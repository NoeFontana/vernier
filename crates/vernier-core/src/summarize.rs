//! Twelve-stat detection summary atop [`crate::Accumulated`].
//!
//! Mirrors `pycocotools.cocoeval.COCOeval.summarize` (cocoeval.py
//! lines 422-475), but as a pure structured value — no stdout side
//! effects (quirks **L5/L6/L7**, dispositioned `corrected`).
//!
//! ## Quirk dispositions
//!
//! - **C5** (`strict`): cells absent from the dataset carry `-1`;
//!   summarization filters them out via `s > -1` before averaging.
//! - **L5** (`corrected`): the print/log side-effect from upstream
//!   `_summarize` is gone. Use [`Summary::pretty_lines`] for the
//!   pycocotools-shaped human-readable rendering.
//! - **L6** (`corrected`): empty-eval `mean(empty)` no longer raises a
//!   numpy RuntimeWarning — the absent case explicitly returns `-1`.
//! - **L7** (`corrected`): the result is a value (`Summary`), not a
//!   property side-effect on the evaluator.

use crate::accumulate::Accumulated;
use crate::error::EvalError;

/// COCO area-range buckets, as enumerated by pycocotools'
/// `Params.areaRngLbl` for detection: `[all, small, medium, large]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AreaRng {
    /// Full range — pycocotools' `[0, 1e10]`. Index 0.
    All,
    /// Small — pycocotools' `[0, 32^2]`. Index 1.
    Small,
    /// Medium — `[32^2, 96^2]`. Index 2.
    Medium,
    /// Large — `[96^2, 1e10]`. Index 3.
    Large,
}

impl AreaRng {
    /// Index in the canonical COCO `areaRng` table.
    fn index(self) -> usize {
        match self {
            Self::All => 0,
            Self::Small => 1,
            Self::Medium => 2,
            Self::Large => 3,
        }
    }
}

/// Discriminator passed to [`mean_slice`] and emitted in
/// [`StatLine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Average Precision — slices `Accumulated::precision`.
    AveragePrecision,
    /// Average Recall — slices `Accumulated::recall`. Quirk **C4**: AR
    /// is the terminal cumulative recall, not an integral of the
    /// precision/recall curve.
    AverageRecall,
}

/// Single line of the COCO 12-stat summary table.
#[derive(Debug, Clone)]
pub struct StatLine {
    /// AP or AR.
    pub metric: Metric,
    /// `None` means averaged across the whole IoU ladder; `Some(t)`
    /// pins a specific threshold (e.g., 0.5 for AP@.50).
    pub iou_threshold: Option<f64>,
    /// Area-range bucket.
    pub area: AreaRng,
    /// Per-image maxDet cap.
    pub max_dets: usize,
    /// Mean over the matching slice, ignoring `-1` sentinels. `-1.0`
    /// when the slice has no non-sentinel entries (quirks **C5/L6**).
    pub value: f64,
}

/// Result of summarizing an [`Accumulated`].
///
/// `stats` is the canonical 12-element vector pycocotools'
/// `summarize()` writes into `eval.stats`. `lines` carries the same
/// values with their slicing metadata so callers can render or
/// post-process them without re-deriving the layout.
#[derive(Debug, Clone)]
pub struct Summary {
    /// Twelve mean values, in the canonical pycocotools order:
    /// `[AP, AP50, AP75, AP_S, AP_M, AP_L, AR_1, AR_10, AR_100, AR_S,
    /// AR_M, AR_L]`.
    pub stats: Vec<f64>,
    /// Same values as `stats`, paired with their slicing metadata.
    pub lines: Vec<StatLine>,
}

impl Summary {
    /// Render the canonical pycocotools text table (12 lines, each in
    /// the upstream `Average Precision (AP) @[ IoU=... | area=... |
    /// maxDets=... ] = 0.xxx` shape). Returned as a `Vec<String>`; the
    /// caller decides whether to print, log, or test against it.
    pub fn pretty_lines(&self) -> Vec<String> {
        self.lines
            .iter()
            .map(|line| {
                let title = match line.metric {
                    Metric::AveragePrecision => "Average Precision",
                    Metric::AverageRecall => "Average Recall",
                };
                let kind = match line.metric {
                    Metric::AveragePrecision => "(AP)",
                    Metric::AverageRecall => "(AR)",
                };
                let iou = match line.iou_threshold {
                    Some(t) => format!("{t:0.2}"),
                    None => "0.50:0.95".to_string(),
                };
                let area = match line.area {
                    AreaRng::All => "   all",
                    AreaRng::Small => " small",
                    AreaRng::Medium => "medium",
                    AreaRng::Large => " large",
                };
                format!(
                    " {title:<18} {kind} @[ IoU={iou:<9} | area={area:>6} | maxDets={:>3} ] = {:0.3}",
                    line.max_dets, line.value
                )
            })
            .collect()
    }
}

/// Twelve-stat COCO detection summary.
///
/// `iou_thresholds` and `max_dets` describe the same grid the
/// `Accumulated` was built against. `iou_thresholds` is needed to map
/// the AP@.50 / AP@.75 selectors to row indices; `max_dets` to map the
/// AR_1 / AR_10 / AR_100 selectors to the M-axis.
///
/// # Errors
///
/// Returns [`EvalError::DimensionMismatch`] if `iou_thresholds` length
/// does not match `accum.precision`'s `T` axis, or if `max_dets`
/// length does not match the `M` axis. Returns
/// [`EvalError::InvalidAnnotation`] (with `detail = "iou_threshold"`)
/// when AP@.50 / AP@.75 are not present in `iou_thresholds`.
pub fn summarize_detection(
    accum: &Accumulated,
    iou_thresholds: &[f64],
    max_dets: &[usize],
) -> Result<Summary, EvalError> {
    let p_shape = accum.precision.shape();
    let r_shape = accum.recall.shape();
    let n_t = p_shape[0];
    let n_m = p_shape[4];

    if n_t != iou_thresholds.len() {
        return Err(EvalError::DimensionMismatch {
            detail: format!(
                "precision T-axis {} != iou_thresholds len {}",
                n_t,
                iou_thresholds.len()
            ),
        });
    }
    if n_m != max_dets.len() {
        return Err(EvalError::DimensionMismatch {
            detail: format!(
                "precision M-axis {} != max_dets len {}",
                n_m,
                max_dets.len()
            ),
        });
    }
    if r_shape[0] != n_t || r_shape[3] != n_m {
        return Err(EvalError::DimensionMismatch {
            detail: format!("recall {r_shape:?} disagrees with precision {p_shape:?}"),
        });
    }

    let m_max = max_dets.len() - 1;
    let m_at = |target: usize| {
        max_dets
            .iter()
            .position(|&d| d == target)
            .ok_or_else(|| EvalError::InvalidAnnotation {
                detail: format!("max_dets does not contain {target}"),
            })
    };

    // Build the 12-stat plan as (metric, iouThr, area, max_dets index).
    // Pycocotools indexes maxDets[0|1|2] for AR_{1,10,100} and
    // maxDets[2] for everything else; we honor whatever the user
    // actually passed for the M-axis but map to the same intent.
    let plan: Vec<(Metric, Option<f64>, AreaRng, usize)> = vec![
        (Metric::AveragePrecision, None, AreaRng::All, m_max),
        (Metric::AveragePrecision, Some(0.5), AreaRng::All, m_max),
        (Metric::AveragePrecision, Some(0.75), AreaRng::All, m_max),
        (Metric::AveragePrecision, None, AreaRng::Small, m_max),
        (Metric::AveragePrecision, None, AreaRng::Medium, m_max),
        (Metric::AveragePrecision, None, AreaRng::Large, m_max),
        (Metric::AverageRecall, None, AreaRng::All, m_at(1)?),
        (Metric::AverageRecall, None, AreaRng::All, m_at(10)?),
        (Metric::AverageRecall, None, AreaRng::All, m_at(100)?),
        (Metric::AverageRecall, None, AreaRng::Small, m_max),
        (Metric::AverageRecall, None, AreaRng::Medium, m_max),
        (Metric::AverageRecall, None, AreaRng::Large, m_max),
    ];

    let mut stats = Vec::with_capacity(plan.len());
    let mut lines = Vec::with_capacity(plan.len());
    for &(metric, iou_thr, area, m_idx) in &plan {
        let value = mean_slice(accum, metric, iou_thr, area, m_idx, iou_thresholds)?;
        stats.push(value);
        lines.push(StatLine {
            metric,
            iou_threshold: iou_thr,
            area,
            max_dets: max_dets[m_idx],
            value,
        });
    }

    Ok(Summary { stats, lines })
}

/// Mean of an `Accumulated` slice, filtering out the `-1` sentinel
/// (quirks **C5/L6**). Returns `-1.0` if every cell in the slice is
/// the sentinel (mirrors pycocotools' `if len(s[s>-1])==0: -1`).
fn mean_slice(
    accum: &Accumulated,
    metric: Metric,
    iou_thr: Option<f64>,
    area: AreaRng,
    m_idx: usize,
    iou_thresholds: &[f64],
) -> Result<f64, EvalError> {
    let t_range = match iou_thr {
        None => 0..iou_thresholds.len(),
        Some(target) => {
            let t = iou_thresholds
                .iter()
                .position(|&v| (v - target).abs() < 1e-12)
                .ok_or_else(|| EvalError::InvalidAnnotation {
                    detail: format!("iou_threshold {target} not in ladder"),
                })?;
            t..(t + 1)
        }
    };
    let a_idx = area.index();

    let mut sum = 0.0_f64;
    let mut count = 0usize;
    match metric {
        Metric::AveragePrecision => {
            let p_shape = accum.precision.shape();
            let n_r = p_shape[1];
            let n_k = p_shape[2];
            // C5: skip -1 sentinels in the average.
            for t in t_range {
                for r in 0..n_r {
                    for k in 0..n_k {
                        let v = accum.precision[(t, r, k, a_idx, m_idx)];
                        if v > -1.0 {
                            sum += v;
                            count += 1;
                        }
                    }
                }
            }
        }
        Metric::AverageRecall => {
            let r_shape = accum.recall.shape();
            let n_k = r_shape[1];
            for t in t_range {
                for k in 0..n_k {
                    let v = accum.recall[(t, k, a_idx, m_idx)];
                    if v > -1.0 {
                        sum += v;
                        count += 1;
                    }
                }
            }
        }
    }
    Ok(if count == 0 { -1.0 } else { sum / count as f64 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulate::{accumulate, AccumulateParams, PerImageEval};
    use crate::parity::{iou_thresholds, recall_thresholds, ParityMode};
    use ndarray::{Array2, Array4, Array5};

    fn perfect_match_eval(t: usize) -> PerImageEval {
        PerImageEval {
            dt_scores: vec![0.9],
            dt_matched: Array2::from_elem((t, 1), true),
            dt_ignore: Array2::from_elem((t, 1), false),
            gt_ignore: vec![false],
        }
    }

    #[test]
    fn perfect_match_summarizes_to_ones() {
        // Single image, single category, all-area only — the simplest
        // valid run that exercises every line of the 12-stat table.
        let iou = iou_thresholds();
        let rec = recall_thresholds();
        let max_dets = [1usize, 10, 100];
        let cell = perfect_match_eval(iou.len());

        // K=1, A=4 (all/small/medium/large), I=1; we populate only the
        // `all` cell. small/medium/large stay None → -1 sentinel.
        let mut grid: Vec<Option<PerImageEval>> = vec![None; 4];
        grid[0] = Some(cell);

        let p = AccumulateParams {
            iou_thresholds: iou,
            recall_thresholds: rec,
            max_dets: &max_dets,
            n_categories: 1,
            n_area_ranges: 4,
            n_images: 1,
        };
        let accum = accumulate(&grid, p, ParityMode::Strict).unwrap();
        let summary = summarize_detection(&accum, iou, &max_dets).unwrap();

        assert_eq!(summary.stats.len(), 12);
        // AP[all], AP50, AP75, AR_1, AR_10, AR_100 should all be ~1.0.
        for &i in &[0usize, 1, 2, 6, 7, 8] {
            let v = summary.stats[i];
            assert!((v - 1.0).abs() < 1e-9, "stat[{i}] = {v}");
        }
        // small / medium / large carry -1 (no data).
        for &i in &[3usize, 4, 5, 9, 10, 11] {
            assert_eq!(summary.stats[i], -1.0, "stat[{i}] should be -1 sentinel");
        }
    }

    #[test]
    fn empty_grid_yields_all_neg_one_stats() {
        let iou = iou_thresholds();
        let rec = recall_thresholds();
        let max_dets = [1usize, 10, 100];
        let p = AccumulateParams {
            iou_thresholds: iou,
            recall_thresholds: rec,
            max_dets: &max_dets,
            n_categories: 1,
            n_area_ranges: 4,
            n_images: 0,
        };
        let accum = accumulate(&[], p, ParityMode::Strict).unwrap();
        let summary = summarize_detection(&accum, iou, &max_dets).unwrap();
        assert!(summary.stats.iter().all(|&v| v == -1.0));
    }

    #[test]
    fn missing_max_det_value_is_typed_error() {
        // AR_1 line requires max_dets to contain 1; if we only hand it
        // [10, 100], summarization fails with InvalidAnnotation.
        let iou = iou_thresholds();
        let max_dets = [10usize, 100];
        let accum = Accumulated {
            precision: Array5::<f64>::from_elem((iou.len(), 101, 1, 4, 2), -1.0),
            recall: Array4::<f64>::from_elem((iou.len(), 1, 4, 2), -1.0),
            scores: Array5::<f64>::from_elem((iou.len(), 101, 1, 4, 2), -1.0),
        };
        let err = summarize_detection(&accum, iou, &max_dets).unwrap_err();
        assert!(matches!(err, EvalError::InvalidAnnotation { .. }));
    }

    #[test]
    fn iou_threshold_dimension_mismatch_is_typed_error() {
        let max_dets = [100usize];
        let accum = Accumulated {
            precision: Array5::<f64>::from_elem((10, 101, 1, 4, 1), -1.0),
            recall: Array4::<f64>::from_elem((10, 1, 4, 1), -1.0),
            scores: Array5::<f64>::from_elem((10, 101, 1, 4, 1), -1.0),
        };
        // pass only 5 thresholds — accum was built with 10.
        let err = summarize_detection(&accum, &[0.5, 0.6, 0.7, 0.8, 0.9], &max_dets).unwrap_err();
        assert!(matches!(err, EvalError::DimensionMismatch { .. }));
    }

    #[test]
    fn pretty_lines_match_pycocotools_shape() {
        let iou = iou_thresholds();
        let max_dets = [1usize, 10, 100];
        let accum = Accumulated {
            precision: Array5::<f64>::from_elem((iou.len(), 101, 1, 4, 3), 1.0),
            recall: Array4::<f64>::from_elem((iou.len(), 1, 4, 3), 1.0),
            scores: Array5::<f64>::from_elem((iou.len(), 101, 1, 4, 3), 1.0),
        };
        let summary = summarize_detection(&accum, iou, &max_dets).unwrap();
        let lines = summary.pretty_lines();
        assert_eq!(lines.len(), 12);
        // Spot-check the first AP line and the first AR line for the
        // pycocotools-shaped layout.
        assert!(lines[0].contains("Average Precision"));
        assert!(lines[0].contains("(AP)"));
        assert!(lines[0].contains("0.50:0.95"));
        assert!(lines[0].contains("maxDets=100"));
        assert!(lines[6].contains("Average Recall"));
        assert!(lines[6].contains("maxDets=  1"));
    }
}
