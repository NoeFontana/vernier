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

use ndarray::Axis;

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

/// AP / AR selector emitted on every [`StatLine`].
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

/// Result of evaluating a summary plan over an [`Accumulated`].
///
/// `lines.len()` matches the plan length; for the canonical pycocotools
/// detection summary built by [`summarize_detection`], that's 12 lines
/// in the order `[AP, AP50, AP75, AP_S, AP_M, AP_L, AR_1, AR_10,
/// AR_100, AR_S, AR_M, AR_L]`. For custom plans evaluated via
/// [`summarize_with`], `lines` mirrors the request order.
#[derive(Debug, Clone)]
pub struct Summary {
    /// One entry per request in the evaluated plan, paired with slicing
    /// metadata.
    pub lines: Vec<StatLine>,
}

impl Summary {
    /// Numeric values in plan order. Equivalent to
    /// `lines.iter().map(|l| l.value).collect()`.
    pub fn stats(&self) -> Vec<f64> {
        self.lines.iter().map(|l| l.value).collect()
    }
    /// Render the canonical pycocotools text table (12 lines, each in
    /// the upstream `Average Precision (AP) @[ IoU=... | area=... |
    /// maxDets=... ] = 0.xxx` shape). Returned as a `Vec<String>`; the
    /// caller decides whether to print, log, or test against it.
    pub fn pretty_lines(&self) -> Vec<String> {
        self.lines
            .iter()
            .map(|line| {
                let (title, kind) = match line.metric {
                    Metric::AveragePrecision => ("Average Precision", "(AP)"),
                    Metric::AverageRecall => ("Average Recall", "(AR)"),
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

/// How a [`StatRequest`] picks an entry on the M-axis of an
/// [`Accumulated`].
///
/// Pycocotools hard-codes `maxDets[0|1|2]` for `AR_{1,10,100}` and
/// `maxDets[-1]` for everything else; this enum lets a plan express
/// that intent — "the largest cap available" or "the entry whose value
/// equals N" — without binding to fixed positional indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxDetSelector {
    /// Pick the largest cap in the supplied `max_dets` slice. This is
    /// what every cocoeval AP line and `AR_S` / `AR_M` / `AR_L` use.
    Largest,
    /// Pick the M-axis entry whose value equals this. Errors via
    /// [`EvalError::InvalidConfig`] if the value is absent.
    Value(usize),
}

/// One line of a summary plan — describes a single mean to compute.
#[derive(Debug, Clone, Copy)]
pub struct StatRequest {
    /// AP or AR.
    pub metric: Metric,
    /// `None` averages across the IoU ladder; `Some(t)` pins one row.
    pub iou_threshold: Option<f64>,
    /// Area-range bucket on the A-axis.
    pub area: AreaRng,
    /// How to pick the M-axis entry.
    pub max_dets: MaxDetSelector,
}

impl StatRequest {
    /// Convenience constructor.
    pub fn new(
        metric: Metric,
        iou_threshold: Option<f64>,
        area: AreaRng,
        max_dets: MaxDetSelector,
    ) -> Self {
        Self {
            metric,
            iou_threshold,
            area,
            max_dets,
        }
    }

    /// The canonical 12-entry pycocotools detection plan, in the
    /// `[AP, AP50, AP75, AP_S, AP_M, AP_L, AR_1, AR_10, AR_100, AR_S,
    /// AR_M, AR_L]` order. Bit-exact with cocoeval is by construction:
    /// [`summarize_detection`] is just `summarize_with(.., this, ..)`.
    pub fn coco_detection_default() -> [Self; 12] {
        use AreaRng::{All, Large, Medium, Small};
        use MaxDetSelector::{Largest, Value};
        use Metric::{AveragePrecision, AverageRecall};
        [
            Self::new(AveragePrecision, None, All, Largest),
            Self::new(AveragePrecision, Some(0.5), All, Largest),
            Self::new(AveragePrecision, Some(0.75), All, Largest),
            Self::new(AveragePrecision, None, Small, Largest),
            Self::new(AveragePrecision, None, Medium, Largest),
            Self::new(AveragePrecision, None, Large, Largest),
            Self::new(AverageRecall, None, All, Value(1)),
            Self::new(AverageRecall, None, All, Value(10)),
            Self::new(AverageRecall, None, All, Value(100)),
            Self::new(AverageRecall, None, Small, Largest),
            Self::new(AverageRecall, None, Medium, Largest),
            Self::new(AverageRecall, None, Large, Largest),
        ]
    }
}

/// Twelve-stat COCO detection summary, bit-exact with cocoeval.
///
/// Thin wrapper over [`summarize_with`] that supplies the canonical
/// 12-entry plan from [`StatRequest::coco_detection_default`].
/// Downstream callers who need a different shape (LVIS area buckets,
/// keypoint `[20]` maxDets, custom AP@.30, …) should call
/// `summarize_with` directly with their own plan; the canonical plan is
/// available via the constructor for those who want to extend rather
/// than replace it.
///
/// # Errors
///
/// Same conditions as [`summarize_with`].
pub fn summarize_detection(
    accum: &Accumulated,
    iou_thresholds: &[f64],
    max_dets: &[usize],
) -> Result<Summary, EvalError> {
    summarize_with(
        accum,
        &StatRequest::coco_detection_default(),
        iou_thresholds,
        max_dets,
    )
}

/// Evaluate an arbitrary summary plan over an [`Accumulated`].
///
/// `iou_thresholds` and `max_dets` describe the grid the `Accumulated`
/// was built against; they are needed to resolve [`StatRequest`]
/// selectors (IoU value → T-axis index, [`MaxDetSelector`] → M-axis
/// index) and to populate the `max_dets` field on each emitted
/// [`StatLine`].
///
/// # Errors
///
/// Returns [`EvalError::DimensionMismatch`] if `iou_thresholds` or
/// `max_dets` lengths disagree with `accum`'s `T`/`M` axes. Returns
/// [`EvalError::InvalidConfig`] if any request names an IoU threshold
/// not present in `iou_thresholds` (within `1e-12`) or a
/// [`MaxDetSelector::Value`] absent from `max_dets`.
pub fn summarize_with(
    accum: &Accumulated,
    plan: &[StatRequest],
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
    let resolve_m = |sel: MaxDetSelector| -> Result<usize, EvalError> {
        match sel {
            MaxDetSelector::Largest => Ok(m_max),
            MaxDetSelector::Value(v) => {
                max_dets
                    .iter()
                    .position(|&d| d == v)
                    .ok_or_else(|| EvalError::InvalidConfig {
                        detail: format!("max_dets does not contain {v}"),
                    })
            }
        }
    };

    let lines = plan
        .iter()
        .map(|req| {
            let m_idx = resolve_m(req.max_dets)?;
            let value = mean_slice(
                accum,
                req.metric,
                req.iou_threshold,
                req.area,
                m_idx,
                iou_thresholds,
            )?;
            Ok(StatLine {
                metric: req.metric,
                iou_threshold: req.iou_threshold,
                area: req.area,
                max_dets: max_dets[m_idx],
                value,
            })
        })
        .collect::<Result<Vec<_>, EvalError>>()?;

    Ok(Summary { lines })
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
                .ok_or_else(|| EvalError::InvalidConfig {
                    detail: format!("iou_threshold {target} not in ladder"),
                })?;
            t..(t + 1)
        }
    };
    let a = area.index();

    // C5: skip -1 sentinels in the mean.
    let mut sum = 0.0_f64;
    let mut count = 0usize;
    let mut tally = |v: f64| {
        if v > -1.0 {
            sum += v;
            count += 1;
        }
    };
    for t in t_range {
        match metric {
            Metric::AveragePrecision => accum
                .precision
                .index_axis(Axis(0), t)
                .index_axis(Axis(2), a)
                .index_axis(Axis(2), m_idx)
                .iter()
                .copied()
                .for_each(&mut tally),
            Metric::AverageRecall => accum
                .recall
                .index_axis(Axis(0), t)
                .index_axis(Axis(1), a)
                .index_axis(Axis(1), m_idx)
                .iter()
                .copied()
                .for_each(&mut tally),
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

        let stats = summary.stats();
        assert_eq!(stats.len(), 12);
        // AP[all], AP50, AP75, AR_1, AR_10, AR_100 should all be ~1.0.
        for &i in &[0usize, 1, 2, 6, 7, 8] {
            let v = stats[i];
            assert!((v - 1.0).abs() < 1e-9, "stat[{i}] = {v}");
        }
        // small / medium / large carry -1 (no data).
        for &i in &[3usize, 4, 5, 9, 10, 11] {
            assert_eq!(stats[i], -1.0, "stat[{i}] should be -1 sentinel");
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
        assert!(summary.stats().iter().all(|&v| v == -1.0));
    }

    #[test]
    fn missing_max_det_value_is_typed_error() {
        // AR_1 line requires max_dets to contain 1; without it,
        // summarization fails with InvalidConfig.
        let iou = iou_thresholds();
        let max_dets = [10usize, 100];
        let accum = Accumulated {
            precision: Array5::<f64>::from_elem((iou.len(), 101, 1, 4, 2), -1.0),
            recall: Array4::<f64>::from_elem((iou.len(), 1, 4, 2), -1.0),
            scores: Array5::<f64>::from_elem((iou.len(), 101, 1, 4, 2), -1.0),
        };
        let err = summarize_detection(&accum, iou, &max_dets).unwrap_err();
        assert!(matches!(err, EvalError::InvalidConfig { .. }));
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
    fn summarize_with_custom_plan_evaluates_only_requested_lines() {
        // Demonstrates the extension point: a 2-entry plan asking for
        // AP@.50 across all areas and AR@.75 (not in the canonical 12)
        // — both at the largest cap. Order is preserved.
        let iou = iou_thresholds();
        let max_dets = [100usize];
        let accum = Accumulated {
            precision: Array5::<f64>::from_elem((iou.len(), 101, 1, 4, 1), 0.5),
            recall: Array4::<f64>::from_elem((iou.len(), 1, 4, 1), 0.7),
            scores: Array5::<f64>::from_elem((iou.len(), 101, 1, 4, 1), 1.0),
        };
        let plan = [
            StatRequest::new(
                Metric::AveragePrecision,
                Some(0.5),
                AreaRng::All,
                MaxDetSelector::Largest,
            ),
            StatRequest::new(
                Metric::AverageRecall,
                Some(0.75),
                AreaRng::All,
                MaxDetSelector::Largest,
            ),
        ];
        let summary = summarize_with(&accum, &plan, iou, &max_dets).unwrap();
        assert_eq!(summary.lines.len(), 2);
        assert!((summary.lines[0].value - 0.5).abs() < 1e-12);
        assert_eq!(summary.lines[0].iou_threshold, Some(0.5));
        assert!((summary.lines[1].value - 0.7).abs() < 1e-12);
        assert_eq!(summary.lines[1].metric, Metric::AverageRecall);
    }

    #[test]
    fn summarize_detection_matches_canonical_plan_via_summarize_with() {
        // The thin-wrapper invariant: results are bit-equal whether the
        // caller invokes summarize_detection or summarize_with with the
        // canonical plan.
        let iou = iou_thresholds();
        let max_dets = [1usize, 10, 100];
        let accum = Accumulated {
            precision: Array5::<f64>::from_elem((iou.len(), 101, 1, 4, 3), 0.5),
            recall: Array4::<f64>::from_elem((iou.len(), 1, 4, 3), 0.7),
            scores: Array5::<f64>::from_elem((iou.len(), 101, 1, 4, 3), 1.0),
        };
        let direct = summarize_detection(&accum, iou, &max_dets).unwrap();
        let via_plan = summarize_with(
            &accum,
            &StatRequest::coco_detection_default(),
            iou,
            &max_dets,
        )
        .unwrap();
        assert_eq!(direct.stats(), via_plan.stats());
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
