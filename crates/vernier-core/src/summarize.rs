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

use std::borrow::Cow;
use std::ops::Range;

use ndarray::Axis;

use crate::accumulate::Accumulated;
use crate::error::EvalError;

/// Tolerance for matching a user-supplied IoU threshold to a value in
/// the `iou_thresholds` ladder. Rounds out the ulp-level error from the
/// `linspace(0.5, 0.95, 10)` build (quirk **L1**).
const IOU_LOOKUP_TOL: f64 = 1e-12;

/// One bucket on the A-axis of an [`Accumulated`] — an index plus a
/// label for rendering.
///
/// The canonical pycocotools detection layout is exposed as
/// [`AreaRng::ALL`] / [`SMALL`](Self::SMALL) / [`MEDIUM`](Self::MEDIUM)
/// / [`LARGE`](Self::LARGE), matching the cocoeval `Params.areaRngLbl`
/// order. Custom layouts (e.g., robotics-style finer buckets) are
/// constructed with [`AreaRng::new`] for owned labels or
/// [`AreaRng::from_static`] for `&'static str` labels.
///
/// The *bounds* that turn an annotation's area into a bucket index
/// live upstream, on the orchestrator that builds [`crate::PerImageEval`]
/// cells; the summarizer only consumes the resulting A-axis index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AreaRng {
    /// Position on the A-axis of [`Accumulated::precision`] /
    /// [`Accumulated::recall`]. Validated against the actual A-axis
    /// length at summarize time, not at construction; an out-of-range
    /// index produces [`EvalError::InvalidConfig`].
    pub index: usize,
    /// Label rendered by [`Summary::pretty_lines`].
    pub label: Cow<'static, str>,
}

impl AreaRng {
    /// Construct from any owned- or borrowed-string label.
    pub fn new(index: usize, label: impl Into<Cow<'static, str>>) -> Self {
        Self {
            index,
            label: label.into(),
        }
    }

    /// `const`-friendly constructor for compile-time labels.
    pub const fn from_static(index: usize, label: &'static str) -> Self {
        Self {
            index,
            label: Cow::Borrowed(label),
        }
    }

    /// COCO `all` bucket — pycocotools' `[0, 1e10]`, A-axis index 0.
    pub const ALL: Self = Self::from_static(0, "all");
    /// COCO `small` bucket — pycocotools' `[0, 32^2]`, A-axis index 1.
    pub const SMALL: Self = Self::from_static(1, "small");
    /// COCO `medium` bucket — pycocotools' `[32^2, 96^2]`, A-axis index 2.
    pub const MEDIUM: Self = Self::from_static(2, "medium");
    /// COCO `large` bucket — pycocotools' `[96^2, 1e10]`, A-axis index 3.
    pub const LARGE: Self = Self::from_static(3, "large");
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
                format!(
                    " {title:<18} {kind} @[ IoU={iou:<9} | area={:>6} | maxDets={:>3} ] = {:0.3}",
                    line.area.label, line.max_dets, line.value
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
#[derive(Debug, Clone)]
pub struct StatRequest {
    /// AP or AR.
    pub metric: Metric,
    /// `None` averages across the IoU ladder; `Some(t)` pins one row.
    /// Looked up against `iou_thresholds` within [`IOU_LOOKUP_TOL`] at
    /// summarize time; values not on the ladder produce
    /// [`EvalError::InvalidConfig`].
    pub iou_threshold: Option<f64>,
    /// Area-range bucket on the A-axis.
    pub area: AreaRng,
    /// How to pick the M-axis entry.
    pub max_dets: MaxDetSelector,
}

impl StatRequest {
    /// Convenience constructor. `const`-callable so [`coco_detection_default`]
    /// and downstream user-defined plans can be assembled in `const`
    /// contexts.
    ///
    /// [`coco_detection_default`]: Self::coco_detection_default
    pub const fn new(
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
    pub const fn coco_detection_default() -> [Self; 12] {
        use MaxDetSelector::{Largest, Value};
        use Metric::{AveragePrecision, AverageRecall};
        [
            Self::new(AveragePrecision, None, AreaRng::ALL, Largest),
            Self::new(AveragePrecision, Some(0.5), AreaRng::ALL, Largest),
            Self::new(AveragePrecision, Some(0.75), AreaRng::ALL, Largest),
            Self::new(AveragePrecision, None, AreaRng::SMALL, Largest),
            Self::new(AveragePrecision, None, AreaRng::MEDIUM, Largest),
            Self::new(AveragePrecision, None, AreaRng::LARGE, Largest),
            Self::new(AverageRecall, None, AreaRng::ALL, Value(1)),
            Self::new(AverageRecall, None, AreaRng::ALL, Value(10)),
            Self::new(AverageRecall, None, AreaRng::ALL, Value(100)),
            Self::new(AverageRecall, None, AreaRng::SMALL, Largest),
            Self::new(AverageRecall, None, AreaRng::MEDIUM, Largest),
            Self::new(AverageRecall, None, AreaRng::LARGE, Largest),
        ]
    }

    /// The canonical 10-entry pycocotools keypoints plan, in the
    /// `[AP, AP50, AP75, AP_M, AP_L, AR, AR50, AR75, AR_M, AR_L]`
    /// order (cocoeval.py:478-499 under `iouType="keypoints"`).
    ///
    /// Differs from [`Self::coco_detection_default`] in three ways,
    /// all per ADR-0012:
    ///
    /// - 10 entries, not 12 — the small-area row is dropped on both
    ///   AP and AR (quirk **D5**).
    /// - Every entry uses [`MaxDetSelector::Largest`], which resolves
    ///   to the kp-canonical `(20,)` ladder; there are no `AR_1` /
    ///   `AR_10` / `AR_100` rows because the kp ladder has only one
    ///   rung.
    /// - The `AreaRng` indices `0/1/2` (all/medium/large) are
    ///   re-indexed for the kp A-axis. Callers must pair this plan
    ///   with [`crate::AreaRange::keypoints_default`] so the A-axis
    ///   indices line up; the const [`AreaRng::ALL`] / `MEDIUM` /
    ///   `LARGE` carry the four-bucket detection-grid indices and
    ///   would index off the end of a three-bucket accumulator.
    pub const fn coco_keypoints_default() -> [Self; 10] {
        use MaxDetSelector::Largest;
        use Metric::{AveragePrecision, AverageRecall};
        // D5: re-indexed kp A-axis (0=all, 1=medium, 2=large), no small.
        // `from_static` is `const`, so each call site materializes a
        // fresh `AreaRng` without an intermediate `clone()` — mirroring
        // `coco_detection_default`'s use of the const `AreaRng::ALL`
        // / `MEDIUM` / `LARGE` constants.
        const ALL: AreaRng = AreaRng::from_static(0, "all");
        const MEDIUM: AreaRng = AreaRng::from_static(1, "medium");
        const LARGE: AreaRng = AreaRng::from_static(2, "large");
        [
            Self::new(AveragePrecision, None, ALL, Largest),
            Self::new(AveragePrecision, Some(0.5), ALL, Largest),
            Self::new(AveragePrecision, Some(0.75), ALL, Largest),
            Self::new(AveragePrecision, None, MEDIUM, Largest),
            Self::new(AveragePrecision, None, LARGE, Largest),
            Self::new(AverageRecall, None, ALL, Largest),
            Self::new(AverageRecall, Some(0.5), ALL, Largest),
            Self::new(AverageRecall, Some(0.75), ALL, Largest),
            Self::new(AverageRecall, None, MEDIUM, Largest),
            Self::new(AverageRecall, None, LARGE, Largest),
        ]
    }
}

/// Twelve-stat COCO detection summary, bit-exact with cocoeval.
///
/// Thin wrapper over [`summarize_with`] that supplies the canonical
/// 12-entry plan from [`StatRequest::coco_detection_default`].
/// Downstream callers who need a different shape (keypoint `[20]`
/// maxDets, custom AP@.30, …) should call `summarize_with` directly
/// with their own plan; the canonical plan is available via the
/// constructor for those who want to extend rather than replace it.
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

    // Resolve every selector before computing any means: a typo in any
    // request fails early without wasting evaluation work, and the
    // compute pass below stays infallible.
    let n_a = p_shape[3];
    let m_max = max_dets.len() - 1;
    let resolved: Vec<(usize, Range<usize>)> = plan
        .iter()
        .map(|req| {
            if req.area.index >= n_a {
                return Err(EvalError::InvalidConfig {
                    detail: format!(
                        "AreaRng index {} is out of range for A-axis (size {})",
                        req.area.index, n_a
                    ),
                });
            }
            let m_idx = match req.max_dets {
                MaxDetSelector::Largest => m_max,
                MaxDetSelector::Value(v) => {
                    max_dets.iter().position(|&d| d == v).ok_or_else(|| {
                        EvalError::InvalidConfig {
                            detail: format!("max_dets does not contain {v}"),
                        }
                    })?
                }
            };
            let t_range = match req.iou_threshold {
                None => 0..n_t,
                Some(target) => {
                    let t = iou_thresholds
                        .iter()
                        .position(|&v| (v - target).abs() < IOU_LOOKUP_TOL)
                        .ok_or_else(|| EvalError::InvalidConfig {
                            detail: format!("iou_threshold {target} not in ladder"),
                        })?;
                    t..(t + 1)
                }
            };
            Ok((m_idx, t_range))
        })
        .collect::<Result<Vec<_>, EvalError>>()?;

    let lines = plan
        .iter()
        .zip(resolved)
        .map(|(req, (m_idx, t_range))| {
            let value = mean_slice(accum, req.metric, t_range, req.area.index, m_idx);
            StatLine {
                metric: req.metric,
                iou_threshold: req.iou_threshold,
                area: req.area.clone(),
                max_dets: max_dets[m_idx],
                value,
            }
        })
        .collect();

    Ok(Summary { lines })
}

/// Mean of an `Accumulated` slice, filtering out the `-1` sentinel
/// (quirks **C5/L6**). Returns `-1.0` if every cell in the slice is
/// the sentinel (mirrors pycocotools' `if len(s[s>-1])==0: -1`).
///
/// The sum is computed via numpy-compatible pairwise summation
/// ([`pairwise_sum`]) so the result is bit-identical to
/// `np.mean(s[s>-1])` for the same input ordering.
///
/// Infallible: callers must validate `t_range`, `area_idx`, and `m_idx`
/// against the `Accumulated`'s shape upfront (see [`summarize_with`]).
fn mean_slice(
    accum: &Accumulated,
    metric: Metric,
    t_range: Range<usize>,
    area_idx: usize,
    m_idx: usize,
) -> f64 {
    let t_count = t_range.len();
    let cap = match metric {
        Metric::AveragePrecision => {
            t_count * accum.precision.shape()[1] * accum.precision.shape()[2]
        }
        Metric::AverageRecall => t_count * accum.recall.shape()[1],
    };
    let mut filtered: Vec<f64> = Vec::with_capacity(cap);
    let mut push = |v: f64| {
        if v > -1.0 {
            filtered.push(v);
        }
    };
    for t in t_range {
        match metric {
            Metric::AveragePrecision => accum
                .precision
                .index_axis(Axis(0), t)
                .index_axis(Axis(2), area_idx)
                .index_axis(Axis(2), m_idx)
                .iter()
                .copied()
                .for_each(&mut push),
            Metric::AverageRecall => accum
                .recall
                .index_axis(Axis(0), t)
                .index_axis(Axis(1), area_idx)
                .index_axis(Axis(1), m_idx)
                .iter()
                .copied()
                .for_each(&mut push),
        }
    }
    if filtered.is_empty() {
        -1.0
    } else {
        pairwise_sum(&filtered) / filtered.len() as f64
    }
}

/// Numpy-compatible pairwise summation for `f64` slices.
///
/// Matches the algorithm used by `np.add.reduce` on contiguous
/// double-precision arrays (see numpy's
/// `numpy/core/src/umath/loops_utils.h.src::pairwise_sum_DOUBLE`):
///
/// - `n < 8`: naive forward sum.
/// - `8 <= n <= PW_BLOCKSIZE` (128): 8 separately accumulated lanes
///   combined via a balanced tree `((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))`,
///   followed by a tail loop for the remainder.
/// - `n > PW_BLOCKSIZE`: split at `n / 2` aligned down to a multiple of
///   8 and recurse on both halves.
///
/// Reproducing this here is a quirk-**C8**-style alignment: the public
/// summary stats ride on top of `np.mean(s[s > -1])`, and any other sum
/// order drifts by ~1 ULP.
fn pairwise_sum(values: &[f64]) -> f64 {
    const PW_BLOCKSIZE: usize = 128;
    let n = values.len();

    if n < 8 {
        let mut s = 0.0_f64;
        for &v in values {
            s += v;
        }
        return s;
    }

    if n <= PW_BLOCKSIZE {
        let mut r = [
            values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
        ];
        let trunc = n - (n % 8);
        let mut i = 8;
        while i < trunc {
            r[0] += values[i];
            r[1] += values[i + 1];
            r[2] += values[i + 2];
            r[3] += values[i + 3];
            r[4] += values[i + 4];
            r[5] += values[i + 5];
            r[6] += values[i + 6];
            r[7] += values[i + 7];
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += values[i];
            i += 1;
        }
        return res;
    }

    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&values[..n2]) + pairwise_sum(&values[n2..])
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
                AreaRng::ALL,
                MaxDetSelector::Largest,
            ),
            StatRequest::new(
                Metric::AverageRecall,
                Some(0.75),
                AreaRng::ALL,
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
    fn custom_area_bucket_with_owned_label_renders_in_pretty_lines() {
        // 5-bucket A-axis (e.g. an orchestrator that adds a "tiny"
        // bucket below "small"). The plan addresses index 4 by name and
        // the label flows through to pretty_lines.
        let iou = iou_thresholds();
        let max_dets = [100usize];
        let accum = Accumulated {
            precision: Array5::<f64>::from_elem((iou.len(), 101, 1, 5, 1), 1.0),
            recall: Array4::<f64>::from_elem((iou.len(), 1, 5, 1), 1.0),
            scores: Array5::<f64>::from_elem((iou.len(), 101, 1, 5, 1), 1.0),
        };
        let plan = [StatRequest::new(
            Metric::AveragePrecision,
            None,
            AreaRng::new(4, "tiny"),
            MaxDetSelector::Largest,
        )];
        let summary = summarize_with(&accum, &plan, iou, &max_dets).unwrap();
        let lines = summary.pretty_lines();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("tiny"), "unexpected line: {}", lines[0]);
    }

    #[test]
    fn out_of_range_area_index_is_typed_error() {
        // Plan addresses A-axis index 4 against a 4-bucket Accumulated.
        let iou = iou_thresholds();
        let max_dets = [100usize];
        let accum = Accumulated {
            precision: Array5::<f64>::from_elem((iou.len(), 101, 1, 4, 1), 1.0),
            recall: Array4::<f64>::from_elem((iou.len(), 1, 4, 1), 1.0),
            scores: Array5::<f64>::from_elem((iou.len(), 101, 1, 4, 1), 1.0),
        };
        let plan = [StatRequest::new(
            Metric::AveragePrecision,
            None,
            AreaRng::new(4, "tiny"),
            MaxDetSelector::Largest,
        )];
        let err = summarize_with(&accum, &plan, iou, &max_dets).unwrap_err();
        assert!(matches!(err, EvalError::InvalidConfig { .. }));
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

    #[test]
    fn pairwise_sum_matches_numpy_add_reduce_bitwise() {
        // 1010 alternating elements is large enough to drive both the
        // 8-lane unrolled block and the recursive split (n > 128). The
        // expected hex below is `np.add.reduce(v).hex()` for the same
        // sequence; naive forward summation lands one ULP higher
        // (`0x1.f900000002309p+8`).
        let v: Vec<f64> = (0..1010)
            .map(|i| if i % 2 == 0 { 1.0 } else { 1e-12 })
            .collect();
        let got = pairwise_sum(&v);
        let expected = f64::from_bits(0x407f_9000_0000_22b4);
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "pairwise_sum drifts from numpy: got {got:e}, expected {expected:e}",
        );
    }

    #[test]
    fn coco_keypoints_default_plan_pins_canonical_order() {
        // ADR-0012 / D5: pycocotools' kp summary is exactly these 10
        // lines, in this order. Pin metric, threshold, A-axis index,
        // and selector so a refactor cannot silently re-order, drop a
        // row, or re-introduce the small bucket.
        let plan = StatRequest::coco_keypoints_default();
        assert_eq!(plan.len(), 10);

        // Each entry: (metric, iou_threshold, area_index, selector).
        let expected: [(Metric, Option<f64>, usize, MaxDetSelector); 10] = [
            (Metric::AveragePrecision, None, 0, MaxDetSelector::Largest), // AP
            (
                Metric::AveragePrecision,
                Some(0.5),
                0,
                MaxDetSelector::Largest,
            ), // AP50
            (
                Metric::AveragePrecision,
                Some(0.75),
                0,
                MaxDetSelector::Largest,
            ), // AP75
            (Metric::AveragePrecision, None, 1, MaxDetSelector::Largest), // AP_M
            (Metric::AveragePrecision, None, 2, MaxDetSelector::Largest), // AP_L
            (Metric::AverageRecall, None, 0, MaxDetSelector::Largest),    // AR
            (Metric::AverageRecall, Some(0.5), 0, MaxDetSelector::Largest), // AR50
            (
                Metric::AverageRecall,
                Some(0.75),
                0,
                MaxDetSelector::Largest,
            ), // AR75
            (Metric::AverageRecall, None, 1, MaxDetSelector::Largest),    // AR_M
            (Metric::AverageRecall, None, 2, MaxDetSelector::Largest),    // AR_L
        ];

        for (i, (metric, iou, idx, sel)) in expected.into_iter().enumerate() {
            assert_eq!(plan[i].metric, metric, "row {i} metric");
            assert_eq!(plan[i].iou_threshold, iou, "row {i} iou_threshold");
            assert_eq!(plan[i].area.index, idx, "row {i} area index");
            assert_eq!(plan[i].max_dets, sel, "row {i} selector");
        }

        // No row addresses A-axis index 3 (would land off the end of a
        // 3-bucket kp accumulator) and no row addresses index 1 of the
        // detection-grid (which is "small" — D5 forbids).
        assert!(plan.iter().all(|r| r.area.index <= 2));
    }

    #[test]
    fn pairwise_sum_handles_short_inputs_with_naive_fallback() {
        // n < 8 uses the simple loop; verify a hand-checked tiny case.
        let v = [1.0_f64, 2.0, 3.0, 4.0];
        assert_eq!(pairwise_sum(&v), 10.0);
        assert_eq!(pairwise_sum(&[]), 0.0);
        assert_eq!(pairwise_sum(&[42.0]), 42.0);
    }
}
