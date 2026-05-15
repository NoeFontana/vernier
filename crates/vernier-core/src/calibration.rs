//! Detection-family calibration summarizer (ADR-0018).
//!
//! Folds the per-image cell store (`Vec<Option<Box<PerImageEval>>>` —
//! the same shape ADR-0013 produces and [`crate::accumulate::accumulate`]
//! consumes) into a reliability table plus the scalar ECE / MCE
//! summaries. The kernel is a *summarizer*, never an edit to
//! [`crate::matching`] / [`crate::accumulate`] / [`crate::summarize`]
//! (ADR-0005 invariant).
//!
//! ## Input shape
//!
//! `eval_imgs` is flattened `(k, a, i)` with stride
//! `[k * A * I + a * I + i]`, the same layout
//! [`crate::stream::PerImageEvalStore::flatten`] produces. Calibration
//! only consults `area_idx = 0` (the `all` bucket); area faceting is
//! out of scope per ADR-0018 (Shape 1 footnote on ADR-0012 D5).
//!
//! ## Quirk dispositions
//!
//! Each `Pn` ID below is sourced from
//! `docs/engineering/calibration-quirks.md` (Unit 4 deliverable; the
//! survey row is the single source of truth for each disposition).
//!
//! - **P1** (`strict`): bin-edges via numpy `quantile(method='linear')`
//!   — see [`crate::parity::quantile_linear`].
//! - **P2** (`strict`): detections matched to ignore regions
//!   (`dt_ignore[t, d] == true`) drop from the histogram entirely
//!   (R3 mitigation).
//! - **P3** (`corrected`): `min_score = 0.05` default cutoff
//!   (DETR-aware divergence from pycocotools, which has no equivalent).
//! - **P4** (`strict`): Wilson 95% CI for per-bin accuracy. Z-score
//!   pinned to `scipy.stats.norm.ppf(0.975) = 1.959963984540054`.
//! - **P5** (`strict`): duplicate-quantile edges are merged on small
//!   samples; the effective bin count is reported in
//!   [`CalibrationSummary::effective_n_bins`] (R1 mitigation).
//! - **P6** (`corrected`): macro per-class aggregation is the default
//!   for the headline scalars (safety-case rationale; minority classes
//!   are not drowned by the majority).
//!
//! ## Numerical policy
//!
//! All histogram math is `f64` end-to-end (ADR-0004). Per-bin score
//! and accuracy sums route through
//! [`crate::summarize::pairwise_sum`] for numpy-compatible reduction
//! order.

use crate::accumulate::PerImageEval;
use crate::error::EvalError;
use crate::parity::{quantile_linear, ParityMode};
use crate::summarize::pairwise_sum;

/// 95% normal quantile, `scipy.stats.norm.ppf(0.975)` to bit-precision.
///
/// Pinned here so the Wilson interval is reproducible across
/// platforms without dragging a stats crate into `vernier-core`.
/// Quirk **P4** (`strict`).
const Z_95: f64 = 1.959_963_984_540_054;

/// Bin-edge construction strategy.
///
/// Quantile is the default (Quirk **P1**, `strict`); equal-width is
/// available for diagnostics. Choice is independent of the
/// confidence-interval choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binning {
    /// Equal-mass bins built via numpy quantile `method='linear'`.
    Quantile,
    /// Equal-width bins between `min_score` (or the observed
    /// minimum) and `1.0`.
    EqualWidth,
}

/// Per-bin confidence-interval flavor.
///
/// Quirk **P4** (`strict`) — Wilson is the only CI shipped in
/// Unit 1; [`ConfidenceKind::ClopperPearson`] is a Phase-2 follow-up
/// and currently returns an error from
/// [`summarize_calibration`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceKind {
    /// Closed-form Wilson interval (z = 1.959963984540054, 95%).
    Wilson,
    /// Exact Clopper-Pearson interval via the regularized incomplete
    /// beta function. **Phase-2 follow-up** — not yet implemented;
    /// [`summarize_calibration`] returns
    /// [`EvalError::InvalidConfig`] when this variant is requested.
    ClopperPearson,
}

/// How per-class headline scalars combine.
///
/// Quirk **P6** (`corrected`) — `Macro` is the default. The kernel
/// records the choice but does **not** consult it for the marginal
/// `ece` / `mce` scalars (those always come from the marginal
/// binning); per-class aggregation is a caller / UI concern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregation {
    /// Unweighted mean of per-class metrics.
    Macro,
    /// Detection-count-weighted mean of per-class metrics.
    Micro,
}

/// Calibration summarizer parameters.
///
/// Defaults follow ADR-0018:
/// `iou_index=0` (IoU=0.5), `n_bins=15`, `Binning::Quantile`,
/// `min_score=0.05`, `ConfidenceKind::Wilson`, `per_class=false`,
/// `Aggregation::Macro`.
#[derive(Debug, Clone)]
pub struct CalibrationParams {
    /// Index into the IoU-threshold ladder driving the match outcome.
    /// `0` selects IoU=0.5 on the COCO ladder. Must satisfy
    /// `iou_index < T` (the T-axis size of the per-image cells), else
    /// [`summarize_calibration`] returns
    /// [`EvalError::InvalidConfig`].
    pub iou_index: usize,
    /// Number of bins requested. `0` is rejected. May be reduced to
    /// the *effective* bin count (see [`CalibrationSummary::effective_n_bins`])
    /// when duplicate quantile edges collapse (Quirk **P5**).
    pub n_bins: usize,
    /// Bin-edge construction strategy. See [`Binning`].
    pub binning: Binning,
    /// Detections with `dt_scores[d] < min_score` are excluded from
    /// the histogram. Quirk **P3** (`corrected`; DETR-aware default).
    pub min_score: f64,
    /// Per-bin confidence interval. Quirk **P4**.
    pub confidence: ConfidenceKind,
    /// When `true`, also emit the per-class breakdown table.
    pub per_class: bool,
    /// Per-class aggregation flavor recorded for callers / UI. Quirk
    /// **P6**. Does **not** alter the marginal `ece` / `mce` scalars.
    pub per_class_aggregation: Aggregation,
}

impl Default for CalibrationParams {
    fn default() -> Self {
        Self {
            iou_index: 0,
            n_bins: 15,
            binning: Binning::Quantile,
            min_score: 0.05,
            confidence: ConfidenceKind::Wilson,
            per_class: false,
            per_class_aggregation: Aggregation::Macro,
        }
    }
}

/// Columnar reliability table.
///
/// One row per *effective* bin (after Quirk **P5** dedup); see
/// [`CalibrationSummary::effective_n_bins`]. Per-bin floats are
/// `f64::NAN` on zero-count bins (Quirk **R2** mitigation in
/// ADR-0018 terminology).
#[derive(Debug, Clone)]
pub struct ReliabilityTable {
    /// Dense bin identifier, `0..effective_n_bins`.
    pub bin_id: Vec<u32>,
    /// Lower edge of each bin (inclusive).
    pub score_lo: Vec<f64>,
    /// Upper edge of each bin (right-inclusive on the last bin only;
    /// see [`assign_bin`] for the exact assignment rule).
    pub score_hi: Vec<f64>,
    /// Per-bin mean score. `NaN` when `count == 0`.
    pub mean_score: Vec<f64>,
    /// Per-bin accuracy = correct / count. `NaN` when `count == 0`.
    pub accuracy: Vec<f64>,
    /// Per-bin detection count.
    pub count: Vec<u64>,
    /// Per-bin gap = `accuracy - mean_score`. `NaN` when `count == 0`.
    pub gap: Vec<f64>,
    /// Per-bin Wilson-CI lower bound. `NaN` when `count == 0`.
    pub ci_lo: Vec<f64>,
    /// Per-bin Wilson-CI upper bound. `NaN` when `count == 0`.
    pub ci_hi: Vec<f64>,
}

/// Per-class calibration breakdown.
///
/// Only emitted when [`CalibrationParams::per_class`] is `true`.
/// `class_id` lists the categories that survived filtering (i.e.,
/// have at least one detection at or above `min_score` and not
/// dropped by Quirk **P2**); the list is sorted ascending.
#[derive(Debug, Clone)]
pub struct PerClassTable {
    /// Category indices `k` present in the table, sorted ascending.
    pub class_id: Vec<u32>,
    /// Per-class ECE — same formula as the marginal scalar, scoped to
    /// the class's detections.
    pub ece: Vec<f64>,
    /// Per-class MCE.
    pub mce: Vec<f64>,
    /// Per-class detection count.
    pub n: Vec<u64>,
}

/// Top-level calibration summary returned by
/// [`summarize_calibration`].
#[derive(Debug, Clone)]
pub struct CalibrationSummary {
    /// Marginal Expected Calibration Error, `NaN` when no detections
    /// survive filtering.
    pub ece: f64,
    /// Marginal Maximum Calibration Error, `NaN` when no detections
    /// survive filtering.
    pub mce: f64,
    /// Total detections that contributed to the marginal histogram.
    pub n_detections: u64,
    /// Effective bin count after Quirk **P5** dedup. May be less
    /// than [`CalibrationParams::n_bins`].
    pub effective_n_bins: usize,
    /// Columnar reliability table; row count equals
    /// [`Self::effective_n_bins`].
    pub reliability: ReliabilityTable,
    /// Per-class breakdown when [`CalibrationParams::per_class`]
    /// was `true`, else `None`.
    pub per_class: Option<PerClassTable>,
}

/// Outcome of one filtered detection contributing to the histogram.
///
/// Encoded as `(score, correct)` with `correct` in `{0.0, 1.0}` so
/// the per-bin sum can route through
/// [`crate::summarize::pairwise_sum`] without a separate integer
/// path. Class id is carried alongside for per-class slicing.
#[derive(Debug, Clone, Copy)]
struct Detection {
    score: f64,
    correct: f64,
    class: u32,
}

/// Collect the filtered detections from the cell store at the
/// requested IoU index. Implements Quirks **P2** (ignore-region drop)
/// and **P3** (min_score cutoff).
fn collect_detections(
    eval_imgs: &[Option<Box<PerImageEval>>],
    n_categories: usize,
    n_area_ranges: usize,
    iou_index: usize,
    min_score: f64,
) -> Result<Vec<Detection>, EvalError> {
    let n_i = if n_categories == 0 || n_area_ranges == 0 {
        0
    } else {
        eval_imgs.len() / (n_categories * n_area_ranges)
    };
    let expected = n_categories * n_area_ranges * n_i;
    if eval_imgs.len() != expected {
        return Err(EvalError::DimensionMismatch {
            detail: format!(
                "eval_imgs len {} != n_categories({}) * n_area_ranges({}) * n_images({}) = {}",
                eval_imgs.len(),
                n_categories,
                n_area_ranges,
                n_i,
                expected
            ),
        });
    }

    let mut out: Vec<Detection> = Vec::new();
    // Calibration only consults the `all` area bucket (a = 0), per
    // ADR-0018 Shape 1.
    let area_idx: usize = 0;
    if n_area_ranges == 0 || area_idx >= n_area_ranges {
        return Ok(out);
    }
    for k in 0..n_categories {
        let nk = k * n_area_ranges * n_i;
        let na = area_idx * n_i;
        for i in 0..n_i {
            let cell = match eval_imgs[nk + na + i].as_deref() {
                Some(c) => c,
                None => continue,
            };
            validate_cell(cell, iou_index)?;
            let n_d = cell.dt_scores.len();
            for d in 0..n_d {
                if cell.dt_ignore[(iou_index, d)] {
                    // Quirk P2 — drop ignore-region detections from
                    // the histogram entirely.
                    continue;
                }
                let s = cell.dt_scores[d];
                if !s.is_finite() {
                    return Err(EvalError::NonFinite {
                        context: "calibration::dt_scores",
                    });
                }
                if s < min_score {
                    // Quirk P3 — DETR-aware min-score cutoff.
                    continue;
                }
                let correct = if cell.dt_matched[(iou_index, d)] {
                    1.0_f64
                } else {
                    0.0_f64
                };
                let class: u32 = u32::try_from(k).map_err(|_| EvalError::InvalidConfig {
                    detail: format!("class index {k} does not fit in u32"),
                })?;
                out.push(Detection {
                    score: s,
                    correct,
                    class,
                });
            }
        }
    }
    Ok(out)
}

fn validate_cell(cell: &PerImageEval, iou_index: usize) -> Result<(), EvalError> {
    if cell.dt_matched.shape() != cell.dt_ignore.shape() {
        return Err(EvalError::DimensionMismatch {
            detail: format!(
                "PerImageEval.dt_matched {:?} != dt_ignore {:?}",
                cell.dt_matched.shape(),
                cell.dt_ignore.shape()
            ),
        });
    }
    if cell.dt_matched.ncols() != cell.dt_scores.len() {
        return Err(EvalError::DimensionMismatch {
            detail: format!(
                "PerImageEval.dt_matched cols {} != dt_scores len {}",
                cell.dt_matched.ncols(),
                cell.dt_scores.len()
            ),
        });
    }
    if iou_index >= cell.dt_matched.nrows() {
        return Err(EvalError::InvalidConfig {
            detail: format!(
                "iou_index {iou_index} out of range for PerImageEval with T={}",
                cell.dt_matched.nrows()
            ),
        });
    }
    Ok(())
}

/// Build bin edges for the marginal histogram. Returns `(edges,
/// effective_n_bins)`. Empty input yields an empty edge list and
/// `effective_n_bins == 0`.
fn build_edges(
    scores_sorted: &[f64],
    n_bins: usize,
    binning: Binning,
    min_score: f64,
) -> (Vec<f64>, usize) {
    if scores_sorted.is_empty() || n_bins == 0 {
        return (Vec::new(), 0);
    }
    let edges = match binning {
        Binning::Quantile => {
            let n_bins_f = n_bins as f64;
            let qs: Vec<f64> = (0..=n_bins).map(|i| (i as f64) / n_bins_f).collect();
            quantile_linear(scores_sorted, &qs)
        }
        Binning::EqualWidth => {
            let observed_min = scores_sorted[0];
            let lo = if min_score > observed_min {
                min_score
            } else {
                observed_min
            };
            // Degenerate ladder (e.g., all-1.0 detector) collapses in
            // the dedup step below.
            crate::parity::linspace(lo, 1.0_f64, n_bins + 1)
        }
    };
    // Quirk P5 — dedupe consecutive identical edges.
    let mut deduped: Vec<f64> = Vec::with_capacity(edges.len());
    for e in edges {
        if let Some(&last) = deduped.last() {
            if last == e {
                continue;
            }
        }
        deduped.push(e);
    }
    let effective = if deduped.len() < 2 {
        0
    } else {
        deduped.len() - 1
    };
    (deduped, effective)
}

/// Assign a score to one of `effective_n_bins` bins given an edge
/// vector. The last bin is right-inclusive (`score <= edges[last]`)
/// to keep `score == 1.0` (and `score == max_observed` under
/// quantile binning) inside the histogram. Returns `None` when no
/// bin matches (e.g., empty edges or score outside the constructed
/// range — the caller drops the detection).
fn assign_bin(score: f64, edges: &[f64]) -> Option<usize> {
    if edges.len() < 2 {
        return None;
    }
    let last_edge = edges[edges.len() - 1];
    let first_edge = edges[0];
    if score < first_edge || score > last_edge {
        return None;
    }
    // Walk the (small) edge ladder. For large n_bins a binary search
    // would help, but n_bins is bounded by ~15 in practice.
    for b in 0..(edges.len() - 1) {
        let lo = edges[b];
        let hi = edges[b + 1];
        let in_bin = if b + 1 == edges.len() - 1 {
            score >= lo && score <= hi
        } else {
            score >= lo && score < hi
        };
        if in_bin {
            return Some(b);
        }
    }
    None
}

/// Wilson 95% CI on `(correct, count)`. `count > 0` precondition.
fn wilson_ci(correct: f64, count: u64) -> (f64, f64) {
    let n = count as f64;
    let phat = correct / n;
    let z = Z_95;
    let zz = z * z;
    let denom = 1.0 + zz / n;
    let center = (phat + zz / (2.0 * n)) / denom;
    let margin = (z / denom)
        * (phat * (1.0 - phat) / n + zz / (4.0 * n * n))
            .max(0.0)
            .sqrt();
    (center - margin, center + margin)
}

/// Compute one reliability table + scalar (ece, mce) over a slice of
/// detections under a shared edge ladder.
fn build_reliability(
    detections: &[Detection],
    edges: &[f64],
    effective_n_bins: usize,
    confidence: ConfidenceKind,
) -> Result<(ReliabilityTable, f64, f64), EvalError> {
    let mut bin_id: Vec<u32> = Vec::with_capacity(effective_n_bins);
    let mut score_lo: Vec<f64> = Vec::with_capacity(effective_n_bins);
    let mut score_hi: Vec<f64> = Vec::with_capacity(effective_n_bins);
    let mut mean_score: Vec<f64> = Vec::with_capacity(effective_n_bins);
    let mut accuracy: Vec<f64> = Vec::with_capacity(effective_n_bins);
    let mut count: Vec<u64> = Vec::with_capacity(effective_n_bins);
    let mut gap: Vec<f64> = Vec::with_capacity(effective_n_bins);
    let mut ci_lo: Vec<f64> = Vec::with_capacity(effective_n_bins);
    let mut ci_hi: Vec<f64> = Vec::with_capacity(effective_n_bins);

    if effective_n_bins == 0 || edges.len() < 2 {
        return Ok((
            ReliabilityTable {
                bin_id,
                score_lo,
                score_hi,
                mean_score,
                accuracy,
                count,
                gap,
                ci_lo,
                ci_hi,
            },
            f64::NAN,
            f64::NAN,
        ));
    }

    // Group detection scores / correctness by bin.
    let mut per_bin_scores: Vec<Vec<f64>> = vec![Vec::new(); effective_n_bins];
    let mut per_bin_correct: Vec<Vec<f64>> = vec![Vec::new(); effective_n_bins];
    for det in detections {
        if let Some(b) = assign_bin(det.score, edges) {
            per_bin_scores[b].push(det.score);
            per_bin_correct[b].push(det.correct);
        }
    }

    let total_n: u64 = per_bin_scores.iter().map(|v| v.len() as u64).sum();
    let total_n_f = total_n as f64;

    let mut ece_acc = 0.0_f64;
    let mut mce_acc: f64 = 0.0_f64;
    let mut any_nonempty = false;

    for b in 0..effective_n_bins {
        let scores_b = &per_bin_scores[b];
        let correct_b = &per_bin_correct[b];
        let n_b = scores_b.len() as u64;
        bin_id.push(u32::try_from(b).map_err(|_| EvalError::InvalidConfig {
            detail: format!("bin id {b} does not fit in u32"),
        })?);
        score_lo.push(edges[b]);
        score_hi.push(edges[b + 1]);
        count.push(n_b);
        if n_b == 0 {
            mean_score.push(f64::NAN);
            accuracy.push(f64::NAN);
            gap.push(f64::NAN);
            ci_lo.push(f64::NAN);
            ci_hi.push(f64::NAN);
            continue;
        }
        any_nonempty = true;
        let sum_s = pairwise_sum(scores_b);
        let sum_c = pairwise_sum(correct_b);
        let n_b_f = n_b as f64;
        let mean_s = sum_s / n_b_f;
        let acc = sum_c / n_b_f;
        let g = acc - mean_s;
        mean_score.push(mean_s);
        accuracy.push(acc);
        gap.push(g);
        match confidence {
            ConfidenceKind::Wilson => {
                let (lo_ci, hi_ci) = wilson_ci(sum_c, n_b);
                ci_lo.push(lo_ci);
                ci_hi.push(hi_ci);
            }
            ConfidenceKind::ClopperPearson => {
                // Phase-2 follow-up — documented error path.
                return Err(EvalError::InvalidConfig {
                    detail: "Clopper-Pearson CI not yet implemented; use Wilson".to_string(),
                });
            }
        }
        let abs_gap = g.abs();
        if total_n > 0 {
            ece_acc += (n_b_f / total_n_f) * abs_gap;
        }
        if abs_gap > mce_acc {
            mce_acc = abs_gap;
        }
    }

    let (ece, mce) = if any_nonempty {
        (ece_acc, mce_acc)
    } else {
        (f64::NAN, f64::NAN)
    };

    Ok((
        ReliabilityTable {
            bin_id,
            score_lo,
            score_hi,
            mean_score,
            accuracy,
            count,
            gap,
            ci_lo,
            ci_hi,
        },
        ece,
        mce,
    ))
}

/// Empty-summary helper for the no-detections short-circuit path.
fn empty_summary() -> CalibrationSummary {
    CalibrationSummary {
        ece: f64::NAN,
        mce: f64::NAN,
        n_detections: 0,
        effective_n_bins: 0,
        reliability: ReliabilityTable {
            bin_id: Vec::new(),
            score_lo: Vec::new(),
            score_hi: Vec::new(),
            mean_score: Vec::new(),
            accuracy: Vec::new(),
            count: Vec::new(),
            gap: Vec::new(),
            ci_lo: Vec::new(),
            ci_hi: Vec::new(),
        },
        per_class: None,
    }
}

/// Fold the per-image cell store into a calibration summary (ADR-0018).
///
/// `eval_imgs` has the flattened `(k, a, i)` layout the streaming
/// store and orchestrator already produce
/// ([`crate::stream::PerImageEvalStore::flatten`]); see this module's
/// top-level doc for the index rule. Only the `all` area bucket
/// (`a = 0`) contributes — calibration is not area-faceted (ADR-0018
/// Shape 1 footnote on ADR-0012 D5).
///
/// `parity_mode` is accepted for API symmetry with the rest of
/// `vernier-core`; today the calibration kernel honors strict
/// dispositions for every quirk in `docs/engineering/calibration-quirks.md`,
/// so the flag is recorded but not branched on.
///
/// # Errors
///
/// - [`EvalError::InvalidConfig`] when `params.n_bins == 0`,
///   `params.iou_index >= T` (the T-axis of any populated cell), or
///   `params.confidence == ConfidenceKind::ClopperPearson`
///   (Phase-2 follow-up).
/// - [`EvalError::DimensionMismatch`] when the cell-store length is
///   not a multiple of `n_categories * n_area_ranges` or any cell's
///   `dt_matched` / `dt_ignore` shapes disagree.
/// - [`EvalError::NonFinite`] on a NaN / infinity detection score.
///
/// Empty input *after* filtering is not an error — the returned
/// summary carries `n_detections = 0`, `effective_n_bins = 0`,
/// `ece = NaN`, `mce = NaN`, and empty tables.
pub fn summarize_calibration(
    eval_imgs: &[Option<Box<PerImageEval>>],
    n_categories: usize,
    n_area_ranges: usize,
    params: &CalibrationParams,
    _parity_mode: ParityMode,
) -> Result<CalibrationSummary, EvalError> {
    if params.n_bins == 0 {
        return Err(EvalError::InvalidConfig {
            detail: "calibration n_bins must be > 0".to_string(),
        });
    }
    // Bounds-check iou_index against any populated cell early — this
    // gives a clearer error than waiting for the per-cell walk to
    // index out-of-range.
    for cell in eval_imgs.iter().flatten() {
        let t = cell.dt_matched.nrows();
        if params.iou_index >= t {
            return Err(EvalError::InvalidConfig {
                detail: format!(
                    "calibration iou_index {} out of range for T={t}",
                    params.iou_index
                ),
            });
        }
    }

    let detections = collect_detections(
        eval_imgs,
        n_categories,
        n_area_ranges,
        params.iou_index,
        params.min_score,
    )?;

    if detections.is_empty() {
        return Ok(empty_summary());
    }

    let mut scores_sorted: Vec<f64> = detections.iter().map(|d| d.score).collect();
    scores_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let (edges, effective_n_bins) = build_edges(
        &scores_sorted,
        params.n_bins,
        params.binning,
        params.min_score,
    );

    let (reliability, ece, mce) =
        build_reliability(&detections, &edges, effective_n_bins, params.confidence)?;
    let n_detections: u64 = reliability.count.iter().sum();

    let per_class = if params.per_class {
        Some(build_per_class(
            &detections,
            &edges,
            effective_n_bins,
            params.confidence,
        )?)
    } else {
        None
    };

    Ok(CalibrationSummary {
        ece,
        mce,
        n_detections,
        effective_n_bins,
        reliability,
        per_class,
    })
}

/// Build the per-class table sharing the marginal edge ladder.
///
/// Sharing edges keeps the per-class breakdown comparable to the
/// marginal table (same x-axis on a reliability diagram). Per-class
/// `n` sums to the marginal `n_detections` since each detection
/// belongs to exactly one class.
fn build_per_class(
    detections: &[Detection],
    edges: &[f64],
    effective_n_bins: usize,
    confidence: ConfidenceKind,
) -> Result<PerClassTable, EvalError> {
    // Group detections by class.
    let mut by_class: std::collections::BTreeMap<u32, Vec<Detection>> =
        std::collections::BTreeMap::new();
    for det in detections {
        by_class.entry(det.class).or_default().push(*det);
    }

    let mut class_id: Vec<u32> = Vec::with_capacity(by_class.len());
    let mut ece_col: Vec<f64> = Vec::with_capacity(by_class.len());
    let mut mce_col: Vec<f64> = Vec::with_capacity(by_class.len());
    let mut n_col: Vec<u64> = Vec::with_capacity(by_class.len());

    for (k, dets) in by_class {
        let (table, ece_k, mce_k) = build_reliability(&dets, edges, effective_n_bins, confidence)?;
        let n_k: u64 = table.count.iter().sum();
        class_id.push(k);
        ece_col.push(ece_k);
        mce_col.push(mce_k);
        n_col.push(n_k);
    }

    Ok(PerClassTable {
        class_id,
        ece: ece_col,
        mce: mce_col,
        n: n_col,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // --- Cell-construction helpers -----------------------------------

    /// Build a 1-image, n_categories-class grid of cells. Each entry of
    /// `per_class_data` is `(scores, matched, ignored)` for category k
    /// at iou_index=0 (T=1).
    fn build_grid(
        per_class_data: &[(Vec<f64>, Vec<bool>, Vec<bool>)],
    ) -> Vec<Option<Box<PerImageEval>>> {
        // K = per_class_data.len(); A = 1 (only `all`); I = 1.
        let mut grid: Vec<Option<Box<PerImageEval>>> = Vec::new();
        for (scores, matched, ignored) in per_class_data {
            let d = scores.len();
            assert_eq!(matched.len(), d);
            assert_eq!(ignored.len(), d);
            if d == 0 {
                grid.push(None);
                continue;
            }
            let mut dt_matched: Array2<bool> = Array2::from_elem((1, d), false);
            let mut dt_ignore: Array2<bool> = Array2::from_elem((1, d), false);
            for (j, &m) in matched.iter().enumerate() {
                dt_matched[(0, j)] = m;
            }
            for (j, &ig) in ignored.iter().enumerate() {
                dt_ignore[(0, j)] = ig;
            }
            grid.push(Some(Box::new(PerImageEval {
                dt_scores: scores.clone(),
                dt_matched,
                dt_ignore,
                gt_ignore: vec![],
            })));
        }
        grid
    }

    fn default_params() -> CalibrationParams {
        // Drop min_score for tests so we control what gets filtered.
        CalibrationParams {
            min_score: 0.0,
            ..CalibrationParams::default()
        }
    }

    // --- Tests --------------------------------------------------------

    #[test]
    fn n_bins_zero_returns_error() {
        let grid = build_grid(&[(vec![0.5], vec![true], vec![false])]);
        let params = CalibrationParams {
            n_bins: 0,
            ..default_params()
        };
        let err = summarize_calibration(&grid, 1, 1, &params, ParityMode::Strict).unwrap_err();
        match err {
            EvalError::InvalidConfig { detail } => {
                assert!(detail.contains("n_bins"), "got: {detail}");
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn iou_index_out_of_range_returns_error() {
        // Cells carry T=1; iou_index=5 is out of range.
        let grid = build_grid(&[(vec![0.5], vec![true], vec![false])]);
        let params = CalibrationParams {
            iou_index: 5,
            ..default_params()
        };
        let err = summarize_calibration(&grid, 1, 1, &params, ParityMode::Strict).unwrap_err();
        match err {
            EvalError::InvalidConfig { detail } => {
                assert!(detail.contains("iou_index"), "got: {detail}");
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn clopper_pearson_is_phase2_error() {
        let grid = build_grid(&[(vec![0.5, 0.6], vec![true, false], vec![false, false])]);
        let params = CalibrationParams {
            confidence: ConfidenceKind::ClopperPearson,
            n_bins: 2,
            ..default_params()
        };
        let err = summarize_calibration(&grid, 1, 1, &params, ParityMode::Strict).unwrap_err();
        match err {
            EvalError::InvalidConfig { detail } => {
                assert!(detail.contains("Clopper-Pearson"), "got: {detail}");
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn empty_input_returns_empty_summary_not_error() {
        // No cells anywhere.
        let grid: Vec<Option<Box<PerImageEval>>> = vec![None];
        let params = default_params();
        let out = summarize_calibration(&grid, 1, 1, &params, ParityMode::Strict).unwrap();
        assert_eq!(out.n_detections, 0);
        assert_eq!(out.effective_n_bins, 0);
        assert!(out.ece.is_nan());
        assert!(out.mce.is_nan());
        assert!(out.reliability.bin_id.is_empty());
        assert!(out.per_class.is_none());
    }

    #[test]
    fn min_score_cutoff_excludes_low_score_detections() {
        // 4 detections: two below cutoff, two above.
        let grid = build_grid(&[(
            vec![0.01, 0.04, 0.6, 0.9],
            vec![true, false, true, true],
            vec![false, false, false, false],
        )]);
        let params = CalibrationParams {
            min_score: 0.05,
            n_bins: 2,
            ..CalibrationParams::default()
        };
        let out = summarize_calibration(&grid, 1, 1, &params, ParityMode::Strict).unwrap();
        // Only the two high-score detections survive.
        assert_eq!(out.n_detections, 2);
    }

    #[test]
    fn ignore_region_detections_drop_from_histogram() {
        // 3 detections: middle one carries dt_ignore=true.
        let grid = build_grid(&[(
            vec![0.4, 0.5, 0.6],
            vec![true, false, true],
            vec![false, true, false],
        )]);
        let params = CalibrationParams {
            n_bins: 2,
            ..default_params()
        };
        let out = summarize_calibration(&grid, 1, 1, &params, ParityMode::Strict).unwrap();
        assert_eq!(out.n_detections, 2);
    }

    #[test]
    fn bin_edge_degeneracy_merges_to_fewer_bins() {
        // All identical scores → all quantile edges collapse to a
        // single value; dedup yields effective_n_bins=0 (only one
        // unique edge survives).
        let grid = build_grid(&[(
            vec![0.5; 5],
            vec![true, true, false, true, false],
            vec![false; 5],
        )]);
        let params = CalibrationParams {
            n_bins: 10,
            binning: Binning::Quantile,
            ..default_params()
        };
        let out = summarize_calibration(&grid, 1, 1, &params, ParityMode::Strict).unwrap();
        // R1: effective_n_bins shrinks. The exact value depends on the
        // unique-edge count; assert it's strictly smaller than the
        // requested 10 and that the reliability table is well-formed.
        assert!(out.effective_n_bins < 10);
        assert_eq!(out.reliability.bin_id.len(), out.effective_n_bins);
        assert_eq!(out.reliability.score_lo.len(), out.effective_n_bins);
    }

    #[test]
    fn zero_count_bin_emits_nan_under_equal_width() {
        // Equal-width [0.0, 1.0] / 4 bins = [0, 0.25, 0.5, 0.75, 1.0].
        // Scores in [0.0, 0.1, 0.95] populate bin 0 and bin 3; bin 1
        // and bin 2 are empty.
        let grid = build_grid(&[(
            vec![0.0, 0.1, 0.95],
            vec![true, false, true],
            vec![false, false, false],
        )]);
        let params = CalibrationParams {
            n_bins: 4,
            binning: Binning::EqualWidth,
            min_score: 0.0,
            ..CalibrationParams::default()
        };
        let out = summarize_calibration(&grid, 1, 1, &params, ParityMode::Strict).unwrap();
        // Find any empty bin and verify NaNs.
        let mut found_empty = false;
        for (b, &c) in out.reliability.count.iter().enumerate() {
            if c == 0 {
                found_empty = true;
                assert!(out.reliability.accuracy[b].is_nan());
                assert!(out.reliability.mean_score[b].is_nan());
                assert!(out.reliability.gap[b].is_nan());
                assert!(out.reliability.ci_lo[b].is_nan());
                assert!(out.reliability.ci_hi[b].is_nan());
            }
        }
        assert!(found_empty, "expected at least one empty bin");
    }

    #[test]
    fn perfect_all_correct_high_score_ece_matches_gap() {
        // Detector emits scores [0.9, 0.95, 0.99], all correct.
        // Under quantile binning with n_bins=1 (or with all scores
        // collapsing into a single non-empty bin), ECE = |acc - mean|.
        // acc = 1.0; mean = (0.9 + 0.95 + 0.99)/3 = 0.94666...; gap =
        // 0.0533...; ECE = same.
        let grid = build_grid(&[(
            vec![0.9, 0.95, 0.99],
            vec![true, true, true],
            vec![false, false, false],
        )]);
        let params = CalibrationParams {
            n_bins: 1,
            binning: Binning::Quantile,
            min_score: 0.0,
            ..CalibrationParams::default()
        };
        let out = summarize_calibration(&grid, 1, 1, &params, ParityMode::Strict).unwrap();
        assert_eq!(out.n_detections, 3);
        let expected_mean = (0.9 + 0.95 + 0.99) / 3.0;
        let expected_ece = (1.0_f64 - expected_mean).abs();
        assert!(
            (out.ece - expected_ece).abs() < 1e-12,
            "ece={} expected~{}",
            out.ece,
            expected_ece
        );
        // MCE on a single populated bin equals ECE.
        assert!((out.mce - expected_ece).abs() < 1e-12);
    }

    #[test]
    fn per_class_breakdown_sums_to_total_detections() {
        // Two classes with distinct calibration profiles. n must sum
        // to marginal n_detections; class_id sorted ascending.
        let grid = build_grid(&[
            (
                vec![0.2, 0.4, 0.6],
                vec![false, false, true],
                vec![false; 3],
            ),
            (vec![0.8, 0.9], vec![true, true], vec![false; 2]),
        ]);
        let params = CalibrationParams {
            n_bins: 2,
            per_class: true,
            min_score: 0.0,
            ..CalibrationParams::default()
        };
        let out = summarize_calibration(&grid, 2, 1, &params, ParityMode::Strict).unwrap();
        let pc = out.per_class.expect("per_class table");
        assert_eq!(pc.class_id, vec![0, 1]);
        let pc_sum: u64 = pc.n.iter().sum();
        assert_eq!(pc_sum, out.n_detections);
        assert_eq!(pc.n.len(), 2);
        assert_eq!(pc.ece.len(), 2);
        assert_eq!(pc.mce.len(), 2);
    }

    #[test]
    fn identical_cells_produce_identical_summaries_iou_type_genericity() {
        // Implicit iou_type-genericity: the kernel reads only
        // dt_scores / dt_matched / dt_ignore. Two hand-built cells
        // with identical values yield identical summaries regardless
        // of the imagined upstream iou_type.
        let grid_a = build_grid(&[(vec![0.3, 0.7], vec![false, true], vec![false, false])]);
        let grid_b = build_grid(&[(vec![0.3, 0.7], vec![false, true], vec![false, false])]);
        let params = CalibrationParams {
            n_bins: 2,
            min_score: 0.0,
            ..CalibrationParams::default()
        };
        let out_a = summarize_calibration(&grid_a, 1, 1, &params, ParityMode::Strict).unwrap();
        let out_b = summarize_calibration(&grid_b, 1, 1, &params, ParityMode::Strict).unwrap();
        assert_eq!(out_a.n_detections, out_b.n_detections);
        assert_eq!(out_a.effective_n_bins, out_b.effective_n_bins);
        assert_eq!(out_a.reliability.count, out_b.reliability.count);
        // ECE / MCE bit-equal because the inputs are identical.
        assert_eq!(out_a.ece.to_bits(), out_b.ece.to_bits());
        assert_eq!(out_a.mce.to_bits(), out_b.mce.to_bits());
    }

    #[test]
    fn wilson_ci_known_values() {
        // Reference (Python; z = scipy.stats.norm.ppf(0.975)):
        //   phat = 0.8, n = 10, z = 1.959963984540054
        //   center = (0.8 + z^2/20) / (1 + z^2/10) = 0.7167401600
        //   margin = (z / (1+z^2/10)) * sqrt(0.8*0.2/10 + z^2/400)
        //          = 0.2265776885
        //   CI = (0.4901624715, 0.9433178485)
        let (lo, hi) = wilson_ci(8.0, 10);
        assert!((lo - 0.490_162_471_5).abs() < 1e-9, "lo={lo}");
        assert!((hi - 0.943_317_848_5).abs() < 1e-9, "hi={hi}");
    }
}
