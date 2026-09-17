//! Per-image evaluation → precision/recall/scores arrays.
//!
//! Mirrors `pycocotools.cocoeval.COCOeval.accumulate` (cocoeval.py
//! lines 315-420). Inputs come from the upstream matching engine
//! (the [`crate::matching`] engine) packaged as one [`PerImageEval`] per
//! `(category, areaRange, image)` cell; outputs are the
//! `(T, R, K, A, M)` precision and `(T, K, A, M)` recall tensors that
//! the summarizer slices into the final 12 stats.
//!
//! ## Quirk dispositions
//!
//! - **A1** (`strict`): the merged-stream sort across one `(K, A, M)`
//!   slice is also a stable mergesort on `-score`, mirroring
//!   `np.argsort(kind='mergesort')` on the concatenated stream.
//! - **C1** (`strict`): recall lookup uses `searchsorted(rc, t,
//!   side='left')` semantics — the leftmost cumulative-recall index
//!   with `rc[i] >= t`.
//! - **C2** (`strict`): right-to-left running max on the precision
//!   array enforces the monotonic precision envelope before
//!   integration.
//! - **C3** (`corrected` implementation, `strict` outputs): the
//!   `try/except` around `dtScoresSorted[pi]` becomes an explicit
//!   bounds check (`pi < n_d`); past the curve we leave `q[ri]` and
//!   `ss[ri]` at `0.0`, matching the silent-skip pycocotools does in
//!   the `except: pass` branch.
//! - **C4** (`strict`): "AR" stored in `recall` is terminal cumulative
//!   recall (the last value of `rc`), not an integral of the
//!   precision/recall curve.
//! - **C5** (`strict`): `(K, A, M)` cells with no eval-img entries OR
//!   `npig == 0` (no non-ignore GTs) leave `precision`/`recall`/`scores`
//!   at the `-1` sentinel; the summarizer filters those before
//!   averaging. Cells that have GTs but no detections (npig > 0,
//!   n_d == 0) reach pycocotools' `q = np.zeros((R,))` write-site and
//!   land at `0.0`, not `-1` — pycocotools writes the zero-initialised
//!   `q` even when the inner `try/except` over an empty `pr` bails on
//!   the first index.
//! - **C7** (`strict`): TP and FP cumsums skip DTs whose `dt_ignore`
//!   flag is set — both B6 (matched-to-ignore) and B7 (out-of-area
//!   unmatched) are folded into `dt_ignore` upstream.
//! - **C8** (`aligned`): precision denominator uses
//!   [`crate::parity::PARITY_EPS`] (= `f64::EPSILON`), bit-equal to
//!   `np.spacing(1)`.
//! - **L1, L2** (`strict`): `iou_thresholds` and `recall_thresholds`
//!   come from [`crate::parity::iou_thresholds`] / [`crate::parity::recall_thresholds`]
//!   and are linspace-built; the accumulator does not assume their
//!   values, only their lengths.
//!
//! Quirks **B7** (out-of-area unmatched DT → `dt_ignore`) and **B6**
//! (DT matched to ignore-GT → `dt_ignore`) are inputs here, not
//! responsibilities. The orchestrator that builds [`PerImageEval`]
//! folds B7 in alongside the matching engine's B6.

use ndarray::{Array2, Array4, Array5, ArrayViewMut3, ArrayViewMut4, Axis};
use rayon::prelude::*;

use crate::error::EvalError;
use crate::parity::{argsort_score_desc, ParityMode, PARITY_EPS};

/// Per `(image, category, areaRange)` slice of evaluation data, in the
/// shape the accumulator consumes.
///
/// Built by the orchestrator from a `MatchResult` (private to the
/// [`crate::matching`] module) plus the
/// per-DT areas needed to apply quirk **B7**. Field orders mirror the
/// matching engine's *sorted* internal orders: `dt_*` rows are
/// score-desc (stable mergesort), `gt_ignore` is ignore-asc.
#[derive(Debug, Clone)]
pub struct PerImageEval {
    /// Detection scores in sorted-DT order. Length `D`.
    pub dt_scores: Vec<f64>,
    /// Per-`(T, D)` match indicator. `true` when the DT matched any GT
    /// at this threshold (regardless of whether the matched GT is an
    /// ignore-GT — that distinction is carried by `dt_ignore`).
    pub dt_matched: Array2<bool>,
    /// Per-`(T, D)` ignore flag. Caller must fold in both B6 (matched
    /// to ignore-GT) and B7 (out-of-area unmatched) before constructing
    /// this struct; the accumulator treats it as authoritative.
    pub dt_ignore: Array2<bool>,
    /// Per-GT ignore flag in sorted-GT order. Length `G`.
    pub gt_ignore: Vec<bool>,
}

/// Inputs to [`accumulate`] that describe the evaluation grid.
///
/// `eval_imgs.len()` must equal `n_categories * n_area_ranges *
/// n_images`, with the layout `eval_imgs[k * A * I + a * I + i]`
/// matching pycocotools' flat indexing of `evalImgs`.
#[derive(Debug, Clone, Copy)]
pub struct AccumulateParams<'p> {
    /// IoU thresholds, length `T`. Use [`crate::parity::iou_thresholds`] for
    /// the canonical 10-point COCO ladder.
    pub iou_thresholds: &'p [f64],
    /// Recall integration thresholds, length `R` (typically 101). Use
    /// [`crate::parity::recall_thresholds`].
    pub recall_thresholds: &'p [f64],
    /// Per-image maxDet caps, length `M`. Pycocotools defaults to
    /// `[1, 10, 100]`. The matching engine should be invoked with the
    /// *largest* of these — the accumulator slices to smaller caps via
    /// `[..max_det]`.
    ///
    /// Must be sorted ascending (quirk **A2** — strict). Pycocotools
    /// silently overwrites `p.maxDets = sorted(p.maxDets)` at
    /// `cocoeval.py:137`, so the M-axis is always laid out
    /// smallest-to-largest. The summarizer's `AR_1 / AR_10 / AR_100`
    /// slot mapping depends on this ordering — passing `[100, 1, 10]`
    /// without sorting would silently swap the slot semantics. Callers
    /// at the FFI boundary use [`sort_max_dets`] to enforce this.
    pub max_dets: &'p [usize],
    /// Number of categories `K` (or `1` when `useCats == 0`).
    pub n_categories: usize,
    /// Number of area ranges `A` (COCO defaults to 4: all/small/medium/
    /// large).
    pub n_area_ranges: usize,
    /// Number of images `I`.
    pub n_images: usize,
}

/// Normalize a `max_dets` ladder to ascending order, in place.
///
/// Mirrors `pycocotools.cocoeval.COCOeval.accumulate`'s opening line
/// (`cocoeval.py:137`):
///
/// ```python
/// p.maxDets = sorted(p.maxDets)
/// ```
///
/// Quirk **A2** (strict). The accumulator's M-axis is laid out in the
/// order of the ladder it receives, and the summarizer's
/// `AR_1 / AR_10 / AR_100` slot mapping is positional — sorting at the
/// param-construction boundary keeps user input order from silently
/// permuting the final stat vector. Stable sort (`Vec::sort`); the
/// ladder is `usize`, so stability matches pycocotools' Python `sorted`
/// (also stable).
pub fn sort_max_dets(max_dets: &mut [usize]) {
    max_dets.sort();
}

/// Output tensors produced by [`accumulate`].
///
/// Cells absent from the dataset (no DTs, or no non-ignore GTs) carry
/// `-1.0` per quirk **C5**. The summarizer filters these before
/// averaging; downstream code that consumes the tensors directly must
/// honor the same convention.
#[derive(Debug, Clone)]
pub struct Accumulated {
    /// Shape `(T, R, K, A, M)`. Right-monotonic precision interpolated
    /// at every recall threshold.
    pub precision: Array5<f64>,
    /// Shape `(T, K, A, M)`. Terminal cumulative recall (quirk **C4**).
    pub recall: Array4<f64>,
    /// Shape `(T, R, K, A, M)`. Detection score at the recall threshold
    /// where each precision sample was taken.
    pub scores: Array5<f64>,
}

/// Accumulate per-image evaluation results into precision / recall /
/// scores tensors.
///
/// The flat `eval_imgs` slice must be laid out as `[k][a][i]` (K-major,
/// then A, then I) — `eval_imgs.len() == K * A * I`.
///
/// # Errors
///
/// Returns [`EvalError::DimensionMismatch`] if `eval_imgs.len()` does
/// not equal `K * A * I`, or if any per-image array shapes disagree
/// with the declared `T` (IoU-threshold count).
pub fn accumulate(
    eval_imgs: &[Option<Box<PerImageEval>>],
    p: AccumulateParams<'_>,
    _parity_mode: ParityMode,
) -> Result<Accumulated, EvalError> {
    let n_t = p.iou_thresholds.len();
    let n_r = p.recall_thresholds.len();
    let n_k = p.n_categories;
    let n_a = p.n_area_ranges;
    let n_m = p.max_dets.len();
    let n_i = p.n_images;

    validate_grid(eval_imgs, n_t, n_k, n_a, n_i)?;

    let mut precision = Array5::<f64>::from_elem((n_t, n_r, n_k, n_a, n_m), -1.0);
    let mut recall = Array4::<f64>::from_elem((n_t, n_k, n_a, n_m), -1.0);
    let mut scores = Array5::<f64>::from_elem((n_t, n_r, n_k, n_a, n_m), -1.0);

    for (k, ((mut p_k, mut r_k), mut s_k)) in precision
        .axis_iter_mut(Axis(2))
        .zip(recall.axis_iter_mut(Axis(1)))
        .zip(scores.axis_iter_mut(Axis(2)))
        .enumerate()
    {
        accumulate_category(eval_imgs, p, k, n_t, n_a, n_i, &mut p_k, &mut r_k, &mut s_k);
    }

    Ok(Accumulated {
        precision,
        recall,
        scores,
    })
}

/// Shape checks shared by [`accumulate`] and [`accumulate_parallel`].
///
/// # Errors
///
/// [`EvalError::DimensionMismatch`] when the grid length disagrees with
/// `K * A * I`, or when a per-image array's shape disagrees with `T`.
fn validate_grid(
    eval_imgs: &[Option<Box<PerImageEval>>],
    n_t: usize,
    n_k: usize,
    n_a: usize,
    n_i: usize,
) -> Result<(), EvalError> {
    let expected = n_k * n_a * n_i;
    if eval_imgs.len() != expected {
        return Err(EvalError::DimensionMismatch {
            detail: format!(
                "eval_imgs len {} != n_categories({}) * n_area_ranges({}) * n_images({}) = {}",
                eval_imgs.len(),
                n_k,
                n_a,
                n_i,
                expected
            ),
        });
    }

    for cell in eval_imgs.iter().flatten() {
        if cell.dt_matched.shape() != cell.dt_ignore.shape() {
            return Err(EvalError::DimensionMismatch {
                detail: format!(
                    "PerImageEval.dt_matched {:?} != dt_ignore {:?}",
                    cell.dt_matched.shape(),
                    cell.dt_ignore.shape()
                ),
            });
        }
        if cell.dt_matched.nrows() != n_t {
            return Err(EvalError::DimensionMismatch {
                detail: format!(
                    "PerImageEval row count {} != iou_thresholds len {}",
                    cell.dt_matched.nrows(),
                    n_t
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
    }

    Ok(())
}

/// Rayon sibling of [`accumulate`], fanned out over the category axis.
///
/// Categories own disjoint slices of all three output tensors and share
/// no accumulator, so this is bit-identical to the sequential walk
/// rather than merely equivalent: no float reduction crosses a thread
/// boundary, and each cell's arithmetic is the same
/// `accumulate_category` call in the same order.
///
/// Worth the fan-out only when the category axis is long. `accumulate`
/// walks `K * A * M` cells and gathers `I` images for each; on COCO
/// (80 categories) that is ~70 ms, but Objects365 val (365 categories,
/// 80 000 images) spends ~2.1 s there — a fifth of the whole
/// evaluation, all of it serial. Callers on the sequential path
/// (`num_threads=None`) keep calling [`accumulate`] and never enter
/// rayon (ADR-0047).
///
/// # Errors
///
/// Same validation as [`accumulate`]: propagates
/// [`EvalError::DimensionMismatch`] on a grid whose length or per-image
/// array shapes disagree with the declared axes.
pub fn accumulate_parallel(
    eval_imgs: &[Option<Box<PerImageEval>>],
    p: AccumulateParams<'_>,
    parity_mode: ParityMode,
) -> Result<Accumulated, EvalError> {
    let n_t = p.iou_thresholds.len();
    let n_r = p.recall_thresholds.len();
    let n_k = p.n_categories;
    let n_a = p.n_area_ranges;
    let n_m = p.max_dets.len();
    let n_i = p.n_images;

    validate_grid(eval_imgs, n_t, n_k, n_a, n_i)?;
    let _ = parity_mode;

    let mut precision = Array5::<f64>::from_elem((n_t, n_r, n_k, n_a, n_m), -1.0);
    let mut recall = Array4::<f64>::from_elem((n_t, n_k, n_a, n_m), -1.0);
    let mut scores = Array5::<f64>::from_elem((n_t, n_r, n_k, n_a, n_m), -1.0);

    // One mutable view per category, collected up front. `ArrayViewMut`
    // is `Send`, and the views are disjoint by construction, so rayon
    // can hold one per worker without any interior mutability.
    let p_views: Vec<ArrayViewMut4<'_, f64>> = precision.axis_iter_mut(Axis(2)).collect();
    let r_views: Vec<ArrayViewMut3<'_, f64>> = recall.axis_iter_mut(Axis(1)).collect();
    let s_views: Vec<ArrayViewMut4<'_, f64>> = scores.axis_iter_mut(Axis(2)).collect();

    p_views
        .into_par_iter()
        .zip(r_views)
        .zip(s_views)
        .enumerate()
        .for_each(|(k, ((mut p_k, mut r_k), mut s_k))| {
            accumulate_category(eval_imgs, p, k, n_t, n_a, n_i, &mut p_k, &mut r_k, &mut s_k);
        });

    Ok(Accumulated {
        precision,
        recall,
        scores,
    })
}

/// One score-descending permutation, shared by every `(area range,
/// max-det)` cell of a category (ADR-0052).
///
/// `scores` is the *cap stream*: the concatenation, in image order, of
/// `cell.dt_scores[..min(len, cap)]` for the largest `maxDet` in the
/// ladder. `perm` is its stable score-descending permutation. Every
/// smaller `maxDet` stream is an induced subsequence of the cap stream,
/// so its permutation is `perm` filtered rather than a second sort.
///
/// Quirk **A1**'s disposition row anticipates an opt-in `corrected`
/// mode that breaks score ties by `(-score, ann_id)` instead of by
/// input position. When it lands it has to be threaded through
/// `SortPlan::build`, which is the category's single sort: a per-cell
/// re-tie-break bolted on after the plan exists would desync the
/// derived streams from the cap permutation they filter, and Claim 2's
/// "restriction of a stable order" argument only holds while every cell
/// of the category is ordered by the same key.
struct SortPlan {
    /// Largest `maxDet` this plan was built for.
    cap: usize,
    /// Per-cell take at `cap`, in cell order.
    takes: Vec<usize>,
    /// Concatenated cap-stream scores.
    scores: Vec<f64>,
    /// Stable score-descending permutation of `scores` (quirk **A1**).
    perm: Vec<usize>,
    /// Whether the cap stream carries a `NaN`.
    ///
    /// `argsort_score_desc` degrades to `partial_cmp(..).unwrap_or(Equal)`
    /// there, which is *intransitive* — so `(-score, position)` is no
    /// longer a total order and neither claim's proof applies. Both
    /// derivations are switched off when this is set: `matches` never
    /// reuses the plan across area ranges (Claim 1) and
    /// `DerivedStream::fill_from` sorts each `maxDet` stream from
    /// scratch (Claim 2).
    has_nan: bool,
}

impl SortPlan {
    /// Build the cap stream for `cells` and sort it once.
    fn build(cells: &[&PerImageEval], cap: usize) -> Self {
        let mut takes: Vec<usize> = Vec::with_capacity(cells.len());
        let mut total = 0usize;
        for cell in cells {
            let take = cell.dt_scores.len().min(cap);
            takes.push(take);
            total += take;
        }
        let mut scores: Vec<f64> = Vec::with_capacity(total);
        for (cell, &take) in cells.iter().zip(&takes) {
            scores.extend_from_slice(&cell.dt_scores[..take]);
        }
        let perm = argsort_score_desc(&scores);
        let has_nan = scores.iter().any(|s| s.is_nan());
        Self {
            cap,
            takes,
            scores,
            perm,
            has_nan,
        }
    }

    /// Whether this plan's cap stream is the one `cells` would build.
    ///
    /// ADR-0052 Claim 1 says it always is for grids built by vernier's
    /// own evaluate paths: `evaluate_cell` runs the four area ranges
    /// over one `CellBuffers`, so the `PerImageEval`s of a
    /// `(category, image)` pair share a `dt_scores` vector and only
    /// their ignore flags differ. The check is what makes that a
    /// verified precondition rather than an assumption — `accumulate`
    /// is public and its grid can be built by hand.
    ///
    /// A plan built from a `NaN` stream is never reused: its
    /// permutation came out of an intransitive comparator, so "the same
    /// stream sorts the same way" is the only property it has, and this
    /// keeps the reuse surface at zero rather than resting on it.
    ///
    /// The per-element comparison is on **bit patterns**, not `==`:
    /// `-0.0 == 0.0` holds, so a plain `==` would let an area range
    /// that presents `-0.0` reuse a plan holding `+0.0` and then emit
    /// the plan's `+0.0` into the scores tensor. The two sort
    /// identically (`partial_cmp` calls them `Equal`), so this costs
    /// nothing in ordering — it only keeps the emitted score the one
    /// that cell actually carried, which the module's bit-exactness
    /// premise asks for.
    fn matches(&self, cells: &[&PerImageEval], cap: usize) -> bool {
        if self.has_nan || cap != self.cap || cells.len() != self.takes.len() {
            return false;
        }
        let mut cursor = 0usize;
        for (cell, &take) in cells.iter().zip(&self.takes) {
            if cell.dt_scores.len().min(cap) != take {
                return false;
            }
            let cached = &self.scores[cursor..cursor + take];
            if cell.dt_scores[..take]
                .iter()
                .zip(cached)
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return false;
            }
            cursor += take;
        }
        true
    }
}

/// Reusable buffers for a `maxDet < cap` stream derived from a
/// [`SortPlan`]. One per category walk, refilled per `(area, maxDet)`.
#[derive(Default)]
struct DerivedStream {
    takes: Vec<usize>,
    scores: Vec<f64>,
    /// Cap-stream position → derived-stream position, [`Self::DROPPED`]
    /// where the cap stream's entry is past this `maxDet`.
    map: Vec<usize>,
    perm: Vec<usize>,
}

impl DerivedStream {
    const DROPPED: usize = usize::MAX;

    /// Refill from `plan`, keeping the first `max_det` detections of
    /// each cell.
    ///
    /// The kept positions carry strictly increasing derived-stream
    /// indices, so filtering `plan.perm` through `map` preserves both
    /// the score order and the input-order tie-break — ADR-0052 Claim 2.
    ///
    /// That argument needs `(-score, position)` to be a *total* order,
    /// which a `NaN` in the stream destroys: `argsort_score_desc` falls
    /// back to `Ordering::Equal` on an incomparable pair, and the
    /// resulting comparator is intransitive, so a restriction of the
    /// cap permutation is no longer the permutation the restricted
    /// stream sorts to. When `plan.has_nan` is set this sorts the
    /// derived stream directly instead — the pre-ADR-0052 path, which
    /// is by definition what that `(a, m)` cell would have produced.
    fn fill_from(&mut self, plan: &SortPlan, max_det: usize) {
        let Self {
            takes,
            scores,
            map,
            perm,
        } = self;
        takes.clear();
        scores.clear();
        perm.clear();
        map.clear();
        let derive = !plan.has_nan;
        if derive {
            map.resize(plan.scores.len(), Self::DROPPED);
        }

        let mut cap_cursor = 0usize;
        let mut out_cursor = 0usize;
        for &take in &plan.takes {
            let keep = take.min(max_det);
            if derive {
                for j in 0..keep {
                    map[cap_cursor + j] = out_cursor + j;
                }
            }
            scores.extend_from_slice(&plan.scores[cap_cursor..cap_cursor + keep]);
            takes.push(keep);
            cap_cursor += take;
            out_cursor += keep;
        }

        if derive {
            perm.extend(plan.perm.iter().filter_map(|&i| {
                let mapped = map[i];
                (mapped != Self::DROPPED).then_some(mapped)
            }));
        } else {
            perm.extend(argsort_score_desc(scores));
        }
    }
}

/// The score stream one `(area range, max-det)` cell sweeps over.
#[derive(Clone, Copy)]
struct StreamView<'s> {
    /// Per-cell take, in cell order. Length equals the cell count.
    takes: &'s [usize],
    /// Concatenated scores, cell-major.
    scores: &'s [f64],
    /// Stable score-descending permutation of `scores`.
    perm: &'s [usize],
}

/// Every `(area range, max-det)` cell of one category, writing into
/// that category's slice of the output tensors.
///
/// Split out of [`accumulate`] so the sequential walk and
/// [`accumulate_parallel`] run byte-identical arithmetic — the only
/// difference between them is which thread holds the category.
///
/// Per ADR-0052 the category sorts its detection stream once: the plan
/// survives across area ranges while they present the same stream, and
/// the shorter `maxDet` ladders filter it instead of re-sorting.
#[allow(clippy::too_many_arguments)]
fn accumulate_category(
    eval_imgs: &[Option<Box<PerImageEval>>],
    p: AccumulateParams<'_>,
    k: usize,
    n_t: usize,
    n_a: usize,
    n_i: usize,
    precision: &mut ArrayViewMut4<'_, f64>,
    recall: &mut ArrayViewMut3<'_, f64>,
    scores: &mut ArrayViewMut4<'_, f64>,
) {
    let nk = k * n_a * n_i;
    // A2 keeps `max_dets` ascending, so the cap is the last entry; take
    // the max anyway rather than depend on it here.
    let cap = p.max_dets.iter().copied().max().unwrap_or(0);
    let mut plan: Option<SortPlan> = None;
    let mut derived = DerivedStream::default();
    let mut cells: Vec<&PerImageEval> = Vec::with_capacity(n_i);

    for a in 0..n_a {
        let na = a * n_i;
        cells.clear();
        cells.extend((0..n_i).filter_map(|i| eval_imgs[nk + na + i].as_deref()));
        if cells.is_empty() {
            continue;
        }
        let npig: usize = cells
            .iter()
            .map(|e| e.gt_ignore.iter().filter(|&&ig| !ig).count())
            .sum();
        if npig == 0 {
            continue;
        }

        // Drop a plan this area range's stream no longer matches, then
        // build on demand: one `SortPlan::build` call site, and no
        // unreachable `None` arm to explain.
        if plan.as_ref().is_some_and(|plan| !plan.matches(&cells, cap)) {
            plan = None;
        }
        let plan = plan.get_or_insert_with(|| SortPlan::build(&cells, cap));

        for (m, &max_det) in p.max_dets.iter().enumerate() {
            let stream = if max_det >= cap {
                StreamView {
                    takes: &plan.takes,
                    scores: &plan.scores,
                    perm: &plan.perm,
                }
            } else {
                derived.fill_from(plan, max_det);
                StreamView {
                    takes: &derived.takes,
                    scores: &derived.scores,
                    perm: &derived.perm,
                }
            };
            accumulate_cell(
                &cells,
                stream,
                npig,
                n_t,
                p.recall_thresholds,
                a,
                m,
                precision,
                recall,
                scores,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_cell(
    cells: &[&PerImageEval],
    stream: StreamView<'_>,
    npig: usize,
    n_t: usize,
    recall_thresholds: &[f64],
    a: usize,
    m: usize,
    precision: &mut ArrayViewMut4<'_, f64>,
    recall: &mut ArrayViewMut3<'_, f64>,
    scores: &mut ArrayViewMut4<'_, f64>,
) {
    let n_d = stream.scores.len();
    if n_d == 0 {
        // No detections, but npig > 0. Pycocotools (cocoeval.py:442-465)
        // initializes `q = np.zeros((R,))` and `ss = np.zeros((R,))`
        // per (t, k, a, m) cell, then overwrites with `pr[pi]` /
        // `dtScoresSorted[pi]` inside a `try/except` — when those
        // arrays are empty (the n_d == 0 case here) the `try` raises
        // on the first index and `q` / `ss` get written verbatim into
        // `precision[t,:,k,a,m]` / `scores[t,:,k,a,m]`, leaving each
        // cell at `0.0`, not at the `-1` sentinel. The `-1` sentinel
        // is reserved for cells that never reach the assignment site
        // at all (`len(E) == 0` or `npig == 0` — the two `continue`s
        // above this block). Mirror that here so the surviving cells
        // get 0.0, not -1.
        for t in 0..n_t {
            recall[(t, a, m)] = 0.0;
            for ri in 0..recall_thresholds.len() {
                precision[(t, ri, a, m)] = 0.0;
                scores[(t, ri, a, m)] = 0.0;
            }
        }
        return;
    }

    let npig_f = npig as f64;
    let mut rc = vec![0.0_f64; n_d];
    let mut pr = vec![0.0_f64; n_d];
    let mut dtm = vec![false; n_d];
    let mut dtg = vec![false; n_d];

    for t in 0..n_t {
        let mut cursor = 0;
        for (cell, &take) in cells.iter().zip(stream.takes) {
            let m_row = cell.dt_matched.row(t);
            let g_row = cell.dt_ignore.row(t);
            for d in 0..take {
                dtm[cursor] = m_row[d];
                dtg[cursor] = g_row[d];
                cursor += 1;
            }
        }

        // C7: cumulative TP/FP exclude ignore-tagged DTs.
        let mut tp = 0.0_f64;
        let mut fp = 0.0_f64;
        for (out_idx, &src_idx) in stream.perm.iter().enumerate() {
            if !dtg[src_idx] {
                if dtm[src_idx] {
                    tp += 1.0;
                } else {
                    fp += 1.0;
                }
            }
            rc[out_idx] = tp / npig_f;
            pr[out_idx] = tp / (tp + fp + PARITY_EPS);
        }

        // C4: terminal cumulative recall.
        recall[(t, a, m)] = rc[n_d - 1];

        // C2: right-to-left running max on precision (envelope).
        for j in (1..n_d).rev() {
            if pr[j] > pr[j - 1] {
                pr[j - 1] = pr[j];
            }
        }

        // C1 + C3: searchsorted-left + bounds-check. Past the curve,
        // slots are filled with 0.0 — overwriting the -1 sentinel so the
        // summarizer's `s > -1` filter keeps them.
        let mut p_lane = precision
            .index_axis_mut(Axis(0), t)
            .index_axis_move(Axis(1), a)
            .index_axis_move(Axis(1), m);
        let mut s_lane = scores
            .index_axis_mut(Axis(0), t)
            .index_axis_move(Axis(1), a)
            .index_axis_move(Axis(1), m);
        for (ri, &target) in recall_thresholds.iter().enumerate() {
            let pi = rc.partition_point(|&v| v < target);
            if pi < n_d {
                p_lane[ri] = pr[pi];
                s_lane[ri] = stream.scores[stream.perm[pi]];
            } else {
                p_lane[ri] = 0.0;
                s_lane[ri] = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn one_threshold_eval(
        scores: Vec<f64>,
        matched: Vec<bool>,
        ignore: Vec<bool>,
        gt_ignore: Vec<bool>,
    ) -> PerImageEval {
        let n = scores.len();
        let dt_matched =
            Array2::from_shape_vec((1, n), matched).expect("dt_matched shape mismatch");
        let dt_ignore = Array2::from_shape_vec((1, n), ignore).expect("dt_ignore shape mismatch");
        PerImageEval {
            dt_scores: scores,
            dt_matched,
            dt_ignore,
            gt_ignore,
        }
    }

    fn params<'p>(
        iou: &'p [f64],
        rec: &'p [f64],
        max_dets: &'p [usize],
        n_images: usize,
    ) -> AccumulateParams<'p> {
        AccumulateParams {
            iou_thresholds: iou,
            recall_thresholds: rec,
            max_dets,
            n_categories: 1,
            n_area_ranges: 1,
            n_images,
        }
    }

    #[test]
    fn empty_grid_returns_all_sentinel() {
        let p = params(&[0.5], &[0.0, 0.5, 1.0], &[100], 0);
        let out = accumulate(&[], p, ParityMode::Strict).unwrap();
        assert!(out.precision.iter().all(|&v| v == -1.0));
        assert!(out.recall.iter().all(|&v| v == -1.0));
    }

    #[test]
    fn no_dt_with_real_gt_yields_zero_recall_and_zero_precision() {
        // Pycocotools' `cocoeval.py:442-465` writes `q = np.zeros((R,))`
        // into `precision[t,:,k,a,m]` for every (k, a, m) cell that
        // passes the `len(E) > 0` and `npig > 0` gates — including the
        // n_d == 0 sub-case, where the try/except over an empty `pr`
        // leaves `q` untouched (all zeros) and still writes it. The
        // `-1` sentinel is reserved for cells skipped before that
        // assignment (`len(E) == 0` or `npig == 0`); a cell with real
        // GT but no detections reaches the assignment and lands at 0.
        let cell = PerImageEval {
            dt_scores: vec![],
            dt_matched: Array2::<bool>::default((2, 0)),
            dt_ignore: Array2::<bool>::default((2, 0)),
            gt_ignore: vec![false],
        };
        let p = params(&[0.5, 0.75], &[0.0, 0.5, 1.0], &[100], 1);
        let out = accumulate(&[Some(Box::new(cell))], p, ParityMode::Strict).unwrap();
        assert_eq!(out.recall[(0, 0, 0, 0)], 0.0);
        assert_eq!(out.recall[(1, 0, 0, 0)], 0.0);
        for ri in 0..3 {
            assert_eq!(out.precision[(0, ri, 0, 0, 0)], 0.0);
            assert_eq!(out.precision[(1, ri, 0, 0, 0)], 0.0);
        }
    }

    #[test]
    fn cell_with_only_ignore_gts_skips_entirely() {
        // npig == 0 short-circuit: outputs stay at -1 (no recall write).
        let cell = one_threshold_eval(vec![0.9], vec![true], vec![true], vec![true]);
        let p = params(&[0.5], &[0.0, 0.5, 1.0], &[100], 1);
        let out = accumulate(&[Some(Box::new(cell))], p, ParityMode::Strict).unwrap();
        assert_eq!(out.recall[(0, 0, 0, 0)], -1.0);
        assert_eq!(out.precision[(0, 0, 0, 0, 0)], -1.0);
    }

    #[test]
    fn perfect_match_yields_ap_one_and_ar_one() {
        // Single DT matches the only real GT → both precision and
        // recall are 1.0 across every recall threshold.
        let cell = one_threshold_eval(vec![0.9], vec![true], vec![false], vec![false]);
        let p = params(&[0.5], &[0.0, 0.5, 1.0], &[100], 1);
        let out = accumulate(&[Some(Box::new(cell))], p, ParityMode::Strict).unwrap();

        assert_eq!(out.recall[(0, 0, 0, 0)], 1.0);
        for ri in 0..3 {
            // Precision is `tp / (tp + fp + eps)` — 1 / (1 + 0 + eps) ≈ 1.
            let pr = out.precision[(0, ri, 0, 0, 0)];
            assert!((pr - 1.0).abs() < 1e-12, "precision[{ri}] = {pr}");
            assert_eq!(out.scores[(0, ri, 0, 0, 0)], 0.9);
        }
    }

    #[test]
    fn lone_fp_yields_zero_recall_zero_precision() {
        // One unmatched detection, one real unmatched GT → recall 0,
        // precision 0 across all recall thresholds. The score column
        // gets a value only at recall=0 (where the curve does exist);
        // recall thresholds past the end of the curve fall through to
        // pycocotools' silent-skip branch, leaving 0.0.
        let cell = one_threshold_eval(vec![0.9], vec![false], vec![false], vec![false]);
        let p = params(&[0.5], &[0.0, 0.5, 1.0], &[100], 1);
        let out = accumulate(&[Some(Box::new(cell))], p, ParityMode::Strict).unwrap();
        assert_eq!(out.recall[(0, 0, 0, 0)], 0.0);
        for ri in 0..3 {
            // 0 / (0 + 1 + eps) ≈ 0 → envelope keeps it at 0.
            assert!(out.precision[(0, ri, 0, 0, 0)].abs() < 1e-12);
        }
        // recall threshold 0.0 lands on the lone curve point (rc[0] =
        // 0.0); 0.5 and 1.0 are past the end → score sentinel 0.0.
        assert_eq!(out.scores[(0, 0, 0, 0, 0)], 0.9);
        assert_eq!(out.scores[(0, 1, 0, 0, 0)], 0.0);
        assert_eq!(out.scores[(0, 2, 0, 0, 0)], 0.0);
    }

    #[test]
    fn ignored_dt_does_not_count_as_fp() {
        // C7: an ignore-tagged DT is invisible to both TP and FP cumsums.
        // Setup: one real GT (matched by DT 0), one DT 1 that misses but
        // is ignore-tagged (e.g. out-of-area unmatched). FP must not
        // appear in the curve.
        let cell = one_threshold_eval(
            vec![0.9, 0.8],
            vec![true, false],
            vec![false, true],
            vec![false],
        );
        let p = params(&[0.5], &[0.0, 0.5, 1.0], &[100], 1);
        let out = accumulate(&[Some(Box::new(cell))], p, ParityMode::Strict).unwrap();

        // tp=1 fp=0 → precision ≈ 1 everywhere on the curve.
        for ri in 0..3 {
            let pr = out.precision[(0, ri, 0, 0, 0)];
            assert!((pr - 1.0).abs() < 1e-12, "precision[{ri}] = {pr}");
        }
        assert_eq!(out.recall[(0, 0, 0, 0)], 1.0);
    }

    #[test]
    fn precision_envelope_runs_right_to_left() {
        // C2: pre-envelope precision dips. Curve: TP, FP, TP → precisions
        // 1.0, 0.5, 0.667. After right-to-left max: 1.0, 0.667, 0.667.
        // Recall thresholds 0.0 and 0.5 (rc = [0.5, 0.5, 1.0]) sample
        // index 0; threshold 1.0 samples index 2.
        let cell = one_threshold_eval(
            vec![0.9, 0.8, 0.7],
            vec![true, false, true],
            vec![false, false, false],
            vec![false, false],
        );
        let p = params(&[0.5], &[0.0, 0.5, 1.0], &[100], 1);
        let out = accumulate(&[Some(Box::new(cell))], p, ParityMode::Strict).unwrap();

        // recall thresholds 0.0 and 0.5 both fall on the first rc cell
        // where rc[0] = 0.5 (TP at j=0 → 1/2). Envelope makes pr[0]=1.0.
        assert!((out.precision[(0, 0, 0, 0, 0)] - 1.0).abs() < 1e-12);
        assert!((out.precision[(0, 1, 0, 0, 0)] - 1.0).abs() < 1e-12);
        // recall threshold 1.0 samples j=2: pr[2] = 2/3.
        assert!((out.precision[(0, 2, 0, 0, 0)] - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn partition_point_matches_numpy_searchsorted_left() {
        // Pinning the stdlib semantics so a future swap (e.g., to a
        // SIMD search) keeps `np.searchsorted(..., side='left')` parity.
        let haystack = [0.1, 0.3, 0.3, 0.7];
        let lookup = |t: f64| haystack.partition_point(|&v| v < t);
        assert_eq!(lookup(0.0), 0);
        assert_eq!(lookup(0.3), 1); // leftmost equal
        assert_eq!(lookup(0.5), 3);
        assert_eq!(lookup(1.0), 4); // past end
    }

    #[test]
    fn merged_sort_breaks_ties_by_input_order() {
        // A1 over the merged stream: two images with one DT each at
        // score 0.7. With stable sort, image-0 DT comes first.
        let img0 = one_threshold_eval(vec![0.7], vec![true], vec![false], vec![false]);
        let img1 = one_threshold_eval(vec![0.7], vec![false], vec![false], vec![false]);
        // grid: K=1, A=1, I=2 → eval_imgs[0..2] is the (k=0, a=0) row.
        let grid = vec![Some(Box::new(img0)), Some(Box::new(img1))];
        let p = params(&[0.5], &[0.0, 0.5, 1.0], &[100], 2);
        let out = accumulate(&grid, p, ParityMode::Strict).unwrap();

        // tp=1, fp=1 → final pr = 0.5; rc = [0.5, 0.5]. With envelope
        // (no monotonicity adjustment needed because pr[1] < pr[0]),
        // recThr 0.0 and 0.5 both sample index 0 (pr ≈ 1.0), recThr 1.0
        // is past the end → 0.0.
        assert!((out.precision[(0, 0, 0, 0, 0)] - 1.0).abs() < 1e-12);
        assert!((out.precision[(0, 1, 0, 0, 0)] - 1.0).abs() < 1e-12);
        assert_eq!(out.precision[(0, 2, 0, 0, 0)], 0.0);
    }

    #[test]
    fn max_det_truncation_drops_low_score_dts_per_image() {
        // Per-image max_det=1: only the top-scoring DT survives, even
        // though more were emitted. With only the FP at score 0.95
        // surviving, AP must collapse.
        let cell = one_threshold_eval(
            vec![0.95, 0.9],
            vec![false, true], // FP first, TP second
            vec![false, false],
            vec![false],
        );
        let p = params(&[0.5], &[0.0, 0.5, 1.0], &[1], 1);
        let out = accumulate(&[Some(Box::new(cell))], p, ParityMode::Strict).unwrap();
        // Only FP survived → tp=0, fp=1, precision ≈ 0 everywhere.
        for ri in 0..3 {
            assert!(out.precision[(0, ri, 0, 0, 0)].abs() < 1e-12);
        }
        assert_eq!(out.recall[(0, 0, 0, 0)], 0.0);
    }

    #[test]
    fn dimension_mismatch_on_grid_size_is_typed_error() {
        let p = params(&[0.5], &[0.0], &[100], 5);
        // Grid claims K*A*I = 1*1*5 = 5 cells; we pass 2 → error.
        let err = accumulate(&[None, None], p, ParityMode::Strict).unwrap_err();
        match err {
            EvalError::DimensionMismatch { detail } => {
                assert!(detail.contains("eval_imgs"));
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn dimension_mismatch_on_per_image_t_is_typed_error() {
        // Per-image dt_matched has 2 rows, params declare 3 IoU
        // thresholds → mismatch reported.
        let cell = PerImageEval {
            dt_scores: vec![0.9],
            dt_matched: array![[true], [true]],
            dt_ignore: array![[false], [false]],
            gt_ignore: vec![false],
        };
        let p = params(&[0.5, 0.75, 0.9], &[0.0], &[100], 1);
        let err = accumulate(&[Some(Box::new(cell))], p, ParityMode::Strict).unwrap_err();
        assert!(matches!(err, EvalError::DimensionMismatch { .. }));
    }

    #[test]
    fn reaccumulate_with_different_area_range_count_is_typed_error() {
        // A3: re-accumulating an `eval_imgs` grid built for one A-axis
        // size against an `AccumulateParams` with a different
        // `n_area_ranges` must surface DimensionMismatch — not silently
        // produce wrong outputs by re-slicing the flat buffer at the new
        // pitch. Build a 4-area-range grid (the COCO default), then try
        // to accumulate it as if it were a 3-area-range grid.
        let n_i = 1;
        let n_a_built = 4;
        let n_k = 1;
        let cell = one_threshold_eval(vec![0.9], vec![true], vec![false], vec![false]);
        // Only the first (k=0, a=0, i=0) slot carries data; remaining
        // slots are None as they would be for an image with no GTs/DTs in
        // those buckets.
        let mut eval_imgs: Vec<Option<Box<PerImageEval>>> = vec![None; n_k * n_a_built * n_i];
        eval_imgs[0] = Some(Box::new(cell));

        // Mismatched params: claim the grid has 3 area ranges. Expected
        // grid size becomes 1*3*1 = 3, but we pass 4 cells → typed error.
        let mut bad = params(&[0.5], &[0.0, 0.5, 1.0], &[100], n_i);
        bad.n_area_ranges = 3;
        let err = accumulate(&eval_imgs, bad, ParityMode::Strict).unwrap_err();
        match err {
            EvalError::DimensionMismatch { detail } => {
                assert!(detail.contains("eval_imgs"), "msg: {detail}");
                assert!(detail.contains("n_area_ranges(3)"), "msg: {detail}");
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn vectorized_inner_sweep_matches_naive_reference() {
        // C6: the inner recall-threshold sweep is vectorized via
        // partition_point + an in-place right-to-left envelope. Pin it
        // against a naive reference that mirrors pycocotools' Python
        // `for ri, pi in enumerate(inds): q[ri] = pr[pi]` line by line.
        //
        // Three hand-crafted PR curves cover the edge cases:
        //  - monotonic-decreasing precision (no envelope work);
        //  - non-monotonic precision (envelope rewrites multiple cells);
        //  - all-1.0 precision with the recall curve ending at 0.5 so
        //    half the recall thresholds fall past the curve (C3 path).
        //
        // Only the precision lane is compared — both implementations
        // share the same recall-index lookup, so the score lane would
        // trivially agree.
        let recall_thresholds: Vec<f64> = (0..=10).map(|i| (i as f64) / 10.0).collect();

        // Naive reference: explicit right-to-left running max + linear
        // searchsorted-left scan.
        fn naive_sweep(rc: &[f64], pr: &[f64], rec_thr: &[f64]) -> Vec<f64> {
            let n = pr.len();
            let mut env = pr.to_vec();
            for j in (1..n).rev() {
                if env[j] > env[j - 1] {
                    env[j - 1] = env[j];
                }
            }
            let mut q = vec![0.0_f64; rec_thr.len()];
            for (ri, &target) in rec_thr.iter().enumerate() {
                let mut pi = n;
                for (j, &r) in rc.iter().enumerate() {
                    if r >= target {
                        pi = j;
                        break;
                    }
                }
                if pi < n {
                    q[ri] = env[pi];
                }
            }
            q
        }

        // Vectorized reference: same shape as `accumulate_cell`'s inner
        // sweep, callable on hand-crafted curves without rebuilding the
        // whole `(T, R, K, A, M)` tensor. Drift between this body and
        // the production sweep is what the test exists to catch.
        fn vectorized_sweep(rc: &[f64], pr: &[f64], rec_thr: &[f64]) -> Vec<f64> {
            let n = pr.len();
            let mut env = pr.to_vec();
            for j in (1..n).rev() {
                if env[j] > env[j - 1] {
                    env[j - 1] = env[j];
                }
            }
            let mut q = vec![0.0_f64; rec_thr.len()];
            for (ri, &target) in rec_thr.iter().enumerate() {
                let pi = rc.partition_point(|&v| v < target);
                if pi < n {
                    q[ri] = env[pi];
                }
            }
            q
        }

        let curves: &[(&[f64], &[f64])] = &[
            // Monotonic-decreasing precision; recall reaches 1.0.
            (&[0.1, 0.3, 0.5, 0.7, 1.0], &[1.0, 0.9, 0.7, 0.5, 0.3]),
            // Non-monotonic precision: envelope rewrites cells 1 and 3.
            (&[0.2, 0.4, 0.6, 0.8, 1.0], &[1.0, 0.4, 0.6, 0.2, 0.5]),
            // All-1.0 precision; recall caps at 0.5 → recall thresholds
            // > 0.5 fall past the curve (C3 silent-skip path → 0.0).
            (&[0.1, 0.2, 0.3, 0.4, 0.5], &[1.0, 1.0, 1.0, 1.0, 1.0]),
        ];

        for (i, (rc, pr)) in curves.iter().enumerate() {
            let q_naive = naive_sweep(rc, pr, &recall_thresholds);
            let q_vec = vectorized_sweep(rc, pr, &recall_thresholds);
            assert_eq!(q_naive.len(), q_vec.len(), "curve {i}");
            for (ri, (a, b)) in q_naive.iter().zip(q_vec.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "curve {i}, recall threshold index {ri}: naive={a}, vec={b}"
                );
            }
        }
    }

    #[test]
    fn sort_max_dets_normalizes_ascending() {
        // Quirk A2: pycocotools' `cocoeval.py:137` does
        // `p.maxDets = sorted(p.maxDets)` — `sort_max_dets` is the
        // mirror at the param-construction boundary.
        let mut ladder = vec![100usize, 1, 10];
        sort_max_dets(&mut ladder);
        assert_eq!(ladder, vec![1, 10, 100]);
    }

    #[test]
    fn sort_max_dets_is_idempotent_on_sorted_input() {
        let mut ladder = vec![1usize, 10, 100];
        sort_max_dets(&mut ladder);
        assert_eq!(ladder, vec![1, 10, 100]);
    }

    #[test]
    fn sort_max_dets_handles_duplicates_and_singletons() {
        let mut singleton = vec![100usize];
        sort_max_dets(&mut singleton);
        assert_eq!(singleton, vec![100]);

        let mut empty: Vec<usize> = Vec::new();
        sort_max_dets(&mut empty);
        assert!(empty.is_empty());

        let mut dups = vec![10usize, 1, 10, 1, 100];
        sort_max_dets(&mut dups);
        assert_eq!(dups, vec![1, 1, 10, 10, 100]);
    }

    #[test]
    fn permuted_ladder_after_sort_matches_canonical_order() {
        // End-to-end: feeding `[100, 1, 10]` after `sort_max_dets`
        // produces a `(T, R, K, A, M)` accumulator whose M-axis is
        // identical to the one built from the canonical `[1, 10, 100]`.
        // Without the sort, the M-axis slots would be swapped and the
        // summarizer's positional `AR_1 / AR_10 / AR_100` mapping would
        // bind to the wrong threshold.
        let cell = one_threshold_eval(
            vec![0.9, 0.8, 0.7],
            vec![true, true, false],
            vec![false, false, false],
            vec![false, false, false],
        );
        let iou = [0.5];
        let rec = [0.0, 0.5, 1.0];

        let canonical = vec![1usize, 10, 100];
        let canonical_acc = accumulate(
            &[Some(Box::new(cell.clone()))],
            params(&iou, &rec, &canonical, 1),
            ParityMode::Strict,
        )
        .unwrap();

        let mut permuted = vec![100usize, 1, 10];
        sort_max_dets(&mut permuted);
        assert_eq!(permuted, canonical);
        let permuted_acc = accumulate(
            &[Some(Box::new(cell))],
            params(&iou, &rec, &permuted, 1),
            ParityMode::Strict,
        )
        .unwrap();

        assert_eq!(canonical_acc.precision, permuted_acc.precision);
        assert_eq!(canonical_acc.recall, permuted_acc.recall);
        assert_eq!(canonical_acc.scores, permuted_acc.scores);
    }

    /// Multi-category grid with ragged per-cell shapes: some cells
    /// empty, some all-ignore (the `npig == 0` skip), some with
    /// detections but no matches. The parallel walk must reproduce the
    /// sequential tensors bit-for-bit, not merely within a tolerance.
    #[test]
    fn parallel_accumulate_is_bit_identical_to_sequential() {
        let n_k = 7;
        let n_a = 4;
        let n_i = 5;
        let iou = [0.5, 0.75];
        let rec: Vec<f64> = (0..101).map(|r| f64::from(r) / 100.0).collect();
        let max_dets = [1usize, 10, 100];

        let mut grid: Vec<Option<Box<PerImageEval>>> = Vec::with_capacity(n_k * n_a * n_i);
        for slot in 0..(n_k * n_a * n_i) {
            // Deterministic variety: empty cells, all-ignored cells,
            // and cells whose scores tie across images.
            let cell = match slot % 5 {
                0 => None,
                1 => Some(two_threshold_eval(
                    vec![0.9, 0.9, 0.4],
                    vec![true, false, true, false, true, false],
                    vec![false; 6],
                    vec![false, true],
                )),
                2 => Some(two_threshold_eval(
                    vec![0.5],
                    vec![true, false],
                    vec![false, false],
                    vec![true],
                )),
                3 => Some(two_threshold_eval(
                    vec![0.8, 0.2],
                    vec![false, true, true, true],
                    vec![true, false, false, false],
                    vec![false],
                )),
                _ => Some(two_threshold_eval(
                    vec![0.99, 0.75, 0.75, 0.1],
                    vec![true; 8],
                    vec![false; 8],
                    vec![false, false, true],
                )),
            };
            grid.push(cell.map(Box::new));
        }

        let p = AccumulateParams {
            iou_thresholds: &iou,
            recall_thresholds: &rec,
            max_dets: &max_dets,
            n_categories: n_k,
            n_area_ranges: n_a,
            n_images: n_i,
        };
        let sequential = accumulate(&grid, p, ParityMode::Strict).expect("sequential");
        let parallel = accumulate_parallel(&grid, p, ParityMode::Strict).expect("parallel");

        assert_eq!(sequential.precision, parallel.precision);
        assert_eq!(sequential.recall, parallel.recall);
        assert_eq!(sequential.scores, parallel.scores);
    }

    /// `T = 2` sibling of [`one_threshold_eval`]; `matched` / `ignore`
    /// are row-major `(2, n)`.
    fn two_threshold_eval(
        scores: Vec<f64>,
        matched: Vec<bool>,
        ignore: Vec<bool>,
        gt_ignore: Vec<bool>,
    ) -> PerImageEval {
        let n = scores.len();
        PerImageEval {
            dt_scores: scores,
            dt_matched: Array2::from_shape_vec((2, n), matched).expect("dt_matched shape"),
            dt_ignore: Array2::from_shape_vec((2, n), ignore).expect("dt_ignore shape"),
            gt_ignore,
        }
    }

    // ---- ADR-0052: one sort per category ------------------------------

    /// Claim 2: the `maxDet = m` stream is the cap stream restricted to
    /// each image's first `m` detections, and filtering the cap
    /// permutation reproduces the stable sort of that restriction.
    /// Checked against `argsort_score_desc` on an independently
    /// gathered stream — the pre-ADR-0052 code path, verbatim.
    #[test]
    fn derived_max_det_stream_matches_an_independent_sort() {
        // Ragged cells, ties inside one image, ties across images, and
        // an empty cell — every shape the filter has to survive.
        let raw = [
            vec![0.9, 0.7, 0.7, 0.2],
            vec![0.9, 0.5],
            vec![],
            vec![0.8, 0.8, 0.8],
        ];
        let owned: Vec<PerImageEval> = raw
            .iter()
            .map(|scores| {
                let n = scores.len();
                one_threshold_eval(scores.clone(), vec![true; n], vec![false; n], vec![false])
            })
            .collect();
        let cells: Vec<&PerImageEval> = owned.iter().collect();

        let cap = 100usize;
        let plan = SortPlan::build(&cells, cap);
        let mut derived = DerivedStream::default();

        for max_det in [0usize, 1, 2, 3, 4, 99] {
            derived.fill_from(&plan, max_det);

            // The reference: gather this maxDet's stream from scratch
            // and sort it, exactly as `accumulate_cell` used to.
            let mut expect_takes = Vec::new();
            let mut expect_scores: Vec<f64> = Vec::new();
            for cell in &cells {
                let take = cell.dt_scores.len().min(max_det);
                expect_takes.push(take);
                expect_scores.extend_from_slice(&cell.dt_scores[..take]);
            }
            let expect_perm = argsort_score_desc(&expect_scores);

            assert_eq!(derived.takes, expect_takes, "takes at maxDet={max_det}");
            assert_eq!(derived.scores, expect_scores, "stream at maxDet={max_det}");
            assert_eq!(derived.perm, expect_perm, "perm at maxDet={max_det}");
        }
    }

    /// The cap itself needs no filtering: the plan is the stream.
    #[test]
    fn plan_stream_equals_the_cap_max_det_stream() {
        let owned = [
            one_threshold_eval(vec![0.6, 0.4], vec![true; 2], vec![false; 2], vec![false]),
            one_threshold_eval(vec![0.5], vec![true], vec![false], vec![false]),
        ];
        let cells: Vec<&PerImageEval> = owned.iter().collect();
        let plan = SortPlan::build(&cells, 100);

        assert_eq!(plan.takes, vec![2, 1]);
        assert_eq!(plan.scores, vec![0.6, 0.4, 0.5]);
        assert_eq!(plan.perm, argsort_score_desc(&[0.6, 0.4, 0.5]));
    }

    /// Claim 1 is verified, not assumed. A grid whose area ranges carry
    /// different score streams — which vernier's own evaluate paths
    /// never build, but a hand-built grid can — must rebuild the plan.
    #[test]
    fn plan_is_rejected_when_the_next_area_range_diverges() {
        let base = one_threshold_eval(vec![0.9, 0.3], vec![true; 2], vec![false; 2], vec![false]);
        let cells = vec![&base];
        let plan = SortPlan::build(&cells, 100);
        assert!(plan.matches(&cells, 100));

        // Different values, same length.
        let other = one_threshold_eval(vec![0.9, 0.2], vec![true; 2], vec![false; 2], vec![false]);
        assert!(!plan.matches(&[&other], 100));

        // Different length.
        let shorter = one_threshold_eval(vec![0.9], vec![true], vec![false], vec![false]);
        assert!(!plan.matches(&[&shorter], 100));

        // Different cell count.
        assert!(!plan.matches(&[&base, &base], 100));

        // Different cap — the takes would differ.
        assert!(!plan.matches(&cells, 1));

        // A plan whose cap stream carries a NaN is never reused: its
        // permutation came out of an intransitive comparator.
        let nan = one_threshold_eval(
            vec![f64::NAN, 0.3],
            vec![true; 2],
            vec![false; 2],
            vec![false],
        );
        assert!(!plan.matches(&[&nan], 100));
        let nan_plan = SortPlan::build(&[&nan], 100);
        assert!(nan_plan.has_nan);
        assert!(!nan_plan.matches(&[&nan], 100));
    }

    /// End-to-end guard check: a grid that violates Claim 1 still
    /// accumulates each area range against its own stream. Each area
    /// range is compared to the same cell accumulated on its own.
    #[test]
    fn divergent_area_ranges_accumulate_against_their_own_streams() {
        let rec: Vec<f64> = (0..101).map(|r| f64::from(r) / 100.0).collect();
        let max_dets = [1usize, 10, 100];
        let iou = [0.5];

        // a=0: two DTs, the top one a TP. a=1: different scores *and* a
        // different match pattern, so reusing a=0's permutation would
        // show up in the tensors.
        let a0 = one_threshold_eval(
            vec![0.9, 0.3],
            vec![true, false],
            vec![false, false],
            vec![false, false],
        );
        let a1 = one_threshold_eval(
            vec![0.8, 0.75, 0.7],
            vec![false, true, true],
            vec![false, false, false],
            vec![false, false],
        );

        let combined = accumulate(
            &[Some(Box::new(a0.clone())), Some(Box::new(a1.clone()))],
            AccumulateParams {
                iou_thresholds: &iou,
                recall_thresholds: &rec,
                max_dets: &max_dets,
                n_categories: 1,
                n_area_ranges: 2,
                n_images: 1,
            },
            ParityMode::Strict,
        )
        .expect("combined");

        for (a, cell) in [a0, a1].into_iter().enumerate() {
            let solo = accumulate(
                &[Some(Box::new(cell))],
                AccumulateParams {
                    iou_thresholds: &iou,
                    recall_thresholds: &rec,
                    max_dets: &max_dets,
                    n_categories: 1,
                    n_area_ranges: 1,
                    n_images: 1,
                },
                ParityMode::Strict,
            )
            .expect("solo");

            for m in 0..max_dets.len() {
                assert_eq!(
                    combined.recall[(0, 0, a, m)],
                    solo.recall[(0, 0, 0, m)],
                    "recall a={a} m={m}"
                );
                for ri in 0..rec.len() {
                    assert_eq!(
                        combined.precision[(0, ri, 0, a, m)],
                        solo.precision[(0, ri, 0, 0, m)],
                        "precision a={a} ri={ri} m={m}"
                    );
                    assert_eq!(
                        combined.scores[(0, ri, 0, a, m)],
                        solo.scores[(0, ri, 0, 0, m)],
                        "scores a={a} ri={ri} m={m}"
                    );
                }
            }
        }
    }

    /// The `NaN` hole in Claim 2. `argsort_score_desc` degrades to an
    /// intransitive comparator on a stream carrying a `NaN`, so
    /// filtering the cap permutation is no longer the permutation the
    /// restricted stream sorts to: for these cells at `maxDet = 1` the
    /// filter yields `[0, 1, 2]` where a fresh sort yields `[1, 0, 2]`,
    /// which would send the sweep to `0.9` before `0.95`.
    #[test]
    fn nan_in_the_cap_stream_sorts_each_derived_max_det_stream() {
        let raw = [
            vec![0.9, f64::NAN, 0.8, 0.7],
            vec![0.95, 0.85, f64::NAN, 0.1],
            vec![f64::NAN, 0.99, 0.5],
        ];
        let owned: Vec<PerImageEval> = raw
            .iter()
            .map(|scores| {
                let n = scores.len();
                one_threshold_eval(scores.clone(), vec![true; n], vec![false; n], vec![false])
            })
            .collect();
        let cells: Vec<&PerImageEval> = owned.iter().collect();

        let plan = SortPlan::build(&cells, 100);
        assert!(plan.has_nan, "cap stream carries NaNs");

        let mut derived = DerivedStream::default();
        derived.fill_from(&plan, 1);

        assert_eq!(derived.takes, vec![1, 1, 1]);
        assert_eq!(derived.scores[0], 0.9);
        assert_eq!(derived.scores[1], 0.95);
        assert!(derived.scores[2].is_nan());

        // The fallback is a fresh sort of the derived stream — by
        // definition what this `(a, m)` cell sorted to before ADR-0052.
        assert_eq!(derived.perm, argsort_score_desc(&derived.scores));
        assert_eq!(derived.perm, vec![1, 0, 2]);

        // And it differs from what filtering the cap permutation gives,
        // which is the pre-fix behaviour this test exists to pin.
        let mut map = vec![DerivedStream::DROPPED; plan.scores.len()];
        let mut cap_cursor = 0usize;
        let mut out_cursor = 0usize;
        for &take in &plan.takes {
            let keep = take.min(1);
            for j in 0..keep {
                map[cap_cursor + j] = out_cursor + j;
            }
            cap_cursor += take;
            out_cursor += keep;
        }
        let filtered: Vec<usize> = plan
            .perm
            .iter()
            .filter_map(|&i| (map[i] != DerivedStream::DROPPED).then_some(map[i]))
            .collect();
        assert_ne!(
            filtered, derived.perm,
            "filter and fresh sort must disagree here, or the repro has gone stale"
        );
    }

    /// End-to-end: on a `NaN`-carrying grid every `maxDet` lane must
    /// equal the same grid pre-truncated to that `maxDet` and
    /// accumulated on its own — one sort per cell, which is what the
    /// fallback restores.
    #[test]
    fn nan_scores_accumulate_like_a_pre_truncated_grid() {
        let rec: Vec<f64> = (0..101).map(|r| f64::from(r) / 100.0).collect();
        let iou = [0.5];
        let max_dets = [1usize, 100];

        let raw: [(Vec<f64>, Vec<bool>); 3] = [
            (
                vec![0.9, f64::NAN, 0.8, 0.7],
                vec![true, false, true, false],
            ),
            (
                vec![0.95, 0.85, f64::NAN, 0.1],
                vec![false, true, false, true],
            ),
            (vec![f64::NAN, 0.99, 0.5], vec![true, true, false]),
        ];
        let truncate = |take: usize| -> Vec<Option<Box<PerImageEval>>> {
            raw.iter()
                .map(|(scores, matched)| {
                    let n = scores.len().min(take);
                    Some(Box::new(one_threshold_eval(
                        scores[..n].to_vec(),
                        matched[..n].to_vec(),
                        vec![false; n],
                        vec![false, false],
                    )))
                })
                .collect()
        };

        let combined = accumulate(
            &truncate(usize::MAX),
            AccumulateParams {
                iou_thresholds: &iou,
                recall_thresholds: &rec,
                max_dets: &max_dets,
                n_categories: 1,
                n_area_ranges: 1,
                n_images: 3,
            },
            ParityMode::Strict,
        )
        .expect("combined");

        for (m, &max_det) in max_dets.iter().enumerate() {
            let ladder = [max_det];
            let solo = accumulate(
                &truncate(max_det),
                AccumulateParams {
                    iou_thresholds: &iou,
                    recall_thresholds: &rec,
                    max_dets: &ladder,
                    n_categories: 1,
                    n_area_ranges: 1,
                    n_images: 3,
                },
                ParityMode::Strict,
            )
            .expect("solo");

            assert_eq!(
                combined.recall[(0, 0, 0, m)].to_bits(),
                solo.recall[(0, 0, 0, 0)].to_bits(),
                "recall at maxDet={max_det}"
            );
            for ri in 0..rec.len() {
                assert_eq!(
                    combined.precision[(0, ri, 0, 0, m)].to_bits(),
                    solo.precision[(0, ri, 0, 0, 0)].to_bits(),
                    "precision ri={ri} at maxDet={max_det}"
                );
                assert_eq!(
                    combined.scores[(0, ri, 0, 0, m)].to_bits(),
                    solo.scores[(0, ri, 0, 0, 0)].to_bits(),
                    "scores ri={ri} at maxDet={max_det}"
                );
            }
        }
    }

    /// Signed zero: `-0.0 == 0.0`, so an `==` guard would call a `-0.0`
    /// area range a match for a cached `+0.0` and then emit the plan's
    /// sign into the scores tensor. `matches` compares bit patterns, so
    /// each area range keeps the zero it was given.
    #[test]
    fn signed_zero_rebuilds_the_plan_and_keeps_its_sign() {
        let pos = one_threshold_eval(
            vec![0.5, 0.0],
            vec![true, true],
            vec![false, false],
            vec![false, false],
        );
        let neg = one_threshold_eval(
            vec![0.5, -0.0],
            vec![true, true],
            vec![false, false],
            vec![false, false],
        );

        let plan = SortPlan::build(&[&pos], 100);
        assert!(plan.matches(&[&pos], 100));
        assert!(!plan.matches(&[&neg], 100), "-0.0 is not +0.0 bitwise");

        // a=0 carries +0.0, a=1 carries -0.0. Recall threshold 1.0
        // samples the second detection in both.
        let rec = [0.0, 1.0];
        let out = accumulate(
            &[Some(Box::new(pos)), Some(Box::new(neg))],
            AccumulateParams {
                iou_thresholds: &[0.5],
                recall_thresholds: &rec,
                max_dets: &[100],
                n_categories: 1,
                n_area_ranges: 2,
                n_images: 1,
            },
            ParityMode::Strict,
        )
        .expect("accumulate");

        assert_eq!(out.scores[(0, 1, 0, 0, 0)].to_bits(), 0.0_f64.to_bits());
        assert_eq!(out.scores[(0, 1, 0, 1, 0)].to_bits(), (-0.0_f64).to_bits());
    }
}
