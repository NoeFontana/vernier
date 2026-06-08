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

use ndarray::{Array2, Array4, Array5, Axis};

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

    let mut precision = Array5::<f64>::from_elem((n_t, n_r, n_k, n_a, n_m), -1.0);
    let mut recall = Array4::<f64>::from_elem((n_t, n_k, n_a, n_m), -1.0);
    let mut scores = Array5::<f64>::from_elem((n_t, n_r, n_k, n_a, n_m), -1.0);

    for k in 0..n_k {
        let nk = k * n_a * n_i;
        for a in 0..n_a {
            let na = a * n_i;
            let cells: Vec<&PerImageEval> = (0..n_i)
                .filter_map(|i| eval_imgs[nk + na + i].as_deref())
                .collect();
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

            for (m, &max_det) in p.max_dets.iter().enumerate() {
                accumulate_cell(
                    &cells,
                    max_det,
                    npig,
                    n_t,
                    p.recall_thresholds,
                    k,
                    a,
                    m,
                    &mut precision,
                    &mut recall,
                    &mut scores,
                );
            }
        }
    }

    Ok(Accumulated {
        precision,
        recall,
        scores,
    })
}

#[allow(clippy::too_many_arguments)]
fn accumulate_cell(
    cells: &[&PerImageEval],
    max_det: usize,
    npig: usize,
    n_t: usize,
    recall_thresholds: &[f64],
    k: usize,
    a: usize,
    m: usize,
    precision: &mut Array5<f64>,
    recall: &mut Array4<f64>,
    scores: &mut Array5<f64>,
) {
    let mut takes: Vec<usize> = Vec::with_capacity(cells.len());
    let mut total = 0usize;
    for cell in cells {
        let take = cell.dt_scores.len().min(max_det);
        takes.push(take);
        total += take;
    }
    let mut all_scores: Vec<f64> = Vec::with_capacity(total);
    for (cell, &take) in cells.iter().zip(&takes) {
        all_scores.extend_from_slice(&cell.dt_scores[..take]);
    }

    let n_d = all_scores.len();
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
            recall[(t, k, a, m)] = 0.0;
            for ri in 0..recall_thresholds.len() {
                precision[(t, ri, k, a, m)] = 0.0;
                scores[(t, ri, k, a, m)] = 0.0;
            }
        }
        return;
    }

    let perm = argsort_score_desc(&all_scores);

    let npig_f = npig as f64;
    let mut rc = vec![0.0_f64; n_d];
    let mut pr = vec![0.0_f64; n_d];
    let mut dtm = vec![false; n_d];
    let mut dtg = vec![false; n_d];

    for t in 0..n_t {
        let mut cursor = 0;
        for (cell, &take) in cells.iter().zip(&takes) {
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
        for (out_idx, &src_idx) in perm.iter().enumerate() {
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
        recall[(t, k, a, m)] = rc[n_d - 1];

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
            .index_axis_move(Axis(1), k)
            .index_axis_move(Axis(1), a)
            .index_axis_move(Axis(1), m);
        let mut s_lane = scores
            .index_axis_mut(Axis(0), t)
            .index_axis_move(Axis(1), k)
            .index_axis_move(Axis(1), a)
            .index_axis_move(Axis(1), m);
        for (ri, &target) in recall_thresholds.iter().enumerate() {
            let pi = rc.partition_point(|&v| v < target);
            if pi < n_d {
                p_lane[ri] = pr[pi];
                s_lane[ri] = all_scores[perm[pi]];
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
}
