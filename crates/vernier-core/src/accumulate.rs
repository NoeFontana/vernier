//! Per-image evaluation → precision/recall/scores arrays.
//!
//! Mirrors `pycocotools.cocoeval.COCOeval.accumulate` (cocoeval.py
//! lines 315-420). Inputs come from the upstream matching engine
//! ([`crate::match_image`]) packaged as one [`PerImageEval`] per
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
//! - **C5** (`strict`): `(K, A, M)` cells with no detections or no
//!   non-ignore GTs leave `precision`/`recall`/`scores` at the `-1`
//!   sentinel; the summarizer filters those before averaging.
//! - **C7** (`strict`): TP and FP cumsums skip DTs whose `dt_ignore`
//!   flag is set — both B6 (matched-to-ignore) and B7 (out-of-area
//!   unmatched) are folded into `dt_ignore` upstream.
//! - **C8** (`aligned`): precision denominator uses
//!   [`PARITY_EPS`](crate::PARITY_EPS) (= `f64::EPSILON`), bit-equal to
//!   `np.spacing(1)`.
//! - **L1, L2** (`strict`): `iou_thresholds` and `recall_thresholds`
//!   come from [`crate::iou_thresholds`] / [`crate::recall_thresholds`]
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
/// Built by the orchestrator from a [`crate::MatchResult`] plus the
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
    /// IoU thresholds, length `T`. Use [`crate::iou_thresholds`] for
    /// the canonical 10-point COCO ladder.
    pub iou_thresholds: &'p [f64],
    /// Recall integration thresholds, length `R` (typically 101). Use
    /// [`crate::recall_thresholds`].
    pub recall_thresholds: &'p [f64],
    /// Per-image maxDet caps, length `M`. Pycocotools defaults to
    /// `[1, 10, 100]`. The matching engine should be invoked with the
    /// *largest* of these — the accumulator slices to smaller caps via
    /// `[..max_det]`.
    pub max_dets: &'p [usize],
    /// Number of categories `K` (or `1` when `useCats == 0`).
    pub n_categories: usize,
    /// Number of area ranges `A` (COCO defaults to 4: all/small/medium/
    /// large).
    pub n_area_ranges: usize,
    /// Number of images `I`.
    pub n_images: usize,
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
    eval_imgs: &[Option<PerImageEval>],
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
                .filter_map(|i| eval_imgs[nk + na + i].as_ref())
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
        // No detections, but npig > 0 — recall collapses to 0; precision
        // and scores keep the -1 sentinel.
        for t in 0..n_t {
            recall[(t, k, a, m)] = 0.0;
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
    fn no_dt_with_real_gt_yields_zero_recall_and_sentinel_precision() {
        // C5: precision stays at -1 sentinel; recall is 0 for every t.
        let cell = PerImageEval {
            dt_scores: vec![],
            dt_matched: Array2::<bool>::default((2, 0)),
            dt_ignore: Array2::<bool>::default((2, 0)),
            gt_ignore: vec![false],
        };
        let p = params(&[0.5, 0.75], &[0.0, 0.5, 1.0], &[100], 1);
        let out = accumulate(&[Some(cell)], p, ParityMode::Strict).unwrap();
        assert_eq!(out.recall[(0, 0, 0, 0)], 0.0);
        assert_eq!(out.recall[(1, 0, 0, 0)], 0.0);
        // No precision write happened — every cell still -1.
        for ri in 0..3 {
            assert_eq!(out.precision[(0, ri, 0, 0, 0)], -1.0);
            assert_eq!(out.precision[(1, ri, 0, 0, 0)], -1.0);
        }
    }

    #[test]
    fn cell_with_only_ignore_gts_skips_entirely() {
        // npig == 0 short-circuit: outputs stay at -1 (no recall write).
        let cell = one_threshold_eval(vec![0.9], vec![true], vec![true], vec![true]);
        let p = params(&[0.5], &[0.0, 0.5, 1.0], &[100], 1);
        let out = accumulate(&[Some(cell)], p, ParityMode::Strict).unwrap();
        assert_eq!(out.recall[(0, 0, 0, 0)], -1.0);
        assert_eq!(out.precision[(0, 0, 0, 0, 0)], -1.0);
    }

    #[test]
    fn perfect_match_yields_ap_one_and_ar_one() {
        // Single DT matches the only real GT → both precision and
        // recall are 1.0 across every recall threshold.
        let cell = one_threshold_eval(vec![0.9], vec![true], vec![false], vec![false]);
        let p = params(&[0.5], &[0.0, 0.5, 1.0], &[100], 1);
        let out = accumulate(&[Some(cell)], p, ParityMode::Strict).unwrap();

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
        let out = accumulate(&[Some(cell)], p, ParityMode::Strict).unwrap();
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
        let out = accumulate(&[Some(cell)], p, ParityMode::Strict).unwrap();

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
        let out = accumulate(&[Some(cell)], p, ParityMode::Strict).unwrap();

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
        let grid = vec![Some(img0), Some(img1)];
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
        let out = accumulate(&[Some(cell)], p, ParityMode::Strict).unwrap();
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
        let err = accumulate(&[Some(cell)], p, ParityMode::Strict).unwrap_err();
        assert!(matches!(err, EvalError::DimensionMismatch { .. }));
    }
}
