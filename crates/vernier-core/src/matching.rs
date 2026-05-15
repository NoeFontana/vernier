//! Greedy GT × DT matching over a precomputed similarity matrix.
//!
//! Mirrors the inner matching loop of
//! `pycocotools.cocoeval.COCOeval.evaluateImg` (lines 268-296). Per
//! ADR-0005, this engine is generic over the matrix only — it knows
//! nothing about bbox, segm, or keypoints. Phase 2 (segm) and Phase 3
//! (OKS) add new [`crate::similarity::Similarity`] impls without touching this file.
//! If they would, the abstraction failed.
//!
//! ## Ordering
//!
//! `match_image` receives inputs in the caller's natural order and
//! produces results in the engine's *internal sorted orders*:
//!
//! - GTs are sorted ascending by `gt_ignore` (quirk **A4** — strict).
//!   The matching loop's [B3](#quirk-dispositions) early-termination
//!   relies on this ordering.
//! - DTs are sorted descending by `dt_scores` (quirk **A1** — strict).
//!   Stable mergesort means tied scores resolve to *input* order; this
//!   matters for parity and is exactly what numpy's
//!   `argsort(kind='mergesort')` produces.
//!
//! `MatchResult::dt_perm` and `MatchResult::gt_perm` expose the
//! permutations so callers can map matched indices back to their
//! input positions if needed.
//!
//! ## Quirk dispositions
//!
//! - **A1** (`strict`): stable mergesort of dts by `-score`. Ties
//!   resolve to input order. The corrected `(-score, ann_id)` tiebreak
//!   from the disposition table is deferred — exposing `ann_id` here
//!   would expand the ADR-0005 signature, so it lands when the FFI
//!   layer plumbs ids through.
//! - **A4** (`strict`): GTs sorted ascending by `_ignore`. B3 depends
//!   on it.
//! - **B1** (`strict`): the per-DT best-IoU tracker is seeded at
//!   `min(t, 1 - IOU_BOUNDARY_EPS)` so a DT whose best overlap exactly
//!   equals the threshold still matches.
//! - **B2** (`strict`): IoU comparison is non-strict (`>=`); on a tie
//!   the *later* GT wins (the comparison is `< best`, which keeps the
//!   update path live for equality).
//! - **B3** (`strict`): once a real match has been made, scanning into
//!   the trailing block of ignore-GTs is short-circuited (the inner
//!   loop `break`s).
//! - **B4** (`strict`): crowd GTs are many-to-one — they are not
//!   removed from contention after a first DT picks them, so further
//!   DTs may also match them.
//! - **B6** (`strict`): a DT matched to an ignore-GT inherits the
//!   ignore flag; downstream this DT disappears from the
//!   precision/recall curve.
//!
//! Quirk **B7** (out-of-area unmatched DT → `dtIg=1`) is *not* handled
//! here — it depends on per-DT area and the active `areaRng`, neither
//! of which the matching engine takes as input. The accumulator
//! applies B7 before building the curve.

use ndarray::{Array2, ArrayView2};

use crate::error::EvalError;
use crate::parity::{argsort_score_desc, ParityMode, IOU_BOUNDARY_EPS};

/// Per-call `(g, d, wall_ns)` recorder, gated on `bench-histogram`.
#[cfg(feature = "bench-histogram")]
pub mod histogram {
    use std::path::Path;
    use std::sync::Mutex;
    use std::time::Instant;

    static RECORDS: Mutex<Vec<(u32, u32, u64)>> = Mutex::new(Vec::new());

    /// Drop-on-end timer. Constructed at the top of `match_image`;
    /// pushes `(g, d, wall_ns)` into the global buffer when it drops.
    pub(super) struct CallTimer {
        g: u32,
        d: u32,
        start: Instant,
    }

    impl CallTimer {
        pub(super) fn new(g: usize, d: usize) -> Self {
            Self {
                g: u32::try_from(g).unwrap_or(u32::MAX),
                d: u32::try_from(d).unwrap_or(u32::MAX),
                start: Instant::now(),
            }
        }
    }

    impl Drop for CallTimer {
        fn drop(&mut self) {
            let elapsed = self.start.elapsed().as_nanos();
            let wall_ns = u64::try_from(elapsed).unwrap_or(u64::MAX);
            if let Ok(mut records) = RECORDS.lock() {
                records.push((self.g, self.d, wall_ns));
            }
        }
    }

    /// Write every recorded `match_image` call to `path` as CSV (header
    /// `g,d,wall_ns`), then clear the in-process buffer. Returns the
    /// number of records written.
    pub fn dump_csv(path: &Path) -> std::io::Result<usize> {
        use std::io::Write;
        let mut records = RECORDS.lock().unwrap_or_else(|p| p.into_inner());
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(file, "g,d,wall_ns")?;
        for (g, d, w) in records.iter() {
            writeln!(file, "{},{},{}", g, d, w)?;
        }
        let n = records.len();
        records.clear();
        file.flush()?;
        Ok(n)
    }

    /// Number of records currently buffered. Test-only hook for the
    /// smoke test.
    #[cfg(test)]
    pub(super) fn len() -> usize {
        let records = RECORDS.lock().unwrap_or_else(|p| p.into_inner());
        records.len()
    }
}

/// Per-image, per-category greedy match between ground-truth and
/// detection annotations across every IoU threshold.
///
/// Output arrays are indexed in the engine's *sorted* orders (see
/// [`crate::matching`] module docs). Use [`MatchResult::dt_perm`] and
/// [`MatchResult::gt_perm`] to recover the input index for any sorted
/// position.
#[derive(Debug, Clone)]
pub(crate) struct MatchResult {
    /// `dt_perm[k]` is the input DT index of the `k`-th highest-scoring
    /// DT (length D). Built by stable mergesort on `-dt_scores`.
    pub dt_perm: Vec<usize>,

    /// `gt_perm[k]` is the input GT index of the `k`-th sorted GT
    /// (length G). Built by stable mergesort on `gt_ignore`
    /// (`false` < `true`).
    pub gt_perm: Vec<usize>,

    /// Shape `(T, D)`. For each `(threshold, sorted-DT k)`, the
    /// sorted-GT position of the matched GT, or `-1` if unmatched.
    /// Use `gt_perm[m as usize]` to recover the input GT index.
    pub dt_matches: Array2<i64>,

    /// Shape `(T, G)`. For each `(threshold, sorted-GT k)`, the
    /// sorted-DT position of the matched DT, or `-1` if unmatched.
    pub gt_matches: Array2<i64>,

    /// Shape `(T, D)`. `true` if the DT at `(threshold, sorted-DT k)`
    /// matched an ignore-GT (quirk **B6**); `false` otherwise. The
    /// accumulator overlays the B7 area-range adjustment on top of
    /// this; the matching engine does not know about area ranges.
    pub dt_ignore: Array2<bool>,
}

/// Greedy assignment of detections to ground-truth annotations across
/// every IoU threshold.
///
/// Inputs are in the caller's natural order. `iou_matrix` has shape
/// `(G, D)` per ADR-0005 (rows = GT, cols = DT), produced by some
/// [`crate::Similarity`] impl. `parity_mode` is plumbed through for the
/// ADR-0002 contract; only **A1**'s corrected tiebreak would change
/// matching behavior, and that tiebreak needs an `ann_id` input the
/// ADR-0005 signature does not carry — so for now both modes match
/// strict numerically.
///
/// # Errors
///
/// Returns [`EvalError::DimensionMismatch`] if `iou_matrix` shape does
/// not equal `(gt_ignore.len(), dt_scores.len())`, or if `gt_ignore`
/// and `gt_iscrowd` have different lengths.
// Signature pinned by ADR-0005. `ArrayView2` is a `Copy` view (a
// `(ptr, dims, strides)` triple), so by-value is idiomatic; clippy's
// `needless_pass_by_value` doesn't recognize that here.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn match_image(
    iou_matrix: ArrayView2<'_, f64>,
    gt_ignore: &[bool],
    gt_iscrowd: &[bool],
    dt_scores: &[f64],
    iou_thresholds: &[f64],
    _parity_mode: ParityMode,
) -> Result<MatchResult, EvalError> {
    let n_g = gt_ignore.len();
    let n_d = dt_scores.len();
    let n_t = iou_thresholds.len();

    #[cfg(feature = "bench-histogram")]
    let _hist_guard = histogram::CallTimer::new(n_g, n_d);

    if gt_iscrowd.len() != n_g {
        return Err(EvalError::DimensionMismatch {
            detail: format!(
                "gt_iscrowd len {} does not match gt_ignore len {}",
                gt_iscrowd.len(),
                n_g
            ),
        });
    }
    if iou_matrix.nrows() != n_g || iou_matrix.ncols() != n_d {
        return Err(EvalError::DimensionMismatch {
            detail: format!(
                "iou_matrix is {}x{}, expected {}x{}",
                iou_matrix.nrows(),
                iou_matrix.ncols(),
                n_g,
                n_d
            ),
        });
    }

    let dt_perm = argsort_score_desc(dt_scores);

    // A4 (strict): stable sort GTs by `_ignore`, so non-ignore precede
    // ignore. The B3 early-termination relies on this layout.
    let mut gt_perm: Vec<usize> = (0..n_g).collect();
    gt_perm.sort_by_key(|&i| gt_ignore[i]);

    let gt_ignore_sorted: Vec<bool> = gt_perm.iter().map(|&i| gt_ignore[i]).collect();
    let gt_iscrowd_sorted: Vec<bool> = gt_perm.iter().map(|&i| gt_iscrowd[i]).collect();

    let mut dt_matches = Array2::<i64>::from_elem((n_t, n_d), -1);
    let mut gt_matches = Array2::<i64>::from_elem((n_t, n_g), -1);
    let mut dt_ignore = Array2::<bool>::default((n_t, n_d));

    if n_g == 0 || n_d == 0 || n_t == 0 {
        return Ok(MatchResult {
            dt_perm,
            gt_perm,
            dt_matches,
            gt_matches,
            dt_ignore,
        });
    }

    for (tind, &t) in iou_thresholds.iter().enumerate() {
        for (k_d, &d_orig) in dt_perm.iter().enumerate() {
            // B1: seed best at `min(t, 1 - 1e-10)` so a DT whose best
            // overlap exactly equals the threshold still matches.
            let mut best = t.min(1.0 - IOU_BOUNDARY_EPS);
            let mut m: i64 = -1;

            for k_g in 0..n_g {
                // B4: skip already-matched GT unless it is a crowd
                // (crowds are many-to-one).
                if gt_matches[(tind, k_g)] >= 0 && !gt_iscrowd_sorted[k_g] {
                    continue;
                }
                // B3: once a non-ignore match has been made, stop at the
                // first ignore-GT. A4 guarantees ignore-GTs are at the
                // tail, so this short-circuits the rest of the row.
                if m >= 0 && !gt_ignore_sorted[m as usize] && gt_ignore_sorted[k_g] {
                    break;
                }

                let g_orig = gt_perm[k_g];
                let iou = iou_matrix[(g_orig, d_orig)];

                // B2: non-strict comparison. On equality the later GT
                // wins (the update path runs when `iou >= best`).
                if iou < best {
                    continue;
                }
                best = iou;
                m = k_g as i64;
            }

            if m < 0 {
                continue;
            }
            let m_idx = m as usize;
            // B6: matched-to-ignore DTs inherit the ignore flag.
            dt_ignore[(tind, k_d)] = gt_ignore_sorted[m_idx];
            dt_matches[(tind, k_d)] = m;
            gt_matches[(tind, m_idx)] = k_d as i64;
        }
    }

    Ok(MatchResult {
        dt_perm,
        gt_perm,
        dt_matches,
        gt_matches,
        dt_ignore,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn run(
        iou: &Array2<f64>,
        gt_ignore: &[bool],
        gt_iscrowd: &[bool],
        dt_scores: &[f64],
        thresholds: &[f64],
    ) -> MatchResult {
        match_image(
            iou.view(),
            gt_ignore,
            gt_iscrowd,
            dt_scores,
            thresholds,
            ParityMode::Strict,
        )
        .unwrap()
    }

    #[test]
    fn perfect_match_all_dts_pair_with_distinct_gts() {
        let iou = array![[1.0, 0.0], [0.0, 1.0]];
        let r = run(&iou, &[false, false], &[false, false], &[0.9, 0.8], &[0.5]);

        // dt order = score-desc = [0, 1]; gt order = ignore-asc = [0, 1].
        assert_eq!(r.dt_perm, vec![0, 1]);
        assert_eq!(r.gt_perm, vec![0, 1]);
        assert_eq!(r.dt_matches[(0, 0)], 0);
        assert_eq!(r.dt_matches[(0, 1)], 1);
        assert_eq!(r.gt_matches[(0, 0)], 0);
        assert_eq!(r.gt_matches[(0, 1)], 1);
        assert!(!r.dt_ignore[(0, 0)] && !r.dt_ignore[(0, 1)]);
    }

    #[test]
    fn no_overlap_yields_no_matches() {
        let iou = array![[0.0, 0.0], [0.0, 0.0]];
        let r = run(&iou, &[false, false], &[false, false], &[0.9, 0.8], &[0.5]);
        assert!(r.dt_matches.iter().all(|&v| v == -1));
        assert!(r.gt_matches.iter().all(|&v| v == -1));
    }

    #[test]
    fn b1_iou_exactly_at_threshold_matches() {
        // IoU == 0.5 exactly. Without B1's `min(t, 1 - 1e-10)` seed, the
        // strict `< best` would reject. With it, the comparison passes.
        let iou = array![[0.5]];
        let r = run(&iou, &[false], &[false], &[0.9], &[0.5]);
        assert_eq!(r.dt_matches[(0, 0)], 0);
    }

    #[test]
    fn b1_iou_at_one_still_matches() {
        // The seed `1 - 1e-10` lets a perfect overlap match even at
        // threshold == 1.0, mirroring the pycocotools edge case.
        let iou = array![[1.0]];
        let r = run(&iou, &[false], &[false], &[0.9], &[1.0]);
        assert_eq!(r.dt_matches[(0, 0)], 0);
    }

    #[test]
    fn b4_crowd_gt_matches_many_dts() {
        // One crowd GT, two DTs both fully inside it. Both should match
        // the same crowd GT (many-to-one).
        let iou = array![[1.0, 1.0]];
        let r = run(&iou, &[false], &[true], &[0.9, 0.8], &[0.5]);
        assert_eq!(r.dt_matches[(0, 0)], 0);
        assert_eq!(r.dt_matches[(0, 1)], 0);
        // B4 corollary: `gt_matches` is overwritten on each successive
        // match, so it ends pointing at the last DT — pinning this
        // matters because pycocotools has the same overwrite semantics.
        assert_eq!(r.gt_matches[(0, 0)], 1);
    }

    #[test]
    fn a1_score_ties_resolve_to_input_order() {
        // Two DTs with identical scores. Stable mergesort keeps input
        // order: dt_perm == [0, 1]. DT 0 should claim GT 0 first; DT 1
        // gets the leftover (GT 1 here, with lower IoU).
        let iou = array![[0.9, 0.6], [0.6, 0.9]];
        let r = run(&iou, &[false, false], &[false, false], &[0.7, 0.7], &[0.5]);
        assert_eq!(r.dt_perm, vec![0, 1]);
        assert_eq!(r.dt_matches[(0, 0)], 0);
        assert_eq!(r.dt_matches[(0, 1)], 1);
    }

    #[test]
    fn b3_ignore_gt_terminates_inner_loop_after_real_match() {
        // GT 0 is non-ignore (becomes sorted-pos 0); GT 1 is ignore
        // (becomes sorted-pos 1). DT 0 overlaps both with IoU 0.8.
        // After matching GT 0, B3 must `break` rather than considering
        // GT 1 — otherwise the LAST-equal-wins behavior of B2 would
        // overwrite the match to the ignore-GT and shift dt_ignore.
        let iou = array![[0.8], [0.8]];
        let r = run(&iou, &[false, true], &[false, false], &[0.9], &[0.5]);
        assert_eq!(r.dt_matches[(0, 0)], 0); // sorted-GT pos 0 = real
        assert!(!r.dt_ignore[(0, 0)]);
    }

    #[test]
    fn b6_dt_matched_to_ignore_inherits_flag() {
        // Only one GT and it is ignore. The DT matches it; dt_ignore
        // must reflect that so the accumulator strips it from the
        // curve.
        let iou = array![[0.8]];
        let r = run(&iou, &[true], &[false], &[0.9], &[0.5]);
        assert_eq!(r.dt_matches[(0, 0)], 0);
        assert!(r.dt_ignore[(0, 0)]);
    }

    #[test]
    fn a4_gt_sort_puts_ignore_at_tail_regardless_of_input_order() {
        // Input has the ignore-GT first. A4's stable mergesort moves it
        // to the tail. gt_perm reflects the permutation.
        let iou = array![[0.0], [0.9]];
        let r = run(&iou, &[true, false], &[false, false], &[0.9], &[0.5]);
        assert_eq!(r.gt_perm, vec![1, 0]);
        // DT 0 matches the non-ignore GT at sorted-pos 0 (input idx 1).
        assert_eq!(r.dt_matches[(0, 0)], 0);
        assert!(!r.dt_ignore[(0, 0)]);
    }

    #[test]
    fn multiple_thresholds_accumulate_independently() {
        let iou = array![[0.6]];
        let r = run(&iou, &[false], &[false], &[0.9], &[0.5, 0.55, 0.6, 0.65]);
        assert_eq!(r.dt_matches[(0, 0)], 0);
        assert_eq!(r.dt_matches[(1, 0)], 0);
        // B1 seed handles the boundary at 0.6 exactly.
        assert_eq!(r.dt_matches[(2, 0)], 0);
        assert_eq!(r.dt_matches[(3, 0)], -1);
    }

    #[test]
    fn empty_inputs_return_empty_arrays() {
        let iou = Array2::<f64>::zeros((0, 0));
        let r = run(&iou, &[], &[], &[], &[0.5]);
        assert_eq!(r.dt_matches.shape(), &[1, 0]);
        assert_eq!(r.gt_matches.shape(), &[1, 0]);
        assert!(r.dt_perm.is_empty());
        assert!(r.gt_perm.is_empty());
    }

    #[test]
    fn empty_thresholds_yield_zero_row_matrices() {
        let iou = array![[0.9]];
        let r = run(&iou, &[false], &[false], &[0.9], &[]);
        assert_eq!(r.dt_matches.shape(), &[0, 1]);
        assert_eq!(r.gt_matches.shape(), &[0, 1]);
    }

    #[test]
    fn iou_matrix_dimension_mismatch_is_typed_error() {
        let iou = Array2::<f64>::zeros((1, 1));
        let err = match_image(
            iou.view(),
            &[false, false],
            &[false, false],
            &[0.9],
            &[0.5],
            ParityMode::Strict,
        )
        .unwrap_err();
        match err {
            EvalError::DimensionMismatch { detail } => {
                assert!(detail.contains("iou_matrix"));
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn gt_iscrowd_length_mismatch_is_typed_error() {
        let iou = Array2::<f64>::zeros((2, 1));
        let err = match_image(
            iou.view(),
            &[false, false],
            &[false],
            &[0.9],
            &[0.5],
            ParityMode::Strict,
        )
        .unwrap_err();
        assert!(matches!(err, EvalError::DimensionMismatch { .. }));
    }
}
