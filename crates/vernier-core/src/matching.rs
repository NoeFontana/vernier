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

/// The `(threshold, DT)` matching ladder.
///
/// Monomorphized on `PREFILTER`: with it off, `dt_best` is unused and
/// the skip test folds away, leaving the loop the compiler saw before
/// the prefilter existed. With it on, `dt_best[d]` is the DT's best
/// overlap with any GT — below the threshold seed, the DT cannot match
/// (`best` only ever rises from the seed, so every `iou < best` test
/// would `continue` and `m` would stay -1, which the `m < 0` guard
/// turns into a no-op), so its whole `G`-long scan is dead work.
/// Skipping it is output-identical, not an approximation.
// `ArrayView2` is a `Copy` view, so by-value is idiomatic here too
// (same rationale as the ADR-0005 entry points below).
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
fn run_ladder<const PREFILTER: bool>(
    iou_matrix: ArrayView2<'_, f64>,
    iou_thresholds: &[f64],
    dt_perm: &[usize],
    gt_perm: &[usize],
    gt_ignore_sorted: &[bool],
    gt_iscrowd_sorted: &[bool],
    dt_best: &[f64],
    dt_matches: &mut Array2<i64>,
    gt_matches: &mut Array2<i64>,
    dt_ignore: &mut Array2<bool>,
) {
    let n_g = gt_ignore_sorted.len();
    for (tind, &t) in iou_thresholds.iter().enumerate() {
        // B1: seed best at `min(t, 1 - 1e-10)` so a DT whose best
        // overlap exactly equals the threshold still matches.
        let seed = t.min(1.0 - IOU_BOUNDARY_EPS);
        for (k_d, &d_orig) in dt_perm.iter().enumerate() {
            if PREFILTER && dt_best[d_orig] < seed {
                continue;
            }
            let mut best = seed;
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
}

/// Minimum `G · D` at which the per-DT prefilter pays for itself.
///
/// The prefilter costs one pass over the IoU matrix plus a `D`-long
/// buffer, and saves a `G`-long scan per `(threshold, skipped DT)`. On
/// a COCO-shaped grid the median non-empty cell sits at `G · D = 1`
/// (see `benches/evaluate_bbox.rs`), where the scan it would skip is
/// shorter than its own setup. Dense cells — the surveillance /
/// autonomous-driving regime at `G · D` in the tens of thousands — are
/// where it earns its keep, so the gate is set well above the COCO
/// median and well below that regime.
const PREFILTER_MIN_CELL: usize = 256;

/// `out[d] = max_g iou[(g, d)]` — the best overlap each DT has with any
/// GT, used to skip DTs that cannot match at a given threshold.
///
/// Walks the matrix in row-major order (the layout ADR-0005 pins) so
/// the read stream stays sequential; the accumulator is the `D`-long
/// output, which is cache-resident for any cell worth prefiltering.
///
/// `NaN` maps to `+inf` rather than being dropped. `f64::max` ignores
/// `NaN`, but the match loop does not: `iou < best` is false for `NaN`,
/// so a `NaN` overlap *matches*. Column maxima must not be able to
/// mask that, and a `NaN` bbox can reach here through the ADR-0030
/// array-ingest path, which does not go via JSON.
#[allow(clippy::needless_pass_by_value)]
fn column_maxima(iou_matrix: ArrayView2<'_, f64>) -> Vec<f64> {
    let mut best = vec![f64::NEG_INFINITY; iou_matrix.ncols()];
    for g in 0..iou_matrix.nrows() {
        for (slot, &v) in best.iter_mut().zip(iou_matrix.row(g)) {
            *slot = if v.is_nan() {
                f64::INFINITY
            } else {
                slot.max(v)
            };
        }
    }
    best
}

/// Greedy assignment of detections to ground-truth annotations across
/// every IoU threshold.
///
/// Inputs are in the caller's natural order. `iou_matrix` has shape
/// `(G, D)` per ADR-0005 (rows = GT, cols = DT), produced by some
/// [`crate::similarity::Similarity`] impl. `parity_mode` is plumbed
/// through for the ADR-0002 contract; only **A1**'s corrected tiebreak
/// would change matching behavior, and that tiebreak needs an `ann_id`
/// input the ADR-0005 signature does not carry — so for now both
/// modes match strict numerically.
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
    parity_mode: ParityMode,
) -> Result<MatchResult, EvalError> {
    // Public entry: the DT order is unknown, so derive the
    // score-descending permutation (quirk A1) here. Callers that already
    // hold DTs in score-descending order — the `evaluate_with` hot path,
    // via `dt_top_indices_for_cell_into` — call [`match_image_with_perm`]
    // directly with an identity permutation to skip the redundant re-sort.
    let dt_perm = argsort_score_desc(dt_scores);
    match_image_with_perm(
        iou_matrix,
        gt_ignore,
        gt_iscrowd,
        dt_perm,
        iou_thresholds,
        parity_mode,
    )
}

/// [`match_image`] with a caller-supplied DT permutation.
///
/// `dt_perm` must index the columns of `iou_matrix` in score-descending
/// order with the stable A1 tiebreak — exactly what [`argsort_score_desc`]
/// returns for the DT scores. The `evaluate_with` hot path passes an
/// identity permutation: [`crate::evaluate::dt_top_indices_for_cell_into`]
/// already score-sorts each cell's DTs once per `(category, image)`, so
/// re-deriving the argsort for each of the four area ranges is pure
/// redundant work. Because that filter and `argsort_score_desc` use the
/// same comparator, the identity permutation is bit-identical to the
/// re-sorted one — parity-safe, including the `-0.0`/`+0.0` tie case.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn match_image_with_perm(
    iou_matrix: ArrayView2<'_, f64>,
    gt_ignore: &[bool],
    gt_iscrowd: &[bool],
    dt_perm: Vec<usize>,
    iou_thresholds: &[f64],
    parity_mode: ParityMode,
) -> Result<MatchResult, EvalError> {
    match_image_with_perm_gated(
        iou_matrix,
        gt_ignore,
        gt_iscrowd,
        dt_perm,
        iou_thresholds,
        parity_mode,
        PREFILTER_MIN_CELL,
    )
}

/// [`match_image_with_perm`] with the prefilter gate as a parameter.
///
/// The prefilter is a pure optimization — `prefilter_min_cell` moves
/// *when* it engages, never what comes out. Tests pin that by running a
/// cell through with the gate at `0` and at `usize::MAX` and comparing
/// the results.
#[allow(clippy::needless_pass_by_value)]
fn match_image_with_perm_gated(
    iou_matrix: ArrayView2<'_, f64>,
    gt_ignore: &[bool],
    gt_iscrowd: &[bool],
    dt_perm: Vec<usize>,
    iou_thresholds: &[f64],
    _parity_mode: ParityMode,
    prefilter_min_cell: usize,
) -> Result<MatchResult, EvalError> {
    let n_g = gt_ignore.len();
    let n_d = dt_perm.len();
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

    // A4 (strict): stable sort GTs by `_ignore`, so non-ignore precede
    // ignore. The B3 early-termination relies on this layout. NB:
    // `slice::sort_by_key` is stable (unstable is `sort_unstable_by_key`),
    // so GTs sharing an `_ignore` value keep their input order.
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

    // The ladder is monomorphized on whether the prefilter is active,
    // so a sub-gate cell compiles to exactly the pre-prefilter loop —
    // no residual per-`(threshold, DT)` branch on an inactive Option.
    // On a COCO-shaped grid, where the median non-empty cell is
    // `G · D = 1`, that branch alone cost ~4 %.
    if n_g * n_d >= prefilter_min_cell {
        let dt_best = column_maxima(iou_matrix);
        run_ladder::<true>(
            iou_matrix,
            iou_thresholds,
            &dt_perm,
            &gt_perm,
            &gt_ignore_sorted,
            &gt_iscrowd_sorted,
            &dt_best,
            &mut dt_matches,
            &mut gt_matches,
            &mut dt_ignore,
        );
    } else {
        run_ladder::<false>(
            iou_matrix,
            iou_thresholds,
            &dt_perm,
            &gt_perm,
            &gt_ignore_sorted,
            &gt_iscrowd_sorted,
            &[],
            &mut dt_matches,
            &mut gt_matches,
            &mut dt_ignore,
        );
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

    // ---- per-DT prefilter (column maxima) ----------------------------

    /// Run one cell twice — prefilter forced on, then forced off — and
    /// return both results. The gate is a pure performance switch, so
    /// the two must agree field for field.
    fn run_both_gates(
        iou: &Array2<f64>,
        gt_ignore: &[bool],
        gt_iscrowd: &[bool],
        dt_scores: &[f64],
        thresholds: &[f64],
    ) -> (MatchResult, MatchResult) {
        let forced = match_image_with_perm_gated(
            iou.view(),
            gt_ignore,
            gt_iscrowd,
            argsort_score_desc(dt_scores),
            thresholds,
            ParityMode::Strict,
            0,
        )
        .expect("prefilter forced on");
        let disabled = match_image_with_perm_gated(
            iou.view(),
            gt_ignore,
            gt_iscrowd,
            argsort_score_desc(dt_scores),
            thresholds,
            ParityMode::Strict,
            usize::MAX,
        )
        .expect("prefilter disabled");
        (forced, disabled)
    }

    fn assert_same_matching(label: &str, a: &MatchResult, b: &MatchResult) {
        assert_eq!(a.dt_perm, b.dt_perm, "{label}: dt_perm");
        assert_eq!(a.gt_perm, b.gt_perm, "{label}: gt_perm");
        assert_eq!(a.dt_matches, b.dt_matches, "{label}: dt_matches");
        assert_eq!(a.gt_matches, b.gt_matches, "{label}: gt_matches");
        assert_eq!(a.dt_ignore, b.dt_ignore, "{label}: dt_ignore");
    }

    /// A dense cell with the full 10-threshold ladder, crowds, ignores,
    /// score ties and a broad spread of overlaps — including values that
    /// sit exactly on a threshold, where the B1 seed decides the match.
    #[test]
    fn prefilter_agrees_with_the_unfiltered_scan_on_a_dense_cell() {
        let n_g = 24;
        let n_d = 32;
        let thresholds = crate::parity::iou_thresholds();

        // Deterministic spread: many columns land below 0.5 (the
        // skippable majority), some straddle individual thresholds.
        let iou = Array2::from_shape_fn((n_g, n_d), |(g, d)| {
            if d % 3 == 0 {
                // Junk DT: overlaps nothing above the lowest rung, so
                // the prefilter skips its whole scan at every threshold.
                0.03 * ((g + d) % 5) as f64
            } else if (g + d) % 7 == 0 {
                // Exactly on a ladder rung: the B1 boundary case.
                thresholds[(g + d) % thresholds.len()]
            } else {
                (((g * 37 + d * 11) % 23) as f64 / 22.0) * 0.95
            }
        });
        let gt_ignore: Vec<bool> = (0..n_g).map(|g| g % 5 == 0).collect();
        let gt_iscrowd: Vec<bool> = (0..n_g).map(|g| g % 9 == 0).collect();
        let dt_scores: Vec<f64> = (0..n_d)
            .map(|d| 1.0 - ((d / 2) as f64) / (n_d as f64))
            .collect();

        let (forced, disabled) =
            run_both_gates(&iou, &gt_ignore, &gt_iscrowd, &dt_scores, thresholds);
        assert_same_matching("dense cell", &forced, &disabled);

        // Guard against a vacuous test: the cell must actually produce
        // matches, and must actually have skippable columns.
        assert!(forced.dt_matches.iter().any(|&m| m >= 0));
        let best = column_maxima(iou.view());
        assert!(best.iter().any(|&b| b < 0.5), "no column is skippable");
    }

    /// An all-low cell: every column is below the lowest rung, so the
    /// prefilter skips every DT at every threshold. Nothing may match.
    #[test]
    fn prefilter_skipping_every_dt_leaves_the_cell_unmatched() {
        let iou = Array2::from_shape_fn((20, 20), |(g, d)| 0.01 * ((g + d) % 4) as f64);
        let gt_ignore = vec![false; 20];
        let gt_iscrowd = vec![false; 20];
        let dt_scores: Vec<f64> = (0..20).map(|d| 1.0 - d as f64 / 20.0).collect();
        let thresholds = crate::parity::iou_thresholds();

        let (forced, disabled) =
            run_both_gates(&iou, &gt_ignore, &gt_iscrowd, &dt_scores, thresholds);
        assert_same_matching("all-low cell", &forced, &disabled);
        assert!(forced.dt_matches.iter().all(|&m| m < 0));
    }

    /// `NaN` overlap *matches* in pycocotools (`iou < best` is false for
    /// `NaN`, so the update path runs). `f64::max` would drop it from
    /// the column maxima and the prefilter would then skip a DT that
    /// the scan matches, so `column_maxima` sends `NaN` to `+inf`.
    /// A `NaN` bbox reaches here through ADR-0030 array ingest, which
    /// does not pass through JSON.
    #[test]
    fn nan_overlap_survives_the_prefilter() {
        let n_g = 20;
        let n_d = 20;
        let mut iou = Array2::from_elem((n_g, n_d), 0.02);
        // One DT whose only non-trivial entry is NaN.
        iou[(3, 7)] = f64::NAN;
        let gt_ignore = vec![false; n_g];
        let gt_iscrowd = vec![false; n_g];
        let dt_scores: Vec<f64> = (0..n_d).map(|d| 1.0 - d as f64 / 100.0).collect();
        let thresholds = [0.5];

        let (forced, disabled) =
            run_both_gates(&iou, &gt_ignore, &gt_iscrowd, &dt_scores, &thresholds);
        assert_same_matching("nan cell", &forced, &disabled);

        assert_eq!(column_maxima(iou.view())[7], f64::INFINITY);
        // DT 7 is at sorted position 7 (scores are strictly decreasing)
        // and matches on both paths. It lands on the *last* GT, not on
        // GT 3: once `best` is `NaN`, every later `iou < best` is false
        // too, so the update path runs for each remaining GT in turn.
        // That is what the reference does, and reproducing it is the
        // point — the prefilter must not turn it into a non-match.
        assert_eq!(forced.dt_matches[(0, 7)], (n_g - 1) as i64);
    }

    /// The gate itself: below `PREFILTER_MIN_CELL` no buffer is built,
    /// above it one is — and neither changes the answer. Checked at the
    /// boundary so a future retune cannot silently flip behaviour.
    #[test]
    fn prefilter_gate_does_not_change_results_at_its_boundary() {
        let thresholds = crate::parity::iou_thresholds();
        for n in [15usize, 16, 17] {
            let iou = Array2::from_shape_fn((n, n), |(g, d)| ((g * 3 + d) % 11) as f64 / 10.0);
            let gt_ignore: Vec<bool> = (0..n).map(|g| g % 4 == 0).collect();
            let gt_iscrowd = vec![false; n];
            let dt_scores: Vec<f64> = (0..n).map(|d| 1.0 - d as f64 / (n as f64)).collect();

            let (forced, disabled) =
                run_both_gates(&iou, &gt_ignore, &gt_iscrowd, &dt_scores, thresholds);
            assert_same_matching(&format!("n={n}"), &forced, &disabled);

            let auto = match_image(
                iou.view(),
                &gt_ignore,
                &gt_iscrowd,
                &dt_scores,
                thresholds,
                ParityMode::Strict,
            )
            .expect("auto gate");
            assert_same_matching(&format!("auto n={n}"), &auto, &disabled);
        }
        // 16x16 = 256 is the first cell that engages the prefilter.
        assert_eq!(PREFILTER_MIN_CELL, 256);
    }
}
