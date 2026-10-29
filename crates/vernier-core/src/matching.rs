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
//!   the trailing block of ignore-GTs is short-circuited. pycocotools
//!   `break`s at the first ignore-GT; here A4's partition makes it
//!   structural — the ignore tail is scanned only when the non-ignore
//!   block produced no match, which is the same set of outcomes.
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

    /// The A4 boundary: sorted GTs `0..n_unignored` are non-ignore, the
    /// rest ignore — so a sorted GT `k` is ignored iff `k >= n_unignored`.
    pub n_unignored: usize,
}

/// One `(threshold, DT)` step of the ladder — the B1-B4 / B6 rules, in
/// one place for both loop orders of [`run_ladder`].
///
/// `iou_at(k)` is the DT's overlap with sorted GT `k`; `taken[k] >= 0`
/// means GT `k` already matched at this threshold (**B4**: out of
/// contention unless a crowd, which is many-to-one). `iou < best` comes
/// first because nearly every overlap falls below the seed, so that exit
/// is the common, well-predicted one (a branch-free select measured up to
/// 60 % slower: it chains every GT on `best`). It is false for a `NaN`,
/// which therefore matches — as in pycocotools. **B2**: on equality the
/// later GT wins.
///
/// **B3 is structural.** GTs arrive A4-partitioned, so the DT scans the
/// non-ignore block `..n_unignored` and goes on into the ignore tail only
/// if nothing matched there — exactly pycocotools' "break at the first
/// ignore-GT once a real match exists", with the test off the inner loop.
///
/// Returns the matched sorted GT, already marked taken by DT `k_d`; the
/// caller records it — **B6** (a DT matched to an ignore-GT inherits the
/// ignore flag) is `m >= n_unignored`.
#[inline(always)]
fn ladder_step(
    iou_at: impl Fn(usize) -> f64,
    seed: f64,
    taken: &mut [i64],
    crowd: &[bool],
    n_unignored: usize,
    k_d: usize,
) -> Option<usize> {
    // B1: the caller passes the seed `min(t, 1 - 1e-10)`.
    let mut best = seed;
    let mut m = usize::MAX;
    let scan = |range: std::ops::Range<usize>, best: &mut f64, m: &mut usize| {
        for k in range {
            let iou = iou_at(k);
            // B2: `<`, so equality updates and the later GT wins.
            // B4: a taken GT is out of contention unless it is a crowd.
            if iou < *best || (taken[k] >= 0 && !crowd[k]) {
                continue;
            }
            *best = iou;
            *m = k;
        }
    };
    scan(0..n_unignored, &mut best, &mut m);
    // B3: the ignore tail is reached only without a real match.
    if m == usize::MAX {
        scan(n_unignored..crowd.len(), &mut best, &mut m);
    }
    if m == usize::MAX {
        return None;
    }
    taken[m] = k_d as i64;
    Some(m)
}

/// Where [`run_ladder`] writes a match, and how a score rank maps to its
/// IoU-matrix column.
///
/// The ladder — quirks **B1**-**B4** and **B6** — exists once; the three
/// output shapes (the [`MatchResult`] index matrices behind
/// `EvalImageMeta`, and the two packed accumulator layouts of ADR-0066)
/// are sinks it is monomorphized over, so a parity fix to the scan lands
/// on every path at once.
trait LadderSink {
    /// IoU-matrix column of the `rank`-th score-sorted DT.
    fn column(&self, rank: usize) -> usize;
    /// DT `rank` matched sorted GT `m` at threshold `tind`; `to_ignore`
    /// is **B6** (`m` is an ignore-GT).
    fn record(&mut self, tind: usize, rank: usize, m: usize, to_ignore: bool);
}

/// The [`MatchResult`] `(T, D)` matrices, flat; columns via `dt_perm`.
/// (`gt_matches` is the ladder's own taken-state.)
struct IndexSink<'a> {
    dt_perm: &'a [usize],
    dt_matches: &'a mut [i64],
    dt_ignore: &'a mut [bool],
}

impl LadderSink for IndexSink<'_> {
    #[inline(always)]
    fn column(&self, rank: usize) -> usize {
        self.dt_perm[rank]
    }

    #[inline(always)]
    fn record(&mut self, tind: usize, rank: usize, m: usize, to_ignore: bool) {
        let at = tind * self.dt_perm.len() + rank;
        self.dt_matches[at] = m as i64;
        self.dt_ignore[at] = to_ignore;
    }
}

/// One `matched_u16 | ignore_u16 << 16` word per pre-sorted DT
/// (`n_t <= 16`), pre-filled with the unmatched-ignore bits.
struct Bits16Sink<'a> {
    flags: &'a mut [u32],
}

impl LadderSink for Bits16Sink<'_> {
    #[inline(always)]
    fn column(&self, rank: usize) -> usize {
        rank
    }

    #[inline(always)]
    fn record(&mut self, tind: usize, rank: usize, _m: usize, to_ignore: bool) {
        let m_bit = 1_u32 << tind;
        let g_bit = 1_u32 << (tind + 16);
        // A match overrides the pre-filled unmatched-ignore (B7 / AA3)
        // bit with B6's: set when the GT is an ignore-GT, cleared if not.
        let slot = &mut self.flags[rank];
        *slot = (*slot | m_bit) & !g_bit | if to_ignore { g_bit } else { 0 };
    }
}

/// `[rank * n_t + tind]` booleans per pre-sorted DT (`n_t > 16`),
/// `ignore` pre-filled with the unmatched-ignore flags.
struct GeneralSink<'a> {
    n_t: usize,
    matched: &'a mut [bool],
    ignore: &'a mut [bool],
}

impl LadderSink for GeneralSink<'_> {
    #[inline(always)]
    fn column(&self, rank: usize) -> usize {
        rank
    }

    #[inline(always)]
    fn record(&mut self, tind: usize, rank: usize, _m: usize, to_ignore: bool) {
        let flat = rank * self.n_t + tind;
        self.matched[flat] = true;
        self.ignore[flat] = to_ignore;
    }
}

/// The greedy matching ladder — pycocotools' loop (`cocoeval.py:268-296`),
/// in two loop orders over one [`ladder_step`].
///
/// **Dense cells** (`PREFILTER`, `G · D >= PREFILTER_MIN_CELL`) run
/// DT-outer, threshold-inner. Each threshold's matching state (its
/// `gt_matches` row) depends only on earlier DTs at that threshold, so
/// the interchange is output-identical, and it lets each surviving DT's
/// IoU column be gathered **once**, in sorted-GT order, into a contiguous
/// buffer reused by all `T` thresholds: `G` strided loads per DT instead
/// of `T · G`, the `gt_perm` indirection out of the inner loop. The
/// prefilter skips a DT below every seed before the gather, and a
/// `(threshold, DT)` whose best overlap is below that seed (`best` only
/// rises from the seed, so it cannot match) — both output-identical.
///
/// **Sparse cells** (the COCO median is `G · D = 1`) keep the
/// threshold-outer order and read through `gt_perm` directly: the matrix
/// is cache-resident there, and both the gather and the interchange
/// measured as pure overhead (+5 % on the COCO-shaped bench).
///
/// `taken` is the flat row-major `(T, G)` taken-state, `-1`-filled on
/// entry (it ends as `gtMatches`); matches go to `sink`. `column` is the
/// dense path's gather buffer.
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
#[inline(always)]
fn run_ladder<const PREFILTER: bool, S: LadderSink>(
    iou_matrix: ArrayView2<'_, f64>,
    iou_thresholds: &[f64],
    n_d: usize,
    gt_perm: &[usize],
    n_unignored: usize,
    gt_iscrowd_sorted: &[bool],
    dt_best: &[f64],
    column: &mut Vec<f64>,
    taken: &mut [i64],
    sink: &mut S,
) {
    let n_g = gt_perm.len();
    // B1: seed best at `min(t, 1 - 1e-10)` so a DT whose best overlap
    // exactly equals the threshold still matches.
    let seed_of = |t: f64| t.min(1.0 - IOU_BOUNDARY_EPS);

    if PREFILTER {
        let min_seed = iou_thresholds
            .iter()
            .map(|&t| seed_of(t))
            .fold(f64::INFINITY, f64::min);
        column.clear();
        column.resize(n_g, 0.0);
        for k_d in 0..n_d {
            let col = sink.column(k_d);
            if dt_best[col] < min_seed {
                continue;
            }
            let iou_col = iou_matrix.column(col);
            for (slot, &g) in column.iter_mut().zip(gt_perm) {
                *slot = iou_col[g];
            }
            for (tind, &t) in iou_thresholds.iter().enumerate() {
                let seed = seed_of(t);
                if dt_best[col] < seed {
                    continue;
                }
                if let Some(m) = ladder_step(
                    |k| column[k],
                    seed,
                    &mut taken[tind * n_g..(tind + 1) * n_g],
                    gt_iscrowd_sorted,
                    n_unignored,
                    k_d,
                ) {
                    sink.record(tind, k_d, m, m >= n_unignored);
                }
            }
        }
    } else {
        for (tind, &t) in iou_thresholds.iter().enumerate() {
            let seed = seed_of(t);
            let gt_row = &mut taken[tind * n_g..(tind + 1) * n_g];
            for k_d in 0..n_d {
                let col = sink.column(k_d);
                if let Some(m) = ladder_step(
                    |k| iou_matrix[(gt_perm[k], col)],
                    seed,
                    gt_row,
                    gt_iscrowd_sorted,
                    n_unignored,
                    k_d,
                ) {
                    sink.record(tind, k_d, m, m >= n_unignored);
                }
            }
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

/// `out[d] = max_g iou[(g, d)]` — the best overlap each DT column has
/// with any GT, used to skip DTs that cannot match at a given threshold.
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
fn column_maxima_into(out: &mut Vec<f64>, iou_matrix: ArrayView2<'_, f64>) {
    #[inline(always)]
    fn fold_row<'r>(out: &mut [f64], row: impl IntoIterator<Item = &'r f64>) {
        for (slot, &v) in out.iter_mut().zip(row) {
            *slot = if v.is_nan() {
                f64::INFINITY
            } else {
                slot.max(v)
            };
        }
    }
    let n_d = iou_matrix.ncols();
    out.clear();
    out.resize(n_d, f64::NEG_INFINITY);
    // Every kernel writes a standard-layout matrix, so the common case is
    // one flat slice: plain `&[f64]` rows let the fold vectorize. (ndarray's
    // generic `rows()` iterator measured 2.3x slower; `row(g)` is the
    // strided fallback.)
    match iou_matrix.as_slice() {
        Some(flat) if n_d > 0 => {
            for row in flat.chunks_exact(n_d) {
                fold_row(out, row);
            }
        }
        _ => {
            for g in 0..iou_matrix.nrows() {
                fold_row(out, iou_matrix.row(g));
            }
        }
    }
}

/// The prefilter input for one `(category, image)` cell: its column
/// maxima when `G · D >= min_cell`, else `None`.
///
/// It depends on the IoU matrix only — not on the area range's GT
/// ignore flags — so it is computed once per cell
/// ([`MatchScratch::prepare_iou`]) and shared by every area range.
#[allow(clippy::needless_pass_by_value)]
fn gated_prefilter<'b>(
    buf: &'b mut Vec<f64>,
    iou_matrix: ArrayView2<'_, f64>,
    min_cell: usize,
) -> Option<&'b [f64]> {
    if iou_matrix.nrows() * iou_matrix.ncols() < min_cell {
        return None;
    }
    column_maxima_into(buf, iou_matrix);
    Some(buf)
}

/// Dispatch [`run_ladder`] on the prefilter. Monomorphized on it, so a
/// sub-gate cell compiles to the plain loop — no per-`(DT, threshold)`
/// branch on an inactive Option. On a COCO-shaped grid (median
/// `G · D = 1`) that branch alone cost ~4 %.
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
#[inline(always)]
fn ladder<S: LadderSink>(
    iou_matrix: ArrayView2<'_, f64>,
    iou_thresholds: &[f64],
    n_d: usize,
    gt_perm: &[usize],
    n_unignored: usize,
    gt_iscrowd_sorted: &[bool],
    column_maxima: Option<&[f64]>,
    column: &mut Vec<f64>,
    taken: &mut [i64],
    sink: &mut S,
) {
    match column_maxima {
        Some(dt_best) => run_ladder::<true, S>(
            iou_matrix,
            iou_thresholds,
            n_d,
            gt_perm,
            n_unignored,
            gt_iscrowd_sorted,
            dt_best,
            column,
            taken,
            sink,
        ),
        None => run_ladder::<false, S>(
            iou_matrix,
            iou_thresholds,
            n_d,
            gt_perm,
            n_unignored,
            gt_iscrowd_sorted,
            &[],
            column,
            taken,
            sink,
        ),
    }
}

/// The shape checks every matcher entry shares.
#[allow(clippy::needless_pass_by_value)]
fn check_cell_dims(
    iou_matrix: ArrayView2<'_, f64>,
    n_g: usize,
    n_d: usize,
    n_crowd: usize,
) -> Result<(), EvalError> {
    if n_crowd != n_g {
        return Err(EvalError::DimensionMismatch {
            detail: format!("gt_iscrowd len {n_crowd} does not match gt_ignore len {n_g}"),
        });
    }
    if iou_matrix.nrows() != n_g || iou_matrix.ncols() != n_d {
        return Err(EvalError::DimensionMismatch {
            detail: format!(
                "iou_matrix is {}x{}, expected {n_g}x{n_d}",
                iou_matrix.nrows(),
                iou_matrix.ncols(),
            ),
        });
    }
    Ok(())
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

/// [`match_image`] with a caller-supplied DT permutation and prefilter.
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
///
/// `column_maxima` is [`MatchScratch::prefilter`] for this cell, shared
/// across its area ranges; `None` runs the unfiltered ladder.
///
/// # Errors
///
/// As [`match_image`], plus [`EvalError::DimensionMismatch`] when
/// `column_maxima` does not cover every DT column.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn match_image_with_perm(
    iou_matrix: ArrayView2<'_, f64>,
    gt_ignore: &[bool],
    gt_iscrowd: &[bool],
    dt_perm: Vec<usize>,
    iou_thresholds: &[f64],
    _parity_mode: ParityMode,
    column_maxima: Option<&[f64]>,
) -> Result<MatchResult, EvalError> {
    let n_g = gt_ignore.len();
    let n_d = dt_perm.len();
    let n_t = iou_thresholds.len();

    #[cfg(feature = "bench-histogram")]
    let _hist_guard = histogram::CallTimer::new(n_g, n_d);

    check_cell_dims(iou_matrix, n_g, n_d, gt_iscrowd.len())?;
    if let Some(best) = column_maxima.filter(|b| b.len() != n_d) {
        return Err(EvalError::DimensionMismatch {
            detail: format!("prefilter covers {} DT columns, expected {n_d}", best.len()),
        });
    }

    // A4 (strict): GTs stably partitioned by `_ignore` — non-ignore first,
    // each block in input order. One O(G) pass, the same permutation a
    // stable sort by the boolean key produces.
    let mut gt_perm: Vec<usize> = Vec::with_capacity(n_g);
    gt_perm.extend((0..n_g).filter(|&g| !gt_ignore[g]));
    let n_unignored = gt_perm.len();
    gt_perm.extend((0..n_g).filter(|&g| gt_ignore[g]));
    let gt_iscrowd_sorted: Vec<bool> = gt_perm.iter().map(|&g| gt_iscrowd[g]).collect();

    let mut dt_matches = Array2::<i64>::from_elem((n_t, n_d), -1);
    let mut gt_matches = Array2::<i64>::from_elem((n_t, n_g), -1);
    let mut dt_ignore = Array2::<bool>::default((n_t, n_d));

    if n_g != 0 && n_d != 0 && n_t != 0 {
        debug_assert!(
            gt_perm[..n_unignored].iter().all(|&g| !gt_ignore[g])
                && gt_perm[n_unignored..].iter().all(|&g| gt_ignore[g]),
            "run_ladder's B3/B6 read the A4 partition positionally"
        );
        // Freshly built arrays are standard layout, so these are `Some`.
        let not_flat = || EvalError::DimensionMismatch {
            detail: "match output is not contiguous".into(),
        };
        let mut sink = IndexSink {
            dt_perm: &dt_perm,
            dt_matches: dt_matches.as_slice_mut().ok_or_else(not_flat)?,
            dt_ignore: dt_ignore.as_slice_mut().ok_or_else(not_flat)?,
        };
        let taken = gt_matches.as_slice_mut().ok_or_else(not_flat)?;
        ladder(
            iou_matrix,
            iou_thresholds,
            n_d,
            &gt_perm,
            n_unignored,
            &gt_iscrowd_sorted,
            column_maxima,
            &mut Vec::new(),
            taken,
            &mut sink,
        );
    }

    Ok(MatchResult {
        dt_perm,
        gt_perm,
        dt_matches,
        gt_matches,
        dt_ignore,
        n_unignored,
    })
}

/// [`match_image_with_perm`] computing its own prefilter at gate
/// `prefilter_min_cell`. The prefilter is a pure optimization — the gate
/// moves *when* it engages, never what comes out; tests pin that by
/// running cells at gates `0` and `usize::MAX`.
#[allow(clippy::needless_pass_by_value)]
fn match_image_with_perm_gated(
    iou_matrix: ArrayView2<'_, f64>,
    gt_ignore: &[bool],
    gt_iscrowd: &[bool],
    dt_perm: Vec<usize>,
    iou_thresholds: &[f64],
    parity_mode: ParityMode,
    prefilter_min_cell: usize,
) -> Result<MatchResult, EvalError> {
    let mut buf = Vec::new();
    let prefilter = gated_prefilter(&mut buf, iou_matrix, prefilter_min_cell);
    match_image_with_perm(
        iou_matrix,
        gt_ignore,
        gt_iscrowd,
        dt_perm,
        iou_thresholds,
        parity_mode,
        prefilter,
    )
}

/// Reusable per-worker scratch for allocation-free matching straight
/// into [`crate::accumulate::PackedCategory`] (ADR-0066).
///
/// The `!builds_meta` hot path: DTs arrive score-sorted (the identity
/// permutation) and no `EvalImageMeta` is wanted, so the ladder writes
/// the accumulator's flag layout directly instead of index matrices.
/// The matching itself is [`run_ladder`], shared with [`match_image`].
#[derive(Default)]
pub(crate) struct MatchScratch {
    gt_perm: Vec<usize>,
    gt_iscrowd_sorted: Vec<bool>,
    taken: Vec<i64>,
    column: Vec<f64>,
    column_maxima: Vec<f64>,
    prefilter: bool,
}

impl MatchScratch {
    /// Compute the cell's prefilter ([`gated_prefilter`] at
    /// [`PREFILTER_MIN_CELL`]): once per `(category, image)` cell, shared
    /// by every area range. Must run before the cell's matcher calls.
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn prepare_iou(&mut self, iou_matrix: ArrayView2<'_, f64>) {
        self.prepare_iou_gated(iou_matrix, PREFILTER_MIN_CELL);
    }

    /// The prefilter [`Self::prepare_iou`] computed for the current cell,
    /// for [`match_image_with_perm`] on the metadata path.
    pub(crate) fn prefilter(&self) -> Option<&[f64]> {
        self.prefilter.then_some(&self.column_maxima[..])
    }

    #[allow(clippy::needless_pass_by_value)]
    fn prepare_iou_gated(&mut self, iou_matrix: ArrayView2<'_, f64>, min_cell: usize) {
        self.prefilter = gated_prefilter(&mut self.column_maxima, iou_matrix, min_cell).is_some();
    }

    /// A4 stable partition into `gt_perm` (non-ignore block first) with
    /// the crowd flags permuted alongside, and the `(T, G)` taken-state
    /// reset. Returns the non-ignore count.
    fn partition_gts(&mut self, gt_ignore: &[bool], gt_iscrowd: &[bool], n_t: usize) -> usize {
        self.gt_perm.clear();
        self.gt_perm
            .extend((0..gt_ignore.len()).filter(|&g| !gt_ignore[g]));
        let n_unignored = self.gt_perm.len();
        self.gt_perm
            .extend((0..gt_ignore.len()).filter(|&g| gt_ignore[g]));
        self.gt_iscrowd_sorted.clear();
        self.gt_iscrowd_sorted
            .extend(self.gt_perm.iter().map(|&g| gt_iscrowd[g]));
        self.taken.clear();
        self.taken.resize(n_t * gt_ignore.len(), -1);
        n_unignored
    }

    #[allow(clippy::needless_pass_by_value)]
    #[inline(always)]
    fn run<S: LadderSink>(
        &mut self,
        iou_matrix: ArrayView2<'_, f64>,
        iou_thresholds: &[f64],
        n_d: usize,
        n_unignored: usize,
        sink: &mut S,
    ) {
        debug_assert!(!self.prefilter || self.column_maxima.len() == iou_matrix.ncols());
        ladder(
            iou_matrix,
            iou_thresholds,
            n_d,
            &self.gt_perm,
            n_unignored,
            &self.gt_iscrowd_sorted,
            self.prefilter.then_some(&self.column_maxima[..]),
            &mut self.column,
            &mut self.taken,
            sink,
        );
    }

    /// Greedy matching of pre-sorted DTs (`n_t <= 16`), appending `n_d`
    /// `matched_u16 | ignore_u16 << 16` words to `out_flags`.
    /// `dt_unmatched_ignore[d]` is the ignore flag an *unmatched* DT
    /// carries (B7 / AA3). Returns `n_unignored`.
    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    pub(crate) fn match_presorted_bits16(
        &mut self,
        iou_matrix: ArrayView2<'_, f64>,
        gt_ignore: &[bool],
        gt_iscrowd: &[bool],
        dt_unmatched_ignore: &[bool],
        iou_thresholds: &[f64],
        out_flags: &mut Vec<u32>,
    ) -> Result<usize, EvalError> {
        let n_g = gt_ignore.len();
        let n_d = dt_unmatched_ignore.len();
        let n_t = iou_thresholds.len();
        debug_assert!(n_t <= 16);

        #[cfg(feature = "bench-histogram")]
        let _hist_guard = histogram::CallTimer::new(n_g, n_d);

        check_cell_dims(iou_matrix, n_g, n_d, gt_iscrowd.len())?;
        let n_unignored = self.partition_gts(gt_ignore, gt_iscrowd, n_t);
        let all_t_ig: u32 = ((1_u32 << n_t) - 1) << 16;
        let base = out_flags.len();
        out_flags.extend(
            dt_unmatched_ignore
                .iter()
                .map(|&ig| if ig { all_t_ig } else { 0 }),
        );
        if n_g != 0 && n_d != 0 && n_t != 0 {
            let mut sink = Bits16Sink {
                flags: &mut out_flags[base..],
            };
            self.run(iou_matrix, iou_thresholds, n_d, n_unignored, &mut sink);
        }
        Ok(n_unignored)
    }

    /// Arbitrary-`n_t` fallback (`n_t > 16`) for pre-sorted DTs:
    /// appends `n_d * n_t` booleans (layout `[d * n_t + tind]`) to
    /// `out_matched` and `out_ignore`. Returns `n_unignored`.
    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    pub(crate) fn match_presorted_general(
        &mut self,
        iou_matrix: ArrayView2<'_, f64>,
        gt_ignore: &[bool],
        gt_iscrowd: &[bool],
        dt_unmatched_ignore: &[bool],
        iou_thresholds: &[f64],
        out_matched: &mut Vec<bool>,
        out_ignore: &mut Vec<bool>,
    ) -> Result<usize, EvalError> {
        let n_g = gt_ignore.len();
        let n_d = dt_unmatched_ignore.len();
        let n_t = iou_thresholds.len();

        #[cfg(feature = "bench-histogram")]
        let _hist_guard = histogram::CallTimer::new(n_g, n_d);

        check_cell_dims(iou_matrix, n_g, n_d, gt_iscrowd.len())?;
        let n_unignored = self.partition_gts(gt_ignore, gt_iscrowd, n_t);
        let base = out_matched.len();
        out_matched.resize(base + n_d * n_t, false);
        out_ignore.reserve(n_d * n_t);
        for &ig in dt_unmatched_ignore {
            out_ignore.extend(std::iter::repeat_n(ig, n_t));
        }
        if n_g != 0 && n_d != 0 && n_t != 0 {
            let mut sink = GeneralSink {
                n_t,
                matched: &mut out_matched[base..],
                ignore: &mut out_ignore[base..],
            };
            self.run(iou_matrix, iou_thresholds, n_d, n_unignored, &mut sink);
        }
        Ok(n_unignored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[allow(clippy::needless_pass_by_value)]
    fn column_maxima(iou: ArrayView2<'_, f64>) -> Vec<f64> {
        let mut out = Vec::new();
        column_maxima_into(&mut out, iou);
        out
    }

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

    /// `dt_best` is indexed by the DT's **original column**, not by its
    /// rank in `dt_perm`. Every other prefilter test above feeds
    /// score-descending `dt_scores`, so `dt_perm` is the identity and
    /// the two indices coincide — which hides the difference entirely.
    ///
    /// The permutation is genuinely non-identity in production:
    /// `crate::tide::assignment` calls [`match_image`] with raw,
    /// unsorted `dt_scores` over whole-category cells, which are large
    /// enough to cross the `PREFILTER_MIN_CELL` gate. So this cell
    /// shuffles the scores and anti-correlates overlap quality with
    /// score rank: a detection with a real match sits at a rank whose
    /// *column index* overlaps nothing. Confusing the two indices skips
    /// every true match and turns it into a false negative.
    #[test]
    fn prefilter_indexes_dt_best_by_original_column_not_by_rank() {
        let n_g = 16;
        let n_d = 24; // 16 * 24 = 384, comfortably above the gate.
        let thresholds = crate::parity::iou_thresholds();

        // A shuffled permutation of `0..n_d` (7 is coprime to 24), so
        // `argsort_score_desc` is nowhere near the identity. Its parity
        // inverts: rank `k` holds column `7 * (n_d - 1 - k) mod n_d`,
        // whose parity is opposite to `k`'s.
        let dt_scores: Vec<f64> = (0..n_d)
            .map(|d| ((d * 7) % n_d) as f64 / n_d as f64)
            .collect();

        // Overlap quality keys off the *column* index: odd columns hold
        // one strong match, even columns overlap nothing above the
        // lowest rung. Combined with the parity inversion above, reading
        // `dt_best` by rank inspects exactly the wrong column every time.
        let iou = Array2::from_shape_fn((n_g, n_d), |(g, d)| {
            if d % 2 == 0 {
                0.04 * ((g + d) % 3) as f64
            } else if g == d / 2 {
                0.95 - 0.01 * d as f64
            } else {
                0.05 * ((g * 5 + d) % 4) as f64
            }
        });
        // Interleaved ignores, so `gt_perm` is non-identity too, plus a
        // crowd — the B3/B4 paths run alongside the shuffled DT order.
        let gt_ignore: Vec<bool> = (0..n_g).map(|g| g % 7 == 3).collect();
        let gt_iscrowd: Vec<bool> = (0..n_g).map(|g| g == 5).collect();

        let (forced, disabled) =
            run_both_gates(&iou, &gt_ignore, &gt_iscrowd, &dt_scores, thresholds);
        assert_same_matching("shuffled dt scores", &forced, &disabled);

        // Guard against the test going vacuous: the permutations must
        // really be non-identity, and the cell must really match.
        assert_ne!(
            forced.dt_perm,
            (0..n_d).collect::<Vec<usize>>(),
            "dt_perm must not be the identity"
        );
        assert_ne!(
            forced.gt_perm,
            (0..n_g).collect::<Vec<usize>>(),
            "gt_perm must not be the identity"
        );
        assert!(forced.dt_matches.iter().any(|&m| m >= 0));

        // The sharp edge. Rank 0 is column `7 * 23 mod 24 == 17`, whose
        // best overlap is 0.78; column 0 — what a rank-indexed lookup
        // would read — tops out at 0.08 and is skippable at every rung.
        let dt_best = column_maxima(iou.view());
        assert_eq!(forced.dt_perm[0], 17);
        assert!(dt_best[17] >= 0.5, "rank 0's column must be matchable");
        assert!(dt_best[0] < 0.5, "column 0 must be skippable");
        assert!(
            forced.dt_matches[(0, 0)] >= 0,
            "the top-scoring DT must still match under the prefilter"
        );
    }
    /// Tiny deterministic xorshift; the test must not depend on a crate RNG.
    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// `cocoeval.py:258-296` transliterated, sharing **nothing** with the
    /// implementation: its own stable argsorts (`kind='mergesort'`),
    /// pycocotools' own encoding (1-based ids, `gtm`/`dtm` zero-filled,
    /// "taken" is `gtm > 0`), the literal `1 - 1e-10`, one loop over the
    /// sorted GTs with the B3 `break` inline. Returns `(dtm, gtm, dtIg)`
    /// with `dtm` holding matched **GT ids** and `gtm` matched **DT ids**.
    fn pycocotools_eval_img(
        iou: &Array2<f64>,
        gt_ignore: &[bool],
        gt_iscrowd: &[bool],
        dt_scores: &[f64],
        thresholds: &[f64],
    ) -> (Array2<i64>, Array2<i64>, Array2<bool>) {
        let (n_g, n_d, n_t) = (gt_ignore.len(), dt_scores.len(), thresholds.len());
        // gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        let mut gtind: Vec<usize> = (0..n_g).collect();
        gtind.sort_by_key(|&g| u8::from(gt_ignore[g]));
        // dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        let mut dtind: Vec<usize> = (0..n_d).collect();
        dtind.sort_by(|&a, &b| (-dt_scores[a]).total_cmp(&(-dt_scores[b])));
        let gt_id = |k: usize| gtind[k] as i64 + 1;
        let dt_id = |k: usize| dtind[k] as i64 + 1;
        let mut gtm = Array2::<i64>::zeros((n_t, n_g));
        let mut dtm = Array2::<i64>::zeros((n_t, n_d));
        let mut dt_ig = Array2::<bool>::default((n_t, n_d));
        for (tind, &t) in thresholds.iter().enumerate() {
            for dind in 0..n_d {
                let mut best = t.min(1.0 - 1e-10);
                let mut m: i64 = -1;
                for gind in 0..n_g {
                    if gtm[(tind, gind)] > 0 && !gt_iscrowd[gtind[gind]] {
                        continue;
                    }
                    if m > -1 && !gt_ignore[gtind[m as usize]] && gt_ignore[gtind[gind]] {
                        break;
                    }
                    let v = iou[(gtind[gind], dtind[dind])];
                    if v < best {
                        continue;
                    }
                    best = v;
                    m = gind as i64;
                }
                if m == -1 {
                    continue;
                }
                dt_ig[(tind, dind)] = gt_ignore[gtind[m as usize]];
                dtm[(tind, dind)] = gt_id(m as usize);
                gtm[(tind, m as usize)] = dt_id(dind);
            }
        }
        (dtm, gtm, dt_ig)
    }

    /// The ladder against [`pycocotools_eval_img`] over random cells:
    /// crowds, interleaved ignores, IoU ties (quantized to 0.05, so
    /// threshold equality and B2 ties occur), `NaN`s, tied and shuffled
    /// scores, both prefilter gates (gathered and direct column reads),
    /// and a tail of larger cells.
    #[test]
    fn ladder_matches_pycocotools_on_random_cells() {
        let coco: Vec<f64> = (0..10).map(|i| 0.5 + 0.05 * f64::from(i)).collect();
        let wide: Vec<f64> = (0..20).map(|i| 0.05 * f64::from(i)).collect();
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        let mut cells_with_matches = 0;
        for cell in 0..600 {
            let n_g = if cell % 10 == 0 {
                64 + rng.below(40) as usize
            } else {
                rng.below(24) as usize
            };
            let n_d = rng.below(40) as usize;
            let iou = Array2::from_shape_fn((n_g, n_d), |_| match rng.below(50) {
                0 => f64::NAN,
                v => (v as f64 * 0.05).min(1.0) * f64::from(u8::from(rng.below(3) != 0)),
            });
            let gt_ignore: Vec<bool> = (0..n_g).map(|_| rng.below(4) == 0).collect();
            let gt_iscrowd: Vec<bool> = (0..n_g).map(|_| rng.below(6) == 0).collect();
            let scores: Vec<f64> = (0..n_d).map(|_| (rng.below(8) as f64) / 8.0).collect();
            for thresholds in [&coco, &wide] {
                let (dtm, gtm, dt_ig) =
                    pycocotools_eval_img(&iou, &gt_ignore, &gt_iscrowd, &scores, thresholds);
                cells_with_matches += usize::from(dtm.iter().any(|&m| m > 0));
                for gate in [0, usize::MAX] {
                    let got = match_image_with_perm_gated(
                        iou.view(),
                        &gt_ignore,
                        &gt_iscrowd,
                        argsort_score_desc(&scores),
                        thresholds,
                        ParityMode::Strict,
                        gate,
                    )
                    .unwrap();
                    // Re-encode the implementation's sorted positions as
                    // pycocotools ids (0 = unmatched).
                    let as_gt_id = got.dt_matches.mapv(|m| {
                        if m < 0 {
                            0
                        } else {
                            got.gt_perm[m as usize] as i64 + 1
                        }
                    });
                    let as_dt_id = got.gt_matches.mapv(|k| {
                        if k < 0 {
                            0
                        } else {
                            got.dt_perm[k as usize] as i64 + 1
                        }
                    });
                    assert_eq!(as_gt_id, dtm, "dtMatches (cell {cell}, gate {gate})");
                    assert_eq!(as_dt_id, gtm, "gtMatches (cell {cell}, gate {gate})");
                    assert_eq!(got.dt_ignore, dt_ig, "dtIgnore (cell {cell}, gate {gate})");
                    assert!(got.gt_perm[got.n_unignored..].iter().all(|&g| gt_ignore[g]));
                    assert!(got.gt_perm[..got.n_unignored]
                        .iter()
                        .all(|&g| !gt_ignore[g]));
                }
            }
        }
        assert!(cells_with_matches > 300, "fixture must exercise matching");
    }

    /// The packed matchers are the `evaluate_with` hot path; the index
    /// matcher feeds `EvalImageMeta`. Both must say the same thing about
    /// every `(threshold, DT)` — with the unmatched-ignore overlay (B7 /
    /// AA3) applied on top of the index matcher's B6 flag, exactly as
    /// `evaluate_cell` does — on both sides of the prefilter gate, with
    /// one scratch reused across cells of every shape.
    #[test]
    fn packed_matchers_agree_with_the_index_matcher() {
        let coco: Vec<f64> = (0..10).map(|i| 0.5 + 0.05 * f64::from(i)).collect();
        let wide: Vec<f64> = (0..20).map(|i| 0.05 * f64::from(i)).collect();
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        let mut scratch = MatchScratch::default();
        let mut cells_with_matches = 0;
        for cell in 0..600 {
            let n_g = if cell % 10 == 0 {
                64 + rng.below(40) as usize
            } else {
                rng.below(24) as usize
            };
            let n_d = rng.below(40) as usize;
            let iou = Array2::from_shape_fn((n_g, n_d), |_| match rng.below(50) {
                0 => f64::NAN,
                v => (v as f64 * 0.05).min(1.0) * f64::from(u8::from(rng.below(3) != 0)),
            });
            let gt_ignore: Vec<bool> = (0..n_g).map(|_| rng.below(4) == 0).collect();
            let gt_iscrowd: Vec<bool> = (0..n_g).map(|_| rng.below(6) == 0).collect();
            let unmatched_ignore: Vec<bool> = (0..n_d).map(|_| rng.below(3) == 0).collect();

            for thresholds in [&coco, &wide] {
                let n_t = thresholds.len();
                let reference = match_image_with_perm_gated(
                    iou.view(),
                    &gt_ignore,
                    &gt_iscrowd,
                    (0..n_d).collect(),
                    thresholds,
                    ParityMode::Strict,
                    usize::MAX,
                )
                .unwrap();
                let expected = |t: usize, d: usize| {
                    let matched = reference.dt_matches[(t, d)] >= 0;
                    let ignore = if matched {
                        reference.dt_ignore[(t, d)]
                    } else {
                        unmatched_ignore[d]
                    };
                    (matched, ignore)
                };
                cells_with_matches += usize::from(reference.dt_matches.iter().any(|&m| m >= 0));

                for gate in [0, usize::MAX] {
                    scratch.prepare_iou_gated(iou.view(), gate);
                    if n_t <= 16 {
                        let mut flags = vec![0xDEAD_BEEF_u32];
                        let got = scratch
                            .match_presorted_bits16(
                                iou.view(),
                                &gt_ignore,
                                &gt_iscrowd,
                                &unmatched_ignore,
                                thresholds,
                                &mut flags,
                            )
                            .unwrap();
                        assert_eq!(got, reference.n_unignored);
                        assert_eq!(flags.len(), 1 + n_d, "appends exactly D words");
                        for d in 0..n_d {
                            for t in 0..n_t {
                                let word = flags[1 + d];
                                let got = (word >> t & 1 == 1, word >> (t + 16) & 1 == 1);
                                assert_eq!(got, expected(t, d), "bits16 (t={t}, d={d}, {gate})");
                            }
                        }
                    }
                    let (mut matched, mut ignore) = (vec![true], vec![true]);
                    let got = scratch
                        .match_presorted_general(
                            iou.view(),
                            &gt_ignore,
                            &gt_iscrowd,
                            &unmatched_ignore,
                            thresholds,
                            &mut matched,
                            &mut ignore,
                        )
                        .unwrap();
                    assert_eq!(got, reference.n_unignored);
                    assert_eq!(matched.len(), 1 + n_d * n_t);
                    for d in 0..n_d {
                        for t in 0..n_t {
                            let flat = 1 + d * n_t + t;
                            assert_eq!(
                                (matched[flat], ignore[flat]),
                                expected(t, d),
                                "general (t={t}, d={d}, gate {gate})"
                            );
                        }
                    }
                }
            }
        }
        assert!(cells_with_matches > 300, "fixture must exercise matching");
    }

    /// The column-maxima fold and the per-DT column gather on a strided
    /// (transposed) view — the non-contiguous fallback the
    /// standard-layout fast path skips — agree with a contiguous copy.
    #[test]
    fn strided_iou_view_matches_its_contiguous_copy() {
        let (n_g, n_d) = (9, 300);
        let mut rng = XorShift(0x2545_F491_4F6C_DD1D);
        // Built (D, G) and viewed transposed, so `as_slice` is `None`.
        let dg = Array2::from_shape_fn((n_d, n_g), |_| match rng.below(40) {
            0 => f64::NAN,
            v => (v as f64 / 40.0) * f64::from(u8::from(rng.below(2) == 0)),
        });
        let strided = dg.t();
        assert!(strided.as_slice().is_none(), "view must be non-contiguous");
        let contiguous = strided.to_owned();
        assert_eq!(
            column_maxima(strided)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            column_maxima(contiguous.view())
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let gt_ignore: Vec<bool> = (0..n_g).map(|g| g % 4 == 1).collect();
        let gt_iscrowd: Vec<bool> = (0..n_g).map(|g| g == 6).collect();
        let scores: Vec<f64> = (0..n_d).map(|d| ((d * 37) % 11) as f64).collect();
        let thresholds: Vec<f64> = (0..10).map(|i| 0.5 + 0.05 * f64::from(i)).collect();
        let run = |m: ArrayView2<'_, f64>| {
            match_image_with_perm_gated(
                m,
                &gt_ignore,
                &gt_iscrowd,
                argsort_score_desc(&scores),
                &thresholds,
                ParityMode::Strict,
                0,
            )
            .unwrap()
        };
        let (a, b) = (run(strided), run(contiguous.view()));
        assert_eq!(a.dt_matches, b.dt_matches);
        assert_eq!(a.gt_matches, b.gt_matches);
        assert_eq!(a.dt_ignore, b.dt_ignore);
        assert!(a.dt_matches.iter().any(|&m| m >= 0));
    }
}
