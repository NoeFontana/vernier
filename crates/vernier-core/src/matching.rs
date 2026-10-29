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

/// Where the greedy ladder's matches land, and how a DT's score rank
/// maps to its IoU-matrix column.
///
/// The ladder itself — quirks **B1**-**B4** and **B6** — exists exactly
/// once, in [`run_ladder`]; the three output shapes (the `MatchResult`
/// index matrices behind `EvalImageMeta`, and the two packed accumulator
/// layouts) are sinks it is monomorphized over. A parity fix to the scan
/// therefore lands on every path at once.
trait LadderSink {
    /// IoU-matrix column of the `rank`-th score-sorted DT.
    fn column(&self, rank: usize) -> usize;
    /// DT `rank` matched sorted GT `m` at threshold `tind`. `to_ignore`
    /// is quirk **B6**: whether `m` is an ignore-GT.
    fn record(&mut self, tind: usize, rank: usize, m: usize, to_ignore: bool);
}

/// `MatchResult` index matrices; columns reached through `dt_perm`.
struct IndexSink<'a> {
    dt_perm: &'a [usize],
    dt_matches: &'a mut Array2<i64>,
    gt_matches: &'a mut Array2<i64>,
    dt_ignore: &'a mut Array2<bool>,
}

impl LadderSink for IndexSink<'_> {
    #[inline(always)]
    fn column(&self, rank: usize) -> usize {
        self.dt_perm[rank]
    }

    #[inline(always)]
    fn record(&mut self, tind: usize, rank: usize, m: usize, to_ignore: bool) {
        self.dt_ignore[(tind, rank)] = to_ignore;
        self.dt_matches[(tind, rank)] = m as i64;
        self.gt_matches[(tind, m)] = rank as i64;
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

/// The `(threshold, DT)` matching ladder — the one implementation of the
/// pycocotools greedy loop (`cocoeval.py:268-296`).
///
/// GTs arrive A4-partitioned (`gt_perm[..n_unignored]` non-ignore, the
/// rest ignore, each block in input order), which turns **B3** into
/// structure rather than a per-GT test: a DT scans the non-ignore block,
/// and only if nothing there matched does it go on into the ignore tail.
/// That is exactly pycocotools' "break at the first ignore-GT once a real
/// match exists" — the tail is only ever reached with `m == -1`.
///
/// Monomorphized on `PREFILTER`: with it off, `dt_best` is unused and
/// the skip test folds away. With it on, `dt_best[col]` is the DT's best
/// overlap with any GT — below the threshold seed the DT cannot match
/// (`best` only rises from the seed, so every `iou < best` test would
/// `continue`), so its whole `G`-long scan is dead work. Skipping it is
/// output-identical, not an approximation.
// `ArrayView2` is a `Copy` view, so by-value is idiomatic.
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
#[inline(always)]
fn run_ladder<const PREFILTER: bool, S: LadderSink>(
    iou_matrix: ArrayView2<'_, f64>,
    iou_thresholds: &[f64],
    n_d: usize,
    n_unignored: usize,
    gt_perm: &[usize],
    gt_iscrowd_sorted: &[bool],
    gt_matched: &mut [bool],
    dt_best: &[f64],
    sink: &mut S,
) {
    let (unignored, ignored) = gt_perm.split_at(n_unignored);
    for (tind, &t) in iou_thresholds.iter().enumerate() {
        gt_matched.fill(false);
        // B1: seed best at `min(t, 1 - 1e-10)` so a DT whose best
        // overlap exactly equals the threshold still matches.
        let seed = t.min(1.0 - IOU_BOUNDARY_EPS);
        for rank in 0..n_d {
            let col = sink.column(rank);
            if PREFILTER && dt_best[col] < seed {
                continue;
            }
            let mut best = seed;
            let mut m = usize::MAX;
            for (k_g, &g_orig) in unignored.iter().enumerate() {
                // B4: an already-taken GT is out of contention; crowds
                // never mark themselves taken (many-to-one).
                if gt_matched[k_g] {
                    continue;
                }
                let iou = iou_matrix[(g_orig, col)];
                // B2: non-strict comparison — on equality the later GT
                // wins, since the update runs when `iou >= best`.
                if iou < best {
                    continue;
                }
                best = iou;
                m = k_g;
            }
            // B3: the ignore tail is reached only without a real match.
            if m == usize::MAX {
                for (k_t, &g_orig) in ignored.iter().enumerate() {
                    let k_g = n_unignored + k_t;
                    if gt_matched[k_g] {
                        continue;
                    }
                    let iou = iou_matrix[(g_orig, col)];
                    if iou < best {
                        continue;
                    }
                    best = iou;
                    m = k_g;
                }
            }
            if m == usize::MAX {
                continue;
            }
            if !gt_iscrowd_sorted[m] {
                gt_matched[m] = true;
            }
            // B6: a DT matched to an ignore-GT inherits the ignore flag.
            sink.record(tind, rank, m, m >= n_unignored);
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
    // one flat slice: plain `&[f64]` rows let the fold vectorize, which
    // ndarray's generic row iterators do not.
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

/// Shape checks every matcher entry shares: `iou_matrix` is `(G, D)`
/// and the crowd flags cover every GT.
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
                "iou_matrix is {}x{}, expected {}x{}",
                iou_matrix.nrows(),
                iou_matrix.ncols(),
                n_g,
                n_d
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

    check_cell_dims(iou_matrix, n_g, n_d, gt_iscrowd.len())?;

    // A4 (strict): stable partition of GTs by `_ignore`, so non-ignore
    // precede ignore and each block keeps its input order — the same
    // permutation a stable sort by key produces.
    let mut scratch = MatchScratch::default();
    let n_unignored = scratch.partition_gts(gt_ignore, gt_iscrowd);

    let mut dt_matches = Array2::<i64>::from_elem((n_t, n_d), -1);
    let mut gt_matches = Array2::<i64>::from_elem((n_t, n_g), -1);
    let mut dt_ignore = Array2::<bool>::default((n_t, n_d));

    if n_g != 0 && n_d != 0 && n_t != 0 {
        let prefilter = n_g * n_d >= prefilter_min_cell;
        if prefilter {
            column_maxima_into(&mut scratch.dt_best, iou_matrix);
        }
        let mut sink = IndexSink {
            dt_perm: &dt_perm,
            dt_matches: &mut dt_matches,
            gt_matches: &mut gt_matches,
            dt_ignore: &mut dt_ignore,
        };
        scratch.ladder(
            iou_matrix,
            iou_thresholds,
            n_d,
            n_unignored,
            prefilter,
            &mut sink,
        );
    }

    Ok(MatchResult {
        dt_perm,
        gt_perm: scratch.gt_perm,
        dt_matches,
        gt_matches,
        dt_ignore,
    })
}

/// Reusable per-worker scratch for allocation-free matching into
/// [`crate::accumulate::PackedCategory`] (ADR-0066).
///
/// Used on the default `!builds_meta` hot path where detections are
/// already pre-sorted descending by score (`dt_perm` is identity) and
/// matched annotation IDs (`EvalImageMeta`) are not requested. The
/// matching itself is [`run_ladder`], shared with [`match_image`].
#[derive(Default)]
pub(crate) struct MatchScratch {
    gt_perm: Vec<usize>,
    gt_iscrowd_sorted: Vec<bool>,
    gt_matched: Vec<bool>,
    dt_best: Vec<f64>,
}

impl MatchScratch {
    /// Precomputes column maxima on `iou_matrix` once per `(category,
    /// image)` cell when `G * D >= PREFILTER_MIN_CELL`, reusing the
    /// result across all area ranges. Must run before the cell's
    /// `match_presorted_*` calls.
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn prepare_iou(&mut self, iou_matrix: ArrayView2<'_, f64>) {
        if iou_matrix.nrows() * iou_matrix.ncols() >= PREFILTER_MIN_CELL {
            column_maxima_into(&mut self.dt_best, iou_matrix);
        } else {
            self.dt_best.clear();
        }
    }

    /// A4 stable partition into `gt_perm` (non-ignore block first) with
    /// the crowd flags permuted alongside. Returns the non-ignore count.
    fn partition_gts(&mut self, gt_ignore: &[bool], gt_iscrowd: &[bool]) -> usize {
        let n_g = gt_ignore.len();
        self.gt_perm.clear();
        self.gt_perm.reserve(n_g);
        self.gt_iscrowd_sorted.clear();
        self.gt_iscrowd_sorted.reserve(n_g);
        for (i, (&ig, &crowd)) in gt_ignore.iter().zip(gt_iscrowd).enumerate() {
            if !ig {
                self.gt_perm.push(i);
                self.gt_iscrowd_sorted.push(crowd);
            }
        }
        let n_unignored = self.gt_perm.len();
        for (i, (&ig, &crowd)) in gt_ignore.iter().zip(gt_iscrowd).enumerate() {
            if ig {
                self.gt_perm.push(i);
                self.gt_iscrowd_sorted.push(crowd);
            }
        }
        // Contents are irrelevant: the ladder clears it per threshold.
        self.gt_matched.resize(n_g, false);
        n_unignored
    }

    /// Dispatch [`run_ladder`] on the prefilter flag.
    #[allow(clippy::needless_pass_by_value)]
    #[inline(always)]
    fn ladder<S: LadderSink>(
        &mut self,
        iou_matrix: ArrayView2<'_, f64>,
        iou_thresholds: &[f64],
        n_d: usize,
        n_unignored: usize,
        prefilter: bool,
        sink: &mut S,
    ) {
        if prefilter {
            debug_assert_eq!(self.dt_best.len(), iou_matrix.ncols());
            run_ladder::<true, S>(
                iou_matrix,
                iou_thresholds,
                n_d,
                n_unignored,
                &self.gt_perm,
                &self.gt_iscrowd_sorted,
                &mut self.gt_matched,
                &self.dt_best,
                sink,
            );
        } else {
            run_ladder::<false, S>(
                iou_matrix,
                iou_thresholds,
                n_d,
                n_unignored,
                &self.gt_perm,
                &self.gt_iscrowd_sorted,
                &mut self.gt_matched,
                &[],
                sink,
            );
        }
    }

    /// Runs greedy matching on pre-sorted detections (`n_t <= 16`) and
    /// appends `n_d` bit-packed `(matched_u16 | (ignore_u16 << 16))`
    /// words to `out_flags`. `dt_unmatched_ignore[d]` is the ignore flag
    /// an *unmatched* DT carries (B7 / AA3). Returns `n_unignored`.
    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    pub(crate) fn match_presorted_bits16(
        &mut self,
        iou_matrix: ArrayView2<'_, f64>,
        gt_ignore: &[bool],
        gt_iscrowd: &[bool],
        dt_unmatched_ignore: &[bool],
        iou_thresholds: &[f64],
        _parity_mode: ParityMode,
        out_flags: &mut Vec<u32>,
    ) -> Result<usize, EvalError> {
        let n_g = gt_ignore.len();
        let n_d = dt_unmatched_ignore.len();
        let n_t = iou_thresholds.len();
        debug_assert!(n_t <= 16);

        #[cfg(feature = "bench-histogram")]
        let _hist_guard = histogram::CallTimer::new(n_g, n_d);

        check_cell_dims(iou_matrix, n_g, n_d, gt_iscrowd.len())?;
        let n_unignored = self.partition_gts(gt_ignore, gt_iscrowd);
        let all_t_ig: u32 = if n_t == 0 {
            0
        } else {
            ((1_u32 << n_t) - 1) << 16
        };
        let base = out_flags.len();
        out_flags.extend(
            dt_unmatched_ignore
                .iter()
                .map(|&ig| if ig { all_t_ig } else { 0 }),
        );
        if n_g == 0 || n_d == 0 || n_t == 0 {
            return Ok(n_unignored);
        }
        let mut sink = Bits16Sink {
            flags: &mut out_flags[base..],
        };
        let prefilter = n_g * n_d >= PREFILTER_MIN_CELL;
        self.ladder(
            iou_matrix,
            iou_thresholds,
            n_d,
            n_unignored,
            prefilter,
            &mut sink,
        );
        Ok(n_unignored)
    }

    /// Arbitrary-`n_t` fallback (`n_t > 16`) for pre-sorted detections.
    /// Appends `n_d * n_t` booleans (layout `[d * n_t + tind]`) to
    /// `out_matched` and `out_ignore`. Returns `n_unignored`.
    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    pub(crate) fn match_presorted_general(
        &mut self,
        iou_matrix: ArrayView2<'_, f64>,
        gt_ignore: &[bool],
        gt_iscrowd: &[bool],
        dt_unmatched_ignore: &[bool],
        iou_thresholds: &[f64],
        _parity_mode: ParityMode,
        out_matched: &mut Vec<bool>,
        out_ignore: &mut Vec<bool>,
    ) -> Result<usize, EvalError> {
        let n_g = gt_ignore.len();
        let n_d = dt_unmatched_ignore.len();
        let n_t = iou_thresholds.len();

        #[cfg(feature = "bench-histogram")]
        let _hist_guard = histogram::CallTimer::new(n_g, n_d);

        check_cell_dims(iou_matrix, n_g, n_d, gt_iscrowd.len())?;
        let n_unignored = self.partition_gts(gt_ignore, gt_iscrowd);
        let base = out_matched.len();
        out_matched.resize(base + n_d * n_t, false);
        out_ignore.reserve(n_d * n_t);
        for &ig in dt_unmatched_ignore {
            out_ignore.extend(std::iter::repeat_n(ig, n_t));
        }
        if n_g == 0 || n_d == 0 || n_t == 0 {
            return Ok(n_unignored);
        }
        let mut sink = GeneralSink {
            n_t,
            matched: &mut out_matched[base..],
            ignore: &mut out_ignore[base..],
        };
        let prefilter = n_g * n_d >= PREFILTER_MIN_CELL;
        self.ladder(
            iou_matrix,
            iou_thresholds,
            n_d,
            n_unignored,
            prefilter,
            &mut sink,
        );
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

    /// Literal transliteration of `cocoeval.py:268-296` — one loop over
    /// the A4-sorted GTs with the B3 `break` inline — kept independent of
    /// [`run_ladder`]'s split-block form so the two can check each other.
    fn pycocotools_ladder(
        iou: &Array2<f64>,
        gt_ignore: &[bool],
        gt_iscrowd: &[bool],
        thresholds: &[f64],
    ) -> (Array2<i64>, Array2<i64>, Array2<bool>) {
        let (n_g, n_d, n_t) = (gt_ignore.len(), iou.ncols(), thresholds.len());
        let mut gt_perm: Vec<usize> = (0..n_g).collect();
        gt_perm.sort_by_key(|&g| gt_ignore[g]);
        let mut dtm = Array2::<i64>::from_elem((n_t, n_d), -1);
        let mut gtm = Array2::<i64>::from_elem((n_t, n_g), -1);
        let mut dt_ig = Array2::<bool>::default((n_t, n_d));
        for (t, &thr) in thresholds.iter().enumerate() {
            for d in 0..n_d {
                let mut best = thr.min(1.0 - IOU_BOUNDARY_EPS);
                let mut m: i64 = -1;
                for (k, &g) in gt_perm.iter().enumerate() {
                    if gtm[(t, k)] >= 0 && !gt_iscrowd[g] {
                        continue;
                    }
                    if m >= 0 && !gt_ignore[gt_perm[m as usize]] && gt_ignore[g] {
                        break;
                    }
                    if iou[(g, d)] < best {
                        continue;
                    }
                    best = iou[(g, d)];
                    m = k as i64;
                }
                if m >= 0 {
                    dt_ig[(t, d)] = gt_ignore[gt_perm[m as usize]];
                    dtm[(t, d)] = m;
                    gtm[(t, m as usize)] = d as i64;
                }
            }
        }
        (dtm, gtm, dt_ig)
    }

    /// The packed matchers are the `evaluate_with` hot path; the index
    /// matcher feeds `EvalImageMeta`. Both must say the same thing about
    /// every `(threshold, DT)` — with the unmatched-ignore overlay (B7 /
    /// AA3) applied on top of the index matcher's B6 flag, exactly as
    /// `evaluate_cell` does. Random cells cover crowds, interleaved
    /// ignores, IoU ties (quantized to 0.05, so threshold equality and
    /// B2 ties occur), `NaN`s, and both sides of the prefilter gate.
    #[test]
    fn packed_matchers_agree_with_the_index_matcher() {
        let coco: Vec<f64> = (0..10).map(|i| 0.5 + 0.05 * f64::from(i)).collect();
        let wide: Vec<f64> = (0..20).map(|i| 0.05 * f64::from(i)).collect();
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        let mut scratch = MatchScratch::default();
        let mut cells_with_matches = 0;
        for _ in 0..600 {
            let n_g = rng.below(24) as usize;
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
                let reference = match_image_with_perm(
                    iou.view(),
                    &gt_ignore,
                    &gt_iscrowd,
                    (0..n_d).collect(),
                    thresholds,
                    ParityMode::Strict,
                )
                .unwrap();
                let (dtm, gtm, dt_ig) =
                    pycocotools_ladder(&iou, &gt_ignore, &gt_iscrowd, thresholds);
                assert_eq!(reference.dt_matches, dtm, "dtMatches vs pycocotools loop");
                assert_eq!(reference.gt_matches, gtm, "gtMatches vs pycocotools loop");
                assert_eq!(reference.dt_ignore, dt_ig, "dtIgnore vs pycocotools loop");
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
                let n_unignored = gt_ignore.iter().filter(|&&ig| !ig).count();

                scratch.prepare_iou(iou.view());
                if n_t <= 16 {
                    let mut flags = vec![0xDEAD_BEEF_u32];
                    let got = scratch
                        .match_presorted_bits16(
                            iou.view(),
                            &gt_ignore,
                            &gt_iscrowd,
                            &unmatched_ignore,
                            thresholds,
                            ParityMode::Strict,
                            &mut flags,
                        )
                        .unwrap();
                    assert_eq!(got, n_unignored);
                    assert_eq!(flags.len(), 1 + n_d, "appends exactly D words");
                    for d in 0..n_d {
                        for t in 0..n_t {
                            let word = flags[1 + d];
                            let got = (word >> t & 1 == 1, word >> (t + 16) & 1 == 1);
                            assert_eq!(got, expected(t, d), "bits16 (t={t}, d={d})");
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
                        ParityMode::Strict,
                        &mut matched,
                        &mut ignore,
                    )
                    .unwrap();
                assert_eq!(got, n_unignored);
                assert_eq!(matched.len(), 1 + n_d * n_t);
                for d in 0..n_d {
                    for t in 0..n_t {
                        let flat = 1 + d * n_t + t;
                        assert_eq!(
                            (matched[flat], ignore[flat]),
                            expected(t, d),
                            "general (t={t}, d={d})"
                        );
                    }
                }
            }
        }
        assert!(cells_with_matches > 300, "fixture must exercise matching");
    }
}
