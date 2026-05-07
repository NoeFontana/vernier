//! Streaming evaluator (ADR-0013).
//!
//! Composes around the locked spine: each `update()` runs `match_image`
//! over the new detections (via the existing `evaluate_with` path) and
//! appends one [`PerImageEval`] per `(category, area, image)` cell to an
//! internal sparse store. [`StreamingEvaluator::snapshot`] and
//! [`StreamingEvaluator::finalize`] call [`crate::accumulate`] over the
//! current store unchanged.
//!
//! Per ADR-0005 this module does NOT edit `matching.rs` or
//! `accumulate.rs`. It only orchestrates.
//!
//! ## v0 deferrals
//!
//! - `checkpoint`/`restore` return [`EvalError::NotImplemented`] (scope
//!   decision; a future ADR adds rkyv-based serialization).
//! - `snapshot_running` delegates to `snapshot()` (TODO: running-mode
//!   PR-curve approximation per ADR-0013 §"Fast snapshot mode").
//! - Strict-mode `(score, stream_position)` tiebreak is not yet wired
//!   through the matching path; `next_dt_id` carries the monotonic
//!   counter for the future implementation.

use std::collections::{HashMap, HashSet};
use std::mem::size_of;

use crate::accumulate::{accumulate, AccumulateParams, PerImageEval};
use crate::dataset::{
    AnnId, CategoryId, CocoDataset, CocoDetection, CocoDetections, DetectionInput, EvalDataset,
    ImageId,
};
use crate::error::EvalError;
use crate::evaluate::{evaluate_with, EvalImageMeta, EvalKernel, OwnedEvaluateParams};
use crate::parity::{recall_thresholds, ParityMode};
use crate::summarize::{summarize_detection, summarize_with, StatRequest, Summary};

/// Default fallback when [`MemoryBudget::system_total_bytes`] returns
/// `None`. 8 GiB.
const DEFAULT_BUDGET_BYTES: usize = 8 * 1024 * 1024 * 1024;

/// Soft-warn fraction of the budget. Once `total_used >= soft_warn *
/// budget`, the next [`UpdateReport`] sets `soft_warn_triggered = true`
/// (one-shot per evaluator).
const DEFAULT_SOFT_WARN_FRACTION: f64 = 0.80;

/// Memory budget for a [`StreamingEvaluator`].
///
/// The streaming evaluator holds one [`PerImageEval`] per
/// `(category, area, image)` cell that has received any detection. This
/// can grow large for big datasets with many categories — the budget is
/// the cap that bounds it.
#[derive(Debug, Clone, Copy)]
pub struct MemoryBudget {
    /// Hard cap, in bytes. `update()` returns
    /// [`EvalError::OutOfBudget`] before inserting any cell that would
    /// take the total past this number.
    pub bytes: usize,
    /// Soft-warn threshold as a fraction of `bytes` (typically `0.80`).
    /// Once crossed, the next [`UpdateReport`] flags
    /// `soft_warn_triggered`. Set to `> 1.0` to disable.
    pub soft_warn_fraction: f64,
}

impl MemoryBudget {
    /// `min(8 GiB, system_total / 2)`. The system-total query is
    /// best-effort; a read failure falls back to 8 GiB.
    pub fn auto_default() -> Self {
        let half_total = Self::system_total_bytes()
            .map(|t| t / 2)
            .unwrap_or(DEFAULT_BUDGET_BYTES);
        Self {
            bytes: DEFAULT_BUDGET_BYTES.min(half_total),
            soft_warn_fraction: DEFAULT_SOFT_WARN_FRACTION,
        }
    }

    /// Read-the-system-once helper. Reads `/proc/meminfo` on Linux,
    /// returns `None` on other platforms or on read/parse failure.
    fn system_total_bytes() -> Option<usize> {
        if cfg!(target_os = "linux") {
            let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
            for line in contents.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    let rest = rest.trim();
                    let kb_part = rest.strip_suffix(" kB")?;
                    let kb: usize = kb_part.trim().parse().ok()?;
                    return Some(kb.saturating_mul(1024));
                }
            }
            None
        } else {
            None
        }
    }
}

/// Static metadata describing the `(K, A, I)` evaluation grid the
/// streaming evaluator targets.
///
/// Built once at construction from the [`CocoDataset`] and the
/// [`OwnedEvaluateParams`] — never mutated thereafter. Mirrors the axes
/// the unchanged batch [`crate::evaluate_with`] orchestrator emits.
#[derive(Debug, Clone)]
pub struct EvalGridMeta {
    /// `K` axis size (number of categories, or `1` when `use_cats=false`).
    pub n_categories: usize,
    /// `A` axis size (number of area ranges).
    pub n_area_ranges: usize,
    /// `I` axis size (number of images in the GT dataset).
    pub n_images: usize,
    /// Maps each GT [`CategoryId`] to its position on the K-axis. Empty
    /// when `use_cats=false`.
    pub category_id_to_idx: HashMap<CategoryId, usize>,
    /// Maps each GT [`ImageId`] to its position on the I-axis.
    pub image_id_to_idx: HashMap<ImageId, usize>,
}

/// Sparse `(k, a, i) -> PerImageEval` store backing the streaming
/// evaluator.
///
/// Cells absent from the store represent the same "no detections, no
/// non-ignore GTs" condition as `None` entries in the batch
/// [`crate::EvalGrid::eval_imgs`]. [`Self::flatten`] re-densifies the
/// store into the dense `Vec<Option<PerImageEval>>` shape
/// [`crate::accumulate`] consumes.
#[derive(Debug, Clone, Default)]
pub struct PerImageEvalStore {
    /// Sparse cells keyed by `(k, a, i)`.
    cells: HashMap<(usize, usize, usize), PerImageEval>,
}

impl PerImageEvalStore {
    /// Empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of populated cells.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// `true` if no cells have been inserted.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Insert (or overwrite) the cell at `(k, a, i)`.
    pub fn insert(&mut self, k: usize, a: usize, i: usize, cell: PerImageEval) {
        self.cells.insert((k, a, i), cell);
    }

    /// Borrow the underlying `(k, a, i) -> PerImageEval` map. Used by
    /// the distributed-eval encoder (ADR-0031) to walk cells in
    /// canonical order.
    pub(crate) fn as_map(&self) -> &HashMap<(usize, usize, usize), PerImageEval> {
        &self.cells
    }

    /// Move-out variant of [`Self::as_map`] used by the
    /// `from_partials` constructor to swap a freshly merged store
    /// in.
    pub(crate) fn from_map(cells: HashMap<(usize, usize, usize), PerImageEval>) -> Self {
        Self { cells }
    }

    /// Densify into the `[k * A * I + a * I + i]`-laid-out
    /// `Vec<Option<Box<PerImageEval>>>` that [`crate::accumulate`]
    /// consumes. Cloning is intentional — `accumulate` borrows the
    /// slice and the store must remain valid for further updates after
    /// a snapshot. The cells are heap-boxed so the dense slot Vec only
    /// pays for a pointer per slot at zero-init time (see
    /// [`crate::EvalGrid::eval_imgs`]).
    pub fn flatten(&self, meta: &EvalGridMeta) -> Vec<Option<Box<PerImageEval>>> {
        let total = meta.n_categories * meta.n_area_ranges * meta.n_images;
        let mut out: Vec<Option<Box<PerImageEval>>> = Vec::with_capacity(total);
        for k in 0..meta.n_categories {
            for a in 0..meta.n_area_ranges {
                for i in 0..meta.n_images {
                    out.push(self.cells.get(&(k, a, i)).cloned().map(Box::new));
                }
            }
        }
        out
    }
}

/// Diagnostics returned from each [`StreamingEvaluator::update`] call.
///
/// Useful for training-loop logging (TensorBoard, console). All counters
/// describe the *batch* that produced this report, not the cumulative
/// totals (see [`StreamingEvaluator::detections_seen`] etc. for those).
#[derive(Debug, Clone)]
pub struct UpdateReport {
    /// Number of detections accepted in this batch (after duplicate
    /// rejection).
    pub n_detections_accepted: usize,
    /// Number of distinct images that received detections in this batch.
    pub n_images_in_batch: usize,
    /// Number of populated cells produced and inserted.
    pub n_cells_inserted: usize,
    /// `true` exactly on the *first* report whose post-insert total
    /// crosses the soft-warn threshold (one-shot per evaluator).
    pub soft_warn_triggered: bool,
}

/// Pre-parsed detection batch.
///
/// The streaming evaluator's normal entry point is [`StreamingEvaluator::update`]
/// (raw JSON bytes); this struct names the parsed-but-not-yet-evaluated
/// shape for callers that want to amortize the parse cost or supply
/// detections from a non-JSON source. Generic over [`EvalKernel`] so the
/// parsed form is type-tied to the evaluator's kernel choice — feeding a
/// `ParsedDetections<BboxIou>` into a `StreamingEvaluator<OksSimilarity>`
/// is a compile-time error.
#[derive(Debug, Clone)]
pub struct ParsedDetections<K: EvalKernel> {
    /// Parsed detections.
    pub detections: CocoDetections,
    /// Type marker tying these detections to a specific kernel.
    _kernel: std::marker::PhantomData<K>,
}

impl<K: EvalKernel> ParsedDetections<K> {
    /// Wrap a pre-built [`CocoDetections`] for use with a streaming
    /// evaluator of kernel `K`.
    pub fn from_detections(detections: CocoDetections) -> Self {
        Self {
            detections,
            _kernel: std::marker::PhantomData,
        }
    }

    /// Parse detections from the `loadRes`-shaped JSON byte slice.
    ///
    /// # Errors
    ///
    /// Propagates [`EvalError::Json`] / [`EvalError::NonFinite`] from
    /// the underlying [`CocoDetections::from_json_bytes`].
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, EvalError> {
        Ok(Self::from_detections(CocoDetections::from_json_bytes(
            bytes,
        )?))
    }
}

/// Streaming COCO evaluator, ADR-0013.
///
/// Holds a [`CocoDataset`] plus a sparse store of [`PerImageEval`] cells
/// produced by per-batch `match_image` calls. [`Self::snapshot`] runs
/// [`accumulate`] + summarize over the current store at any time;
/// [`Self::finalize`] consumes the evaluator and returns the same
/// [`Summary`]. Bit-identical to a batch run over the union of all
/// `update()` batches submitted in order.
///
/// When `params.retain_iou` is `true`, the evaluator additionally
/// retains a [`crate::RetainedIous`] store keyed by `(k, i)`, populated
/// incrementally as each batch's `evaluate_with` returns its per-batch
/// retentions. Consumed by the per_pair / per_detection table builders.
#[derive(Debug)]
pub struct StreamingEvaluator<K: EvalKernel> {
    dataset: CocoDataset,
    kernel: K,
    params: OwnedEvaluateParams,
    parity_mode: ParityMode,
    grid_meta: EvalGridMeta,
    cells: PerImageEvalStore,
    /// Per-`(k, a, i)` `EvalImageMeta` retained alongside `cells` when
    /// `params.retain_iou` is true. Empty otherwise. Required by the
    /// per_detection / per_pair builders for `dt_ids` / `gt_ids` / etc.
    meta_cells: HashMap<(usize, usize, usize), EvalImageMeta>,
    /// Per-`(category, image)` IoU matrices retained across `update()`
    /// calls. `None` on the default path (`params.retain_iou=false`);
    /// `Some` when retention was opted into at construction.
    retained_ious: Option<crate::tables::RetainedIous>,
    /// Detection records accumulated across `update()` calls, kept only
    /// when `params.retain_iou` is true (consumed by `per_detection`'s
    /// `area` / optional `bbox` columns). Records are owned and carry
    /// their original ids; flushed into a fresh `CocoDetections` view
    /// at finalize/snapshot time.
    dets_seen: Vec<CocoDetection>,
    seen_images: HashSet<i64>,
    /// Image-grid indices for every entry in `seen_images`. Maintained
    /// incrementally so `compute_summary` can decide which GT-only
    /// cells to overlay without re-walking `seen_images` every call.
    seen_image_indices: HashSet<usize>,
    /// Lazily-built GT-only `(K, A, I)` grid used by `compute_summary`
    /// to file cells for images that have not yet received any
    /// detection. GT is immutable, so the grid is computed at most
    /// once across the evaluator's lifetime.
    gt_only_cells: Option<Vec<Option<Box<PerImageEval>>>>,
    n_detections: usize,
    /// Monotonic DT-id counter. Reserved for the strict-mode
    /// `(score, stream_position)` tiebreak (ADR-0013 §Determinism); not
    /// yet consumed by the matching path.
    next_dt_id: i64,
    /// Optional rank identifier for distributed-eval merge (ADR-0031).
    /// `None` for single-rank usage. Set via [`Self::with_rank`] before
    /// the first `update`. Carried in the partial header and used as
    /// the strict-mode `(rank_id, local_position)` tiebreak when the
    /// matching path consumes it.
    rank_id: Option<crate::distributed::RankId>,
    bytes_cells_struct: usize,
    bytes_dt_scores: usize,
    bytes_match_flags: usize,
    budget: MemoryBudget,
    soft_warn_fired: bool,
}

impl<K: EvalKernel> StreamingEvaluator<K> {
    /// Construct a new streaming evaluator.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError::InvalidConfig`] if `params.area_ranges` is
    /// empty (the batch path tolerates this; the streaming evaluator
    /// rejects it because the `(K, A, I)` grid would be degenerate).
    pub fn new(
        dataset: CocoDataset,
        kernel: K,
        params: OwnedEvaluateParams,
        parity_mode: ParityMode,
        budget: MemoryBudget,
    ) -> Result<Self, EvalError> {
        if params.area_ranges.is_empty() {
            return Err(EvalError::InvalidConfig {
                detail: "OwnedEvaluateParams.area_ranges must be non-empty".into(),
            });
        }
        let grid_meta = build_grid_meta(&dataset, &params);
        let retained_ious = if params.retain_iou {
            Some(crate::tables::RetainedIous::new())
        } else {
            None
        };
        Ok(Self {
            dataset,
            kernel,
            params,
            parity_mode,
            grid_meta,
            cells: PerImageEvalStore::new(),
            meta_cells: HashMap::new(),
            retained_ious,
            dets_seen: Vec::new(),
            seen_images: HashSet::new(),
            seen_image_indices: HashSet::new(),
            gt_only_cells: None,
            n_detections: 0,
            next_dt_id: 1,
            rank_id: None,
            bytes_cells_struct: 0,
            bytes_dt_scores: 0,
            bytes_match_flags: 0,
            budget,
            soft_warn_fired: false,
        })
    }

    /// Set this evaluator's rank identifier for distributed-eval merge
    /// (ADR-0031). Builder shape: returns `Self`. Calling this after
    /// the first [`Self::update`] is a programming error and returns
    /// [`EvalError::InvalidConfig`] — rank identity is a
    /// construction-time property, not a mid-run mutable parameter.
    ///
    /// # Errors
    ///
    /// [`EvalError::InvalidConfig`] when `n_detections > 0`.
    pub fn with_rank(mut self, rank_id: crate::distributed::RankId) -> Result<Self, EvalError> {
        if self.n_detections > 0 {
            return Err(EvalError::InvalidConfig {
                detail: "with_rank must be called before any update; rank identity is fixed at construction".into(),
            });
        }
        self.rank_id = Some(rank_id);
        Ok(self)
    }

    /// The rank id this evaluator was tagged with, if any.
    pub fn rank_id(&self) -> Option<crate::distributed::RankId> {
        self.rank_id
    }

    /// Number of distinct images with at least one accepted detection.
    pub fn images_seen(&self) -> usize {
        self.seen_images.len()
    }

    /// Number of detections accepted across all `update()` calls.
    pub fn detections_seen(&self) -> usize {
        self.n_detections
    }

    /// Number of GT images that have not yet received any detection.
    pub fn images_pending(&self) -> usize {
        self.grid_meta.n_images.saturating_sub(self.images_seen())
    }

    /// Total bytes the evaluator currently holds (sum of the three
    /// breakdown components).
    pub fn memory_used_bytes(&self) -> usize {
        self.bytes_cells_struct + self.bytes_dt_scores + self.bytes_match_flags
    }

    /// View of the configured budget.
    pub fn budget(&self) -> MemoryBudget {
        self.budget
    }

    /// Read-only access to the static grid metadata.
    pub fn grid_meta(&self) -> &EvalGridMeta {
        &self.grid_meta
    }

    /// Read-only access to the per-`(category, image)` IoU matrices
    /// retained when `params.retain_iou` was set at construction. `None`
    /// on the default no-retention path.
    pub fn retained_ious(&self) -> Option<&crate::tables::RetainedIous> {
        self.retained_ious.as_ref()
    }

    /// Update with a new batch of detections, parsed from
    /// `loadRes`-shaped JSON bytes.
    ///
    /// # Errors
    ///
    /// Propagates [`EvalError`] from the parse path, the underlying
    /// [`evaluate_with`] call, and the budget check
    /// ([`EvalError::OutOfBudget`]). On any error the evaluator state
    /// is unchanged and remains usable.
    pub fn update(&mut self, json_bytes: &[u8]) -> Result<UpdateReport, EvalError> {
        let parsed = ParsedDetections::<K>::from_json_bytes(json_bytes)?;
        self.update_parsed(parsed)
    }

    /// Update with a pre-parsed batch.
    ///
    /// # Errors
    ///
    /// - [`EvalError::InvalidAnnotation`] if any detection's `image_id`
    ///   has already been processed in a prior `update()` call (no
    ///   silent merge — submit duplicates as a single batch instead).
    /// - [`EvalError::OutOfBudget`] if the projected post-insert total
    ///   crosses [`MemoryBudget::bytes`]. State is unchanged on error.
    /// - Any [`EvalError`] from the underlying [`evaluate_with`] call.
    pub fn update_parsed(
        &mut self,
        parsed: ParsedDetections<K>,
    ) -> Result<UpdateReport, EvalError> {
        let detections = parsed.detections;

        // Reject any detection whose image_id was already seen in a
        // prior batch. This keeps update() additive: each cell is built
        // exactly once and never mutated, which is what makes
        // finalize() bit-identical to a batch run.
        let mut batch_image_ids: HashSet<i64> = HashSet::new();
        for dt in detections.detections() {
            let id = dt.image_id.0;
            if self.seen_images.contains(&id) {
                return Err(EvalError::InvalidAnnotation {
                    detail: format!(
                        "image_id={id} was already submitted in a prior update(); \
                         StreamingEvaluator does not silently merge — submit all \
                         detections for an image in a single batch"
                    ),
                });
            }
            batch_image_ids.insert(id);
        }

        // Run the unchanged batch orchestrator over just this batch's
        // detections. The grid it returns has the same `(K, A, I)`
        // shape as the streaming evaluator's target grid (we share the
        // dataset and params), but the orchestrator iterates the *full*
        // GT image set — every image with any GTs produces cells, even
        // when no detection in this batch landed on it. We filter those
        // out below: streaming semantics file a cell exactly once,
        // when its image first appears in a batch.
        let mut grid = evaluate_with(
            &self.dataset,
            &detections,
            self.params.borrow(),
            self.parity_mode,
            &self.kernel,
        )?;

        // Map batch image_ids to their I-axis indices so we can keep
        // only the cells whose image appears in this batch. Anything
        // else would (a) double-count GTs on every empty / partial
        // update and (b) trip the duplicate-image_id guard the next
        // time those images receive their own detections.
        let mut batch_image_indices: HashSet<usize> = HashSet::with_capacity(batch_image_ids.len());
        for id in &batch_image_ids {
            if let Some(&idx) = self.grid_meta.image_id_to_idx.get(&ImageId(*id)) {
                batch_image_indices.insert(idx);
            }
            // Unknown image_ids fall through silently; the underlying
            // gather treats them as "no GTs / no DTs of interest" so
            // they produce no cells anyway.
        }

        // Pre-compute insertion cost. Iterate batch images first so the
        // walk is `O(batch_size * K * A)` instead of `O(I * K * A)` —
        // critical at COCO scale where `I` is in the thousands but a
        // typical training-loop batch covers tens of images.
        let n_t = self.params.iou_thresholds.len();
        let n_k = grid.n_categories;
        let n_a = grid.n_area_ranges;
        let n_i = grid.n_images;
        let mut staged: Vec<(usize, usize, usize, PerImageEval, CellCost)> = Vec::new();
        let mut cost_total = CellCost::default();
        for &i in &batch_image_indices {
            for k in 0..n_k {
                for a in 0..n_a {
                    let flat = k * n_a * n_i + a * n_i + i;
                    if let Some(cell) = grid.eval_imgs.get(flat).and_then(|opt| opt.as_deref()) {
                        let cost = cell_cost(cell, n_t);
                        cost_total = cost_total.add(cost);
                        staged.push((k, a, i, cell.clone(), cost));
                    }
                }
            }
        }

        // Budget check: do not mutate state on overflow.
        let projected = self.memory_used_bytes() + cost_total.total();
        if projected > self.budget.bytes {
            let mut breakdown: HashMap<&'static str, usize> = HashMap::new();
            breakdown.insert(
                "cells_store",
                self.bytes_cells_struct + cost_total.cells_struct,
            );
            breakdown.insert("scores", self.bytes_dt_scores + cost_total.dt_scores);
            breakdown.insert(
                "match_flags",
                self.bytes_match_flags + cost_total.match_flags,
            );
            return Err(EvalError::OutOfBudget {
                used_bytes: projected,
                budget_bytes: self.budget.bytes,
                breakdown,
            });
        }

        // All-or-nothing commit. The error paths above ran before any
        // mutation; from here, every step is infallible.
        let n_cells_inserted = staged.len();
        for (k, a, i, cell, cost) in staged {
            self.cells.insert(k, a, i, cell);
            self.bytes_cells_struct += cost.cells_struct;
            self.bytes_dt_scores += cost.dt_scores;
            self.bytes_match_flags += cost.match_flags;
        }

        // Streaming rejects duplicate image_ids earlier, so each (k, i)
        // entry is inserted at most once across all `update()` calls.
        if let (Some(store), Some(per_batch)) =
            (self.retained_ious.as_mut(), grid.retained_ious.as_mut())
        {
            for k in 0..n_k {
                for &i in &batch_image_indices {
                    if let Some(iou) = per_batch.remove(k, i) {
                        store.insert(k, i, iou);
                    }
                }
            }
        }

        // Retain `EvalImageMeta` and detection records when `retain_iou`
        // is on — both are needed by per_detection / per_pair table
        // builders. Cells without retention skip this work entirely.
        if self.params.retain_iou {
            for &i in &batch_image_indices {
                for k in 0..n_k {
                    for a in 0..n_a {
                        let flat = k * n_a * n_i + a * n_i + i;
                        if let Some(meta) = grid
                            .eval_imgs_meta
                            .get_mut(flat)
                            .and_then(Option::take)
                            .map(|b| *b)
                        {
                            self.meta_cells.insert((k, a, i), meta);
                        }
                    }
                }
            }
            self.dets_seen
                .extend(detections.detections().iter().cloned());
        }

        let n_detections_accepted = detections.detections().len();
        self.n_detections += n_detections_accepted;
        self.next_dt_id = self.next_dt_id.saturating_add(n_detections_accepted as i64);
        for id in &batch_image_ids {
            self.seen_images.insert(*id);
        }
        for idx in &batch_image_indices {
            self.seen_image_indices.insert(*idx);
        }

        let total_used = self.memory_used_bytes();
        let threshold = (self.budget.bytes as f64 * self.budget.soft_warn_fraction) as usize;
        let soft_warn_triggered = total_used >= threshold && !self.soft_warn_fired;
        if soft_warn_triggered {
            self.soft_warn_fired = true;
        }

        Ok(UpdateReport {
            n_detections_accepted,
            n_images_in_batch: batch_image_ids.len(),
            n_cells_inserted,
            soft_warn_triggered,
        })
    }

    /// Compute a [`Summary`] over the current store. Cheap to call
    /// repeatedly. Bit-identical to a batch run over the union of all
    /// detections submitted via `update()` so far (modulo stream-order
    /// ULP wobble in `corrected` mode — see ADR-0013 §Determinism).
    ///
    /// Takes `&mut self` because the first call materializes a cached
    /// GT-only `(K, A, I)` grid for images that haven't received any
    /// detection yet; subsequent snapshots reuse the cache.
    ///
    /// # Errors
    ///
    /// Propagates [`EvalError`] from the underlying [`accumulate`] or
    /// summarize call.
    pub fn snapshot(&mut self) -> Result<Summary, EvalError> {
        self.compute_summary()
    }

    /// "Fast" snapshot. v0: identical to [`Self::snapshot`].
    ///
    /// # Errors
    ///
    /// Same conditions as [`Self::snapshot`].
    pub fn snapshot_running(&mut self) -> Result<Summary, EvalError> {
        // TODO(ADR-0013): running-mode PR-curve approximation.
        self.snapshot()
    }

    /// Consume the evaluator and return its final [`Summary`].
    ///
    /// # Errors
    ///
    /// Propagates [`EvalError`] from the underlying [`accumulate`] or
    /// summarize call.
    pub fn finalize(mut self) -> Result<Summary, EvalError> {
        self.compute_summary()
    }

    /// Consume the evaluator and return both its final [`Summary`] and
    /// the requested result tables.
    ///
    /// v0.5 supports the *cheap* tables ([`crate::TablesRequest::per_image`],
    /// [`crate::TablesRequest::per_class`]) on the streaming path —
    /// neither needs the per-cell `EvalImageMeta` the cells store would
    /// have to also retain. `per_detection` / `per_pair` on streaming
    /// returns [`EvalError::NotImplemented`]; callers who need those
    /// today should run the same workload via [`crate::evaluate_with`]
    /// in batch mode.
    ///
    /// # Errors
    ///
    /// - [`EvalError::NotImplemented`] when `request.per_detection` or
    ///   `request.per_pair` is set.
    /// - Any error from the underlying [`accumulate`] / summarize /
    ///   [`crate::build_per_image`] / [`crate::build_per_class`] calls.
    pub fn finalize_with_tables(
        mut self,
        request: crate::TablesRequest,
        config: &crate::TablesConfig,
    ) -> Result<(Summary, crate::Tables), EvalError> {
        self.compute_summary_and_tables(request, config)
    }

    /// Mid-stream version of [`Self::finalize_with_tables`]. See
    /// [`Self::snapshot`] for the determinism caveat.
    ///
    /// # Errors
    ///
    /// Same conditions as [`Self::finalize_with_tables`].
    pub fn snapshot_with_tables(
        &mut self,
        request: crate::TablesRequest,
        config: &crate::TablesConfig,
    ) -> Result<(Summary, crate::Tables), EvalError> {
        self.compute_summary_and_tables(request, config)
    }

    fn compute_summary_and_tables(
        &mut self,
        request: crate::TablesRequest,
        config: &crate::TablesConfig,
    ) -> Result<(Summary, crate::Tables), EvalError> {
        if request.requires_iou_retention() && !self.params.retain_iou {
            return Err(EvalError::InvalidConfig {
                detail: "per_detection / per_pair require retain_iou=True at \
                         StreamingEvaluator construction; rebuild the evaluator \
                         with retain_iou=True to opt in"
                    .into(),
            });
        }

        // Build a synthetic EvalGrid from the cells store + GT-only
        // overlay, identical in shape to what `evaluate_with` would
        // have produced for the union of all submitted batches. The
        // dense `eval_imgs` slice fully drives `build_per_image` /
        // `build_per_class`; `eval_imgs_meta` is densified from
        // `meta_cells` for per_detection / per_pair, and
        // `retained_ious` is cloned from the streaming store.
        let n_k = self.grid_meta.n_categories;
        let n_a = self.grid_meta.n_area_ranges;
        let n_i = self.grid_meta.n_images;
        let total = n_k * n_a * n_i;
        let mut eval_imgs = self.cells.flatten(&self.grid_meta);
        let eval_imgs_meta: Vec<Option<Box<EvalImageMeta>>> = if self.params.retain_iou {
            let mut out: Vec<Option<Box<EvalImageMeta>>> = Vec::with_capacity(total);
            for k in 0..n_k {
                for a in 0..n_a {
                    for i in 0..n_i {
                        out.push(self.meta_cells.get(&(k, a, i)).cloned().map(Box::new));
                    }
                }
            }
            out
        } else {
            vec![None; total]
        };
        if self.images_seen() < self.grid_meta.n_images {
            self.ensure_gt_only_cells()?;
            let gt_only = self
                .gt_only_cells
                .as_ref()
                .ok_or_else(|| EvalError::InvalidConfig {
                    detail: "gt_only_cells cache missing after init".into(),
                })?;
            for i in 0..n_i {
                if self.seen_image_indices.contains(&i) {
                    continue;
                }
                for k in 0..n_k {
                    for a in 0..n_a {
                        let flat = k * n_a * n_i + a * n_i + i;
                        if let Some(cell) = gt_only.get(flat).and_then(|opt| opt.as_ref()) {
                            eval_imgs[flat] = Some(cell.clone());
                            // GT-only images contribute no detections;
                            // leave `eval_imgs_meta[flat]` at None so
                            // per_detection / per_pair simply skip them.
                        }
                    }
                }
            }
        }

        let synthetic_grid = crate::EvalGrid {
            eval_imgs,
            eval_imgs_meta,
            n_categories: n_k,
            n_area_ranges: n_a,
            n_images: n_i,
            retained_ious: self.retained_ious.clone(),
        };

        // Standard COCO ladder per the existing `compute_summary` path.
        let max_dets: [usize; 3] = [1, 10, 100];
        let accum_params = AccumulateParams {
            iou_thresholds: &self.params.iou_thresholds,
            recall_thresholds: recall_thresholds(),
            max_dets: &max_dets,
            n_categories: n_k,
            n_area_ranges: n_a,
            n_images: n_i,
        };
        let accumulated = accumulate(&synthetic_grid.eval_imgs, accum_params, self.parity_mode)?;
        let summary = if self.kernel.is_keypoints() {
            let kp_max_dets: [usize; 1] = [20];
            let accum_params_kp = AccumulateParams {
                iou_thresholds: &self.params.iou_thresholds,
                recall_thresholds: recall_thresholds(),
                max_dets: &kp_max_dets,
                n_categories: n_k,
                n_area_ranges: n_a,
                n_images: n_i,
            };
            let accumulated_kp =
                accumulate(&synthetic_grid.eval_imgs, accum_params_kp, self.parity_mode)?;
            let plan = StatRequest::coco_keypoints_default();
            summarize_with(
                &accumulated_kp,
                &plan,
                &self.params.iou_thresholds,
                &kp_max_dets,
            )?
        } else {
            summarize_detection(&accumulated, &self.params.iou_thresholds, &max_dets)?
        };

        // Build a fresh `CocoDetections` view over every detection seen
        // so far when per_detection is requested; cheaper tables don't
        // need the records, so skip the work entirely there.
        let detections_view = if request.per_detection {
            Some(CocoDetections::from_records(self.dets_seen.clone()))
        } else {
            None
        };
        let tables = crate::build_tables(
            &synthetic_grid,
            &accumulated,
            &self.dataset,
            detections_view.as_ref(),
            self.retained_ious.as_ref(),
            &self.params.iou_thresholds,
            &max_dets,
            request,
            config,
        )?;
        Ok((summary, tables))
    }

    /// Serialize the current evaluator state to an opaque byte blob
    /// (ADR-0031 partial wire format). Non-consuming variant — the
    /// evaluator stays usable for further `update` calls.
    ///
    /// # Errors
    ///
    /// [`EvalError::PartialFormatMismatch`] if rkyv archiving fails;
    /// [`EvalError::InvalidConfig`] if params hashing fails.
    pub fn snapshot_to_partial(&self) -> Result<Vec<u8>, EvalError> {
        crate::distributed::encode(&self.encode_input())
    }

    /// Consuming variant of [`Self::snapshot_to_partial`]. The
    /// evaluator is dropped after the partial is produced — the
    /// expected shape for the rank-local final state in a
    /// distributed-eval gather (ADR-0031).
    ///
    /// # Errors
    ///
    /// Same as [`Self::snapshot_to_partial`].
    pub fn finalize_to_partial(self) -> Result<Vec<u8>, EvalError> {
        crate::distributed::encode(&self.encode_input())
    }

    fn encode_input(&self) -> crate::distributed::EncodeInput<'_, K> {
        crate::distributed::EncodeInput {
            dataset: &self.dataset,
            kernel: &self.kernel,
            params: &self.params,
            parity_mode: self.parity_mode,
            rank_id: self.rank_id,
            n_categories: self.grid_meta.n_categories as u32,
            n_area_ranges: self.grid_meta.n_area_ranges as u32,
            n_images: self.grid_meta.n_images as u32,
            n_detections: self.n_detections as u64,
            next_dt_id: self.next_dt_id,
            seen_images: &self.seen_images,
            cells: self.cells.as_map(),
            meta_cells: if self.params.retain_iou {
                Some(&self.meta_cells)
            } else {
                None
            },
            retained_ious: self.retained_ious.as_ref(),
            dets_seen: if self.params.retain_iou {
                Some(self.dets_seen.as_slice())
            } else {
                None
            },
            retain_iou: self.params.retain_iou,
        }
    }

    /// Equivalent to [`Self::snapshot_to_partial`]. Retained for
    /// ADR-0013 API stability — the public surface ships under both
    /// names so users tracking either ADR find the call they expect.
    ///
    /// # Errors
    ///
    /// Same as [`Self::snapshot_to_partial`].
    pub fn checkpoint(&self) -> Result<Vec<u8>, EvalError> {
        self.snapshot_to_partial()
    }

    /// Reconstruct an evaluator from a single
    /// [`Self::checkpoint`] / [`Self::snapshot_to_partial`] blob.
    /// Thin wrapper over [`Self::from_partials`] with a
    /// single-element slice; preserves the ADR-0013 signature for
    /// crash-recovery callers.
    ///
    /// # Errors
    ///
    /// All variants of [`Self::from_partials`].
    pub fn restore(
        dataset: CocoDataset,
        kernel: K,
        params: OwnedEvaluateParams,
        parity_mode: ParityMode,
        budget: MemoryBudget,
        bytes: &[u8],
    ) -> Result<Self, EvalError> {
        Self::from_partials(dataset, kernel, params, parity_mode, budget, &[bytes])
    }

    /// Construct an evaluator equivalent to a batch run over the
    /// union of all partials' submitted detections (ADR-0031).
    ///
    /// All partials must share `dataset_hash`, `params_hash`,
    /// `parity_mode`, kernel kind, `retain_iou`, and grid dimensions.
    /// In strict mode, every partial must declare a distinct
    /// `rank_id`. Image-id sets across partials must be disjoint.
    ///
    /// # Errors
    ///
    /// - [`EvalError::PartialFormatMismatch`] on framing or rkyv
    ///   validation failures (magic, version, CRC, kernel kind, grid
    ///   dims, parity, retain_iou).
    /// - [`EvalError::PartialDatasetMismatch`] on dataset_hash
    ///   divergence.
    /// - [`EvalError::PartialParamsMismatch`] on params_hash
    ///   divergence.
    /// - [`EvalError::PartialPartitionOverlap`] when two partials
    ///   cover the same `image_id`.
    /// - [`EvalError::PartialRankCollision`] when two strict-mode
    ///   partials share a `rank_id`.
    pub fn from_partials(
        dataset: CocoDataset,
        kernel: K,
        params: OwnedEvaluateParams,
        parity_mode: ParityMode,
        budget: MemoryBudget,
        partials: &[&[u8]],
    ) -> Result<Self, EvalError> {
        let mut ev = Self::new(dataset, kernel, params, parity_mode, budget)?;
        let expected = crate::distributed::instance_expectation(
            &ev.dataset,
            &ev.kernel,
            &ev.params,
            parity_mode,
            ev.grid_meta.n_categories as u32,
            ev.grid_meta.n_area_ranges as u32,
            ev.grid_meta.n_images as u32,
        )?;
        let mut acc =
            crate::distributed::InstanceMergeAccumulator::new(parity_mode == ParityMode::Strict);
        acc.set_retain_iou(ev.params.retain_iou);
        for bytes in partials {
            vernier_partial::with_validated_envelope(bytes, &expected, |view| acc.ingest(&view))?;
        }
        ev.install_merged_state(acc)?;
        Ok(ev)
    }

    /// Swap a freshly merged [`crate::distributed::InstanceMergeAccumulator`]
    /// into this evaluator's spine state. Internal helper for
    /// [`Self::from_partials`].
    fn install_merged_state(
        &mut self,
        acc: crate::distributed::InstanceMergeAccumulator,
    ) -> Result<(), EvalError> {
        // Bind every load-bearing field by name (`base`, `retain_iou`
        // are merge-internal bookkeeping that doesn't carry into the
        // spine — destructure asserts a future field addition is a
        // compile error rather than silent loss).
        let crate::distributed::InstanceMergeAccumulator {
            base,
            n_detections,
            next_dt_id,
            cells,
            meta_cells,
            retained_ious_map,
            dets_seen,
            retain_iou: _,
        } = acc;
        self.n_detections = n_detections;
        self.next_dt_id = next_dt_id;
        // base.image_ids() is the union image-id set;
        // seen_image_indices is the parallel local-index set under
        // the live grid_meta.
        self.seen_image_indices = base
            .image_ids()
            .filter_map(|id| self.grid_meta.image_id_to_idx.get(&ImageId(id)).copied())
            .collect();
        self.seen_images = base.image_ids().collect();
        self.cells = PerImageEvalStore::from_map(cells);
        self.meta_cells = meta_cells;
        if self.params.retain_iou {
            self.retained_ious = Some(crate::tables::RetainedIous::from_map(retained_ious_map));
        }
        self.dets_seen = dets_seen;
        Ok(())
    }

    /// Compute the GT-only `(K, A, I)` grid once and cache it. Subsequent
    /// snapshots reuse the cached grid; GT is immutable, so the
    /// underlying `evaluate_with` result never goes stale.
    fn ensure_gt_only_cells(&mut self) -> Result<(), EvalError> {
        if self.gt_only_cells.is_some() {
            return Ok(());
        }
        let empty_dt = CocoDetections::from_inputs(Vec::new())?;
        let grid = evaluate_with(
            &self.dataset,
            &empty_dt,
            self.params.borrow(),
            self.parity_mode,
            &self.kernel,
        )?;
        self.gt_only_cells = Some(grid.eval_imgs);
        Ok(())
    }

    /// Internal: shared implementation of `snapshot` and `finalize`.
    /// Mutates `self` only to populate the lazy `gt_only_cells` cache
    /// on its first call.
    fn compute_summary(&mut self) -> Result<Summary, EvalError> {
        let mut eval_imgs = self.cells.flatten(&self.grid_meta);

        // Overlay GT-only cells for images that never received any
        // detection across the entire stream. The batch path files
        // these (with `dt_scores=[]` + populated `gt_ignore`); without
        // this overlay, streaming `finalize().stats` diverges from
        // `Evaluator.evaluate(...).stats` whenever the GT contains
        // images with no DTs anywhere in the stream — see ADR-0013
        // §"Per-image cell coverage".
        if self.images_seen() < self.grid_meta.n_images {
            self.ensure_gt_only_cells()?;
            let n_k = self.grid_meta.n_categories;
            let n_a = self.grid_meta.n_area_ranges;
            let n_i = self.grid_meta.n_images;
            let gt_only = self
                .gt_only_cells
                .as_ref()
                .ok_or_else(|| EvalError::InvalidConfig {
                    detail: "gt_only_cells cache missing after init".into(),
                })?;
            for i in 0..n_i {
                if self.seen_image_indices.contains(&i) {
                    continue;
                }
                for k in 0..n_k {
                    for a in 0..n_a {
                        let flat = k * n_a * n_i + a * n_i + i;
                        if let Some(cell) = gt_only.get(flat).and_then(|opt| opt.as_ref()) {
                            eval_imgs[flat] = Some(cell.clone());
                        }
                    }
                }
            }
        }
        // Standard COCO ladder; pycocotools applies it unconditionally.
        let max_dets: [usize; 3] = [1, 10, 100];
        let accum_params = AccumulateParams {
            iou_thresholds: &self.params.iou_thresholds,
            recall_thresholds: recall_thresholds(),
            max_dets: &max_dets,
            n_categories: self.grid_meta.n_categories,
            n_area_ranges: self.grid_meta.n_area_ranges,
            n_images: self.grid_meta.n_images,
        };
        let accumulated = accumulate(&eval_imgs, accum_params, self.parity_mode)?;
        if self.kernel.is_keypoints() {
            // ADR-0012: keypoints uses a 10-stat plan with a 1-rung
            // max-dets ladder pinned at 20.
            let kp_max_dets: [usize; 1] = [20];
            // Re-run accumulate with the kp-canonical ladder so the
            // M-axis lengths line up with the summary plan.
            let accum_params_kp = AccumulateParams {
                iou_thresholds: &self.params.iou_thresholds,
                recall_thresholds: recall_thresholds(),
                max_dets: &kp_max_dets,
                n_categories: self.grid_meta.n_categories,
                n_area_ranges: self.grid_meta.n_area_ranges,
                n_images: self.grid_meta.n_images,
            };
            let accumulated_kp = accumulate(&eval_imgs, accum_params_kp, self.parity_mode)?;
            let plan = StatRequest::coco_keypoints_default();
            summarize_with(
                &accumulated_kp,
                &plan,
                &self.params.iou_thresholds,
                &kp_max_dets,
            )
        } else {
            summarize_detection(&accumulated, &self.params.iou_thresholds, &max_dets)
        }
    }
}

/// Per-cell memory cost breakdown.
#[derive(Debug, Default, Clone, Copy)]
struct CellCost {
    cells_struct: usize,
    dt_scores: usize,
    match_flags: usize,
}

impl CellCost {
    fn total(self) -> usize {
        self.cells_struct + self.dt_scores + self.match_flags
    }
    fn add(self, other: Self) -> Self {
        Self {
            cells_struct: self.cells_struct + other.cells_struct,
            dt_scores: self.dt_scores + other.dt_scores,
            match_flags: self.match_flags + other.match_flags,
        }
    }
}

/// Compute the memory cost of a single [`PerImageEval`] under the
/// budget accounting policy (`bytes_cells_struct + bytes_dt_scores +
/// bytes_match_flags`).
fn cell_cost(cell: &PerImageEval, n_iou_thresholds: usize) -> CellCost {
    let n_d = cell.dt_scores.len();
    CellCost {
        cells_struct: size_of::<PerImageEval>(),
        dt_scores: cell.dt_scores.capacity() * size_of::<f64>(),
        // dt_matched + dt_ignore are both `(T, D)` Bool arrays.
        match_flags: n_iou_thresholds
            .saturating_mul(n_d)
            .saturating_mul(size_of::<bool>())
            .saturating_mul(2),
    }
}

/// Build [`EvalGridMeta`] from the dataset and params, mirroring the
/// axis layout the batch [`crate::evaluate_with`] orchestrator emits.
fn build_grid_meta(dataset: &CocoDataset, params: &OwnedEvaluateParams) -> EvalGridMeta {
    let n_area_ranges = params.area_ranges.len();
    let n_images = dataset.images().len();

    // Image ids: same id-ascending sort the batch orchestrator uses, so
    // I-axis indices match between the two paths.
    let mut image_ids: Vec<ImageId> = dataset.images().iter().map(|im| im.id).collect();
    image_ids.sort_unstable_by_key(|id| id.0);
    let mut image_id_to_idx: HashMap<ImageId, usize> = HashMap::with_capacity(n_images);
    for (i, id) in image_ids.into_iter().enumerate() {
        image_id_to_idx.insert(id, i);
    }

    let (n_categories, category_id_to_idx) = if params.use_cats {
        let mut cat_ids: Vec<CategoryId> = dataset.categories().iter().map(|c| c.id).collect();
        cat_ids.sort_unstable_by_key(|c| c.0);
        let mut map: HashMap<CategoryId, usize> = HashMap::with_capacity(cat_ids.len());
        for (k, id) in cat_ids.iter().enumerate() {
            map.insert(*id, k);
        }
        (cat_ids.len(), map)
    } else {
        (1, HashMap::new())
    };

    EvalGridMeta {
        n_categories,
        n_area_ranges,
        n_images,
        category_id_to_idx,
        image_id_to_idx,
    }
}

// Silence the "imported but unused" warning on `CocoDetection` /
// `DetectionInput` / `AnnId` — these are part of the documented surface
// but unused by the compiled body. They live in the `use` block to
// keep the module's public-API touchpoints visible at a glance.
#[allow(dead_code)]
fn _docs_typecheck(_a: AnnId, _b: CocoDetection, _c: DetectionInput) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Bbox, CategoryMeta, CocoAnnotation, ImageMeta};
    use crate::evaluate::AreaRange;
    use crate::parity::iou_thresholds;
    use crate::similarity::BboxIou;

    fn img(id: i64, w: u32, h: u32) -> ImageMeta {
        ImageMeta {
            id: ImageId(id),
            width: w,
            height: h,
            file_name: None,
        }
    }

    fn cat(id: i64, name: &str) -> CategoryMeta {
        CategoryMeta {
            id: CategoryId(id),
            name: name.into(),
            supercategory: None,
        }
    }

    fn ann(id: i64, image: i64, cat: i64, bbox: (f64, f64, f64, f64)) -> CocoAnnotation {
        CocoAnnotation {
            id: AnnId(id),
            image_id: ImageId(image),
            category_id: CategoryId(cat),
            area: bbox.2 * bbox.3,
            is_crowd: false,
            ignore_flag: None,
            bbox: Bbox {
                x: bbox.0,
                y: bbox.1,
                w: bbox.2,
                h: bbox.3,
            },
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
        }
    }

    fn tiny_dataset() -> CocoDataset {
        let images = vec![img(1, 100, 100), img(2, 100, 100)];
        let cats = vec![cat(1, "thing")];
        let anns = vec![
            ann(1, 1, 1, (0.0, 0.0, 10.0, 10.0)),
            ann(2, 2, 1, (50.0, 50.0, 10.0, 10.0)),
        ];
        CocoDataset::from_parts(images, anns, cats).unwrap()
    }

    fn default_params() -> OwnedEvaluateParams {
        OwnedEvaluateParams {
            iou_thresholds: iou_thresholds().to_vec(),
            area_ranges: AreaRange::coco_default().to_vec(),
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        }
    }

    #[test]
    fn auto_default_budget_is_nonzero() {
        let b = MemoryBudget::auto_default();
        assert!(b.bytes > 0);
        assert!((b.soft_warn_fraction - DEFAULT_SOFT_WARN_FRACTION).abs() < 1e-12);
    }

    #[test]
    fn fresh_evaluator_reports_zero_counters() {
        let ds = tiny_dataset();
        let ev = StreamingEvaluator::new(
            ds,
            BboxIou,
            default_params(),
            ParityMode::Strict,
            MemoryBudget::auto_default(),
        )
        .unwrap();
        assert_eq!(ev.images_seen(), 0);
        assert_eq!(ev.detections_seen(), 0);
        assert_eq!(ev.memory_used_bytes(), 0);
        // 2 images in the dataset, 0 seen → 2 pending.
        assert_eq!(ev.images_pending(), 2);
        // K=1 cat, A=4 ranges, I=2 images.
        assert_eq!(ev.grid_meta().n_categories, 1);
        assert_eq!(ev.grid_meta().n_area_ranges, 4);
        assert_eq!(ev.grid_meta().n_images, 2);
    }

    #[test]
    fn empty_update_returns_zero_counters() {
        let ds = tiny_dataset();
        let mut ev = StreamingEvaluator::new(
            ds,
            BboxIou,
            default_params(),
            ParityMode::Strict,
            MemoryBudget::auto_default(),
        )
        .unwrap();
        let report = ev.update(b"[]").unwrap();
        assert_eq!(report.n_detections_accepted, 0);
        assert_eq!(report.n_images_in_batch, 0);
        assert_eq!(report.n_cells_inserted, 0);
        assert!(!report.soft_warn_triggered);
        assert_eq!(ev.detections_seen(), 0);
        assert_eq!(ev.images_seen(), 0);
        assert_eq!(ev.memory_used_bytes(), 0);
    }

    #[test]
    fn finalize_returns_summary_with_canonical_shape() {
        // 12 stats for a detection kernel; we don't pin values here —
        // the parity tests do that. Smoke check only.
        let ds = tiny_dataset();
        let ev = StreamingEvaluator::new(
            ds,
            BboxIou,
            default_params(),
            ParityMode::Strict,
            MemoryBudget::auto_default(),
        )
        .unwrap();
        let summary = ev.finalize().unwrap();
        assert_eq!(summary.lines.len(), 12);
    }

    #[test]
    fn duplicate_image_id_across_updates_is_rejected() {
        let ds = tiny_dataset();
        let mut ev = StreamingEvaluator::new(
            ds,
            BboxIou,
            default_params(),
            ParityMode::Strict,
            MemoryBudget::auto_default(),
        )
        .unwrap();
        // First batch: a single DT on image 1.
        let batch1 =
            br#"[{"image_id": 1, "category_id": 1, "score": 0.9, "bbox": [0, 0, 10, 10]}]"#;
        ev.update(batch1).unwrap();
        assert_eq!(ev.images_seen(), 1);

        // Second batch: another DT on the same image — must be rejected.
        let batch2 =
            br#"[{"image_id": 1, "category_id": 1, "score": 0.8, "bbox": [50, 50, 10, 10]}]"#;
        let err = ev.update(batch2).unwrap_err();
        assert!(matches!(err, EvalError::InvalidAnnotation { .. }));
        // State unchanged: still one image seen, original counters intact.
        assert_eq!(ev.images_seen(), 1);
        assert_eq!(ev.detections_seen(), 1);
    }

    #[test]
    fn out_of_budget_does_not_mutate_state() {
        let ds = tiny_dataset();
        let tiny_budget = MemoryBudget {
            bytes: 1, // pathologically small — first cell will overflow
            soft_warn_fraction: 0.80,
        };
        let mut ev = StreamingEvaluator::new(
            ds,
            BboxIou,
            default_params(),
            ParityMode::Strict,
            tiny_budget,
        )
        .unwrap();
        let batch = br#"[{"image_id": 1, "category_id": 1, "score": 0.9, "bbox": [0, 0, 10, 10]}]"#;
        let err = ev.update(batch).unwrap_err();
        match err {
            EvalError::OutOfBudget {
                used_bytes,
                budget_bytes,
                breakdown,
            } => {
                assert!(used_bytes > budget_bytes);
                assert_eq!(budget_bytes, 1);
                assert!(breakdown.contains_key("cells_store"));
                assert!(breakdown.contains_key("scores"));
                assert!(breakdown.contains_key("match_flags"));
            }
            other => panic!("expected OutOfBudget, got {other:?}"),
        }
        // State unchanged.
        assert_eq!(ev.images_seen(), 0);
        assert_eq!(ev.detections_seen(), 0);
        assert_eq!(ev.memory_used_bytes(), 0);
    }

    fn dt_json(image_id: i64, score: f64, bbox: (f64, f64, f64, f64)) -> Vec<u8> {
        let body = format!(
            r#"[{{"image_id":{image_id},"category_id":1,"score":{score},"bbox":[{},{},{},{}]}}]"#,
            bbox.0, bbox.1, bbox.2, bbox.3
        );
        body.into_bytes()
    }

    #[test]
    fn checkpoint_round_trip_yields_equal_summary() {
        let ds = tiny_dataset();
        let mut ev = StreamingEvaluator::new(
            ds.clone(),
            BboxIou,
            default_params(),
            ParityMode::Corrected,
            MemoryBudget::auto_default(),
        )
        .unwrap();
        ev.update(&dt_json(1, 0.9, (0.0, 0.0, 10.0, 10.0))).unwrap();
        ev.update(&dt_json(2, 0.8, (50.0, 50.0, 10.0, 10.0)))
            .unwrap();

        let blob = ev.checkpoint().unwrap();
        let summary_a = ev.finalize().unwrap();

        let restored = StreamingEvaluator::<BboxIou>::restore(
            ds,
            BboxIou,
            default_params(),
            ParityMode::Corrected,
            MemoryBudget::auto_default(),
            &blob,
        )
        .unwrap();
        let summary_b = restored.finalize().unwrap();

        assert_eq!(summary_a.stats(), summary_b.stats());
    }

    #[test]
    fn from_partials_two_disjoint_partitions_equals_combined_stream() {
        // Rank 0 sees image 1; rank 1 sees image 2. Merge should produce
        // the same summary as a single evaluator that ate both images.
        let ds = tiny_dataset();
        let mut combined = StreamingEvaluator::new(
            ds.clone(),
            BboxIou,
            default_params(),
            ParityMode::Corrected,
            MemoryBudget::auto_default(),
        )
        .unwrap();
        combined
            .update(&dt_json(1, 0.9, (0.0, 0.0, 10.0, 10.0)))
            .unwrap();
        combined
            .update(&dt_json(2, 0.8, (50.0, 50.0, 10.0, 10.0)))
            .unwrap();
        let combined_summary = combined.finalize().unwrap();

        let mut rank0 = StreamingEvaluator::new(
            ds.clone(),
            BboxIou,
            default_params(),
            ParityMode::Corrected,
            MemoryBudget::auto_default(),
        )
        .unwrap()
        .with_rank(0)
        .unwrap();
        rank0
            .update(&dt_json(1, 0.9, (0.0, 0.0, 10.0, 10.0)))
            .unwrap();
        let p0 = rank0.finalize_to_partial().unwrap();

        let mut rank1 = StreamingEvaluator::new(
            ds.clone(),
            BboxIou,
            default_params(),
            ParityMode::Corrected,
            MemoryBudget::auto_default(),
        )
        .unwrap()
        .with_rank(1)
        .unwrap();
        rank1
            .update(&dt_json(2, 0.8, (50.0, 50.0, 10.0, 10.0)))
            .unwrap();
        let p1 = rank1.finalize_to_partial().unwrap();

        let merged = StreamingEvaluator::<BboxIou>::from_partials(
            ds,
            BboxIou,
            default_params(),
            ParityMode::Corrected,
            MemoryBudget::auto_default(),
            &[&p0, &p1],
        )
        .unwrap();
        let merged_summary = merged.finalize().unwrap();

        assert_eq!(combined_summary.stats(), merged_summary.stats());
    }

    #[test]
    fn from_partials_overlap_returns_partition_overlap_error() {
        let ds = tiny_dataset();
        let mut a = StreamingEvaluator::new(
            ds.clone(),
            BboxIou,
            default_params(),
            ParityMode::Corrected,
            MemoryBudget::auto_default(),
        )
        .unwrap()
        .with_rank(0)
        .unwrap();
        a.update(&dt_json(1, 0.9, (0.0, 0.0, 10.0, 10.0))).unwrap();
        let pa = a.finalize_to_partial().unwrap();

        // Both partials cover image 1 — the partition rule must reject.
        let mut b = StreamingEvaluator::new(
            ds.clone(),
            BboxIou,
            default_params(),
            ParityMode::Corrected,
            MemoryBudget::auto_default(),
        )
        .unwrap()
        .with_rank(1)
        .unwrap();
        b.update(&dt_json(1, 0.7, (5.0, 5.0, 10.0, 10.0))).unwrap();
        let pb = b.finalize_to_partial().unwrap();

        let err = StreamingEvaluator::<BboxIou>::from_partials(
            ds,
            BboxIou,
            default_params(),
            ParityMode::Corrected,
            MemoryBudget::auto_default(),
            &[&pa, &pb],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EvalError::PartialPartitionOverlap {
                rank_a: 0,
                rank_b: 1,
                image_id: 1,
            }
        ));
    }

    #[test]
    fn from_partials_strict_mode_rank_collision_rejected() {
        let ds = tiny_dataset();
        let mut a = StreamingEvaluator::new(
            ds.clone(),
            BboxIou,
            default_params(),
            ParityMode::Strict,
            MemoryBudget::auto_default(),
        )
        .unwrap()
        .with_rank(7)
        .unwrap();
        a.update(&dt_json(1, 0.9, (0.0, 0.0, 10.0, 10.0))).unwrap();
        let pa = a.finalize_to_partial().unwrap();

        let mut b = StreamingEvaluator::new(
            ds.clone(),
            BboxIou,
            default_params(),
            ParityMode::Strict,
            MemoryBudget::auto_default(),
        )
        .unwrap()
        .with_rank(7)
        .unwrap();
        b.update(&dt_json(2, 0.8, (50.0, 50.0, 10.0, 10.0)))
            .unwrap();
        let pb = b.finalize_to_partial().unwrap();

        let err = StreamingEvaluator::<BboxIou>::from_partials(
            ds,
            BboxIou,
            default_params(),
            ParityMode::Strict,
            MemoryBudget::auto_default(),
            &[&pa, &pb],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EvalError::PartialRankCollision { rank_id: 7 }
        ));
    }

    #[test]
    fn from_partials_dataset_hash_mismatch_rejected() {
        let ds_a = tiny_dataset();
        // ds_b shifts the bbox of one annotation by 1 px → different hash.
        let images = vec![img(1, 100, 100), img(2, 100, 100)];
        let cats = vec![cat(1, "thing")];
        let anns = vec![
            ann(1, 1, 1, (1.0, 0.0, 10.0, 10.0)), // shifted from (0,0,10,10)
            ann(2, 2, 1, (50.0, 50.0, 10.0, 10.0)),
        ];
        let ds_b = CocoDataset::from_parts(images, anns, cats).unwrap();
        assert_ne!(ds_a.dataset_hash(), ds_b.dataset_hash());

        let mut ev = StreamingEvaluator::new(
            ds_a,
            BboxIou,
            default_params(),
            ParityMode::Corrected,
            MemoryBudget::auto_default(),
        )
        .unwrap();
        ev.update(&dt_json(1, 0.9, (0.0, 0.0, 10.0, 10.0))).unwrap();
        let blob = ev.finalize_to_partial().unwrap();

        let err = StreamingEvaluator::<BboxIou>::from_partials(
            ds_b,
            BboxIou,
            default_params(),
            ParityMode::Corrected,
            MemoryBudget::auto_default(),
            &[&blob],
        )
        .unwrap_err();
        assert!(matches!(err, EvalError::PartialDatasetMismatch { .. }));
    }

    #[test]
    fn from_partials_params_hash_mismatch_rejected() {
        let ds = tiny_dataset();
        let mut ev = StreamingEvaluator::new(
            ds.clone(),
            BboxIou,
            default_params(),
            ParityMode::Corrected,
            MemoryBudget::auto_default(),
        )
        .unwrap();
        ev.update(&dt_json(1, 0.9, (0.0, 0.0, 10.0, 10.0))).unwrap();
        let blob = ev.finalize_to_partial().unwrap();

        let mut other_params = default_params();
        other_params.max_dets_per_image = 50; // diverges from the default 100.

        let err = StreamingEvaluator::<BboxIou>::from_partials(
            ds,
            BboxIou,
            other_params,
            ParityMode::Corrected,
            MemoryBudget::auto_default(),
            &[&blob],
        )
        .unwrap_err();
        assert!(matches!(err, EvalError::PartialParamsMismatch { .. }));
    }

    #[test]
    fn with_rank_after_update_is_rejected() {
        let ds = tiny_dataset();
        let mut ev = StreamingEvaluator::new(
            ds,
            BboxIou,
            default_params(),
            ParityMode::Corrected,
            MemoryBudget::auto_default(),
        )
        .unwrap();
        ev.update(&dt_json(1, 0.9, (0.0, 0.0, 10.0, 10.0))).unwrap();
        let err = ev.with_rank(0).unwrap_err();
        assert!(matches!(err, EvalError::InvalidConfig { .. }));
    }

    #[test]
    fn corrupted_partial_returns_format_mismatch() {
        let ds = tiny_dataset();
        let mut ev = StreamingEvaluator::new(
            ds.clone(),
            BboxIou,
            default_params(),
            ParityMode::Corrected,
            MemoryBudget::auto_default(),
        )
        .unwrap();
        ev.update(&dt_json(1, 0.9, (0.0, 0.0, 10.0, 10.0))).unwrap();
        let mut blob = ev.finalize_to_partial().unwrap();
        // Corrupt the magic bytes — earliest validation step.
        blob[0] = b'X';

        let err = StreamingEvaluator::<BboxIou>::from_partials(
            ds,
            BboxIou,
            default_params(),
            ParityMode::Corrected,
            MemoryBudget::auto_default(),
            &[&blob],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EvalError::PartialFormatMismatch {
                kind: crate::error::PartialFormatErrorKind::WrongMagic { .. }
            }
        ));
    }

    #[test]
    fn flatten_round_trips_to_dense_layout() {
        let mut store = PerImageEvalStore::new();
        // Insert one cell at (0, 0, 0) of a 1x1x2 grid.
        let cell = PerImageEval {
            dt_scores: vec![0.5],
            dt_matched: ndarray::Array2::default((1, 1)),
            dt_ignore: ndarray::Array2::default((1, 1)),
            gt_ignore: vec![false],
        };
        store.insert(0, 0, 0, cell);
        let meta = EvalGridMeta {
            n_categories: 1,
            n_area_ranges: 1,
            n_images: 2,
            category_id_to_idx: HashMap::new(),
            image_id_to_idx: HashMap::new(),
        };
        let dense = store.flatten(&meta);
        assert_eq!(dense.len(), 2);
        assert!(dense[0].is_some());
        assert!(dense[1].is_none());
    }
}
