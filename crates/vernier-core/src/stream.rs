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
use crate::evaluate::{evaluate_with, EvalKernel, OwnedEvaluateParams};
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

    /// Densify into the `[k * A * I + a * I + i]`-laid-out
    /// `Vec<Option<PerImageEval>>` that [`crate::accumulate`] consumes.
    /// Cloning is intentional — `accumulate` borrows the slice and the
    /// store must remain valid for further updates after a snapshot.
    pub fn flatten(&self, meta: &EvalGridMeta) -> Vec<Option<PerImageEval>> {
        let total = meta.n_categories * meta.n_area_ranges * meta.n_images;
        let mut out: Vec<Option<PerImageEval>> = Vec::with_capacity(total);
        for k in 0..meta.n_categories {
            for a in 0..meta.n_area_ranges {
                for i in 0..meta.n_images {
                    out.push(self.cells.get(&(k, a, i)).cloned());
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
#[derive(Debug)]
pub struct StreamingEvaluator<K: EvalKernel> {
    dataset: CocoDataset,
    kernel: K,
    params: OwnedEvaluateParams,
    parity_mode: ParityMode,
    grid_meta: EvalGridMeta,
    cells: PerImageEvalStore,
    seen_images: HashSet<i64>,
    /// Image-grid indices for every entry in `seen_images`. Maintained
    /// incrementally so `compute_summary` can decide which GT-only
    /// cells to overlay without re-walking `seen_images` every call.
    seen_image_indices: HashSet<usize>,
    /// Lazily-built GT-only `(K, A, I)` grid used by `compute_summary`
    /// to file cells for images that have not yet received any
    /// detection. GT is immutable, so the grid is computed at most
    /// once across the evaluator's lifetime.
    gt_only_cells: Option<Vec<Option<PerImageEval>>>,
    n_detections: usize,
    /// Monotonic DT-id counter. Reserved for the strict-mode
    /// `(score, stream_position)` tiebreak (ADR-0013 §Determinism); not
    /// yet consumed by the matching path.
    next_dt_id: i64,
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
        Ok(Self {
            dataset,
            kernel,
            params,
            parity_mode,
            grid_meta,
            cells: PerImageEvalStore::new(),
            seen_images: HashSet::new(),
            seen_image_indices: HashSet::new(),
            gt_only_cells: None,
            n_detections: 0,
            next_dt_id: 1,
            bytes_cells_struct: 0,
            bytes_dt_scores: 0,
            bytes_match_flags: 0,
            budget,
            soft_warn_fired: false,
        })
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
        let grid = evaluate_with(
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
                    if let Some(cell) = grid.eval_imgs.get(flat).and_then(|opt| opt.as_ref()) {
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

    /// Serialize evaluator state to an opaque byte blob suitable for
    /// crash recovery (per ADR-0013).
    ///
    /// # Errors
    ///
    /// **v0**: always returns [`EvalError::NotImplemented`]. A future
    /// ADR re-introduces a real implementation (rkyv-based).
    pub fn checkpoint(&self) -> Result<Vec<u8>, EvalError> {
        Err(EvalError::NotImplemented {
            feature: "StreamingEvaluator::checkpoint",
        })
    }

    /// Restore an evaluator from a [`Self::checkpoint`] blob.
    ///
    /// # Errors
    ///
    /// **v0**: always returns [`EvalError::NotImplemented`].
    pub fn restore(_bytes: &[u8]) -> Result<Self, EvalError> {
        Err(EvalError::NotImplemented {
            feature: "StreamingEvaluator::restore",
        })
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

    #[test]
    fn checkpoint_and_restore_return_not_implemented() {
        let ds = tiny_dataset();
        let ev = StreamingEvaluator::new(
            ds,
            BboxIou,
            default_params(),
            ParityMode::Strict,
            MemoryBudget::auto_default(),
        )
        .unwrap();
        let err = ev.checkpoint().unwrap_err();
        assert!(matches!(err, EvalError::NotImplemented { .. }));
        let err = StreamingEvaluator::<BboxIou>::restore(&[]).unwrap_err();
        assert!(matches!(err, EvalError::NotImplemented { .. }));
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
