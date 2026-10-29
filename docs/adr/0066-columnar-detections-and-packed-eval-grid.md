# ADR-0066: Columnar detection storage, `DetectionsBatch` ingest, and packed evaluation grids

- **Status:** proposed
- **Date:** 2026-10-28
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

Profiling `vernier` 0.5.4 against `ultrafast-pycocotools` (`ufcoco` 0.1.14) on RF-DETR validation workloads (`5,000` images × `300` unthresholded queries = `1,500,000` detections across `80` categories, `35,000` ground truths) exposed five compounding architectural bottlenecks:

1. **Python columnar repackaging (`_samples.py:_detections`).** Because ADR-0030's `Detections` dict takes a scalar `image_id: int`, `coco_inputs` and `coco_inputs_from_columns` could not hand already-concatenated arrays across FFI directly: `bbox` allocated an `84 MB` `(N, 7)` `float64` copy via `np.column_stack` (widening `int64` image and category IDs to `float64` only for Rust to round-trip check `3,000,000` floats back to `i64`), while `segm` constructed `5,000` per-image Python `Detections` dicts.
2. **Double AoS materialization & SSE2 `libm::trunc` calls at FFI.** `result_ingest` and `array_ingest` materialized `Vec<DetectionInput>` (`~160 B × 1.5M = 240 MB`, plus `3,000,000` `f64::fract` external `libm::trunc` calls on baseline x86_64 SSE2 builds) under the GIL, which `CocoDetections::from_inputs_with_area` immediately copied into a second AoS `Vec<CocoDetection>` (`~144 B × 1.5M = 216 MB`, `80 B` of dead `None` option padding per bbox detection).
3. **Per-key heap vector indexing.** `CocoDetections` indexed `(image_id, category_id)` and `image_id` using `std::collections::HashMap<..., Vec<usize>>` (SipHash, unreserved), allocating and later freeing `~393,000` individual heap `Vec<usize>` buffers on dense-query DETR outputs.
4. **Per-cell temporary heap churn on empty-GT DETR cells.** Unthresholded 300-query DETR outputs populate `~390,000` of `400,000` `(image, category)` cells even though `93.6%` (`~365,000`) contain zero ground truths (`n_g == 0`). Across `4` area ranges (`1,560,000` `evaluate_cell` calls), `evaluate_cell` and `match_image_with_perm_gated` executed `8–11` temporary heap allocations per cell (`~14M` `malloc`/`free` pairs), allocating and immediately dropping `dt_identity_perm`, `gt_perm`, and `(T, D)` / `(T, G)` `Array2<i64>` match matrices even when `retain_meta == false` and `n_g == 0`.
5. **Pointer-chased boxed `EvalGrid` (`7.8M` heap blocks).** Storing `EvalGrid` eagerly as `Vec<Option<Box<PerImageEval>>>` allocated `5` heap blocks per occupied `(k, a, i)` cell (`7.8M` allocations across Rayon worker arenas, followed by `7.8M` cross-arena `free()` calls on the Python thread at drop and `384 ms` of pointer-chasing inside `accumulate`).

## Decision drivers

- **First-principles hardware efficiency:** minimize bytes touched, soft page faults, allocator round-trips, and pointer indirections so throughput tracks memory bandwidth and L1/L2 cache speed.
- **ADR-0057 single convergence point:** every detection ingest route must converge on one constructor owning quirks **J1**, **J3**, **E2/J4**, and non-finite score rejection so routes cannot drift.
- **ADR-0005 layer separation:** `matching` and `accumulate` remain strictly agnostic to IoU kernels and dataset types.
- **ADR-0002 bit-exact parity:** zero change to IEEE-754 floating-point evaluation order across `strict` and `corrected` modes, and `#![forbid(unsafe_code)]` in `vernier-core`.
- **Clean 0.6.0 minor-version break:** rather than carrying permanent AoS compatibility shims in `vernier-core`, adopt the long-term Structure-of-Arrays (SoA) and packed-arena representations directly.

## Decision outcome

1. **Widen `DetectionsInput` with `DetectionsBatch`.**
   `vernier._array_types.DetectionsBatch` accepts whole-dataset or multi-image batch columns in one `TypedDict`:
   - `boxes`: `(N, 4)` C-contiguous `float64` (`xywh`)
   - `scores`: `(N,)` `float64`
   - `labels`: `(N,)` `int64`
   - `image_ids`: `(N,)` `int64` when `counts` is omitted, or `(M,)` `int64` paired with `counts: (M,) int64` (`counts[m] >= 0`, `sum(counts) == N`)
   - `rles: Sequence[RLEInput]` (length `N`), required under `segm` / `boundary` exactly as on the per-image `Detections` dict, and `keypoints: (N, K, 3) float64`, required under `keypoints`.
   `DetectionsBatch` is the array surface's multi-image form, so it takes that surface's rule: quirk **J2**'s bbox-rectangle fallback stays with the routes that mirror a `loadRes` file (JSON bytes, result dicts, the `(N, 7)` matrix). On an array route a missing mask column is a caller bug, and J2 would turn it into a plausible, wrong segm AP.
   A dict carrying `image_ids` routes to `DetectionsBatch`; specifying both `image_id` and `image_ids` is refused explicitly. `coco_inputs` and `coco_inputs_from_columns` emit `DetectionsBatch` directly with zero Python array copies on both `bbox` and `segm`, attaching `rles` exactly when the prediction mask column covers every row.
   The numeric columns are copied out of the DLPack views **under the GIL**: the views alias the caller's live, mutable buffers, and ADR-0006 requires conversion to finish before `Python::detach`. The CSR index build that follows runs detached.

2. **Converge all detection routes on SoA `DetectionColumnsInput` → `CocoDetections` (`DetectionColumns` + `CsrIndex`).**
   - `CocoDetections` stores a private `Arc<DetectionColumns>` (compact `IdColumn::Sequential` when no explicit IDs are supplied, `AreaColumn::FromBbox` computing `w * h` inline under default **J3**, dense `image_ids`/`category_ids`/`scores`/`bboxes`, and sparse modality columns allocated only when present) plus two crate-private flat `CsrIndex` structures (an `FxHashMap<K, usize>` slot map over contiguous `offsets` and `indices` vectors, also adopted by `CocoDataset`). Index lookups return `&[usize]`.
   - Row-wise producers (typed `DetectionInput`s, result dicts at the FFI) append through one `DetectionColumnsInput::push`, the single owner of the lazy sparse-column rule.
   - `EvalKernel::build_dt_anns` and `build_dt_anns_into` take `&CocoDetections` directly instead of `&[CocoDetection]`.
   - `result_ingest::exact_int` replaces `f64::fract` with the exact IEEE-754 round-trip `(v as i64) as f64 != v` over `[-2^53, 2^53]`.

3. **Store `EvalGrid` in a category-major `PackedEvalGrid` arena with lazy `PerImageEval` materialization.**
   - `EvalGrid` stores one `PackedCategory` per category `k`: sorted `occupied_images`, `dt_offsets`, a single `dt_scores` buffer shared across all area ranges, `O(1)` `npig_by_area` + `(n_gt, n_unignored)` counts (reconstructing `gt_ignore` losslessly since **A4** stably partitions non-ignored before ignored GTs), and bit-packed `(matched: u16, ignore: u16)` masks per `(detection, area_range)` when `T <= 16` (with a flat boolean fallback for `T > 16`).
   - The per-cell pass short-circuits `n_g == 0 && !build_meta` algebraically and reuses per-worker `MatchScratch` when `n_g > 0 && !build_meta`, eliminating per-cell temporary heap allocations.
   - The pycocotools greedy matcher (quirks **B1**–**B4**, **B6**) exists once: a single ladder, monomorphized over output sinks — the `MatchResult` index matrices behind `EvalImageMeta`, and the two packed flag layouts. A4's stable partition makes **B3** structural (the ignore tail is scanned only without a real match). A randomized test pins every sink against an independent transliteration of `cocoeval.py:268-296`.
   - The packed arena addresses images and detection rows with `u32`. A pass whose image, detection or annotation count exceeds `u32::MAX` is refused up front (`EvalError::InvalidConfig`), so every offset is an exact conversion; chunk concatenation refuses mismatched `(n_a, n_t)` rather than dropping flags.
   - `EvalGrid` holds exactly one representation — the packed arena, or dense cells for grids assembled via `EvalGrid::from_eval_imgs` — and caches nothing beside it. `accumulate_grid` / `accumulate_grid_parallel` consume the arena directly. `EvalGrid::cell` unpacks one cell on demand (`Cow`); `EvalGrid::category_cells(k, a)` walks a category's populated cells in image order, which is what the table builders and LRP use; `EvalGrid::eval_imgs()` builds the dense `[k][a][i]` view per call for the consumers that need all of it (partitioned evaluation, calibration handles, Python `EvalGrid.eval_imgs()`).
   - One `GridPlan` (axes, occupancy, per-pass quirk switches) feeds both the sequential and the parallel evaluator. The `I` axis — images in id-ascending order — is computed once per `CocoDataset` and shared by every grid consumer.
   - `StreamingEvaluator::update` evaluates only the batch's images. The subset pass returns its packed cells with *sparse* metadata and retained IoUs, so an update costs `O(batch)` plus an `O(K log K)` category scaffold (and, on federated LVIS ground truth, one `O(I)` fill of the per-image federated lookup), never an `O(I log I)` sort or a `K·A·I` grid. Snapshots rebuild a packed grid from the populated cells (`O(C log C)`) and accumulate it with the batch kernel; the GT-only overlay for unseen images is itself packed.

## Consequences

- **Positive:** Eliminates `84 MB` Python matrix copies on `coco_inputs_from_columns`, cuts Rust detection storage from `456 MB` (`DetectionInput` + `CocoDetection`) to `84 MB` SoA, collapses `~393,000` index `Vec`s + `7.8M` `PerImageEval` heap blocks to `~400` flat buffers, and makes `accumulate` L1/L2 cache-resident.
- **Negative (0.6.0 breaking change in Rust `vernier-core`):** `CocoDetections::detections()` (`&[CocoDetection]`), `EvalKernel::build_dt_anns*(&[CocoDetection])`, and the raw field `EvalGrid::eval_imgs` become SoA accessors (`CocoDetections::len`, `record_at`, `records`, column slices), `EvalKernel::build_dt_anns*(&CocoDetections)`, and `EvalGrid::eval_imgs(&self) -> Cow<'_, [Option<Box<PerImageEval>>]>` respectively; `EvalGrid::cell` returns `Option<Cow<'_, PerImageEval>>`. Python callers gain `DetectionsBatch` on `DetectionsInput`; existing `bytes`, `Detections`, `ResultAnnotation`, and `DetectionMatrix` inputs continue to work unchanged.
- **Negative:** the `DetectionsBatch` / `(N, 7)` column copies hold the GIL (one streaming pass over the caller's arrays). Releasing it there would race any Python thread writing the same buffers.
- **Neutral (deferred):** the streaming evaluator's long-lived store keeps the ADR-0013 / ADR-0031 `(k, a, i) → PerImageEval` cell format — the distributed partial wire format and the memory-budget accounting are defined over it — so each update still unpacks its batch's cells into that map. Moving the store itself to packed per-batch runs is a separate decision, to be taken with ADR-0031's wire format.

## Implementation (staged)

The decision lands as a series of independently reviewable PRs, so the
non-breaking parts ship in a 0.5.x patch release ahead of the 0.6.0 break:

- **0.5.x (additive / internal):** the `DetectionsBatch` route and the
  adapters that emit it; inline `(N, 7)` id checks; in-place `xyxy`
  widening; seeded flat CSR indexes; the cached grid image order and
  `O(batch)` streaming matching; the matcher's column gather, per-cell
  prefilter and structural B3; grid-level accessors (`accumulate_grid*`,
  `EvalGrid::eval_imgs()`, `category_cells`) so callers can migrate ahead
  of the break; the DETR-300q bench scenario.
- **0.6.0 (breaking):** Structure-of-Arrays `CocoDetections` with columnar
  FFI ingest; the packed `EvalGrid` and its packed accumulation and
  streaming snapshots.

