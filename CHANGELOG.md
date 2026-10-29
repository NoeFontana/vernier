# Changelog

All notable changes to this project will be documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The project graduated out of the 0.0.x line with 0.1.0 — the four
evaluation paradigms (instance, panoptic, semantic, LVIS federated)
plus the LRP / oLRP and detection-calibration diagnostic surfaces are
all wired with strict bit-equal parity against their oracles, the
local bench harness is stable, and the public API surface has held its
shape across the last three patches. Pre-1.0 still means the API can
break between minor versions; the bar moves from "every patch is
exploratory" to "minor bumps signal breakage, patch bumps are
additive / perf / docs".

## [Unreleased]

### Added

- **Multi-image columnar `DetectionsBatch` on `DetectionsInput`** (ADR-0066).
  `vernier.instance.DetectionsBatch` carries whole-dataset or multi-image
  detection columns (`image_ids`, optional per-image `counts`, `boxes`,
  `scores`, `labels`, and optional `rles` / `keypoints`) across every
  `DetectionsInput` entry point (`Evaluator.evaluate`, `evaluate_*_grid`,
  `BackgroundEvaluator.submit`, partitioned evaluation, and TIDE / LRP /
  confusion-matrix / FP-IoU-histogram diagnostics).
- **Direct-to-SoA Rust constructors and packed-grid accumulation** (ADR-0066).
  `CocoDetections::from_columns`, `DetectionColumns`, `DetectionColumnsInput`,
  `CsrIndex`, `CategoryPackedCells`, `PackedEvalGrid`, `MatchScratch`,
  `accumulate_grid`, and `accumulate_grid_parallel` in `vernier-core` (and
  re-exported through `vernier`).

### Changed

- **Breaking (0.6.0) — `vernier.adapters.coco_inputs` and
  `coco_inputs_from_columns` return `DetectionsBatch`** on both `bbox` and
  `segm`, replacing the `(N, 7)` float64 `np.column_stack` copy on `bbox` and
  the per-image `list[Detections]` dict loop on `segm`. Callers forwarding the
  returned `(CocoDataset, DetectionsInput)` pair to `vernier.instance` require
  no code changes.
- **Breaking (0.6.0) — `vernier-core` / `vernier` Rust API:**
  - `CocoDetections` is backed by `Arc<DetectionColumns>` (Structure of Arrays)
    and flat 2-buffer `CsrIndex` tables rather than `Vec<CocoDetection>`; index
    lookups return `&[u32]` row indices, and `records()` returns an exact-size
    iterator yielding `CocoDetection` by value.
  - `EvalGrid.eval_imgs` is now a lazy method `EvalGrid::eval_imgs(&self) -> &[Option<Box<PerImageEval>>]`
    backed by `OnceLock` over `PackedEvalGrid` instead of an eager struct field.
  - `EvalKernel::build_dt_anns` and `build_dt_anns_into` take `&CocoDetections`
    directly instead of `&[&CocoDetection]`.

### Performance

- **6.8×–8.2× faster end-to-end RF-DETR validation (`5,000` images × `300`
  queries = `1.5M` detections, `80` classes)** (ADR-0066):
  - `coco_inputs_from_columns` drops from `157 ms` to `2.4 ms` (`float64 xywh`)
    / `24.0 ms` (`float32 xyxy → float64 xywh` in-place widening).
  - FFI detection routes (`DetectionsBatch`, `Detections`, `ResultAnnotation`,
    and `(N, 7)` `DetectionMatrix`) write directly into `DetectionColumnsInput`
    off the GIL (`py.detach`), with branchless inline SSE2 `cvttsd2si64`
    round-trip integer checks replacing `libm::trunc` calls on `(N, 7)` IDs.
  - `match_image_into_packed` reuses `MatchScratch` across cells and short-circuits
    empty-GT cells (`n_g == 0`, ~93% of unthresholded 300-query DETR cells) on
    infallible kernels (`BboxIou`).
  - Category-major `PackedEvalGrid` + L1/L2-resident `accumulate_grid_parallel`
    cut 4-thread PR accumulation from `965 ms` to `67 ms` (`14.3×`) and fused
    4-thread `evaluate_bbox_summary` from `1,454 ms` to `211.5 ms` (`6.9×`).
  - `StreamingEvaluator::update_with` now evaluates only the `O(batch_images)`
    subset present in each incoming batch rather than scanning all GT images.

## [0.5.4] - 2026-09-24

### Added

- **The parsed pair on every instance surface** (ADR-0064). TIDE
  (`error_decomposition`), LRP (`optimal_lrp`, including its
  `manifest=` form), `confusion_matrix`, `fp_iou_histogram` and
  `Evaluator.evaluate`'s `tables=` / `manifest=` paths now take a
  `CocoDataset` handle for `gt` and the whole `DetectionsInput` union
  for `dt`, like `Evaluator.evaluate` and `evaluate_*_grid`. So the
  `(CocoDataset, DetectionsInput)` pair from `coco_inputs` reaches every
  instance surface without a COCO file. The four standalone diagnostics
  gain `cast_inputs` (default `False`, as on `Evaluator`).

  Those four refuse LVIS federated ground truth (a
  `CocoDataset.from_lvis_json` handle) with `NotImplementedError`: they
  have no federated disposition.

### Fixed

- `BackgroundEvaluator` / `Evaluator.background` (and the core
  `StreamingEvaluator`) now apply the ADR-0026 AC2 per-image detection
  cap to a federated `CocoDataset`, per `submit()` (ADR-0065). This is
  exact because an image's detections must arrive in one batch. A
  streamed LVIS evaluation previously matched uncapped and could
  disagree with the batch evaluator and lvis-api.
- `Evaluator.evaluate(lvis_handle, dt)` — neither `tables=` nor
  `manifest=` — now applies the ADR-0026 AC2 per-image detection cap, as
  the grid path always has. It matched a federated handle untrimmed, so
  an image carrying more detections than the largest `max_dets` scored
  differently from `tables=`, `manifest=`, `evaluate_*_grid` and
  lvis-api.

### Performance

- The segm / boundary diagnostics reuse a `CocoDataset` handle's GT
  caches, shared with `Evaluator.evaluate`, and TIDE shares one cache
  across its eight passes on either `gt` spelling. On 500 val2017
  images, boundary TIDE runs ~27% faster, and boundary LRP / confusion
  matrix / FP-IoU histogram ~33% faster on a warm handle. Results are
  bit-identical. The `vernier-core` per-kernel diagnostic wrappers
  switch to the scratch-reusing mask kernels too.
- The `gt` bytes path borrows the payload across the GIL release rather
  than copying it (~20 MB per call on val2017), on `evaluate_*_summary`
  as well as the diagnostics. `BackgroundEvaluator` now parses bytes GT
  off the GIL.
- `manifest=` (partitioned AP and LRP) validates the manifest before
  running the evaluation or reading the detections.

## Released versions

Frozen history, newest first, in [`docs/changelog/`](docs/changelog/); leave it
out of searches and sweeps.

- [0.5.3](docs/changelog/0.5.3.md) — 2026-09-23
- [0.5.2](docs/changelog/0.5.2.md) — 2026-09-22
- [0.5.1](docs/changelog/0.5.1.md) — 2026-09-19
- [0.5.0](docs/changelog/0.5.0.md) — 2026-09-19
- [0.4.1](docs/changelog/0.4.1.md) — 2026-09-18
- [0.4.0](docs/changelog/0.4.0.md) — 2026-09-18
- [0.3.0](docs/changelog/0.3.0.md) — 2026-09-16
- [0.2.0](docs/changelog/0.2.0.md) — 2026-06-09
- [0.1.0](docs/changelog/0.1.0.md) — 2026-05-19
- [0.0.4](docs/changelog/0.0.4.md) — 2026-05-16
- [0.0.3](docs/changelog/0.0.3.md) — 2026-05-15
- [0.0.2](docs/changelog/0.0.2.md) — 2026-05-12
- [0.0.1](docs/changelog/0.0.1.md) — 2026-04-30

[Unreleased]: https://github.com/NoeFontana/vernier/compare/v0.5.4...HEAD
[0.5.4]: https://github.com/NoeFontana/vernier/compare/v0.5.3...v0.5.4
