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
  `scores`, `labels`, and `rles` / `keypoints` where the IoU type needs
  them) across every `DetectionsInput` entry point (`Evaluator.evaluate`,
  `evaluate_*_grid`, `BackgroundEvaluator.submit`, partitioned evaluation,
  and TIDE / LRP / confusion-matrix / FP-IoU-histogram diagnostics). Like
  the per-image `Detections` dict it generalizes, it requires `rles` under
  `iou_type="segm"` / `"boundary"`: quirk **J2**'s bbox-rectangle fallback
  stays with the routes that mirror a `loadRes` file (JSON bytes, result
  dicts, the `(N, 7)` matrix).
- **Columnar Rust ingest and packed-grid accumulation** (ADR-0066), in
  `vernier-core` and through the `vernier::instance` facade:
  `CocoDetections::from_columns` / `from_columns_with_area` over
  `dataset::DetectionColumnsInput` (with a `with_capacity` / `push` row
  builder that keeps optional columns sparse), `accumulate::accumulate_grid`
  and `accumulate::accumulate_grid_parallel`, and
  `EvalGrid::category_cells(k, a)`, which walks one category's populated
  cells in image order.

### Changed

- **Breaking (0.6.0) — `vernier.adapters.coco_inputs` and
  `coco_inputs_from_columns` return `DetectionsBatch`** on both `bbox` and
  `segm`, replacing the `(N, 7)` float64 `np.column_stack` copy on `bbox` and
  the per-image `list[Detections]` dict loop on `segm`. Callers forwarding the
  returned `(CocoDataset, DetectionsInput)` pair to `vernier.instance` require
  no code changes. Predictions without masks beside masked ground truth are
  still refused by a `segm` pass, as they were before.
- **Breaking (0.6.0) — `vernier-core` / `vernier` Rust API:**
  - `CocoDetections` is backed by private Structure-of-Arrays columns and
    flat CSR indexes rather than `Vec<CocoDetection>`. `detections()` is
    replaced by `len`, `record_at`, `records()` (an exact-size iterator
    yielding `CocoDetection` by value) and per-column accessors
    (`image_ids`, `category_ids`, `scores`, `bboxes`, `area_at`, …);
    `indices_for` / `indices_for_image` still return `&[usize]`.
  - `EvalKernel::build_dt_anns` / `build_dt_anns_into` take
    `&CocoDetections` instead of `&[CocoDetection]`.
  - `EvalGrid` stores its cells in a packed category-major arena. The
    `eval_imgs` field is gone: `EvalGrid::eval_imgs()` returns
    `Cow<'_, [Option<Box<PerImageEval>>]>`, built per call (and not
    cached) for an evaluator-produced grid; `EvalGrid::cell` returns
    `Option<Cow<'_, PerImageEval>>`, unpacking just that cell.

### Performance

- **Faster end-to-end evaluation on dense-query detectors** (ADR-0066),
  measured on RF-DETR validation (`5,000` images × `300` queries = `1.5M`
  detections, `80` classes):
  - `coco_inputs_from_columns` drops from `157 ms` to `2.4 ms` (`float64 xywh`)
    / `24.0 ms` (`float32 xyxy → float64 xywh` in-place widening).
  - The `DetectionsBatch` and `(N, 7)` routes copy each column once,
    straight into `DetectionColumnsInput` (under the GIL, per ADR-0006,
    since the source buffers are the caller's), and the `(N, 7)` id columns
    are round-trip checked with an inline integer conversion instead of a
    `libm::trunc` call; the CSR index build then runs detached.
  - Matching reuses per-worker scratch across cells and short-circuits
    empty-GT cells (`n_g == 0`, ~93% of unthresholded 300-query DETR cells),
    with one greedy-matching implementation shared by every output layout.
  - The packed `EvalGrid` + `accumulate_grid_parallel` cut 4-thread PR
    accumulation from `965 ms` to `67 ms` and fused 4-thread
    `evaluate_bbox_summary` from `1,454 ms` to `211.5 ms`. Diagnostics that
    read cells (`tables`, LRP) walk only populated cells, and nothing
    dense is cached alongside the packed grid.
  - `StreamingEvaluator::update` evaluates only the batch's images, with no
    per-call `O(I log I)` image sort (the order is cached on the dataset)
    and no `K·A·I` metadata grid; snapshots accumulate a packed grid built
    from the populated cells, and the GT-only overlay for unseen images is
    packed rather than a dense `K·A·I` grid.

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
