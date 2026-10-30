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

## [0.6.0] - 2026-10-30

### Changed (BREAKING - pre-1.0)

- **`CocoDetections` is Structure-of-Arrays** (ADR-0066, Rust API only;
  the Python API is unchanged). Detections live in private dense columns.
  Ids are derived as `1..=N` under quirk **J1** unless supplied. Areas are
  `w·h` from the bbox under **J3** unless supplied. The segmentation and
  keypoint columns are allocated only when present.
  `CocoDetections::detections() -> &[CocoDetection]` is removed; use
  `len`, `record_at`, `records()` (an exact-size iterator of
  `CocoDetection` by value) and the column accessors `image_ids`,
  `category_ids`, `scores`, `bboxes`, `area_at`, `bbox_at` and `id_at`.
  `EvalKernel::build_dt_anns` / `build_dt_anns_into` take
  `&CocoDetections` instead of `&[CocoDetection]`.
- **`EvalGrid` storage is private** (ADR-0066, Rust API only). Evaluators
  emit a packed category-major arena instead of a dense `K·A·I` grid of
  boxed cells. `EvalGrid::eval_imgs` is a method returning
  `Cow<[Option<Box<PerImageEval>>]>` (the dense view, built on demand),
  not a field; `EvalGrid::cell` returns `Option<Cow<PerImageEval>>`; an
  `EvalGrid` is built with `EvalGrid::from_eval_imgs`, not a struct
  literal; its axis sizes are read through `n_categories()`,
  `n_area_ranges()` and `n_images()` instead of public fields.
  `accumulate_grid`, `category_cells` and `into_eval_imgs` work on either
  storage.

### Added

- `CocoDetections::from_columns` / `from_columns_with_area` and
  `DetectionColumnsInput` (`with_capacity`, `push`). Every ingest route
  converges on this one constructor, which owns **J1**, **J3**,
  **E2**/**J4** and non-finite score rejection.
- `EvalKernel::dt_anns_can_fail`, defaulting to `true`. A kernel whose
  detection annotations cannot fail to build (`BboxIou`) lets empty-GT
  cells skip building them.
- `DetectionFields`, `KernelKind::detection_fields`,
  `EvalKernel::DETECTION_FIELDS` and `CocoDetections::from_json_bytes_with`
  / `from_json_bytes_parallel_with`: a detection JSON load can skip the
  payloads its kernel never reads; `ParsedDetections::<K>::from_json_bytes`
  does so for `K`.

### Performance

- Detection ingest writes straight into columns, with no per-detection
  intermediate. 1.5M rows, input build included: 311.6 → 80.7 ms (3.9×).
  The `Detections` and `DetectionsBatch` FFI routes share one column
  extractor.
- **Packed evaluation grid.** `accumulate` sweeps the packed arena
  directly, empty-GT cells skip the matcher, and the parallel evaluator
  fans out over image chunks of each category, sized to the thread pool.
  On a DETR-shaped grid (500 images × 80 categories × 300 queries),
  evaluate + accumulate go from 171 to 39.6 ms on one thread (4.3×) and
  from 76.6 to 8.9 ms on eight (8.6×). COCO-shaped cells are 5.0× faster;
  dense single-class cells are flat. End to end from Python,
  `evaluate_bbox_summary` on 1.5M detections over 5,000 images goes from
  2.55 s to 0.69 s on one thread and from 1.60 s to 0.19 s on eight,
  where 0.5.5 had stopped scaling past four. Bit-identical output.
- **`accumulate` sweeps true positives, not the whole PR curve.** The
  precision envelope can only peak at a true positive, and a recall
  threshold lands on a fixed true-positive count, so each lane records
  its true positives in one pass over the 16-bit flag words (all
  thresholds at once) and reads the 101 recall points off them: no
  per-detection division, curve stores or binary searches. The dense
  per-cell accumulator (partitioned evaluation, non-packed grids) runs
  the same sweep (ADR-0068). One thread, COCO val2017: RF-DETR Nano 132 → 52 ms
  (−60%), jittered 37.5 → 17 ms (−55%); a two-slice partitioned
  evaluation of RF-DETR Nano 1056 → 726 ms (−31%). Bit-identical output.
- **`segmentation` parses without serde's untagged buffering.** The
  field (and the `iscrowd` / `ignore` flags) dispatch on the JSON shape
  instead of `#[serde(untagged)]`, which copied every polygon
  coordinate and RLE string into an intermediate tree before trying
  each variant. COCO val2017 GT parse: 95 → 57 ms (−40%), on every IoU
  type. RF-DETR Seg-Nano results, one thread, end to end: bbox
  768 → 587 ms (−24%), segm 1941 → 1783 ms (−8%). Every shape a COCO
  file carries reads as before, and byte-string RLE counts (from a format
  with a byte type) read as the compressed form. A malformed
  `segmentation` now reports what it expected instead of "data did not
  match any variant of untagged enum", and the positional-array RLE
  (`[[h, w], counts]`) that serde's derive admitted is refused, as
  pycocotools reads any list as polygons.
- **Polygon rasterizer.** Boundary crossings are generated edge by edge
  instead of densifying every edge into a point buffer and scanning it:
  an x-major edge visits only the one step in five that can land on a
  pixel boundary (quirk H5), with the C's own per-step expressions. The
  H4/H5 downsample uses exact integer forms instead of `fract()` /
  `ceil()`, which the x86-64 baseline the wheel targets lowers to libm
  calls. One thread, COCO val2017, evaluate stage: jittered segm −37%,
  RF-DETR Seg-Nano segm −17%, jittered boundary −6%. Bit-identical
  output.
- **Detection parsing skips the fields the kernel never reads.** bbox
  skips `segmentation` and the keypoint fields, segm / boundary skip the
  keypoint fields, keypoints skips `segmentation` — on every route: grid,
  summary, streaming, background and diagnostics, JSON bytes or result
  dicts alike. A results file that also carries masks no longer pays to
  parse them under bbox: RF-DETR Seg-Nano on COCO val2017, one thread,
  753 → 525 ms end to end (−30%). A malformed value in a skipped field no
  longer fails the evaluation, matching pycocotools, which never reads it
  there, and both routes accept the same payloads (ADR-0057).
- **Detection JSON parses in parallel when the GT is a `CocoDataset`,
  in one pool per call.** The grid and summary entry points that take a
  pre-parsed dataset parsed the detection payload on one thread at any
  `num_threads`. Each instance entry point now installs one scoped pool
  that its parse, evaluate and accumulate stages share, instead of each
  stage building and joining its own. RF-DETR Nano on COCO val2017,
  eight threads, parse + evaluate: 252 → 133 ms (−47%); Seg-Nano segm
  722 → 587 ms (−19%).
- **Faster RLE decoding and mask walks.** Compressed counts decode on a
  dedicated path for one- and two-char runs (99% of runs on real masks)
  into a reused buffer copied once into the mask's shared storage, and
  `decode_area` sums a compressed mask's area without materializing its
  counts. The walks that derive a mask's runs, area, bbox and raster take
  column and row by an exact multiply-shift instead of 64-bit divisions.
  430k RF-DETR Seg-Nano masks, one thread: decode 222 → 173 ms (−22%),
  walk 176 → 136 ms (−23%); Seg-Nano segm evaluate −6.6%. Malformed
  strings still fail with the same error.
- **Parallel evaluation balances by work.** Each category's images are
  cut into tasks of about equal estimated cost instead of equal image
  counts — a category is weighed from the ground-truth and detection
  totals the occupancy pass now counts on its way — and every task is
  its own stealable job running on its worker's reused scratch. Before,
  the `person` chunks (30–40 ms each against a few ms for most
  categories) all sat at the front of the order and one worker ran them
  back to back while the rest idled. COCO val2017, eight threads, parse +
  evaluate: jittered segm −30%, RF-DETR Seg-Nano segm −15%; Objects365
  unchanged. Bit-identical output.

- **Detection JSON splits in parallel.** The parallel loader no longer
  walks the whole array serially to find every element before parsing:
  it cuts the array into segments and scans each once, in parallel, for
  both possible string states — an escaped quote is told apart without
  knowing the state — then composes the true state to find one element
  start and count per run, exactly. Runs parse with strict separator
  checks, then their columns concatenate and the detection indexes build
  concurrently. COCO val2017, parse + evaluate, eight / two threads:
  RF-DETR Seg-Nano bbox −43% / −16%, Nano bbox −24% / −10%, Seg-Nano
  segm −20% / −5%. Bit-identical output.

## Released versions

Frozen history, newest first, in [`docs/changelog/`](docs/changelog/); leave it
out of searches and sweeps.

- [0.5.5](docs/changelog/0.5.5.md) — 2026-10-29
- [0.5.4](docs/changelog/0.5.4.md) — 2026-09-24
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

[Unreleased]: https://github.com/NoeFontana/vernier/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/NoeFontana/vernier/compare/v0.5.5...v0.6.0
