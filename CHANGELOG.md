# Changelog

All notable changes to this project will be documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning stays on the 0.0.x release line until the core and extended
feature set is complete; moving to 0.1.0+ is a deliberate later decision.

## [Unreleased]

## [0.0.3] — 2026-05-15

This is the diagnostic-surfaces and scenario-slicing release: instance
gains an oLRP error decomposition (Oksuz et al.), a detection-family
calibration summarizer (ECE / MCE / reliability), and a manifest-driven
slice-and-aggregate lane that runs one matching pass across N scenario
cells. Panoptic picks up boundary PQ. No paradigm shifts, no
crates.io additions — every kernel slots into the existing
vernier-core / vernier-panoptic / vernier-semantic surface.

### Added

- **LRP / oLRP error decomposition** (ADR-0043, ADR-0044, ADR-0045) —
  Oksuz et al. (ECCV 2018 / TPAMI 2021) Localization Recall Precision
  as an opt-in metric alongside AP. `vernier.instance.optimal_lrp(gt,
  dt, iou=Bbox()|Segm()|Boundary()|Keypoints())` decomposes detection
  performance into `oLRP_Loc + oLRP_FP + oLRP_FN`, minimised over a
  per-class confidence threshold `tau`. CLI gains `--metric {ap,olrp}`
  with `ap` preserving the existing headline-table contract. The Rust
  core lives in `crates/vernier-core/src/lrp/`; the ADR-0005 firewall
  is held (no edits to `matching.rs` / `accumulate.rs` / `evaluate.rs`).
  Pure-NumPy oracle is the correctness contract (ADR-0043);
  `kemaloksuz/LRP-Error` is an opt-in tripwire, not a parity gate.
  `vernier.panoptic.optimal_lrp` is a typed `NotImplementedError` stub
  — panoptic predictions carry no per-segment score so the tau sweep
  has nothing to scan; extension is a follow-up ADR.
- **Boundary Panoptic Quality** (ADR-0025 §Z1/Z2 amendment) —
  `PanopticEvaluator(boundary=True, dilation_ratio=0.02)` now ships
  under both `parity_mode="strict"` (bit-exact reproduction of
  `bowenc0221/boundary-iou-api`'s `coco_panoptic_api/evaluation.py`
  at SHA `37d25586a677`) and `parity_mode="corrected"` (deterministic,
  snapshot-based; segment-id-sorted iteration). Composition is
  `iou = min(mask_iou, boundary_iou)` — identical to the instance
  Boundary case (the prior Q3 row of `boundary-iou-quirks.md` had
  miscalled this; corrected in the same amendment). FN/FP attribution
  is unchanged; U6/U7/V1-V7/W1/W7 stand. The streaming runner threads
  boundary state per image with `BoundaryScratch` reuse, and
  distributed-eval partials hash the `dilation_ratio` into
  `params_hash` so silent boundary/instance partial mixing is rejected
  at envelope-validation time. No `FORMAT_VERSION` bump. Cityscapes
  panoptic (Z3) remains deferred.
- **Detection-family calibration summarizer** (ADR-0018) —
  ECE / MCE / reliability table for bbox / segm / boundary /
  keypoints. Opt-in via `Evaluator.evaluate(..., calibration=True)`;
  the lazy `result.calibration(iou=..., n_bins=15,
  binning="quantile", min_score=0.05, per_class=False, ...)` re-fold
  returns a `vernier.calibration.CalibrationResult` (polars
  `reliability` / `per_class` plus scalar `ece` / `mce`). Re-folding
  with different params does not re-run matching. Streaming pairing:
  `BackgroundEvaluator.finalize_with_cells()` plus the
  `vernier.calibration.StreamingSnapshot` wrapper. Clean-room NumPy
  oracle is the correctness contract; 16/16 parity bit-equal at
  strict mode. Panoptic and semantic calibration are deferred
  (data-model prerequisites per the ADR's per-paradigm shape map).
- **Slice-and-aggregate** (ADR-0046) — manifest-driven scenario
  slicing across all three paradigms plus the `vernier aggregate`
  fan-in verb. Python:
  `Evaluator.evaluate(..., manifest=..., cross_axes=...)` accepts a
  dict, JSON / CSV path, or Arrow PyCapsule manifest and returns
  `EvalResult.slices` as a polars `DataFrame` (one row per
  `(axis, value)` cell). CLI: `vernier eval --manifest weather.json
  [--cross weather,time_of_day] [--label NAME] [--metric {ap,olrp}]`
  emits a v2 envelope; un-partitioned `vernier eval` keeps emitting
  v1 verbatim. `vernier.aggregate(results, manifest, *,
  baseline=None, metric=None)` and `vernier aggregate result1.json
  result2.json --manifest runs.json --baseline clean` fan N runs
  into a comparative table with `<metric>` (mPC) and
  `<metric>__rpc` (rPC) columns when `--baseline` is set. The
  `tables=` + `manifest=` cross product is a deliberate non-feature
  with a client-side recipe at
  [`docs/how-to/per-class-by-slice.md`](docs/how-to/per-class-by-slice.md).
  New reference schemas: [`manifest-schema.md`](docs/reference/manifest-schema.md),
  [`aggregate-schema.md`](docs/reference/aggregate-schema.md).

## [0.0.2] — 2026-05-12

This is the three-paradigm release: instance gains panoptic and semantic
siblings, distributed eval lands across all three, and the bench harness
brings real-model + alternatives numbers to the docs site. Two new
crates ship to crates.io (`vernier-panoptic`, `vernier-semantic`) plus
the `vernier-partial` leaf that holds the shared partial wire envelope.

### Added

- **Distributed-eval entry points on `Evaluator`** (ADR-0035) — each
  paradigm's public `Evaluator` gains
  `evaluate_to_partial(..., *, rank_id) -> bytes` and a
  classmethod `from_partials(...) -> Summary`. Per-paradigm shapes:
  instance takes JSON bytes, semantic takes `Dataset/Predictions`,
  panoptic takes per-image tuples + `categories=` (the one
  asymmetry — `PanopticDataset` doesn't yet expose per-image
  accessors; closing that gap is a follow-up). The streaming
  substrate, the `vernier-partial` wire format, `FORMAT_VERSION`,
  partition-disjointness invariant, and the five paradigm-shared
  `Partial*` exception classes are all unchanged. The same DDP
  recipe works on instance, semantic, and panoptic.
- **Distributed evaluation wire format** (ADR-0031, ADR-0032) — new
  `vernier-partial` workspace crate holds the shared partial-envelope
  (magic + `FORMAT_VERSION` + framing + the five `Partial*` typed
  errors) used by all three paradigms. `FORMAT_VERSION` is a 1→2
  hard break (pre-1.0 policy). Cross-paradigm merge is structurally
  rejected (paradigm tag in the envelope). Determinism contract is
  paradigm-specific: instance preserves bit-exactness, semantic
  preserves it for any partition, panoptic only when the partition
  order matches the original GT order. `BackgroundEvaluator` reuses
  the same substrate via `finalize_to_partial`.

### Changed

- **Public-surface consolidation** (ADR-0035, supersedes the public
  `StreamingEvaluator` portion of ADR-0013; amends ADR-0014, ADR-0031,
  ADR-0032). Each paradigm now exposes two classes: `Evaluator`
  (frozen config dataclass; batch + DDP entry points) and
  `BackgroundEvaluator` (in-training entry point; `submit` /
  `finalize` / `finalize_with_tables` / `finalize_to_partial` /
  context manager). The streaming pyclasses are removed from Python
  entirely; the Rust substrate stays and is reachable via new
  PyO3 functions (`evaluate_*_to_partial`, `merge_*_partials`) and
  via `BackgroundEvaluator`.

### Removed

- `vernier.{instance,panoptic,semantic}.StreamingEvaluator` — the
  three streaming pyclasses are removed from Python entirely. They no
  longer appear on `vernier._core`, on any paradigm namespace, or
  under a `vernier._impl` shim. The Rust streaming substrate
  (`vernier_core::stream::StreamingEvaluator<K>`,
  `StreamingPanopticEvaluator`, `StreamingSemanticEvaluator`) remains
  as the implementation behind the new
  `evaluate_*_to_partial` / `merge_*_partials` PyO3 functions and
  `BackgroundEvaluator`'s worker. No public deprecation shim — pre-1.0
  hard break.
- `Evaluator.stream(...)` factory on `vernier.{panoptic,semantic}` —
  removed alongside the public streaming class. Use
  `BackgroundEvaluator(...)` directly, or `Evaluator.evaluate_to_partial`
  / `Evaluator.from_partials` for DDP.
- `StreamingEvaluator.snapshot(running=True)` and its Rust-side
  `snapshot_running()` method — the biased fast path that ADR-0013
  itself flagged as inappropriate for quality gates.
- `StreamingEvaluator.checkpoint()` / `restore()` — these were
  `NotImplemented` thin wrappers around `snapshot_to_partial` /
  `from_partials`. The persistence story is now exclusively
  `evaluate_to_partial` → store bytes → `from_partials` on resume.
- `BackgroundEvaluator.snapshot()`, `snapshot(peek=True)`,
  `snapshot_with_tables()`, and the non-finalize `to_partial()` on
  all three paradigms. Public surface is the consuming
  `finalize` / `finalize_with_tables` / `finalize_to_partial` only.
- `BackgroundPanopticEvaluator.from_partials` /
  `BackgroundSemanticEvaluator.from_partials` — vestigial (return-
  type bug carried them; no caller used them).

- **Semantic-segmentation user docs** (ADR-0028 PR-B10) — three new
  pages in `docs/`: `migrate/from-mmsegmentation.md` (semantic-side
  migration recipe with preset / streaming / NaN-vs-0.0 /
  binary-mask coverage), and `explanation/three-paradigms.md` (paradigm
  picker — when to reach for instance vs panoptic vs semantic, why
  they're sibling submodules rather than a single evaluator with a
  knob). README updated to
  feature the three-paradigm surface in a top-level section
  alongside the install commands; `mkdocs.yml` nav surfaces both
  new pages plus the previously-orphaned panoptic migration
  guide.
- **Semantic-segmentation streaming evaluator** (ADR-0028 PR-B9
  partial — streaming only; Breakdown / result-tables follow-ups
  scoped to a future PR). New
  `vernier_semantic::StreamingSemanticEvaluator` is a flat
  `O(n_classes²)` accumulator over `ConfusionMatrix`: `update(image_id,
  gt, dt)` folds via the same `accumulate_confusion` kernel the batch
  path uses; `snapshot()` is constant-time relative to image count
  (per ADR-0013, no fast-vs-running mode distinction needed). FFI
  pyclass `vernier._core.StreamingSemanticEvaluator` is registered
  on the module; the Python `Evaluator.stream(n_classes,
  ignore_label=None)` factory returns a fresh streaming evaluator
  carrying the parent's `parity_mode`. **Load-bearing invariant**
  (pinned by `tests/python/test_semantic_streaming.py::test_streaming_finalize_bit_equals_batch_evaluate`):
  streaming `finalize()` is bit-equal to batch `evaluate(...)` over
  the same images on f64 outputs. 10 new Python tests + 7 new Rust
  tests; total workspace 472 Rust + 376 Python tests pass.
- **Semantic-segmentation Python wrapper + per-dataset presets**
  (ADR-0028 PR-B5) — new `vernier.semantic` submodule (per ADR-0029)
  exposing `Dataset` / `Predictions` / `Evaluator` frozen dataclasses
  plus `Summary` / `ClassSemanticStats` / `ConfusionMatrix`
  re-exports of the FFI pyclasses (under their unprefixed names).
  `Dataset.from_arrays` and `Predictions.from_arrays` accept any
  unsigned-integer dtype; the wrapper preserves the input dtype and
  the FFI/kernel walks at native dtype (since ADR-0037).
  `Dataset.from_files` / `Predictions.from_files` decode single-
  channel PNG label maps via lazy-imported Pillow (raises a
  structured `ImportError` if Pillow is missing); RGB-encoded panoptic
  PNGs are rejected with a typed message pointing at
  `vernier.panoptic.Dataset`. `Predictions.from_binary_masks`
  implements the **AN2** per-class binary-mask merge with explicit
  `merge ∈ {"argmax", "first", "highest_class_id"}` selector and
  `unlabeled_class` parameter (quirks **AN3**, **AN4**). Per-dataset
  presets `Dataset.cityscapes` / `ade20k` / `pascal_voc` bake the
  canonical `(n_classes, ignore_label)` constants from
  `vernier_semantic::parity::*`. 23 new Python tests cover the
  wrapper round-trip, dtype handling, ignore-label / label-remap
  propagation, binary-mask merge rules, RGB rejection, and
  end-to-end PNG decode + evaluate.
- **Semantic-segmentation FFI surface** (ADR-0028 PR-B4) —
  `vernier._core.evaluate_semantic_from_arrays(gt_label_maps,
  dt_label_maps, n_classes, parity_mode, *, ignore_label=None,
  label_remap=None)` is the load-bearing pyfunction that drives the
  Rust kernel + summarize pass under `py.detach` (ADR-0006). Inputs
  are dicts mapping image_id (int) → 2-D `numpy.ndarray` of dtype
  `uint32`. New pyclasses `SemanticSummary`, `ClassSemanticStats`,
  `ConfusionMatrix` expose the per-class and global metrics; the
  confusion matrix is materialized as a 2-D `numpy.uint64` array
  via `ConfusionMatrix.counts()` (ADR-0028 §F1 first-class output).
  GT image-id ordering is sorted for deterministic accumulation
  (quirk **AM5** aligned). `label_remap` is pre-applied to DT
  buffers at the FFI boundary (quirk **AK2**) so the hot kernel
  loop avoids per-pixel dict lookups. PNG-decode (`from_files`) and
  binary-mask (`from_binary_masks`) variants land in PR-B5
  alongside the per-dataset preset constructors that drive them.
  14 Python smoke tests pass; full workspace 465 Rust + 343 Python
  green.
- **Semantic-segmentation kernel + summarize** (ADR-0028 PR-B3) —
  `vernier_semantic::kernel::accumulate_confusion` per-image
  histogram fold (one pass over flattened `(H, W)` slices into a
  `u64` `(n_classes, n_classes)` matrix; ignore-label mask before
  the bincount per quirk **AJ2**; out-of-range DT silent-skip per
  **AI4** strict-MS path). `ConfusionMatrix` is a flat-`Vec<u64>`
  row-major shape that doubles as the FFI `(N, N)` numpy-view
  source. `vernier_semantic::summarize::summarize` derives the
  seven headline outputs (mIoU, FWIoU, pixel accuracy, mean
  accuracy, per-class IoU/accuracy/precision, plus the confusion
  matrix as a first-class output per **AL8**). `parity_mode`
  selects NaN vs. 0.0 for zero-support per-class entries (quirk
  **AL2**); means skip zero-support classes regardless of mode
  (**AL3**, mirroring panopticapi **W2** and LVIS **AB3**). 16
  unit tests (kernel + summarize) on hand-computed fixtures, all
  pass in `--release` and debug. No SIMD per ADR-0028 §"Numerical
  layout" — the kernel is integer/memory-bandwidth bound. Dataset
  constructors and FFI surface land in PR-B5 / PR-B4 respectively.
- **Semantic-segmentation crate scaffold** (ADR-0028 PR-B2) — new
  workspace member `crates/vernier-semantic/` with `Cargo.toml` /
  `lib.rs` / `error.rs` / `parity.rs`. Re-exports
  `vernier_core::parity::ParityMode` per ADR-0028 §"Workspace and
  dependency direction" — the first dep-edge asymmetry vs.
  `vernier-panoptic ⊥ vernier-core`, justified by concrete reuse.
  Pins the per-dataset ignore-label conventions
  (`CITYSCAPES_IGNORE_LABEL=255`, `ADE20K_IGNORE_LABEL=0`,
  `PASCAL_VOC_IGNORE_LABEL=255`), class counts, and
  `SEMANTIC_PARITY_EPS` placeholder. `SemanticError` enum surfaces
  the corrected-disposition rows (AI3, AI4, AI6, AM1, AJ5) at the
  dataset-constructor boundary. The kernel, summarize, dataset, and
  FFI surfaces land in subsequent PRs (PR-B3..PR-B5).
- **User-parametrizable evaluation grids** (ADR-0040, ADR-0041,
  ADR-0042) — each paradigm's `Evaluator` accepts a structured
  config object describing the slice/aggregation surface (instance:
  IoU thresholds, area buckets, max-dets, per-class filter; semantic:
  `class_filter` + `class_grouping`; panoptic: things/stuff split
  override + per-class filter). The instance kernel grew the
  `Breakdown` axis abstraction (ADR-0039 Phase 2A/2B): area buckets,
  class groups, and CategoryFilter compose orthogonally; cells whose
  combined `(IoU × area × class)` shape is empty short-circuit. New
  `InvalidEvalParams` exception hierarchy (ADR-0039 Phase 1) replaces
  the prior assert-based validation with structured Python errors
  pointing at the offending param. Defaults reproduce the canonical
  COCO / LVIS / panoptic / mIoU shapes — opt in only when you need
  custom slicing.
- **TIDE error decomposition** (ADR-0022, weeks 1–5) — new
  `vernier.instance.tide` module returning the canonical six-bin
  decomposition (`Cls`, `Loc`, `Both`, `Dupe`, `Bkg`, `Miss`) for
  bbox / segm / boundary IoU kinds. Public Python surface +
  debugging tutorial under `docs/explanation/`; per-image
  confusion-matrix capability; FP-IoU histogram extractor + CLI for
  cross-model `t_b` ratification rounds. Validated against a numpy
  oracle on six synthetic fixtures; rf-detr-anchored real-model
  harness drives the COCO val2017 cross-check. The bbox `t_b` row
  in ADR-0022 is ratified; segm + boundary `t_b` rows remain
  tentative (ratification pending the val2017 cross-model run).
- **Result tables — opt-in Arrow surface** (ADR-0019, ADR-0038) —
  `Evaluator.evaluate(..., tables=[...])` materializes per-detection,
  per-pair, and per-class rows as zero-copy Arrow record batches
  (PyArrow / pandas / polars zero-copy import). Tables stream out of
  `BackgroundEvaluator` and `StreamingEvaluator` as well. Per-class
  tables now extend to panoptic and semantic (ADR-0038): both
  paradigms expose `per_class` schemas via the same
  `RequestedTables` mechanism.
- **Compressed-RLE + 2-D bitmask ingest on `Detections.rles`**
  (ADR-0030) — instance accepts pycocotools-encoded compressed-RLE
  bytes and 2-D `numpy.bool_`/`uint8` bitmasks alongside the
  pre-existing decoded-RLE tuple form. The FFI path is zero-copy
  via DLPack for array inputs, decoded-once for byte inputs, and
  routes through `vernier_mask::Rle::from_*` constructors so all
  three forms produce the same `Rle` representation downstream.
- **Semantic uint8 / uint16 / uint32 ingest + fused PNG decode**
  (ADR-0037) — `Dataset.from_arrays` / `Predictions.from_arrays`
  accept any of the three unsigned-integer dtypes; the kernel walks
  at the input dtype, so uint8 from a torch tensor avoids the 4×
  upcast earlier wheel versions paid. New `Evaluator.evaluate_from_pngs`
  fuses libpng decode + label-map fold in Rust under `py.detach`,
  eliminating the per-image NumPy round-trip; `submit_png` is the
  matching streaming entry point on `BackgroundEvaluator`. The
  semantic kernel was generalized over a `ClassId` trait so the
  fused path is monomorphized per dtype.
- **`BackgroundEvaluator` generalized to semantic + panoptic**
  (ADR-0014 follow-up) — the in-training entry point now exists for
  all three paradigms with the same shape (`submit` / `finalize` /
  `finalize_with_tables` / `finalize_to_partial` / context manager,
  bounded queue + worker-thread offload + soft-warn memory budget).
- **Parsed-once `Dataset` handle** (ADR-0020) — the GT cache
  produced by parsing `Dataset.from_coco_json` is retained as an
  opaque PyO3 handle that subsequent `Evaluator.evaluate(...)`
  calls reuse without re-parsing. `EvalGrid` keeps the `Dataset`
  alive across the `tables=` second pass, removing the prior
  double-parse.
- **`vernier-bench` cross-paradigm benchmark harness** (ADR-0017,
  ADR-0033) — multi-paradigm runner orchestrates vernier vs.
  pycocotools / faster-coco-eval / lvis-api / panopticapi /
  mmsegmentation across a workload ladder (synthetic, COCO val2017
  perfect-DT, mask-space jittered, real-model). `compare` /
  `report` / `bench-sync` subcommands; release-mode pin, machine
  fingerprint, IQR gate; bbox-IoU histogram dump on shutdown for
  Stage-0 instrumentation; `--with-images` cache for inference
  harnesses. Numbers feed `docs/comparison.md` and
  `docs/benchmarks.md`.
- **LVIS bench oracle** (ADR-0026 + ADR-0033) — `lvis-api 0.5.3` is
  wired as the federated-AP strict-tier oracle alongside
  pycocotools, with `Frequency`-aware K-axis cells.
- **Semantic bench oracle** (ADR-0036) — mmsegmentation is wired as
  the semantic strict-tier oracle (vernier-only cells published; the
  ADE20K/mmseg parity gate remains externally blocked).

### Changed (BREAKING — pre-1.0)

- **Per-paradigm namespace restructure** (ADR-0029) — the public
  Python surface splits across submodules: AP-fold types live under
  `vernier.instance` (`Evaluator`, `Bbox`, `Segm`, `Boundary`,
  `Keypoints`, `IouKind`, `Summary`, `EvalResult`, `Dataset`,
  `StreamingEvaluator`, `BackgroundEvaluator`, the TIDE / table /
  confusion-matrix surface, and the FFI exception classes); panoptic
  types live under `vernier.panoptic` (`Evaluator`, `Dataset`,
  `Predictions`, `Summary`, `ClassPanopticStats` — note the dropped
  `Panoptic` prefix on the unqualified type names). The cross-paradigm
  shared types (`ParityMode`, `Frequency`) and the pycocotools
  migration shim (`COCOeval`, `patch_pycocotools`) stay at the root.
  Per ADR-0029 §B1, no flat-root re-exports for moved symbols —
  `from vernier import Evaluator` raises `ImportError`.

### Added

- **Panoptic-quality (PQ) evaluation** (ADR-0025) — new sibling
  workspace crate `vernier-panoptic`, parallel to `vernier-core`,
  for the third leg of COCO evaluation
  (Kirillov et al. 2019, arXiv:1801.00868). Surface:
  `PanopticEvaluator(parity_mode='corrected', things_stuff_split=True)`
  with `.evaluate(gt: PanopticDataset, dt: PanopticPredictions)`
  returning a `PanopticSummary` (global PQ/SQ/RQ + things/stuff
  buckets + per-class rows). `PanopticDataset.from_arrays` and
  `PanopticPredictions.from_arrays` accept dicts of uint32 label
  maps + JSON segments_info; both run S1/S7/S11 validation and
  S3 PNG-marginal area recompute on the DT side. Single-threaded
  per ADR-0006 + X1 corrected disposition (bypasses panopticapi's
  multiprocessing pool entirely). `PanopticEvaluator(boundary=True)`
  raises `NotImplementedError` pointing at the Q3/Z1 follow-up
  ADR. ADR-0005 invariant preserved: zero edits to
  `crates/vernier-core/`; the firewall is structural (the new
  crate has no edge to vernier-core).
- **Panoptic parity oracle** — `cocodataset/panopticapi` vendored at
  SHA `7bb4655548f98f3fedc07bf37e9040a992b054b0` under
  `tests/python/parity_panoptic/oracle/panopticapi/`; pinned
  constants in `crates/vernier-panoptic/src/parity.rs`. Strict-mode
  bit-equality on the All/Things/Stuff + per-class shape is verified
  against `pq_compute_single_core(proc_id=0)` by `just
  test-parity-panoptic`. The multi-process pool is bypassed
  intentionally (X1 corrected; multi-process traces match under
  `Aligned` only, with `PANOPTIC_PARITY_EPS` placeholder
  `1e-9` until Q6 val measurement lands).
- **Migration guide** — `docs/explanation/panoptic-migration.md`
  covers the API mapping (`pq_compute` -> `PanopticEvaluator`),
  things/stuff semantics, sentinel divergence (panoptic `0.0` vs
  LVIS `-1.0`), single-vs-multi-process tolerance gotcha, and the
  boundary-PQ deferral.
- **LVIS federated evaluation** (ADR-0026) — long-tail benchmark
  support landed as modules in `vernier-core`. `Dataset.from_lvis_json`
  loads per-image `pos`/`neg`/`not_exhaustive_category_ids` and
  per-category `frequency`; the orchestrator's federated cell-skip
  (AA4) + `dt_ignore` extension (AA3) flow above the locked spine
  (ADR-0005, `matching.rs` and `accumulate.rs` unchanged).
  `Accumulated.summarize_lvis(dataset)` returns the canonical 13-entry
  plan (`AP`, `AP50`, `AP75`, `APs/m/l`, `APr/c/f`, `AR@300`,
  `ARs/m/l@300`); `CategoryFilter::{All, Frequency, ByIds}` is the
  K-axis subset selector behind it. `CocoDetections::lvis_trim`
  reproduces `LVISResults.limit_dets_per_image` (per-image top-K
  across all categories, AC2). `Frequency` enum (`r`/`c`/`f`) is the
  Python-facing tag.
- **LVIS parity oracle** — `lvis==0.5.3` vendored at
  `tests/python/parity_lvis/oracle/lvis_api/`; pinned constants in
  `crates/vernier-core/src/lvis_parity.rs`. Strict-mode bit-equality
  on the 13-entry summary against `LVISEval` is verified by
  `just test-parity-lvis-val`.
- **Migration guide** — `docs/explanation/lvis-migration.md` covers
  the silent-federated-semantics gotcha, the AF6 sentinel
  cross-reference (LVIS `-1` vs panoptic `0` vs uninitialized `nan`),
  and the explicit `max_dets=300` requirement.

### Performance

- **Bbox-IoU**: `pulp` `Arch` dispatch hoisted out of the inner loop
  + bool-mask prefilter; small-cell fast path bypasses dispatch
  overhead entirely.
- **Boundary**: bbox-cropped erode + bbox-cropped XOR scan skip
  per-mask full-image work; u64-packed row pass eliminates the
  strided gather/scatter; band derivation skips prefilter-empty
  rows/cols; per-image `BoundaryGtCache` + scratch reuse.
  COCO val2017 perfect-DT: 21.4s → 3.1s (-85.5%).
- **Segm**: bbox + area + offsets fused into a single counts walk;
  `SegmGtCache` + scratch reuse; `SegmentTable` + offset-based
  intersect closes the boundary regression; sparse-table AND-fold
  for `min_filter_binary`; single-pass band derivation skips the
  RLE round-trip; XOR fused into the segment scan.
  COCO val2017 segm: -21%.
- **Panoptic**: streaming runner + `FxHash` on per-image hashmaps
  drops val2017 perfect-DT 85.6s / 21.17 GiB → 32.3s / 127 MiB
  (now 1.11× faster than panopticapi); `submit_png` fuses libpng
  decode + RGB→id + S3 area recompute in Rust; row-streamed libpng
  output; thread-local DT-lookup scratch; `SegmentLookup` dispatch
  hoisted out of `decode_dt`; dense intersection matrix in
  `pq_image_with_id`.
- **Semantic**: per-image streaming through the FFI; deduped
  `parity_mode` parser; `evaluate_from_pngs` is the bench runner's
  default path.
- **FFI**: zero-copy GT/DT bytes via `PyBackedBytes` across
  `py.detach`; per-cell results boxed so `EvalGrid` skips the
  prior 268 MB zero-init; scratch buffers for cell-level + per-area
  gathers; `from_inputs` HashMap built off the GIL; `Dataset` /
  `CocoDetections` retained on `PyEvalGrid` so the `tables=` path
  skips a double-parse.

### Fixed

- **LVIS GT area filter** (quirk AG6) — strict mode now mirrors the
  oracle's `area > 0` ground-truth filter, eliminating a 0.06%
  divergence on two categories at the federated K=168 / K=817
  cells.
- **Panoptic `isthing` ingest** — tolerates `int 0/1` and numpy
  `int64` in `segments_info` (previously required `bool`).
- **sdist** — `LICENSE-APACHE` and `LICENSE-MIT` are included in
  the source distribution.

## [0.0.1] — 2026-04-30

First release with code. The placeholder 0.0.0 reservations on
crates.io and PyPI exposed no public API; 0.0.1 is the first wheel and
crate set that ship the evaluator.

### Added

- **Bbox parity** with `pycocotools==2.0.11` — strict-mode byte-equality
  on `evaluate()` / `accumulate()` / `summarize()` over COCO val2017,
  via the `EvalKernel` trait (ADR-0005) and the IoU-type-agnostic
  matching engine (ADR-0004).
- **Segm parity** — COCO RLE codec, polygon rasterizer, and mask ops in
  the leaf `vernier-mask` crate (ADR-0009); RLE bbox-IoU prefilter for
  the typical-pair speedup (quirk I1).
- **Boundary IoU** — bowenc0221 `boundary-iou-api` is the strict-mode
  oracle; `--dilation-ratio` selects band thickness (ADR-0010,
  isolated subsystem).
- **OKS keypoints** — per-category sigmas via `IouKind::Keypoints`
  (ADR-0012); `kpt_oks_sigmas` does not leak across `iouType`s; kp
  10-stat summarizer plan with `maxDets = [20]`.
- **Generalized `Breakdown` axis** (ADR-0016) — `Breakdown { axis,
  buckets }` lifts the hard-coded small/medium/large area buckets;
  closed-on-both-ends `contains` per quirk D6.
- **`StreamingEvaluator`** (ADR-0013) — push-batches surface for
  out-of-core inference workloads; bounded memory, snapshot/finalize.
- **`BackgroundEvaluator`** (ADR-0014) — async wrapper that offloads
  matching/accumulation onto a worker thread with backpressure and a
  soft-warn memory budget.
- **`patch_pycocotools()` shim** — replaces `pycocotools.cocoeval.COCOeval`
  in `sys.modules` so existing user code transparently exercises
  vernier (ADR-0007).
- **`Evaluator` extended-API class** — Rust-native builder surface
  exposing strict / aligned / corrected parity modes (ADR-0002).
- **`COCOeval` drop-in class** — faithful replication of the `Params`
  mutability pattern; `iouType ∈ {bbox, segm, boundary, keypoints}`.
- **`vernier-cli`** — `vernier eval` workspace binary (ADR-0015) with
  text + JSON v1 emit formatters; strict-mode stdout byte-equal to
  `COCOeval(...).summarize()`. Schema version is independent of the
  package version.
- **Stable-Rust SIMD via `pulp`** (ADR-0003) — runtime CPU-feature
  dispatch on bbox / boundary IoU inner loops.
- **Quirk survey** — `docs/engineering/pycocotools-quirks.md` enumerates
  61 quirks (A1–L8) with three-tier dispositions; cited verbatim
  throughout the codebase.
- **Parity fixtures** — minimal per-quirk fixtures plus full COCO
  val2017 perfect-DT smoke for bbox, segm, boundary, and keypoints.

[Unreleased]: https://github.com/NoeFontana/vernier/compare/v0.0.3...HEAD
[0.0.3]: https://github.com/NoeFontana/vernier/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/NoeFontana/vernier/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/NoeFontana/vernier/releases/tag/v0.0.1
