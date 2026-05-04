# Changelog

All notable changes to this project will be documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning stays on the 0.0.x release line until the core and extended
feature set is complete; moving to 0.1.0+ is a deliberate later decision.

## [Unreleased]

### Added

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
  `from vernier import Evaluator` raises `ImportError`. Migration:
  `python tools/migrate_imports.py --tree path/to/your/code` rewrites
  most call sites mechanically; the script ships in this release for
  external 0.0.x users to replay and is expected to be deleted in a
  follow-up patch.

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

[Unreleased]: https://github.com/NoeFontana/vernier/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/NoeFontana/vernier/releases/tag/v0.0.1
