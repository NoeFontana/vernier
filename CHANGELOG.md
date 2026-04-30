# Changelog

All notable changes to this project will be documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning stays on the 0.0.x release line until the core and extended
feature set is complete; moving to 0.1.0+ is a deliberate later decision.

## [Unreleased]

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
