# Architecture Decision Records

This directory holds vernier's Architecture Decision Records (ADRs): the
immutable record of *why* the project is built the way it is. Format is
[MADR](https://adr.github.io/madr/).

To propose a new ADR: copy `template.md` to `NNNN-short-kebab-title.md`
with the next free number, fill in *Context / Decision / Consequences*,
open a PR with status `proposed`. On merge, status flips to `accepted`.
Accepted ADRs are immutable — supersede with a later ADR rather than
editing in place.

## Suggested reading order for newcomers

Five themes explain the project's load-bearing decisions; everything
else is detail.

1. **[0002 — three-tier parity model](0002-three-tier-parity-model.md).**
   The contract that distinguishes vernier from every other fast COCO
   evaluator: each pycocotools quirk gets a `strict`, `aligned`, or
   `corrected` disposition.
2. **[0005 — Similarity trait + matching engine API](0005-similarity-trait-and-matching-engine-api.md).**
   The kernel-vs-engine split that lets bbox / segm / boundary /
   keypoints share one matching loop.
3. **[0009 — vernier-mask crate split](0009-split-mask-kernels-into-vernier-mask-crate.md)**
   and **[0025 — panoptic sibling crate](0025-panoptic-api.md)** /
   **[0028 — semantic sibling crate](0028-sem-seg.md).** The
   architectural firewall that keeps AP, PQ, and mIoU from leaking
   into each other.
4. **[0029 — per-paradigm submodules](0029-namespace.md).** Why the
   public Python surface is `vernier.{instance, panoptic, semantic}`
   instead of one flat namespace.
5. **[0017 — local bench harness](0017-local-bench-harness.md).** How
   the numbers in [`docs/benchmarks.md`](../benchmarks.md) and
   [`docs/comparison.md`](../comparison.md) are produced and gated by
   parity.

## Index

### Foundations

| #    | Title                                                       | Status   |
| ---- | ----------------------------------------------------------- | -------- |
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions (the meta-ADR) | accepted |
| [0027](0027-doc-framework.md) | Diátaxis + mkdocs-material as the docs framework, gated in CI | accepted |

### Parity contract

| #    | Title                                                       | Status   |
| ---- | ----------------------------------------------------------- | -------- |
| [0002](0002-three-tier-parity-model.md) | Three-tier parity model — `strict` / `aligned` / `corrected` | accepted |
| [0007](0007-patch-pycocotools-policy.md) | `patch_pycocotools` — opt-in `sys.modules` monkey-patch shim | accepted |

### Numerics

| #    | Title                                                       | Status   |
| ---- | ----------------------------------------------------------- | -------- |
| [0003](0003-stable-rust-simd-via-pulp.md) | `pulp` for stable-Rust SIMD with runtime CPU dispatch | accepted |
| [0004](0004-numerical-layout-policy.md) | f32 internal, f64 boundary, SoA, pinned constants | accepted (bbox clause superseded by 0008) |
| [0008](0008-bbox-iou-f64-end-to-end.md) | Bbox IoU computes in f64 end-to-end | accepted (supersedes 0004's bbox clause) |

### Architecture

| #    | Title                                                       | Status   |
| ---- | ----------------------------------------------------------- | -------- |
| [0005](0005-similarity-trait-and-matching-engine-api.md) | Lock the `Similarity` trait + matching-engine API for Phases 1–3 | accepted |
| [0006](0006-threading-model.md) | GIL-drop at every PyO3 entry; single-threaded compute for Phase 1 | accepted |
| [0009](0009-split-mask-kernels-into-vernier-mask-crate.md) | Split mask kernels into the `vernier-mask` workspace crate | accepted |
| [0011](0011-discriminated-kernel-config.md) | Discriminated kernel config replaces the `iou_type` string literal | accepted |
| [0029](0029-namespace.md) | Public Python API as per-paradigm submodules | accepted |

### Instance kernels

| #    | Title                                                       | Status   |
| ---- | ----------------------------------------------------------- | -------- |
| [0010](0010-boundary-iou-isolated-subsystem.md) | Boundary IoU as an isolated subsystem with its own oracle + quirks | accepted |
| [0012](0012-oks-keypoints-surface.md) | OKS keypoints public surface and kernel-coupled defaults | accepted |

### Public surfaces — dataset, results, streaming, CLI

| #    | Title                                                       | Status   |
| ---- | ----------------------------------------------------------- | -------- |
| [0013](0013-streaming-evaluator.md) | Streaming evaluator — store per-image evals, fold on snapshot | accepted |
| [0014](0014-background-evaluator.md) | `BackgroundEvaluator` — single-worker async wrapper | accepted |
| [0015](0015-vernier-cli.md) | Ship `vernier-cli` as a workspace binary linking `vernier-core` directly | accepted |
| [0016](0016-generalized-breakdown-axis.md) | Generalize the A-axis as a value-typed `Breakdown` | accepted |
| [0019](0019-result-tables.md) | Result tables — opt-in, Arrow-backed, zero-overhead by default | accepted |
| [0020](0020-parsed-once-dataset-handle.md) | Parsed-once `Dataset` handle as the GT-side derivation cache | accepted |
| [0030](0030-buffer-protocol.md) | Accept detection arrays alongside JSON bytes in streaming update | accepted (amended 2026-05-09 — bitmask + compressed-bytes ingest readmitted) |

### Sibling paradigms

| #    | Title                                                       | Status   |
| ---- | ----------------------------------------------------------- | -------- |
| [0025](0025-panoptic-api.md) | Add panoptic-quality (PQ) evaluation as a sibling crate | accepted |
| [0026](0026-lvis-support.md) | Add LVIS federated evaluation in `vernier-core` | accepted |
| [0028](0028-sem-seg.md) | Add semantic segmentation as a `vernier-semantic` sibling crate | accepted |

### Distributed evaluation

| #    | Title                                                       | Status   |
| ---- | ----------------------------------------------------------- | -------- |
| [0031](0031-dist-eval.md) | `from_partials` and the partial wire format (instance) | accepted (cross-paradigm carve-out superseded by 0032) |
| [0032](0032-dist-eval-paradigms.md) | Distributed evaluation across paradigms | accepted (supersedes 0031's cross-paradigm carve-out) |

### Bench harness

| #    | Title                                                       | Status   |
| ---- | ----------------------------------------------------------- | -------- |
| [0017](0017-local-bench-harness.md) | Local bench harness — subprocess-isolated, uv-managed, parity-coupled | accepted (streaming carve-out superseded by 0033) |
| [0033](0033-multi-paradigm-bench.md) | Extend the bench harness across paradigms (panoptic, semantic, streaming) | accepted (supersedes 0017's streaming carve-out) |

### TIDE design package — proposed cluster

The four ADRs below were drafted together (2026-05-02) as a single
TIDE-error-decomposition design. Read them as a set; the kernel
implementation under `crates/vernier-core/src/tide/` cites them
individually and the per-decision precision matters for future
supersession.

| #    | Title                                                       | Status   |
| ---- | ----------------------------------------------------------- | -------- |
| [0021](0021-tide-oracle.md) | TIDE error decomposition — numpy oracle as the correctness model | proposed |
| [0022](0022-tide-thresholds.md) | TIDE thresholds and per-kernel defaults | proposed (machinery shipped; val2017 cross-model run + amendment pending) |
| [0023](0023-tide-cross-class-strategy.md) | Cross-class IoU as an orchestrator-level side pass | proposed |
| [0024](0024-tide-keypoints-deferred.md) | TIDE on keypoints (OKS) — deferred | proposed |

### Calibration — proposed

| #    | Title                                                       | Status   |
| ---- | ----------------------------------------------------------- | -------- |
| [0018](0018-calibration.md) | Calibration metrics for modern detection architectures (Phase 5) | proposed |

### Operational

| #    | Title                                                       | Status   |
| ---- | ----------------------------------------------------------- | -------- |
| [0034](0034-aarch64-linux-gnu-release-target.md) | Add `aarch64-unknown-linux-gnu` to the `vernier-cli` Release target list | proposed |
