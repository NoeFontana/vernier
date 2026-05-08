# Architecture Decision Records

This directory holds vernier's Architecture Decision Records (ADRs): the
immutable record of *why* the project is built the way it is.

The format is [MADR](https://adr.github.io/madr/) (Markdown Architecture
Decision Records). To propose a new ADR:

1. Copy `template.md` to `NNNN-short-kebab-title.md` with the next available
   number.
2. Fill in *Context*, *Decision*, *Consequences*. Set status to `proposed`.
3. Open a PR. Discuss until consensus.
4. On merge, status becomes `accepted`. From this point the ADR is
   immutable — supersede it with a later ADR rather than editing it.

## Index

| #    | Title                                                    | Status   |
| ---- | -------------------------------------------------------- | -------- |
| 0001 | Record architecture decisions                            | accepted |
| 0002 | Adopt a three-tier parity model against pycocotools      | accepted |
| 0003 | Use `pulp` for stable-Rust SIMD with runtime CPU dispatch| accepted |
| 0004 | Numerical layout policy — f32 internal, f64 boundary, SoA, pinned constants | accepted |
| 0005 | Lock the `Similarity` trait and matching-engine API for Phases 1–3 | accepted |
| 0006 | Threading model — GIL-drop at every PyO3 entry, single-threaded compute for Phase 1 | accepted |
| 0007 | `patch_pycocotools` — opt-in `sys.modules` monkey-patch as the migration path | accepted |
| 0008 | Bbox IoU computes in `f64` end-to-end (supersedes ADR-0004's bbox clause) | accepted |
| 0009 | Split mask kernels into a `vernier-mask` workspace crate | accepted |
| 0010 | Boundary IoU as an isolated subsystem with its own oracle, quirks, and performance baseline | accepted |
| 0011 | Discriminated kernel config replaces the `iou_type` string literal | accepted |
| 0012 | OKS keypoints public surface and quirk dispositions      | accepted |
| 0013 | Streaming evaluator — store per-image evals, fold on snapshot and finalize | accepted |
| 0014 | `BackgroundEvaluator` — single-worker, bounded-queue async wrapper around `StreamingEvaluator` | accepted |
| 0015 | Ship `vernier-cli` as a workspace binary that links `vernier-core` directly | accepted |
| 0016 | Generalize the A-axis as a value-typed `Breakdown`       | proposed |
| 0017 | Local bench harness — subprocess-isolated, uv-managed, parity-coupled | proposed |
| 0018 | Calibration metrics for modern detection architectures   | proposed |
| 0019 | Result tables — opt-in, Arrow-backed, zero-overhead by default | proposed |
| 0020 | Parsed-once `Dataset` handle as the GT-side derivation cache | proposed |
| 0021 | TIDE error decomposition — NumPy oracle as the correctness model | proposed |
| 0022 | TIDE thresholds and per-kernel defaults | proposed |
| 0023 | Cross-class IoU as an orchestrator-level side pass | proposed |
| 0024 | TIDE on keypoints (OKS) — deferred | proposed |
| 0025 | Add panoptic-quality evaluation as a sibling crate | accepted |
| 0026 | Add LVIS federated evaluation in `vernier-core` | accepted |
| 0027 | Documentation framework — Diátaxis on `mkdocs-material`, code-tested, gated in CI | accepted |
| 0028 | Add semantic segmentation evaluation as a `vernier-semantic` sibling crate | accepted |
| 0029 | Restructure public Python API into per-paradigm submodules | accepted |
| 0034 | Add `aarch64-unknown-linux-gnu` to the `vernier-cli` GitHub Release target list | proposed |

(Update this table as ADRs land. Eventually we may automate it from the
front-matter, but until there are enough ADRs to make that worthwhile, hand
maintenance is fine. Note: rows 0030–0033 are missing from this index — a
separate cleanup, not in scope for ADR-0034.)
