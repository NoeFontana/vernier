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
| 0009 | Split mask kernels into a `vernier-mask` workspace crate | proposed |

(Update this table as ADRs land. Eventually we may automate it from the
front-matter, but until there are enough ADRs to make that worthwhile, hand
maintenance is fine.)
