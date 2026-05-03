# ADR-0023: Cross-class IoU as an orchestrator-level side pass

- **Status:** proposed
- **Date:** 2026-05-02
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

The TIDE bin assignment (ADR-0021) needs `IoU(DT_class_A, GT_class_B)`
for two of the six bins:

- **Cls** — a detection of class A whose best wrong-class GT (some
  class B ≠ A) overlaps at IoU ≥ `t_f`. The detection looks like a
  classification mistake.
- **Both** — same shape, but with overlap in `[t_b, t_f)`. Localization
  *and* classification mistakes compounded.

Confusion matrix output (a sibling capability shipping in the same
release) needs the same data: per-DT, the best-overlapping GT across
all classes.

The matching engine (ADR-0005) is generic over the IoU matrix; category
filtering happens one level up in the orchestrator at `evaluate.rs:721`,
which builds per-`(image, category)` GT/DT slices and calls
`Similarity::compute(&gt_slice, &dt_slice, &mut iou)`. The same-class
invariant is enforced by construction of the slices, not by a check
inside matching.

`RetainedIous` (`tables.rs:686`) already exists for the result-tables
path: same-class IoU matrices keyed `(category_idx, image_idx)`. There
is no cross-class storage today.

Where does the cross-class IoU data come from, and where is it stored?
Three layers could host the work:

- **Inside matching** — extend matching to optionally take an
  un-filtered matrix and return cross-class match data. Touches the
  ADR-0005 invariant.
- **At the orchestrator level** — a side pass that calls the same
  `Similarity::compute` kernel with unfiltered per-image GT/DT lists.
  No matching.rs change.
- **Inside the bin-assignment loop** — recompute IoU per detection on
  demand against same-image GTs of every class.

This ADR picks the layer.

## Decision drivers

- **Preserve the ADR-0005 matching invariant.** Matching is generic
  over the matrix; same-class is the orchestrator's job. Pushing
  cross-class awareness into matching expands its contract for one
  caller (TIDE) at the cost of every other caller's mental model.
- **Reuse the existing kernel.** `Similarity::compute(&gts, &dts,
  &mut out)` is class-agnostic by signature (`crates/vernier-core/
  src/similarity/mod.rs:41-61`). Calling it with unfiltered slices
  is the kernel doing exactly what it already does, on a larger
  matrix.
- **Confusion matrix funds the same pass.** If both
  `error_decomposition` and `confusion_matrix` need the data,
  paying for it twice is waste; storage that both can read amortizes
  the cost.
- **Memory budget must be bounded.** COCO val (~5000 images × ~100
  dets × ~30 GTs × 8 B) ≈ 120 MB worst case. Acceptable as a
  default for an opt-in capability that runs offline. Dense-prediction
  workloads (DETR-style: 300 dets/image, 1000+ classes) blow this
  budget; need an escape hatch.
- **Determinism.** Whatever path we take must be deterministic across
  runs — the matching engine and similarity kernels both already are
  (ADR-0006 threading model), so the side pass inherits this for free
  if we use the same kernel.

## Considered options

1. **Modify matching to take an un-filtered matrix.** Add a
   `cross_class_mode` flag to `match_image`; matching returns
   per-class match data plus cross-class best-IoU records.
2. **Orchestrator-level side pass via `Similarity::compute`.** New
   per-image pass calling the same kernel with unfiltered per-image
   GT/DT lists. New `CrossClassIous` storage keyed by `image_idx`.
   Bin assignment and confusion matrix both read from it.
3. **Recompute on demand inside the bin-assignment loop.** No storage;
   per detection, ask the kernel for IoU against the image's GTs.
4. **Top-K per detection only.** Storage is per-DT top-K cross-class
   IoU values; full matrix never materialized.

## Decision outcome

Chosen option: **(2) orchestrator-level side pass via
`Similarity::compute`,** materializing a per-image cross-class IoU
matrix into a new `CrossClassIous` store. **(4) ships as an opt-in
knob** (`cross_class_topk: Option<usize>`, default `None` = full
materialize) for dense-prediction workloads.

Mechanics. A new orchestrator entry point
`evaluate_with_retention` in `crates/vernier-core/src/evaluate.rs`
(sibling of `evaluate_with` at line 721) walks images, builds the
unfiltered per-image GT and DT slices, and calls
`kernel.compute(&gts_unfiltered, &dts_unfiltered, &mut matrix)` once
per image. The matrix is stored in:

```rust
// crates/vernier-core/src/tables.rs (next to RetainedIous)
pub struct CrossClassIous {
    inner: HashMap<usize /* image_idx */, Array2<f64> /* (D_total, G_total) */>,
    dt_classes: HashMap<usize, Vec<usize>>,  // per-image DT category indices
    gt_classes: HashMap<usize, Vec<usize>>,  // per-image GT category indices
}
```

The `dt_classes` / `gt_classes` side vectors carry the category index
per row/column so the bin-assignment layer can answer "what class is
the best-overlapping GT?" without touching the dataset again.

The same `CrossClassIous` is consumed by both `error_decomposition`
(for Cls/Both bin assignment) and `confusion_matrix` (for the
`(gt_class, dt_class)` cell counts). One pass, two consumers.

Matching is unchanged. The ADR-0005 invariant is preserved
verbatim — the side pass calls a kernel, not matching.

The `cross_class_topk` knob, when set, replaces the full
`Array2<f64>` per image with a `(D_total × K)` array of top-K
cross-class IoU values plus a parallel `(D_total × K)` array of GT
indices. K=20 covers the realistic worst case (a DT meaningfully
overlapping ≥20 GTs is improbable). Implemented in 0.5.0 as a
feature-complete escape hatch; the bin-assignment layer reads through
a small adapter trait so the consuming code does not branch.

### Consequences

- **Positive.** ADR-0005 untouched. The kernel pulp dispatch
  (ADR-0003) on bbox and the SegmGtCache / BoundaryGtCache hits
  (ADR-0020) all apply to the side pass for free — no new perf
  story to write. Confusion matrix and TIDE share one pass. The
  `cross_class_topk` knob bounds the worst case.
- **Negative.** Up to ~120 MB of IoU storage on COCO val for the
  duration of `error_decomposition`. The ADR-0020 `Dataset` handle
  bounds the lifetime tightly: the side-pass output lives only for
  the call. Dense workloads need the topk knob; not invisible.
- **Neutral.** Two storage types that look similar (`RetainedIous`
  vs `CrossClassIous`) with different keying. The shape is correct
  (one is for a same-class result-tables product, the other is for
  TIDE/confusion); naming makes the difference clear. We do not
  unify them.

## What this ADR explicitly does not decide

- **Numpy oracle and correctness model.** ADR-0021.
- **Threshold defaults.** ADR-0022.
- **TIDE-on-OKS / keypoints.** ADR-0024 (deferred — OKS is single-
  class in COCO, so the cross-class story does not apply).
- **Caching the cross-class matrices across calls.** Out of scope.
  TIDE is a one-shot offline call; reuse across calls is a follow-up
  if real workloads warrant. The `Dataset` handle (ADR-0020) caches
  GT-side derivations; cross-class IoU has a DT-side input and so
  cannot share that lifetime.
- **Streaming-evaluator integration.** TIDE on a streaming evaluator
  is out of scope for 0.5.0; the side pass requires the full image
  set known up front, which the streaming evaluator does not give us.
  Documented in the explanation page.
- **Top-K storage layout details.** The trait adapter the
  bin-assignment layer reads through has its concrete shape settled
  in implementation; this ADR commits to "topk is an escape hatch
  with `None` = full materialize", not to the byte layout.

## Pros and cons of the options

### (1) Modify matching to take an un-filtered matrix

- 👍 Single pass through the matching engine; no second walk.
- 👎 Expands matching's contract for one caller. The ADR-0005
  invariant ("matching is generic over the matrix only") becomes
  "matching is generic over the matrix unless TIDE asks for
  cross-class." Any future caller now has to know which mode
  matching was called in. We do not modify a public-API invariant
  for one feature.
- 👎 Matching's output type (`MatchResult`) would need a
  cross-class extension; every other caller pays for a field they
  do not use.

### (2) Orchestrator-level side pass (chosen)

- 👍 Zero matching.rs change. The kernel does what it already does.
  Reviewers see one new orchestrator function, not a contract
  expansion in core matching.
- 👍 The side pass output is consumed by both TIDE and confusion
  matrix; one pass funds two features.
- 👍 The `cross_class_topk` escape hatch composes naturally — it is
  a property of the side-pass output, not of the kernel.
- 👎 Walks images twice in the worst case (once for matching, once
  for cross-class). On bbox COCO val this is ~2 s extra per call;
  on dense workloads the topk knob caps it.
- 👎 Two IoU-storage types in `tables.rs` (`RetainedIous` for
  result tables, `CrossClassIous` for TIDE/confusion) — accepted
  as the cost of clean per-feature ownership.

### (3) Recompute on demand inside the bin-assignment loop

- 👍 No storage at all. Smallest memory footprint.
- 👎 The bin-assignment loop reads every detection's cross-class
  IoU at least once. Confusion matrix reads it again. Lazy = paying
  twice. Saves no work because we never have a reader who *doesn't*
  read a given DT.
- 👎 Couples the bin-assignment layer to the kernel directly; the
  layer becomes responsible for kernel-dispatch concerns it
  shouldn't know about.

### (4) Top-K per detection only

- 👍 Smallest fixed storage; bounded independently of `n_classes`.
- 👎 As the *only* mode, surfaces in the public `TideConfig` for
  every user, including the COCO-val-fits-in-200 MB user who does
  not need it.
- ✅ Adopted as an opt-in knob layered on (2), not as the default.

## Links and references

- ADR-0001 — Record architecture decisions.
- ADR-0003 — Stable-Rust SIMD via `pulp` (the dispatch the side
  pass inherits for free on bbox).
- ADR-0005 — Similarity trait and matching engine API (the
  invariant this ADR preserves).
- ADR-0006 — Threading model (the determinism the side pass
  inherits).
- ADR-0010 — Boundary IoU isolated subsystem (the kernel whose
  caching strategy applies to the side pass without modification).
- ADR-0019 — Result tables (the `RetainedIous` precedent for IoU
  storage in `tables.rs`).
- ADR-0020 — Parsed-once `Dataset` handle (the lifetime that bounds
  the side-pass output).
- ADR-0021 — TIDE numpy oracle and correctness model (the consumer
  whose Cls/Both bins read the cross-class data).
- ADR-0022 — TIDE thresholds and per-kernel defaults (the `t_b` /
  `t_f` cutoffs that slice the cross-class data into bins).
