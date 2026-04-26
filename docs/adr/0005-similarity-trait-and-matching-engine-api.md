# ADR-0005: Lock the `Similarity` trait and matching-engine API for Phases 1–3

- **Status:** proposed
- **Date:** 2026-04-26
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

vernier supports three IoU types (bbox in Phase 1, segm in Phase 2,
keypoints/OKS in Phase 3) and several non-COCO datasets (LVIS, CrowdPose,
custom). The roadmap's architectural commitment is that **the matching
engine and accumulator are written once in Phase 1 and never edited** —
later phases add new IoU types and new datasets without touching that
code.

The architectural test of Phase 1 is whether Phase 2 (segm) and Phase 3
(keypoints) require any edits to `matching.rs` or `accumulate.rs`. If they
do, the abstraction failed. The question this ADR answers is: what trait
shape makes that invariant possible?

The natural shape is a `Similarity` trait that produces an IoU matrix from
a slice of GTs and a slice of DTs. The matching engine is generic over
the produced matrix only — never over the IoU type. New IoU types are new
trait implementations; matching code does not change.

Without an explicit ADR locking the trait shape, the trait will accumulate
drift: a `Phase 2` author adds a `&self` parameter for a mask cache; a
`Phase 3` author adds a `cat_id` because OKS sigmas are per-category; a
later author adds an `image_size` because boundary IoU needs the image
dimensions. Each individual change is reasonable in isolation; together,
they break the invariant that `match_image` is type-agnostic.

## Decision drivers

- The trait shape must support all three Phase 1–3 IoU types
  (bbox, segm with crowd asymmetry, OKS) without expanding.
- The trait must permit batch / SIMD inner loops (per ADR-0003 and
  ADR-0004): the impl receives slices, not single pairs, so it can choose
  its own inner-loop layout.
- The trait must be shaped so that quirks E1 (crowd asymmetric IoU) and
  F1 (per-category OKS sigmas) live inside the impl, not in the matching
  engine. Threading "is this a crowd?" or "what category?" through
  matching pulls IoU-type knowledge into the wrong layer.
- Public Rust API (per ADR-0001 §"Affect the public API"). Crosses the
  FFI boundary (per ADR-0001 §"Cross the FFI boundary").

## Considered options

1. **Trait-per-pair: `fn iou(gt, dt) -> f64`.** Simplest; one method
   per pair. Forces the matching engine to build an outer-loop matrix and
   leaves the impl with no opportunity for SIMD. Wrong layer for batch.
2. **Single batch method on the trait, IoU-type-specific param structs
   passed in.** The matching engine knows about `BboxParams`,
   `SegmParams`, etc. Pulls IoU-type knowledge into matching.
3. **Single batch method on the trait, IoU-type-specific config stored on
   the impl.** Matching engine sees only the trait. Each impl carries its
   own config (sigmas, prefilter thresholds, parity-mode flag). Matching
   code is type-agnostic.
4. **No trait — separate `compute_iou_bbox`, `compute_iou_segm`,
   `compute_iou_oks` functions.** Forces matching to be three near-
   duplicates (one per IoU type) or to take a function pointer; no
   ergonomic reuse.

## Decision outcome

Chosen option: **Option 3 — single batch method, type-specific config on
the impl.**

### Trait

```rust
/// Computes a similarity matrix between ground-truth and detection
/// annotations. One impl per IoU type (bbox, segm, OKS, boundary, …).
pub trait Similarity: Send + Sync {
    /// The annotation type this impl consumes. For bbox this is
    /// `BboxAnn`, for segm `SegmAnn`, for OKS `KeypointAnn`. The shared
    /// metadata (image_id, category_id, area, is_crowd) is on the
    /// outer `Annotation` struct, not the variant; the impl extracts
    /// only what it needs.
    type Annotation;

    /// Writes pairwise similarity into `out`, where rows index gts and
    /// columns index dts. Out-of-band signals (impossible match) are
    /// expressed as `0.0`, never as a sentinel. Errors are reported
    /// via the typed `EvalError`; impls do not panic.
    fn compute(
        &self,
        gts: &[Self::Annotation],
        dts: &[Self::Annotation],
        out: &mut ndarray::ArrayViewMut2<'_, f64>,
    ) -> Result<(), EvalError>;
}
```

The impl owns whatever it needs:

- `BboxIou` is a unit struct (no state).
- `SegmIou` is a unit struct.
- `BoundaryIou { dilation_pixels: u32 }`.
- `OksSimilarity { sigmas: Arc<HashMap<CategoryId, Vec<f32>>> }`.
- `MaskedSegmIou { ignore_mask_cache: ... }` (a hypothetical Phase 4+ impl).

### Matching engine

```rust
/// Greedy assignment over a precomputed similarity matrix.
///
/// Generic over the matrix only. Knows nothing about bbox, segm, or
/// keypoints. If a future phase needs to edit this signature, the
/// abstraction has failed and the right response is an architectural
/// review, not a workaround.
pub fn match_image(
    iou_matrix: ndarray::ArrayView2<'_, f64>,
    gt_ignore: &[bool],
    gt_iscrowd: &[bool],
    dt_scores: &[f64],
    iou_thresholds: &[f64],
    parity_mode: ParityMode,
) -> MatchResult { /* ... */ }
```

`ParityMode` (an enum from `parity.rs`) carries the strict/aligned/
corrected flag from ADR-0002. This is the only IoU-type-agnostic config
parameter the matching engine needs.

### Accumulator

```rust
pub fn accumulate(
    matches: &[MatchResult],
    params: &EvalParams,
    parity_mode: ParityMode,
) -> Accumulated { /* ... */ }
```

Same shape: takes precomputed match arrays, knows nothing about IoU type.

### Invariant

Phases 2 and 3 add new `Similarity` impls and new `EvalDataset` impls.
They **do not** add parameters to `match_image` or to `accumulate`. PRs
that propose to add such parameters are rejected and routed back into a
new `Similarity` impl or a new summarizer mode.

The single exception that justifies a future ADR: if a quirk is discovered
where the matching algorithm itself differs by IoU type (e.g., a reading
of pycocotools that we haven't anticipated), the response is a new ADR
that supersedes this one, not an in-place edit.

### Accepted consequences

- **Positive.** Phase 2 and Phase 3 work is bounded: new files, no edits to
  the matching/accumulator spine. Reviewers have a one-line standard:
  "does this PR edit `matching.rs` or `accumulate.rs`?". The trait method
  takes slices (not pairs), so the impl chooses its own SIMD layout per
  ADR-0003.
- **Negative.** The `Annotation` associated type means the trait is not
  object-safe (`dyn Similarity` won't compile because of the type-tag).
  This is acceptable: dispatch happens at the Python boundary
  (`Evaluator::iou_type` selects which impl to instantiate), and inside
  Rust each impl is monomorphized. If we later want runtime-pluggable
  similarity (e.g., user-provided IoU via a callback from Python), we
  introduce a separate object-safe wrapper trait at the FFI layer.
- **Neutral.** `Send + Sync` bounds are required for the
  `BackgroundEvaluator` (its own ADR). Impls with non-Sync state
  (e.g., a mutable cache) must wrap it in `Mutex` or `RwLock`.

## Pros and cons of the options

### Option 1 — pair method `fn iou(gt, dt)`

- 👍 Trivial to implement; trivial to mock in tests.
- 👎 No batching. The matching engine builds the matrix; the impl gets
  no chance to SIMD-ify. Throws away the entire performance argument
  for the Rust port.

### Option 2 — batch method, type-specific params at call site

- 👍 Concrete params are explicit at the call site.
- 👎 Matching engine has to know about `BboxParams`, `OksParams`. Adds
  a generic parameter or a branch. Phase 2 immediately requires editing
  matching to thread the new param. Invariant fails.

### Option 3 (chosen) — batch method, config on the impl

- 👍 Matching engine is type-agnostic. Each impl encapsulates its own
  state (sigmas, dilation, prefilter thresholds). Phase 2 and 3 add
  files, not parameters. Slice-in-slice-out matches the SIMD-friendly
  layout. Crowd asymmetric IoU (E1) lives in the impl. Per-category
  OKS sigmas (F1) live in the impl.
- 👎 Not object-safe; runtime-pluggable user-provided similarity needs a
  separate FFI wrapper if it ever becomes a requirement.

### Option 4 — three free functions

- 👍 No trait machinery.
- 👎 Matching code becomes three near-duplicates or takes a function
  pointer. Loses the unifying abstraction. Dataset adapters that want to
  swap similarity at runtime have to dispatch by enum at the call site.

## Links and references

- ADR-0001 — Record architecture decisions.
- ADR-0002 — Three-tier parity model.
- ADR-0003 — `pulp` for stable-Rust SIMD with runtime dispatch.
- ADR-0004 — Numerical layout policy (f32 internal / f64 boundary, SoA).
- `docs/engineering/pycocotools-quirks.md` — E1 (crowd IoU asymmetry),
  F1 (per-category sigmas).
