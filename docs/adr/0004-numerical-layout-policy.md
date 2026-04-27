# ADR-0004: Numerical layout policy — f32 internal, f64 boundary, SoA, pinned constants

- **Status:** accepted (bbox-IoU clause superseded by ADR-0008)
- **Date:** 2026-04-26
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

The numerical behavior of vernier's evaluator is the parity contract.
`pycocotools` performs every IoU and every accumulation in Python's `float`
(IEEE 754 binary64). `numpy` arrays are mostly `float64`. The reference
oracle's outputs are bit-keyed to that choice.

But: SIMD throughput on a given vector width is twice as high in `f32` as
in `f64` (AVX2 = 8 lanes vs 4; AVX-512 = 16 vs 8; NEON = 4 vs 2). For
vernier's hot kernels, going `f32` doubles the speedup ceiling. The IoU
computation itself is numerically robust at `f32` — typical bbox / RLE
intersections have a relative-error budget several orders of magnitude
larger than `f32` precision. The total error budget is dominated by the
accumulation step, not by IoU.

Without a written policy, every kernel author independently decides the
numeric type. The result is silent precision drift along the eval pipeline:
some kernels are `f64`, some are `f32`, some round-trip through `f32`
internally and present `f64` at the boundary, and parity tests fail in ways
that take a week to track down.

A separate axis: data layout. `pycocotools` allocates Python lists of
`np.array([x, y, w, h], dtype=float64)` per detection — array of structures
(AoS). vernier wants structure of arrays (SoA: `Vec<f32>` for `x`, for `y`,
for `w`, for `h` separately) so that SIMD lanes load contiguous memory.
Without a policy, the same drift happens on the layout axis.

A third axis: constants. The accumulator uses `np.spacing(1) ≈ 2.22e-16`
as a divide-by-zero guard. Numpy's `searchsorted(side='left')` is what
pycocotools uses to bucket recall thresholds. Stable mergesort is what
pycocotools uses to argsort scores. Without an explicit list of pinned
constants, parity drifts when a kernel author picks a "reasonable"
substitute (`f32::EPSILON`, `partition_point`, `sort_unstable`).

## Decision drivers

- The strict-mode parity contract from ADR-0002 must produce bit-equal
  outputs to pycocotools 2.0.11 on the reference fixtures.
- The aligned-mode parity contract requires bit-equal outputs (or
  documented tolerance) with cleaner implementations.
- SIMD throughput should not be sacrificed for numerical conservatism that
  doesn't change observable outputs.
- Reviewers should not need to negotiate type and layout per PR.

## Considered options

1. **All-f64 throughout.** Mirrors pycocotools type-for-type. Halves SIMD
   throughput; no ambiguity.
2. **All-f32 throughout.** Doubles SIMD throughput; observably diverges
   from pycocotools at the accumulator (recall thresholds, eps, summary
   stats), so strict-mode parity fails.
3. **f32 internal, f64 boundary.** Inner kernels (IoU compute, mask area,
   keypoint-distance squared) use `f32`. Cross-kernel matrices and
   accumulator state use `f64`. Conversion happens at module boundaries.
4. **Per-kernel author's choice.** No policy.

## Decision outcome

Chosen option: **Option 3 — f32 internal, f64 boundary**, plus an
explicit pinned-constants list and a workspace-wide SoA layout convention.

### Numeric type policy

- Inner-loop kernels (`crates/vernier-core/src/similarity/*`) compute
  intermediate values in `f32` where the algorithm's error budget allows
  it. Specifically:
  - bbox IoU: `f32` (the intersection / union ratio is robust at `f32`
    relative to the tolerance budget).
  - segm RLE counts and area: `u32` for run lengths, `u64` for cumulative
    area sums (areas can exceed 4e9 pixels for large images), IoU result
    cast to `f32`.
  - OKS per-keypoint distance squared: `f32`.
- The matrix produced by every `Similarity::compute` is `ArrayViewMut2<f64>`
  at the boundary, regardless of the internal type. The cast happens at
  the last write. `f32 → f64` is exact and lossless, so this introduces no
  rounding error relative to a hypothetical pure-`f64` kernel.
- Matching, accumulation, and summarization are entirely `f64`. The score
  array `dt_scores: &[f64]` is the canonical type. Cumulative TP/FP, the
  recall integration, the precision envelope, and the eps guard are all
  `f64`.

### Layout policy

- Detection and ground-truth annotation collections are stored SoA at the
  dataset level. `BboxBatch { x, y, w, h: Vec<f32> }`, not `Vec<Bbox>`.
- The single `Arc<Vec<AnnotationData>>` plus index ranges from the dataset
  trait (Phase 1 Week 1 of the roadmap) is SoA at one level up — the
  enum payload is per-annotation, but the by-image and by-category
  groupings are contiguous slice ranges, so iteration over a category
  is a contiguous sweep.
- `Vec<f32>` lanes for SIMD kernels are naturally aligned to the
  architecture's vector width via `pulp::Arch` (per ADR-0003); kernel code
  does not assume alignment beyond what the allocator provides.

### Pinned constants and primitives

These constants and primitives are load-bearing for parity. They are pinned
in source as named constants in `crates/vernier-core/src/parity.rs` (a new
module introduced by this ADR), with a doc comment citing the pycocotools
quirk ID.

| Name | Value | Quirk | Notes |
|---|---|---|---|
| `PARITY_EPS` | `f64::EPSILON` (= 2.22e-16) | C8 | Substitute for `np.spacing(1)`. Identical bits on every supported platform. |
| `IOU_BOUNDARY_EPS` | `1e-10` | B1 | Used as `min(t, 1.0 - IOU_BOUNDARY_EPS)` for initial best-IoU. |
| `OKS_AREA_EPS` | `f64::EPSILON` | F2 | Divide-by-zero guard for OKS area. |
| `IOU_THRESHOLDS` | `linspace(0.5, 0.95, 10)` | L1 | Constructed via the same formula pycocotools uses (not arange). |
| `RECALL_THRESHOLDS` | `linspace(0.0, 1.0, 101)` | L2 | Same. |

The `linspace` function used to construct the threshold arrays must
reproduce numpy's `np.linspace(start, stop, num, endpoint=True)` to all
bits. A unit test in `parity.rs` asserts this against a fixture extracted
from numpy 2.0+.

### Pinned algorithms

- **Argsort.** Score sorts use a stable sort that orders ties by input
  position. This matches numpy's `kind='mergesort'`. In Rust, the chosen
  primitive is `slice::sort_by` with `Ord::cmp` on `(NotNan<f64>, usize)`
  pairs, where the `usize` is the input index. Strict mode uses input
  position as the tiebreaker; `corrected` mode (per ADR-0002) breaks ties
  by `(score, ann_id)`. Quirk A1.
- **Bucket lookup for recall thresholds.** Equivalent to numpy's
  `searchsorted(rc, recThrs, side='left')`. The Rust primitive is
  `partition_point` over the cumulative-recall slice, with the predicate
  `|r| r < threshold`. This produces identical indices. Quirk C1.

### Aligned-tier tolerance

When an aligned implementation deviates from a strict-tier implementation,
the deviation is bounded by **`4 * f64::EPSILON` relative**, asserted in the
parity harness. Rows in the survey marked aligned must demonstrate they
meet this bound, in their unit tests, before merge. Rows that cannot
(none anticipated as of this ADR) are reclassified strict.

### Consequences

- **Positive.** Type and layout decisions are made once. Reviewers point
  at this ADR rather than re-arguing per PR. SIMD throughput is doubled
  on the inner loops where it matters. Pinned constants give parity tests
  a fixed target. Aligned tolerance is a single number, not a per-row
  argument.
- **Negative.** Mixed precision means bugs at the boundary (forgetting to
  cast `f32 → f64` before `partition_point`, or vice versa) will be
  silent until a parity fixture catches them. We mitigate with type
  signatures and a `#[deny(clippy::cast_possible_truncation)]` lint
  attribute on the relevant modules.
- **Neutral.** The `parity.rs` module becomes a load-bearing dependency for
  every algorithm crate. Changes to it require an ADR on top of this one.

## Pros and cons of the options

### Option 1 — all-f64

- 👍 No mixed-precision boundaries.
- 👎 Halves SIMD lane count; gives back the throughput gain that motivated
  the Rust port.

### Option 2 — all-f32

- 👍 Maximum SIMD throughput.
- 👎 Strict-mode parity fails at the accumulator. Recall thresholds and
  eps are not bit-equal at f32. Out of scope.

### Option 3 (chosen) — f32 internal, f64 boundary

- 👍 SIMD throughput where it matters; bit-equal at the boundary; clear
  type policy.
- 👎 Mixed-precision boundary requires discipline.

### Option 4 — author's choice

- 👍 Maximum local flexibility.
- 👎 Drift is silent; parity tests turn into a per-kernel debugging
  treadmill.

## Links and references

- ADR-0001 — Record architecture decisions.
- ADR-0002 — Three-tier parity model (strict / aligned / corrected).
- ADR-0003 — `pulp` for stable-Rust SIMD with runtime dispatch.
- `docs/engineering/pycocotools-quirks.md` — quirk citations (A1, B1, C1,
  C8, F2, L1, L2).
