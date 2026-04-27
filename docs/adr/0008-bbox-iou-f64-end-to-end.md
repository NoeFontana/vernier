# ADR-0008: Bbox IoU computes in `f64` end-to-end

- **Status:** accepted
- **Date:** 2026-04-26
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

ADR-0004 established a "f32 internal, f64 boundary" policy across all
similarity kernels, citing throughput from doubled SIMD lane counts and
the claim that "f32 → f64 is exact and lossless, so this introduces no
rounding error relative to a hypothetical pure-f64 kernel."

That last claim is wrong. Widening f32 to f64 is exact, but the f32
*computation* has already lost bits relative to a pure-f64 computation
of the same algebraic expression. The widened result carries those lost
bits forward as an f64 value that does not equal the f64-computed
reference.

This is not theoretical. The whole-dataset perfect-DT smoke
(`test_coco_val2017_bbox_parity_perfect_dt`) surfaced concrete cases
where a single bbox sits inside a crowd region. Pycocotools' f64
`maskUtils.iou` produces non-trivial values like
`0.9999999999999991` (crowd path, asymmetric `inter / dt_area`) and
`0.9999999999999983` (non-crowd path, self-intersect through the
`(min - max).max(0)` formulation). The matching loop's "later equal
wins" rule (quirk **B2**, strict) picks the crowd because its IoU is
strictly greater by 8 ULPs of f64.

Vernier's f32 kernel produced `1.0` exactly for the non-crowd
self-intersect (the f32 subtraction cancels cleanly when both boxes
are equal) but `1.0000001192…` for the crowd path (the iw subtraction
picks up an extra ULP on the f32 representation of `dxa + dw`). After
widening to f64, these are *different* bit patterns from
pycocotools — and they happen to flip the matching decision, which
ripples through `dtMatches` and breaks strict parity on a long tail of
val2017 cells.

The cells affected are exactly the configurations that motivated
strict-mode parity in the first place: overlapping crowd and non-crowd
GTs, where the matching outcome is decided by the last bits of an IoU
calculation.

## Decision drivers

- The strict-mode parity contract from ADR-0002 is the headline claim;
  if it fails on real datasets, the project loses its raison d'être.
- The bbox kernel is fast enough already that f32 lane-count gains
  matter at the noise level for typical detection workloads (5000
  images × 80 categories × ~100 dets = ~10⁵ small IoU matrices, each
  a few microseconds). The accumulator and Python boundary dominate.
- ADR-0004 is immutable per ADR-0001; the right move is a superseding
  ADR scoped narrowly to the bbox clause.
- Future similarity kernels (segm, OKS) face the same trade-off but
  with different error surfaces — segm uses integer run-length
  arithmetic, OKS uses Gaussian decay where f32 precision is fine.
  The supersession is bbox-only.

## Considered options

1. **Keep ADR-0004 as written; accept the divergence as `aligned`.**
   Document a tolerance and stop claiming strict parity for bbox.
2. **Compute bbox IoU in f64 end-to-end.** Supersede ADR-0004's bbox
   clause; pulp dispatch still applies but with f64 lane counts.
3. **Branch on parity mode at runtime.** `ParityMode::Strict` uses
   f64; `ParityMode::Aligned` / `Corrected` use f32.

## Decision outcome

Chosen option: **Option 2 — bbox IoU in f64 end-to-end**, with this
ADR superseding only the *bbox-IoU* line in ADR-0004's "Numeric type
policy". All other ADR-0004 clauses (segm RLE counts at u32, OKS
distances at f32, SoA layout, pinned constants, aligned-tier
tolerance) stand.

The kernel still runs inside `pulp::Arch::dispatch` per ADR-0003.
AVX2 has 4 f64 lanes (vs 8 f32), AVX-512 has 8 f64 lanes (vs 16),
NEON has 2 f64 lanes (vs 4) — half the throughput of f32 but the
same auto-vectorization story. For the bbox kernel this is ample;
the kernel is not the workload bottleneck.

### Consequences

- **Positive.** Strict-mode parity holds bit-for-bit on COCO val2017
  for bbox. The headline parity claim is real. Future bbox-IoU
  divergences are real bugs to track down, not "well, f32 rounding".
- **Negative.** Half the SIMD lane count for the bbox kernel
  specifically. We accept this — bbox IoU is not a hot enough loop
  to justify breaking parity for the gain.
- **Neutral.** ADR-0004's per-kernel table now has one entry that
  reads "bbox IoU: f64 (per ADR-0008)" instead of "bbox IoU: f32".
  Reviewers checking type policy follow the supersession link.

## Pros and cons of the options

### Option 1 — accept divergence as aligned

- 👍 No code changes.
- 👎 Walks back the strict-mode parity claim that the project sells
  on. Failing the perfect-DT smoke right after declaring it the
  headline parity goal is not a recoverable position.

### Option 2 (chosen) — f64 end-to-end for bbox

- 👍 Strict parity actually holds. One narrow change. Easy to verify.
- 👎 Half the SIMD lane count for one kernel.

### Option 3 — runtime branch on parity mode

- 👍 Strict gets f64; aligned/corrected get f32 throughput.
- 👎 Two code paths to test. The f32 path's only advantage is
  throughput on a kernel that isn't the bottleneck. Not worth the
  maintenance cost.

## Links and references

- ADR-0001 — Record architecture decisions (immutability rule).
- ADR-0002 — Three-tier parity model (strict / aligned / corrected).
- ADR-0003 — `pulp` for stable-Rust SIMD with runtime dispatch.
- ADR-0004 — Numerical layout policy (superseded *only* on the
  bbox-IoU clause; all other clauses stand).
- `docs/engineering/pycocotools-quirks.md` — quirks **B2**, **E1**,
  **I3**, **I4** (the matching and IoU rules whose interaction with
  f32 rounding caused the divergence).
- `docs/engineering/coco-val-parity.md` — the test that surfaced the
  bug.
