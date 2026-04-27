# ADR-0009: Split mask kernels into a `vernier-mask` workspace crate

- **Status:** proposed
- **Date:** 2026-04-27
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

Phase 2 introduces COCO segmentation parity. The work breaks into two
distinct surfaces:

1. **A mask data layer.** RLE struct, the LEB128-style 6-bit string
   codec, encode / decode, area, bbox, intersection, union, merge, and
   the polygon and bbox rasterizers. The full set of `pycocotools.mask`
   entry points. This is data-and-codec code with no dependence on
   detection, evaluation, or scoring.
2. **A segm `Similarity` impl.** Glue that consumes the mask data layer
   and produces the GT × DT IoU matrix the matching engine reads. Tiny
   (a hundred-ish lines) and structurally identical to the bbox
   `Similarity` impl shipped in Phase 1.

These have different audiences. The mask data layer is reusable by any
Rust project that needs to read or write COCO masks (annotation tools,
training-data loaders, custom evaluators, ROS perception nodes). The
segm `Similarity` impl is meaningful only to vernier's matching engine.

We also have a registry constraint: `vernier-mask` was reserved on
crates.io in Phase 0 (`tools/reservations/crates/vernier-mask`,
version `0.0.0`). The reservation commits us to publishing a real crate
under that name.

The decision in front of us is the *boundary*: where exactly does
`vernier-mask` end, and what stays in `vernier-core`?

## Decision drivers

- **Single source of truth.** Every Phase 2 quirk (G1–G6, H1–H7,
  I1–I6, K1–K4) has a disposition row in
  `docs/engineering/pycocotools-quirks.md`. The implementation must map
  one-to-one to those rows, regardless of which crate owns the code.
- **Reusability outside vernier.** The mask codec is the only piece of
  pycocotools that has reuse value beyond evaluation. A Rust user with
  a COCO RLE who wants `decode + bbox + area` should not have to depend
  on the matching engine, accumulator, summarizer, or pyo3.
- **Wheel size and compile-time discipline.** The polygon rasterizer
  and string codec add several hundred lines of integer-heavy code that
  bbox-only users do not exercise. A separate crate gives Cargo a
  clean unit to cache and a clean unit to leave out of small builds.
- **Architectural acid test from Phase 1.** ADR-0005 locked the
  `Similarity` trait so new IoU types plug in without editing the
  matching engine. Phase 2 is the first test of that lock. We should
  not muddle the result by also restructuring how `Similarity` impls
  are organized.
- **Leaf dependency direction.** `vernier-mask` must not depend on
  `vernier-core`. The reverse is allowed and expected. A cycle is a
  red flag that the boundary is in the wrong place.
- **No-Python policy.** `vernier-mask` follows `vernier-core`:
  `#![forbid(unsafe_code)]`, no Python deps. PyO3 surfacing, when
  needed, lives in `vernier-ffi`.

## Considered options

1. **Module inside `vernier-core`.** Add `vernier-core::mask` and
   `vernier-core::similarity::segm`. No new crate. Possibly gated by a
   Cargo feature (`segm`).
2. **`vernier-mask` owns the codec; segm `Similarity` lives in
   `vernier-core`.** Mask data layer is a pure leaf crate; the segm
   IoU impl is a `Similarity` colocated with `BboxIou`.
3. **`vernier-mask` owns codec *and* segm `Similarity`.** The crate
   takes a dependency on the `Similarity` trait (which would have to
   move to a tiny `vernier-traits` crate, or `vernier-mask` takes a
   `vernier-core` dep, breaking leaf direction).

## Decision outcome

Chosen option: **Option 2.**

`crates/vernier-mask/` is added as a workspace member. Its public API
mirrors `pycocotools.mask` but Rusty:

- `Rle { counts: Vec<u32>, h: u32, w: u32 }` (field-compatible with
  the COCOAPI in-memory layout).
- Encode / decode against the 6-bit char string format (quirks
  G1–G3, K3).
- `area`, `bbox`, `intersection`, `union`, `merge` (quirks G4–G6,
  H2).
- `polygon_to_rle`, `bbox_to_rle`, `from_uncompressed_rle` factories
  (quirks H3–H6, K1, K2).
- `iou` for RLE × RLE with the bbox prefilter (quirk I1).
- Errors via a `MaskError` enum; no `0` / `-1` sentinels (quirks H1,
  H2, I2, I6 — `corrected` per ADR-0002).

The segm `Similarity` impl ships in `vernier-core::similarity::segm`,
depending on `vernier-mask` for the `Rle` type and IoU primitives.
The matching engine, accumulator, summarizer, and FFI surfaces are
**not edited** in Phase 2 — that is the architectural test of
ADR-0005.

The crate publishes to crates.io as `vernier-mask 0.1.0`, replacing
the `0.0.0` reservation. SemVer is independent of the wheel; the
wheel pins a compatible range.

`pulp::Arch::dispatch` (ADR-0003) is **not** committed to in this
ADR. RLE intersection / union is integer-bound and most of the
realistic per-image work is bookkeeping; if profiling on real
workloads shows a hot loop, SIMD lands in a follow-up under ADR-0003.

### Consequences

- **Positive.** Mask data layer is independently usable by Rust
  consumers — the only piece of vernier with broad reuse value
  outside evaluation. The `Similarity` trait passes its first
  extension test cleanly. Quirk rows G/H/I/K map one-to-one to
  modules in one crate.
- **Negative.** A second workspace crate to keep in sync (lints,
  edition, MSRV, license headers). We accept this — the workspace
  inheritance machinery already absorbs the per-crate boilerplate.
- **Neutral.** The reservation (`tools/reservations/crates/vernier-mask`)
  remains where it is, deliberately outside the workspace per
  `docs/engineering/registry-reservations.md`. The first real release
  steps over `0.0.0` to `0.1.0`.

## Pros and cons of the options

### Option 1 — module inside vernier-core

- 👍 No new Cargo unit; one place to read, one place to test.
- 👍 No leaf-direction question.
- 👎 Reusable mask code is buried under an evaluator. A Rust user who
  wants COCO RLE reads must depend on the matching engine,
  accumulator, summarizer, and (transitively, via dev-deps) the FFI
  test surface.
- 👎 The `segm` Cargo feature is a fault line that has to be
  maintained on every PR — the alternative (just always compile it)
  defeats the wheel-size point.

### Option 2 (chosen) — vernier-mask data layer; segm `Similarity` in vernier-core

- 👍 Clean leaf direction. `vernier-mask` is reusable. `vernier-core`
  is where every `Similarity` impl lives, matching ADR-0005.
- 👍 The G/H/I/K quirks all land in one crate. The segm parity bug
  hunt has a single grep target.
- 👎 Two crates instead of one. Boilerplate cost is real but bounded
  by workspace inheritance.

### Option 3 — vernier-mask owns segm `Similarity` too

- 👍 All segm-related code in one crate.
- 👎 Forces `vernier-mask` to depend on the `Similarity` trait, which
  drags in either a leaf-direction violation (`vernier-mask` →
  `vernier-core`) or a third crate (`vernier-traits`) whose only
  reason to exist is this dependency. Either resolution is worse than
  the small split in Option 2.
- 👎 `Similarity` impls would no longer be colocated, weakening the
  "one place to look" property the matching engine depends on.

## Links and references

- ADR-0001 — Record architecture decisions (build-target rule).
- ADR-0002 — Three-tier parity model (the dispositions of every
  Phase 2 quirk, including `corrected` for H1/H2/I2/I6/K1).
- ADR-0003 — `pulp` for stable-Rust SIMD with runtime dispatch (not
  invoked in this ADR; reserved for a follow-up if profiling warrants).
- ADR-0004 — Numerical layout policy (RLE counts at `u32` per the
  segm clause, still in force).
- ADR-0005 — `Similarity` trait and matching-engine API lock; the
  segm `Similarity` impl is the first non-bbox test of this lock.
- `docs/engineering/pycocotools-quirks.md` — quirks G1–G6, H1–H7,
  I1–I6, K1–K4.
- `docs/engineering/registry-reservations.md` — why the reservation
  package sits outside the workspace.
