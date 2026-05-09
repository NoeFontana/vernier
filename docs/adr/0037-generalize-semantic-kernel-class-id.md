# ADR-0037: Generalize the semantic kernel over class-id type

- **Status:** proposed
- **Date:** 2026-05-09
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

## Context and problem statement

ADR-0028 §"Algorithm" pinned the semantic confusion-matrix kernel to
`fn accumulate_confusion(gt: &[u32], dt: &[u32], ...)`. At the time
this was a deliberately conservative choice: u32 is the canonical
class-id width across the rest of vernier (panoptic segment ids, the
distributed-eval partials format, the FFI boundary type), and the
v1 surface only needed one monomorphization.

Two forces have shifted since:

1. **Real-image label maps decode at u8.** PNG semantic label maps —
   the workload in PR-B7's val2017 perfect-DT smoke and in every
   panoptic-derived semantic cache — store a single byte per pixel.
   Forcing a `astype(np.uint32)` cast at the FFI boundary means
   walking the same buffer twice (once to widen, once to fold) and
   holding 4× the memory. On val2017 (5000 images, ~270k px each)
   that's ~10 GB of bandwidth nobody asked for.

2. **A fused-decode FFI path is queued (PR #2).** The same perf
   round that motivated the streaming OOM fix (PR-B7) wants a Rust
   `evaluate_semantic_from_pngs(...)` that decodes libpng into a
   row-buffer and folds in the same walk. Decoding into a `Vec<u8>`
   and then upcasting to `Vec<u32>` before the kernel walks it is
   exactly the bandwidth waste a fused path is meant to eliminate.

The kernel's hot loop is one `as usize` index per pixel into a
`(n_classes, n_classes)` `u64` matrix. The `as usize` is identical
on `u8`, `u16`, and `u32` — at the assembly level the cast is free
on smaller types. Pinning the slice element type to u32 forces every
caller to pay the upcast even when it's algorithmically unnecessary.

## Decision drivers

- The fused-PNG-decode path (PR #2) needs a kernel that walks `&[u8]`
  without an intermediate `Vec<u32>` materialization.
- u32 must remain the canonical wire format for the array-input FFI
  and for the distributed-eval partials format (ADR-0032). No
  hard break to existing callers.
- The kernel is the single source of truth for the parity contract
  (ADR-0028). It must remain `#![forbid(unsafe_code)]` and the
  per-pixel arithmetic must not change — only the element type.
- Monomorphization bloat must be bounded: at most three concrete
  versions of a ~40-line function.

## Considered options

1. **Option A — Keep the kernel at `&[u32]`. Cast inside the FFI submit.**
   Status quo plus a friendlier error path. Kernel unchanged.
2. **Option B — Generalize the kernel over a `ClassId` trait
   (u8/u16/u32).** Walks at native dtype; widens to `u32` per pixel
   for ignore-label comparison and matrix indexing.
3. **Option C — Generalize over an arbitrary `Into<u32>` bound.**
   Same shape as B but more permissive (`i16`, `usize`, etc. would
   compile).

## Decision outcome

Chosen option: **Option B**, because it lets the upcoming fused-PNG
path walk a `&[u8]` row-buffer without a 4× memory expansion *and*
keeps the kernel surface type-safe (only the three natural class-id
widths compile). The amendment to ADR-0028 is mild: the algorithm
is unchanged, only the element type widens via a sealed trait.

### Consequences

- **Positive:**
  - PR #2's `evaluate_semantic_from_pngs` becomes a one-line `accumulate_confusion(gt_u8, dt_u8, ...)` on the libpng row buffer; no upcast.
  - The same kernel serves PR #4's loosened `submit(arr: u8|u16|u32)` FFI without dispatching to three different bodies.
  - PR-B7's existing parity tests (`u8` PNG path, `u32` ndarray path) continue to compile by changing the call site dtype only.
  - Cache-friendlier on the `u8` path: 4× fewer bytes touched per pixel.
- **Negative:**
  - Three monomorphizations of `accumulate_confusion` plus a `ClassId` trait ($\sim$ 30 LoC delta in the kernel module). Codegen bloat is bounded — the function is small.
  - Existing call sites that passed integer literals (`accumulate_confusion(&[0, 1], &[0, 1], ...)`) need an explicit dtype suffix (`&[0u32, 1]`), since the generic eats the default-i32 inference. Caught at compile time, low touch.
- **Neutral:**
  - `ignore_label` stays `Option<u32>`. Strictly more permissive than parameterizing on `Option<T>` and lets `u8` callers pass a u8 sentinel via `Some(255_u32)` without a separate impl.
  - The partials format and the array-input FFI path keep `u32` as the wire type. No ADR-0032 amendment needed.

## Pros and cons of the options

### Option A — keep kernel at `&[u32]`

- 👍 Zero amendment to ADR-0028. Smallest possible diff.
- 👎 Doesn't unblock the fused-PNG path. The cast moves from Python to Rust but doesn't disappear.
- 👎 Forces every future fused-decode entry point to materialize a `Vec<u32>` even when the source data is u8.

### Option B — `ClassId` trait over u8/u16/u32 (chosen)

- 👍 Fused-PNG path walks at native u8 dtype.
- 👍 Unblocks PR #4's multi-dtype `submit(arr)` without three near-duplicate FFI bodies.
- 👍 Sealed surface — only the three natural widths implement the trait.
- 👎 Three monomorphizations. Codegen bloat is real but bounded (~120 LoC of asm in release).
- 👎 Mild ADR-0028 amendment.

### Option C — `T: Into<u32>`

- 👍 Same perf shape as B with no new trait.
- 👎 Surface widens unintentionally (i8, isize, etc. compile). Loses the "class ids are unsigned" invariant we get from `ClassId`.
- 👎 Worse error messages for the rare misuse case.

## Links and references

- ADR-0028 — semantic-segmentation evaluation paradigm (the algorithm
  snippet pinning `&[u32]` is the clause this ADR amends).
- ADR-0032 — distributed-eval partials format (u32 wire type
  unaffected).
- PR-B7 — val2017 perfect-DT smoke (current array-input path).
- PR #2 — `evaluate_semantic_from_pngs` (the fused-decode path that
  consumes this generalization).
