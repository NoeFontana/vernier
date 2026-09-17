# ADR-0056: Pin the no-FP-contraction invariant for the bbox IoU kernel

- **Status:** proposed
- **Date:** 2026-09-17
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

ADR-0008 moved the bbox IoU kernel to `f64` end-to-end so that every
cell is bit-equal to pycocotools' `maskUtils.iou`. That ADR is precise
about *width* (f64, not f32) and silent about *association and
rounding count* — and the second property is doing just as much work as
the first.

`iou_pair` (`crates/vernier-core/src/similarity/bbox.rs`) computes the
union denominator as:

```rust
let d_area = dw * dh;
let inter  = iw * ih;
let denom  = g_area + d_area - inter;   // left-associative
```

That expression contains two multiply-then-add patterns a compiler is
free to fuse into a single FMA whenever floating-point *contraction* is
enabled:

| site  | source                                 | fused form            |
| ----- | -------------------------------------- | --------------------- |
| `sum` | `g_area + d_area`, `d_area = dw * dh`  | `fma(dw, dh, g_area)` |
| `sub` | `sum - inter`, `inter = iw * ih`       | `fnmadd(iw, ih, sum)` |

Fusing *improves* accuracy — it drops an intermediate rounding — and for
exactly that reason it breaks parity. pycocotools' reference C
(`bbIou`) evaluates `u = da + ga - i` as three separately-rounded
operations, and the C reference is the definition, not an
approximation of one.

The magnitude is not cosmetic. A random search over COCO-scale boxes
(x ∈ [0, 640], y ∈ [0, 480], w/h ∈ [1, 300]) finds contraction-sensitive
pairs at a rate of roughly one in a million, moving the final IoU by
1–2 ULP. ADR-0008 is the record that this matters: it documents
concrete val2017 cells where the matching loop's "later equal wins"
rule (quirk **B2**, `strict`) resolved a crowd-vs-non-crowd tie on an
8-ULP difference, and the wrong resolution rippled into `dtMatches` and
AP. 1–2 ULP is inside that blast radius.

Three properties make this worth an explicit decision rather than an
assumption:

1. **The guarantee is inherited, not stated.** rustc has no
   `-ffast-math` and leaves LLVM's `fp-contract` off for Rust codegen,
   so the unfused form is what ships today. Nothing in this repository
   says so, tests it, or would notice if it changed.
2. **The fused instruction is always available on the shipped path.**
   Per ADR-0003 the kernel runs inside `pulp::Arch::dispatch`, which
   compiles the loop body with `fma` (x86) or `neon` (aarch64) target
   features enabled. On aarch64, `fmadd` is baseline regardless.
3. **The failure mode is per-architecture and silent.** We ship
   aarch64 wheels (`.github/workflows/wheels.yml`, ADR-0034). A
   contraction difference would produce a wrong AP on one wheel and not
   another, with nothing panicking and no error raised — only the last
   digits moving, on the subset of images with overlapping crowd GT.

A related gap surfaced while writing this up: **no pre-existing test
exercised the dispatched path at all.** Every bbox unit test used a
cell of at most 3×3, and `SMALL_CELL_THRESHOLD = 32` routes those to
the plain scalar body. The `pulp` path — the one compiled with FMA
available, and the one that runs on dense cells in production — had
zero bit-exactness coverage.

## Decision drivers

- Strict-mode parity (ADR-0002) is the project's headline claim; a
  silent per-architecture divergence in the single most-reused kernel
  is the worst available way to lose it.
- Stable Rust only. `-Zfp-contract` is nightly-gated, so there is no
  build-level switch available to us.
- The pin must live where it cannot be satisfied vacuously: a test that
  passes because it stopped testing anything is worse than no test.
- The bbox kernel is shared. `BboxIou::compute` backs standalone bbox
  eval, TIDE, LRP, LVIS, and the streaming evaluator; a single pin
  covers all of them.

## Considered options

1. **Status quo — trust rustc's default.**
2. **Golden bit-pattern regression tests, asserted on x86_64 *and*
   aarch64, in the release profile.**
3. **Defeat contraction structurally in the source** — force each
   rounding with `core::hint::black_box` (or equivalent) between steps.
4. **Set `-Cllvm-args` / `RUSTFLAGS` in `.cargo/config.toml`.**

## Decision outcome

Chosen option: **Option 2.**

A `fp_contract_pin` module in
`crates/vernier-core/src/similarity/bbox.rs` holds a golden table of
`(gt, dt, is_crowd, expected_bits)` cases and asserts them:

- on the **scalar** path (`G·D = 1`),
- on the **dispatched** path, with the grid tiled until `G·D` clears
  `SMALL_CELL_THRESHOLD` — sized *from the constant*, so raising the
  threshold cannot silently stop covering the `pulp` path,
- with a **self-validation** test asserting the table discriminates
  against each fusion site independently (recomputing both fused forms
  via `mul_add` and requiring at least one case to differ per site).
  Without this, a future edit could replace the constants with
  contraction-insensitive values and leave a green test protecting
  nothing.

The table also pins the two semantics that sit next to the rounding
question and are easy to "simplify" away: crowd GT divides by `d_area`
alone (quirk **E1** — it is IoA, not IoU) and non-overlapping pairs
produce exact `+0.0` rather than a computed negative or `-0.0` (quirks
**I3** / **I4**).

CI gains a `test-rust-cross-arch` job running the pin on
`ubuntu-latest` and `ubuntu-24.04-arm`. Two deliberate differences from
the existing `test-rust` job:

- **`--release`.** Contraction and auto-vectorization only happen under
  optimization. `test-rust` runs the default debug profile and would
  pass even on a toolchain that contracts, so a debug-only pin is
  security theatre. The job builds with the plain release profile — not
  `target-cpu=native` — matching the shipped wheel per this repo's
  bench discipline.
- **`--no-tests=fail`.** A nextest filter expression that matches
  nothing exits 0, so a rename would otherwise convert this job into a
  vacuous green.

### Scope: which kernels are actually exposed

- **bbox** — exposed on its output path. Pinned here.
- **segm** (`SegmIou`) and **boundary** (`BoundaryIou`) — **not**
  exposed. Both accumulate intersection and union as integer `u64`
  areas and perform a single `as f64` divide at the end
  (`similarity/segm.rs`, `similarity/boundary.rs`). Integer arithmetic
  cannot contract.
- **The shared bbox prefilter** (`BboxIou::compute_overlap_mask`) —
  not exposed; it contains no multiply, only `min`/`max`/`sub` and a
  comparison.
- **OKS** (`similarity/oks.rs`) — **exposed**, at `dx * dx + dy * dy`
  in both the in-bbox and F4-surrogate branches, which can fuse to
  `fma(dx, dx, dy * dy)`. Not pinned by this ADR; recorded below as a
  follow-up so it is tracked rather than forgotten.

### Consequences

- **Positive.** The rounding contract of the most-reused kernel is
  stated and enforced, on both shipped architectures, in the profile
  that ships. The `pulp` dispatch path gains its first bit-exactness
  coverage.
- **Positive.** The pin is cheap: six cases, five tests, microseconds
  of runtime, and one small CI job whose aarch64 leg is free for public
  repositories.
- **Negative.** Golden bit patterns are opaque to a reader who does not
  know why they are there. Mitigated by keeping the derivation, the
  fusion-site table, and the self-validation test next to the
  constants.
- **Negative.** One extra CI job (two legs). Accepted: it is the only
  thing in CI that executes vernier code on aarch64 at all — `wheels.yml`
  cross-builds aarch64 but never runs the result.
- **Neutral.** No kernel code changed. This ADR pins existing behavior;
  it does not alter any output.

## Pros and cons of the options

### Option 1 — trust rustc's default

- 👍 Zero work; correct today.
- 👎 Correct today by accident. Nothing detects a toolchain default
  change, an added flag, or a well-meaning `mul_add` "optimization".
- 👎 Leaves the dispatched path untested regardless.

### Option 2 (chosen) — cross-arch golden bits in release

- 👍 Detects every route to contraction, including ones we have not
  thought of, because it asserts the *output*.
- 👍 Runs on the architecture where the risk is highest.
- 👎 Golden constants need their rationale carried alongside them.

### Option 3 — defeat contraction structurally with `black_box`

- 👍 Makes the guarantee local to the source, independent of CI.
- 👎 `black_box` is explicitly a no-guarantee hint, so it would be
  trading one unstated assumption for another.
- 👎 It is an optimization barrier in the hottest inner loop in the
  crate, and it would block the auto-vectorization ADR-0003 and
  ADR-0008 both rely on. Paying a throughput cost for a guarantee we
  already have, in order to avoid writing a test, is the wrong trade.

### Option 4 — `RUSTFLAGS` in `.cargo/config.toml`

- 👍 Would state the intent at the build level.
- 👎 There is no stable flag to state it with; `-Zfp-contract` is
  nightly-only.
- 👎 `.cargo/config.toml` flags are overridden wholesale by a
  `RUSTFLAGS` environment variable, so downstream builders (including
  `maturin` in `wheels.yml`) could drop them without noticing. A
  guarantee that a downstream env var can silently remove is not a
  guarantee.

## Open questions and follow-ups

- **OKS contraction site.** `dx * dx + dy * dy` in `similarity/oks.rs`
  is the one remaining `f64` contraction site in the crate. It feeds an
  `exp()` and then a sum, so a 1-ULP input shift propagates rather than
  cancels. It should get the same treatment; it is deferred here only
  to keep this ADR's change set to one kernel.
- **The reference side of the claim.** Strict parity is a two-sided
  statement, and this ADR only pins *our* side. pycocotools' `bbIou` is
  C compiled by the wheel builder; GCC's default for C is
  `-ffp-contract=fast`. On x86-64 the baseline ISA has no FMA, so
  stock manylinux x86_64 wheels cannot contract. On aarch64, `fmadd`
  *is* baseline. Whether the published `pycocotools` aarch64 wheel
  contracts in `bbIou` is unverified — it needs an aarch64 host to
  check, which was not available when this was written. If it does,
  strict bbox parity on aarch64 is a claim about a moving reference and
  needs its own disposition row in
  `docs/engineering/pycocotools-quirks.md`. Flagging it as a known
  risk rather than asserting either way.

## Links and references

- ADR-0002 — Three-tier parity model (`strict` is what is at stake).
- ADR-0003 — `pulp` for stable-Rust SIMD with runtime dispatch (the
  reason `fma` / `neon` target features are enabled on the shipped
  path).
- ADR-0004 — Numerical layout policy.
- ADR-0008 — Bbox IoU computes in `f64` end-to-end (this ADR pins the
  unstated rounding-count half of that decision).
- ADR-0034 — `aarch64-unknown-linux-gnu` release target.
- ADR-0049 — Bench CPU budget (why the pin job builds plain release
  rather than `target-cpu=native`).
- `crates/vernier-core/src/similarity/bbox.rs` — the kernel and the
  `fp_contract_pin` module.
- `.github/workflows/ci.yml` — job `test-rust-cross-arch`.
- `docs/engineering/pycocotools-quirks.md` — quirks **B2**, **E1**,
  **I3**, **I4**.
