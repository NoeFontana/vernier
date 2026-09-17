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
exercised the dispatched path at all.** The largest cell any bbox unit
test built was 5×5 (`overlap_mask_survivor_bit_matches_full_iou`, with
one 4×4 alongside it), and `SMALL_CELL_THRESHOLD = 32` routes
everything below 32 to the plain scalar body, so all of them took it. The `pulp` path — the one compiled with FMA
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
- The bbox kernel is shared — *once the duplicates are gone*.
  `BboxIou::compute` backs standalone bbox eval, TIDE cross-class, LRP,
  LVIS and the streaming evaluator, so a single pin covers all of them.
  It did not cover TIDE's **same-class** matching, which carried a
  second, private `f64` union expression
  (`tide::assignment::bbox_iou_pair`) that no pin reached; this ADR's
  change set removes it (see §"One kernel, or the pin is a half-pin").
  A pin on a shared kernel is only worth what the sharing is worth.

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
  against each fusion site independently *and* against both sites fused
  together (recomputing all three forms via `mul_add` and requiring at
  least one case to differ per form).

  The third arm is what makes the test load-bearing. The realistic
  decay is not "someone invents contraction-insensitive constants"; it
  is "a future toolchain contracts, the golden test goes red, and a
  contributor refreshes the constants from the new output on both
  architectures". A table regenerated that way is still sensitive to
  each site *individually* — the counts merely swap, 2/1 → 1/2 — so
  single-site assertions stay green on a pin that now pins the
  contracted form. It is, by construction, identical to the both-fused
  form, which is exactly what the third assertion rejects.

  Backing that up, the constants carry an explicit **provenance** note:
  they are the *reference's* values, derived from pycocotools' `bbIou`,
  and are never to be regenerated from vernier's own output. A
  disagreement between this crate and the table means the crate has
  diverged, not that the table is stale.

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

### One kernel, or the pin is a half-pin

A golden test on `BboxIou::compute` pins every caller of
`BboxIou::compute` and nothing else. That distinction was not academic:
`tide::assignment::same_class_match_one_category` built its own IoU
matrix from a private `bbox_iou_pair`, a verbatim restatement of the
same union expression. All three fusion candidates were present there,
and unlike pycocotools' C — where every multiply feeding the union has a
second consumer, see below — each multiply in that copy had exactly one
use, so not even that argument applied. Contraction could have moved
TIDE's same-class IoU by 1-2 ULP, flipped a match at `t_f`, and silently
re-attributed error types, with the bbox golden test green on both
architectures throughout.

**This ADR's change set reroutes that call site through
`BboxIou::compute`** rather than adding a second golden table. The two
expressions were verified bit-identical before the switch, which they
are by construction:

- same association — `(g_area + d_area) - inter` on both sides
  (the TIDE oracle writes `dt_area + gt_area - inter`; `+` is
  commutative and exactly rounded, so the bits agree);
- same intersection — `(min - max).max(0.0)` with the same operand
  order, so quirks **I4** and the exact `+0.0` carry over;
- same zero guard — `denom > 0.0` and `union <= 0.0` are complements on
  every non-NaN input, and a NaN denominator cannot arise from finite
  box coordinates (the kernel's guard is the safer of the two, returning
  `+0.0` where the old copy returned `NaN`);
- the **one** semantic the kernel adds is quirk **E1**, the crowd IoA
  denominator, which the TIDE oracle (`oracle.py::bbox_iou`, ADR-0021)
  does not have — crowd GTs reach TIDE through `gt_ignore`, not through
  the denominator. The call site therefore passes `is_crowd: false` on
  both sides, which selects the union branch and leaves every value
  unchanged.

`tide::assignment`'s `tide_same_class_iou_matches_oracle_bitwise`
asserts the first three points against a transcription of the oracle
over a 64×64 deterministic COCO-scale sample (which also clears
`SMALL_CELL_THRESHOLD`, so it runs the dispatched body), and
`tide_same_class_iou_suppresses_the_crowd_asymmetry` pins the fourth.
Output is unchanged; `just test-parity` confirms it.

The general rule this leaves behind: a new `f64` bbox-IoU expression
anywhere in the crate is a new unpinned kernel, not a duplication of
code. Route it through `BboxIou::compute` or extend the table.

### Scope: which kernels are actually exposed

- **bbox** — exposed on its output path. Pinned here, at
  `BboxIou::compute`, which after this change set is the crate's only
  `f64` bbox-IoU expression.
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

### The reference side: pin to the x86-64 reference on every architecture

Strict parity is a two-sided statement, and the golden table pins only
*our* side. pycocotools' `bbIou` is C compiled by the wheel builder, and
GCC's default for C is `-ffp-contract=fast`. On x86-64 the baseline ISA
has no FMA, so a stock manylinux x86_64 wheel *cannot* contract. On
aarch64, `fmadd` is baseline and nothing stops it.

**Decision: vernier's bbox IoU output is architecture-independent.** The
golden bits encode the non-contracting (x86-64) evaluation of
`da + ga - i`, and that same table is asserted on every architecture we
ship. We do not make vernier's output track a per-architecture
reference.

The rationale is that the alternative is worse in every direction. An
arch-dependent kernel would mean the same wheel version produces
different APs on different machines: `from_partials` (ADR-0031 /
ADR-0032) would stop being well-defined across a heterogeneous cluster,
a cached result would stop being portable, and a published number would
need its CPU recorded next to it. A 1-2 ULP tie against one build of the
reference is a much smaller cost than losing reproducibility of our own
output.

#### What the reference actually does (verified 2026-09-17)

This ADR originally flagged aarch64 contraction in `bbIou` as a plausible
but unverified risk. It has since been checked directly, by disassembling
the published binaries rather than by running them — no aarch64 host
required:

| pycocotools 2.0.11 binary (arm64)               | toolchain    | fused ops in `bbIou` |
| ----------------------------------------------- | ------------ | -------------------- |
| `cp312-abi3-manylinux_2_17_aarch64`              | GCC, glibc   | none                 |
| `cp39-manylinux_2_17_aarch64`                    | GCC, glibc   | none                 |
| `cp312-abi3-musllinux_1_2_aarch64`               | GCC, musl    | none                 |
| `macosx-universal2`, **arm64 slice**             | Apple clang  | none                 |
| `win_arm64`, `_mask.pyd`                         | MSVC         | none                 |

`bbIou` is an exported symbol, so each check is exact: the kernel emits
separate `fmul` / `fadd` / `fsub` / `fdiv` and **no** `fmadd` / `fmsub` /
`fnmsub` / `fmla`.

**What this does and does not claim.** pycocotools 2.0.11 publishes on
the order of twenty files that carry arm64 code; the five above are a
sample, not the set. They were chosen to span every compiler family
pycocotools ships arm64 with — GCC against glibc and against musl,
Apple clang, MSVC — so the result is not a statement about one
toolchain's flags, and in particular it is not limited to manylinux.
Every arm64 binary inspected is unfused; nothing here rules out an
uninspected file, and nothing enforces the property on an ongoing basis
(see the last Consequences bullet).

The mechanism is partly structural rather than wholly lucky, but weaker
than a first reading suggests. In `maskApi.c:125-136`:

```c
BB D=dt+d*4; da=D[2]*D[3]; ...
i=w*h; u = crowd ? da : da+ga-i; o[g*m+d]=i/u;
```

`i` is used both in `u` and as the numerator of `i/u`, and `da` is used
both in `da+ga` and as the whole of `u` on the crowd branch (quirk
**E1**). Contracting either one would force the compiler to keep the
rounded value *and* recompute the product, so `-ffp-contract=fast` has
nothing to gain there and leaves the arithmetic alone. That argument is
solid, and it covers two of the three candidates.

It does **not** cover the third. `ga = G[2]*G[3]` has exactly one
consumer — the `da+ga` in the union — so the multi-use argument says
nothing about it, and `fmadd(G[2], G[3], da)` would be a legal
contraction. What blocks it in practice is that `ga` is loop-invariant
and gets hoisted to its own `fmul` above the inner `d` loop; folding it
back into the loop body to save a rounding would cost a multiply per
iteration, which no cost heuristic wants. That is a compiler *judgement*
call, not a correctness constraint, and a different optimizer could
decide otherwise. It happens to be unfused in all five binaries above;
treat that as an empirical result with a plausible explanation rather
than as a guarantee.

#### The residual divergence, and where it is recorded

What is *not* covered is a pycocotools built from the sdist on the user's
own machine (`pip install --no-binary pycocotools`, a distro package, a
conda-forge build) with a compiler that contracts more eagerly than the
wheel builder's GCC. Nothing in this repository can constrain that
toolchain. On such a build, on aarch64, a locally compiled `bbIou` may
contract and land 1-2 ULP away from vernier on roughly one box pair in
10^6 — a divergence introduced by the user's compiler, not by vernier.

That is recorded as quirk **I7** in
`docs/engineering/pycocotools-quirks.md`, dispositioned **strict**: the
parity claim is against `pycocotools==2.0.11` *as published on PyPI*,
which is the artifact `pyproject.toml` pins and the harness installs, and
against that artifact vernier is bit-equal on both architectures. The
two-tier vocabulary (ADR-0002, as amended 2026-05-10) has no cell for a
per-architecture disposition and this row does not need one — the
condition is "reference rebuilt with a different compiler", not
"architecture", and it is out of scope of the pin either way.

### Consequences

- **Positive.** The rounding contract of the most-reused kernel is
  stated and enforced, on both shipped architectures, in the profile
  that ships. The `pulp` dispatch path gains its first bit-exactness
  coverage.
- **Positive.** The pin is cheap: six cases, five tests in
  `fp_contract_pin` plus two bit-equality tests on the rerouted TIDE
  path, microseconds of runtime, and one small CI job whose aarch64 leg
  is free for public repositories.
- **Negative.** Golden bit patterns are opaque to a reader who does not
  know why they are there. Mitigated by keeping the derivation, the
  fusion-site table, and the self-validation test next to the
  constants.
- **Negative.** One extra CI job (two legs). Accepted: it is the only
  thing in CI that executes vernier code on aarch64 at all — `wheels.yml`
  cross-builds aarch64 but never runs the result.
- **Neutral.** No kernel *arithmetic* changed. One call site moved:
  TIDE's same-class matching now calls `BboxIou::compute` instead of a
  private copy of the same expression, which is bit-identical (with the
  E1 branch suppressed, as the TIDE oracle requires) and brings that
  path under the pin. This ADR pins existing behavior; it does not
  alter any output.
- **Neutral.** The reference side is now checked rather than assumed,
  but nothing enforces it on an ongoing basis — a future pycocotools
  release could be built by a compiler that contracts. Re-running the
  disassembly is part of the work of bumping the `pycocotools` pin,
  which is already an ADR-level decision.

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
  (both the in-bbox and F4-surrogate branches) is the remaining exposed
  `f64` contraction site among the similarity kernels — the one known
  unpinned site, now that TIDE's private bbox copy is gone. It feeds an
  `exp()` and then a sum, so a 1-ULP input shift propagates rather than
  cancels. It should get the same treatment; it is deferred here only
  to keep this ADR's change set to one kernel.
- **The reference side of the claim.** *Resolved* — see
  §"[The reference side: pin to the x86-64 reference on every
  architecture](#the-reference-side-pin-to-the-x86-64-reference-on-every-architecture)".
  vernier pins the non-contracting form everywhere; the published
  aarch64 wheels were disassembled and do not contract; the residual
  locally-recompiled case is quirk **I7**.

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
- `crates/vernier-core/src/tide/assignment.rs` — the same-class matching
  path rerouted onto that kernel, and the bit-equality tests that hold
  it there.
- `.github/workflows/ci.yml` — job `test-rust-cross-arch`.
- `docs/engineering/pycocotools-quirks.md` — quirks **B2**, **E1**,
  **I3**, **I4**, and **I7** (the reference-side row this ADR adds).
- `docs/migrate/from-pycocotools.md` — §"Bit-for-bit, and against
  which build", the user-facing statement of the same caveat.
