# ADR-0003: Use `pulp` for stable-Rust SIMD with runtime CPU dispatch

- **Status:** accepted
- **Date:** 2026-04-26
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

The hot loops of the evaluator — bbox IoU, segm IoU over RLE intersections,
OKS over per-keypoint distances — are vectorizable. On COCO-scale workloads
SIMD is the difference between "comparable to pycocotools" and "5–10× faster
than pycocotools". The performance ceiling of vernier as a project is set
by how well these loops vectorize.

Three constraints frame the choice of how to write them:

1. `rust-toolchain.toml` pins stable Rust. The `std::simd` (a.k.a.
   `portable_simd`) module is unstable and would force a nightly toolchain
   for every contributor and every CI runner.
2. vernier's wheels target multiple x86_64 microarchitectures (SSE4.2,
   AVX2, AVX-512) and aarch64 (NEON). Compiling for a specific feature set
   either pessimizes new hardware or breaks on old hardware. Runtime
   dispatch is the only way to ship one wheel that takes advantage of what
   the host actually has.
3. The evaluation crate is `#![forbid(unsafe_code)]`. Hand-rolled
   `core::arch` intrinsics require `unsafe` blocks at every call site.

Without an explicit decision, the path of least resistance is "use whatever
feels simplest in the moment", which produces a mix of `core::arch`,
auto-vectorization hopes, and one-off intrinsics by file. That mix is
slow to maintain, hard to benchmark consistently, and incompatible with
the unsafe-forbidden policy.

## Decision drivers

- Stable Rust only (per `rust-toolchain.toml`).
- `unsafe` is forbidden in `vernier-core` (per its `lib.rs`).
- Runtime CPU-feature dispatch — one wheel, multiple SIMD variants, picked
  at process start.
- Cross-architecture: x86_64 (SSE/AVX2/AVX-512) and aarch64 (NEON), not
  just x86 with NEON as an afterthought.
- The crate must remain reviewable. Inner loops written once, in
  source-portable form, not duplicated per ISA.
- The dependency must be load-bearing for Faer or another well-maintained
  numerical-Rust ecosystem so it's not a one-author crate at risk of
  abandonment.

## Considered options

1. **`std::simd` (`portable_simd`).** The intended long-term answer; nightly-only.
2. **`wide` crate.** Stable-Rust SIMD types (`f32x8`, `f64x4`, …); compile-time
   lane selection; no runtime dispatch.
3. **`safe_arch` crate.** Stable-Rust safe wrappers around `core::arch`
   intrinsics; per-ISA at the call site.
4. **Hand-rolled `core::arch` with `multiversion` for dispatch.** Lowest
   level; per-target codegen via `#[multiversion]`; requires unsafe.
5. **`pulp` crate (Faer's portable SIMD abstraction).** Stable-Rust
   architecture-agnostic SIMD with runtime dispatch via `Arch::new()`;
   cross-architecture (x86 + aarch64); load-bearing in Faer.

## Decision outcome

Chosen option: **`pulp`**.

`pulp` provides a portable `Simd` trait abstraction, a runtime `Arch`
selector that picks the best ISA available on the host at process start,
and a single `with_simd` style entrypoint that the inner-loop code is
written against. It compiles on stable Rust, requires no `unsafe` at the
call site of vernier's algorithms, and is part of the Faer ecosystem
(`faer` itself, `faer-core`, etc., depend on it), so its maintenance is
load-bearing for a major numerical-Rust project rather than a one-off.

`pulp` is added once to the workspace `Cargo.toml`'s
`[workspace.dependencies]` table, pinned by minor version. Crates that
need it inherit via `pulp.workspace = true`; crates that don't (e.g.
`vernier-ffi` if it stays purely data-conversion) leave it out.

Inner-loop policy:

- Hot kernels live in `crates/vernier-core/src/<module>/simd.rs` files,
  written against `pulp::Simd`.
- The dispatching `Arch` value is constructed once at module load (or via
  a `OnceLock`) and reused; `Arch::new()` is not called per call.
- A scalar reference implementation is kept side by side (`<module>/scalar.rs`)
  and used by tests and microbenchmarks to verify equivalence and to
  measure speedup. Divan microbenches assert the SIMD path is at least 2×
  the scalar path on the CI runner; failure is a regression.
- `vernier-core` keeps `#![forbid(unsafe_code)]`. `pulp` itself contains
  `unsafe` internally (it has to, to call ISA intrinsics) but the
  forbidden-unsafe rule applies to **vernier's source**, not to its
  dependency tree. This is consistent with how the lints are already
  configured: `unsafe_code` is denied workspace-wide for our code, and
  `cargo-deny` audits the dep graph for advisories.

The chosen toolchain (stable Rust) is unchanged. The MSRV in
`rust-toolchain.toml` and `rust-version` in `Cargo.toml` are unchanged.

### Consequences

- **Positive.** One source for each kernel; runtime dispatch picks the best
  ISA without a fat-binary build. Stable Rust. No nightly. No call-site
  `unsafe`. Maintained as part of Faer.
- **Negative.** New top-level dep. `pulp`'s API has evolved across versions
  and pins matter; bumping it ripples to every kernel. The abstraction is
  thinner than `std::simd` will eventually be — some patterns (e.g.
  gather/scatter) require dropping to scalar fallbacks today.
- **Neutral.** When `std::simd` stabilizes, this ADR may be superseded by
  a migration ADR. The kernel layout (`<module>/simd.rs` + scalar
  reference) is structured to make that migration mechanical.

## Pros and cons of the options

### Option 1 — `std::simd` / `portable_simd`

- 👍 Long-term answer; eventually no third-party dep needed.
- 👎 Nightly-only as of 2026-04. Forces the project to move off stable
  Rust. Out of scope for the project's current platform constraints.

### Option 2 — `wide`

- 👍 Stable Rust; small dep; well-maintained.
- 👎 Compile-time lane selection only. To get AVX-512 on hosts that have it,
  the build target must enable AVX-512, which then doesn't run on hosts
  that don't. No single-wheel runtime dispatch story.

### Option 3 — `safe_arch`

- 👍 Stable Rust; safe wrappers around `core::arch`; no dependency tree
  pressure.
- 👎 Per-ISA at the call site: x86 code is one block, aarch64 another.
  Source duplication grows linearly with kernels × ISAs. No portable
  abstraction.

### Option 4 — Hand-rolled `core::arch` + `multiversion`

- 👍 Most direct control over codegen; runtime dispatch via attribute.
- 👎 Requires `unsafe` blocks at every intrinsic call. Conflicts with
  `#![forbid(unsafe_code)]` in `vernier-core`. We would have to relocate
  every SIMD kernel to a sibling crate that allows unsafe, splitting the
  evaluator's logic across crates for purely-mechanical reasons.

### Option 5 (chosen) — `pulp`

- 👍 Stable Rust; portable; runtime dispatch; cross-arch; safe at the call
  site; load-bearing in Faer.
- 👎 New top-level dep, with its own versioning cadence. API surface is
  smaller than `std::simd`'s ultimate scope.

## Links and references

- ADR-0001 — Record architecture decisions.
- `pulp` crate — <https://docs.rs/pulp>.
- Faer — <https://github.com/sarah-quinones/faer-rs>, the primary user
  whose maintenance load-bears `pulp`.
- `rust-toolchain.toml` — pins stable Rust; precondition for this ADR.
