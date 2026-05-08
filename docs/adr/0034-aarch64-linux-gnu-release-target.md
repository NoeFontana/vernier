# ADR-0034: Add `aarch64-unknown-linux-gnu` to the `vernier-cli` GitHub Release target list

- **Status:** proposed
- **Date:** 2026-05-08
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

ADR-0015 ratified `vernier-cli` as a workspace binary and committed the
project to publishing pre-built binaries for **three** target triples on
GitHub Releases: `x86_64-unknown-linux-gnu`, `aarch64-apple-darwin`, and
`x86_64-pc-windows-msvc`. That list was the minimum surface that covered
the three personas the CLI was originally written for (CI-shell users on
x86 Linux runners, macOS-on-Apple-Silicon developers, Windows desktop
users) and was deliberately conservative — Homebrew, conda-forge, and
other distribution channels were explicitly out of scope.

What that list does *not* cover is **`aarch64-unknown-linux-gnu`** — Linux
on 64-bit ARM. Three audience segments fall through this gap:

1. **Edge / robotics hardware.** NVIDIA Jetson devices (Orin, Xavier, Nano)
   are aarch64-Linux. They are also a non-trivial slice of the audience for
   a fast COCO-evaluation binary — replay pipelines on a real-world robot
   benchmark are exactly the "out-of-Python, in-shell" persona ADR-0015 was
   written for, just on aarch64 silicon instead of x86.
2. **AWS Graviton (and other aarch64 cloud).** Graviton instance types are
   the price-leader on AWS for general-purpose Linux compute and are
   widely used for CI, batch evaluation jobs, and dataset-management
   tooling. A team that runs their CI on Graviton today has to either
   `cargo install vernier-cli` (which assumes a Rust toolchain on the CI
   image) or fall back to the wheel (which assumes a Python interpreter,
   defeating the CLI's reason to exist).
3. **GitHub Actions aarch64 Linux runners.** These became generally
   available in 2024 and are now the standard on-ramp for OSS aarch64 CI.
   Any project that wants to consume `vernier-cli` from a `runs-on:
   ubuntu-latest-arm64` job today has the same `cargo install` /
   wheel-fallback choice.

The gap is mechanical, not architectural. The wheel matrix already builds
on `aarch64-unknown-linux-gnu` (`.github/workflows/wheels.yml` lines
24–25 cover Linux x86_64 and aarch64 with manylinux + musllinux variants),
so the project already has the cross-compilation infrastructure in CI;
adding `aarch64-unknown-linux-gnu` to the binary release matrix is one
cargo-dist target entry and an additional GitHub Actions runner allocation
per release.

CLAUDE.md's ADR policy is explicit: "ADRs are immutable once `accepted`;
supersede with a new ADR rather than edit." This ADR follows that rule by
extending — not editing — ADR-0015's target list. ADR-0015's status stays
`accepted`; this ADR is a focused follow-up that adds a single target.

## Decision drivers

- **ADR-0015's reason-to-exist applies symmetrically across architectures.**
  The "skip the Python interpreter" value proposition is no less true on
  aarch64 Linux than on x86 Linux. A Jetson developer who has to install
  Python + numpy + the wheel just to evaluate a robot's detection output
  is in exactly the position ADR-0015 was written to fix.
- **The build infrastructure already exists.** The wheel matrix builds
  aarch64 manylinux + musllinux wheels today. cargo-dist can target the
  same triple using the same cross-compilation toolchain. No new CI
  runner type, no new container image, no new toolchain pin.
- **Marginal CI cost is small.** One additional cross-compile job per
  release tag. Bandwidth for the additional release asset is in single-
  digit MB. Neither is a meaningful budget line.
- **Symmetry with `cargo binstall`.** `cargo binstall vernier-cli` works
  by mapping the host triple to a release asset. A user on
  `aarch64-unknown-linux-gnu` who runs `cargo binstall` today falls back
  to a from-source build, defeating the binstall reason-to-exist for a
  user who picked binstall specifically to avoid a Rust toolchain.
- **No commitment beyond the wheel matrix.** This ADR adds one target
  that the wheel matrix already covers. It does not commit to expanding
  the matrix further; in particular, `x86_64-apple-darwin` and
  `aarch64-pc-windows-msvc` are explicitly out of scope (Intel Macs and
  Windows-on-ARM are real but smaller audiences and the cost / value
  trade-off is different — both are individual follow-up ADRs if and when
  the audience signal warrants).
- **No retroactive guarantees.** This ADR applies prospectively from the
  next release tag forward. Older releases (0.0.1) are not back-filled
  with aarch64-Linux binaries; a user on that platform installing 0.0.1
  continues to fall back to `cargo install` from source.

## Considered options

1. **Add `aarch64-unknown-linux-gnu` only.** Single-target extension. The
   audience exists, the build path exists, the cost is one matrix entry.
2. **Add `aarch64-unknown-linux-gnu` and `x86_64-apple-darwin` together.**
   Symmetric expansion — covers Linux ARM64 *and* Intel Macs. Intel Macs
   are a shrinking share of the macOS audience but still real.
3. **Add a broader matrix in one stroke** (Linux ARM64, macOS x86_64,
   Windows ARM64, Linux musl variants for the binary). Maximally generous
   but several of those targets have unclear audience signal and the
   maintenance overhead grows non-linearly.
4. **Defer entirely; rely on `cargo install`.** Treat the gap as
   acceptable because Rust users can compile from source. Loses the
   binstall persona and the "no toolchain on the CI image" use case.
5. **Wait for explicit user demand.** Only add the target after a user
   reports the gap. Operationally simple but punts on a known audience
   the project already builds wheels for.

## Decision outcome

**Chosen option: 1 — add `aarch64-unknown-linux-gnu` only.**

The audience is concrete (Jetson, Graviton, GitHub Actions ARM64
runners), the cost is small (one cargo-dist matrix entry, one CI job),
and the build infrastructure already exists in the wheel matrix. The
expansion is the smallest move that closes the most acute gap.

Adding `x86_64-apple-darwin` (Option 2) is plausible but the trade-off
is different in shape: Intel Macs are a shrinking share, the macOS dual-
arch picture has historical complications (Universal binaries vs.
separate triples), and the audience signal is weaker than for aarch64
Linux. We treat that as a separate decision; if it lands, it lands as a
sibling ADR.

The broader matrix (Option 3) is rejected as scope creep — each
additional target carries its own audience-validation question and
should be argued separately.

Deferring entirely (Option 4) keeps `cargo install` as the only path for
aarch64-Linux users, which both undermines `cargo binstall` and forces a
Rust toolchain on every Jetson / Graviton CI image. The cost-of-delay is
real and the cost-of-action is small; deferring is the wrong call.

Waiting for explicit demand (Option 5) is operationally tempting but the
project already publishes aarch64-Linux *wheels* — which means the
audience is presumed to exist for the Python frontend but not for the
CLI. That asymmetry is unprincipled.

### Consequences

- **Positive.** Jetson / Graviton / GHA aarch64 users get a pre-built
  `vernier` binary they can curl-pipe-bash or `cargo binstall`. The CI-
  shell persona ADR-0015 was written for now applies symmetrically
  across the two Linux architectures the wheel matrix already covers.
  Symmetry between the wheel and the binary on aarch64 Linux is
  restored.
- **Negative.** The release matrix grows by one job. Each release tag
  takes marginally longer to complete (the additional cross-compile is
  in parallel with the existing matrix, so the wall-clock impact is
  bounded by the longest existing job, not additive). One additional
  binary per release accumulates in the GitHub Releases store. The
  binstall heuristic now has one more target to resolve, which we still
  expect to work via the default templates (per ADR-0015's binstall
  rule).
- **Neutral.** ADR-0015's three-target list stays the historical
  decision; this ADR is recorded as the extension. Future consumers of
  the target list look at both ADRs.

### Implementation

- The cargo-dist configuration in workspace `Cargo.toml`'s
  `[workspace.metadata.dist]` lists four targets:
  `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`,
  `aarch64-apple-darwin`, `x86_64-pc-windows-msvc`.
- The release runbook (`docs/engineering/release-runbook.md`) is updated
  to list the four binaries in the post-tag verification checklist.
- The smoke test step in `wheels.yml` is unaffected (it gates wheel
  publishing, not binary asset upload). cargo-dist's separate
  `release.yml` workflow handles the binary matrix.
- Per ADR-0015's binstall rule: we add `[package.metadata.binstall]` to
  `crates/vernier-cli/Cargo.toml` *only if* cargo-dist's default
  binstall-conventional naming heuristic fails for the new target. We
  expect it to work; we verify with `cargo dist plan` before tagging.

### What this ADR explicitly does *not* decide

- **`x86_64-apple-darwin`.** Intel Macs are a separate decision, weighed
  on different grounds (shrinking audience, Universal-binary option vs.
  separate triple). If they land, they land as a follow-up ADR.
- **`aarch64-pc-windows-msvc`.** Windows-on-ARM is a real platform with
  unclear vernier-audience signal today. Same disposition as Intel Macs.
- **musl variants of the Linux binary.** The wheel matrix builds both
  manylinux and musllinux variants because the wheel ABI requires it.
  The CLI binary is a single statically-linked executable; we ship the
  glibc variant as the default and let static-linking advocates either
  `cargo install --target=x86_64-unknown-linux-musl` or wait for a
  follow-up ADR if a musl audience materializes.
- **GPG signing of release assets.** ADR-0015 mentioned signing as a
  later step; this ADR does not commit to it on any timeline. Signing,
  if it happens, applies to all targets uniformly and is a separate
  decision.
- **Conda-forge / Homebrew / OS package formulae.** Out of scope per
  ADR-0015, unchanged here.

## Pros and cons of the options

### Option 1 (chosen) — add `aarch64-unknown-linux-gnu` only

- 👍 Smallest change that closes the most acute gap.
- 👍 Build infrastructure already exists in the wheel matrix.
- 👍 Audience is concrete and named (Jetson, Graviton, GHA ARM64).
- 👎 Doesn't help Intel Mac users; doesn't help Windows-on-ARM users.

### Option 2 — add aarch64-Linux *and* x86_64-macOS

- 👍 Symmetric expansion; covers two distinct audiences in one move.
- 👎 Intel Mac audience is shrinking and has Universal-binary complications.
- 👎 Couples two decisions that have different audience-signal shapes.

### Option 3 — broader matrix in one stroke

- 👍 Maximally generous; no follow-up ADRs needed for a while.
- 👎 Several targets have unclear audience signal; the maintenance
  overhead grows non-linearly with target count; cargo-dist's binstall
  heuristic gets more chances to fail.

### Option 4 — defer; rely on `cargo install`

- 👍 Operationally simple.
- 👎 Forces a Rust toolchain on every aarch64-Linux CI image; defeats
  binstall on the same triples; asymmetric with the wheel matrix.

### Option 5 — wait for explicit user demand

- 👍 Lowest commitment; reactive.
- 👎 The audience already exists (the wheel matrix presumes it). Waiting
  for a bug report is unprincipled when the cost of action is small.

## Links and references

- ADR-0001 — Record architecture decisions (§"Add or remove a build
  target"). This ADR adds a build target.
- ADR-0015 — `vernier-cli` workspace binary. The "Distribution" section
  pins the original three-target list; this ADR extends it.
- `.github/workflows/wheels.yml` — the existing wheel matrix that
  already builds `aarch64-unknown-linux-gnu`.
- `docs/engineering/release-runbook.md` — updated alongside this ADR's
  implementation to reflect the four-target binary matrix.
