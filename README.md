# vernier

[![CI](https://github.com/NoeFontana/vernier/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/NoeFontana/vernier/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/vernier.svg?label=pypi%20%7C%20vernier)](https://pypi.org/project/vernier/)
[![crates.io vernier-core](https://img.shields.io/crates/v/vernier-core.svg?label=crates.io%20%7C%20vernier-core)](https://crates.io/crates/vernier-core)
[![crates.io vernier-mask](https://img.shields.io/crates/v/vernier-mask.svg?label=crates.io%20%7C%20vernier-mask)](https://crates.io/crates/vernier-mask)
[![crates.io vernier-cli](https://img.shields.io/crates/v/vernier-cli.svg?label=crates.io%20%7C%20vernier-cli)](https://crates.io/crates/vernier-cli)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

> High-performance, parity-preserving COCO-style evaluation for object detection,
> instance segmentation, and keypoints. Rust core, Python frontend, optional CLI.

**Status:** early development. Public API is unstable and the project is
pre-1.0. See `docs/adr/` for the design decisions that are shaping it.

## Why vernier

Existing COCO evaluation tooling presents a forced choice: the reference
`pycocotools` is correct-by-definition but slow and unmaintained; faster
reimplementations exist but each comes with its own drift from the reference.
vernier aims to provide a single library that is:

- **Bitwise identical** to the pycocotools reference on the minimal API
- **Fast** — Rust core, SIMD-friendly data layout, zero-copy on the hot path
- **Honest about quirks** — extended API offers corrected definitions
  alongside the reference, with the difference documented per-quirk
- **Embeddable** — pure-Rust core usable from CLI, ROS2 nodes, or any other
  Rust program without dragging Python along

## Install

**Python** (wheels for linux x86_64/aarch64 glibc+musl, macOS x86_64/arm64,
windows x64):

```bash
pip install vernier
```

**Rust library**:

```bash
cargo add vernier-core
```

**Rust CLI** — installs the `vernier` binary:

```bash
cargo install vernier-cli
vernier eval --gt instances_val2017.json --dt predictions.json --iou-type bbox
```

The umbrella `vernier` crate name on crates.io is held as a `0.0.0`
placeholder; `vernier-core` is the real Rust entry point. See
[`docs/engineering/registry-reservations.md`](docs/engineering/registry-reservations.md)
for the rationale.

## Layout

```
crates/
  vernier-core/     pure Rust evaluation logic; no Python dependency
  vernier-mask/     pure Rust COCO RLE codec, polygon rasterizer, mask ops (ADR-0009)
  vernier-ffi/      PyO3 bindings; data conversion only, no business logic
  vernier-cli/      `vernier` binary — workspace member per ADR-0015
python/
  vernier/          thin Python wrapper; the user-facing API lives here
tools/
  reservations/     placeholder packages holding registry names; outside the workspace
docs/
  adr/              Architecture Decision Records
  ...               Diátaxis-organized documentation
tests/
  rust/             Rust integration tests
  python/           Python tests against the FFI boundary
```

## Quickstart (development)

Prerequisites:

- [Rust stable](https://rustup.rs/) (`rustc >= 1.83`, pinned in `rust-toolchain.toml`)
- [uv](https://docs.astral.sh/uv/) for the Python toolchain (Python `>= 3.10`)
- [just](https://github.com/casey/just) for task running
- [`cargo-nextest`](https://nexte.st/) for the Rust test runner
- [`cargo-deny`](https://github.com/EmbarkStudios/cargo-deny) for `just audit`

```bash
# One-time setup
just bootstrap

# Iterate
just develop      # fast incremental rebuild
just test         # Rust + Python tests
just lint         # clippy + ruff + pyright (read-only, CI-equivalent)
just fmt          # auto-format everything
just audit        # cargo-deny check
```

## Project governance

Architectural decisions are recorded in `docs/adr/`. The first ADR
(`0001-record-architecture-decisions.md`) establishes the process itself.
Significant changes — anything that affects the public API, the FFI boundary,
or the parity contract — start as an ADR draft.

## License

Dual-licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT license](LICENSE-MIT) at your option.

## Third-party code

vernier vendors a small number of test-only reference implementations
to support parity testing. None of this code is included in published
wheels or linked into the Rust binary. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the full
inventory and license attributions.
