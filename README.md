# vernier

[![CI](https://github.com/NoeFontana/vernier/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/NoeFontana/vernier/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/vernier.svg?label=pypi%20%7C%20vernier)](https://pypi.org/project/vernier/)
[![crates.io vernier-core](https://img.shields.io/crates/v/vernier-core.svg?label=crates.io%20%7C%20vernier-core)](https://crates.io/crates/vernier-core)
[![crates.io vernier-mask](https://img.shields.io/crates/v/vernier-mask.svg?label=crates.io%20%7C%20vernier-mask)](https://crates.io/crates/vernier-mask)
[![crates.io vernier-cli](https://img.shields.io/crates/v/vernier-cli.svg?label=crates.io%20%7C%20vernier-cli)](https://crates.io/crates/vernier-cli)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

> High-performance, parity-preserving evaluation for object detection,
> instance / panoptic / semantic segmentation, and keypoints. Rust core,
> Python frontend, optional CLI.

**Status:** early development. Public API is unstable and the project is
pre-1.0. See `docs/adr/` for the design decisions that are shaping it.

## Three evaluation paradigms

vernier ships three sibling submodules — pick the one whose **input shape**
matches your model's output:

```python
import vernier

# Detections (bbox / segm / boundary / keypoints) with scores → AP fold
vernier.instance.Evaluator()

# RGB-encoded panoptic PNGs + segments_info JSON → PQ
vernier.panoptic.Evaluator()

# Single-channel class-id label maps → mIoU / FWIoU / pAcc / mAcc
vernier.semantic.Evaluator()
```

The submodules are mutually exclusive (different data models, different
matching rules, different parity oracles). See
[Three paradigms: instance, panoptic, semantic][three-paradigms] for when to
use which.

[three-paradigms]: docs/explanation/three-paradigms.md

## Why vernier

`pycocotools==2.0.11` is the reference for COCO evaluation and it is slow
and unmaintained. Faster reimplementations exist, but each silently fixes
some quirks and not others, leaving users to discover the divergences
empirically. vernier takes a third path:

- **Auditable parity.** Every divergence from `pycocotools` is filed in a
  quirks survey under one of three dispositions — `strict`, `aligned`, or
  `corrected` ([ADR-0002][adr-0002]). Strict mode reproduces `pycocotools`
  bit-for-bit; opt-in corrected fixes are itemized.
- **A unified evaluation toolkit.** Whether you need bbox / segm /
  keypoints AP, boundary IoU, panoptic PQ, semantic mIoU, or LVIS
  federated, it all lives inside vernier. Instead of wrestling with
  fragmented dependencies like `pycocotools`, `boundary-iou-api`,
  `panopticapi`, `lvis-api`, or `mmsegmentation` — and reconciling
  their competing JSON conventions — you get one consistent interface.
  Each has a per-paradigm migration guide under
  [`docs/migrate/`](docs/migrate/).
- **Drop-in shim.** `vernier.patch_pycocotools()` swaps the `COCOeval`
  symbol in place so existing pycocotools-based scripts switch with one
  line ([ADR-0007][adr-0007]).
- **Rust core.** The matching kernel is Rust with runtime SIMD dispatch;
  the FFI layer is data conversion only. The CLI ships as a static binary,
  so CI pipelines can call vernier without provisioning a Python
  interpreter.

[adr-0002]: docs/adr/0002-three-tier-parity-model.md
[adr-0007]: docs/adr/0007-patch-pycocotools-policy.md

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
