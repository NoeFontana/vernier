# vernier

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

## Layout

```
crates/
  vernier-core/   pure Rust evaluation logic; no Python dependency
  vernier-ffi/    PyO3 bindings; data conversion only, no business logic
python/
  vernier/        thin Python wrapper; the user-facing API lives here
docs/
  adr/            Architecture Decision Records
  ...             Diátaxis-organized documentation
tests/
  rust/           Rust integration tests
  python/         Python tests against the FFI boundary
```

## Quickstart (development)

Prerequisites: [Rust stable](https://rustup.rs/), [uv](https://docs.astral.sh/uv/),
[just](https://github.com/casey/just), and `cargo-nextest`.

```bash
# One-time setup
just bootstrap

# Iterate
just develop      # fast incremental rebuild
just test         # Rust + Python tests
just lint         # clippy + ruff + pyright (read-only, CI-equivalent)
just fmt          # auto-format everything
```

## Project governance

Architectural decisions are recorded in `docs/adr/`. The first ADR
(`0001-record-architecture-decisions.md`) establishes the process itself.
Significant changes — anything that affects the public API, the FFI boundary,
or the parity contract — start as an ADR draft.

## License

Dual-licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT license](LICENSE-MIT) at your option.
