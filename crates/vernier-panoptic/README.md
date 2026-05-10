# vernier-panoptic

[![Crates.io](https://img.shields.io/crates/v/vernier-panoptic.svg)](https://crates.io/crates/vernier-panoptic)
[![Docs.rs](https://docs.rs/vernier-panoptic/badge.svg)](https://docs.rs/vernier-panoptic)

Panoptic-quality (PQ) evaluation for the
[vernier](https://github.com/NoeFontana/vernier) library
([ADR-0025](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0025-panoptic-api.md)).

This crate is a sibling to `vernier-core`: both depend on `vernier-mask` as a
leaf, neither depends on the other. Panoptic and instance evaluation use
different matching rules and different data models — the architectural firewall
keeps the AP fold (`matching.rs` / `accumulate.rs` / `Similarity` trait) and
the PQ fold from accidentally drifting toward each other.

## Parity

vernier-panoptic reproduces
[`cocodataset/panopticapi`](https://github.com/cocodataset/panopticapi)
semantics under the parity model in
[ADR-0002](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0002-three-tier-parity-model.md):
strict mode is bit-equal at the per-class TP/FP/FN counts. The strict-mode
parity harness in `tests/python/parity_panoptic/` runs both libraries on every
fixture and diffs every intermediate.

## Installation

```toml
[dependencies]
vernier-panoptic = "0.0.1"
```

Stable Rust only (MSRV in `rust-toolchain.toml`, currently 1.83).

## Minimal usage

```rust
use vernier_panoptic::{evaluate, dataset::PanopticDataset, parity::ParityMode};

let gt = PanopticDataset::from_files(/* ... */)?;
let dt = PanopticDataset::from_files(/* ... */)?;
let summary = evaluate(&gt, &dt, ParityMode::Strict)?;
println!("PQ = {:.4}", summary.pq);
```

Per [ADR-0025](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0025-panoptic-api.md),
the kernel uses a single PNG decoder pin (`png` crate, ~150 KB) rather than the
`image` umbrella crate — the input shape is well-defined and an opinionated
decoder is the right fit. PNG decoding and the per-image matching are both
single-threaded by design.

## Performance

The val2017 perfect-DT panoptic workload runs ~32.3 s on
`vernier-panoptic` (post-2026-05 streaming runner + FxHash optimizations),
beating `panopticapi` 1.11×. See
[`docs/engineering/benchmarking/`](https://github.com/NoeFontana/vernier/tree/main/docs/engineering/benchmarking)
for the current snapshot and methodology.

## License

Dual-licensed under MIT or Apache-2.0, at your option.
