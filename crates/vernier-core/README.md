# vernier-core

[![Crates.io](https://img.shields.io/crates/v/vernier-core.svg)](https://crates.io/crates/vernier-core)
[![Docs.rs](https://docs.rs/vernier-core/badge.svg)](https://docs.rs/vernier-core)

Pure-Rust core for the [vernier](https://github.com/NoeFontana/vernier) evaluation library.

This crate is the source of truth for vernier's instance-paradigm evaluation
semantics — bounding-box AP, segmentation AP, boundary-IoU AP, and OKS keypoints.
It links no Python and is usable directly from Rust binaries, CLI tools, or
embedded contexts.

It also covers two cross-paradigm surfaces:

- **Streaming** — fold per-image evals incrementally and finalize on demand
  ([ADR-0013](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0013-streaming-evaluator.md)).
- **Distributed partials** — encode/decode the rkyv wire format for sharding
  evaluation across ranks
  ([ADR-0031](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0031-dist-eval.md),
  [ADR-0032](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0032-dist-eval-paradigms.md)).

## Why a separate crate

The Python wheel (`pip install vernier`) is a thin PyO3 shim over this crate.
By design, every evaluation decision lives here. `vernier-ffi` is a
data-conversion layer; `vernier-cli` orchestrates eval invocations; both depend
on `vernier-core` and never the other way around.

If your project is pure-Rust — a robotics replay pipeline, a Rust-side training
benchmark, a custom evaluation tool — `vernier-core` is the entry point.

## Installation

```toml
[dependencies]
vernier-core = "0.0.1"
```

Stable Rust only (MSRV in `rust-toolchain.toml`, currently 1.83).

## Minimal usage

```rust
use vernier_core::{
    evaluate_bbox, AreaRange, CocoDataset, CocoDetections, EvalDataset,
    EvaluateParams, ParityMode,
};

let gt: CocoDataset = serde_json::from_slice(&std::fs::read("gt.json")?)?;
let dt: CocoDetections = serde_json::from_slice(&std::fs::read("dt.json")?)?;
let params = EvaluateParams::default()
    .with_parity_mode(ParityMode::Strict)
    .with_max_dets(vec![1, 10, 100]);

let summary = evaluate_bbox(&gt.into_eval_dataset(), &dt, &params)?;
println!("{}", summary.pretty_lines().join("\n"));
```

A worked example is in [`examples/cache_speedup_val2017.rs`](examples/cache_speedup_val2017.rs).

## Parity contract

`vernier-core` reproduces `pycocotools==2.0.11` semantics under the three-tier
parity model in
[ADR-0002](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0002-three-tier-parity-model.md).
Each documented quirk is dispositioned as `strict` (bit-equal),
`aligned` (within tolerance), or `corrected` (opt-in fix). The full disposition
table lives in
[`docs/engineering/pycocotools-quirks.md`](https://github.com/NoeFontana/vernier/blob/main/docs/engineering/pycocotools-quirks.md).

## License

Dual-licensed under MIT or Apache-2.0, at your option. See
[`LICENSE-MIT`](https://github.com/NoeFontana/vernier/blob/main/LICENSE-MIT) and
[`LICENSE-APACHE`](https://github.com/NoeFontana/vernier/blob/main/LICENSE-APACHE)
in the workspace root.
