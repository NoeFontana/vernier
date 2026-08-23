# vernier

[![Crates.io](https://img.shields.io/crates/v/vernier.svg)](https://crates.io/crates/vernier)
[![Docs.rs](https://docs.rs/vernier/badge.svg)](https://docs.rs/vernier)

One dependency for the whole [vernier](https://github.com/NoeFontana/vernier)
evaluation toolkit — detection / instance-segmentation AP, panoptic quality,
semantic mIoU, COCO mask primitives, and the distributed-eval wire format.

This crate is a **facade**
([ADR-0048](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0048-vernier-facade-crate.md)).
It contains no code of its own: it re-exports the published paradigm crates
under one dependency and one module map. Its public API *is* the union of
theirs, by construction, so it cannot drift from them.

```toml
[dependencies]
vernier = "0.2"
```

## Module map

| Module | Crate | Paradigm |
|---|---|---|
| `vernier::instance` | [`vernier-core`](https://crates.io/crates/vernier-core) | Detection / instance-segmentation AP — bbox, segm, boundary, OKS keypoints, LVIS federated, LRP, TIDE, calibration |
| `vernier::mask` | [`vernier-mask`](https://crates.io/crates/vernier-mask) | COCO RLE codec, polygon rasterizer, mask ops |
| `vernier::panoptic` | [`vernier-panoptic`](https://crates.io/crates/vernier-panoptic) | Panoptic quality (PQ / SQ / RQ) — `panoptic` feature |
| `vernier::semantic` | [`vernier-semantic`](https://crates.io/crates/vernier-semantic) | Semantic segmentation (mIoU / FWIoU / pixel accuracy) — `semantic` feature |
| `vernier::partial` | [`vernier-partial`](https://crates.io/crates/vernier-partial) | Distributed-eval wire envelope shared by the three paradigms — `partial` feature |

The module-per-paradigm shape is forced rather than chosen: `vernier-core`,
`vernier-panoptic`, and `vernier-semantic` each export a `ParityMode` and a
`VERSION`, so a flat glob re-export would not compile. It lands on the same
shape
[ADR-0029](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0029-namespace.md)
chose for the Python surface (`vernier.{instance, panoptic, semantic}`), so one
mental model carries across both languages.

## Minimal usage

```rust
use vernier::mask::Rle;
use vernier::semantic::{kernel::accumulate_confusion, summarize, ConfusionMatrix, ParityMode};

// COCO RLE round-trip.
let rle = Rle::from_counts(4, 4, vec![4, 6, 6]);
assert_eq!(rle.area(), 6);

// Semantic mIoU over a 4-pixel, 2-class image.
let mut confusion = ConfusionMatrix::zeros(2);
accumulate_confusion(&[0u8, 0, 1, 1], &[0u8, 0, 1, 1], None, &mut confusion);
assert_eq!(summarize(confusion, ParityMode::Strict).miou, 1.0);
```

Each module's rustdoc carries a worked example; the leaf crates carry the
reference material.

## Features

All three are additive, re-export-only, and on by default. A feature here
changes what is *nameable*, never what is *computed* — none is ever forwarded
to a paradigm crate, so "one wheel, one behavior"
([ADR-0047](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0047-threading-model.md))
holds unchanged.

| Feature | Default | Gates |
|---|---|---|
| `panoptic` | on | `vernier::panoptic` |
| `semantic` | on | `vernier::semantic` |
| `partial` | on | `vernier::partial` |

`instance` and `mask` are unconditional: `vernier-core` already depends on
`vernier-mask` ([ADR-0009](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0009-split-mask-kernels-into-vernier-mask-crate.md)),
so gating `mask` would save no compilation, and making `instance` optional
would let `default-features = false` yield a crate with no public items. The
crate is non-empty in every reachable feature combination.

```toml
[dependencies]
vernier = { version = "0.2", default-features = false }   # instance + mask only
```

Depending on a leaf crate directly is equally supported and always was — the
facade is convenience for the common case, never a gate.

## What is not here

- **The CLI.** `cargo install vernier-cli` installs the `vernier` binary; this
  crate is the library. A library dep on the binary's package would drag `clap`
  into every consumer's tree
  ([ADR-0015](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0015-vernier-cli.md)).
- **The Python bindings.** `vernier-ffi` ships as the `vernier._core` extension
  module inside the `vernier` wheel (`pip install vernier`) and is not
  published to crates.io.
- **Logic of any kind.** `src/lib.rs` is the only source file and there never
  will be another.

Stable Rust only (MSRV in `rust-toolchain.toml`, currently 1.83).

## License

Dual-licensed under MIT or Apache-2.0, at your option. See
[`LICENSE-MIT`](https://github.com/NoeFontana/vernier/blob/main/LICENSE-MIT) and
[`LICENSE-APACHE`](https://github.com/NoeFontana/vernier/blob/main/LICENSE-APACHE)
in the workspace root.
