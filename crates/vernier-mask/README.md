# vernier-mask

[![Crates.io](https://img.shields.io/crates/v/vernier-mask.svg)](https://crates.io/crates/vernier-mask)
[![Docs.rs](https://docs.rs/vernier-mask/badge.svg)](https://docs.rs/vernier-mask)

Pure-Rust [COCO](https://cocodataset.org/) mask kernels: RLE codec, polygon
rasterizer, and binary-mask ops. A leaf crate of the
[vernier](https://github.com/NoeFontana/vernier) evaluation library, designed to
be useful on its own.

## Why a separate crate

The mask data layer is reusable beyond evaluation: annotation tools,
training-data loaders, dataset converters, and custom evaluators all need to
read or write COCO RLE without pulling in pycocotools' C extension.
`vernier-mask` exposes the same primitives in stable, safe Rust:

- `Rle` and the `encode_counts` / `decode_counts` codec for COCO's
  comma-less RLE counts string.
- Polygon-to-mask rasterization (`raster::polygon_to_rle`) matching pycocotools'
  even-odd fill rule.
- Mask ops: `intersect_area_offsets` (bbox-cropped pairwise IoU areas),
  `erode_chebyshev_ball` (boundary-band erosion), `boundary_band` (boundary IoU
  precomputation).

## What's different from pycocotools.mask

vernier-mask returns typed `Result<_, MaskError>` instead of pycocotools' `0` /
`-1` / empty-RLE sentinels. Dimension mismatches, malformed RLE strings, and
ambiguous polygon input all surface as named error variants you can match on.
The `corrected` quirk dispositions (H1, H2, I2, I6, K1) under
[ADR-0002](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0002-three-tier-parity-model.md)
are documented in
[`docs/engineering/pycocotools-quirks.md`](https://github.com/NoeFontana/vernier/blob/main/docs/engineering/pycocotools-quirks.md).

## Installation

```toml
[dependencies]
vernier-mask = "0.0.1"
```

Stable Rust only (MSRV in workspace `rust-toolchain.toml`, currently 1.83). No
dependencies beyond `thiserror`.

## Minimal usage

```rust
use vernier_mask::{decode_counts, encode_counts, Rle};

// Round-trip a COCO RLE string
let counts = "PPYL2";
let decoded = decode_counts(counts.as_bytes())?;
let rle = Rle { size: [240, 320], counts: decoded };
let re_encoded = encode_counts(&rle.counts);
assert_eq!(re_encoded, counts.as_bytes());
```

## Architectural placement

Per [ADR-0009](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0009-split-mask-kernels-into-vernier-mask-crate.md),
this is a leaf crate: `vernier-core` and `vernier-panoptic` consume it,
nothing the other way around. The segm `Similarity` impl that consumes these
primitives lives in `vernier-core::similarity::segm`; the matching engine,
accumulator, and summarizer are unaware of `vernier-mask`.

## License

Dual-licensed under MIT or Apache-2.0, at your option. See
[`LICENSE-MIT`](https://github.com/NoeFontana/vernier/blob/main/LICENSE-MIT) and
[`LICENSE-APACHE`](https://github.com/NoeFontana/vernier/blob/main/LICENSE-APACHE)
in the workspace root.
