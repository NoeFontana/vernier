//! COCO mask kernels — RLE codec, polygon rasterizer, and mask ops.
//!
//! Per ADR-0009, this crate is a pure-Rust leaf: no Python deps, no
//! dependency on `vernier-core`. The mask data layer is reusable by
//! any Rust project that needs to read or write COCO RLE masks
//! (annotation tools, training-data loaders, perception nodes,
//! custom evaluators).
//!
//! The segm `Similarity` impl that consumes these primitives lives in
//! `vernier-core::similarity::segm`; the matching engine, accumulator,
//! and summarizer are unaware of this crate.
//!
//! Public surface mirrors `pycocotools.mask` but with a Rusty error
//! model: dimension mismatches, malformed RLE, and ambiguous polygon
//! input return a [`MaskError`] rather than COCOAPI's `0` / `-1` /
//! empty-RLE sentinels (the `corrected` dispositions on quirks H1,
//! H2, I2, I6, K1 per ADR-0002).
//!
//! See `docs/engineering/pycocotools-quirks.md` for the G/H/I/K
//! quirk rows this crate dispositions.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod codec;
pub mod error;
pub mod ops;
pub mod polygon;
pub mod raster;
pub mod rle;

pub use codec::{decode_counts, encode_counts};
pub use error::MaskError;
pub use ops::{
    boundary_band, boundary_band_into, erode_chebyshev_ball, erode_chebyshev_ball_into,
    ErodeScratch,
};
pub use rle::Rle;

/// Library version string. Useful for parity tracing and debugging
/// FFI mismatches.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }
}
