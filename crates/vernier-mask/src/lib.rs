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

use thiserror::Error;

/// Errors raised by mask operations.
///
/// Variants cover the cases where pycocotools silently returns a
/// sentinel value (`0` from `rleDecode` on overflow, `-1` from
/// `rleIou` on dimension mismatch, an empty 0×0 RLE from `rleMerge`
/// on dimension mismatch). Vernier's contract per ADR-0002 is to
/// raise rather than silently corrupt downstream computations.
#[derive(Debug, Error)]
pub enum MaskError {
    /// Operands have inconsistent `(h, w)` shapes. Quirks H2, I2.
    #[error("mask dimension mismatch: expected {expected:?}, got {got:?}")]
    DimensionMismatch {
        /// Expected `(h, w)`.
        expected: (u32, u32),
        /// Actual `(h, w)`.
        got: (u32, u32),
    },
}

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
