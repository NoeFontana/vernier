//! Error type for mask operations.

use thiserror::Error;

/// Errors raised by mask operations. See crate-level docs for the
/// sentinel-vs-error rationale.
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

    /// RLE counts string is malformed: a byte outside the legal
    /// `[0x30, 0x6F]` range, a string ending mid-run (continuation
    /// bit set on the final char), or a decoded run length that does
    /// not fit in `u32`.
    ///
    /// `corrected` per ADR-0002. Pycocotools' decoder silently
    /// produces garbage on out-of-range bytes (`mc:259`); vernier
    /// rejects them.
    #[error("malformed RLE counts string: {reason}")]
    MalformedRle {
        /// Human-readable explanation.
        reason: &'static str,
    },
}
