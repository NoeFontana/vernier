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

    /// RLE counts string is malformed. `corrected` per ADR-0002:
    /// pycocotools' decoder silently produces garbage on out-of-range
    /// bytes (`mc:259`); vernier rejects them.
    #[error("malformed RLE counts string: {0}")]
    MalformedRle(MalformedRleReason),

    /// Raster byte slice length doesn't match the declared
    /// `(h, w)` shape. `corrected` per ADR-0002: the C `rleEncode`
    /// (`mc:32-41`) trusts the declared `h*w` and reads past the
    /// end of `M` on mismatch.
    #[error(
        "raster length {got} bytes does not match shape ({h}, {w}) (expected h*w = {expected})"
    )]
    RasterLengthMismatch {
        /// Declared mask height.
        h: u32,
        /// Declared mask width.
        w: u32,
        /// Expected length: `h * w`.
        expected: u64,
        /// Actual byte-slice length received.
        got: usize,
    },
}

/// Concrete reason a counts string failed to decode. Programmatically
/// inspectable so callers can distinguish a truncated payload from a
/// out-of-range byte without string matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MalformedRleReason {
    /// Continuation bit set on the final byte; expected another char.
    #[error("string ended mid-run (continuation bit set on final byte)")]
    TruncatedString,
    /// Byte outside the wire alphabet `[0x30, 0x6F]`.
    #[error("byte outside legal [0x30, 0x6F] range")]
    ByteOutOfRange,
    /// Run encoding exceeded the algebraic upper bound on chars per
    /// run (a `u32` plus sign needs at most 7 5-bit chunks).
    #[error("run encoding exceeds maximum char count")]
    ChunkLimitExceeded,
    /// Decoded run length does not fit in `u32` after the differential
    /// undo step.
    #[error("decoded run length outside u32 range")]
    U32Overflow,
}
