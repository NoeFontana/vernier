//! Run-length-encoded binary mask.
//!
//! Field-compatible with the COCOAPI in-memory layout
//! (`maskApi.h`'s `RLE { siz h, w, m; uint *cnts; }`). The `m`
//! field is implicit — `counts.len()`.

use crate::codec::{decode_counts, encode_counts};
use crate::error::MaskError;

/// A run-length-encoded binary mask of shape `(h, w)`.
///
/// Per quirk **G5**, the encoding always starts with a background
/// run (possibly zero-length); foreground runs sit at odd indices
/// of [`Rle::counts`].
///
/// Layout matches `pycocotools` so that consumers porting from the
/// C / Cython side can reuse mental models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rle {
    /// Mask height in pixels.
    pub h: u32,
    /// Mask width in pixels.
    pub w: u32,
    /// Run lengths. `counts[0]` is a background run (per **G5**);
    /// runs alternate background / foreground from there.
    pub counts: Vec<u32>,
}

impl Rle {
    /// Encodes `self` to the COCO 6-bit char string. Output bytes
    /// are in `b'0'..=b'o'` and round-trip through
    /// [`Rle::from_string_bytes`].
    pub fn to_string_bytes(&self) -> Vec<u8> {
        encode_counts(&self.counts)
    }

    /// Parses a COCO 6-bit char string into an `Rle` of shape
    /// `(h, w)`.
    ///
    /// Errors on malformed input per [`MaskError::MalformedRle`].
    /// The `(h, w)` shape is taken on trust — the codec does not
    /// validate that `counts` sums to `h * w`, matching pycocotools'
    /// behavior.
    pub fn from_string_bytes(bytes: &[u8], h: u32, w: u32) -> Result<Self, MaskError> {
        let counts = decode_counts(bytes)?;
        Ok(Self { h, w, counts })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_via_struct() {
        let rle = Rle {
            h: 10,
            w: 10,
            counts: vec![3, 2, 1, 4, 90],
        };
        let s = rle.to_string_bytes();
        let parsed = Rle::from_string_bytes(&s, 10, 10).unwrap();
        assert_eq!(parsed, rle);
    }

    #[test]
    fn empty_rle_round_trip() {
        let rle = Rle {
            h: 0,
            w: 0,
            counts: vec![],
        };
        assert_eq!(rle.to_string_bytes(), b"");
        assert_eq!(Rle::from_string_bytes(b"", 0, 0).unwrap(), rle);
    }
}
