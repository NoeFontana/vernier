//! Run-length-encoded binary mask.
//!
//! Field-compatible with the COCOAPI in-memory layout
//! (`maskApi.h`'s `RLE { siz h, w, m; uint *cnts; }`). The `m`
//! field is implicit — `counts.len()`.

use std::sync::{Arc, LazyLock};

use crate::codec::{decode_counts, encode_counts};
use crate::error::MaskError;

/// A run-length-encoded binary mask of shape `(h, w)`.
///
/// Per quirk **G5**, the encoding always starts with a background
/// run (possibly zero-length); foreground runs sit at odd indices
/// of [`Rle::counts`].
///
/// `counts` is shared via [`Arc`] so the dataset-cached value and
/// every per-pair kernel call reuse the same buffer without an
/// O(N) memcpy on each clone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rle {
    /// Mask height in pixels.
    pub h: u32,
    /// Mask width in pixels.
    pub w: u32,
    /// Run lengths. `counts[0]` is a background run (per **G5**);
    /// runs alternate background / foreground from there.
    pub counts: Arc<[u32]>,
}

static EMPTY_COUNTS: LazyLock<Arc<[u32]>> = LazyLock::new(|| Arc::from(Vec::<u32>::new()));

impl Rle {
    /// Builds an `Rle` from an owned counts buffer, taking ownership
    /// of the `Vec` so the wrapping `Arc<[u32]>` allocation reuses
    /// the existing storage when capacity matches.
    pub fn from_counts(h: u32, w: u32, counts: Vec<u32>) -> Self {
        Self {
            h,
            w,
            counts: counts.into(),
        }
    }

    /// All-background `Rle` of shape `(h, w)` with an empty counts
    /// buffer. Reuses a process-wide empty `Arc<[u32]>` so repeated
    /// empty-mask construction (e.g., zero-area images,
    /// `from_polygons(&[])`) is allocation-free.
    pub fn empty(h: u32, w: u32) -> Self {
        Self {
            h,
            w,
            counts: Arc::clone(&EMPTY_COUNTS),
        }
    }

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
        Ok(Self::from_counts(h, w, decode_counts(bytes)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_via_struct() {
        let rle = Rle::from_counts(10, 10, vec![3, 2, 1, 4, 90]);
        let s = rle.to_string_bytes();
        let parsed = Rle::from_string_bytes(&s, 10, 10).unwrap();
        assert_eq!(parsed, rle);
    }

    #[test]
    fn empty_rle_round_trip() {
        let rle = Rle::empty(0, 0);
        assert_eq!(rle.to_string_bytes(), b"");
        assert_eq!(Rle::from_string_bytes(b"", 0, 0).unwrap(), rle);
    }
}
