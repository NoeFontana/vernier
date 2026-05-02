//! Raster ↔ RLE conversion.
//!
//! Mirrors `rleEncode` (`mc:32-41`) and `rleDecode` (`mc:45-63`) from
//! `pycocotools-2.0.11/common/maskApi.c`.
//!
//! Rasters are flat `Vec<u8>` / `&[u8]` of length `h * w` in
//! column-major (Fortran) order: pixel `(x, y)` lives at index
//! `x * h + y`. This matches pycocotools' `byte *M` layout and lets
//! callers reinterpret the slice as a numpy `(h, w)` array with
//! `order='F'` without copying.
//!
//! Per quirk **G6** disposition: `strict` — any non-zero byte is
//! treated as foreground (`1`) on encode, matching pycocotools'
//! `T[j]!=p` comparison (`mc:32-41`). A uint8 mask with values
//! `{0, 2}` thus encodes identically to `{0, 1}`, bit-for-bit with
//! the reference.

use crate::error::{MalformedRleReason, MaskError};
use crate::rle::Rle;

/// Initial capacity for the encoder's `counts` vec. Eliminates the
/// first several doublings on typical masks without overshooting on
/// tiny ones. Capped to keep small masks from over-allocating.
const ENCODE_COUNTS_CAPACITY_HINT: usize = 64;

impl Rle {
    /// Encodes a column-major byte mask of shape `(h, w)` into an
    /// RLE.
    ///
    /// `mask` must have length `h * w`; mismatch returns
    /// [`MaskError::RasterLengthMismatch`]. Per quirk **G6**, every
    /// non-zero byte is foreground.
    ///
    /// Returns the empty `0x0` RLE for `h == 0 || w == 0`.
    pub fn from_raster_bytes(mask: &[u8], h: u32, w: u32) -> Result<Self, MaskError> {
        let expected = (h as u64) * (w as u64);
        if mask.len() as u64 != expected {
            return Err(MaskError::RasterLengthMismatch {
                h,
                w,
                expected,
                got: mask.len(),
            });
        }
        if expected == 0 {
            return Ok(Rle {
                h,
                w,
                counts: vec![],
            });
        }
        let mut counts: Vec<u32> =
            Vec::with_capacity((mask.len() + 1).min(ENCODE_COUNTS_CAPACITY_HINT));
        let mut phase: u8 = 0;
        let mut run: u64 = 0;
        for &byte in mask {
            let bit = u8::from(byte != 0);
            if bit != phase {
                counts.push(
                    u32::try_from(run)
                        .map_err(|_| MaskError::MalformedRle(MalformedRleReason::U32Overflow))?,
                );
                run = 0;
                phase = bit;
            }
            run += 1;
        }
        counts.push(
            u32::try_from(run)
                .map_err(|_| MaskError::MalformedRle(MalformedRleReason::U32Overflow))?,
        );
        Ok(Rle { h, w, counts })
    }

    /// Decodes the RLE into a freshly allocated column-major byte
    /// mask of length `h * w`. Foreground pixels are `1`, background
    /// `0`.
    ///
    /// Assumes a well-formed RLE (`counts` summing to `h * w`). The
    /// length of the returned vector reflects the actual sum of
    /// counts; for a well-formed RLE this equals `h * w`.
    pub fn to_raster_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity((self.h as usize).saturating_mul(self.w as usize));
        self.to_raster_bytes_into(&mut out);
        out
    }

    /// Decodes the RLE into a caller-owned byte buffer, reusing its
    /// capacity. The buffer is `clear()`-ed first, then grown to
    /// `h * w`. Same semantics as [`Self::to_raster_bytes`] otherwise.
    ///
    /// Hot-path callers (the boundary-band kernel decodes ~36 k masks
    /// per `evaluate_boundary` on val2017) can hold a single
    /// `Vec<u8>` and pass it on every call to amortize the per-mask
    /// allocation cost.
    pub fn to_raster_bytes_into(&self, buf: &mut Vec<u8>) {
        buf.clear();
        let total = (self.h as usize).saturating_mul(self.w as usize);
        buf.reserve(total);
        let mut v: u8 = 0;
        for &len in &self.counts {
            buf.resize(buf.len() + len as usize, v);
            v ^= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn rle(h: u32, w: u32, counts: Vec<u32>) -> Rle {
        Rle { h, w, counts }
    }

    #[test]
    fn empty_zero_zero_round_trips() {
        let r = Rle::from_raster_bytes(&[], 0, 0).unwrap();
        assert_eq!(r, rle(0, 0, vec![]));
        assert_eq!(r.to_raster_bytes(), Vec::<u8>::new());
    }

    #[test]
    fn empty_nonzero_shape_with_empty_slice_errors() {
        let err = Rle::from_raster_bytes(&[], 2, 3).unwrap_err();
        assert!(matches!(
            err,
            MaskError::RasterLengthMismatch {
                h: 2,
                w: 3,
                expected: 6,
                got: 0
            }
        ));
    }

    #[test]
    fn length_mismatch_errors() {
        let err = Rle::from_raster_bytes(&[0; 5], 2, 3).unwrap_err();
        assert!(matches!(
            err,
            MaskError::RasterLengthMismatch {
                h: 2,
                w: 3,
                expected: 6,
                got: 5
            }
        ));
    }

    #[test]
    fn all_background_encodes_to_single_run() {
        let r = Rle::from_raster_bytes(&[0; 4], 2, 2).unwrap();
        assert_eq!(r, rle(2, 2, vec![4]));
        assert_eq!(r.to_raster_bytes(), vec![0; 4]);
    }

    #[test]
    fn all_foreground_starts_with_zero_length_background() {
        let r = Rle::from_raster_bytes(&[1; 4], 2, 2).unwrap();
        assert_eq!(r, rle(2, 2, vec![0, 4]));
        assert_eq!(r.to_raster_bytes(), vec![1; 4]);
    }

    #[test]
    fn nonzero_bytes_binarize_per_g6() {
        // Mixed values 0/2/255/0 → binarized as 0/1/1/0 → counts [1,2,1].
        let r = Rle::from_raster_bytes(&[0, 2, 255, 0], 2, 2).unwrap();
        assert_eq!(r, rle(2, 2, vec![1, 2, 1]));
        assert_eq!(r.to_raster_bytes(), vec![0, 1, 1, 0]);
    }

    #[test]
    fn column_major_pixel_layout() {
        // 2x3 with one fg pixel at (x=1, y=1) → flat idx = x*h + y = 3.
        let mut mask = vec![0u8; 6];
        mask[3] = 1;
        let r = Rle::from_raster_bytes(&mask, 2, 3).unwrap();
        assert_eq!(r, rle(2, 3, vec![3, 1, 2]));
        assert_eq!(r.bbox(), [1, 1, 1, 1]);
    }

    #[test]
    fn run_spanning_columns_round_trips() {
        // 2x3 mask, fg from idx 1..=4 (length 4): [0,1,1,1,1,0].
        let mask = vec![0, 1, 1, 1, 1, 0];
        let r = Rle::from_raster_bytes(&mask, 2, 3).unwrap();
        assert_eq!(r, rle(2, 3, vec![1, 4, 1]));
        assert_eq!(r.to_raster_bytes(), mask);
    }

    proptest! {
        #[test]
        fn raster_round_trip(bytes in proptest::collection::vec(0u8..=1, 0..120)) {
            let len = bytes.len() as u32;
            // Pick (h, w) such that h*w = len. Simplest: h=1, w=len.
            let r = Rle::from_raster_bytes(&bytes, 1, len)?;
            prop_assert_eq!(r.to_raster_bytes(), bytes);
        }

        #[test]
        fn raster_to_rle_to_raster_with_arbitrary_bytes(bytes in proptest::collection::vec(any::<u8>(), 0..120)) {
            let len = bytes.len() as u32;
            let r = Rle::from_raster_bytes(&bytes, 1, len)?;
            let expected: Vec<u8> = bytes.iter().map(|&b| u8::from(b != 0)).collect();
            prop_assert_eq!(r.to_raster_bytes(), expected);
        }

        #[test]
        fn area_matches_byte_count(bytes in proptest::collection::vec(0u8..=1, 0..120)) {
            let len = bytes.len() as u32;
            let r = Rle::from_raster_bytes(&bytes, 1, len)?;
            let expected: u64 = bytes.iter().map(|&b| b as u64).sum();
            prop_assert_eq!(r.area(), expected);
        }
    }
}
