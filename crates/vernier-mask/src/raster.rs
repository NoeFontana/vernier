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

use std::ops::Range;

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
            return Ok(Rle::empty(h, w));
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
        Ok(Rle::from_counts(h, w, counts))
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
        for &len in self.counts.iter() {
            buf.resize(buf.len() + len as usize, v);
            v ^= 1;
        }
    }

    /// Decodes only the bbox region of the RLE into a contiguous
    /// `bw * bh` column-major byte buffer. `buf` is `clear()`-ed and
    /// grown to `bw * bh` (zero-filled), then foreground pixels inside
    /// the bbox are overwritten with `1`.
    ///
    /// `bbox` is `[bx, by, bw, bh]` in pixel-integer form (matching
    /// [`crate::Rle::bbox`]), and must contain every foreground pixel
    /// of the RLE. `[0; 4]` is the empty-foreground case — a `0`-sized
    /// buffer is returned (matching the `bw == 0 || bh == 0` early
    /// exit).
    ///
    /// Used by the boundary-IoU hot path
    /// ([`crate::ops::boundary_band_segments_into`]) to skip the
    /// `(h - bh) * w + h * (w - bw)` outside-bbox bytes that the
    /// full-image [`Self::to_raster_bytes_into`] decode wastes
    /// background-fills on. On val2017 this typically saves a 30×
    /// reduction in per-mask write traffic since instance bboxes are
    /// small relative to the 480×640 image.
    ///
    /// Walks `counts` once (`O(num_runs)`), bbox-clips each fg run to
    /// the columns and rows it touches, and emits the foreground bytes
    /// into the cropped buffer.
    pub fn decode_bbox_into(&self, buf: &mut Vec<u8>, bbox: [u32; 4]) {
        let h = self.h as usize;
        let bx = bbox[0] as usize;
        let by = bbox[1] as usize;
        let bw = bbox[2] as usize;
        let bh = bbox[3] as usize;
        buf.clear();
        buf.resize(bw * bh, 0);
        if bw == 0 || bh == 0 || h == 0 {
            return;
        }
        // Walk runs in counts order, alternating bg/fg starting at bg
        // (G5). For each fg run, clip its flat-offset range to the
        // bbox columns and rows, and fill the surviving slice.
        let mut is_fg = false;
        let mut cum: usize = 0;
        for &len in self.counts.iter() {
            let run_len = len as usize;
            if !is_fg || run_len == 0 {
                cum += run_len;
                is_fg = !is_fg;
                continue;
            }
            // Foreground run [cum, cum + run_len). May span columns.
            let mut idx = cum;
            let run_end = cum + run_len;
            while idx < run_end {
                let x = idx / h;
                let col_end_flat = (x + 1) * h;
                let chunk_end = run_end.min(col_end_flat);
                if x < bx || x >= bx + bw {
                    idx = chunk_end;
                    continue;
                }
                // Row range within this column, intersected with
                // bbox rows [by, by + bh).
                let y_lo = idx - x * h;
                let y_hi = chunk_end - x * h;
                let yb_lo = y_lo.max(by);
                let yb_hi = y_hi.min(by + bh);
                if yb_lo < yb_hi {
                    let bbox_col = x - bx;
                    let dst_start = bbox_col * bh + (yb_lo - by);
                    let dst_len = yb_hi - yb_lo;
                    buf[dst_start..dst_start + dst_len].fill(1);
                }
                idx = chunk_end;
            }
            cum = run_end;
            is_fg = !is_fg;
        }
    }

    /// Calls `f(x, y_start..y_end)` for every contiguous vertical foreground
    /// span in column `x`.
    ///
    /// `x` is in `0..w`, and `ys` is a half-open range `y_start..y_end`
    /// satisfying `0 <= ys.start < ys.end <= h`.
    ///
    /// Traverses `counts` in a single pass with zero heap allocations.
    /// Used by renderers, RGBA blitters, WebGL texture loaders, and
    /// contour generators to visit foreground pixels directly without
    /// materializing an intermediate binary mask.
    pub fn for_each_fg_span<F>(&self, mut f: F)
    where
        F: FnMut(u32, Range<u32>),
    {
        let h = self.h as usize;
        let w = self.w as usize;
        if h == 0 || w == 0 {
            return;
        }
        let mut is_fg = false;
        let mut curr_x = 0usize;
        let mut curr_y = 0usize;

        for &len in self.counts.iter() {
            let mut run_len = len as usize;
            if !is_fg || run_len == 0 {
                if run_len > 0 {
                    let total_y = curr_y + run_len;
                    curr_x += total_y / h;
                    curr_y = total_y % h;
                    if curr_x >= w {
                        break;
                    }
                }
                is_fg = !is_fg;
                continue;
            }

            while run_len > 0 && curr_x < w {
                let rem_in_col = h - curr_y;
                if run_len < rem_in_col {
                    f(curr_x as u32, (curr_y as u32)..((curr_y + run_len) as u32));
                    curr_y += run_len;
                    break;
                } else {
                    f(curr_x as u32, (curr_y as u32)..(h as u32));
                    run_len -= rem_in_col;
                    curr_x += 1;
                    curr_y = 0;
                }
            }

            if curr_x >= w {
                break;
            }
            is_fg = !is_fg;
        }
    }

    /// Calls `f(x, y)` for every foreground pixel in the mask.
    ///
    /// Pixels are visited column-by-column, top-to-bottom within each
    /// column, matching COCO RLE's native column-major layout.
    pub fn for_each_fg_pixel<F>(&self, mut f: F)
    where
        F: FnMut(u32, u32),
    {
        self.for_each_fg_span(|x, ys| {
            for y in ys {
                f(x, y);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn empty_zero_zero_round_trips() {
        let r = Rle::from_raster_bytes(&[], 0, 0).unwrap();
        assert_eq!(r, Rle::from_counts(0, 0, vec![]));
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
        assert_eq!(r, Rle::from_counts(2, 2, vec![4]));
        assert_eq!(r.to_raster_bytes(), vec![0; 4]);
    }

    #[test]
    fn all_foreground_starts_with_zero_length_background() {
        let r = Rle::from_raster_bytes(&[1; 4], 2, 2).unwrap();
        assert_eq!(r, Rle::from_counts(2, 2, vec![0, 4]));
        assert_eq!(r.to_raster_bytes(), vec![1; 4]);
    }

    #[test]
    fn nonzero_bytes_binarize_per_g6() {
        // Mixed values 0/2/255/0 → binarized as 0/1/1/0 → counts [1,2,1].
        let r = Rle::from_raster_bytes(&[0, 2, 255, 0], 2, 2).unwrap();
        assert_eq!(r, Rle::from_counts(2, 2, vec![1, 2, 1]));
        assert_eq!(r.to_raster_bytes(), vec![0, 1, 1, 0]);
    }

    #[test]
    fn column_major_pixel_layout() {
        // 2x3 with one fg pixel at (x=1, y=1) → flat idx = x*h + y = 3.
        let mut mask = vec![0u8; 6];
        mask[3] = 1;
        let r = Rle::from_raster_bytes(&mask, 2, 3).unwrap();
        assert_eq!(r, Rle::from_counts(2, 3, vec![3, 1, 2]));
        assert_eq!(r.bbox(), [1, 1, 1, 1]);
    }

    #[test]
    fn run_spanning_columns_round_trips() {
        // 2x3 mask, fg from idx 1..=4 (length 4): [0,1,1,1,1,0].
        let mask = vec![0, 1, 1, 1, 1, 0];
        let r = Rle::from_raster_bytes(&mask, 2, 3).unwrap();
        assert_eq!(r, Rle::from_counts(2, 3, vec![1, 4, 1]));
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

        // Bbox-cropped decode must match the full-image decode
        // restricted to the same bbox region. Catches: off-by-one in
        // the column iteration, bbox-row clipping bugs, and bg-run
        // skipping bugs.
        #[test]
        fn decode_bbox_into_matches_full_decode_restricted_to_bbox(
            (h, w, raster) in (1usize..=12, 1usize..=12).prop_flat_map(|(h, w)| {
                let len = h * w;
                (Just(h), Just(w), proptest::collection::vec(0u8..=1, len..=len))
            }),
        ) {
            let r = Rle::from_raster_bytes(&raster, h as u32, w as u32)?;
            let bbox = r.bbox();
            let bx = bbox[0] as usize;
            let by = bbox[1] as usize;
            let bw = bbox[2] as usize;
            let bh = bbox[3] as usize;

            let mut bbox_buf = Vec::new();
            r.decode_bbox_into(&mut bbox_buf, bbox);
            prop_assert_eq!(bbox_buf.len(), bw * bh);

            // Compare to the full decode reshaped to the same bbox.
            let full = r.to_raster_bytes();
            for x in 0..bw {
                for y in 0..bh {
                    let full_idx = (bx + x) * h + (by + y);
                    let bbox_idx = x * bh + y;
                    prop_assert_eq!(
                        bbox_buf[bbox_idx],
                        full[full_idx],
                        "mismatch at bbox ({}, {}) → full ({}, {})",
                        x, y, bx + x, by + y
                    );
                }
            }
        }

        #[test]
        fn for_each_fg_span_matches_raster(
            (h, w, raster) in (1usize..=10, 1usize..=10).prop_flat_map(|(h, w)| {
                let len = h * w;
                (Just(h), Just(w), proptest::collection::vec(0u8..=1, len..=len))
            }),
        ) {
            let r = Rle::from_raster_bytes(&raster, h as u32, w as u32)?;
            let mut visited_count = 0u64;
            let mut reconstructed = vec![0u8; h * w];

            r.for_each_fg_span(|x, ys| {
                assert!(ys.start < ys.end);
                assert!(ys.end <= h as u32);
                assert!(x < w as u32);
                visited_count += (ys.end - ys.start) as u64;
                for y in ys {
                    let idx = (x as usize) * h + (y as usize);
                    reconstructed[idx] = 1;
                }
            });

            prop_assert_eq!(visited_count, r.area());
            prop_assert_eq!(&reconstructed, &raster);

            // Also verify for_each_fg_pixel visits exactly the same set.
            let mut pixel_reconstructed = vec![0u8; h * w];
            r.for_each_fg_pixel(|x, y| {
                let idx = (x as usize) * h + (y as usize);
                pixel_reconstructed[idx] = 1;
            });
            prop_assert_eq!(&pixel_reconstructed, &raster);
        }
    }

    #[test]
    fn for_each_fg_span_empty_mask() {
        let r0 = Rle::empty(0, 0);
        let mut count = 0;
        r0.for_each_fg_span(|_, _| count += 1);
        assert_eq!(count, 0);

        let r_bg = Rle::from_counts(4, 4, vec![16]);
        let mut count_bg = 0;
        r_bg.for_each_fg_span(|_, _| count_bg += 1);
        assert_eq!(count_bg, 0);
    }

    #[test]
    fn for_each_fg_span_single_pixel() {
        // 2x3 with pixel at (x=1, y=1) -> flat idx 3.
        let mut mask = vec![0u8; 6];
        mask[3] = 1;
        let r = Rle::from_raster_bytes(&mask, 2, 3).unwrap();
        let mut spans = Vec::new();
        r.for_each_fg_span(|x, ys| spans.push((x, ys)));
        assert_eq!(spans, vec![(1, 1..2)]);
    }

    #[test]
    fn for_each_fg_span_spanning_columns() {
        // 2x3 mask with fg from flat idx 1 to 4: [0, 1, 1, 1, 1, 0]
        let mask = vec![0, 1, 1, 1, 1, 0];
        let r = Rle::from_raster_bytes(&mask, 2, 3).unwrap();
        let mut spans = Vec::new();
        r.for_each_fg_span(|x, ys| spans.push((x, ys)));
        assert_eq!(spans, vec![(0, 1..2), (1, 0..2), (2, 0..1)]);
    }
}
