//! Erosion of an RLE binary mask by a Chebyshev (L-infinity) ball.
//!
//! [`erode_chebyshev_ball`] is the erosion operator used by the
//! boundary-IoU metric (Cheng et al. 2021). See the boundary-IoU
//! quirks survey for the full algorithmic contract: rows **M1**, **M5**
//! (kernel shape and size), **N1**–**N2** (1-pixel zero pad applied
//! once before erosion, stripped after), **N3** (pad value is hard
//! zero), **N4** (single-pass output bit-equal to iterative on integer
//! binary input).
//!
//! Per ADR-0010 §"Algorithm specification (A2)" the implementation is
//! separable — a 1D row pass (along x) followed by a 1D column pass
//! (along y), each sliding a window of size `2d+1`. On integer binary
//! input this is bit-equal to iterative 3×3 erosion applied `d`
//! times: the (3×3 all-ones)^d structuring element under iterated
//! dilation equals the (2d+1)-square all-ones kernel, and on a
//! min-filter erosion is the dual. Equivalence is asserted by a
//! property test against an iterative reference kept in this file's
//! `tests` module.
//!
//! Reads outside the once-padded `(h+2) × (w+2)` image return `1`,
//! matching `cv2.erode` default `BORDER_CONSTANT` for binary input.
//! The only zeros that contribute to the sliding min are the explicit
//! 1-pixel zero ring; this is what makes the single-pass form
//! bit-equal to the iterative reference at the `(d-1)` ring near the
//! image edge for `d ≥ 2`.

use crate::error::MaskError;
use crate::rle::Rle;

/// Erodes an RLE binary mask by a `(2 * radius_pixels + 1)`-square
/// L-infinity (Chebyshev) ball structuring element.
///
/// `radius_pixels = 0` is the identity. Empty masks (`h == 0` or
/// `w == 0`) round-trip unchanged. Per quirks **N1**–**N2** the mask
/// is padded with a 1-pixel zero border before erosion, then the pad
/// is stripped after — so border-touching foreground pixels are
/// eroded against the zero pad, which is the metric-defining
/// behaviour rather than an implementation quirk.
pub fn erode_chebyshev_ball(rle: &Rle, radius_pixels: u32) -> Result<Rle, MaskError> {
    if rle.h == 0 || rle.w == 0 || radius_pixels == 0 {
        return Ok(rle.clone());
    }
    let h = rle.h as usize;
    let w = rle.w as usize;
    let d = radius_pixels as usize;
    let raster = rle.to_raster_bytes();
    let eroded = erode_raster_chebyshev(&raster, h, w, d);
    Rle::from_raster_bytes(&eroded, rle.h, rle.w)
}

/// Pad → row pass → column pass → strip pad on a column-major byte
/// raster. Crate-private so the proptest in this file can compare
/// against the iterative reference without going through RLE
/// encode/decode.
pub(crate) fn erode_raster_chebyshev(raster: &[u8], h: usize, w: usize, d: usize) -> Vec<u8> {
    let ph = h + 2;
    let pw = w + 2;

    // 1. Pad to (h+2, w+2) column-major with a 1-pixel zero ring (N1).
    //    Original (x, y) maps to padded (x+1, y+1).
    let mut padded = vec![0u8; ph * pw];
    for x in 0..w {
        for y in 0..h {
            padded[(x + 1) * ph + (y + 1)] = raster[x * h + y];
        }
    }

    // 2. Row pass: column-major means x-stride = ph (non-contiguous),
    //    so for each y we gather a contiguous row of length pw, run
    //    the 1D min, and scatter the result. The row scratch buffers
    //    are reused across rows.
    let mut scratch = vec![0u8; ph * pw];
    let mut row_in = vec![0u8; pw];
    let mut row_out = vec![0u8; pw];
    for y in 0..ph {
        for x in 0..pw {
            row_in[x] = padded[x * ph + y];
        }
        min_filter_binary(&row_in, d, &mut row_out);
        for x in 0..pw {
            scratch[x * ph + y] = row_out[x];
        }
    }

    // 3. Column pass: y-stride = 1 means each column is a contiguous
    //    `ph`-element slice. `padded` is unused after the row pass, so
    //    reuse it as the column-pass output buffer.
    let mut output = padded;
    for x in 0..pw {
        let col = x * ph;
        let (col_in, col_out) = (&scratch[col..col + ph], &mut output[col..col + ph]);
        min_filter_binary(col_in, d, col_out);
    }

    // 4. Strip the pad: copy interior (1..=w) × (1..=h) back to (h, w)
    //    column-major (N2).
    let mut stripped = vec![0u8; h * w];
    for x in 0..w {
        for y in 0..h {
            stripped[x * h + y] = output[(x + 1) * ph + (y + 1)];
        }
    }
    stripped
}

/// 1D sliding-window minimum on a binary `{0, 1}` sequence with kernel
/// size `2 * d + 1`. Reads outside the sequence are treated as `1`, so
/// they don't pull the min below — this matches the OOB behaviour of
/// `cv2.erode` with a `BORDER_CONSTANT` value of `255` on binary input
/// (the OpenCV default for erosion).
///
/// O(n) regardless of `d`: maintains a running count of zeros in the
/// in-bounds portion of the sliding window. Output is `1` iff that
/// count is zero. This is one named instance of the van Herk /
/// Gil-Werman family — for binary input the per-element work
/// collapses to two conditional increments and a comparison.
fn min_filter_binary(input: &[u8], d: usize, output: &mut [u8]) {
    let n = input.len();
    debug_assert_eq!(n, output.len());
    if n == 0 {
        return;
    }
    let mut zeros: u32 = 0;
    let initial_hi = d.min(n - 1);
    for v in &input[0..=initial_hi] {
        if *v == 0 {
            zeros += 1;
        }
    }
    output[0] = u8::from(zeros == 0);
    for i in 1..n {
        // Slide by 1: position (i - d - 1) leaves the window iff it
        // was in-bounds before — i.e., iff i > d.
        if i > d && input[i - d - 1] == 0 {
            zeros -= 1;
        }
        // Position (i + d) enters the window iff in-bounds.
        if i + d < n && input[i + d] == 0 {
            zeros += 1;
        }
        output[i] = u8::from(zeros == 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Iterative reference: pad once with a 1-pixel zero ring, apply a
    /// 3×3 min-filter `d` times with OOB-fills-with-1, strip the pad.
    /// This is what `cv2.erode(padded, np.ones((3,3)), iterations=d)`
    /// computes on a binary `np.uint8` mask under the OpenCV pin range
    /// covered by quirk **M6**.
    fn erode_iterative_reference(raster: &[u8], h: usize, w: usize, d: usize) -> Vec<u8> {
        if h == 0 || w == 0 || d == 0 {
            return raster.to_vec();
        }
        let ph = h + 2;
        let pw = w + 2;
        let mut buf = vec![0u8; ph * pw];
        for x in 0..w {
            for y in 0..h {
                buf[(x + 1) * ph + (y + 1)] = raster[x * h + y];
            }
        }
        let mut scratch = vec![0u8; ph * pw];
        for _ in 0..d {
            for x in 0..pw {
                for y in 0..ph {
                    let mut m: u8 = 1;
                    for dx in -1i32..=1 {
                        for dy in -1i32..=1 {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            let v = if nx >= 0 && nx < pw as i32 && ny >= 0 && ny < ph as i32 {
                                buf[(nx as usize) * ph + (ny as usize)]
                            } else {
                                1
                            };
                            if v < m {
                                m = v;
                            }
                        }
                    }
                    scratch[x * ph + y] = m;
                }
            }
            std::mem::swap(&mut buf, &mut scratch);
        }
        let mut out = vec![0u8; h * w];
        for x in 0..w {
            for y in 0..h {
                out[x * h + y] = buf[(x + 1) * ph + (y + 1)];
            }
        }
        out
    }

    fn rle(h: u32, w: u32, counts: Vec<u32>) -> Rle {
        Rle { h, w, counts }
    }

    #[test]
    fn empty_mask_round_trips_unchanged() {
        let r = rle(0, 0, vec![]);
        assert_eq!(erode_chebyshev_ball(&r, 5).unwrap(), r);
    }

    #[test]
    fn radius_zero_is_identity() {
        let r = rle(2, 2, vec![1, 2, 1]);
        assert_eq!(erode_chebyshev_ball(&r, 0).unwrap(), r);
    }

    #[test]
    fn full_3x3_erodes_to_single_center_pixel_at_d1() {
        // 3×3 fully-foreground mask. With the zero pad (N1) the kernel
        // sees zeros around the perimeter, so erosion by the 3×3
        // Chebyshev ball (d=1) leaves only the center pixel: M5
        // (square not cross) is what makes this a single pixel rather
        // than a + shape.
        let r = rle(3, 3, vec![0, 9]);
        let eroded = erode_chebyshev_ball(&r, 1).unwrap();
        // Center pixel at (x=1, y=1) → flat idx 1*3 + 1 = 4. RLE
        // canonicalises as [bg=4, fg=1, bg=4].
        assert_eq!(eroded.counts, vec![4, 1, 4]);
    }

    #[test]
    fn full_5x5_eroded_by_d1_yields_3x3_interior_block() {
        // 5×5 fully-fg, padded → eroded by d=1 → interior 3×3 block
        // (1..=3, 1..=3) of foreground = 9 fg pixels. Frames the
        // metric: the boundary band of a fully-fg image is a frame of
        // width d (P3).
        let r = rle(5, 5, vec![0, 25]);
        let eroded = erode_chebyshev_ball(&r, 1).unwrap();
        let fg_count: u32 = eroded.to_raster_bytes().iter().map(|&b| u32::from(b)).sum();
        assert_eq!(fg_count, 9);
    }

    #[test]
    fn kernel_larger_than_image_erodes_everything() {
        // Quirk P1: when (2d+1) > min(h, w), erosion of fully-fg
        // yields fully-zero — the structuring element cannot fit
        // inside the foreground anywhere.
        let r = rle(3, 3, vec![0, 9]);
        let eroded = erode_chebyshev_ball(&r, 10).unwrap();
        assert_eq!(eroded.area(), 0);
    }

    #[test]
    fn empty_foreground_erosion_is_empty() {
        // P2: erosion of an empty mask is empty.
        let r = rle(4, 4, vec![16]);
        let eroded = erode_chebyshev_ball(&r, 2).unwrap();
        assert_eq!(eroded.area(), 0);
        assert_eq!(eroded.h, 4);
        assert_eq!(eroded.w, 4);
    }

    #[test]
    fn single_pixel_mask_erodes_to_zero_at_d1() {
        // P4: a single-pixel mask + d=1 erosion → empty. The pad pulls
        // every neighbour to zero, so the min over the 3×3 window is 0.
        let mut raster = vec![0u8; 9];
        raster[4] = 1; // (x=1, y=1) flat idx = 4
        let r = Rle::from_raster_bytes(&raster, 3, 3).unwrap();
        let eroded = erode_chebyshev_ball(&r, 1).unwrap();
        assert_eq!(eroded.area(), 0);
    }

    #[test]
    fn van_herk_matches_iterative_on_hand_traced_fixture() {
        // 4×5 mask with a non-trivial pattern, exercising columns
        // partially eroded plus columns fully zeroed plus a column
        // straddling the kernel boundary.
        let raster: Vec<u8> = vec![
            // x=0: column-major, h=4 rows.
            0, 1, 0, 0, // (0,0..4)
            1, 1, 1, 1, // (1,0..4)
            1, 1, 1, 1, // (2,0..4)
            0, 0, 0, 0, // (3,0..4)
            1, 1, 0, 0, // (4,0..4)
        ];
        for d in 1..=3 {
            let single_pass = erode_raster_chebyshev(&raster, 4, 5, d);
            let iterative = erode_iterative_reference(&raster, 4, 5, d);
            assert_eq!(single_pass, iterative, "d={d}");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]
        #[test]
        fn single_pass_matches_iterative_on_random_binary_masks(
            (h, w, d, raster) in (1usize..=8, 1usize..=8, 1usize..=4)
                .prop_flat_map(|(h, w, d)| {
                    let len = h * w;
                    (
                        Just(h),
                        Just(w),
                        Just(d),
                        proptest::collection::vec(0u8..=1, len..=len),
                    )
                }),
        ) {
            let single_pass = erode_raster_chebyshev(&raster, h, w, d);
            let iterative = erode_iterative_reference(&raster, h, w, d);
            prop_assert_eq!(single_pass, iterative);
        }

        #[test]
        fn erosion_is_subset_of_input(
            bytes in proptest::collection::vec(0u8..=1, 0..120),
            d in 1u32..=4,
        ) {
            // For any mask M and any radius r, erode(M, r) ⊆ M
            // because erosion is a min-filter. This is the property
            // the boundary-band XOR step (N5) relies on: the band is
            // M XOR erode(M) only because erode(M) ⊆ M.
            let len = bytes.len() as u32;
            let r = Rle::from_raster_bytes(&bytes, 1, len)?;
            let eroded = erode_chebyshev_ball(&r, d)?;
            let m = r.to_raster_bytes();
            let e = eroded.to_raster_bytes();
            for (mi, ei) in m.iter().zip(&e) {
                prop_assert!(*ei <= *mi);
            }
        }
    }
}
