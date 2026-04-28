//! Boundary band of an RLE binary mask.
//!
//! [`boundary_band`] computes the thin frame of pixels lying on the
//! mask's boundary, parameterised by `dilation_ratio` (Cheng et al.
//! 2021; default 0.02). The band is the region the boundary-IoU
//! metric integrates over.
//!
//! Per the boundary-IoU quirks survey:
//! - **M2**: dilation pixels = `round_ties_even(ratio * sqrt(h² + w²))`
//!   (Python's `round` is banker's rounding; `f64::round_ties_even` is
//!   the matching primitive).
//! - **M3**: clamp `dilation ≥ 1`.
//! - **M4**: default `dilation_ratio = 0.02` (lives in
//!   `vernier_core::boundary_parity::BOUNDARY_DILATION_RATIO_DEFAULT`).
//! - **N5**: the band is `mask AND NOT eroded` rather than the
//!   reference's `mask - eroded` subtraction. Bit-equal on integer
//!   binary input (eroded ⊆ mask is guaranteed by erosion being a
//!   min-filter), but avoids the latent uint8-wraparound risk if the
//!   subset invariant ever broke.

use crate::error::MaskError;
use crate::ops::erode::erode_chebyshev_ball;
use crate::rle::Rle;

/// Computes the boundary band of an RLE binary mask: the set of
/// foreground pixels lying within `d` pixels of the mask's complement,
/// where `d = round_ties_even(dilation_ratio * sqrt(h² + w²))` clamped
/// to ≥ 1.
///
/// Returns the empty mask unchanged when `h == 0` or `w == 0`. Per
/// quirk **P2** the band of an empty (no-foreground) mask is itself
/// empty.
pub fn boundary_band(rle: &Rle, dilation_ratio: f64) -> Result<Rle, MaskError> {
    if rle.h == 0 || rle.w == 0 {
        return Ok(rle.clone());
    }
    let radius = dilation_pixels(rle.h, rle.w, dilation_ratio);
    let eroded = erode_chebyshev_ball(rle, radius)?;
    let mut band = rle.to_raster_bytes();
    let eroded_raster = eroded.to_raster_bytes();
    // N5: mask AND NOT eroded == mask XOR eroded under the eroded ⊆ mask
    // invariant (asserted by the proptest in erode.rs).
    for (m, &e) in band.iter_mut().zip(&eroded_raster) {
        *m ^= e;
    }
    Rle::from_raster_bytes(&band, rle.h, rle.w)
}

/// Quirks **M2** + **M3**: pixel dilation distance for a `(h, w)`
/// mask at the given ratio. Half-to-even rounding, clamped to ≥ 1.
///
/// Crate-private: the public boundary surface takes `dilation_ratio`
/// directly and computes `d` internally. Exposed at crate scope so the
/// boundary parity tests can pin the rounding behaviour without going
/// through `boundary_band`.
pub(crate) fn dilation_pixels(h: u32, w: u32, dilation_ratio: f64) -> u32 {
    let diag = ((f64::from(h)).powi(2) + (f64::from(w)).powi(2)).sqrt();
    let raw = (dilation_ratio * diag).round_ties_even();
    if raw.is_finite() && raw >= 1.0 {
        // Post-rounding the value is an integer in (-2^53, 2^53);
        // cast to u32 is exact for non-negative values within range.
        raw as u32
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rle(h: u32, w: u32, counts: Vec<u32>) -> Rle {
        Rle { h, w, counts }
    }

    #[test]
    fn empty_shape_round_trips_unchanged() {
        let r = rle(0, 0, vec![]);
        assert_eq!(boundary_band(&r, 0.02).unwrap(), r);
    }

    #[test]
    fn empty_foreground_yields_empty_band() {
        // P2: band of an empty mask is empty.
        let r = rle(8, 8, vec![64]);
        let band = boundary_band(&r, 0.02).unwrap();
        assert_eq!(band.area(), 0);
        assert_eq!((band.h, band.w), (8, 8));
    }

    #[test]
    fn full_image_band_is_outer_frame() {
        // P3: a fully-foreground image has its boundary band equal to
        // the outer frame of width d. For 5×5 at d=1 the frame is the
        // 16 perimeter pixels. Ratio 0.15 → round_ties_even(0.15 *
        // sqrt(50)) = round(1.0606) = 1.
        let r = rle(5, 5, vec![0, 25]);
        let band = boundary_band(&r, 0.15).unwrap();
        assert_eq!(band.area(), 16);
    }

    #[test]
    fn single_pixel_mask_band_equals_mask() {
        // P4: 1px mask, d≥1 erosion empties everything, so band == mask.
        let r = Rle::from_raster_bytes(&[0, 0, 1, 0], 2, 2).unwrap();
        let band = boundary_band(&r, 0.5).unwrap();
        assert_eq!(band.area(), 1);
        assert_eq!(band.to_raster_bytes(), vec![0, 0, 1, 0]);
    }

    #[test]
    fn dilation_pixels_clamps_below_one() {
        // M3: even at ratio = 0 (or any value where the product
        // rounds to 0), we clamp up to 1.
        assert_eq!(dilation_pixels(100, 100, 0.0), 1);
        assert_eq!(dilation_pixels(10, 10, 0.001), 1);
    }

    #[test]
    fn dilation_pixels_typical_coco_image() {
        // M4 default 0.02 on a typical 1280×720 image:
        // sqrt(1280² + 720²) = sqrt(2_156_800) ≈ 1468.6
        // 0.02 * 1468.6 ≈ 29.37 → round → 29.
        assert_eq!(dilation_pixels(720, 1280, 0.02), 29);
    }

    #[test]
    fn dilation_pixels_uses_banker_rounding_at_half() {
        // M2: Python's round() is half-to-even; verify the primitive at
        // exactly-representable .5 cases (constructing a (h, w, ratio)
        // triple that hits 0.5 exactly through f64 multiplication is
        // not generally possible).
        assert_eq!(0.5_f64.round_ties_even(), 0.0);
        assert_eq!(1.5_f64.round_ties_even(), 2.0);
        assert_eq!(2.5_f64.round_ties_even(), 2.0);
    }

    #[test]
    fn dilation_pixels_rejects_non_finite_input() {
        // Defensive: NaN ratio → clamp to 1 (don't overflow / panic).
        assert_eq!(dilation_pixels(100, 100, f64::NAN), 1);
        assert_eq!(dilation_pixels(100, 100, f64::INFINITY), 1);
        assert_eq!(dilation_pixels(100, 100, -0.5), 1);
    }

    #[test]
    fn band_is_subset_of_mask() {
        // Conceptual N5 check: band ⊆ mask, because band = mask & !eroded.
        let raster: Vec<u8> = vec![
            0, 1, 1, 0, //
            1, 1, 1, 1, //
            1, 1, 1, 1, //
            0, 1, 1, 0, //
        ];
        let r = Rle::from_raster_bytes(&raster, 4, 4).unwrap();
        let band = boundary_band(&r, 0.3).unwrap();
        let m = r.to_raster_bytes();
        let b = band.to_raster_bytes();
        for (mi, bi) in m.iter().zip(&b) {
            assert!(*bi <= *mi);
        }
    }
}
