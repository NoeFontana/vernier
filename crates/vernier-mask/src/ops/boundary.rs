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
use crate::ops::erode::{erode_raster_into_scratch, ErodeScratch};
use crate::ops::SegmentTable;
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
    let mut scratch = ErodeScratch::new();
    boundary_band_into(rle, dilation_ratio, &mut scratch)
}

/// `_into` variant of [`boundary_band`] reusing a caller-owned
/// [`ErodeScratch`]. Same semantics, but skips the intermediate
/// eroded-RLE encode/decode roundtrip by XORing the mask with the
/// eroded raster in place — the boundary-IoU dataset-wide pass on
/// val2017 amortizes the per-mask allocations across ~36k calls.
pub fn boundary_band_into(
    rle: &Rle,
    dilation_ratio: f64,
    scratch: &mut ErodeScratch,
) -> Result<Rle, MaskError> {
    if rle.h == 0 || rle.w == 0 {
        return Ok(rle.clone());
    }
    xor_band_raster_into_scratch(rle, dilation_ratio, scratch);
    Rle::from_raster_bytes(&scratch.raster, rle.h, rle.w)
}

/// Computes the boundary band of `rle` and pushes its foreground
/// segment offsets onto `segments` while returning the band area.
///
/// Same erosion + N5 XOR semantics as [`boundary_band_into`], but
/// skips the intermediate band-`Rle` encode and the two follow-up
/// `counts` walks (one for `area`, one for the offsets decode) that
/// the boundary-IoU kernel performs after `boundary_band_into`. The
/// XOR'd band raster — already in `scratch.raster` — is walked once
/// via [`SegmentTable::push_from_raster`].
///
/// Use this on the boundary-IoU hot path; keep
/// [`boundary_band_into`] for callers that need the band as an
/// `Rle` (e.g. round-tripping through serialization).
pub fn boundary_band_segments_into(
    rle: &Rle,
    dilation_ratio: f64,
    scratch: &mut ErodeScratch,
    segments: &mut SegmentTable,
) -> Result<u64, MaskError> {
    if rle.h == 0 || rle.w == 0 {
        segments.push_segments(&[]);
        return Ok(0);
    }
    xor_band_raster_into_scratch(rle, dilation_ratio, scratch);
    Ok(segments.push_from_raster(&scratch.raster))
}

/// Fills `scratch.raster` with the band raster: RLE → mask raster →
/// erode → mask XOR eroded. Shared by [`boundary_band_into`] and
/// [`boundary_band_segments_into`]; the only difference between the
/// two public entry points is what they do with the resulting raster.
///
/// N5: `mask AND NOT eroded == mask XOR eroded` under the
/// `eroded ⊆ mask` invariant (asserted by the proptest in `erode.rs`).
/// XOR in place so we don't need a separate band buffer.
fn xor_band_raster_into_scratch(rle: &Rle, dilation_ratio: f64, scratch: &mut ErodeScratch) {
    let radius = dilation_pixels(rle.h, rle.w, dilation_ratio);
    let h = rle.h as usize;
    let w = rle.w as usize;
    let d = radius as usize;
    rle.to_raster_bytes_into(&mut scratch.raster);
    erode_raster_into_scratch(scratch, h, w, d);
    for (m, &e) in scratch.raster.iter_mut().zip(&scratch.eroded) {
        *m ^= e;
    }
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

    #[test]
    fn boundary_band_segments_into_matches_boundary_band_into() {
        // Pin parity vs the band-as-RLE path: the fused composer's
        // `(area, segments)` must equal `boundary_band_into(...).area()`
        // and `decode_fg_offsets_into` for the same input.
        let raster: Vec<u8> = vec![
            0, 1, 1, 0, //
            1, 1, 1, 1, //
            1, 1, 1, 1, //
            0, 1, 1, 0, //
        ];
        let r = Rle::from_raster_bytes(&raster, 4, 4).unwrap();
        let mut erode = ErodeScratch::new();
        let band = boundary_band_into(&r, 0.3, &mut erode).unwrap();
        let mut expected_offsets = Vec::new();
        band.decode_fg_offsets_into(&mut expected_offsets);

        let mut erode2 = ErodeScratch::new();
        let mut segments = SegmentTable::new();
        let area =
            boundary_band_segments_into(&r, 0.3, &mut erode2, &mut segments).unwrap();

        assert_eq!(area, band.area());
        assert_eq!(segments.row(0), expected_offsets.as_slice());
    }

    #[test]
    fn boundary_band_segments_into_empty_shape() {
        // Zero-shape RLE → empty row, area 0 (matches boundary_band_into
        // returning the empty mask unchanged).
        let r = rle(0, 0, vec![]);
        let mut erode = ErodeScratch::new();
        let mut segments = SegmentTable::new();
        let area =
            boundary_band_segments_into(&r, 0.02, &mut erode, &mut segments).unwrap();
        assert_eq!(area, 0);
        assert!(segments.row(0).is_empty());
    }
}
