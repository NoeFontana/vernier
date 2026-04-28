//! Boundary IoU (`iouType="boundary"`).
//!
//! Implements ADR-0010 §"Algorithm specification (A2)" and the
//! IoU-sweep skeleton of ADR-0010 §"IoU sweep (D2)": for each `(g, d)`
//! pair, compute mask IoU and boundary IoU and return their `min`.
//! The boundary band of each annotation is precomputed once via
//! [`vernier_mask::ops::boundary_band`] (Cheng et al. 2021); the
//! per-pair sweep then folds two `intersect_area` calls plus a `min`.
//!
//! Bespoke kernel, not delegating to [`super::SegmIou::compute`]: by
//! computing both IoUs inline we run the bbox prefilter once, the area
//! math once, and the `min` once per cell — saving the second prefilter
//! pass plus the second per-pair RLE sweep that delegation would imply
//! (ADR-0010 §"IoU sweep (D2)").
//!
//! Per ADR-0008, every divide is `f64` so each cell matches the
//! reference oracle's double-precision result.
//!
//! ## Quirk dispositions
//!
//! See `docs/engineering/boundary-iou-quirks.md` for the canonical
//! survey. Dispositions implemented here:
//!
//! - **E1** (`strict`): crowd asymmetry. When GT is crowd, the mask
//!   denominator is `dt_mask_area`, not the union. Applied identically
//!   on the bbox prefilter and on the final RLE-pair denominator —
//!   inherited from the segm kernel.
//! - **I1** (`strict`): bbox-IoU prefilter on the tight RLE bboxes.
//!   Pairs whose bboxes don't overlap are zero by construction; they
//!   skip the boundary-band intersection sweep below. The prefilter is
//!   sound for the `min` fold because `min(a, b) <= a` and the bbox
//!   prefilter already upper-bounds the mask term.
//! - **F5** (`aligned`): empty `gts` or `dts` returns the zero-shape
//!   matrix unchanged.
//! - **H2** (`corrected`): all RLEs in one call must share `(h, w)`.
//!   Mismatch raises [`EvalError::DimensionMismatch`] instead of the
//!   `-1` sentinel pycocotools' `rleIou` writes per cell.
//! - **O1 / O2** (`strict`): for crowd GT the boundary IoU is
//!   suppressed and the cell carries the mask IoU alone. The reference
//!   oracle skips the boundary fold on crowd rows; vernier mirrors that
//!   so the `min` is never taken against a boundary-band term whose
//!   crowd-side semantics are undefined.

use ndarray::ArrayViewMut2;
use vernier_mask::ops::boundary_band;
use vernier_mask::Rle;

use super::bbox::{BboxAnn, BboxIou};
use super::segm::{to_bbox_ann, SegmAnn};
use super::Similarity;
use crate::boundary_parity::BOUNDARY_DILATION_RATIO_DEFAULT;
use crate::error::EvalError;

/// Boundary IoU [`Similarity`] impl. Carries its `dilation_ratio`
/// configuration; the matching engine reads only the [`Similarity`]
/// trait so the knob lives here, not in matching (per ADR-0005).
///
/// The annotation type is reused from the segm kernel
/// ([`SegmAnn`]): boundary IoU consumes the same RLE plus crowd-flag
/// shape — the discriminator is the impl, not the data.
#[derive(Debug, Clone, Copy)]
pub struct BoundaryIou {
    /// Chebyshev-ball dilation ratio (Cheng et al. 2021). Default
    /// [`BOUNDARY_DILATION_RATIO_DEFAULT`] = 0.02; LVIS uses 0.008.
    /// Quirk **M4** disposition `corrected`: surfaced as a public field
    /// rather than hardcoded at the call site.
    pub dilation_ratio: f64,
}

impl Default for BoundaryIou {
    fn default() -> Self {
        Self {
            dilation_ratio: BOUNDARY_DILATION_RATIO_DEFAULT,
        }
    }
}

impl Similarity for BoundaryIou {
    type Annotation = SegmAnn;

    fn compute(
        &self,
        gts: &[SegmAnn],
        dts: &[SegmAnn],
        out: &mut ArrayViewMut2<'_, f64>,
    ) -> Result<(), EvalError> {
        if out.nrows() != gts.len() || out.ncols() != dts.len() {
            return Err(EvalError::DimensionMismatch {
                detail: format!(
                    "boundary IoU output is {}x{}, expected {}x{}",
                    out.nrows(),
                    out.ncols(),
                    gts.len(),
                    dts.len()
                ),
            });
        }
        if gts.is_empty() || dts.is_empty() {
            return Ok(());
        }

        let (h, w) = (gts[0].rle.h, gts[0].rle.w);
        for r in gts.iter().chain(dts.iter()).map(|a| &a.rle) {
            if r.h != h || r.w != w {
                return Err(EvalError::DimensionMismatch {
                    detail: format!(
                        "boundary IoU expects all RLEs at [{h}, {w}]; got [{}, {}]",
                        r.h, r.w
                    ),
                });
            }
        }

        // I1 prefilter: bbox IoU on the tight RLE bboxes. Cells whose
        // bbox IoU is 0 stay zero; non-zero cells get overwritten with
        // the boundary IoU below. BboxIou honors the same E1 crowd
        // asymmetry, so the gate is correct on crowd rows too.
        let g_bbox: Vec<BboxAnn> = gts
            .iter()
            .map(|g| to_bbox_ann(&g.rle, g.is_crowd))
            .collect();
        let d_bbox: Vec<BboxAnn> = dts.iter().map(|d| to_bbox_ann(&d.rle, false)).collect();
        BboxIou.compute(&g_bbox, &d_bbox, out)?;

        // O1/O2: skip B(g) for crowd GTs — the boundary fold is
        // suppressed on crowd rows, so computing the band is wasted
        // work proportional to the (often large) crowd-mask area.
        // An empty placeholder keeps Vec indices aligned with `gts`
        // and is never read past the crowd guard below.
        // TODO(ADR-0010 §"Performance baseline"): parallelise both
        // g_band and d_band precomputation via rayon when total count
        // exceeds ~16. Sequential ships first.
        let crowd_placeholder = Rle {
            h,
            w,
            counts: vec![],
        };
        let g_band: Vec<Rle> = gts
            .iter()
            .map(|g| {
                if g.is_crowd {
                    Ok(crowd_placeholder.clone())
                } else {
                    boundary_band(&g.rle, self.dilation_ratio)
                }
            })
            .collect::<Result<_, _>>()?;
        let d_band: Vec<Rle> = dts
            .iter()
            .map(|d| boundary_band(&d.rle, self.dilation_ratio))
            .collect::<Result<_, _>>()?;

        let g_mask_area: Vec<u64> = gts.iter().map(|g| g.rle.area()).collect();
        let d_mask_area: Vec<u64> = dts.iter().map(|d| d.rle.area()).collect();
        let g_bound_area: Vec<u64> = g_band.iter().map(Rle::area).collect();
        let d_bound_area: Vec<u64> = d_band.iter().map(Rle::area).collect();

        for g in 0..gts.len() {
            let crowd = gts[g].is_crowd;
            for d in 0..dts.len() {
                if out[[g, d]] <= 0.0 {
                    continue;
                }
                let inter_mask = gts[g].rle.intersect_area(&dts[d].rle)?;
                let mask_denom = if crowd {
                    d_mask_area[d]
                } else {
                    g_mask_area[g] + d_mask_area[d] - inter_mask
                };
                let mask_iou = if mask_denom > 0 && inter_mask > 0 {
                    (inter_mask as f64) / (mask_denom as f64)
                } else {
                    0.0
                };

                // Folding `min` against the crowd-side band term would
                // invent semantics the spec does not define (O1/O2),
                // and we skipped its precomputation above.
                if crowd {
                    out[[g, d]] = mask_iou;
                    continue;
                }

                let inter_bound = g_band[g].intersect_area(&d_band[d])?;
                let bound_denom = g_bound_area[g] + d_bound_area[d] - inter_bound;
                let bound_iou = if bound_denom > 0 && inter_bound > 0 {
                    (inter_bound as f64) / (bound_denom as f64)
                } else {
                    0.0
                };

                out[[g, d]] = mask_iou.min(bound_iou);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn ann(rle: Rle, is_crowd: bool) -> SegmAnn {
        SegmAnn { rle, is_crowd }
    }

    fn rle(h: u32, w: u32, counts: Vec<u32>) -> Rle {
        Rle { h, w, counts }
    }

    fn compute(gts: &[SegmAnn], dts: &[SegmAnn]) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((gts.len(), dts.len()));
        BoundaryIou::default()
            .compute(gts, dts, &mut out.view_mut())
            .unwrap();
        out
    }

    /// Builds an RLE for a filled axis-aligned rectangle inside `(h,
    /// w)`. Column-major: `counts = [bg_before_col, fg_h, bg_between,
    /// fg_h, ..., bg_after]`.
    fn filled_rect(h: u32, w: u32, x0: u32, y0: u32, rw: u32, rh: u32) -> Rle {
        let mut raster = vec![0u8; (h as usize) * (w as usize)];
        for x in x0..x0 + rw {
            for y in y0..y0 + rh {
                raster[(x as usize) * (h as usize) + (y as usize)] = 1;
            }
        }
        Rle::from_raster_bytes(&raster, h, w).unwrap()
    }

    #[test]
    fn perfect_overlap_is_one() {
        // Identical masks → mask IoU = 1, band IoU = 1, min = 1.
        let r = rle(2, 2, vec![0, 4]);
        let m = compute(&[ann(r.clone(), false)], &[ann(r, false)]);
        assert_eq!(m[[0, 0]].to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn disjoint_masks_are_zero_via_bbox_prefilter() {
        // GT covers the upper-left pixel; DT covers the lower-right
        // pixel. Their bboxes don't overlap, so I1 short-circuits to 0
        // without computing band intersections.
        let g = rle(2, 2, vec![0, 1, 3]);
        let d = rle(2, 2, vec![3, 1]);
        let m = compute(&[ann(g, false)], &[ann(d, false)]);
        assert_eq!(m[[0, 0]].to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn small_mask_band_clamps_to_full_mask() {
        // A 4×4 image gives sqrt(32) ≈ 5.66; at ratio 0.02,
        // round(0.113) = 0 → clamped to d = 1 (M3). Erosion by radius 1
        // of a 1×1 mask is empty, so the band equals the mask. With
        // both bands == masks, boundary_iou == mask_iou and `min` is a
        // no-op. GT area 1, DT area 2, inter 1 → IoU = 1/2.
        let g = rle(4, 4, vec![0, 1, 15]);
        let d = rle(4, 4, vec![0, 2, 14]);
        let m = compute(&[ann(g, false)], &[ann(d, false)]);
        assert_eq!(m[[0, 0]].to_bits(), (1.0_f64 / 2.0_f64).to_bits());
    }

    #[test]
    fn partial_overlap_equals_min_of_mask_and_bound_iou() {
        // Two 10×10 squares offset by 5 columns inside a 20×20 image.
        // Big enough for d=1 erosion to leave non-trivial bands.
        // sqrt(800) ≈ 28.28; at ratio 0.04, round(1.131) = 1.
        //
        // GT: rect at (x=0, y=5), 10×10. DT: rect at (x=5, y=5), 10×10.
        // Mask intersect = 5 cols × 10 rows = 50.
        // Mask union = 100 + 100 - 50 = 150 → mask_iou = 1/3.
        //
        // The bands are the 1-pixel frames of each square (each band
        // has area 100 - 64 = 36). The two frames overlap; we compute
        // the band IoU directly from the same primitives and verify
        // that the kernel returned min(mask_iou, band_iou).
        let h = 20;
        let w = 20;
        let gt = filled_rect(h, w, 0, 5, 10, 10);
        let dt = filled_rect(h, w, 5, 5, 10, 10);
        let kernel = BoundaryIou {
            dilation_ratio: 0.04,
        };
        let mut out = Array2::<f64>::zeros((1, 1));
        kernel
            .compute(
                &[ann(gt.clone(), false)],
                &[ann(dt.clone(), false)],
                &mut out.view_mut(),
            )
            .unwrap();

        let g_band = boundary_band(&gt, 0.04).unwrap();
        let d_band = boundary_band(&dt, 0.04).unwrap();
        let inter_mask = gt.intersect_area(&dt).unwrap();
        let mask_iou = (inter_mask as f64) / ((gt.area() + dt.area() - inter_mask) as f64);
        let inter_bound = g_band.intersect_area(&d_band).unwrap();
        let bound_iou =
            (inter_bound as f64) / ((g_band.area() + d_band.area() - inter_bound) as f64);
        let expected = mask_iou.min(bound_iou);

        // Sanity: this is the case the test was written to exercise —
        // the bands really do score lower than the masks, so `min` is
        // a non-trivial fold.
        assert!(bound_iou < mask_iou);
        assert_eq!(out[[0, 0]].to_bits(), expected.to_bits());
    }

    #[test]
    fn e1_o1_crowd_gt_uses_mask_iou_alone() {
        // GT covers the whole 4×4 image (area 16) as crowd.
        // DT is a single pixel inside (area 1). E1: crowd mask IoU =
        // inter / dt_area = 1/1 = 1.0. O1/O2: boundary suppressed for
        // crowd GT. If the kernel mistakenly folded the band term in,
        // the cell would be < 1.0 (the bands would not be identical),
        // so this fixture pins both quirks at once.
        let gt_full = rle(4, 4, vec![0, 16]);
        let dt_pixel = rle(4, 4, vec![5, 1, 10]);
        let m = compute(&[ann(gt_full, true)], &[ann(dt_pixel, false)]);
        assert_eq!(m[[0, 0]].to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn dt_iscrowd_flag_is_ignored() {
        // E2/J4: DT iscrowd is enforced 0 at load. A smuggled
        // is_crowd=true on the DT side must not change the answer.
        let g = rle(2, 2, vec![0, 1, 3]);
        let d = rle(2, 2, vec![0, 2, 2]);
        let with_flag = compute(&[ann(g.clone(), false)], &[ann(d.clone(), true)]);
        let without = compute(&[ann(g, false)], &[ann(d, false)]);
        assert_eq!(with_flag[[0, 0]].to_bits(), without[[0, 0]].to_bits());
    }

    #[test]
    fn empty_masks_pair_is_zero_not_nan() {
        // Empty GT and DT: areas all zero, denominators all zero,
        // guards return 0.0 on both mask and band terms; min is 0.
        let empty = rle(2, 2, vec![4]);
        let dt_one = rle(2, 2, vec![0, 1, 3]);
        let m = compute(&[ann(empty.clone(), false)], &[ann(dt_one, false)]);
        assert!(m[[0, 0]].is_finite());
        assert_eq!(m[[0, 0]].to_bits(), 0.0_f64.to_bits());
        let m = compute(&[ann(empty.clone(), false)], &[ann(empty, false)]);
        assert_eq!(m[[0, 0]].to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn empty_inputs_return_unchanged_matrix() {
        let dts: Vec<SegmAnn> = (0..3).map(|_| ann(rle(2, 2, vec![4]), false)).collect();
        let mut out = Array2::<f64>::from_elem((0, 3), 7.0);
        BoundaryIou::default()
            .compute(&[], &dts, &mut out.view_mut())
            .unwrap();
        assert_eq!(out.shape(), &[0, 3]);
    }

    #[test]
    fn output_shape_mismatch_returns_typed_error() {
        let g = ann(rle(2, 2, vec![4]), false);
        let d = ann(rle(2, 2, vec![4]), false);
        let mut out = Array2::<f64>::zeros((2, 3));
        let err = BoundaryIou::default()
            .compute(&[g], &[d], &mut out.view_mut())
            .unwrap_err();
        assert!(matches!(err, EvalError::DimensionMismatch { .. }));
    }

    #[test]
    fn rle_dimension_mismatch_returns_typed_error() {
        let g = ann(rle(4, 4, vec![16]), false);
        let d = ann(rle(8, 8, vec![64]), false);
        let mut out = Array2::<f64>::zeros((1, 1));
        let err = BoundaryIou::default()
            .compute(&[g], &[d], &mut out.view_mut())
            .unwrap_err();
        match err {
            EvalError::DimensionMismatch { detail } => {
                assert!(detail.contains("[4, 4]"));
                assert!(detail.contains("[8, 8]"));
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn default_dilation_ratio_is_pinned_constant() {
        assert_eq!(
            BoundaryIou::default().dilation_ratio,
            BOUNDARY_DILATION_RATIO_DEFAULT
        );
    }

    #[test]
    fn custom_dilation_ratio_flows_through_to_bands() {
        // Same fixture as `partial_overlap_…` (which pins ratio 0.04
        // bit-exactly). At ratio 0.10, sqrt(800) ≈ 28.28 →
        // round(2.828) = 3, so the bands widen and the min-folded
        // output shifts. Pin the d=3 case bit-exactly against
        // primitives, then assert the two ratios disagree — proves
        // the public `dilation_ratio` field actually reaches the
        // kernel and isn't shadowed by the default.
        let h = 20;
        let w = 20;
        let gt = filled_rect(h, w, 0, 5, 10, 10);
        let dt = filled_rect(h, w, 5, 5, 10, 10);

        let run = |ratio: f64| -> f64 {
            let mut out = Array2::<f64>::zeros((1, 1));
            BoundaryIou {
                dilation_ratio: ratio,
            }
            .compute(
                &[ann(gt.clone(), false)],
                &[ann(dt.clone(), false)],
                &mut out.view_mut(),
            )
            .unwrap();
            out[[0, 0]]
        };

        let large_ratio = 0.10;
        let g_band = boundary_band(&gt, large_ratio).unwrap();
        let d_band = boundary_band(&dt, large_ratio).unwrap();
        let inter_mask = gt.intersect_area(&dt).unwrap();
        let mask_iou = (inter_mask as f64) / ((gt.area() + dt.area() - inter_mask) as f64);
        let inter_bound = g_band.intersect_area(&d_band).unwrap();
        let bound_iou =
            (inter_bound as f64) / ((g_band.area() + d_band.area() - inter_bound) as f64);
        let expected_large = mask_iou.min(bound_iou);

        let actual_small = run(0.04);
        let actual_large = run(large_ratio);
        assert_eq!(actual_large.to_bits(), expected_large.to_bits());
        assert_ne!(actual_small.to_bits(), actual_large.to_bits());
    }

    #[test]
    fn impl_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BoundaryIou>();
    }
}
