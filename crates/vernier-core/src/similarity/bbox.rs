//! Axis-aligned bbox IoU.
//!
//! Mirrors `pycocotools.cocoeval.COCOeval.computeIoU` for `iouType="bbox"`.
//! Per ADR-0008, intermediates are `f64` end-to-end so the kernel matches
//! pycocotools' `maskUtils.iou` (also f64) bit-for-bit. Per ADR-0003, the
//! inner loop is wrapped in [`pulp::Arch::dispatch`] so it compiles to
//! AVX2 / AVX-512 / NEON variants picked at process start.
//!
//! ## Quirk dispositions
//!
//! - **E1** (`strict`): when GT is a crowd region, IoU is asymmetric —
//!   `intersect / dt_area`, *not* `intersect / union`. A small DT inside
//!   a large crowd scores 1.0. The asymmetry lives here so that
//!   matching code stays IoU-type-agnostic (per ADR-0005).
//! - **I3** (`aligned`): pycocotools uses two different zero guards
//!   (`u==0` for RLE, `w<=0 || h<=0` for bbox). Both yield IoU=0; we
//!   express it as a single `denom > 0` guard at the last step.
//! - **I4** (`strict`): edge-sharing boxes (e.g. `[0,0,1,1]` and
//!   `[1,0,1,1]`) yield zero IoU. Falls out of the `(min - max).max(0)`
//!   intersection formula automatically.
//!
//! Quirks **E2** and **J4** (DT `iscrowd` is always 0) are enforced at
//! the future `loadRes`-equivalent on the dataset side, not here. The
//! `dts` slice this kernel receives may carry an `is_crowd` field for
//! storage symmetry, but it is ignored — only `gts[g].is_crowd` drives
//! the asymmetric branch.

use ndarray::ArrayViewMut2;

use super::Similarity;
use crate::dataset::Bbox;
use crate::error::EvalError;

/// Annotation shape consumed by [`BboxIou`]. The matching engine
/// constructs these from a concrete [`crate::dataset::CocoAnnotation`]
/// (or any future [`crate::dataset::EvalDataset`] impl) before invoking
/// [`Similarity::compute`].
///
/// Kept deliberately minimal: only the fields the kernel actually reads.
/// Other metadata (image_id, category_id, area, score) flows through
/// the matching engine's parallel arrays, not through here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BboxAnn {
    /// Axis-aligned bounding box (COCO `(x, y, w, h)` convention).
    pub bbox: Bbox,
    /// Crowd flag. Drives the E1 asymmetry on the GT side; ignored on
    /// the DT side.
    pub is_crowd: bool,
}

/// Bbox IoU [`Similarity`] impl. Stateless.
#[derive(Debug, Default, Clone, Copy)]
pub struct BboxIou;

impl Similarity for BboxIou {
    type Annotation = BboxAnn;

    fn compute(
        &self,
        gts: &[BboxAnn],
        dts: &[BboxAnn],
        out: &mut ArrayViewMut2<'_, f64>,
    ) -> Result<(), EvalError> {
        if out.nrows() != gts.len() || out.ncols() != dts.len() {
            return Err(EvalError::DimensionMismatch {
                detail: format!(
                    "bbox IoU output is {}x{}, expected {}x{}",
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

        // `dispatch` runs the closure with the best-available SIMD
        // target features enabled, so LLVM auto-vectorizes the inner
        // loop across AVX2 / AVX-512 / NEON without per-arch source
        // duplication. The crowd flag (E1) is hoisted to the outer loop
        // so each inner pass is branch-free FMA-chain math.
        let arch = pulp::Arch::new();
        arch.dispatch(|| {
            for (g, gt) in gts.iter().enumerate() {
                let gxa = gt.bbox.x;
                let gya = gt.bbox.y;
                let gw = gt.bbox.w;
                let gh = gt.bbox.h;
                let gxb = gxa + gw;
                let gyb = gya + gh;
                let g_area = gw * gh;

                let mut row = out.row_mut(g);
                if gt.is_crowd {
                    for (d, dt) in dts.iter().enumerate() {
                        row[d] = iou_pair(gxa, gya, gxb, gyb, dt.bbox, CrowdDenom);
                    }
                } else {
                    for (d, dt) in dts.iter().enumerate() {
                        row[d] = iou_pair(gxa, gya, gxb, gyb, dt.bbox, UnionDenom(g_area));
                    }
                }
            }
        });

        Ok(())
    }
}

/// Marker trait for the E1 crowd branch hoisted out of the inner loop.
///
/// Crowd GT uses the asymmetric `intersect / dt_area`; non-crowd GT uses
/// the symmetric `intersect / (g_area + d_area - intersect)`. Choosing
/// once per GT row keeps each inner loop branch-free.
trait Denom: Copy {
    fn denom(self, d_area: f64, inter: f64) -> f64;
}

#[derive(Clone, Copy)]
struct CrowdDenom;
impl Denom for CrowdDenom {
    #[inline(always)]
    fn denom(self, d_area: f64, _inter: f64) -> f64 {
        d_area
    }
}

#[derive(Clone, Copy)]
struct UnionDenom(f64);
impl Denom for UnionDenom {
    #[inline(always)]
    fn denom(self, d_area: f64, inter: f64) -> f64 {
        self.0 + d_area - inter
    }
}

#[inline(always)]
fn iou_pair<D: Denom>(gxa: f64, gya: f64, gxb: f64, gyb: f64, dt: Bbox, denom: D) -> f64 {
    let dxa = dt.x;
    let dya = dt.y;
    let dw = dt.w;
    let dh = dt.h;
    let dxb = dxa + dw;
    let dyb = dya + dh;
    let d_area = dw * dh;

    // Quirk I4: edge-sharing → zero. `(min - max).max(0)` gives 0 when
    // the boxes touch on a side rather than overlap.
    let iw = (gxb.min(dxb) - gxa.max(dxa)).max(0.0);
    let ih = (gyb.min(dyb) - gya.max(dya)).max(0.0);
    let inter = iw * ih;

    let denom = denom.denom(d_area, inter);
    // Quirk I3: single zero-denominator guard.
    if denom > 0.0 {
        inter / denom
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_ann(x: f64, y: f64, w: f64, h: f64, is_crowd: bool) -> BboxAnn {
        BboxAnn {
            bbox: Bbox { x, y, w, h },
            is_crowd,
        }
    }

    fn compute(gts: &[BboxAnn], dts: &[BboxAnn]) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((gts.len(), dts.len()));
        BboxIou.compute(gts, dts, &mut out.view_mut()).unwrap();
        out
    }

    #[test]
    fn perfect_overlap_is_one() {
        let gts = [make_ann(0.0, 0.0, 10.0, 10.0, false)];
        let dts = [make_ann(0.0, 0.0, 10.0, 10.0, false)];
        let m = compute(&gts, &dts);
        assert_eq!(m[[0, 0]].to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn no_overlap_is_zero() {
        let gts = [make_ann(0.0, 0.0, 1.0, 1.0, false)];
        let dts = [make_ann(10.0, 10.0, 1.0, 1.0, false)];
        let m = compute(&gts, &dts);
        assert_eq!(m[[0, 0]].to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn i4_edge_sharing_is_zero() {
        // Quirk I4: boxes that share an edge but do not overlap have
        // zero IoU. `[0,0,1,1]` and `[1,0,1,1]` touch at x=1.
        let gts = [make_ann(0.0, 0.0, 1.0, 1.0, false)];
        let dts = [make_ann(1.0, 0.0, 1.0, 1.0, false)];
        let m = compute(&gts, &dts);
        assert_eq!(m[[0, 0]].to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn quarter_overlap_matches_hand_traced_value() {
        // GT [0,0,2,2] (area 4); DT [1,1,2,2] (area 4); intersect 1×1=1.
        // IoU = 1 / (4 + 4 - 1) = 1/7, bit-equal to f64 1/7 (ADR-0008).
        let gts = [make_ann(0.0, 0.0, 2.0, 2.0, false)];
        let dts = [make_ann(1.0, 1.0, 2.0, 2.0, false)];
        let m = compute(&gts, &dts);
        let expected = 1.0_f64 / 7.0_f64;
        assert_eq!(m[[0, 0]].to_bits(), expected.to_bits());
    }

    #[test]
    fn e1_crowd_gt_uses_dt_area_denominator() {
        // GT covers the whole image as a crowd; DT is a 1×1 inside it.
        // Symmetric IoU = 1/100 = 0.01. Crowd IoU = inter/dt_area = 1/1
        // = 1.0. The asymmetry is the test.
        let gts_crowd = [make_ann(0.0, 0.0, 10.0, 10.0, true)];
        let gts_normal = [make_ann(0.0, 0.0, 10.0, 10.0, false)];
        let dts = [make_ann(2.0, 2.0, 1.0, 1.0, false)];
        let crowd_m = compute(&gts_crowd, &dts);
        let normal_m = compute(&gts_normal, &dts);
        assert_eq!(crowd_m[[0, 0]].to_bits(), 1.0_f64.to_bits());
        let expected_normal = 1.0_f64 / 100.0_f64;
        assert_eq!(normal_m[[0, 0]].to_bits(), expected_normal.to_bits());
    }

    #[test]
    fn dt_iscrowd_flag_is_ignored() {
        // Quirks E2/J4: DT iscrowd is enforced 0 at load. Even if the
        // caller smuggles `is_crowd: true` into a DT, the kernel must
        // not branch on it — only GT.is_crowd drives the E1 asymmetry.
        let gts = [make_ann(0.0, 0.0, 2.0, 2.0, false)];
        let dts_marked = [make_ann(1.0, 1.0, 2.0, 2.0, true)];
        let dts_clean = [make_ann(1.0, 1.0, 2.0, 2.0, false)];
        let with_flag = compute(&gts, &dts_marked);
        let without = compute(&gts, &dts_clean);
        assert_eq!(with_flag[[0, 0]].to_bits(), without[[0, 0]].to_bits());
    }

    #[test]
    fn zero_area_gt_with_zero_inter_yields_zero_not_nan() {
        // Degenerate GT (w=0). g_area = 0, inter = 0, union = 0 + d_area
        // - 0 = d_area > 0. Returns 0.0, never NaN. Quirk I3.
        let gts = [make_ann(5.0, 5.0, 0.0, 5.0, false)];
        let dts = [make_ann(0.0, 0.0, 10.0, 10.0, false)];
        let m = compute(&gts, &dts);
        assert!(m[[0, 0]].is_finite());
        assert_eq!(m[[0, 0]].to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn zero_area_gt_and_dt_both_zero_yields_zero_via_denom_guard() {
        // Degenerate on both sides: g_area = d_area = inter = 0 so
        // denom = 0. The I3 single-guard returns 0, not NaN.
        let gts = [make_ann(5.0, 5.0, 0.0, 0.0, false)];
        let dts = [make_ann(5.0, 5.0, 0.0, 0.0, false)];
        let m = compute(&gts, &dts);
        assert_eq!(m[[0, 0]].to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn dimension_mismatch_returns_typed_error() {
        let gts = [make_ann(0.0, 0.0, 1.0, 1.0, false); 2];
        let dts = [make_ann(0.0, 0.0, 1.0, 1.0, false); 3];
        let mut out = Array2::<f64>::zeros((1, 1));
        let err = BboxIou
            .compute(&gts, &dts, &mut out.view_mut())
            .unwrap_err();
        match err {
            EvalError::DimensionMismatch { detail } => {
                assert!(detail.contains("2"));
                assert!(detail.contains("3"));
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn empty_inputs_return_unchanged_matrix() {
        // 0 × 3 and 3 × 0 are valid: nothing to compute. The matrix
        // shape just needs to match.
        let dts = [make_ann(0.0, 0.0, 1.0, 1.0, false); 3];
        let mut out = Array2::<f64>::from_elem((0, 3), 7.0);
        BboxIou.compute(&[], &dts, &mut out.view_mut()).unwrap();
        assert_eq!(out.shape(), &[0, 3]);
    }

    #[test]
    fn three_by_three_matrix_all_pairs_evaluated() {
        let gts = [
            make_ann(0.0, 0.0, 2.0, 2.0, false),
            make_ann(5.0, 5.0, 2.0, 2.0, false),
            make_ann(0.0, 0.0, 10.0, 10.0, true),
        ];
        let dts = [
            make_ann(0.0, 0.0, 2.0, 2.0, false),
            make_ann(1.0, 1.0, 2.0, 2.0, false),
            make_ann(20.0, 20.0, 1.0, 1.0, false),
        ];
        let m = compute(&gts, &dts);

        assert_eq!(m[[0, 0]].to_bits(), 1.0_f64.to_bits());
        assert_eq!(m[[0, 1]].to_bits(), (1.0_f64 / 7.0_f64).to_bits());
        assert_eq!(m[[0, 2]].to_bits(), 0.0_f64.to_bits());

        assert_eq!(m[[1, 0]].to_bits(), 0.0_f64.to_bits());
        assert_eq!(m[[1, 1]].to_bits(), 0.0_f64.to_bits());
        assert_eq!(m[[1, 2]].to_bits(), 0.0_f64.to_bits());

        assert_eq!(m[[2, 0]].to_bits(), 1.0_f64.to_bits());
        assert_eq!(m[[2, 1]].to_bits(), 1.0_f64.to_bits());
        assert_eq!(m[[2, 2]].to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn impl_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BboxIou>();
    }
}
