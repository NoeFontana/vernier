//! RLE-mask IoU (`iouType="segm"`).
//!
//! Mirrors `pycocotools.maskUtils.iou` for RLE inputs: a bbox-IoU
//! prefilter (quirk **I1**) eliminates non-overlapping pairs cheaply,
//! then a per-pair RLE sweep computes the precise intersection area.
//! The bbox prefilter reuses [`BboxIou`] verbatim — that is the entire
//! reason it's a [`Similarity`] impl rather than a free function.
//!
//! Per ADR-0008, every divide is `f64` so each cell is bit-equal to
//! pycocotools' double-precision result.
//!
//! ## Quirk dispositions
//!
//! - **E1** (`strict`): crowd asymmetry. When GT is crowd, the
//!   denominator is `dt_area`, not the union. Applied identically on
//!   the bbox prefilter and on the final RLE-pair denominator.
//! - **I1** (`strict`): bbox-IoU prefilter. Pairs whose tight bboxes
//!   don't overlap are zero by construction; they skip the RLE sweep.
//! - **F5** (`aligned`): empty `gts` or `dts` returns the zero-shape
//!   matrix unchanged.
//! - **H2** (`corrected`): all RLEs in one call must share `(h, w)`.
//!   Mismatch raises [`EvalError::DimensionMismatch`] instead of the
//!   `-1` sentinel pycocotools' `rleIou` writes per cell.
//! - **E2 / J4**: DT `is_crowd` is enforced 0 at the dataset boundary;
//!   here we simply ignore the field on DT side, matching [`BboxIou`].

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use ndarray::ArrayViewMut2;
use vernier_mask::Rle;

use super::bbox::{BboxAnn, BboxIou};
use super::Similarity;
use crate::dataset::Bbox;
use crate::error::EvalError;

/// Annotation shape consumed by [`SegmIou`]. The matching engine
/// constructs these from a [`crate::dataset::CocoAnnotation`] (after
/// calling [`crate::segmentation::Segmentation::to_rle`]) before
/// invoking [`Similarity::compute`].
///
/// All `rle` fields in one `compute` call must share `(h, w)` — the
/// per-image image dimensions. Mismatch is **H2** `corrected`.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmAnn {
    /// Pre-rasterized mask. Polygons are normalized via
    /// [`Rle::from_polygons`] (quirk K2) at the dataset boundary.
    pub rle: Rle,
    /// Crowd flag. Drives the **E1** asymmetry on the GT side; ignored
    /// on the DT side (quirks **E2** / **J4** enforce DT `iscrowd=0`).
    pub is_crowd: bool,
    /// Source annotation id from the dataset (`CocoAnnotation::id` /
    /// `CocoDetection::id`). Used by
    /// [`crate::similarity::BoundaryGtCache`] as the GT cache key in
    /// the boundary kernel; ignored by [`SegmIou`].
    pub ann_id: i64,
}

/// Segm IoU [`Similarity`] impl. Stateless.
#[derive(Debug, Default, Clone, Copy)]
pub struct SegmIou;

impl Similarity for SegmIou {
    type Annotation = SegmAnn;

    fn compute(
        &self,
        gts: &[SegmAnn],
        dts: &[SegmAnn],
        out: &mut ArrayViewMut2<'_, f64>,
    ) -> Result<(), EvalError> {
        let mut scratch = SegmComputeScratch::new();
        segm_iou_compute(gts, dts, out, &mut scratch, None)
    }
}

/// Reusable per-call buffers for the segm-IoU kernel. Lets the
/// dataset-wide path (`evaluate_segm` via the private `SegmIouCached`
/// kernel) amortize per-cell allocations across the ~36 k anns of a
/// val2017 pass — `Vec::clear` + amortized capacity instead of fresh
/// `Vec::new` per `(image, category)` cell. The GT-side buffers double
/// as the destination for [`SegmGtCache`] hits when one is supplied.
#[derive(Default)]
pub(crate) struct SegmComputeScratch {
    g_bbox: Vec<BboxAnn>,
    d_bbox: Vec<BboxAnn>,
    g_area: Vec<u64>,
    d_area: Vec<u64>,
}

impl SegmComputeScratch {
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// Cross-call cache of GT bbox + area for the segm IoU kernel.
///
/// In a training loop, validation passes call `evaluate_segm`
/// repeatedly against the same GT but a fresh DT each epoch. Both
/// `Rle::bbox` (the I1 prefilter input) and `Rle::area` (the union
/// denominator) walk the GT counts on every call. A cache amortises
/// those walks over the training run. Pass an instance to
/// [`crate::evaluate_segm_cached`] and reuse it across calls.
///
/// Keyed by GT annotation id ([`SegmAnn::ann_id`], populated from
/// `CocoAnnotation::id` at the dataset boundary).
///
/// Threadsafe via an internal [`Mutex`] (the kernel needs `Sync`).
/// Single-threaded use is uncontended.
#[derive(Default)]
pub struct SegmGtCache {
    inner: Mutex<HashMap<i64, (BboxAnn, u64)>>,
}

impl SegmGtCache {
    /// Constructs an empty cache. Equivalent to [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of GT annotations currently held.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Returns `true` if no GT entries are currently cached.
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// Drops all cached entries. Useful when the GT dataset changes
    /// mid-loop so stale `(ann_id, bbox, area)` triples don't pollute
    /// the next call.
    pub fn clear(&self) {
        self.lock().clear();
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<i64, (BboxAnn, u64)>> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }
}

/// Scratch-aware segm-IoU compute. Same semantics as
/// [`SegmIou::compute`] but reuses caller-owned buffers across the
/// dataset pass.
///
/// When `gt_cache` is `Some`, GT bbox + area are looked up by
/// [`SegmAnn::ann_id`]; misses fall through to a fresh derivation and
/// populate the cache.
pub(crate) fn segm_iou_compute(
    gts: &[SegmAnn],
    dts: &[SegmAnn],
    out: &mut ArrayViewMut2<'_, f64>,
    scratch: &mut SegmComputeScratch,
    gt_cache: Option<&SegmGtCache>,
) -> Result<(), EvalError> {
    if out.nrows() != gts.len() || out.ncols() != dts.len() {
        return Err(EvalError::DimensionMismatch {
            detail: format!(
                "segm IoU output is {}x{}, expected {}x{}",
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
                    "segm IoU expects all RLEs at [{h}, {w}]; got [{}, {}]",
                    r.h, r.w
                ),
            });
        }
    }

    // I1 prefilter: bbox IoU on the tight RLE bboxes. Cells with
    // bb_iou == 0 stay zero; non-zero cells get overwritten with
    // the exact RLE IoU below. BboxIou honors the same E1 crowd
    // asymmetry, so the gate is correct on crowd rows too.
    //
    // GT-side bbox + area are populated together — both walk the same
    // counts array, so a `SegmGtCache` amortises both at once.
    scratch.g_bbox.clear();
    scratch.g_area.clear();
    populate_gt_bbox_area(gts, scratch, gt_cache);
    scratch.d_bbox.clear();
    scratch.d_bbox.extend(dts.iter().map(|d| to_bbox_ann(&d.rle, false)));
    scratch.d_area.clear();
    scratch.d_area.extend(dts.iter().map(|d| d.rle.area()));
    BboxIou.compute(&scratch.g_bbox, &scratch.d_bbox, out)?;

    for g in 0..gts.len() {
        let crowd = gts[g].is_crowd;
        let ga = scratch.g_area[g];
        for d in 0..dts.len() {
            if out[[g, d]] <= 0.0 {
                continue;
            }
            let inter = gts[g].rle.intersect_area(&dts[d].rle)?;
            let denom = if crowd {
                scratch.d_area[d]
            } else {
                ga + scratch.d_area[d] - inter
            };
            out[[g, d]] = if denom > 0 && inter > 0 {
                (inter as f64) / (denom as f64)
            } else {
                0.0
            };
        }
    }

    Ok(())
}

/// Populate `scratch.g_bbox` + `scratch.g_area` for `gts`, consulting
/// `cache` when present. Vecs are assumed already cleared with
/// capacity preserved. Hits avoid the RLE walks that
/// [`Rle::bbox`] / [`Rle::area`] would otherwise do per call.
fn populate_gt_bbox_area(
    gts: &[SegmAnn],
    scratch: &mut SegmComputeScratch,
    cache: Option<&SegmGtCache>,
) {
    let Some(cache) = cache else {
        scratch.g_bbox.extend(gts.iter().map(|g| to_bbox_ann(&g.rle, g.is_crowd)));
        scratch.g_area.extend(gts.iter().map(|g| g.rle.area()));
        return;
    };
    let mut inner = cache.lock();
    for g in gts {
        let entry = inner
            .entry(g.ann_id)
            .or_insert_with(|| (to_bbox_ann(&g.rle, g.is_crowd), g.rle.area()));
        scratch.g_bbox.push(entry.0);
        scratch.g_area.push(entry.1);
    }
}

pub(super) fn to_bbox_ann(rle: &Rle, is_crowd: bool) -> BboxAnn {
    let [x, y, w, h] = rle.bbox();
    BboxAnn {
        bbox: Bbox {
            x: f64::from(x),
            y: f64::from(y),
            w: f64::from(w),
            h: f64::from(h),
        },
        is_crowd,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn ann(rle: Rle, is_crowd: bool) -> SegmAnn {
        SegmAnn {
            rle,
            is_crowd,
            ann_id: 0,
        }
    }

    fn rle(h: u32, w: u32, counts: Vec<u32>) -> Rle {
        Rle { h, w, counts }
    }

    fn compute(gts: &[SegmAnn], dts: &[SegmAnn]) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((gts.len(), dts.len()));
        SegmIou.compute(gts, dts, &mut out.view_mut()).unwrap();
        out
    }

    #[test]
    fn perfect_overlap_is_one() {
        let r = rle(2, 2, vec![0, 4]);
        let m = compute(&[ann(r.clone(), false)], &[ann(r, false)]);
        assert_eq!(m[[0, 0]].to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn disjoint_masks_are_zero_via_bbox_prefilter() {
        // GT covers the upper-left pixel; DT covers the lower-right
        // pixel. Their bboxes don't overlap, so I1 short-circuits to 0
        // without invoking the RLE sweep.
        let g = rle(2, 2, vec![0, 1, 3]);
        let d = rle(2, 2, vec![3, 1]);
        let m = compute(&[ann(g, false)], &[ann(d, false)]);
        assert_eq!(m[[0, 0]].to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn partial_overlap_matches_hand_traced_ratio() {
        // GT area 1, DT area 2, inter 1 → IoU = 1 / (1 + 2 - 1) = 1/2.
        let g = rle(2, 2, vec![0, 1, 3]);
        let d = rle(2, 2, vec![0, 2, 2]);
        let m = compute(&[ann(g, false)], &[ann(d, false)]);
        assert_eq!(m[[0, 0]].to_bits(), (1.0_f64 / 2.0_f64).to_bits());
    }

    #[test]
    fn e1_crowd_gt_uses_dt_area_denominator() {
        // GT covers the whole 4×4 image (area 16) as crowd.
        // DT is a single pixel inside (area 1).
        // Symmetric IoU = 1/16; crowd IoU = 1/1 = 1.0.
        let gt_full = rle(4, 4, vec![0, 16]);
        let dt_pixel = rle(4, 4, vec![5, 1, 10]);
        let crowd_m = compute(
            &[ann(gt_full.clone(), true)],
            &[ann(dt_pixel.clone(), false)],
        );
        let normal_m = compute(&[ann(gt_full, false)], &[ann(dt_pixel, false)]);
        assert_eq!(crowd_m[[0, 0]].to_bits(), 1.0_f64.to_bits());
        assert_eq!(normal_m[[0, 0]].to_bits(), (1.0_f64 / 16.0_f64).to_bits());
    }

    #[test]
    fn dt_iscrowd_flag_is_ignored() {
        // E2/J4: DT iscrowd is enforced 0 at load. Even if a caller
        // smuggles is_crowd=true into a DT, the IoU must equal the
        // clean version.
        //
        // E3 cross-ref: the asymmetric crowd `iou()` API is enforced by
        // the type signature — `Similarity::compute` takes one GT slice
        // and one DT slice with no parallel `dt_iscrowd` vector — so
        // there is no DT-side crowd input the kernel could branch on.
        // This runtime check pins the observable behavior; the type
        // system covers the structural side.
        let g = rle(2, 2, vec![0, 1, 3]);
        let d = rle(2, 2, vec![0, 2, 2]);
        let with_flag = compute(&[ann(g.clone(), false)], &[ann(d.clone(), true)]);
        let without = compute(&[ann(g, false)], &[ann(d, false)]);
        assert_eq!(with_flag[[0, 0]].to_bits(), without[[0, 0]].to_bits());
    }

    #[test]
    fn empty_gt_or_dt_pair_is_zero_not_nan() {
        // Empty mask has area 0. Non-crowd: denom = 0 + d_area - 0;
        // if d_area is also 0, denom=0 and the guard returns 0.0.
        let empty = rle(2, 2, vec![4]);
        let dt_one = rle(2, 2, vec![0, 1, 3]);
        let m = compute(&[ann(empty.clone(), false)], &[ann(dt_one, false)]);
        assert!(m[[0, 0]].is_finite());
        assert_eq!(m[[0, 0]].to_bits(), 0.0_f64.to_bits());
        let m = compute(&[ann(empty.clone(), false)], &[ann(empty, false)]);
        assert_eq!(m[[0, 0]].to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn rle_dimension_mismatch_returns_typed_error() {
        let g = ann(rle(4, 4, vec![16]), false);
        let d = ann(rle(8, 8, vec![64]), false);
        let mut out = Array2::<f64>::zeros((1, 1));
        let err = SegmIou
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
    fn output_shape_mismatch_returns_typed_error() {
        let g = ann(rle(2, 2, vec![4]), false);
        let d = ann(rle(2, 2, vec![4]), false);
        let mut out = Array2::<f64>::zeros((2, 3));
        let err = SegmIou
            .compute(&[g], &[d], &mut out.view_mut())
            .unwrap_err();
        assert!(matches!(err, EvalError::DimensionMismatch { .. }));
    }

    #[test]
    fn empty_inputs_return_unchanged_matrix() {
        let dts: Vec<SegmAnn> = (0..3).map(|_| ann(rle(2, 2, vec![4]), false)).collect();
        let mut out = Array2::<f64>::from_elem((0, 3), 7.0);
        SegmIou.compute(&[], &dts, &mut out.view_mut()).unwrap();
        assert_eq!(out.shape(), &[0, 3]);
    }

    #[test]
    fn three_by_three_matrix_exercises_prefilter_and_crowd() {
        // Layout in a 4×4 image (column-major):
        // - g0: pixel (0,0) only.
        // - g1: pixel (3,3) only — disjoint from g0 and most dts.
        // - g2: full image, crowd.
        // - d0: pixel (0,0) — equals g0.
        // - d1: pixels (0,0) and (1,0) — partial overlap with g0.
        // - d2: pixel (3,3) — equals g1.
        let g0 = rle(4, 4, vec![0, 1, 15]);
        let g1 = rle(4, 4, vec![15, 1]);
        let g2 = rle(4, 4, vec![0, 16]);
        let d0 = rle(4, 4, vec![0, 1, 15]);
        let d1 = rle(4, 4, vec![0, 1, 3, 1, 11]);
        let d2 = rle(4, 4, vec![15, 1]);

        let m = compute(
            &[ann(g0, false), ann(g1, false), ann(g2, true)],
            &[ann(d0, false), ann(d1, false), ann(d2, false)],
        );

        assert_eq!(m[[0, 0]].to_bits(), 1.0_f64.to_bits());
        assert_eq!(m[[0, 1]].to_bits(), (1.0_f64 / 2.0_f64).to_bits());
        assert_eq!(m[[0, 2]].to_bits(), 0.0_f64.to_bits());

        assert_eq!(m[[1, 0]].to_bits(), 0.0_f64.to_bits());
        assert_eq!(m[[1, 1]].to_bits(), 0.0_f64.to_bits());
        assert_eq!(m[[1, 2]].to_bits(), 1.0_f64.to_bits());

        assert_eq!(m[[2, 0]].to_bits(), 1.0_f64.to_bits());
        assert_eq!(m[[2, 1]].to_bits(), 1.0_f64.to_bits());
        assert_eq!(m[[2, 2]].to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn impl_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SegmIou>();
    }
}
