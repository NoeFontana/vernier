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

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use ndarray::ArrayViewMut2;
use vernier_mask::ops::{
    boundary_band_segments_into, intersect_area_offsets, ErodeScratch, SegmentTable,
};

use super::bbox::{BboxAnn, BboxIou};
use super::segm::{to_bbox_ann, SegmAnn};
use super::Similarity;
use crate::boundary_parity::BOUNDARY_DILATION_RATIO_DEFAULT;
use crate::error::EvalError;

/// Reusable per-call buffers for the boundary-IoU kernel. Mirrors
/// `SegmComputeScratch` so the dataset-wide path
/// (`evaluate_boundary` via the private `BoundaryIouCached` kernel)
/// amortises per-cell allocations across the val2017 pass — one
/// `SegmentTable` allocation per buffer instead of one nested
/// `Vec<u64>` per annotation.
#[derive(Default)]
pub(crate) struct BoundaryComputeScratch {
    erode: ErodeScratch,
    g_bbox: Vec<BboxAnn>,
    d_bbox: Vec<BboxAnn>,
    g_mask_area: Vec<u64>,
    d_mask_area: Vec<u64>,
    g_band_area: Vec<u64>,
    d_band_area: Vec<u64>,
    g_mask_segments: SegmentTable,
    g_band_segments: SegmentTable,
    d_mask_segments: SegmentTable,
    d_band_segments: SegmentTable,
}

impl BoundaryComputeScratch {
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// Cross-call cache of GT boundary bands for the boundary IoU
/// kernel.
///
/// In a training loop, validation passes call `evaluate_boundary`
/// repeatedly against the same GT but a fresh DT each epoch. GT
/// band derivation is the dominant per-annotation cost in the
/// boundary kernel (see `docs/engineering/benchmarking/`), and a
/// cache amortises it across calls. Pass an instance to
/// [`crate::evaluate_boundary_cached`] and reuse it across calls.
///
/// Keyed by GT annotation id ([`SegmAnn::ann_id`], populated from
/// `CocoAnnotation::id` at the dataset boundary). Invalidated
/// wholesale when [`crate::evaluate_boundary_cached`] is invoked at
/// a different `dilation_ratio` than the entries were computed at —
/// ratio is a static configuration knob in practice, so per-call
/// invalidation is the simplest invariant.
///
/// Threadsafe via an internal [`Mutex`] (the kernel needs `Sync`).
/// Single-threaded use is uncontended.
#[derive(Default)]
pub struct BoundaryGtCache {
    inner: Mutex<CacheInner>,
}

#[derive(Default)]
struct CacheInner {
    bands: HashMap<i64, BoundaryGtEntry>,
    /// `None` until the first [`crate::evaluate_boundary_cached`]
    /// call populates entries. Subsequent calls compare and clear on
    /// mismatch.
    ratio: Option<f64>,
}

#[derive(Clone)]
struct BoundaryGtEntry {
    band_area: u64,
    mask_offsets: Vec<u64>,
    band_offsets: Vec<u64>,
}

impl BoundaryGtCache {
    /// Constructs an empty cache. Equivalent to [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of GT annotation bands currently held.
    pub fn len(&self) -> usize {
        self.lock().bands.len()
    }

    /// Returns `true` if no GT bands are currently cached.
    pub fn is_empty(&self) -> bool {
        self.lock().bands.is_empty()
    }

    /// Drops all cached bands. Useful when the GT dataset changes
    /// mid-loop so stale `(ann_id, band)` pairs don't pollute the
    /// next call.
    pub fn clear(&self) {
        let mut inner = self.lock();
        inner.bands.clear();
        inner.ratio = None;
    }

    /// On entry to a cached evaluate, ensure the cached entries
    /// agree with `ratio`. If a different ratio populated the cache
    /// previously, drop those entries — they would yield wrong
    /// boundary bands at the new ratio.
    pub(crate) fn align_ratio(&self, ratio: f64) {
        let mut inner = self.lock();
        if inner.ratio != Some(ratio) {
            inner.bands.clear();
            inner.ratio = Some(ratio);
        }
    }

    fn lock(&self) -> MutexGuard<'_, CacheInner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }
}

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
        let mut scratch = BoundaryComputeScratch::new();
        boundary_iou_compute(self.dilation_ratio, gts, dts, out, &mut scratch, None)
    }
}

/// Scratch-aware boundary-IoU compute. Same semantics as
/// [`BoundaryIou::compute`] but reuses a caller-owned
/// [`BoundaryComputeScratch`] across band derivations + segment-table
/// builds — letting the dataset-wide path (`evaluate_boundary` via the
/// private `BoundaryIouCached` kernel) amortize per-mask + per-cell
/// allocations across the ~36k anns of a val2017 pass.
///
/// When `gt_cache` is `Some`, GT bands are looked up by
/// [`SegmAnn::ann_id`]; misses fall through to a fresh derivation and
/// populate the cache. The cache must already be aligned to
/// `dilation_ratio` (callers go through
/// [`crate::evaluate_boundary_cached`], which calls
/// [`BoundaryGtCache::align_ratio`] once per evaluate).
pub(crate) fn boundary_iou_compute(
    dilation_ratio: f64,
    gts: &[SegmAnn],
    dts: &[SegmAnn],
    out: &mut ArrayViewMut2<'_, f64>,
    scratch: &mut BoundaryComputeScratch,
    gt_cache: Option<&BoundaryGtCache>,
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

    // I1 prefilter: bbox-overlap mask on the tight RLE bboxes. Writes
    // `1.0` where bbox intersection is strictly positive, `0.0`
    // otherwise; the cheaper variant skips the IoU divide because
    // boundary consumes only the survivor-bit (the `<= 0.0` gate
    // below) and unconditionally overwrites passing cells with
    // `min(mask_iou, bound_iou)`. The mask is crowd-agnostic —
    // `inter > 0` iff `bbIoU > 0` for both crowd and non-crowd, so
    // the gate stays correct without the prefilter honoring E1
    // itself.
    scratch.g_bbox.clear();
    scratch
        .g_bbox
        .extend(gts.iter().map(|g| to_bbox_ann(&g.rle, g.is_crowd)));
    scratch.d_bbox.clear();
    scratch
        .d_bbox
        .extend(dts.iter().map(|d| to_bbox_ann(&d.rle, false)));
    BboxIou.compute_overlap_mask(&scratch.g_bbox, &scratch.d_bbox, out)?;

    // O1/O2: skip B(g) for crowd GTs — the boundary fold is
    // suppressed on crowd rows, so computing the band is wasted
    // work proportional to the (often large) crowd-mask area. The
    // mask-side offsets are still needed for `inter_mask` (E1 crowd
    // mask IoU = inter / dt_area), so we push mask segments
    // unconditionally; only band segments / band area get skipped.
    scratch.g_mask_area.clear();
    scratch.g_band_area.clear();
    scratch.g_mask_segments.clear();
    scratch.g_band_segments.clear();
    for g in gts {
        scratch.g_mask_area.push(g.rle.area());
        if g.is_crowd {
            scratch.g_mask_segments.push_from_rle(&g.rle);
            scratch.g_band_area.push(0);
            scratch.g_band_segments.push_segments(&[]);
        } else {
            populate_gt_entry(g, dilation_ratio, scratch, gt_cache)?;
        }
    }
    scratch.d_mask_area.clear();
    scratch.d_band_area.clear();
    scratch.d_mask_segments.clear();
    scratch.d_band_segments.clear();
    for d in dts {
        scratch.d_mask_area.push(d.rle.area());
        scratch.d_mask_segments.push_from_rle(&d.rle);
        let band_area = boundary_band_segments_into(
            &d.rle,
            dilation_ratio,
            &mut scratch.erode,
            &mut scratch.d_band_segments,
        )?;
        scratch.d_band_area.push(band_area);
    }

    for g in 0..gts.len() {
        let crowd = gts[g].is_crowd;
        let g_mask_seg = scratch.g_mask_segments.row(g);
        let g_band_seg = scratch.g_band_segments.row(g);
        for d in 0..dts.len() {
            if out[[g, d]] <= 0.0 {
                continue;
            }
            let inter_mask = intersect_area_offsets(g_mask_seg, scratch.d_mask_segments.row(d));
            let mask_denom = if crowd {
                scratch.d_mask_area[d]
            } else {
                scratch.g_mask_area[g] + scratch.d_mask_area[d] - inter_mask
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

            let inter_bound = intersect_area_offsets(g_band_seg, scratch.d_band_segments.row(d));
            let bound_denom = scratch.g_band_area[g] + scratch.d_band_area[d] - inter_bound;
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

/// Resolve one GT annotation's band area + mask/band fg offsets and
/// append them to `scratch`'s segment tables. With a cache, hit on
/// `ann_id` skips erosion + decode; on miss the entry is computed and
/// inserted before being pushed.
fn populate_gt_entry(
    ann: &SegmAnn,
    ratio: f64,
    scratch: &mut BoundaryComputeScratch,
    cache: Option<&BoundaryGtCache>,
) -> Result<(), EvalError> {
    if let Some(cache) = cache {
        let mut inner = cache.lock();
        if let Some(entry) = inner.bands.get(&ann.ann_id) {
            scratch.g_band_area.push(entry.band_area);
            scratch.g_mask_segments.push_segments(&entry.mask_offsets);
            scratch.g_band_segments.push_segments(&entry.band_offsets);
            return Ok(());
        }
        scratch.g_mask_segments.push_from_rle(&ann.rle);
        let band_area = boundary_band_segments_into(
            &ann.rle,
            ratio,
            &mut scratch.erode,
            &mut scratch.g_band_segments,
        )?;
        scratch.g_band_area.push(band_area);
        let mask_offsets = scratch.g_mask_segments.last_row().to_vec();
        let band_offsets = scratch.g_band_segments.last_row().to_vec();
        inner.bands.insert(
            ann.ann_id,
            BoundaryGtEntry {
                band_area,
                mask_offsets,
                band_offsets,
            },
        );
        return Ok(());
    }
    scratch.g_mask_segments.push_from_rle(&ann.rle);
    let band_area = boundary_band_segments_into(
        &ann.rle,
        ratio,
        &mut scratch.erode,
        &mut scratch.g_band_segments,
    )?;
    scratch.g_band_area.push(band_area);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use vernier_mask::ops::boundary_band;
    use vernier_mask::Rle;

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
