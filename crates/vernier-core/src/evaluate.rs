//! Per-image evaluation orchestrator.
//!
//! The bridge between the dataset layer ([`crate::CocoDataset`] /
//! [`crate::CocoDetections`]) and the IoU-type-agnostic spine
//! ([`crate::match_image`] → [`crate::accumulate`]). Pycocotools fuses
//! these in `evaluate()` (cocoeval.py 174-216); we keep the layers
//! separate so the spine stays untouchable per ADR-0005.
//!
//! The pass is generic over [`EvalKernel`] — a `Similarity` supertrait
//! that adds the dataset-bridging methods that turn a `(image, category)`
//! cell into kernel-typed annotations. Bbox and segm reuse the same
//! orchestrator with [`BboxIou`] and [`SegmIou`] respectively; future
//! kernels (OKS, Boundary IoU) plug in by adding one
//! `impl EvalKernel for FooIou` block — `match_image`, `accumulate`,
//! and `summarize_*` stay untouched.
//!
//! ## What this layer does
//!
//! For each `(image, category)` cell:
//!
//! 1. Gather GTs and DTs from the dataset indices.
//! 2. Pre-filter DTs to the top `max_dets_per_image` by score (the
//!    matching engine and accumulator both rely on this cap; smaller
//!    `max_dets` values are sliced downstream by `accumulate`).
//! 3. Build the kernel's annotation slices via
//!    [`EvalKernel::build_gt_anns`] / [`EvalKernel::build_dt_anns`] and
//!    compute the GT × DT IoU matrix once via [`Similarity::compute`].
//! 4. For each area range, build the per-call `_ignore` vector
//!    (quirk **D3**) from the dataset's base ignore (D1) plus the area
//!    filter (D6/D7), run [`crate::match_image`], apply quirk **B7** by
//!    flipping `dt_ignore` for unmatched DTs whose area is outside the
//!    active range, and pack the result as a [`crate::PerImageEval`] at
//!    `[k][a][i]`.
//!
//! ## Quirk dispositions handled here
//!
//! - **D3** (`aligned`): per-call `_ignore` computed without mutating
//!   the dataset.
//! - **D6/D7** (`strict`): area filter uses non-strict `<=` / `>=` on
//!   both bounds (mirrors `cocoeval.py:251`'s
//!   `g['area'] < aRng[0] or g['area'] > aRng[1]` exclusion). An
//!   annotation whose area equals a bucket boundary lands in *both*
//!   adjacent buckets. Inequality direction matches the eval-time filter
//!   in pycocotools, *not* `getAnnIds(areaRng=...)`.
//! - **B7** (`strict`): unmatched DTs whose area is out of range get
//!   `dt_ignore=true` so they do not contribute to the precision/recall
//!   curve in this area cell.
//! - **AA3** (`strict`, ADR-0026): when the dataset carries LVIS
//!   federated metadata and the current `(image, category)` cell is in
//!   `not_exhaustive_category_ids[image]`, every unmatched DT in the
//!   cell has its `dt_ignore` set to `true`. Mirrors lvis-api
//!   `eval.py:269-278`'s OR into the area-bucket `dt_ig_mask`. The
//!   matching engine is unchanged: the flag piggybacks on the same
//!   `dt_ignore` field B7 already drives.
//! - **AA4** (`strict`, ADR-0026): on a federated dataset and with
//!   `use_cats=true`, a cell `(image I, category C)` is evaluated only
//!   when `C ∈ pos[I] ∪ neg[I]`. Cells with no GT (so `C ∉ pos[I]`)
//!   and no `neg` listing produce no `eval_imgs` entry — the existing
//!   `Option<PerImageEval>` distinction (`None` vs an empty cell) is
//!   the same one lvis-api's `eval.py:336` filter relies on.
//! - **L4** (`aligned`): `use_cats=false` collapses every category onto
//!   a single virtual `k=0` bucket, with `category_id` carried through
//!   matching as a no-op.
//! - **E2 / J4** (`strict`): DTs never carry an `is_crowd` flag — the
//!   [`crate::CocoDetection`] type lacks the field. Only GT crowdness
//!   drives the E1 asymmetry inside the kernel.
//! - **J3** (`strict`): DT areas are read from
//!   [`crate::CocoDetection::area`], which the dataset layer derives
//!   from the bbox at construction.
//! - **J2** (`strict`): under [`ParityMode::Strict`], a DT lacking a
//!   `segmentation` field under `iouType="segm"` has its bbox
//!   synthesized into a 4-point rectangle polygon
//!   `[[x1,y1, x1,y2, x2,y2, x2,y1]]` and rasterized — bit-for-bit the
//!   path `pycocotools/coco.py:341` follows. Under
//!   [`ParityMode::Corrected`] (the default for net-new users) the
//!   synthesis is refused with [`EvalError::InvalidAnnotation`]: silent
//!   coercion of bbox results to rectangle masks is a footgun, and
//!   users who want strict parity opt in.
//! - **J6** (`corrected`): per-entry dispatch — every detection is
//!   inspected independently for the segm/bbox kind. Under
//!   [`ParityMode::Corrected`] heterogeneous DT lists (some entries
//!   with `segmentation`, some without) are rejected up-front rather
//!   than silently routed through the first-entry-decides dispatch
//!   pycocotools follows at `coco.py:330-363`.

use ndarray::{Array2, ArrayView2, ArrayViewMut2};

use crate::accumulate::PerImageEval;
use crate::dataset::{
    Bbox, CategoryId, CocoAnnotation, CocoDataset, CocoDetection, CocoDetections, EvalDataset,
    ImageId, ImageMeta,
};
use crate::error::EvalError;
use crate::matching::{match_image, MatchResult};
use crate::parity::{argsort_score_desc, ParityMode};
use crate::segmentation::Segmentation;
use crate::similarity::{
    boundary_iou_compute, segm_iou_compute, BboxAnn, BboxIou, BoundaryComputeScratch,
    BoundaryGtCache, BoundaryIou, OksAnn, OksSimilarity, SegmAnn, SegmComputeScratch, SegmGtCache,
    SegmIou, Similarity,
};
use std::collections::HashMap;
use std::sync::Mutex;
use vernier_mask::Rle;

/// Sentinel `category_id` emitted on every cell when `use_cats=false`.
/// Mirrors pycocotools' `p.catIds = [-1]` collapse (quirk **L4**).
pub const COLLAPSED_CATEGORY_SENTINEL: i64 = -1;

/// Sentinel upper bound for "unbounded" area buckets, mirroring the
/// `1e10` pycocotools uses for `all` / `large`.
pub const AREA_UNBOUNDED: f64 = 1e10;

/// Closed `[lo, hi]` area bucket — both bounds are inclusive per quirks
/// **D6/D7**, so an annotation with area exactly equal to a bound lands
/// in this bucket (and in the adjacent one when the boundary is shared).
///
/// `index` is the position on the `Accumulated` A-axis the resulting
/// [`PerImageEval`] feeds into; matched at summarize time against
/// [`crate::AreaRng::index`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AreaRange {
    /// A-axis position. `0` is conventionally the `all` bucket, matching
    /// [`crate::AreaRng::ALL`].
    pub index: usize,
    /// Lower bound (inclusive — quirks D6/D7).
    pub lo: f64,
    /// Upper bound (inclusive — quirks D6/D7). Use [`AREA_UNBOUNDED`]
    /// for "no upper bound".
    pub hi: f64,
}

impl AreaRange {
    /// Pycocotools' default detection grid: `all`, `small`, `medium`,
    /// `large`. Indices line up with [`crate::AreaRng`]'s `ALL` /
    /// `SMALL` / `MEDIUM` / `LARGE` constants.
    pub fn coco_default() -> [Self; 4] {
        [
            Self {
                index: 0,
                lo: 0.0,
                hi: AREA_UNBOUNDED,
            },
            Self {
                index: 1,
                lo: 0.0,
                hi: 32.0 * 32.0,
            },
            Self {
                index: 2,
                lo: 32.0 * 32.0,
                hi: 96.0 * 96.0,
            },
            Self {
                index: 3,
                lo: 96.0 * 96.0,
                hi: AREA_UNBOUNDED,
            },
        ]
    }

    /// Keypoints area grid (per ADR-0012, quirk **D5**): `all`, `medium`,
    /// `large` — pycocotools drops the `small` bucket for kp eval. The
    /// A-axis is compressed to 3 entries with indices `0 = all`,
    /// `1 = medium`, `2 = large`. Pair with
    /// [`crate::summarize::StatRequest::coco_keypoints_default`] so the
    /// summarizer's `req.area.index` lookups land on the right slice.
    pub fn keypoints_default() -> [Self; 3] {
        [
            Self {
                index: 0,
                lo: 0.0,
                hi: AREA_UNBOUNDED,
            },
            Self {
                index: 1,
                lo: 32.0 * 32.0,
                hi: 96.0 * 96.0,
            },
            Self {
                index: 2,
                lo: 96.0 * 96.0,
                hi: AREA_UNBOUNDED,
            },
        ]
    }

    fn contains(&self, area: f64) -> bool {
        // D6 (strict): pycocotools (cocoeval.py:251) keeps a GT/DT in a
        // bucket when `not (area < lo or area > hi)`, i.e. non-strict
        // inclusion on both ends. An area equal to a bucket boundary
        // (e.g. 32² = 1024) therefore lands in *both* adjacent buckets.
        area >= self.lo && area <= self.hi
    }
}

/// Inputs to [`evaluate_bbox`] / [`evaluate_segm`] / [`evaluate_boundary`] / [`evaluate_with`].
/// IoU-agnostic — kernel-specific configuration (sigmas, prefilter
/// thresholds, …) lives on the [`EvalKernel`] passed alongside.
#[derive(Debug, Clone, Copy)]
pub struct EvaluateParams<'p> {
    /// IoU thresholds, length `T`. Use [`crate::iou_thresholds`] for the
    /// canonical 10-point COCO ladder.
    pub iou_thresholds: &'p [f64],
    /// Area ranges. The `index` field of each entry is the A-axis
    /// position the resulting [`PerImageEval`] is filed under; the
    /// orchestrator emits exactly `area_ranges.len()` cells per
    /// `(image, category)`.
    pub area_ranges: &'p [AreaRange],
    /// Top-N filter applied to DTs per `(image, category)` cell before
    /// matching. Should be the largest entry of the eventual
    /// [`crate::AccumulateParams::max_dets`] ladder; smaller caps are
    /// sliced downstream.
    pub max_dets_per_image: usize,
    /// Quirk **L4** (`aligned`): when `false`, every category is
    /// collapsed onto a single bucket `k=0` and `category_id` is ignored
    /// for gather purposes.
    pub use_cats: bool,
    /// When `true`, [`evaluate_with`] retains the per-`(category,
    /// image)` IoU matrix on [`EvalGrid::retained_ious`] so the
    /// `per_pair` / `per_detection` result tables can read it. Default
    /// `false`; the no-retention path allocates nothing extra and is
    /// bit-identical to the 0.0.1 release.
    pub retain_iou: bool,
}

/// Owned counterpart to [`EvaluateParams`].
///
/// The streaming evaluator holds its config across many `update()`
/// calls and cannot borrow per-call slices the way the batch entry
/// points do. [`Self::borrow`] reconstructs an [`EvaluateParams`] view
/// that reuses this struct's storage, so handing the owned form to the
/// unchanged `evaluate_with` path is zero-cost.
#[derive(Debug, Clone)]
pub struct OwnedEvaluateParams {
    /// IoU thresholds, length `T`.
    pub iou_thresholds: Vec<f64>,
    /// Area ranges (owned).
    pub area_ranges: Vec<AreaRange>,
    /// Top-N filter applied to DTs per `(image, category)` cell before matching.
    pub max_dets_per_image: usize,
    /// Quirk **L4** collapse flag.
    pub use_cats: bool,
    /// IoU-matrix retention flag — see [`EvaluateParams::retain_iou`].
    pub retain_iou: bool,
}

impl OwnedEvaluateParams {
    /// Borrowed view. Reuses `self`'s storage; no allocation.
    pub fn borrow(&self) -> EvaluateParams<'_> {
        EvaluateParams {
            iou_thresholds: &self.iou_thresholds,
            area_ranges: &self.area_ranges,
            max_dets_per_image: self.max_dets_per_image,
            use_cats: self.use_cats,
            retain_iou: self.retain_iou,
        }
    }
}

/// Bridges a [`CocoDataset`] / [`CocoDetections`] cell to a kernel's
/// annotation type.
///
/// Per ADR-0005, the per-image pass is generic over this trait so a new
/// IoU type plugs in via one `impl EvalKernel for FooIou` block — the
/// matching engine, accumulator, and summarizer never see the new type.
///
/// Implementors do the per-cell rasterization / lookup that a [`Similarity`]
/// kernel can't (because [`Similarity`] is dataset-agnostic by design).
/// `image` carries the `(h, w)` segm impls need for [`crate::Segmentation::to_rle`].
pub trait EvalKernel: Similarity {
    /// Build the kernel's GT annotation slice for one `(image, category)`
    /// cell. `indices` selects from `gt_anns` in the order the cell
    /// matcher will see.
    fn build_gt_anns(
        &self,
        gt_anns: &[CocoAnnotation],
        indices: &[usize],
        image: &ImageMeta,
    ) -> Result<Vec<Self::Annotation>, EvalError>;

    /// Build the kernel's DT annotation slice for one `(image, category)`
    /// cell, in score-descending sorted order matching `dt_indices`.
    ///
    /// `parity_mode` is threaded through so kernels with parity-aware
    /// fallbacks (segm's J2 bbox→polygon synthesis under
    /// [`ParityMode::Strict`]) can dispatch on it without reaching back
    /// up the call stack.
    fn build_dt_anns(
        &self,
        dt_anns: &[CocoDetection],
        indices: &[usize],
        image: &ImageMeta,
        parity_mode: ParityMode,
    ) -> Result<Vec<Self::Annotation>, EvalError>;

    /// Optional kernel-specific GT ignore override. Default `false` (no
    /// kernel reason to ignore).
    ///
    /// The orchestrator OR-s the result with the dataset-level
    /// [`CocoAnnotation::effective_ignore`] (quirk **D1**) when building
    /// `gt_base_ignore`. [`OksSimilarity`] overrides this to fold in
    /// quirk **D2** (`strict`): GT with zero visible keypoints is
    /// treated as an implicit ignore region, OR-ed with the existing
    /// ignore. Bbox / segm / boundary kernels keep the default — D2 is
    /// keypoints-specific and must not bleed across kernels.
    fn extra_gt_ignore(&self, _ann: &CocoAnnotation) -> bool {
        false
    }

    /// Marker: is this kernel the keypoints (OKS) kernel?
    ///
    /// The streaming evaluator dispatches its summarizer choice on this
    /// flag: keypoints kernels resolve to the 10-stat
    /// [`crate::StatRequest::coco_keypoints_default`] plan, every other
    /// kernel resolves to the 12-stat detection plan. Default `false`;
    /// [`OksSimilarity`] overrides to `true`. Additive trait method —
    /// existing implementors keep the default.
    fn is_keypoints(&self) -> bool {
        false
    }
}

impl EvalKernel for BboxIou {
    fn build_gt_anns(
        &self,
        gt_anns: &[CocoAnnotation],
        indices: &[usize],
        _image: &ImageMeta,
    ) -> Result<Vec<BboxAnn>, EvalError> {
        Ok(indices
            .iter()
            .map(|&j| BboxAnn {
                bbox: gt_anns[j].bbox,
                is_crowd: gt_anns[j].is_crowd,
            })
            .collect())
    }

    fn build_dt_anns(
        &self,
        dt_anns: &[CocoDetection],
        indices: &[usize],
        _image: &ImageMeta,
        _parity_mode: ParityMode,
    ) -> Result<Vec<BboxAnn>, EvalError> {
        // E2/J4: DT never carries crowd.
        Ok(indices
            .iter()
            .map(|&j| BboxAnn {
                bbox: dt_anns[j].bbox,
                is_crowd: false,
            })
            .collect())
    }
}

impl EvalKernel for SegmIou {
    fn build_gt_anns(
        &self,
        gt_anns: &[CocoAnnotation],
        indices: &[usize],
        image: &ImageMeta,
    ) -> Result<Vec<SegmAnn>, EvalError> {
        build_segm_gt_anns(gt_anns, indices, image)
    }

    fn build_dt_anns(
        &self,
        dt_anns: &[CocoDetection],
        indices: &[usize],
        image: &ImageMeta,
        parity_mode: ParityMode,
    ) -> Result<Vec<SegmAnn>, EvalError> {
        build_segm_dt_anns(dt_anns, indices, image, parity_mode)
    }
}

impl EvalKernel for BoundaryIou {
    fn build_gt_anns(
        &self,
        gt_anns: &[CocoAnnotation],
        indices: &[usize],
        image: &ImageMeta,
    ) -> Result<Vec<SegmAnn>, EvalError> {
        build_segm_gt_anns(gt_anns, indices, image)
    }

    fn build_dt_anns(
        &self,
        dt_anns: &[CocoDetection],
        indices: &[usize],
        image: &ImageMeta,
        parity_mode: ParityMode,
    ) -> Result<Vec<SegmAnn>, EvalError> {
        build_segm_dt_anns(dt_anns, indices, image, parity_mode)
    }
}

impl EvalKernel for OksSimilarity {
    fn build_gt_anns(
        &self,
        gt_anns: &[CocoAnnotation],
        indices: &[usize],
        _image: &ImageMeta,
    ) -> Result<Vec<OksAnn>, EvalError> {
        indices
            .iter()
            .map(|&j| {
                let ann = &gt_anns[j];
                let kps = ann
                    .keypoints
                    .as_deref()
                    .ok_or_else(|| missing_keypoints_err("GT", ann.id.0, ann.image_id.0))?;
                let num_keypoints = ann
                    .num_keypoints
                    .unwrap_or_else(|| count_visible_keypoints(kps));
                Ok(OksAnn {
                    category_id: ann.category_id.0,
                    keypoints: kps.to_vec(),
                    num_keypoints,
                    bbox: ann.bbox.into(),
                    area: ann.area,
                })
            })
            .collect()
    }

    fn build_dt_anns(
        &self,
        dt_anns: &[CocoDetection],
        indices: &[usize],
        _image: &ImageMeta,
        _parity_mode: ParityMode,
    ) -> Result<Vec<OksAnn>, EvalError> {
        // E2/J4: DT never carries crowd. There is no parity-mode J2
        // analog for keypoints — pycocotools has no bbox→keypoint
        // synthesis path, so a missing `keypoints` field is always an
        // [`EvalError::InvalidAnnotation`] regardless of mode.
        indices
            .iter()
            .map(|&j| {
                let dt = &dt_anns[j];
                let kps = dt
                    .keypoints
                    .as_deref()
                    .ok_or_else(|| missing_keypoints_err("DT", dt.id.0, dt.image_id.0))?;
                let num_keypoints = dt
                    .num_keypoints
                    .unwrap_or_else(|| count_visible_keypoints(kps));
                Ok(OksAnn {
                    category_id: dt.category_id.0,
                    keypoints: kps.to_vec(),
                    num_keypoints,
                    bbox: dt.bbox.into(),
                    area: dt.area,
                })
            })
            .collect()
    }

    fn extra_gt_ignore(&self, ann: &CocoAnnotation) -> bool {
        // D2 (`strict`): GT with zero visible keypoints is an implicit
        // ignore region. Annotations without a `keypoints` field at all
        // are treated as zero-visible — `build_gt_anns` will reject
        // them downstream, but this hook runs before that and must
        // stay total.
        let visible = ann
            .num_keypoints
            .or_else(|| ann.keypoints.as_deref().map(count_visible_keypoints))
            .unwrap_or(0);
        visible == 0
    }

    fn is_keypoints(&self) -> bool {
        true
    }
}

/// Count of *visible* keypoints (`v > 0`) in a flat
/// `[x, y, v, ...]` triplet vector. Used as the fallback for
/// pycocotools-precomputed `num_keypoints` on inputs that omit it.
fn count_visible_keypoints(kps: &[f64]) -> u32 {
    kps.chunks_exact(3).filter(|t| t[2] > 0.0).count() as u32
}

/// OKS path equivalent of [`missing_segmentation_err`] — names the
/// offending kind/id/image when a `keypoints` field is required and
/// absent. Unlike segm there is no parity-mode escape hatch.
fn missing_keypoints_err(kind: &str, ann_id: i64, image_id: i64) -> EvalError {
    EvalError::InvalidAnnotation {
        detail: format!(
            "{kind} id={ann_id} on image {image_id} has no `keypoints` field; \
             OKS eval requires keypoints on every entry. There is no \
             pycocotools-equivalent bbox-synthesis fallback for keypoints \
             (unlike segm quirk J2)."
        ),
    }
}

fn build_segm_gt_anns(
    gt_anns: &[CocoAnnotation],
    indices: &[usize],
    image: &ImageMeta,
) -> Result<Vec<SegmAnn>, EvalError> {
    indices
        .iter()
        .map(|&j| {
            let ann = &gt_anns[j];
            let seg = ann
                .segmentation
                .as_ref()
                .ok_or_else(|| missing_segmentation_err("GT", ann.id.0, image.id.0))?;
            Ok(SegmAnn {
                rle: seg.to_rle(image.height, image.width)?,
                is_crowd: ann.is_crowd,
                ann_id: ann.id.0,
            })
        })
        .collect()
}

fn build_segm_dt_anns(
    dt_anns: &[CocoDetection],
    indices: &[usize],
    image: &ImageMeta,
    parity_mode: ParityMode,
) -> Result<Vec<SegmAnn>, EvalError> {
    indices
        .iter()
        .map(|&j| {
            let dt = &dt_anns[j];
            let rle = match (&dt.segmentation, parity_mode) {
                (Some(seg), _) => seg.to_rle(image.height, image.width)?,
                // J2 (`strict`): pycocotools' coco.py:341 synthesizes
                // a rectangular polygon `[[x1,y1, x1,y2, x2,y2, x2,y1]]`
                // from the bbox when a DT under iouType="segm" lacks
                // a `segmentation` field. We reproduce that path
                // bit-for-bit so strict-mode parity covers bbox-only
                // result files.
                (None, ParityMode::Strict) => {
                    synthesize_dt_segm_from_bbox(&dt.bbox, image.height, image.width)?
                }
                // J2 (`corrected`) + J6 (`corrected`): silent coercion
                // of bbox results to rectangle masks is a footgun.
                // Refusing here also turns a heterogeneous DT list
                // (some entries with segm, some without) under
                // iouType="segm" into a clean, per-entry-pinpointed
                // error rather than the first-entry-decides dispatch
                // pycocotools follows.
                (None, ParityMode::Corrected) => {
                    return Err(missing_segmentation_err("DT", dt.id.0, image.id.0));
                }
            };
            Ok(SegmAnn {
                rle,
                is_crowd: false,
                ann_id: dt.id.0,
            })
        })
        .collect()
}

/// J2 (`strict`): synthesize a 4-point rectangle polygon from a DT bbox
/// and rasterize it at the image's `(h, w)`. Mirrors
/// `pycocotools/coco.py:341` exactly:
/// `[[x1,y1, x1,y2, x2,y2, x2,y1]]` where `(x1, y1)` is the top-left and
/// `(x2, y2) = (x1 + w, y1 + h)`.
fn synthesize_dt_segm_from_bbox(bbox: &Bbox, h: u32, w: u32) -> Result<Rle, EvalError> {
    let x1 = bbox.x;
    let y1 = bbox.y;
    let x2 = bbox.x + bbox.w;
    let y2 = bbox.y + bbox.h;
    let polygon = vec![x1, y1, x1, y2, x2, y2, x2, y1];
    let segm = Segmentation::Polygons(vec![polygon]);
    segm.to_rle(h, w)
}

/// J2 (`corrected`) / J6 (`corrected`) error path: a DT lacks the
/// `segmentation` field under `iouType="segm"`. The detail names the
/// offending kind (`GT` or `DT`), id, and image so a heterogeneous
/// DT list pinpoints the first entry without segm rather than failing
/// with a global "wrong shape" error.
fn missing_segmentation_err(kind: &str, ann_id: i64, image_id: i64) -> EvalError {
    EvalError::InvalidAnnotation {
        detail: format!(
            "{kind} id={ann_id} on image {image_id} has no `segmentation` field; \
             segm eval in corrected mode requires one on every entry. \
             pycocotools synthesizes a bbox-rectangle polygon here \
             (quirks J2/J6); pass `ParityMode::Strict` to opt into that \
             behavior."
        ),
    }
}

/// Pycocotools-shaped per-cell bookkeeping that the matching engine
/// strips out when packing [`PerImageEval`]. Surfaced separately so the
/// accumulator stays narrow per ADR-0005, and FFI / `COCOeval` drop-in
/// consumers can reconstruct `evalImgs` dicts without re-running eval.
///
/// All `dt_*` axes are in score-descending sorted order (stable
/// mergesort, quirk **A1**); all `gt_*` axes are in ignore-ascending
/// sorted order (quirk **A4**). `dt_matches` and `gt_matches` carry
/// pycocotools' value semantics: `i64` annotation ids on a hit, `0` on a
/// miss (matching `dtm`/`gtm` initialization in `cocoeval.py`).
#[derive(Debug, Clone)]
pub struct EvalImageMeta {
    /// COCO image id for this cell.
    pub image_id: i64,
    /// COCO category id, or [`COLLAPSED_CATEGORY_SENTINEL`] when
    /// `use_cats=false`.
    pub category_id: i64,
    /// Active area range as `[lo, hi]`, mirroring pycocotools' `aRng`.
    pub area_rng: [f64; 2],
    /// `max_dets_per_image` cap that produced this cell's DT slice.
    pub max_det: usize,
    /// DT annotation ids in sorted-DT order, length `D`.
    pub dt_ids: Vec<i64>,
    /// GT annotation ids in sorted-GT order, length `G`.
    pub gt_ids: Vec<i64>,
    /// Shape `(T, D)`. GT id matched at `(threshold, sorted-DT k)`, or
    /// `0` if unmatched (pycocotools sentinel; safe because COCO ids are
    /// `>= 1` per spec, and vernier's auto-id assignment also starts at 1).
    pub dt_matches: Array2<i64>,
    /// Shape `(T, G)`. DT id matched at `(threshold, sorted-GT k)`, or
    /// `0` if unmatched (same `>= 1` invariant as `dt_matches`).
    pub gt_matches: Array2<i64>,
}

/// Output of [`evaluate_bbox`] / [`evaluate_segm`] / [`evaluate_boundary`]
/// — the flat `(K, A, I)` grid of
/// [`PerImageEval`] cells the accumulator consumes, plus the dimensions
/// needed to construct [`crate::AccumulateParams`].
#[derive(Debug, Clone)]
pub struct EvalGrid {
    /// `Some(cell)` per `(k, a, i)` triple where the cell ran; `None`
    /// where pycocotools would emit `None` (image absent from
    /// detections, no GTs and no DTs in the cell). Layout is K-major,
    /// then A, then I — `eval_imgs[k * A * I + a * I + i]`.
    pub eval_imgs: Vec<Option<PerImageEval>>,
    /// Pycocotools-shaped bookkeeping for each populated cell (same
    /// `[k][a][i]` layout as `eval_imgs`; `None` wherever `eval_imgs` is
    /// `None`).
    pub eval_imgs_meta: Vec<Option<EvalImageMeta>>,
    /// `K` axis size: the number of categories used for evaluation, or
    /// `1` when `use_cats=false`.
    pub n_categories: usize,
    /// `A` axis size: equal to `params.area_ranges.len()`.
    pub n_area_ranges: usize,
    /// `I` axis size: number of images iterated over (every image in the
    /// GT dataset, in deterministic id-ascending order).
    pub n_images: usize,
    /// Per-`(category, image)` IoU matrices retained when the caller
    /// passed [`EvaluateParams::retain_iou`] = `true`. `None` on the
    /// default no-retention path; one discriminant byte wide there.
    pub retained_ious: Option<crate::tables::RetainedIous>,
}

impl EvalGrid {
    /// Cell at `(category_index, area_index, image_index)`. Returns
    /// `None` when the indices are in bounds but no cell ran (image
    /// absent from detections, or no GTs and no DTs in the cell);
    /// returns `None` for out-of-bounds indices as well.
    pub fn cell(&self, k: usize, a: usize, i: usize) -> Option<&PerImageEval> {
        let idx = self.flat_index(k, a, i)?;
        self.eval_imgs.get(idx).and_then(Option::as_ref)
    }

    /// Pycocotools-shaped bookkeeping at `(category_index, area_index,
    /// image_index)`. `None` exactly when [`EvalGrid::cell`] is `None`.
    pub fn cell_meta(&self, k: usize, a: usize, i: usize) -> Option<&EvalImageMeta> {
        let idx = self.flat_index(k, a, i)?;
        self.eval_imgs_meta.get(idx).and_then(Option::as_ref)
    }

    fn flat_index(&self, k: usize, a: usize, i: usize) -> Option<usize> {
        if k >= self.n_categories || a >= self.n_area_ranges || i >= self.n_images {
            return None;
        }
        Some(k * self.n_area_ranges * self.n_images + a * self.n_images + i)
    }
}

/// Run the per-image evaluation pass with the given [`EvalKernel`].
///
/// Iterates `(image, category)` cells, computes the IoU matrix once per
/// cell via the kernel, runs [`crate::match_image`] once per area range,
/// and packs the results into a flat `[k][a][i]`-ordered grid suitable
/// for [`crate::accumulate`].
///
/// Most callers want [`evaluate_bbox`], [`evaluate_segm`], or
/// [`evaluate_boundary`]; this entry point is exposed for downstream
/// code that ships its own kernel.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying [`Similarity`],
/// [`EvalKernel::build_gt_anns`] / [`EvalKernel::build_dt_anns`], and
/// [`crate::match_image`] calls.
pub fn evaluate_with<K: EvalKernel>(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: EvaluateParams<'_>,
    parity_mode: ParityMode,
    kernel: &K,
) -> Result<EvalGrid, EvalError> {
    // Image and category ordering: id-ascending, deterministic across runs.
    let mut images: Vec<&ImageMeta> = gt.images().iter().collect();
    images.sort_unstable_by_key(|im| im.id.0);
    let n_i = images.len();
    let n_a = params.area_ranges.len();

    // L4: collapse to a single virtual bucket when `use_cats=false`.
    let category_buckets: Vec<Option<CategoryId>> = if params.use_cats {
        let mut cats: Vec<_> = gt.categories().iter().map(|c| c.id).collect();
        cats.sort_unstable_by_key(|id| id.0);
        cats.into_iter().map(Some).collect()
    } else {
        vec![None]
    };
    let n_k = category_buckets.len();

    let mut eval_imgs: Vec<Option<PerImageEval>> = vec![None; n_k * n_a * n_i];
    let mut eval_imgs_meta: Vec<Option<EvalImageMeta>> = vec![None; n_k * n_a * n_i];
    // Optional IoU retention, keyed by `(k, i)` — IoU is geometry-only,
    // so storing per-area would duplicate ~4× under the COCO grid.
    let mut retained_ious_map: Option<std::collections::HashMap<(usize, usize), Array2<f64>>> =
        if params.retain_iou {
            Some(std::collections::HashMap::new())
        } else {
            None
        };

    // ADR-0026 federated metadata is consumed only when `use_cats=true`;
    // LVIS evaluation is per-category by construction. With
    // `use_cats=false` the federated maps are intentionally ignored
    // and the eval falls back to COCO semantics (the L4 `k=0` collapse
    // never carries federated state).
    let (federated_neg, federated_nel) = if params.use_cats && gt.is_federated() {
        (gt.neg_category_ids(), gt.not_exhaustive_category_ids())
    } else {
        (None, None)
    };

    for (k, cat) in category_buckets.iter().enumerate() {
        let nk = k * n_a * n_i;
        let category_id = cat.map_or(COLLAPSED_CATEGORY_SENTINEL, |c| c.0);
        for (i, image) in images.iter().enumerate() {
            let image_id = image.id;
            let gt_indices = gt_indices_for_cell(gt, image_id, *cat);
            let dt_indices = dt_top_indices_for_cell(dt, image_id, *cat, params.max_dets_per_image);
            if gt_indices.is_empty() && dt_indices.is_empty() {
                continue;
            }

            // AA4 cell-skip and AA3 not_exhaustive flag. `cat.is_some()`
            // is implied by `federated_neg.is_some()` (we filtered
            // on `use_cats=true` above) but we re-check defensively so
            // the unwrap on `cat` is local and obvious.
            let mut not_exhaustive_for_cell = false;
            if let (Some(c), Some(neg_map), Some(nel_map)) =
                (cat.as_ref(), federated_neg, federated_nel)
            {
                let in_neg = neg_map.get(&image_id).is_some_and(|s| s.contains(c));
                // pos[I] is derived from GTs at load: `C ∈ pos[I]`
                // exactly when `gt_indices` is non-empty for this
                // cell. Skip when the cell is outside `pos ∪ neg`.
                if gt_indices.is_empty() && !in_neg {
                    continue;
                }
                not_exhaustive_for_cell = nel_map.get(&image_id).is_some_and(|s| s.contains(c));
            }

            // Area-invariant per-cell buffers — built once, reused
            // across every area range.
            let gt_anns = gt.annotations();
            let dt_anns = dt.detections();
            let gt_areas: Vec<f64> = gt_indices.iter().map(|&j| gt_anns[j].area).collect();
            let gt_iscrowd: Vec<bool> = gt_indices.iter().map(|&j| gt_anns[j].is_crowd).collect();
            // D1: parity-mode fork lives on the annotation; pass through.
            // Kernel-specific ignore reasons (OKS quirk **D2**) are
            // OR-ed in via [`EvalKernel::extra_gt_ignore`].
            let gt_base_ignore: Vec<bool> = gt_indices
                .iter()
                .map(|&j| {
                    gt_anns[j].effective_ignore(parity_mode) || kernel.extra_gt_ignore(&gt_anns[j])
                })
                .collect();
            let gt_ids: Vec<i64> = gt_indices.iter().map(|&j| gt_anns[j].id.0).collect();
            let dt_areas: Vec<f64> = dt_indices.iter().map(|&j| dt_anns[j].area).collect();
            let dt_scores: Vec<f64> = dt_indices.iter().map(|&j| dt_anns[j].score).collect();
            let dt_ids: Vec<i64> = dt_indices.iter().map(|&j| dt_anns[j].id.0).collect();

            let gt_kernel = kernel.build_gt_anns(gt_anns, gt_indices, image)?;
            let dt_kernel = kernel.build_dt_anns(dt_anns, &dt_indices, image, parity_mode)?;

            let mut iou = Array2::<f64>::zeros((gt_kernel.len(), dt_kernel.len()));
            if !gt_kernel.is_empty() && !dt_kernel.is_empty() {
                kernel.compute(&gt_kernel, &dt_kernel, &mut iou.view_mut())?;
            }

            let buffers = CellBuffers {
                image_id: image_id.0,
                category_id,
                max_det: params.max_dets_per_image,
                gt_areas: &gt_areas,
                gt_iscrowd: &gt_iscrowd,
                gt_base_ignore: &gt_base_ignore,
                gt_ids: &gt_ids,
                dt_areas: &dt_areas,
                dt_scores: &dt_scores,
                dt_ids: &dt_ids,
                iou: iou.view(),
                not_exhaustive: not_exhaustive_for_cell,
            };
            for (a, area) in params.area_ranges.iter().enumerate() {
                let (cell, meta) =
                    evaluate_cell(&buffers, area, params.iou_thresholds, parity_mode)?;
                let flat = nk + a * n_i + i;
                eval_imgs[flat] = Some(cell);
                eval_imgs_meta[flat] = Some(meta);
            }

            // Retain a clone of the IoU matrix exactly when the caller
            // asked. The check is at end-of-cell so the area-range
            // loop above runs on the borrow `buffers.iou`; cloning
            // here costs O(G*D) f64s, only when retention is active.
            if let Some(map) = retained_ious_map.as_mut() {
                map.insert((k, i), iou);
            }
        }
    }

    Ok(EvalGrid {
        eval_imgs,
        eval_imgs_meta,
        n_categories: n_k,
        n_area_ranges: n_a,
        n_images: n_i,
        retained_ious: retained_ious_map.map(crate::tables::RetainedIous::from_map),
    })
}

/// Run the per-image evaluation pass *and* the cross-class IoU side
/// pass (per ADR-0023) in a single call.
///
/// Returns the standard [`EvalGrid`] alongside a
/// [`crate::tables::CrossClassIous`] populated by walking each image's
/// un-class-filtered GT and DT lists through the same kernel
/// [`evaluate_with`] uses internally. Future TIDE callers consume both
/// outputs from one call so they do not pay the matching cost twice.
///
/// The matching engine is unchanged — the side pass is a separate
/// kernel pass at the orchestrator level, preserving the ADR-0005
/// invariant that matching is generic over the IoU matrix only. The
/// side pass shares `params.max_dets_per_image` with the matching path
/// so the DT row indexing across the two passes is consistent.
///
/// # Errors
///
/// Propagates [`EvalError`] from either pass.
pub fn evaluate_with_retention<K: EvalKernel>(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: EvaluateParams<'_>,
    parity_mode: ParityMode,
    kernel: &K,
) -> Result<(EvalGrid, crate::tables::CrossClassIous), EvalError> {
    let grid = evaluate_with(gt, dt, params, parity_mode, kernel)?;
    let cross_class = crate::tide::compute_cross_class_ious(
        gt,
        dt,
        kernel,
        parity_mode,
        params.max_dets_per_image,
    )?;
    Ok((grid, cross_class))
}

/// Run the per-image bbox evaluation pass. Thin wrapper over
/// [`evaluate_with`] with the [`BboxIou`] kernel.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying kernel and matching
/// calls.
pub fn evaluate_bbox(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: EvaluateParams<'_>,
    parity_mode: ParityMode,
) -> Result<EvalGrid, EvalError> {
    evaluate_with(gt, dt, params, parity_mode, &BboxIou)
}

/// Run the per-image segmentation-mask evaluation pass. Thin wrapper
/// over [`evaluate_with`] with the [`SegmIou`] kernel.
///
/// GTs must carry a `segmentation` field. DT handling is parity-mode
/// aware (quirks **J2** / **J6**):
///
/// - [`ParityMode::Strict`] reproduces `pycocotools/coco.py:341` —
///   DTs missing a `segmentation` field have a 4-point rectangle
///   polygon synthesized from their bbox and rasterized.
/// - [`ParityMode::Corrected`] (the default for net-new users) raises
///   [`EvalError::InvalidAnnotation`] instead, which also rejects
///   heterogeneous DT lists (some entries with segm, some without)
///   per-entry rather than via pycocotools' first-entry-decides
///   dispatch.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying kernel and matching
/// calls.
pub fn evaluate_segm(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: EvaluateParams<'_>,
    parity_mode: ParityMode,
) -> Result<EvalGrid, EvalError> {
    evaluate_with(gt, dt, params, parity_mode, &segm_kernel(None))
}

/// Cached variant of [`evaluate_segm`]: reuses GT bbox + area across
/// calls via a caller-owned [`SegmGtCache`].
///
/// Use this when the same GT dataset is evaluated repeatedly against
/// changing detections — e.g. validation passes inside a training
/// loop. The first call populates the cache; each subsequent call
/// skips the `Rle::bbox` and `Rle::area` walks on the GT side. DT-side
/// derivations are always fresh (predictions change per call).
///
/// The cache is keyed by GT [`crate::dataset::CocoAnnotation::id`].
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying kernel and matching
/// calls.
pub fn evaluate_segm_cached(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: EvaluateParams<'_>,
    parity_mode: ParityMode,
    cache: &SegmGtCache,
) -> Result<EvalGrid, EvalError> {
    evaluate_with(gt, dt, params, parity_mode, &segm_kernel(Some(cache)))
}

fn segm_kernel(gt_cache: Option<&SegmGtCache>) -> SegmIouCached<'_> {
    SegmIouCached {
        scratch: Mutex::new(SegmComputeScratch::new()),
        gt_cache,
    }
}

/// Internal kernel used by [`evaluate_segm`] and
/// [`evaluate_segm_cached`]: same semantics as [`SegmIou`] but threads
/// a single [`SegmComputeScratch`] across every `compute` call (so the
/// dataset-wide pass amortizes per-cell `Vec` allocations across the
/// ~36 k anns of a val2017 pass) and optionally consults a
/// [`SegmGtCache`] for cross-call GT bbox+area reuse. Held by
/// [`Mutex`] to satisfy `Similarity: Send + Sync`; the lock is
/// uncontended in single-threaded use.
struct SegmIouCached<'a> {
    scratch: Mutex<SegmComputeScratch>,
    gt_cache: Option<&'a SegmGtCache>,
}

impl Similarity for SegmIouCached<'_> {
    type Annotation = SegmAnn;

    fn compute(
        &self,
        gts: &[SegmAnn],
        dts: &[SegmAnn],
        out: &mut ArrayViewMut2<'_, f64>,
    ) -> Result<(), EvalError> {
        let mut scratch = self
            .scratch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        segm_iou_compute(gts, dts, out, &mut scratch, self.gt_cache)
    }
}

impl EvalKernel for SegmIouCached<'_> {
    fn build_gt_anns(
        &self,
        gt_anns: &[CocoAnnotation],
        indices: &[usize],
        image: &ImageMeta,
    ) -> Result<Vec<SegmAnn>, EvalError> {
        build_segm_gt_anns(gt_anns, indices, image)
    }

    fn build_dt_anns(
        &self,
        dt_anns: &[CocoDetection],
        indices: &[usize],
        image: &ImageMeta,
        parity_mode: ParityMode,
    ) -> Result<Vec<SegmAnn>, EvalError> {
        build_segm_dt_anns(dt_anns, indices, image, parity_mode)
    }
}

/// Run the per-image boundary-IoU evaluation pass (ADR-0010). Thin
/// wrapper over [`evaluate_with`] with the [`BoundaryIou`] kernel.
///
/// `dilation_ratio` controls the boundary band width per ADR-0010 §A2:
/// `0.02` is the COCO default and `0.008` is the LVIS variant.
///
/// GT/DT segmentation handling is identical to [`evaluate_segm`] — same
/// J2/J6 parity-mode dispatch on missing DT segmentations, same
/// "missing GT segmentation" error.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying kernel and matching
/// calls.
pub fn evaluate_boundary(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: EvaluateParams<'_>,
    parity_mode: ParityMode,
    dilation_ratio: f64,
) -> Result<EvalGrid, EvalError> {
    evaluate_with(gt, dt, params, parity_mode, &kernel(dilation_ratio, None))
}

/// Cached variant of [`evaluate_boundary`]: reuses GT bands across
/// calls via a caller-owned [`BoundaryGtCache`].
///
/// Use this when the same GT dataset is evaluated repeatedly against
/// changing detections — e.g. validation passes inside a training
/// loop. The first call populates the cache; each subsequent call
/// skips GT band derivation. DT bands are always derived fresh
/// (predictions change per call).
///
/// The cache is keyed by GT [`crate::dataset::CocoAnnotation::id`].
/// If `dilation_ratio` differs from the previous call's, the cache
/// is cleared and re-populated — the bands depend on the ratio.
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying kernel and matching
/// calls.
pub fn evaluate_boundary_cached(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: EvaluateParams<'_>,
    parity_mode: ParityMode,
    dilation_ratio: f64,
    cache: &BoundaryGtCache,
) -> Result<EvalGrid, EvalError> {
    cache.align_ratio(dilation_ratio);
    evaluate_with(
        gt,
        dt,
        params,
        parity_mode,
        &kernel(dilation_ratio, Some(cache)),
    )
}

fn kernel(dilation_ratio: f64, gt_cache: Option<&BoundaryGtCache>) -> BoundaryIouCached<'_> {
    BoundaryIouCached {
        dilation_ratio,
        scratch: Mutex::new(BoundaryComputeScratch::new()),
        gt_cache,
    }
}

/// Internal kernel used by [`evaluate_boundary`] and
/// [`evaluate_boundary_cached`]: same semantics as [`BoundaryIou`] but
/// threads a single [`BoundaryComputeScratch`] across every `compute`
/// call (so the dataset-wide pass amortizes per-mask + per-cell
/// allocations) and optionally consults a [`BoundaryGtCache`] for
/// cross-call GT band reuse. Held by [`Mutex`] to satisfy
/// `Similarity: Send + Sync`; the lock is uncontended in
/// single-threaded use.
struct BoundaryIouCached<'a> {
    dilation_ratio: f64,
    scratch: Mutex<BoundaryComputeScratch>,
    gt_cache: Option<&'a BoundaryGtCache>,
}

impl Similarity for BoundaryIouCached<'_> {
    type Annotation = SegmAnn;

    fn compute(
        &self,
        gts: &[SegmAnn],
        dts: &[SegmAnn],
        out: &mut ArrayViewMut2<'_, f64>,
    ) -> Result<(), EvalError> {
        let mut scratch = self
            .scratch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        boundary_iou_compute(
            self.dilation_ratio,
            gts,
            dts,
            out,
            &mut scratch,
            self.gt_cache,
        )
    }
}

impl EvalKernel for BoundaryIouCached<'_> {
    fn build_gt_anns(
        &self,
        gt_anns: &[CocoAnnotation],
        indices: &[usize],
        image: &ImageMeta,
    ) -> Result<Vec<SegmAnn>, EvalError> {
        build_segm_gt_anns(gt_anns, indices, image)
    }

    fn build_dt_anns(
        &self,
        dt_anns: &[CocoDetection],
        indices: &[usize],
        image: &ImageMeta,
        parity_mode: ParityMode,
    ) -> Result<Vec<SegmAnn>, EvalError> {
        build_segm_dt_anns(dt_anns, indices, image, parity_mode)
    }
}

/// Run the per-image OKS (`iouType="keypoints"`) evaluation pass per
/// ADR-0012. Thin wrapper over [`evaluate_with`] with the
/// [`OksSimilarity`] kernel.
///
/// `sigmas` is the per-category sigma override map consumed by
/// [`OksSimilarity::new`]: an empty map means "use
/// [`crate::COCO_PERSON_SIGMAS`] for every category" (quirk **F1**,
/// `corrected`). Sigma resolution rules — including the COCO-person
/// default and the 17-keypoint length contract — are documented on
/// [`OksSimilarity`].
///
/// ## Caller responsibilities
///
/// - **Area ranges (quirk D5).** The keypoints-canonical 3-entry grid
///   (`all`, `medium`, `large` — pycocotools omits `small`) lives on the
///   caller side; pass it through `params.area_ranges`. Reusing the
///   detection-canonical 4-entry grid silently introduces an empty
///   `small` bucket that diverges from the parity oracle.
/// - `params.use_cats=true` is the standard configuration for
///   keypoints; per-category sigmas resolve via [`OksSimilarity`]
///   regardless.
///
/// ## Quirks honored here
///
/// - **D2** (`strict`): GT with zero visible keypoints is treated as an
///   implicit ignore region, OR-ed with the dataset-level ignore
///   ([`CocoAnnotation::effective_ignore`]) via
///   [`EvalKernel::extra_gt_ignore`].
/// - **F1**/**F2**/**F3**/**F4**/**F5**: inherited from
///   [`OksSimilarity::compute`].
///
/// GTs and DTs must carry a `keypoints` field; absence raises
/// [`EvalError::InvalidAnnotation`]. There is no
/// parity-mode-conditional bbox synthesis fallback for keypoints (no
/// J2 analog).
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying kernel and matching
/// calls.
pub fn evaluate_keypoints(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: EvaluateParams<'_>,
    parity_mode: ParityMode,
    sigmas: HashMap<i64, Vec<f64>>,
) -> Result<EvalGrid, EvalError> {
    evaluate_with(gt, dt, params, parity_mode, &OksSimilarity::new(sigmas))
}

fn gt_indices_for_cell(gt: &CocoDataset, image: ImageId, cat: Option<CategoryId>) -> &[usize] {
    match cat {
        Some(c) => gt.ann_indices_for(image, c),
        None => gt.ann_indices_for_image(image),
    }
}

pub(crate) fn dt_top_indices_for_cell(
    dt: &CocoDetections,
    image: ImageId,
    cat: Option<CategoryId>,
    max_dets: usize,
) -> Vec<usize> {
    let indices: &[usize] = match cat {
        Some(c) => dt.indices_for(image, c),
        None => dt.indices_for_image(image),
    };
    let dts = dt.detections();
    // Stable mergesort tiebreak (quirk A1) is part of the parity contract;
    // do not swap for select_nth_unstable.
    let scores: Vec<f64> = indices.iter().map(|&i| dts[i].score).collect();
    let perm = argsort_score_desc(&scores);
    perm.into_iter()
        .take(max_dets)
        .map(|k| indices[k])
        .collect()
}

/// Area-invariant per-cell buffers shared across every area-range pass.
struct CellBuffers<'a> {
    image_id: i64,
    category_id: i64,
    max_det: usize,
    gt_areas: &'a [f64],
    gt_iscrowd: &'a [bool],
    gt_base_ignore: &'a [bool],
    gt_ids: &'a [i64],
    dt_areas: &'a [f64],
    dt_scores: &'a [f64],
    dt_ids: &'a [i64],
    iou: ArrayView2<'a, f64>,
    /// LVIS federated AA3: when `true`, the entire `(image, category)`
    /// cell is in `not_exhaustive_category_ids[image]`, so every
    /// unmatched DT in the cell gets `dt_ignore = true` (mirrors
    /// lvis-api `eval.py:278`). `false` outside LVIS evaluation.
    not_exhaustive: bool,
}

fn evaluate_cell(
    buf: &CellBuffers<'_>,
    area: &AreaRange,
    iou_thresholds: &[f64],
    parity_mode: ParityMode,
) -> Result<(PerImageEval, EvalImageMeta), EvalError> {
    // D3 + D6/D7: per-call ignore = base | out-of-area.
    let gt_ignore: Vec<bool> = buf
        .gt_base_ignore
        .iter()
        .zip(buf.gt_areas)
        .map(|(&base, &a)| base || !area.contains(a))
        .collect();

    let MatchResult {
        dt_perm,
        gt_perm,
        dt_matches: dt_matches_pos,
        gt_matches: gt_matches_pos,
        mut dt_ignore,
    } = match_image(
        buf.iou,
        &gt_ignore,
        buf.gt_iscrowd,
        buf.dt_scores,
        iou_thresholds,
        parity_mode,
    )?;

    let n_t = iou_thresholds.len();
    let n_d = buf.dt_scores.len();
    let n_g = gt_ignore.len();

    let dt_scores_sorted: Vec<f64> = dt_perm.iter().map(|&k| buf.dt_scores[k]).collect();
    let dt_in_range_sorted: Vec<bool> = dt_perm
        .iter()
        .map(|&k| area.contains(buf.dt_areas[k]))
        .collect();
    let gt_ignore_sorted: Vec<bool> = gt_perm.iter().map(|&k| gt_ignore[k]).collect();
    let dt_ids_sorted: Vec<i64> = dt_perm.iter().map(|&k| buf.dt_ids[k]).collect();
    let gt_ids_sorted: Vec<i64> = gt_perm.iter().map(|&k| buf.gt_ids[k]).collect();

    let mut dt_matched = Array2::<bool>::default((n_t, n_d));
    let mut dt_matches_id = Array2::<i64>::zeros((n_t, n_d));
    let mut gt_matches_id = Array2::<i64>::zeros((n_t, n_g));
    for t in 0..n_t {
        for d in 0..n_d {
            let m = dt_matches_pos[(t, d)];
            let matched = m >= 0;
            dt_matched[(t, d)] = matched;
            if matched {
                dt_matches_id[(t, d)] = gt_ids_sorted[m as usize];
            }
            // B7: unmatched AND out-of-area → ignore.
            // AA3 (LVIS): unmatched in a not_exhaustive cell → ignore.
            // Both branches share the same `dt_ignore` field; the
            // matching engine never sees the LVIS-specific flag.
            if !matched && (!dt_in_range_sorted[d] || buf.not_exhaustive) {
                dt_ignore[(t, d)] = true;
            }
        }
        for g in 0..n_g {
            let p = gt_matches_pos[(t, g)];
            if p >= 0 {
                gt_matches_id[(t, g)] = dt_ids_sorted[p as usize];
            }
        }
    }

    let cell = PerImageEval {
        dt_scores: dt_scores_sorted,
        dt_matched,
        dt_ignore,
        gt_ignore: gt_ignore_sorted,
    };
    let meta = EvalImageMeta {
        image_id: buf.image_id,
        category_id: buf.category_id,
        area_rng: [area.lo, area.hi],
        max_det: buf.max_det,
        dt_ids: dt_ids_sorted,
        gt_ids: gt_ids_sorted,
        dt_matches: dt_matches_id,
        gt_matches: gt_matches_id,
    };
    Ok((cell, meta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulate::{accumulate, AccumulateParams};
    use crate::dataset::{AnnId, Bbox, CategoryMeta, CocoAnnotation, DetectionInput, ImageMeta};
    use crate::parity::{iou_thresholds, recall_thresholds};
    use crate::summarize::summarize_detection;

    fn img(id: i64, w: u32, h: u32) -> ImageMeta {
        ImageMeta {
            id: ImageId(id),
            width: w,
            height: h,
            file_name: None,
        }
    }

    fn cat(id: i64, name: &str) -> CategoryMeta {
        CategoryMeta {
            id: CategoryId(id),
            name: name.into(),
            supercategory: None,
        }
    }

    fn ann(id: i64, image: i64, cat: i64, bbox: (f64, f64, f64, f64)) -> CocoAnnotation {
        CocoAnnotation {
            id: AnnId(id),
            image_id: ImageId(image),
            category_id: CategoryId(cat),
            area: bbox.2 * bbox.3,
            is_crowd: false,
            ignore_flag: None,
            bbox: Bbox {
                x: bbox.0,
                y: bbox.1,
                w: bbox.2,
                h: bbox.3,
            },
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
        }
    }

    fn dt_input(image: i64, cat: i64, score: f64, bbox: (f64, f64, f64, f64)) -> DetectionInput {
        DetectionInput {
            id: None,
            image_id: ImageId(image),
            category_id: CategoryId(cat),
            score,
            bbox: Bbox {
                x: bbox.0,
                y: bbox.1,
                w: bbox.2,
                h: bbox.3,
            },
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
        }
    }

    fn perfect_match_grid() -> EvalGrid {
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "thing")];
        let anns = vec![
            ann(1, 1, 1, (0.0, 0.0, 10.0, 10.0)),
            ann(2, 1, 1, (50.0, 50.0, 10.0, 10.0)),
        ];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        let dts = CocoDetections::from_inputs(vec![
            dt_input(1, 1, 0.9, (0.0, 0.0, 10.0, 10.0)),
            dt_input(1, 1, 0.8, (50.0, 50.0, 10.0, 10.0)),
        ])
        .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        evaluate_bbox(&gt, &dts, params, ParityMode::Strict).unwrap()
    }

    #[test]
    fn d4_coco_default_area_ranges_pin_literal_values() {
        // D4: the four COCO buckets are (0, 1e10), (0, 1024),
        // (1024, 9216), (9216, 1e10), labelled "all" / "small" /
        // "medium" / "large". Pin the literal numbers — the 1e10 sentinel
        // and the 32² / 96² boundaries are the parity contract; bumping
        // either silently in source would shift bucket membership
        // throughout the suite.
        let ranges = AreaRange::coco_default();
        assert_eq!(ranges.len(), 4);
        assert_eq!(
            (ranges[0].lo, ranges[0].hi),
            (0.0, 1e10),
            "all bucket bounds"
        );
        assert_eq!(
            (ranges[1].lo, ranges[1].hi),
            (0.0, 1024.0),
            "small bucket bounds"
        );
        assert_eq!(
            (ranges[2].lo, ranges[2].hi),
            (1024.0, 9216.0),
            "medium bucket bounds"
        );
        assert_eq!(
            (ranges[3].lo, ranges[3].hi),
            (9216.0, 1e10),
            "large bucket bounds"
        );

        // A-axis indices line up with crate::AreaRng's labelled
        // constants. The summarizer keys on `index`, so this is the
        // bridge between the orchestrator and the canonical labels.
        use crate::summarize::AreaRng;
        assert_eq!(ranges[0].index, AreaRng::ALL.index);
        assert_eq!(AreaRng::ALL.label.as_ref(), "all");
        assert_eq!(ranges[1].index, AreaRng::SMALL.index);
        assert_eq!(AreaRng::SMALL.label.as_ref(), "small");
        assert_eq!(ranges[2].index, AreaRng::MEDIUM.index);
        assert_eq!(AreaRng::MEDIUM.label.as_ref(), "medium");
        assert_eq!(ranges[3].index, AreaRng::LARGE.index);
        assert_eq!(AreaRng::LARGE.label.as_ref(), "large");

        // The 1e10 upper bound is bit-equal to pycocotools' `1e5 ** 2`.
        // Pinning the bit pattern guarantees the strict-mode area filter
        // makes the same `>` / `<` decisions the Python reference does.
        let pyco_unbounded: f64 = 1e5_f64.powi(2);
        assert_eq!(pyco_unbounded.to_bits(), 1e10_f64.to_bits());
        assert_eq!(ranges[0].hi.to_bits(), 1e10_f64.to_bits());
        assert_eq!(ranges[3].hi.to_bits(), 1e10_f64.to_bits());
    }

    #[test]
    fn perfect_match_produces_one_cell_per_area_range() {
        let grid = perfect_match_grid();
        assert_eq!(grid.n_categories, 1);
        assert_eq!(grid.n_area_ranges, 4);
        assert_eq!(grid.n_images, 1);
        // Both DTs perfectly overlap their GTs → all four area cells exist.
        let cells: Vec<_> = grid.eval_imgs.iter().filter(|c| c.is_some()).collect();
        assert_eq!(cells.len(), 4);
        // The "all" bucket (a=0) has both DTs matched at every threshold.
        let all_cell = grid.cell(0, 0, 0).unwrap();
        assert_eq!(all_cell.dt_scores.len(), 2);
        assert!(all_cell.dt_matched.iter().all(|&m| m));
        assert!(all_cell.dt_ignore.iter().all(|&ig| !ig));
    }

    #[test]
    fn perfect_match_summarizes_to_one() {
        let grid = perfect_match_grid();
        let max_dets = vec![1usize, 10, 100];
        let acc = accumulate(
            &grid.eval_imgs,
            AccumulateParams {
                iou_thresholds: iou_thresholds(),
                recall_thresholds: recall_thresholds(),
                max_dets: &max_dets,
                n_categories: grid.n_categories,
                n_area_ranges: grid.n_area_ranges,
                n_images: grid.n_images,
            },
            ParityMode::Strict,
        )
        .unwrap();
        let summary = summarize_detection(&acc, iou_thresholds(), &max_dets).unwrap();
        let stats = summary.stats();
        // GTs are 10x10 → area 100, which falls inside `small` (< 32²)
        // and `all`. `medium` and `large` see no in-range GTs, so AP and
        // AR collapse to the -1 sentinel (quirk C5).
        assert!((stats[0] - 1.0).abs() < 1e-12, "AP={}", stats[0]);
        assert!((stats[3] - 1.0).abs() < 1e-12, "AP_S={}", stats[3]);
        assert_eq!(stats[4], -1.0, "AP_M should be -1 with no medium GTs");
        assert_eq!(stats[5], -1.0, "AP_L should be -1 with no large GTs");
        assert!((stats[8] - 1.0).abs() < 1e-12, "AR@100={}", stats[8]);
    }

    #[test]
    fn b7_unmatched_dt_outside_area_range_is_ignored() {
        // GT and DT both 200x200 (40000 area, "large" bucket). The
        // small-area cell (a=1, range [0, 32²)) sees the GT as ignored
        // (D6/D7) and the unmatched DT as ignored (B7).
        let images = vec![img(1, 300, 300)];
        let cats = vec![cat(1, "thing")];
        let anns = vec![ann(1, 1, 1, (0.0, 0.0, 200.0, 200.0))];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        let dts =
            CocoDetections::from_inputs(vec![dt_input(1, 1, 0.5, (200.0, 200.0, 50.0, 50.0))])
                .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let grid = evaluate_bbox(&gt, &dts, params, ParityMode::Strict).unwrap();
        let small = grid.cell(0, 1, 0).unwrap();
        // GT is out-of-area, so gt_ignore=true.
        assert_eq!(small.gt_ignore, vec![true]);
        // DT is unmatched (no IoU with GT) AND out-of-area → B7 sets ignore.
        assert!(small.dt_ignore.iter().all(|&ig| ig));
        assert!(small.dt_matched.iter().all(|&m| !m));
    }

    #[test]
    fn d6_boundary_area_lands_in_both_buckets() {
        // D6 (strict): pycocotools (cocoeval.py:251) uses non-strict
        // inclusion on both ends, so a GT/DT with area exactly equal to a
        // bucket boundary (32² = 1024) lands in *both* adjacent buckets.
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "thing")];
        // 32x32 → area 1024 exactly.
        let anns = vec![ann(1, 1, 1, (0.0, 0.0, 32.0, 32.0))];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        let dts =
            CocoDetections::from_inputs(vec![dt_input(1, 1, 0.5, (0.0, 0.0, 32.0, 32.0))]).unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let grid = evaluate_bbox(&gt, &dts, params, ParityMode::Strict).unwrap();
        // small (lo=0, hi=32²=1024): area 1024 == hi → included.
        let small = grid.cell(0, 1, 0).unwrap();
        assert_eq!(small.gt_ignore, vec![false]);
        // medium (lo=1024, hi=96²=9216): area 1024 == lo → included.
        let medium = grid.cell(0, 2, 0).unwrap();
        assert_eq!(medium.gt_ignore, vec![false]);
        // all (lo=0, hi=1e10): area 1024 lies inside.
        let all = grid.cell(0, 0, 0).unwrap();
        assert_eq!(all.gt_ignore, vec![false]);
        // large (lo=96²=9216, hi=1e10): area 1024 < 9216 → ignored.
        let large = grid.cell(0, 3, 0).unwrap();
        assert_eq!(large.gt_ignore, vec![true]);
    }

    #[test]
    fn l4_use_cats_false_collapses_categories() {
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "a"), cat(2, "b")];
        let anns = vec![
            ann(1, 1, 1, (0.0, 0.0, 10.0, 10.0)),
            ann(2, 1, 2, (50.0, 50.0, 10.0, 10.0)),
        ];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        // DT with category=1 overlapping the cat-2 GT — only matches
        // when use_cats=false.
        let dts = CocoDetections::from_inputs(vec![dt_input(1, 1, 0.9, (50.0, 50.0, 10.0, 10.0))])
            .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: false,
            retain_iou: false,
        };
        let grid = evaluate_bbox(&gt, &dts, params, ParityMode::Strict).unwrap();
        assert_eq!(grid.n_categories, 1);
        let all = grid.cell(0, 0, 0).unwrap();
        // Both GTs land in the single bucket; the DT matches the second.
        assert_eq!(all.gt_ignore.len(), 2);
        assert_eq!(all.dt_scores.len(), 1);
        assert!(all.dt_matched.iter().all(|&m| m));
    }

    #[test]
    fn max_dets_per_image_caps_top_n_by_score() {
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "thing")];
        let anns = vec![ann(1, 1, 1, (0.0, 0.0, 10.0, 10.0))];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        let dts = CocoDetections::from_inputs(vec![
            dt_input(1, 1, 0.1, (50.0, 50.0, 5.0, 5.0)),
            dt_input(1, 1, 0.9, (0.0, 0.0, 10.0, 10.0)),
            dt_input(1, 1, 0.5, (50.0, 50.0, 5.0, 5.0)),
        ])
        .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 2,
            use_cats: true,
            retain_iou: false,
        };
        let grid = evaluate_bbox(&gt, &dts, params, ParityMode::Strict).unwrap();
        let all = grid.cell(0, 0, 0).unwrap();
        // Only the top-2 by score survive the cap.
        assert_eq!(all.dt_scores.len(), 2);
        assert_eq!(all.dt_scores[0], 0.9);
        assert_eq!(all.dt_scores[1], 0.5);
    }

    #[test]
    fn d1_parity_mode_propagates_to_base_ignore() {
        // GT with iscrowd=0 and explicit ignore=1.
        // Strict (pycocotools): ignore := iscrowd → false, the GT
        // counts and the matching DT scores a TP.
        // Corrected: respects user's ignore=1 → true, the GT becomes
        // ignored and the DT picks it up via B6 (dt_ignore=true).
        const ANN_JSON: &str = r#"{
            "images": [{"id": 1, "width": 100, "height": 100}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [0, 0, 10, 10], "area": 100,
                 "iscrowd": 0, "ignore": 1}
            ],
            "categories": [{"id": 1, "name": "thing"}]
        }"#;
        let gt = CocoDataset::from_json_bytes(ANN_JSON.as_bytes()).unwrap();
        let dts =
            CocoDetections::from_inputs(vec![dt_input(1, 1, 0.9, (0.0, 0.0, 10.0, 10.0))]).unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };

        let strict = evaluate_bbox(&gt, &dts, params, ParityMode::Strict).unwrap();
        let strict_all = strict.cell(0, 0, 0).unwrap();
        assert_eq!(strict_all.gt_ignore, vec![false]);
        assert!(strict_all.dt_ignore.iter().all(|&ig| !ig));

        let corrected = evaluate_bbox(&gt, &dts, params, ParityMode::Corrected).unwrap();
        let corrected_all = corrected.cell(0, 0, 0).unwrap();
        assert_eq!(corrected_all.gt_ignore, vec![true]);
        // DT matched the now-ignored GT → B6 inherits the ignore flag.
        assert!(corrected_all.dt_ignore.iter().all(|&ig| ig));
    }

    #[test]
    fn cell_meta_carries_pycocotools_shape() {
        let grid = perfect_match_grid();
        // The "all" bucket sees both DTs matched.
        let meta = grid.cell_meta(0, 0, 0).unwrap();
        assert_eq!(meta.image_id, 1);
        assert_eq!(meta.category_id, 1);
        assert_eq!(meta.area_rng, [0.0, AREA_UNBOUNDED]);
        assert_eq!(meta.max_det, 100);
        // DTs sorted score-desc: id=1 (score 0.9) before id=2 (score 0.8).
        assert_eq!(meta.dt_ids, vec![1, 2]);
        // GTs sorted ignore-asc: both non-ignore, stable order preserved.
        assert_eq!(meta.gt_ids, vec![1, 2]);
        let n_t = iou_thresholds().len();
        assert_eq!(meta.dt_matches.shape(), &[n_t, 2]);
        assert_eq!(meta.gt_matches.shape(), &[n_t, 2]);
        // dt_matches carries the matched GT id (or 0); both DTs perfectly
        // overlap their same-position GT at every threshold.
        for t in 0..n_t {
            assert_eq!(meta.dt_matches[(t, 0)], 1, "dt[0] -> gt[1] at t={t}");
            assert_eq!(meta.dt_matches[(t, 1)], 2, "dt[1] -> gt[2] at t={t}");
            assert_eq!(meta.gt_matches[(t, 0)], 1, "gt[1] -> dt[1] at t={t}");
            assert_eq!(meta.gt_matches[(t, 1)], 2, "gt[2] -> dt[2] at t={t}");
        }
    }

    #[test]
    fn cell_meta_unmatched_dt_uses_zero_sentinel() {
        // Single GT, single DT with no overlap → unmatched at every threshold.
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "thing")];
        let anns = vec![ann(7, 1, 1, (0.0, 0.0, 10.0, 10.0))];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        let dts = CocoDetections::from_inputs(vec![dt_input(1, 1, 0.5, (50.0, 50.0, 10.0, 10.0))])
            .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let grid = evaluate_bbox(&gt, &dts, params, ParityMode::Strict).unwrap();
        let meta = grid.cell_meta(0, 0, 0).unwrap();
        assert_eq!(meta.gt_ids, vec![7]);
        // Auto-assigned DT id starts at 1 (first detection).
        assert_eq!(meta.dt_ids.len(), 1);
        assert!(meta.dt_matches.iter().all(|&x| x == 0));
        assert!(meta.gt_matches.iter().all(|&x| x == 0));
    }

    #[test]
    fn cell_meta_use_cats_false_emits_sentinel_category() {
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "a"), cat(2, "b")];
        let anns = vec![ann(1, 1, 1, (0.0, 0.0, 10.0, 10.0))];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        let dts =
            CocoDetections::from_inputs(vec![dt_input(1, 1, 0.9, (0.0, 0.0, 10.0, 10.0))]).unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: false,
            retain_iou: false,
        };
        let grid = evaluate_bbox(&gt, &dts, params, ParityMode::Strict).unwrap();
        let meta = grid.cell_meta(0, 0, 0).unwrap();
        assert_eq!(meta.category_id, COLLAPSED_CATEGORY_SENTINEL);
    }

    #[test]
    fn missing_dt_image_yields_none_cells() {
        // Pycocotools' `evaluateImg` returns a record (not None) when
        // GTs exist but DTs do not — vernier matches that.
        let images = vec![img(1, 100, 100), img(2, 100, 100)];
        let cats = vec![cat(1, "thing")];
        let anns = vec![ann(1, 1, 1, (0.0, 0.0, 10.0, 10.0))];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        let dts = CocoDetections::from_inputs(vec![]).unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let grid = evaluate_bbox(&gt, &dts, params, ParityMode::Strict).unwrap();
        for a in 0..4 {
            assert!(grid.cell(0, a, 0).is_some(), "image 1 area {a}");
            assert!(grid.cell(0, a, 1).is_none(), "image 2 area {a}");
        }
    }

    fn square_polygon(x: f64, y: f64, side: f64) -> Segmentation {
        Segmentation::Polygons(vec![vec![
            x,
            y,
            x + side,
            y,
            x + side,
            y + side,
            x,
            y + side,
        ]])
    }

    fn ann_with_segm(
        id: i64,
        image: i64,
        cat: i64,
        bbox: (f64, f64, f64, f64),
        segm: Segmentation,
    ) -> CocoAnnotation {
        CocoAnnotation {
            id: AnnId(id),
            image_id: ImageId(image),
            category_id: CategoryId(cat),
            area: bbox.2 * bbox.3,
            is_crowd: false,
            ignore_flag: None,
            bbox: Bbox {
                x: bbox.0,
                y: bbox.1,
                w: bbox.2,
                h: bbox.3,
            },
            segmentation: Some(segm),
            keypoints: None,
            num_keypoints: None,
        }
    }

    fn dt_input_with_segm(
        image: i64,
        cat: i64,
        score: f64,
        bbox: (f64, f64, f64, f64),
        segm: Segmentation,
    ) -> DetectionInput {
        DetectionInput {
            id: None,
            image_id: ImageId(image),
            category_id: CategoryId(cat),
            score,
            bbox: Bbox {
                x: bbox.0,
                y: bbox.1,
                w: bbox.2,
                h: bbox.3,
            },
            segmentation: Some(segm),
            keypoints: None,
            num_keypoints: None,
        }
    }

    #[test]
    fn segm_perfect_overlap_summarizes_to_one() {
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "thing")];
        let anns = vec![ann_with_segm(
            1,
            1,
            1,
            (10.0, 10.0, 20.0, 20.0),
            square_polygon(10.0, 10.0, 20.0),
        )];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        let dts = CocoDetections::from_inputs(vec![dt_input_with_segm(
            1,
            1,
            0.9,
            (10.0, 10.0, 20.0, 20.0),
            square_polygon(10.0, 10.0, 20.0),
        )])
        .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let grid = evaluate_segm(&gt, &dts, params, ParityMode::Strict).unwrap();
        let max_dets = vec![1usize, 10, 100];
        let acc = accumulate(
            &grid.eval_imgs,
            AccumulateParams {
                iou_thresholds: iou_thresholds(),
                recall_thresholds: recall_thresholds(),
                max_dets: &max_dets,
                n_categories: grid.n_categories,
                n_area_ranges: grid.n_area_ranges,
                n_images: grid.n_images,
            },
            ParityMode::Strict,
        )
        .unwrap();
        let summary = summarize_detection(&acc, iou_thresholds(), &max_dets).unwrap();
        let stats = summary.stats();
        assert!((stats[0] - 1.0).abs() < 1e-12, "AP={}", stats[0]);
    }

    #[test]
    fn segm_disjoint_masks_summarize_to_zero() {
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "thing")];
        let anns = vec![ann_with_segm(
            1,
            1,
            1,
            (0.0, 0.0, 10.0, 10.0),
            square_polygon(0.0, 0.0, 10.0),
        )];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        let dts = CocoDetections::from_inputs(vec![dt_input_with_segm(
            1,
            1,
            0.9,
            (50.0, 50.0, 10.0, 10.0),
            square_polygon(50.0, 50.0, 10.0),
        )])
        .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let grid = evaluate_segm(&gt, &dts, params, ParityMode::Strict).unwrap();
        let all = grid.cell(0, 0, 0).unwrap();
        // No overlap → no match at any threshold.
        assert!(all.dt_matched.iter().all(|&m| !m));
    }

    #[test]
    fn segm_missing_gt_segmentation_surfaces_typed_error() {
        // GT has no `segmentation` field; running segm eval against it
        // must surface InvalidAnnotation, not silently treat as empty.
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "thing")];
        let anns = vec![ann(7, 1, 1, (0.0, 0.0, 10.0, 10.0))];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        let dts = CocoDetections::from_inputs(vec![dt_input_with_segm(
            1,
            1,
            0.9,
            (0.0, 0.0, 10.0, 10.0),
            square_polygon(0.0, 0.0, 10.0),
        )])
        .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let err = evaluate_segm(&gt, &dts, params, ParityMode::Strict).unwrap_err();
        match err {
            EvalError::InvalidAnnotation { detail } => {
                assert!(detail.contains("GT id=7"), "msg: {detail}");
            }
            other => panic!("expected InvalidAnnotation, got {other:?}"),
        }
    }

    #[test]
    fn j2_bbox_only_dt_under_segm_iou_type_raises_in_corrected_mode() {
        // Quirk J2 (`corrected`): vernier refuses to silently coerce a
        // bbox-only DT into a rectangle mask under iouType="segm". The
        // typed error cites the offending DT id and image.
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "thing")];
        let anns = vec![ann_with_segm(
            1,
            1,
            1,
            (0.0, 0.0, 10.0, 10.0),
            square_polygon(0.0, 0.0, 10.0),
        )];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        // DT without a segmentation field — only bbox.
        let dts =
            CocoDetections::from_inputs(vec![dt_input(1, 1, 0.9, (0.0, 0.0, 10.0, 10.0))]).unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let err = evaluate_segm(&gt, &dts, params, ParityMode::Corrected).unwrap_err();
        match err {
            EvalError::InvalidAnnotation { detail } => {
                assert!(detail.contains("DT"), "expected DT in msg: {detail}");
                assert!(detail.contains("J2"), "expected J2 cite in msg: {detail}");
            }
            other => panic!("expected InvalidAnnotation, got {other:?}"),
        }
    }

    #[test]
    fn j2_bbox_only_dt_under_segm_iou_type_synthesizes_in_strict_mode() {
        // Quirk J2 (`strict`): pycocotools/coco.py:341 synthesizes a
        // 4-point rectangle polygon `[[x1,y1, x1,y2, x2,y2, x2,y1]]`
        // from the DT bbox and rasterizes it. A GT polygon perfectly
        // covering the same rectangle therefore IoU=1 against the
        // synthesized DT mask.
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "thing")];
        // GT polygon covers a 10×10 square at (0, 0).
        let anns = vec![ann_with_segm(
            1,
            1,
            1,
            (0.0, 0.0, 10.0, 10.0),
            square_polygon(0.0, 0.0, 10.0),
        )];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        // DT bbox covers the same rectangle but carries no `segmentation`.
        let dts =
            CocoDetections::from_inputs(vec![dt_input(1, 1, 0.9, (0.0, 0.0, 10.0, 10.0))]).unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let grid = evaluate_segm(&gt, &dts, params, ParityMode::Strict).unwrap();
        let all = grid.cell(0, 0, 0).unwrap();
        // Synthesized rectangle exactly covers the GT polygon → match
        // at every threshold.
        assert!(all.dt_matched.iter().all(|&m| m), "expected matches");
    }

    #[test]
    fn j6_heterogeneous_dt_list_first_with_segm_second_without_raises_in_corrected_mode() {
        // Quirk J6 (`corrected`): per-entry dispatch. A heterogeneous DT
        // list under iouType="segm" — DT[0] carries a `segmentation`,
        // DT[1] does not — is rejected up-front in corrected mode rather
        // than silently routed through pycocotools' first-entry-decides
        // dispatch (`coco.py:330-363`). Verifies that vernier inspects
        // each entry independently rather than dispatching from `anns[0]`.
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "thing")];
        let anns = vec![ann_with_segm(
            1,
            1,
            1,
            (0.0, 0.0, 10.0, 10.0),
            square_polygon(0.0, 0.0, 10.0),
        )];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        // DT[0] has segm, DT[1] does not. pycocotools' first-entry
        // dispatch would route into the segm path on `anns[0]`, then
        // crash on `anns[1]` reading `ann['segmentation']`. vernier
        // raises InvalidAnnotation pinpointing the offending entry.
        let dts = CocoDetections::from_inputs(vec![
            dt_input_with_segm(
                1,
                1,
                0.9,
                (0.0, 0.0, 10.0, 10.0),
                square_polygon(0.0, 0.0, 10.0),
            ),
            dt_input(1, 1, 0.8, (50.0, 50.0, 10.0, 10.0)),
        ])
        .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let err = evaluate_segm(&gt, &dts, params, ParityMode::Corrected).unwrap_err();
        assert!(matches!(err, EvalError::InvalidAnnotation { .. }));
    }

    #[test]
    fn j6_heterogeneous_dt_list_first_without_segm_second_with_raises_in_corrected_mode() {
        // Mirror of the previous test with the order reversed. If the
        // dispatch were first-entry-decides (the pycocotools quirk J6
        // documents), DT[0] without `segmentation` would route to a
        // bbox-synthesis path and DT[1]'s segm would be ignored. Vernier
        // inspects every entry: missing segm anywhere in corrected mode
        // raises.
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "thing")];
        let anns = vec![ann_with_segm(
            1,
            1,
            1,
            (0.0, 0.0, 10.0, 10.0),
            square_polygon(0.0, 0.0, 10.0),
        )];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        let dts = CocoDetections::from_inputs(vec![
            dt_input(1, 1, 0.9, (0.0, 0.0, 10.0, 10.0)),
            dt_input_with_segm(
                1,
                1,
                0.8,
                (50.0, 50.0, 10.0, 10.0),
                square_polygon(50.0, 50.0, 10.0),
            ),
        ])
        .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let err = evaluate_segm(&gt, &dts, params, ParityMode::Corrected).unwrap_err();
        assert!(matches!(err, EvalError::InvalidAnnotation { .. }));
    }

    #[test]
    fn j6_heterogeneous_dt_list_in_strict_mode_synthesizes_per_entry() {
        // Quirk J2 (`strict`) layered with J6: per-entry dispatch under
        // strict mode means DTs without `segmentation` get the
        // bbox→polygon synthesis (matching pycocotools), while DTs with
        // a `segmentation` keep theirs. No first-entry-decides
        // global dispatch — every entry is handled independently.
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "thing")];
        let anns = vec![
            ann_with_segm(
                1,
                1,
                1,
                (0.0, 0.0, 10.0, 10.0),
                square_polygon(0.0, 0.0, 10.0),
            ),
            ann_with_segm(
                2,
                1,
                1,
                (50.0, 50.0, 10.0, 10.0),
                square_polygon(50.0, 50.0, 10.0),
            ),
        ];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        // DT[0] has segm covering GT[0]; DT[1] has only bbox covering GT[1].
        let dts = CocoDetections::from_inputs(vec![
            dt_input_with_segm(
                1,
                1,
                0.9,
                (0.0, 0.0, 10.0, 10.0),
                square_polygon(0.0, 0.0, 10.0),
            ),
            dt_input(1, 1, 0.8, (50.0, 50.0, 10.0, 10.0)),
        ])
        .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let grid = evaluate_segm(&gt, &dts, params, ParityMode::Strict).unwrap();
        let all = grid.cell(0, 0, 0).unwrap();
        // Both DTs match their respective GTs (DT[1] via synthesized
        // rectangle), so every threshold sees both as TPs.
        assert_eq!(all.dt_matched.shape(), &[iou_thresholds().len(), 2]);
        assert!(all.dt_matched.iter().all(|&m| m));
    }

    #[test]
    fn boundary_perfect_overlap_summarizes_to_one() {
        // Pins the wrapper end-to-end (kernel → grid → accumulate →
        // summarize) at AP=1; a regression in any stage trips this.
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "thing")];
        let anns = vec![ann_with_segm(
            1,
            1,
            1,
            (10.0, 10.0, 20.0, 20.0),
            square_polygon(10.0, 10.0, 20.0),
        )];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        let dts = CocoDetections::from_inputs(vec![dt_input_with_segm(
            1,
            1,
            0.9,
            (10.0, 10.0, 20.0, 20.0),
            square_polygon(10.0, 10.0, 20.0),
        )])
        .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let grid = evaluate_boundary(&gt, &dts, params, ParityMode::Strict, 0.02).unwrap();
        let max_dets = vec![1usize, 10, 100];
        let acc = accumulate(
            &grid.eval_imgs,
            AccumulateParams {
                iou_thresholds: iou_thresholds(),
                recall_thresholds: recall_thresholds(),
                max_dets: &max_dets,
                n_categories: grid.n_categories,
                n_area_ranges: grid.n_area_ranges,
                n_images: grid.n_images,
            },
            ParityMode::Strict,
        )
        .unwrap();
        let summary = summarize_detection(&acc, iou_thresholds(), &max_dets).unwrap();
        let stats = summary.stats();
        assert!((stats[0] - 1.0).abs() < 1e-12, "AP={}", stats[0]);
    }

    #[test]
    fn boundary_disjoint_masks_summarize_to_zero() {
        // Disjoint masks → bbox prefilter zeros the cell; no match at
        // any threshold.
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "thing")];
        let anns = vec![ann_with_segm(
            1,
            1,
            1,
            (0.0, 0.0, 10.0, 10.0),
            square_polygon(0.0, 0.0, 10.0),
        )];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        let dts = CocoDetections::from_inputs(vec![dt_input_with_segm(
            1,
            1,
            0.9,
            (50.0, 50.0, 10.0, 10.0),
            square_polygon(50.0, 50.0, 10.0),
        )])
        .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let grid = evaluate_boundary(&gt, &dts, params, ParityMode::Strict, 0.02).unwrap();
        let all = grid.cell(0, 0, 0).unwrap();
        assert!(all.dt_matched.iter().all(|&m| !m));
    }

    /// Two-image, two-category fixture exercised by the cache tests
    /// below. Returns gt + two distinct DT sets so a "second eval" is
    /// genuinely the same GT against fresh DTs (the training-loop
    /// validation pattern the cache is for).
    fn boundary_cache_fixture() -> (
        CocoDataset,
        CocoDetections,
        CocoDetections,
        OwnedEvaluateParams,
    ) {
        let images = vec![img(1, 100, 100), img(2, 100, 100)];
        let cats = vec![cat(1, "thing"), cat(2, "other")];
        let anns = vec![
            ann_with_segm(
                10,
                1,
                1,
                (10.0, 10.0, 20.0, 20.0),
                square_polygon(10.0, 10.0, 20.0),
            ),
            ann_with_segm(
                11,
                1,
                2,
                (50.0, 50.0, 15.0, 15.0),
                square_polygon(50.0, 50.0, 15.0),
            ),
            ann_with_segm(
                12,
                2,
                1,
                (5.0, 5.0, 25.0, 25.0),
                square_polygon(5.0, 5.0, 25.0),
            ),
        ];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        let dts_a = CocoDetections::from_inputs(vec![
            dt_input_with_segm(
                1,
                1,
                0.9,
                (10.0, 10.0, 20.0, 20.0),
                square_polygon(10.0, 10.0, 20.0),
            ),
            dt_input_with_segm(
                2,
                1,
                0.8,
                (5.0, 5.0, 25.0, 25.0),
                square_polygon(5.0, 5.0, 25.0),
            ),
        ])
        .unwrap();
        // dts_b shifts both predictions a little so the grid changes
        // but GT bands don't: this is the regime the cache is for.
        let dts_b = CocoDetections::from_inputs(vec![
            dt_input_with_segm(
                1,
                1,
                0.7,
                (12.0, 12.0, 20.0, 20.0),
                square_polygon(12.0, 12.0, 20.0),
            ),
            dt_input_with_segm(
                2,
                1,
                0.6,
                (8.0, 8.0, 25.0, 25.0),
                square_polygon(8.0, 8.0, 25.0),
            ),
        ])
        .unwrap();
        let params = OwnedEvaluateParams {
            iou_thresholds: iou_thresholds().to_vec(),
            area_ranges: AreaRange::coco_default().to_vec(),
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        (gt, dts_a, dts_b, params)
    }

    fn boundary_grid_cells(grid: &EvalGrid) -> Vec<f64> {
        grid.eval_imgs
            .iter()
            .filter_map(|c| c.as_ref())
            .flat_map(|c| c.dt_scores.iter().copied())
            .collect()
    }

    #[test]
    fn boundary_cached_matches_uncached_bit_exact() {
        // Same-GT, same-DT call via the cached entry point must
        // produce a grid bit-equal to the uncached entry point — the
        // cache is a memoization, never a semantic shift.
        let (gt, dts, _, params) = boundary_cache_fixture();
        let p = params.borrow();
        let baseline = evaluate_boundary(&gt, &dts, p, ParityMode::Strict, 0.02).unwrap();
        let cache = BoundaryGtCache::new();
        let cached_first =
            evaluate_boundary_cached(&gt, &dts, p, ParityMode::Strict, 0.02, &cache).unwrap();
        let cached_second =
            evaluate_boundary_cached(&gt, &dts, p, ParityMode::Strict, 0.02, &cache).unwrap();

        let baseline_scores = boundary_grid_cells(&baseline);
        let first_scores = boundary_grid_cells(&cached_first);
        let second_scores = boundary_grid_cells(&cached_second);
        assert_eq!(baseline_scores.len(), first_scores.len());
        for (b, c) in baseline_scores.iter().zip(first_scores.iter()) {
            assert_eq!(b.to_bits(), c.to_bits());
        }
        for (b, c) in baseline_scores.iter().zip(second_scores.iter()) {
            assert_eq!(b.to_bits(), c.to_bits());
        }
    }

    #[test]
    fn boundary_cache_populates_lazily_per_evaluated_cell() {
        // The cache fills as bands are derived, which only happens on
        // (image, category) cells that have a non-empty DT side. The
        // fixture has 3 GTs but only 2 ever participate under
        // `use_cats: true`: GT 11 (cat 2) has no matching DT, so its
        // band is never computed. Pinning the count documents the
        // lazy-load contract — entries we never need stay out of the
        // cache, keeping memory proportional to actual work.
        let (gt, dts, _, params) = boundary_cache_fixture();
        let cache = BoundaryGtCache::new();
        assert!(cache.is_empty());
        evaluate_boundary_cached(&gt, &dts, params.borrow(), ParityMode::Strict, 0.02, &cache)
            .unwrap();
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn boundary_cache_invalidates_on_ratio_change() {
        // Bands depend on dilation_ratio; reusing entries computed at
        // ratio R₁ when the call is at ratio R₂ would silently return
        // wrong numerics. The cache must drop+repopulate.
        let (gt, dts, _, params) = boundary_cache_fixture();
        let cache = BoundaryGtCache::new();
        evaluate_boundary_cached(&gt, &dts, params.borrow(), ParityMode::Strict, 0.02, &cache)
            .unwrap();
        let after_first = cache.len();
        evaluate_boundary_cached(&gt, &dts, params.borrow(), ParityMode::Strict, 0.05, &cache)
            .unwrap();
        // Same GT count, but every entry was re-derived at the new
        // ratio: parity below proves the entries reflect R=0.05, not
        // stale R=0.02 data.
        assert_eq!(cache.len(), after_first);
        let fresh =
            evaluate_boundary(&gt, &dts, params.borrow(), ParityMode::Strict, 0.05).unwrap();
        let cached =
            evaluate_boundary_cached(&gt, &dts, params.borrow(), ParityMode::Strict, 0.05, &cache)
                .unwrap();
        let fresh_scores = boundary_grid_cells(&fresh);
        let cached_scores = boundary_grid_cells(&cached);
        for (f, c) in fresh_scores.iter().zip(cached_scores.iter()) {
            assert_eq!(f.to_bits(), c.to_bits());
        }
    }

    #[test]
    fn boundary_cache_clear_resets_state() {
        let (gt, dts, _, params) = boundary_cache_fixture();
        let cache = BoundaryGtCache::new();
        evaluate_boundary_cached(&gt, &dts, params.borrow(), ParityMode::Strict, 0.02, &cache)
            .unwrap();
        assert!(!cache.is_empty());
        cache.clear();
        assert!(cache.is_empty());
        // Post-clear the next call must repopulate from scratch and
        // still produce the right answer.
        let after =
            evaluate_boundary_cached(&gt, &dts, params.borrow(), ParityMode::Strict, 0.02, &cache)
                .unwrap();
        let baseline =
            evaluate_boundary(&gt, &dts, params.borrow(), ParityMode::Strict, 0.02).unwrap();
        let after_scores = boundary_grid_cells(&after);
        let baseline_scores = boundary_grid_cells(&baseline);
        for (a, b) in after_scores.iter().zip(baseline_scores.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn boundary_cache_survives_changing_dt() {
        // The training-loop pattern: same GT, fresh DT each call.
        // Cache size must stay constant across DT swaps (the cache
        // only ever holds GT bands), and parity vs uncached must
        // hold for both DT sets.
        let (gt, dts_a, dts_b, params) = boundary_cache_fixture();
        let cache = BoundaryGtCache::new();
        let cached_a = evaluate_boundary_cached(
            &gt,
            &dts_a,
            params.borrow(),
            ParityMode::Strict,
            0.02,
            &cache,
        )
        .unwrap();
        let len_after_a = cache.len();
        let cached_b = evaluate_boundary_cached(
            &gt,
            &dts_b,
            params.borrow(),
            ParityMode::Strict,
            0.02,
            &cache,
        )
        .unwrap();
        assert_eq!(cache.len(), len_after_a);

        let baseline_a =
            evaluate_boundary(&gt, &dts_a, params.borrow(), ParityMode::Strict, 0.02).unwrap();
        let baseline_b =
            evaluate_boundary(&gt, &dts_b, params.borrow(), ParityMode::Strict, 0.02).unwrap();
        for (lhs, rhs) in boundary_grid_cells(&cached_a)
            .iter()
            .zip(boundary_grid_cells(&baseline_a).iter())
        {
            assert_eq!(lhs.to_bits(), rhs.to_bits());
        }
        for (lhs, rhs) in boundary_grid_cells(&cached_b)
            .iter()
            .zip(boundary_grid_cells(&baseline_b).iter())
        {
            assert_eq!(lhs.to_bits(), rhs.to_bits());
        }
    }

    // ---------------------------------------------------------------
    // SegmGtCache (mirrors the BoundaryGtCache suite above; the segm
    // fixture is built on the same pieces but lives here so the
    // boundary tests stay focused on band-specific behaviour).
    // ---------------------------------------------------------------

    #[test]
    fn segm_cached_matches_uncached_bit_exact() {
        let (gt, dts, _, params) = boundary_cache_fixture();
        let p = params.borrow();
        let baseline = evaluate_segm(&gt, &dts, p, ParityMode::Strict).unwrap();
        let cache = SegmGtCache::new();
        let cached_first = evaluate_segm_cached(&gt, &dts, p, ParityMode::Strict, &cache).unwrap();
        let cached_second = evaluate_segm_cached(&gt, &dts, p, ParityMode::Strict, &cache).unwrap();

        let baseline_scores = boundary_grid_cells(&baseline);
        let first_scores = boundary_grid_cells(&cached_first);
        let second_scores = boundary_grid_cells(&cached_second);
        assert_eq!(baseline_scores.len(), first_scores.len());
        for (b, c) in baseline_scores.iter().zip(first_scores.iter()) {
            assert_eq!(b.to_bits(), c.to_bits());
        }
        for (b, c) in baseline_scores.iter().zip(second_scores.iter()) {
            assert_eq!(b.to_bits(), c.to_bits());
        }
    }

    #[test]
    fn segm_cache_populates_lazily_per_evaluated_cell() {
        // Same lazy-load contract as the boundary cache: only GTs
        // that participate in an evaluated `(image, category)` cell
        // — i.e. one with at least one DT — get cached. The
        // boundary fixture has 3 GTs but only 2 such cells under
        // `use_cats: true`.
        let (gt, dts, _, params) = boundary_cache_fixture();
        let cache = SegmGtCache::new();
        assert!(cache.is_empty());
        evaluate_segm_cached(&gt, &dts, params.borrow(), ParityMode::Strict, &cache).unwrap();
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn segm_cache_clear_resets_state() {
        let (gt, dts, _, params) = boundary_cache_fixture();
        let cache = SegmGtCache::new();
        evaluate_segm_cached(&gt, &dts, params.borrow(), ParityMode::Strict, &cache).unwrap();
        assert!(!cache.is_empty());
        cache.clear();
        assert!(cache.is_empty());
        let after =
            evaluate_segm_cached(&gt, &dts, params.borrow(), ParityMode::Strict, &cache).unwrap();
        let baseline = evaluate_segm(&gt, &dts, params.borrow(), ParityMode::Strict).unwrap();
        for (a, b) in boundary_grid_cells(&after)
            .iter()
            .zip(boundary_grid_cells(&baseline).iter())
        {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn segm_cache_survives_changing_dt() {
        // Training-loop pattern: same GT, fresh DT each call. Cache
        // size must stay constant across DT swaps (the cache only
        // holds GT entries) and parity vs uncached must hold for
        // both DT sets.
        let (gt, dts_a, dts_b, params) = boundary_cache_fixture();
        let cache = SegmGtCache::new();
        let cached_a =
            evaluate_segm_cached(&gt, &dts_a, params.borrow(), ParityMode::Strict, &cache).unwrap();
        let len_after_a = cache.len();
        let cached_b =
            evaluate_segm_cached(&gt, &dts_b, params.borrow(), ParityMode::Strict, &cache).unwrap();
        assert_eq!(cache.len(), len_after_a);

        let baseline_a = evaluate_segm(&gt, &dts_a, params.borrow(), ParityMode::Strict).unwrap();
        let baseline_b = evaluate_segm(&gt, &dts_b, params.borrow(), ParityMode::Strict).unwrap();
        for (lhs, rhs) in boundary_grid_cells(&cached_a)
            .iter()
            .zip(boundary_grid_cells(&baseline_a).iter())
        {
            assert_eq!(lhs.to_bits(), rhs.to_bits());
        }
        for (lhs, rhs) in boundary_grid_cells(&cached_b)
            .iter()
            .zip(boundary_grid_cells(&baseline_b).iter())
        {
            assert_eq!(lhs.to_bits(), rhs.to_bits());
        }
    }

    // ---------------------------------------------------------------
    // Phase 3: keypoints (OKS) eval pipeline (ADR-0012).
    // ---------------------------------------------------------------

    /// Builds a flat `[x, y, v, ...]` keypoint vector at a single point.
    /// `len` controls the per-category sigma length the kernel expects
    /// (17 for COCO-person).
    fn const_kps_vec(x: f64, y: f64, v: u32, len: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(3 * len);
        for _ in 0..len {
            out.push(x);
            out.push(y);
            out.push(f64::from(v));
        }
        out
    }

    fn ann_with_kps(
        id: i64,
        image: i64,
        cat: i64,
        bbox: (f64, f64, f64, f64),
        keypoints: Vec<f64>,
        num_keypoints: Option<u32>,
    ) -> CocoAnnotation {
        CocoAnnotation {
            id: AnnId(id),
            image_id: ImageId(image),
            category_id: CategoryId(cat),
            area: bbox.2 * bbox.3,
            is_crowd: false,
            ignore_flag: None,
            bbox: Bbox {
                x: bbox.0,
                y: bbox.1,
                w: bbox.2,
                h: bbox.3,
            },
            segmentation: None,
            keypoints: Some(keypoints),
            num_keypoints,
        }
    }

    fn dt_input_with_kps(
        image: i64,
        cat: i64,
        score: f64,
        bbox: (f64, f64, f64, f64),
        keypoints: Vec<f64>,
    ) -> DetectionInput {
        DetectionInput {
            id: None,
            image_id: ImageId(image),
            category_id: CategoryId(cat),
            score,
            bbox: Bbox {
                x: bbox.0,
                y: bbox.1,
                w: bbox.2,
                h: bbox.3,
            },
            segmentation: None,
            keypoints: Some(keypoints),
            num_keypoints: None,
        }
    }

    #[test]
    fn test_evaluate_keypoints_perfect_match() {
        // 1 image, 1 GT person, 1 DT person matching exactly. Every
        // keypoint aligns → OKS = 1.0 → matched at every threshold,
        // and the meta gt_matches matrix carries the matched DT id.
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "person")];
        let kps = const_kps_vec(50.0, 50.0, 2, 17);
        let anns = vec![ann_with_kps(
            1,
            1,
            1,
            (40.0, 40.0, 20.0, 20.0),
            kps.clone(),
            None,
        )];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        let dts = CocoDetections::from_inputs(vec![dt_input_with_kps(
            1,
            1,
            0.9,
            (40.0, 40.0, 20.0, 20.0),
            kps,
        )])
        .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let grid =
            evaluate_keypoints(&gt, &dts, params, ParityMode::Strict, HashMap::new()).unwrap();
        let cell = grid.cell(0, 0, 0).unwrap();
        // gt_ignore is false (visible keypoints), so the GT is in play.
        assert_eq!(cell.gt_ignore, vec![false]);
        // Every threshold matches the DT at score 0.9.
        assert!(cell.dt_matched.iter().all(|&m| m));
        // Meta carries the matched DT id at every threshold for this GT.
        let meta = grid.cell_meta(0, 0, 0).unwrap();
        assert!(
            meta.gt_matches.iter().all(|&id| id > 0),
            "every threshold should match the DT id (>0)",
        );
    }

    #[test]
    fn test_evaluate_keypoints_zero_overlap() {
        // 1 GT and 1 DT keypoints far apart (separated by ~1000 px on
        // a 10×10 bbox). OKS drops well below 0.5 → no match at any
        // threshold ≥ 0.5.
        let images = vec![img(1, 2000, 2000)];
        let cats = vec![cat(1, "person")];
        let gt_kps = const_kps_vec(50.0, 50.0, 2, 17);
        let dt_kps = const_kps_vec(1500.0, 1500.0, 2, 17);
        let anns = vec![ann_with_kps(
            1,
            1,
            1,
            (40.0, 40.0, 20.0, 20.0),
            gt_kps,
            None,
        )];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        let dts = CocoDetections::from_inputs(vec![dt_input_with_kps(
            1,
            1,
            0.9,
            (1490.0, 1490.0, 20.0, 20.0),
            dt_kps,
        )])
        .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let grid =
            evaluate_keypoints(&gt, &dts, params, ParityMode::Strict, HashMap::new()).unwrap();
        let cell = grid.cell(0, 0, 0).unwrap();
        assert!(
            cell.dt_matched.iter().all(|&m| !m),
            "DTs far from GT should not match at any IoU threshold",
        );
    }

    #[test]
    fn test_evaluate_keypoints_d2_implicit_ignore() {
        // D2 (`strict`): GT with `num_keypoints == 0` is treated as an
        // implicit ignore region, OR-ed with the existing ignore. This
        // GT carries v=0 on every triplet (so num_keypoints derives to
        // 0 even without the precomputed field) and is not is_crowd.
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "person")];
        let gt_kps = const_kps_vec(50.0, 50.0, 0, 17);
        let dt_kps = const_kps_vec(50.0, 50.0, 2, 17);
        let anns = vec![ann_with_kps(
            1,
            1,
            1,
            (40.0, 40.0, 20.0, 20.0),
            gt_kps,
            // Explicit Some(0) covers the precomputed-num_keypoints
            // path; the kernel treats it identically to deriving from
            // visibility flags.
            Some(0),
        )];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        let dts = CocoDetections::from_inputs(vec![dt_input_with_kps(
            1,
            1,
            0.9,
            (40.0, 40.0, 20.0, 20.0),
            dt_kps,
        )])
        .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let grid =
            evaluate_keypoints(&gt, &dts, params, ParityMode::Strict, HashMap::new()).unwrap();
        let cell = grid.cell(0, 0, 0).unwrap();
        assert_eq!(
            cell.gt_ignore,
            vec![true],
            "D2: zero-visible-keypoints GT must be ignored",
        );
    }

    #[test]
    fn test_evaluate_keypoints_per_category_sigmas() {
        // Two GTs in different categories; sigmas provided per category.
        // Each row of the OKS matrix uses the right sigma vector — we
        // verify by asserting the cell evaluates without error and that
        // both DTs match their same-category GT with the override-tuned
        // sigmas. We pick large sigmas (0.5) so a 1-pixel offset still
        // OKS≈1, ensuring matches at every threshold.
        let images = vec![img(1, 200, 200)];
        let cats = vec![cat(1, "person"), cat(2, "dog")];
        let gt_kps = const_kps_vec(50.0, 50.0, 2, 17);
        let anns = vec![
            ann_with_kps(1, 1, 1, (40.0, 40.0, 20.0, 20.0), gt_kps, None),
            ann_with_kps(
                2,
                1,
                2,
                (140.0, 140.0, 20.0, 20.0),
                const_kps_vec(150.0, 150.0, 2, 17),
                None,
            ),
        ];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        // DT[0] near GT[0] (cat 1), DT[1] near GT[1] (cat 2). Both off
        // by 1 pixel.
        let dts = CocoDetections::from_inputs(vec![
            dt_input_with_kps(
                1,
                1,
                0.9,
                (40.0, 40.0, 20.0, 20.0),
                const_kps_vec(51.0, 50.0, 2, 17),
            ),
            dt_input_with_kps(
                1,
                2,
                0.8,
                (140.0, 140.0, 20.0, 20.0),
                const_kps_vec(151.0, 150.0, 2, 17),
            ),
        ])
        .unwrap();
        let mut sigmas: HashMap<i64, Vec<f64>> = HashMap::new();
        sigmas.insert(1, vec![0.5_f64; 17]);
        sigmas.insert(2, vec![0.5_f64; 17]);
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let grid = evaluate_keypoints(&gt, &dts, params, ParityMode::Strict, sigmas).unwrap();
        // K-axis is [cat 1, cat 2]; each cell sees one GT and one DT.
        let cell_cat1 = grid.cell(0, 0, 0).unwrap();
        let cell_cat2 = grid.cell(1, 0, 0).unwrap();
        assert!(
            cell_cat1.dt_matched.iter().all(|&m| m),
            "cat-1 DT should match cat-1 GT under override sigmas",
        );
        assert!(
            cell_cat2.dt_matched.iter().all(|&m| m),
            "cat-2 DT should match cat-2 GT under override sigmas",
        );
    }

    #[test]
    fn test_evaluate_keypoints_missing_dt_kps_rejected() {
        // DT entry without `keypoints` field → the kernel build path
        // surfaces InvalidAnnotation. There is no parity-mode J2 analog
        // for keypoints (no bbox-synthesis fallback).
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "person")];
        let gt_kps = const_kps_vec(50.0, 50.0, 2, 17);
        let anns = vec![ann_with_kps(
            1,
            1,
            1,
            (40.0, 40.0, 20.0, 20.0),
            gt_kps,
            None,
        )];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        // DT has bbox + score but no keypoints — uses the existing
        // bbox-only `dt_input` helper.
        let dts = CocoDetections::from_inputs(vec![dt_input(1, 1, 0.9, (40.0, 40.0, 20.0, 20.0))])
            .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let err =
            evaluate_keypoints(&gt, &dts, params, ParityMode::Strict, HashMap::new()).unwrap_err();
        match err {
            EvalError::InvalidAnnotation { detail } => {
                assert!(detail.contains("DT"), "expected DT in msg: {detail}");
                assert!(
                    detail.contains("keypoints"),
                    "expected keypoints in msg: {detail}",
                );
            }
            other => panic!("expected InvalidAnnotation, got {other:?}"),
        }
    }

    #[test]
    fn test_keypoints_default_ignore_for_other_kernels() {
        // The D2 implicit-ignore clause must not bleed across kernels.
        // BboxIou::extra_gt_ignore (default impl) returns false even for
        // an annotation with num_keypoints=0; only OksSimilarity
        // overrides it.
        let ann_zero_kps = ann_with_kps(
            1,
            1,
            1,
            (0.0, 0.0, 10.0, 10.0),
            const_kps_vec(0.0, 0.0, 0, 17),
            Some(0),
        );
        assert!(
            !BboxIou.extra_gt_ignore(&ann_zero_kps),
            "BboxIou must keep the default `false` ignore",
        );
        assert!(
            !SegmIou.extra_gt_ignore(&ann_zero_kps),
            "SegmIou must keep the default `false` ignore",
        );
        assert!(
            !BoundaryIou {
                dilation_ratio: 0.02,
            }
            .extra_gt_ignore(&ann_zero_kps),
            "BoundaryIou must keep the default `false` ignore",
        );
        // And the OKS kernel does flip it on the same annotation.
        assert!(
            OksSimilarity::default().extra_gt_ignore(&ann_zero_kps),
            "OksSimilarity must flip D2 to true on zero-visible-keypoints GT",
        );
    }

    #[test]
    fn boundary_missing_gt_segmentation_surfaces_typed_error() {
        // Boundary reuses the segm GT-build path, so missing GT segm
        // surfaces the same typed error. Pinned here so a future
        // refactor that splits the build paths can't silently regress.
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "thing")];
        let anns = vec![ann(7, 1, 1, (0.0, 0.0, 10.0, 10.0))];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        let dts = CocoDetections::from_inputs(vec![dt_input_with_segm(
            1,
            1,
            0.9,
            (0.0, 0.0, 10.0, 10.0),
            square_polygon(0.0, 0.0, 10.0),
        )])
        .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let err = evaluate_boundary(&gt, &dts, params, ParityMode::Strict, 0.02).unwrap_err();
        match err {
            EvalError::InvalidAnnotation { detail } => {
                assert!(detail.contains("GT id=7"), "msg: {detail}");
            }
            other => panic!("expected InvalidAnnotation, got {other:?}"),
        }
    }

    // -- ADR-0026: federated cell-skip and dt_ignore extension ---------------

    /// Build an LVIS-style GT dataset directly: a `CocoDataset` whose
    /// federated metadata sets are populated. Mirrors what
    /// `from_lvis_json_bytes` produces, but lets tests pin the maps
    /// without round-tripping through JSON.
    fn lvis_dataset(
        images: &[ImageMeta],
        annotations: &[CocoAnnotation],
        categories: &[CategoryMeta],
        neg: &[(i64, Vec<i64>)],
        nel: &[(i64, Vec<i64>)],
        freq: &[(i64, crate::Frequency)],
    ) -> CocoDataset {
        // Build LVIS JSON bytes through the public loader so the
        // resulting dataset uses the same code path the FFI exercises.
        // (Constructing through `from_parts` would leave the federated
        // fields `None`.)
        let images_json: Vec<serde_json::Value> = images
            .iter()
            .map(|im| {
                let neg_for: Vec<i64> = neg
                    .iter()
                    .find(|(id, _)| *id == im.id.0)
                    .map(|(_, v)| v.clone())
                    .unwrap_or_default();
                let nel_for: Vec<i64> = nel
                    .iter()
                    .find(|(id, _)| *id == im.id.0)
                    .map(|(_, v)| v.clone())
                    .unwrap_or_default();
                serde_json::json!({
                    "id": im.id.0,
                    "width": im.width,
                    "height": im.height,
                    "neg_category_ids": neg_for,
                    "not_exhaustive_category_ids": nel_for,
                })
            })
            .collect();
        let cats_json: Vec<serde_json::Value> = categories
            .iter()
            .map(|c| {
                let f = freq
                    .iter()
                    .find(|(id, _)| *id == c.id.0)
                    .map(|(_, f)| match f {
                        crate::Frequency::Rare => "r",
                        crate::Frequency::Common => "c",
                        crate::Frequency::Frequent => "f",
                    })
                    .expect("test fixture must include frequency for every category");
                serde_json::json!({
                    "id": c.id.0,
                    "name": c.name,
                    "frequency": f,
                })
            })
            .collect();
        let anns_json = serde_json::to_value(annotations).unwrap();
        let payload = serde_json::json!({
            "images": images_json,
            "annotations": anns_json,
            "categories": cats_json,
        });
        let bytes = serde_json::to_vec(&payload).unwrap();
        CocoDataset::from_lvis_json_bytes(&bytes).unwrap()
    }

    #[test]
    fn aa4_skips_cells_outside_pos_union_neg() {
        // Two images, two categories. Image 1 has GTs of cat 1 only;
        // image 2 has GTs of cat 2 only. Neither image lists anything
        // in `neg`. The DT set predicts cat 2 on image 1 (a category
        // for which image 1 has no GT and no neg listing) — the
        // federated cell-skip MUST drop the resulting (image 1,
        // cat 2) cell entirely. Without AA4 the DT counts as a FP and
        // tanks AP.
        let images = vec![img(1, 100, 100), img(2, 100, 100)];
        let cats = vec![cat(1, "a"), cat(2, "b")];
        let anns = vec![
            ann(1, 1, 1, (0.0, 0.0, 10.0, 10.0)),
            ann(2, 2, 2, (0.0, 0.0, 10.0, 10.0)),
        ];
        let gt_lvis = lvis_dataset(
            &images,
            &anns,
            &cats,
            &[(1, vec![]), (2, vec![])],
            &[(1, vec![]), (2, vec![])],
            &[
                (1, crate::Frequency::Frequent),
                (2, crate::Frequency::Frequent),
            ],
        );
        let gt_coco = CocoDataset::from_parts(images, anns, cats).unwrap();
        // DT: a "stray" cat 2 prediction on image 1 — federated wants
        // it dropped, COCO will score it as a FP.
        let dts = CocoDetections::from_inputs(vec![
            dt_input(1, 1, 0.9, (0.0, 0.0, 10.0, 10.0)),
            dt_input(1, 2, 0.7, (50.0, 50.0, 10.0, 10.0)),
            dt_input(2, 2, 0.9, (0.0, 0.0, 10.0, 10.0)),
        ])
        .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let grid_lvis = evaluate_bbox(&gt_lvis, &dts, params, ParityMode::Strict).unwrap();
        let grid_coco = evaluate_bbox(&gt_coco, &dts, params, ParityMode::Strict).unwrap();

        // Cell layout: K=[cat 1, cat 2], A=[all], I=[image 1, image 2].
        // (image 1, cat 2) sits at k=1, a=0, i=0 — federated dataset
        // skips it (None), COCO dataset evaluates it (Some).
        let lvis_cell = grid_lvis.cell(1, 0, 0);
        let coco_cell = grid_coco.cell(1, 0, 0);
        assert!(lvis_cell.is_none(), "AA4: federated cell must be skipped");
        assert!(
            coco_cell.is_some(),
            "control: COCO dataset must evaluate the same cell"
        );
        // The (image 1, cat 1) cell is unaffected — federated and
        // COCO must agree there because cat 1 ∈ pos[1].
        assert_eq!(
            grid_lvis.cell(0, 0, 0).map(|c| c.dt_scores.len()),
            grid_coco.cell(0, 0, 0).map(|c| c.dt_scores.len()),
        );
    }

    #[test]
    fn aa4_keeps_neg_cells_with_no_gts() {
        // Same shape as the previous test, but image 1 lists cat 2 in
        // its `neg` set: the cell now stays (so we score recall on a
        // verified-absent category) and unmatched DTs become FPs.
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "a"), cat(2, "b")];
        let anns = vec![ann(1, 1, 1, (0.0, 0.0, 10.0, 10.0))];
        let gt = lvis_dataset(
            &images,
            &anns,
            &cats,
            &[(1, vec![2])], // cat 2 ∈ neg[1]
            &[(1, vec![])],
            &[
                (1, crate::Frequency::Frequent),
                (2, crate::Frequency::Frequent),
            ],
        );
        let dts = CocoDetections::from_inputs(vec![
            dt_input(1, 1, 0.9, (0.0, 0.0, 10.0, 10.0)),
            dt_input(1, 2, 0.7, (50.0, 50.0, 10.0, 10.0)),
        ])
        .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let grid = evaluate_bbox(&gt, &dts, params, ParityMode::Strict).unwrap();
        let cell = grid
            .cell(1, 0, 0)
            .expect("cat 2 ∈ neg[1] must produce an evaluated cell");
        // The cell has no GTs and one DT; the DT is an unmatched FP
        // (not ignored, because the cell is not in `not_exhaustive`).
        assert_eq!(cell.dt_scores.len(), 1);
        assert!(cell.dt_ignore.iter().all(|&ig| !ig));
    }

    #[test]
    fn aa3_dt_ignore_extension_in_not_exhaustive_cell() {
        // Image 1 has GTs of cat 1 and lists cat 1 in its
        // `not_exhaustive` set. The DT set has two predictions for
        // cat 1: one matches the GT (TP), the other is unmatched.
        // Quirk **AA3** says the unmatched DT must have
        // `dt_ignore = true`; the matched DT keeps `dt_ignore =
        // false`.
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "a")];
        let anns = vec![ann(1, 1, 1, (0.0, 0.0, 10.0, 10.0))];
        let gt = lvis_dataset(
            &images,
            &anns,
            &cats,
            &[(1, vec![])],
            &[(1, vec![1])], // cat 1 ∈ not_exhaustive[1]
            &[(1, crate::Frequency::Frequent)],
        );
        let dts = CocoDetections::from_inputs(vec![
            dt_input(1, 1, 0.9, (0.0, 0.0, 10.0, 10.0)),   // TP
            dt_input(1, 1, 0.7, (50.0, 50.0, 10.0, 10.0)), // unmatched FP candidate
        ])
        .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let grid = evaluate_bbox(&gt, &dts, params, ParityMode::Strict).unwrap();
        let cell = grid.cell(0, 0, 0).expect("cell must evaluate");
        let n_t = cell.dt_ignore.shape()[0];
        // sorted-DT order is descending by score: [TP, FP]. TP must
        // never be `dt_ignore = true` (B6 only flips ignore on
        // *unmatched* DTs); FP must be `true` for every IoU
        // threshold.
        for t in 0..n_t {
            assert!(!cell.dt_ignore[(t, 0)], "TP should not be dt_ignore");
            assert!(
                cell.dt_ignore[(t, 1)],
                "AA3: unmatched DT in not_exhaustive cell must be dt_ignore"
            );
        }
    }

    #[test]
    fn aa3_dt_ignore_only_unmatched() {
        // Mirror of the previous test but with `not_exhaustive` empty:
        // the same DT pair must produce `dt_ignore = false` on both
        // entries (the unmatched DT is now a real FP).
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "a")];
        let anns = vec![ann(1, 1, 1, (0.0, 0.0, 10.0, 10.0))];
        let gt = lvis_dataset(
            &images,
            &anns,
            &cats,
            &[(1, vec![])],
            &[(1, vec![])],
            &[(1, crate::Frequency::Frequent)],
        );
        let dts = CocoDetections::from_inputs(vec![
            dt_input(1, 1, 0.9, (0.0, 0.0, 10.0, 10.0)),
            dt_input(1, 1, 0.7, (50.0, 50.0, 10.0, 10.0)),
        ])
        .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        let grid = evaluate_bbox(&gt, &dts, params, ParityMode::Strict).unwrap();
        let cell = grid.cell(0, 0, 0).expect("cell must evaluate");
        assert!(cell.dt_ignore.iter().all(|&ig| !ig));
    }

    #[test]
    fn federated_dataset_with_use_cats_false_falls_back_to_coco() {
        // Federated logic requires `use_cats=true`. With `use_cats=false`
        // the L4 collapse merges every category into one bucket; we
        // explicitly skip the federated checks so a misconfigured
        // caller still sees deterministic COCO-grade output.
        let images = vec![img(1, 100, 100), img(2, 100, 100)];
        let cats = vec![cat(1, "a"), cat(2, "b")];
        let anns = vec![
            ann(1, 1, 1, (0.0, 0.0, 10.0, 10.0)),
            ann(2, 2, 2, (0.0, 0.0, 10.0, 10.0)),
        ];
        let gt = lvis_dataset(
            &images,
            &anns,
            &cats,
            &[(1, vec![]), (2, vec![])],
            &[(1, vec![]), (2, vec![])],
            &[
                (1, crate::Frequency::Frequent),
                (2, crate::Frequency::Frequent),
            ],
        );
        let dts = CocoDetections::from_inputs(vec![
            dt_input(1, 1, 0.9, (0.0, 0.0, 10.0, 10.0)),
            dt_input(1, 2, 0.7, (50.0, 50.0, 10.0, 10.0)),
        ])
        .unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: false,
            retain_iou: false,
        };
        // No panic, no skipped cell — the K-axis is collapsed to one
        // sentinel category so AA4 cannot apply.
        let grid = evaluate_bbox(&gt, &dts, params, ParityMode::Strict).unwrap();
        assert_eq!(grid.n_categories, 1);
        // (k=0, a=0, i=0) is the only image-1 cell; it must contain
        // both DTs (cat 1 and cat 2 collapsed onto k=0).
        let cell = grid.cell(0, 0, 0).expect("collapsed cell must evaluate");
        assert_eq!(cell.dt_scores.len(), 2);
    }

    #[test]
    fn coco_dataset_unaffected_by_federated_machinery() {
        // The federated branches must be no-ops when
        // `is_federated()` is false. Pin this with a regression check
        // against the perfect_match_grid fixture: the cell shape
        // must be byte-identical to what the function returned
        // before the AA3/AA4 patch.
        let g = perfect_match_grid();
        // 1 category, 4 area ranges, 1 image. (k=0, a=0, i=0) holds
        // the all-area cell with both DTs matched.
        let cell = g.cell(0, 0, 0).expect("perfect_match cell must exist");
        assert_eq!(cell.dt_scores.len(), 2);
        assert!(cell.dt_ignore.iter().all(|&ig| !ig));
    }
}
