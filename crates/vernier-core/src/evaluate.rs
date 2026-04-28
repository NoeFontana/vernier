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
//! - **D6/D7** (`strict`): area filter uses strict `<` and `>` so a GT
//!   with area exactly equal to a boundary value is *kept* (its
//!   `_ignore` stays at the base value). Inequality direction matches
//!   the eval-time filter in pycocotools, *not* `getAnnIds(areaRng=...)`.
//! - **B7** (`strict`): unmatched DTs whose area is out of range get
//!   `dt_ignore=true` so they do not contribute to the precision/recall
//!   curve in this area cell.
//! - **L4** (`aligned`): `use_cats=false` collapses every category onto
//!   a single virtual `k=0` bucket, with `category_id` carried through
//!   matching as a no-op.
//! - **E2 / J4** (`strict`): DTs never carry an `is_crowd` flag — the
//!   [`crate::CocoDetection`] type lacks the field. Only GT crowdness
//!   drives the E1 asymmetry inside the kernel.
//! - **J3** (`strict`): DT areas are read from
//!   [`crate::CocoDetection::area`], which the dataset layer derives
//!   from the bbox at construction.

use ndarray::{Array2, ArrayView2};

use crate::accumulate::PerImageEval;
use crate::dataset::{
    CategoryId, CocoAnnotation, CocoDataset, CocoDetection, CocoDetections, EvalDataset, ImageId,
    ImageMeta,
};
use crate::error::EvalError;
use crate::matching::{match_image, MatchResult};
use crate::parity::{argsort_score_desc, ParityMode};
use crate::similarity::{BboxAnn, BboxIou, SegmAnn, SegmIou, Similarity};

/// Sentinel `category_id` emitted on every cell when `use_cats=false`.
/// Mirrors pycocotools' `p.catIds = [-1]` collapse (quirk **L4**).
pub const COLLAPSED_CATEGORY_SENTINEL: i64 = -1;

/// Sentinel upper bound for "unbounded" area buckets, mirroring the
/// `1e10` pycocotools uses for `all` / `large`.
pub const AREA_UNBOUNDED: f64 = 1e10;

/// Open `(lo, hi)` area bucket — both bounds are strict per quirks
/// **D6/D7**, so an annotation with area exactly equal to a bound is
/// excluded.
///
/// `index` is the position on the `Accumulated` A-axis the resulting
/// [`PerImageEval`] feeds into; matched at summarize time against
/// [`crate::AreaRng::index`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AreaRange {
    /// A-axis position. `0` is conventionally the `all` bucket, matching
    /// [`crate::AreaRng::ALL`].
    pub index: usize,
    /// Lower bound (exclusive — quirks D6/D7).
    pub lo: f64,
    /// Upper bound (exclusive — quirks D6/D7). Use [`AREA_UNBOUNDED`]
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

    fn contains(&self, area: f64) -> bool {
        area > self.lo && area < self.hi
    }
}

/// Inputs to [`evaluate_bbox`] / [`evaluate_segm`] / [`evaluate_with`].
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
    fn build_dt_anns(
        &self,
        dt_anns: &[CocoDetection],
        indices: &[usize],
        image: &ImageMeta,
    ) -> Result<Vec<Self::Annotation>, EvalError>;
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
                })
            })
            .collect()
    }

    fn build_dt_anns(
        &self,
        dt_anns: &[CocoDetection],
        indices: &[usize],
        image: &ImageMeta,
    ) -> Result<Vec<SegmAnn>, EvalError> {
        indices
            .iter()
            .map(|&j| {
                let dt = &dt_anns[j];
                let seg = dt
                    .segmentation
                    .as_ref()
                    .ok_or_else(|| missing_segmentation_err("DT", dt.id.0, image.id.0))?;
                Ok(SegmAnn {
                    rle: seg.to_rle(image.height, image.width)?,
                    is_crowd: false,
                })
            })
            .collect()
    }
}

fn missing_segmentation_err(kind: &str, ann_id: i64, image_id: i64) -> EvalError {
    EvalError::InvalidAnnotation {
        detail: format!(
            "{kind} id={ann_id} on image {image_id} has no `segmentation` field; \
             segm eval requires one on every entry"
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

/// Output of [`evaluate_bbox`] — the flat `(K, A, I)` grid of
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
/// Most callers want [`evaluate_bbox`] or [`evaluate_segm`]; this entry
/// point is exposed for downstream code that ships its own kernel.
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

            // Area-invariant per-cell buffers — built once, reused
            // across every area range.
            let gt_anns = gt.annotations();
            let dt_anns = dt.detections();
            let gt_areas: Vec<f64> = gt_indices.iter().map(|&j| gt_anns[j].area).collect();
            let gt_iscrowd: Vec<bool> = gt_indices.iter().map(|&j| gt_anns[j].is_crowd).collect();
            // D1: parity-mode fork lives on the annotation; pass through.
            let gt_base_ignore: Vec<bool> = gt_indices
                .iter()
                .map(|&j| gt_anns[j].effective_ignore(parity_mode))
                .collect();
            let gt_ids: Vec<i64> = gt_indices.iter().map(|&j| gt_anns[j].id.0).collect();
            let dt_areas: Vec<f64> = dt_indices.iter().map(|&j| dt_anns[j].area).collect();
            let dt_scores: Vec<f64> = dt_indices.iter().map(|&j| dt_anns[j].score).collect();
            let dt_ids: Vec<i64> = dt_indices.iter().map(|&j| dt_anns[j].id.0).collect();

            let gt_kernel = kernel.build_gt_anns(gt_anns, gt_indices, image)?;
            let dt_kernel = kernel.build_dt_anns(dt_anns, &dt_indices, image)?;

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
            };
            for (a, area) in params.area_ranges.iter().enumerate() {
                let (cell, meta) =
                    evaluate_cell(&buffers, area, params.iou_thresholds, parity_mode)?;
                let flat = nk + a * n_i + i;
                eval_imgs[flat] = Some(cell);
                eval_imgs_meta[flat] = Some(meta);
            }
        }
    }

    Ok(EvalGrid {
        eval_imgs,
        eval_imgs_meta,
        n_categories: n_k,
        n_area_ranges: n_a,
        n_images: n_i,
    })
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
/// Every GT and DT must carry a `segmentation` field; running segm eval
/// against bbox-only inputs raises
/// [`EvalError::InvalidAnnotation`] with the offending id, instead of
/// silently treating absent masks as empty (the disposition documented
/// alongside quirk **K3**).
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
    evaluate_with(gt, dt, params, parity_mode, &SegmIou)
}

fn gt_indices_for_cell(gt: &CocoDataset, image: ImageId, cat: Option<CategoryId>) -> &[usize] {
    match cat {
        Some(c) => gt.ann_indices_for(image, c),
        None => gt.ann_indices_for_image(image),
    }
}

fn dt_top_indices_for_cell(
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
            if !matched && !dt_in_range_sorted[d] {
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
    fn d6_d7_strict_inequality_keeps_boundary_areas() {
        // GT exactly at the small/medium boundary (32² = 1024). With a
        // medium range of (1024, 96²), strict `>` excludes the boundary.
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
        };
        let grid = evaluate_bbox(&gt, &dts, params, ParityMode::Strict).unwrap();
        // small (lo=0, hi=32²=1024): area 1024 fails `< 1024` → ignored.
        let small = grid.cell(0, 1, 0).unwrap();
        assert_eq!(small.gt_ignore, vec![true]);
        // medium (lo=1024, hi=96²=9216): area 1024 fails `> 1024` → ignored.
        let medium = grid.cell(0, 2, 0).unwrap();
        assert_eq!(medium.gt_ignore, vec![true]);
        // all (lo=0, hi=1e10): area 1024 lies inside.
        let all = grid.cell(0, 0, 0).unwrap();
        assert_eq!(all.gt_ignore, vec![false]);
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
        };
        let grid = evaluate_bbox(&gt, &dts, params, ParityMode::Strict).unwrap();
        for a in 0..4 {
            assert!(grid.cell(0, a, 0).is_some(), "image 1 area {a}");
            assert!(grid.cell(0, a, 1).is_none(), "image 2 area {a}");
        }
    }

    use crate::segmentation::Segmentation;

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
    fn segm_missing_dt_segmentation_surfaces_typed_error() {
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
        // DT without a segmentation field.
        let dts =
            CocoDetections::from_inputs(vec![dt_input(1, 1, 0.9, (0.0, 0.0, 10.0, 10.0))]).unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
        };
        let err = evaluate_segm(&gt, &dts, params, ParityMode::Strict).unwrap_err();
        assert!(matches!(err, EvalError::InvalidAnnotation { .. }));
    }
}
