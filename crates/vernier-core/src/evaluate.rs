//! Per-image bbox evaluation orchestrator.
//!
//! The bridge between the dataset layer ([`crate::CocoDataset`] /
//! [`crate::CocoDetections`]) and the IoU-type-agnostic spine
//! ([`crate::match_image`] → [`crate::accumulate`]). Pycocotools fuses
//! these in `evaluate()` (cocoeval.py 174-216); we keep the layers
//! separate so the spine stays untouchable per ADR-0005, and Phase 2/3
//! orchestrators (segm / keypoints) reuse the same shape with a
//! different [`crate::Similarity`] impl.
//!
//! ## What this layer does
//!
//! For each `(image, category)` cell:
//!
//! 1. Gather GTs and DTs from the dataset indices.
//! 2. Pre-filter DTs to the top `max_dets_per_image` by score (the
//!    matching engine and accumulator both rely on this cap; smaller
//!    `max_dets` values are sliced downstream by `accumulate`).
//! 3. Compute the GT × DT IoU matrix once via [`crate::BboxIou`].
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
//!   drives the E1 asymmetry inside [`crate::BboxIou`].
//! - **J3** (`strict`): DT areas are read from
//!   [`crate::CocoDetection::area`], which the dataset layer derives
//!   from the bbox at construction.

use ndarray::{Array2, ArrayView2};

use crate::accumulate::PerImageEval;
use crate::dataset::{CategoryId, CocoDataset, CocoDetections, EvalDataset, ImageId};
use crate::error::EvalError;
use crate::matching::{match_image, MatchResult};
use crate::parity::{argsort_score_desc, ParityMode};
use crate::similarity::{BboxAnn, BboxIou, Similarity};

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

/// Inputs to [`evaluate_bbox`]. Mirrors the small slice of pycocotools'
/// `Params` that actually feeds the matcher — the rest (recall
/// thresholds, full max-dets ladder) lives on [`crate::AccumulateParams`]
/// downstream.
#[derive(Debug, Clone, Copy)]
pub struct EvaluateBboxParams<'p> {
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
        if k >= self.n_categories || a >= self.n_area_ranges || i >= self.n_images {
            return None;
        }
        let idx = k * self.n_area_ranges * self.n_images + a * self.n_images + i;
        self.eval_imgs.get(idx).and_then(Option::as_ref)
    }
}

/// Run the per-image bbox evaluation pass.
///
/// Iterates `(image, category)` cells, computes the IoU matrix once per
/// cell with [`crate::BboxIou`], runs [`crate::match_image`] once per
/// area range, and packs the results into a flat
/// `[k][a][i]`-ordered grid suitable for [`crate::accumulate`].
///
/// # Errors
///
/// Propagates [`EvalError`] from the underlying [`crate::Similarity`]
/// and [`crate::match_image`] calls.
pub fn evaluate_bbox(
    gt: &CocoDataset,
    dt: &CocoDetections,
    params: EvaluateBboxParams<'_>,
    parity_mode: ParityMode,
) -> Result<EvalGrid, EvalError> {
    // Image and category ordering: id-ascending, deterministic across runs.
    let mut image_ids: Vec<ImageId> = gt.images().iter().map(|im| im.id).collect();
    image_ids.sort_unstable_by_key(|id| id.0);
    let n_i = image_ids.len();
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

    let bbox_iou = BboxIou;
    let mut eval_imgs: Vec<Option<PerImageEval>> = vec![None; n_k * n_a * n_i];

    for (k, cat) in category_buckets.iter().enumerate() {
        let nk = k * n_a * n_i;
        for (i, image_id) in image_ids.iter().enumerate() {
            let gt_indices = gt_indices_for_cell(gt, *image_id, *cat);
            let dt_indices =
                dt_top_indices_for_cell(dt, *image_id, *cat, params.max_dets_per_image);
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
            let dt_areas: Vec<f64> = dt_indices.iter().map(|&j| dt_anns[j].area).collect();
            let dt_scores: Vec<f64> = dt_indices.iter().map(|&j| dt_anns[j].score).collect();

            let gt_kernel: Vec<BboxAnn> = gt_indices
                .iter()
                .zip(&gt_iscrowd)
                .map(|(&j, &is_crowd)| BboxAnn {
                    bbox: gt_anns[j].bbox,
                    is_crowd,
                })
                .collect();
            // E2/J4: DT never carries crowd.
            let dt_kernel: Vec<BboxAnn> = dt_indices
                .iter()
                .map(|&j| BboxAnn {
                    bbox: dt_anns[j].bbox,
                    is_crowd: false,
                })
                .collect();

            let mut iou = Array2::<f64>::zeros((gt_kernel.len(), dt_kernel.len()));
            if !gt_kernel.is_empty() && !dt_kernel.is_empty() {
                bbox_iou.compute(&gt_kernel, &dt_kernel, &mut iou.view_mut())?;
            }

            let buffers = CellBuffers {
                gt_areas: &gt_areas,
                gt_iscrowd: &gt_iscrowd,
                gt_base_ignore: &gt_base_ignore,
                dt_areas: &dt_areas,
                dt_scores: &dt_scores,
                iou: iou.view(),
            };
            for (a, area) in params.area_ranges.iter().enumerate() {
                let cell = evaluate_cell(&buffers, area, params.iou_thresholds, parity_mode)?;
                eval_imgs[nk + a * n_i + i] = Some(cell);
            }
        }
    }

    Ok(EvalGrid {
        eval_imgs,
        n_categories: n_k,
        n_area_ranges: n_a,
        n_images: n_i,
    })
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
    gt_areas: &'a [f64],
    gt_iscrowd: &'a [bool],
    gt_base_ignore: &'a [bool],
    dt_areas: &'a [f64],
    dt_scores: &'a [f64],
    iou: ArrayView2<'a, f64>,
}

fn evaluate_cell(
    buf: &CellBuffers<'_>,
    area: &AreaRange,
    iou_thresholds: &[f64],
    parity_mode: ParityMode,
) -> Result<PerImageEval, EvalError> {
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
        dt_matches,
        mut dt_ignore,
        ..
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

    let dt_scores_sorted: Vec<f64> = dt_perm.iter().map(|&k| buf.dt_scores[k]).collect();
    let dt_in_range_sorted: Vec<bool> = dt_perm
        .iter()
        .map(|&k| area.contains(buf.dt_areas[k]))
        .collect();
    let gt_ignore_sorted: Vec<bool> = gt_perm.iter().map(|&k| gt_ignore[k]).collect();

    let mut dt_matched = Array2::<bool>::default((n_t, n_d));
    for t in 0..n_t {
        for d in 0..n_d {
            let matched = dt_matches[(t, d)] >= 0;
            dt_matched[(t, d)] = matched;
            // B7: unmatched AND out-of-area → ignore.
            if !matched && !dt_in_range_sorted[d] {
                dt_ignore[(t, d)] = true;
            }
        }
    }

    Ok(PerImageEval {
        dt_scores: dt_scores_sorted,
        dt_matched,
        dt_ignore,
        gt_ignore: gt_ignore_sorted,
    })
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
        let params = EvaluateBboxParams {
            iou_thresholds: iou_thresholds(),
            area_ranges: &area,
            max_dets_per_image: 100,
            use_cats: true,
        };
        evaluate_bbox(&gt, &dts, params, ParityMode::Strict).unwrap()
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
        let params = EvaluateBboxParams {
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
        let params = EvaluateBboxParams {
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
        let params = EvaluateBboxParams {
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
        let params = EvaluateBboxParams {
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
        let params = EvaluateBboxParams {
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
    fn missing_dt_image_yields_none_cells() {
        // Pycocotools' `evaluateImg` returns a record (not None) when
        // GTs exist but DTs do not — vernier matches that.
        let images = vec![img(1, 100, 100), img(2, 100, 100)];
        let cats = vec![cat(1, "thing")];
        let anns = vec![ann(1, 1, 1, (0.0, 0.0, 10.0, 10.0))];
        let gt = CocoDataset::from_parts(images, anns, cats).unwrap();
        let dts = CocoDetections::from_inputs(vec![]).unwrap();
        let area = AreaRange::coco_default();
        let params = EvaluateBboxParams {
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
}
