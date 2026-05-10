//! Confusion matrix sibling capability of TIDE error decomposition.
//!
//! Per ADR-0023, the same cross-class IoU side pass that powers the
//! TIDE Cls / Both bins also funds the confusion-matrix output: per-DT,
//! the best-overlapping GT across all classes is exactly the input the
//! confusion-matrix needs to count `(true_class, predicted_class)`
//! pairs. One pass through [`crate::tide::compute_cross_class_ious`]
//! serves both consumers.
//!
//! Output shape — counts keyed by `(Option<usize>, Option<usize>)` over
//! the **category-index** space (id-ascending; the same coordinate
//! system the cross-class side pass uses, *not* the raw COCO category
//! ids). The `None` sentinel maps to the FFI's `"__none__"` row /
//! column at the Python boundary:
//!
//! - `(Some(gt), Some(dt))` with `gt == dt` — true positives.
//! - `(Some(gt), Some(dt))` with `gt != dt` — classification confusion.
//! - `(None, Some(dt))` — the DT had no GT at IoU ≥ `iou_threshold`
//!   on the same image; counts as a false positive.
//! - `(Some(gt), None)` — the GT was not covered by any DT at the
//!   threshold; counts as a missed GT (false negative).
//!
//! The matching engine (ADR-0005) is unchanged; this module computes
//! its own argmax over the side-pass matrix rather than reusing the
//! same-class matching path. The shape is genuinely different — every
//! DT is compared against every GT on the image regardless of class —
//! and the matching engine's per-cell same-class invariant would have
//! to be peeled apart to model it.
//!
//! ## GT-side ignore handling
//!
//! pycocotools treats `iscrowd=1` (and the optional `ignore` field)
//! GTs as silent: a DT matched to one is dropped from the FP/TP count;
//! an unmatched ignore-GT is **not** a missed GT. Per ADR-0023's
//! recommendation, the side-pass storage [`crate::tables::CrossClassIous`] only
//! carries category indices and an opaque `(D, G)` matrix — no
//! per-column ignore flag — so this module recomputes the per-image
//! GT-annotation indices via [`crate::dataset::EvalDataset::ann_indices_for_image`]
//! and reads the original [`crate::dataset::CocoAnnotation::is_crowd`]
//! / [`crate::dataset::CocoAnnotation::ignore_flag`] to decide. Cheap
//! recompute over a small per-image list; keeps the storage type
//! single-purpose.

use std::collections::{HashMap, HashSet};

use crate::dataset::{CocoDataset, CocoDetections, EvalDataset, ImageMeta};
use crate::error::EvalError;
use crate::evaluate::EvalKernel;
use crate::parity::ParityMode;
use crate::tide::cross_class::compute_cross_class_ious;

/// Aggregated confusion-matrix counts across an entire dataset.
///
/// Indices in [`Self::counts`]'s keys reference [`Self::category_ids`]
/// — both the row (GT) and column (DT) indices live in the
/// id-ascending category-index space the cross-class side pass uses.
/// `None` is the sentinel for "no class" — false-positive row when in
/// the GT slot, missed-GT column when in the DT slot. The FFI surfaces
/// the sentinel as the literal string `"__none__"`.
#[derive(Debug, Clone, Default)]
pub struct ConfusionMatrixCounts {
    /// `(gt_category_idx_or_none, dt_category_idx_or_none) -> count`.
    /// Pairs absent from the map have count zero (the FFI fills in
    /// the dense long-format only for pairs that fired).
    pub counts: HashMap<(Option<usize>, Option<usize>), u64>,
    /// Category ids in the same index space the matrix uses.
    /// `category_ids[i]` is the user-visible COCO category id for
    /// matrix index `i`.
    pub category_ids: Vec<i64>,
}

/// Threshold used to decide whether a DT covers a GT. The DT's
/// best-overlapping GT (across all classes) at IoU ≥ `iou_threshold`
/// is the matched pair; anything below the threshold counts as a
/// false positive and the GT (if not yet covered by a higher-scoring
/// DT) eventually counts as missed.
///
/// `max_dets_per_image` matches the matching path's per-image cap so
/// the rows of the side-pass matrix this function reads line up with
/// the post-cap DT slice the matching engine saw on the same dataset.
///
/// # Errors
///
/// Propagates [`EvalError`] from [`compute_cross_class_ious`] —
/// kernel construction failures and category-id-not-found errors flow
/// through unchanged.
pub fn compute_confusion_matrix<K: EvalKernel>(
    gt: &CocoDataset,
    dt: &CocoDetections,
    kernel: &K,
    iou_threshold: f64,
    max_dets_per_image: usize,
    parity_mode: ParityMode,
) -> Result<ConfusionMatrixCounts, EvalError> {
    let cross = compute_cross_class_ious(gt, dt, kernel, parity_mode, max_dets_per_image)?;

    // Category-id list in id-ascending order — same axis the side
    // pass keys on. `category_ids[i]` is the COCO id at matrix index
    // `i`; consumers use this to map from `Some(idx)` back to a class
    // string at the FFI boundary.
    let mut category_ids: Vec<i64> = gt.categories().iter().map(|c| c.id.0).collect();
    category_ids.sort_unstable();

    let mut images: Vec<&ImageMeta> = gt.images().iter().collect();
    images.sort_unstable_by_key(|im| im.id.0);

    let gt_anns = gt.annotations();
    let mut counts: HashMap<(Option<usize>, Option<usize>), u64> = HashMap::new();

    for (image_idx, image) in images.iter().enumerate() {
        // Rebuild `gt_indices` for this image to read the ignore
        // flags. Storage-shape rationale lives in the module doc.
        let gt_indices = gt.ann_indices_for_image(image.id);

        // The side pass inserts iou + dt_classes + gt_classes
        // atomically per image; if the matrix is missing the image
        // is empty (no DTs and no GTs) and there is nothing to count.
        let (Some(iou), Some(dt_classes), Some(gt_classes)) = (
            cross.get(image_idx),
            cross.dt_classes(image_idx),
            cross.gt_classes(image_idx),
        ) else {
            continue;
        };

        let n_d = iou.shape()[0];
        let n_g = iou.shape()[1];

        // Per-image: which GT columns are already taken by some DT?
        // Walk DTs in row order, which is score-descending per the
        // side pass (and the matching path's A1 ordering). The first
        // DT to claim a GT keeps it; subsequent DTs ignore that
        // column when picking their argmax.
        let mut gt_taken: HashSet<usize> = HashSet::new();

        for d in 0..n_d {
            // argmax over G of iou[d, g] restricted to non-taken GTs.
            // No need to also restrict against ignore-GTs here: if a
            // DT's best overlap is an ignore-GT and the threshold
            // fires, pycocotools-style semantics say we **drop** the
            // DT (not count it as FP, not count the ignore-GT as
            // matched). We model that by skipping the count entirely.
            let mut best_g: Option<usize> = None;
            let mut best_iou = f64::NEG_INFINITY;
            for g in 0..n_g {
                if gt_taken.contains(&g) {
                    continue;
                }
                let v = iou[(d, g)];
                if v > best_iou {
                    best_iou = v;
                    best_g = Some(g);
                }
            }

            let dt_class_idx = dt_classes[d];

            if let Some(g) = best_g {
                if best_iou >= iou_threshold {
                    if is_ignore_gt(&gt_anns[gt_indices[g]]) {
                        // Match against an ignore-GT: drop the DT.
                        // Don't count as FP, don't mark the GT as
                        // covered (so a non-ignore DT later can't
                        // claim it — but ignore-GTs are excluded
                        // from the missed pass anyway, so this is
                        // moot for the missed-GT row).
                        continue;
                    }
                    gt_taken.insert(g);
                    *counts
                        .entry((Some(gt_classes[g]), Some(dt_class_idx)))
                        .or_insert(0) += 1;
                    continue;
                }
            }

            // Either no GT on this image at all, or best overlap
            // didn't clear the threshold → false positive in the
            // `__none__` row.
            *counts.entry((None, Some(dt_class_idx))).or_insert(0) += 1;
        }

        // After walking every DT, count missed (uncovered, non-ignore)
        // GTs in the `__none__` column.
        for (g, &gt_class_idx) in gt_classes.iter().enumerate() {
            if gt_taken.contains(&g) || is_ignore_gt(&gt_anns[gt_indices[g]]) {
                continue;
            }
            *counts.entry((Some(gt_class_idx), None)).or_insert(0) += 1;
        }
    }

    Ok(ConfusionMatrixCounts {
        counts,
        category_ids,
    })
}

/// Mirror the `effective_ignore` semantics from
/// `crate::tide::assignment` (D1): a GT is silent when `iscrowd=1` or
/// the optional `ignore` field is set. Inlined here rather than reaching
/// for [`crate::dataset::CocoAnnotation::effective_ignore`] because that
/// method takes a [`ParityMode`] and the confusion-matrix layer treats
/// strict and corrected identically — both fold iscrowd into ignore.
fn is_ignore_gt(ann: &crate::dataset::CocoAnnotation) -> bool {
    ann.is_crowd || ann.ignore_flag.unwrap_or(false)
}

#[cfg(test)]
fn category_index_for_id(counts: &ConfusionMatrixCounts, category_id: i64) -> Option<usize> {
    counts.category_ids.iter().position(|&id| id == category_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{
        AnnId, Bbox, CategoryId, CategoryMeta, CocoAnnotation, DetectionInput, ImageId, ImageMeta,
    };
    use crate::similarity::BboxIou;

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

    fn ann(
        id: i64,
        image: i64,
        cat: i64,
        bbox: (f64, f64, f64, f64),
        iscrowd: bool,
    ) -> CocoAnnotation {
        CocoAnnotation {
            id: AnnId(id),
            image_id: ImageId(image),
            category_id: CategoryId(cat),
            area: bbox.2 * bbox.3,
            is_crowd: iscrowd,
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

    /// Two GTs of distinct classes, two DTs perfectly aligned with
    /// their same-class GTs. Confusion matrix is diagonal-only.
    #[test]
    fn diagonal_only_when_every_dt_matches_same_class_gt() {
        let images = vec![img(1, 200, 200)];
        let cats = vec![cat(1, "a"), cat(2, "b")];
        let anns = vec![
            ann(1, 1, 1, (10.0, 10.0, 40.0, 40.0), false),
            ann(2, 1, 2, (100.0, 100.0, 40.0, 40.0), false),
        ];
        let gt = CocoDataset::from_parts(images, anns, cats).expect("dataset builds");
        let dts = CocoDetections::from_inputs(vec![
            dt_input(1, 1, 0.9, (10.0, 10.0, 40.0, 40.0)),
            dt_input(1, 2, 0.8, (100.0, 100.0, 40.0, 40.0)),
        ])
        .expect("detections build");

        let cm = compute_confusion_matrix(&gt, &dts, &BboxIou, 0.5, 100, ParityMode::Strict)
            .expect("confusion matrix runs");

        let idx_a = category_index_for_id(&cm, 1).expect("class 1 in matrix");
        let idx_b = category_index_for_id(&cm, 2).expect("class 2 in matrix");
        assert_eq!(cm.counts.get(&(Some(idx_a), Some(idx_a))), Some(&1));
        assert_eq!(cm.counts.get(&(Some(idx_b), Some(idx_b))), Some(&1));
        // No off-diagonal, no FP/Missed.
        assert_eq!(cm.counts.len(), 2);
    }

    /// Two GTs and two DTs, every DT wears the wrong class at the
    /// right location. Counts off-diagonal cells only.
    #[test]
    fn off_diagonal_when_every_dt_is_wrong_class() {
        let images = vec![img(1, 200, 200)];
        let cats = vec![cat(1, "a"), cat(2, "b")];
        let anns = vec![
            ann(1, 1, 1, (10.0, 10.0, 40.0, 40.0), false),
            ann(2, 1, 2, (100.0, 100.0, 40.0, 40.0), false),
        ];
        let gt = CocoDataset::from_parts(images, anns, cats).expect("dataset builds");
        let dts = CocoDetections::from_inputs(vec![
            dt_input(1, 2, 0.9, (10.0, 10.0, 40.0, 40.0)),
            dt_input(1, 1, 0.9, (100.0, 100.0, 40.0, 40.0)),
        ])
        .expect("detections build");

        let cm = compute_confusion_matrix(&gt, &dts, &BboxIou, 0.5, 100, ParityMode::Strict)
            .expect("confusion matrix runs");

        let idx_a = category_index_for_id(&cm, 1).expect("class 1");
        let idx_b = category_index_for_id(&cm, 2).expect("class 2");
        // GT class A → DT class B (DT 1 location matches GT 1 cat 1).
        assert_eq!(cm.counts.get(&(Some(idx_a), Some(idx_b))), Some(&1));
        assert_eq!(cm.counts.get(&(Some(idx_b), Some(idx_a))), Some(&1));
        // No diagonal, no FP/Missed.
        assert_eq!(cm.counts.len(), 2);
    }

    /// All DTs are background (no overlap). Two DTs with overlap
    /// (score 0.5) cover their same-class GTs; two DTs are pure
    /// background (no overlap, score 0.9). Verifies the FP row and
    /// the lack of missed GTs.
    #[test]
    fn fp_row_for_background_dts_and_no_missed_for_covered_gts() {
        let images = vec![img(1, 1000, 1000)];
        let cats = vec![cat(1, "a"), cat(2, "b")];
        let anns = vec![
            ann(1, 1, 1, (10.0, 10.0, 40.0, 40.0), false),
            ann(2, 1, 2, (100.0, 100.0, 40.0, 40.0), false),
        ];
        let gt = CocoDataset::from_parts(images, anns, cats).expect("dataset builds");
        // Two background DTs (high score, no overlap) plus two
        // covering DTs.
        let dts = CocoDetections::from_inputs(vec![
            dt_input(1, 1, 0.9, (500.0, 500.0, 30.0, 30.0)),
            dt_input(1, 2, 0.9, (600.0, 500.0, 30.0, 30.0)),
            dt_input(1, 1, 0.5, (10.0, 10.0, 40.0, 40.0)),
            dt_input(1, 2, 0.5, (100.0, 100.0, 40.0, 40.0)),
        ])
        .expect("detections build");

        let cm = compute_confusion_matrix(&gt, &dts, &BboxIou, 0.5, 100, ParityMode::Strict)
            .expect("confusion matrix runs");

        let idx_a = category_index_for_id(&cm, 1).expect("class 1");
        let idx_b = category_index_for_id(&cm, 2).expect("class 2");
        // Two FPs (the high-score background DTs).
        assert_eq!(cm.counts.get(&(None, Some(idx_a))), Some(&1));
        assert_eq!(cm.counts.get(&(None, Some(idx_b))), Some(&1));
        // Two covering DTs land on the diagonal.
        assert_eq!(cm.counts.get(&(Some(idx_a), Some(idx_a))), Some(&1));
        assert_eq!(cm.counts.get(&(Some(idx_b), Some(idx_b))), Some(&1));
        // No missed GTs (both covered).
        assert!(!cm.counts.contains_key(&(Some(idx_a), None)));
        assert!(!cm.counts.contains_key(&(Some(idx_b), None)));
    }

    /// All DTs are background AND no DT covers any GT → the FP row
    /// fires for the DTs, the Missed column fires for the GTs.
    #[test]
    fn fp_and_missed_when_dts_and_gts_dont_overlap_at_all() {
        let images = vec![img(1, 1000, 1000)];
        let cats = vec![cat(1, "a"), cat(2, "b")];
        let anns = vec![
            ann(1, 1, 1, (10.0, 10.0, 40.0, 40.0), false),
            ann(2, 1, 2, (100.0, 100.0, 40.0, 40.0), false),
        ];
        let gt = CocoDataset::from_parts(images, anns, cats).expect("dataset builds");
        let dts = CocoDetections::from_inputs(vec![
            dt_input(1, 1, 0.9, (500.0, 500.0, 30.0, 30.0)),
            dt_input(1, 2, 0.9, (600.0, 500.0, 30.0, 30.0)),
        ])
        .expect("detections build");

        let cm = compute_confusion_matrix(&gt, &dts, &BboxIou, 0.5, 100, ParityMode::Strict)
            .expect("confusion matrix runs");

        let idx_a = category_index_for_id(&cm, 1).expect("class 1");
        let idx_b = category_index_for_id(&cm, 2).expect("class 2");
        // FP row.
        assert_eq!(cm.counts.get(&(None, Some(idx_a))), Some(&1));
        assert_eq!(cm.counts.get(&(None, Some(idx_b))), Some(&1));
        // Missed column.
        assert_eq!(cm.counts.get(&(Some(idx_a), None)), Some(&1));
        assert_eq!(cm.counts.get(&(Some(idx_b), None)), Some(&1));
    }

    /// One iscrowd GT (image 1) and one regular GT (image 2). A DT
    /// landing on the crowd is dropped (no FP, no TP); the regular
    /// GT is matched. No missed-GT count for the crowd.
    #[test]
    fn ignore_gt_neither_matched_nor_missed() {
        let images = vec![img(1, 1000, 1000), img(2, 1000, 1000)];
        let cats = vec![cat(1, "a")];
        let anns = vec![
            ann(1, 1, 1, (10.0, 10.0, 40.0, 40.0), true), // iscrowd
            ann(2, 2, 1, (10.0, 10.0, 40.0, 40.0), false),
        ];
        let gt = CocoDataset::from_parts(images, anns, cats).expect("dataset builds");
        let dts = CocoDetections::from_inputs(vec![
            // Lands on the crowd → dropped.
            dt_input(1, 1, 0.9, (10.0, 10.0, 40.0, 40.0)),
            // Pure FP on image 1.
            dt_input(1, 1, 0.5, (500.0, 500.0, 30.0, 30.0)),
            // Covers the regular GT on image 2.
            dt_input(2, 1, 0.9, (10.0, 10.0, 40.0, 40.0)),
        ])
        .expect("detections build");

        let cm = compute_confusion_matrix(&gt, &dts, &BboxIou, 0.5, 100, ParityMode::Strict)
            .expect("confusion matrix runs");

        let idx_a = category_index_for_id(&cm, 1).expect("class 1");
        // Diagonal — covered regular GT.
        assert_eq!(cm.counts.get(&(Some(idx_a), Some(idx_a))), Some(&1));
        // FP — the second DT on image 1.
        assert_eq!(cm.counts.get(&(None, Some(idx_a))), Some(&1));
        // No missed-GT for the crowd.
        assert!(!cm.counts.contains_key(&(Some(idx_a), None)));
        // The crowd-matching DT is silent — no entry was created for it.
        assert_eq!(cm.counts.values().sum::<u64>(), 2);
    }
}
