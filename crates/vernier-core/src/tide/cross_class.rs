//! Cross-class IoU side pass.
//!
//! Per ADR-0023, TIDE's Cls and Both bins (and the sibling confusion-
//! matrix capability) need IoU between detections of class A and GTs of
//! some class B ≠ A. The matching engine only ever sees same-class
//! per-cell slices (the class restriction lives one level up in the
//! orchestrator); cross-class data therefore comes from a separate
//! side pass that calls the same [`crate::similarity::Similarity`]
//! kernel with un-class-filtered per-image GT and DT lists.
//!
//! The output is [`CrossClassIous`]: per-image dense IoU matrices with
//! parallel category-index vectors so the bin-assignment layer can
//! answer "what class is the best-overlapping GT?" without touching the
//! dataset again. The storage type itself lives in
//! [`crate::tables`] alongside [`crate::tables::RetainedIous`] (per
//! ADR-0023's pointer to `tables.rs`); this module owns the side-pass
//! driver.
//!
//! Matching is unchanged. The ADR-0005 invariant ("matching is generic
//! over the matrix only") is preserved verbatim.

use std::collections::HashMap;

use ndarray::Array2;

use crate::dataset::{CategoryId, CocoDataset, CocoDetections, EvalDataset, ImageMeta};
use crate::error::EvalError;
use crate::evaluate::{dt_top_indices_for_cell, EvalKernel};
use crate::parity::ParityMode;
use crate::tables::CrossClassIous;

/// Compute the cross-class IoU side pass over every image in the
/// dataset.
///
/// Walks images in id-ascending order (the same ordering
/// [`crate::evaluate_with`] uses for its `I` axis), gathers the
/// un-class-filtered per-image GT and DT lists, applies the same
/// `max_dets_per_image` cap on the DT side as the matching path
/// (with the stable score-descending tie-break from quirk **A1**),
/// and calls the kernel once per image to fill the dense
/// `(D_total, G_total)` matrix.
///
/// The caller-provided `max_dets_per_image` should be the same value
/// passed to [`crate::evaluate_with`] so the DT row indexing
/// downstream consumers see lines up across the two passes. Per
/// ADR-0023 the side pass *materializes the full matrix*; the
/// `cross_class_topk` escape hatch lives on a future refinement and
/// is not implemented here.
///
/// # Errors
///
/// Propagates [`EvalError`] from the kernel's
/// [`EvalKernel::build_gt_anns`] / [`EvalKernel::build_dt_anns`] and
/// from [`crate::similarity::Similarity::compute`]. Returns
/// [`EvalError::InvalidAnnotation`] when an annotation references a
/// category id absent from the GT dataset's category list.
pub fn compute_cross_class_ious<K: EvalKernel>(
    gt: &CocoDataset,
    dt: &CocoDetections,
    kernel: &K,
    parity_mode: ParityMode,
    max_dets_per_image: usize,
) -> Result<CrossClassIous, EvalError> {
    let mut images: Vec<&ImageMeta> = gt.images().iter().collect();
    images.sort_unstable_by_key(|im| im.id.0);

    // Category-id → category-index map matching evaluate_with's K-axis
    // ordering (id-ascending). Indexing rather than the raw id keeps
    // the bin-assignment layer reasoning in the same coordinate
    // system as the rest of the spine.
    let mut category_ids: Vec<CategoryId> = gt.categories().iter().map(|c| c.id).collect();
    category_ids.sort_unstable_by_key(|c| c.0);
    let category_index: HashMap<CategoryId, usize> = category_ids
        .iter()
        .enumerate()
        .map(|(idx, id)| (*id, idx))
        .collect();

    let gt_anns = gt.annotations();
    let dt_anns = dt.detections();
    let mut store = CrossClassIous::new();

    for (image_idx, image) in images.iter().enumerate() {
        let image_id = image.id;
        let gt_indices = gt.ann_indices_for_image(image_id);
        // DT side: A1 stable score-descending sort + max_dets cap,
        // shared with the matching path so rows of the side-pass
        // matrix line up with the post-cap DT slice the matching
        // engine saw. `cat = None` selects every category on the image.
        let dt_indices = dt_top_indices_for_cell(dt, image_id, None, max_dets_per_image);

        if gt_indices.is_empty() && dt_indices.is_empty() {
            continue;
        }

        let dt_classes: Vec<usize> = dt_indices
            .iter()
            .map(|&i| {
                let cat = dt_anns[i].category_id;
                lookup_category_index(&category_index, cat, "DT", dt_anns[i].id.0, image_id.0)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let gt_classes: Vec<usize> = gt_indices
            .iter()
            .map(|&i| {
                let cat = gt_anns[i].category_id;
                lookup_category_index(&category_index, cat, "GT", gt_anns[i].id.0, image_id.0)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let gt_kernel = kernel.build_gt_anns(gt_anns, gt_indices, image)?;
        let dt_kernel = kernel.build_dt_anns(dt_anns, &dt_indices, image, parity_mode)?;

        let mut iou = Array2::<f64>::zeros((gt_kernel.len(), dt_kernel.len()));
        if !gt_kernel.is_empty() && !dt_kernel.is_empty() {
            kernel.compute(&gt_kernel, &dt_kernel, &mut iou.view_mut())?;
        }

        // Storage convention is (D_total, G_total) per ADR-0023, so
        // transpose the (G, D) matrix the kernel writes. The reversed
        // shape matches the bin-assignment layer's per-DT iteration.
        let iou_dg = iou.reversed_axes().to_owned();
        store.insert(image_idx, iou_dg, dt_classes, gt_classes);
    }

    Ok(store)
}

fn lookup_category_index(
    map: &HashMap<CategoryId, usize>,
    cat: CategoryId,
    kind: &str,
    ann_id: i64,
    image_id: i64,
) -> Result<usize, EvalError> {
    map.get(&cat)
        .copied()
        .ok_or_else(|| EvalError::InvalidAnnotation {
            detail: format!(
                "{kind} id={ann_id} on image {image_id} references category_id={} \
                 not present in the GT dataset's category list; the cross-class \
                 IoU side pass needs every annotation's category to be known.",
                cat.0
            ),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{
        AnnId, Bbox, CategoryMeta, CocoAnnotation, DetectionInput, ImageId, ImageMeta,
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

    #[test]
    fn cross_class_iou_matches_axis_aligned_overlap_on_two_class_image() {
        // One image, two categories, two GTs, two DTs. The two DTs and
        // two GTs are colocated in pairs by class but the side pass
        // computes the full 2x2 cross-class matrix — we assert both
        // the same-class diagonal (perfect overlap) and one off-
        // diagonal cross-class IoU we can compute by hand.
        //
        //   GT_A: bbox (0, 0, 10, 10), category 1
        //   GT_B: bbox (5, 0, 10, 10), category 2
        //   DT_A: bbox (0, 0, 10, 10), category 1, score 0.9
        //   DT_B: bbox (5, 0, 10, 10), category 2, score 0.8
        //
        // Same-class IoUs: DT_A vs GT_A = 1.0, DT_B vs GT_B = 1.0.
        // Cross-class IoUs:
        //   DT_A (cat 1) vs GT_B (cat 2): boxes (0,0,10,10) and
        //     (5,0,10,10) overlap in (5,0,5,10) = area 50; union =
        //     100 + 100 − 50 = 150; IoU = 50/150 = 1/3.
        //   DT_B (cat 2) vs GT_A (cat 1): symmetric, IoU = 1/3.
        let images = vec![img(1, 100, 100)];
        let cats = vec![cat(1, "a"), cat(2, "b")];
        let anns = vec![
            ann(1, 1, 1, (0.0, 0.0, 10.0, 10.0)),
            ann(2, 1, 2, (5.0, 0.0, 10.0, 10.0)),
        ];
        let gt = CocoDataset::from_parts(images, anns, cats).expect("dataset builds");
        let dts = CocoDetections::from_inputs(vec![
            dt_input(1, 1, 0.9, (0.0, 0.0, 10.0, 10.0)),
            dt_input(1, 2, 0.8, (5.0, 0.0, 10.0, 10.0)),
        ])
        .expect("detections build");

        let store = compute_cross_class_ious(&gt, &dts, &BboxIou, ParityMode::Strict, 100)
            .expect("side pass runs");

        assert_eq!(store.len(), 1);
        let iou = store.get(0).expect("image 0 retained");
        // (D_total, G_total) per the storage convention.
        assert_eq!(iou.shape(), &[2, 2]);

        let dt_classes = store.dt_classes(0).expect("dt_classes present");
        let gt_classes = store.gt_classes(0).expect("gt_classes present");
        // Categories sorted id-ascending → id 1 → idx 0, id 2 → idx 1.
        // DT order is score-descending, so DT_A (0.9, cat 1) is row 0
        // and DT_B (0.8, cat 2) is row 1. GTs are in dataset insertion
        // order → GT_A (cat 1) is col 0 and GT_B (cat 2) is col 1.
        assert_eq!(dt_classes, &[0, 1]);
        assert_eq!(gt_classes, &[0, 1]);

        // Same-class diagonals — perfect overlap.
        let eps = 1e-12;
        assert!((iou[(0, 0)] - 1.0).abs() < eps, "DT_A vs GT_A");
        assert!((iou[(1, 1)] - 1.0).abs() < eps, "DT_B vs GT_B");
        // Cross-class off-diagonals — the hand-computed 1/3.
        let one_third = 1.0 / 3.0;
        assert!(
            (iou[(0, 1)] - one_third).abs() < eps,
            "DT_A vs GT_B: got {}, expected 1/3",
            iou[(0, 1)]
        );
        assert!(
            (iou[(1, 0)] - one_third).abs() < eps,
            "DT_B vs GT_A: got {}, expected 1/3",
            iou[(1, 0)]
        );
    }
}
