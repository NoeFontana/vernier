//! Per-image PQ kernel: intersection histogram, IoU, matching loop.
//!
//! Implements the `pq_compute_single_core` body from
//! `panopticapi/evaluation.py:117-163` in pure Rust. Two structural
//! deltas vs upstream:
//!
//! - **T2 aligned.** The intersection histogram is built directly into
//!   a `HashMap<(u32, u32), u32>` in one pass over the GT/DT label-map
//!   slices. panopticapi materializes a uint64 buffer
//!   (`pan_gt * OFFSET + pan_pred`) and decodes it via `np.unique`;
//!   vernier's hashmap output is bit-equal but skips the uint64
//!   intermediate (no allocation of an `H * W * 8`-byte buffer).
//!
//! - **U1 aligned.** The matching loop iterates the histogram in a
//!   sorted `(gt, pred)` order rather than dict-iteration order. By
//!   quirk **U9** this is irrelevant — once a `(gt, pred)` pair clears
//!   the IoU > 0.5 threshold, both labels go into `gt_matched` /
//!   `pred_matched` and no other pair can re-match either side. The
//!   sorted iteration produces a deterministic TP set under property
//!   tests across input shuffles.
//!
//! FP / FN attribution lives in [`crate::attribute`]; this module
//! emits a [`PqImageReport`] consumed there.

use std::collections::HashMap;

use crate::dataset::{CategoryId, ImageEntry, ImageId};
use crate::error::PanopticError;
use crate::{PANOPTIC_IOU_THRESHOLD, PANOPTIC_VOID};

/// Per-image kernel output: the matched TP pairs (with their IoUs)
/// and the GT/DT segments that did not match. Consumed by
/// [`crate::attribute::attribute_image`] which folds in V1–V7 to
/// produce final per-category counts.
#[derive(Debug, Clone)]
pub struct PqImageReport {
    /// COCO image id.
    pub image_id: ImageId,
    /// One entry per matched TP — `(gt_id, dt_id, category_id, iou)`.
    /// IoU is the panoptic-union form (U6); category is the agreed
    /// category (U5 ensures GT and DT match).
    pub tp_pairs: Vec<TpPair>,
    /// GT segment ids that did not match any DT (for the FN/crowd
    /// classification step in [`crate::attribute`]).
    pub unmatched_gt: Vec<u32>,
    /// DT segment ids that did not match any GT (for the
    /// FP/V4-exclusion step in [`crate::attribute`]).
    pub unmatched_dt: Vec<u32>,
    /// Sparse intersection map keyed by `(gt_id, dt_id)`. Read by the
    /// attribute step for the V4 FP-exclusion test
    /// (`hist[(VOID, dt)] + same_cat_crowd_overlap > 0.5 * dt_area`).
    pub intersections: HashMap<(u32, u32), u32>,
}

/// One matched TP pair with the metadata needed by the summarize
/// layer (per-category fold of `sum_iou` and TP counts).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TpPair {
    /// GT segment id matched in this pair.
    pub gt_id: u32,
    /// DT segment id matched in this pair.
    pub dt_id: u32,
    /// Category id (GT and DT agree per U5).
    pub category_id: CategoryId,
    /// IoU computed under the panoptic-union form (U6).
    pub iou: f64,
}

/// Run the per-image PQ kernel. Returns the matched-pair set + the
/// unmatched-segment lists; the attribute step consumes both to fold
/// FP/FN per category.
///
/// `pred_areas_recomputed` denotes whether [`ImageEntry::recompute_areas_from_png`]
/// has been called on the DT side already — quirk **S3** demands the
/// pred area be the PNG marginal (the JSON value is silently
/// overwritten upstream). Callers that route through
/// [`from_arrays`](crate) with raw user data must invoke the recompute
/// before calling this kernel; `from_files_bytes` is expected to do
/// so as part of PNG decode. The kernel itself does not recompute
/// because doing so on every call would walk the label map twice on
/// the hot path.
pub fn pq_image(gt: &ImageEntry, dt: &ImageEntry) -> Result<PqImageReport, PanopticError> {
    if gt.height != dt.height || gt.width != dt.width {
        return Err(PanopticError::ShapeMismatch {
            image_id: 0,
            gt_shape: (gt.height, gt.width),
            dt_shape: (dt.height, dt.width),
        });
    }
    pq_image_with_id(0, gt, dt)
}

/// Variant of [`pq_image`] that carries the offending image id so a
/// shape-mismatch error attributes correctly. The orchestrator threads
/// the GT image id in.
pub fn pq_image_with_id(
    image_id: ImageId,
    gt: &ImageEntry,
    dt: &ImageEntry,
) -> Result<PqImageReport, PanopticError> {
    if gt.height != dt.height || gt.width != dt.width {
        return Err(PanopticError::ShapeMismatch {
            image_id,
            gt_shape: (gt.height, gt.width),
            dt_shape: (dt.height, dt.width),
        });
    }

    let intersections = build_intersection_histogram(&gt.label_map, &dt.label_map);

    let mut sorted: Vec<((u32, u32), u32)> = intersections.iter().map(|(&k, &v)| (k, v)).collect();
    sorted.sort_unstable_by_key(|&(k, _)| k);

    let mut tp_pairs: Vec<TpPair> = Vec::new();
    let mut gt_matched: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut dt_matched: std::collections::HashSet<u32> = std::collections::HashSet::new();

    for ((gt_id, dt_id), intersection) in sorted {
        let gt_seg = match gt.segments.get(&gt_id) {
            Some(s) => s,
            None => continue, // U2: catches gt_id == VOID
        };
        let dt_seg = match dt.segments.get(&dt_id) {
            Some(s) => s,
            None => continue, // U3: dt_id == VOID is impossible if S1 was enforced
        };
        if gt_seg.iscrowd {
            continue; // U4: crowd GT cannot be a TP
        }
        if gt_seg.category_id != dt_seg.category_id {
            continue; // U5: category disagreement
        }

        let void_overlap_with_dt = intersections
            .get(&(PANOPTIC_VOID, dt_id))
            .copied()
            .unwrap_or(0);
        let union = (gt_seg.area + dt_seg.area)
            .saturating_sub(intersection as u64)
            .saturating_sub(void_overlap_with_dt as u64);
        if union == 0 {
            continue; // pathological; would be NaN under U8
        }
        let iou = (intersection as f64) / (union as f64);
        if iou > PANOPTIC_IOU_THRESHOLD {
            tp_pairs.push(TpPair {
                gt_id,
                dt_id,
                category_id: gt_seg.category_id,
                iou,
            });
            gt_matched.insert(gt_id);
            dt_matched.insert(dt_id);
        }
    }

    // Sort unmatched ids so the downstream FP/FN attribution
    // (crowd_labels_dict last-wins under V3 strict, FP enumeration
    // order) is deterministic. Without this the HashMap iteration
    // order leaks into strict-mode reproduction.
    let mut unmatched_gt: Vec<u32> = gt
        .segments
        .keys()
        .copied()
        .filter(|id| !gt_matched.contains(id))
        .collect();
    unmatched_gt.sort_unstable();
    let mut unmatched_dt: Vec<u32> = dt
        .segments
        .keys()
        .copied()
        .filter(|id| !dt_matched.contains(id))
        .collect();
    unmatched_dt.sort_unstable();

    Ok(PqImageReport {
        image_id,
        tp_pairs,
        unmatched_gt,
        unmatched_dt,
        intersections,
    })
}

/// Build the `(gt_id, dt_id) -> intersection_pixels` histogram in one
/// pass over the parallel label maps. Quirk **T2** aligned: bit-equal
/// to panopticapi's `np.unique` over the OFFSET-encoded uint64
/// buffer, but allocates only the resulting hashmap (no uint64
/// intermediate).
///
/// `gt` and `dt` must have the same length; the orchestrator validates
/// this via [`pq_image_with_id`]'s `ShapeMismatch` check before
/// reaching here.
fn build_intersection_histogram(gt: &[u32], dt: &[u32]) -> HashMap<(u32, u32), u32> {
    debug_assert_eq!(gt.len(), dt.len());
    let mut out: HashMap<(u32, u32), u32> = HashMap::with_capacity(gt.len() / 16 + 16);
    for (g, d) in gt.iter().zip(dt.iter()) {
        *out.entry((*g, *d)).or_insert(0) += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::SegmentInfo;
    use std::collections::HashMap;

    fn entry(
        height: u32,
        width: u32,
        label_map: Vec<u32>,
        segments: &[(u32, CategoryId, bool, u64)],
    ) -> ImageEntry {
        let mut map = HashMap::new();
        for &(id, category_id, iscrowd, area) in segments {
            map.insert(
                id,
                SegmentInfo {
                    id,
                    category_id,
                    iscrowd,
                    area,
                },
            );
        }
        ImageEntry {
            height,
            width,
            label_map,
            segments: map,
        }
    }

    #[test]
    fn perfect_overlap_produces_one_tp() {
        // 2x2 image, single GT segment id=1, single DT segment id=10,
        // both cover all 4 pixels. Categories agree. IoU = 1.0.
        let gt = entry(2, 2, vec![1; 4], &[(1, 100, false, 4)]);
        let dt = entry(2, 2, vec![10; 4], &[(10, 100, false, 4)]);
        let report = pq_image_with_id(7, &gt, &dt).unwrap();
        assert_eq!(report.tp_pairs.len(), 1);
        assert_eq!(report.tp_pairs[0].gt_id, 1);
        assert_eq!(report.tp_pairs[0].dt_id, 10);
        assert_eq!(report.tp_pairs[0].category_id, 100);
        assert_eq!(report.tp_pairs[0].iou, 1.0);
        assert!(report.unmatched_gt.is_empty());
        assert!(report.unmatched_dt.is_empty());
    }

    #[test]
    fn shape_mismatch_returns_typed_error() {
        let gt = entry(2, 3, vec![1; 6], &[(1, 100, false, 6)]);
        let dt = entry(3, 2, vec![10; 6], &[(10, 100, false, 6)]);
        let err = pq_image_with_id(42, &gt, &dt).unwrap_err();
        match err {
            PanopticError::ShapeMismatch {
                image_id,
                gt_shape,
                dt_shape,
            } => {
                assert_eq!(image_id, 42);
                assert_eq!(gt_shape, (2, 3));
                assert_eq!(dt_shape, (3, 2));
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn category_disagreement_skips_pair() {
        // GT category 100, DT category 200. No TP, both unmatched.
        let gt = entry(2, 2, vec![1; 4], &[(1, 100, false, 4)]);
        let dt = entry(2, 2, vec![10; 4], &[(10, 200, false, 4)]);
        let report = pq_image_with_id(0, &gt, &dt).unwrap();
        assert!(report.tp_pairs.is_empty());
        assert_eq!(report.unmatched_gt, vec![1]);
        assert_eq!(report.unmatched_dt, vec![10]);
    }

    #[test]
    fn crowd_gt_cannot_be_tp_u4() {
        // Single GT marked iscrowd=true, perfect-overlapping DT.
        // U4: crowd GT is excluded from TP. Both end up unmatched
        // (the crowd FN-exclusion + V4 FP-exclusion happen in attribute.rs,
        // not here).
        let gt = entry(2, 2, vec![1; 4], &[(1, 100, true, 4)]);
        let dt = entry(2, 2, vec![10; 4], &[(10, 100, false, 4)]);
        let report = pq_image_with_id(0, &gt, &dt).unwrap();
        assert!(report.tp_pairs.is_empty());
        assert!(report.unmatched_gt.contains(&1));
        assert!(report.unmatched_dt.contains(&10));
    }

    #[test]
    fn iou_at_exactly_half_does_not_match_u7() {
        // Construct a pair with IoU exactly 1/2 in f64. The naive
        // attempt — single GT at one pixel, DT at two pixels with
        // VOID in between — fails because the U6 panoptic union
        // subtracts the (VOID, dt) intersection, which inflates the
        // IoU back to 1.0.
        //
        // Correct fixture: 1x2 image with GT id=1 (cat=100) at pixel
        // 0 and GT id=2 (cat=999) at pixel 1 (categories disagree
        // with DT, so this pair is skipped by U5 — keeps it out of
        // the candidate pool). DT id=10 (cat=100) at both pixels.
        // (gt=1, dt=10) intersection=1, dt_area=2, gt_area=1,
        // void_overlap=0, union=2, iou=1/2. U7 strict-greater rejects.
        let gt = entry(1, 2, vec![1, 2], &[(1, 100, false, 1), (2, 999, false, 1)]);
        let dt = entry(1, 2, vec![10, 10], &[(10, 100, false, 2)]);
        let report = pq_image_with_id(0, &gt, &dt).unwrap();
        assert!(report.tp_pairs.is_empty());
    }

    #[test]
    fn u9_property_one_dt_overlaps_two_gts() {
        // 2x4 image. GT: pixels [0..3] = id 1 (3 px), pixel 3 = id 2
        // (1 px). DT: all 4 pixels = id 10. So DT id 10 overlaps GT
        // id 1 by 3 px and GT id 2 by 1 px. Both candidate IoUs:
        //   (gt=1, dt=10): intersection=3, union = 3 + 4 - 3 = 4, iou = 3/4 = 0.75
        //   (gt=2, dt=10): intersection=1, union = 1 + 4 - 1 = 4, iou = 1/4 = 0.25
        // Only (gt=1, dt=10) clears U7. The (gt=2, dt=10) pair would
        // not match anyway by IoU; U9 ensures that even if both
        // cleared the threshold, only the first would match (and the
        // second would become FN).
        let gt = entry(
            1,
            4,
            vec![1, 1, 1, 2],
            &[(1, 100, false, 3), (2, 100, false, 1)],
        );
        let dt = entry(1, 4, vec![10; 4], &[(10, 100, false, 4)]);
        let report = pq_image_with_id(0, &gt, &dt).unwrap();
        assert_eq!(report.tp_pairs.len(), 1);
        assert_eq!(report.tp_pairs[0].gt_id, 1);
        assert!(report.unmatched_gt.contains(&2));
        assert!(report.unmatched_dt.is_empty()); // dt 10 matched
    }

    #[test]
    fn void_pixels_subtract_from_union_u6() {
        // 1x4 image. GT id=1 covers pixel 0 (1 px), pixels 1-3 are
        // VOID. DT id=10 covers all 4 pixels.
        // panopticapi union: gt.area + dt.area - intersection - hist[(VOID, dt)]
        //                  = 1 + 4 - 1 - 3 = 1
        // IoU = 1 / 1 = 1.0 → matches under U7.
        // (Without the VOID subtraction, union would be 4 and IoU = 0.25 → no match.)
        let gt = entry(1, 4, vec![1, 0, 0, 0], &[(1, 100, false, 1)]);
        let dt = entry(1, 4, vec![10; 4], &[(10, 100, false, 4)]);
        let report = pq_image_with_id(0, &gt, &dt).unwrap();
        assert_eq!(report.tp_pairs.len(), 1);
        assert_eq!(report.tp_pairs[0].iou, 1.0);
    }

    #[test]
    fn iteration_order_independence_property_u9() {
        // Build the same fixture but reverse the segment-id ordering
        // (in `segments` HashMap, which is order-independent already)
        // and the label_map pixel ordering. The TP set must be
        // identical.
        let gt_a = entry(
            1,
            4,
            vec![1, 1, 1, 2],
            &[(1, 100, false, 3), (2, 100, false, 1)],
        );
        let dt_a = entry(1, 4, vec![10, 10, 10, 10], &[(10, 100, false, 4)]);
        let report_a = pq_image_with_id(0, &gt_a, &dt_a).unwrap();

        // Pixel-shuffle: swap pixels 0 and 3 on both sides. GT now:
        // [2, 1, 1, 1] (gt id 2 at pos 0, gt id 1 at pos 1-3). DT
        // unchanged in label content. Intersections shift but the TP
        // set must still be {(1, 10)}.
        let gt_b = entry(
            1,
            4,
            vec![2, 1, 1, 1],
            &[(1, 100, false, 3), (2, 100, false, 1)],
        );
        let dt_b = entry(1, 4, vec![10, 10, 10, 10], &[(10, 100, false, 4)]);
        let report_b = pq_image_with_id(0, &gt_b, &dt_b).unwrap();

        assert_eq!(
            report_a
                .tp_pairs
                .iter()
                .map(|p| (p.gt_id, p.dt_id))
                .collect::<Vec<_>>(),
            report_b
                .tp_pairs
                .iter()
                .map(|p| (p.gt_id, p.dt_id))
                .collect::<Vec<_>>()
        );
    }
}
