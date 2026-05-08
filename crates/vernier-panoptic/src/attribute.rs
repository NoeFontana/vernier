//! FP / FN attribution: V1–V7 of the ADR-0025 quirks survey.
//!
//! Consumes a [`PqImageReport`] from [`crate::kernel`] and folds in
//! the per-image FP/FN counts that complete the per-category PQ
//! summary. The matching loop in `kernel.rs` only emits TP pairs and
//! the unmatched-segment lists; the FP-exclusion logic (V4) and the
//! crowd-FN policy (V2/V3) live here so the kernel stays a thin
//! "match by IoU" routine and the policy questions are isolated.
//!
//! The split mirrors panopticapi's evaluation.py structure:
//! `evaluation.py:121-138` is the matching loop (kernel),
//! `evaluation.py:140-163` is FP/FN attribution (this module).

use std::collections::HashMap;

use crate::dataset::{CategoryId, ImageEntry};
use crate::kernel::{PqImageReport, TpPair};
use crate::parity::ParityMode;
use crate::PANOPTIC_VOID;

/// Per-category cumulative panoptic stats (sum over images).
///
/// The four fields fold together into the per-category PQ/SQ/RQ
/// formulas at summarize time (W1):
/// `PQ_c = sum_iou_c / (n_tp_c + 0.5*n_fp_c + 0.5*n_fn_c)`,
/// `SQ_c = sum_iou_c / n_tp_c` (or 0 if `n_tp_c == 0`),
/// `RQ_c = n_tp_c / (n_tp_c + 0.5*n_fp_c + 0.5*n_fn_c)`.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct PqStat {
    /// Sum of TP IoUs (U10 — summed per category, averaged at summarize).
    pub sum_iou: f64,
    /// Number of true positives (matched pairs).
    pub n_tp: u64,
    /// Number of false positives (unmatched DT not excluded by V4).
    pub n_fp: u64,
    /// Number of false negatives (unmatched non-crowd GT, V1).
    pub n_fn: u64,
}

impl PqStat {
    /// In-place add two per-category accumulators. Used by the
    /// summarize layer to fold per-image reports into a single
    /// per-category map.
    pub fn add_assign(&mut self, other: &PqStat) {
        self.sum_iou += other.sum_iou;
        self.n_tp += other.n_tp;
        self.n_fp += other.n_fp;
        self.n_fn += other.n_fn;
    }
}

/// Apply V1–V7 FP/FN attribution to one image's [`PqImageReport`],
/// returning the per-category [`PqStat`] delta this image contributes.
///
/// Quirks honored:
/// - **V1.** Each unmatched non-crowd GT increments `fn_per_cat[gt_cat]`.
/// - **V2.** Unmatched crowd GT does *not* count as FN; it goes into
///   `crowd_labels_dict[category_id]` for V3 / V4.
/// - **V3.** Two same-category crowds on one image: `Strict` reproduces
///   panopticapi's last-wins behavior (`crowd_labels_dict[cat] =
///   gt_label`); `Corrected` (default) sums overlaps from *all*
///   same-category crowds. The corrected disposition is the one
///   load-bearing strict-vs-corrected divergence in this module.
/// - **V4.** Unmatched DT is excluded from FP if `void_overlap +
///   same_cat_crowd_overlap > 0.5 * dt_area` (strict greater-than,
///   matching U7's direction).
/// - **V5.** The crowd term in V4 reads from the same `intersections`
///   map populated by the kernel.
/// - **V6.** A V4-excluded prediction contributes neither TP nor FP.
/// - **V7.** Unmatched-and-not-excluded DT is FP-counted against the
///   *predicted* category, regardless of any cross-category GT
///   overlap. PQ has no cross-class confusion concept.
pub fn attribute_image(
    gt: &ImageEntry,
    dt: &ImageEntry,
    report: &PqImageReport,
    mode: ParityMode,
) -> HashMap<CategoryId, PqStat> {
    let mut acc: HashMap<CategoryId, PqStat> = HashMap::new();

    // TP fold (U10 sum, W1 division at summarize).
    for &TpPair {
        category_id, iou, ..
    } in &report.tp_pairs
    {
        let s = acc.entry(category_id).or_default();
        s.sum_iou += iou;
        s.n_tp += 1;
    }

    // V3 storage diverges by mode: strict keeps a single last-wins
    // `gt_id` per category (panopticapi `crowd_labels_dict[cat] =
    // gt_label`); corrected accumulates every same-category crowd
    // and sums their overlaps. Picking the storage shape upfront
    // skips the unused alternative.
    enum CrowdMap {
        Strict(HashMap<CategoryId, u32>),
        Corrected(HashMap<CategoryId, Vec<u32>>),
    }
    let mut crowd: CrowdMap = match mode {
        ParityMode::Strict => CrowdMap::Strict(HashMap::new()),
        ParityMode::Corrected => CrowdMap::Corrected(HashMap::new()),
    };

    // FN classification + crowd_labels_dict population (V1, V2, V3).
    for &gt_id in &report.unmatched_gt {
        let Some(seg) = gt.segments.get(&gt_id) else {
            continue;
        };
        if seg.iscrowd {
            // V2: crowd GT is excluded from FN.
            match &mut crowd {
                CrowdMap::Strict(m) => {
                    m.insert(seg.category_id, gt_id);
                }
                CrowdMap::Corrected(m) => m.entry(seg.category_id).or_default().push(gt_id),
            }
            continue;
        }
        // V1: unmatched non-crowd GT → FN.
        let s = acc.entry(seg.category_id).or_default();
        s.n_fn += 1;
    }

    // FP attribution (V4–V7).
    for &dt_id in &report.unmatched_dt {
        let Some(seg) = dt.segments.get(&dt_id) else {
            continue;
        };
        let void_overlap = report
            .intersections
            .get(&(PANOPTIC_VOID, dt_id))
            .copied()
            .unwrap_or(0) as u64;
        let lookup = |gt_id: u32| -> u64 {
            report
                .intersections
                .get(&(gt_id, dt_id))
                .copied()
                .unwrap_or(0) as u64
        };
        let crowd_overlap: u64 = match &crowd {
            CrowdMap::Strict(m) => m.get(&seg.category_id).copied().map(lookup).unwrap_or(0),
            CrowdMap::Corrected(m) => m
                .get(&seg.category_id)
                .map(|gt_ids| gt_ids.iter().copied().map(lookup).sum::<u64>())
                .unwrap_or(0),
        };
        // V4: strict greater-than. Equality → not excluded → counts as FP.
        if (void_overlap + crowd_overlap) * 2 > seg.area {
            // V6: V4-excluded predictions contribute neither TP nor FP.
            continue;
        }
        // V7: FP attributed to the *predicted* category.
        let s = acc.entry(seg.category_id).or_default();
        s.n_fp += 1;
    }

    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::SegmentInfo;
    use crate::kernel::pq_image_with_id;
    use rustc_hash::FxHashMap;

    fn entry(
        height: u32,
        width: u32,
        label_map: Vec<u32>,
        segments: &[(u32, CategoryId, bool, u64)],
    ) -> ImageEntry {
        let mut map = FxHashMap::default();
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
    fn unmatched_non_crowd_gt_becomes_fn_v1() {
        // GT id=1 cat=100, no DT covers it. Expected: 1 FN at cat=100.
        let gt = entry(1, 2, vec![1, 0], &[(1, 100, false, 1)]);
        let dt = entry(1, 2, vec![0, 0], &[]);
        let report = pq_image_with_id(0, &gt, &dt).unwrap();
        let acc = attribute_image(&gt, &dt, &report, ParityMode::Corrected);
        assert_eq!(acc.get(&100).map(|s| s.n_fn), Some(1));
        assert_eq!(acc.get(&100).map(|s| s.n_tp), Some(0));
    }

    #[test]
    fn unmatched_crowd_gt_does_not_become_fn_v2() {
        // GT id=1 cat=100 iscrowd=true, no DT. V2: crowd is excluded
        // from FN. Expected: no entry for cat=100 (or zero FN).
        let gt = entry(1, 2, vec![1, 0], &[(1, 100, true, 1)]);
        let dt = entry(1, 2, vec![0, 0], &[]);
        let report = pq_image_with_id(0, &gt, &dt).unwrap();
        let acc = attribute_image(&gt, &dt, &report, ParityMode::Corrected);
        assert_eq!(acc.get(&100).map(|s| s.n_fn).unwrap_or(0), 0);
    }

    #[test]
    fn unmatched_dt_becomes_fp_v7() {
        // DT id=10 cat=200, GT id=1 cat=100 covers the same 4 px, no
        // VOID, no crowd. Categories disagree so no TP. V7: FP
        // attributed to the predicted (cat=200) category. V1: GT also
        // counts as one FN at cat=100.
        let gt = entry(1, 4, vec![1, 1, 1, 1], &[(1, 100, false, 4)]);
        let dt = entry(1, 4, vec![10, 10, 10, 10], &[(10, 200, false, 4)]);
        let report = pq_image_with_id(0, &gt, &dt).unwrap();
        let acc = attribute_image(&gt, &dt, &report, ParityMode::Corrected);
        assert_eq!(acc.get(&200).map(|s| s.n_fp), Some(1));
        assert_eq!(acc.get(&100).map(|s| s.n_fn), Some(1));
    }

    #[test]
    fn dt_excluded_from_fp_when_void_overlap_dominates_v4() {
        // DT id=10 covers 4 pixels. GT covers 1 pixel as id=1, the
        // other 3 are VOID. Categories disagree (or no GT in this
        // category) so no TP. void_overlap = 3, dt_area = 4.
        // V4 strict greater-than: 3 * 2 = 6 > 4 → excluded → no FP.
        let gt = entry(1, 4, vec![1, 0, 0, 0], &[(1, 100, false, 1)]);
        let dt = entry(1, 4, vec![10, 10, 10, 10], &[(10, 200, false, 4)]);
        let report = pq_image_with_id(0, &gt, &dt).unwrap();
        let acc = attribute_image(&gt, &dt, &report, ParityMode::Corrected);
        // No FP at cat=200 because V4 excluded it.
        assert_eq!(acc.get(&200).map(|s| s.n_fp).unwrap_or(0), 0);
    }

    #[test]
    fn v4_strict_greater_than_at_exactly_half() {
        // void_overlap = 2, dt_area = 4. 2 * 2 = 4 == 4 → NOT
        // excluded under strict greater-than. So DT contributes 1 FP.
        let gt = entry(1, 4, vec![1, 1, 0, 0], &[(1, 100, false, 2)]);
        let dt = entry(1, 4, vec![10, 10, 10, 10], &[(10, 200, false, 4)]);
        let report = pq_image_with_id(0, &gt, &dt).unwrap();
        let acc = attribute_image(&gt, &dt, &report, ParityMode::Corrected);
        assert_eq!(acc.get(&200).map(|s| s.n_fp), Some(1));
    }

    #[test]
    fn v3_strict_vs_corrected_diverge_on_two_same_category_crowds() {
        // Two same-category crowds on one image, sized so that the
        // sum of their overlaps with an unmatched DT is enough to
        // V4-exclude it but neither one alone is.
        //
        // 1x10 image. GT crowd id=1 (cat=100, area=2) covers [0..2];
        // GT crowd id=2 (cat=100, area=4) covers [2..6]; GT id=3
        // (NOT crowd, cat=999, area=4) covers [6..10]. DT id=10
        // (cat=100, area=10) covers all 10 px.
        //
        // Categories agree only between GT crowds and DT — but U4
        // skips crowd matches, so no TP.
        // GT 3 is unmatched non-crowd → V1 FN at cat=999 (both modes).
        //
        // FP analysis for unmatched DT 10 (cat=100):
        //   void_overlap = 0; crowd ∩ DT: id1=2, id2=4 (both cat=100).
        //   Strict: same_cat_crowd_overlap = max(2, 4) (sort makes
        //     last-wins = max). 2*(0 + 4) = 8 ≤ 10 → NOT excluded
        //     → FP=1. (The lower-id crowd would also give 2*(0+2)=4 ≤
        //     10, so the strict result is FP=1 regardless of which
        //     crowd wins the last-wins race.)
        //   Corrected: same_cat_crowd_overlap = 2 + 4 = 6.
        //     2*(0 + 6) = 12 > 10 → EXCLUDED → FP=0.
        let gt = entry(
            1,
            10,
            vec![1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            &[(1, 100, true, 2), (2, 100, true, 4), (3, 999, false, 4)],
        );
        let dt = entry(1, 10, vec![10; 10], &[(10, 100, false, 10)]);
        let report = pq_image_with_id(0, &gt, &dt).unwrap();

        let acc_strict = attribute_image(&gt, &dt, &report, ParityMode::Strict);
        let acc_corrected = attribute_image(&gt, &dt, &report, ParityMode::Corrected);

        assert_eq!(acc_strict.get(&100).map(|s| s.n_fp), Some(1));
        assert_eq!(acc_corrected.get(&100).map(|s| s.n_fp).unwrap_or(0), 0);
        // FN at cat=999 is identical across modes (V1 isn't mode-gated).
        assert_eq!(acc_strict.get(&999).map(|s| s.n_fn), Some(1));
        assert_eq!(acc_corrected.get(&999).map(|s| s.n_fn), Some(1));
    }
}
