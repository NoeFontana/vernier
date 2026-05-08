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

use rustc_hash::FxHashMap;

use crate::dataset::{CategoryId, ImageEntry, ImageId};
use crate::error::PanopticError;
use crate::parity::{PANOPTIC_IOU_THRESHOLD, PANOPTIC_VOID};

/// Cap on the dense `id → row/col` lookup-table size before falling
/// back to a hashmap-backed remap. COCO panoptic encodes segment ids
/// as `R + 256·G + 256²·B` with B nonzero on most images, so raw ids
/// reach the 10–16 million range; allocating
/// `Vec<u32; max_id + 1>` per call would dominate per-image cost.
/// 1 M caps the dense-table memory at 4 MiB; the sparse fallback
/// uses an `FxHashMap` whose memory is bounded by the segment count
/// (typically ≤ 64 entries).
const DENSE_LOOKUP_MAX_ID: u32 = 1_000_000;

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
    /// Dense intersection matrix keyed by `(gt_row, dt_col)` plus
    /// per-image `gt_id → row` and `dt_id → col` remap tables. The
    /// VOID row at index `n_gt` carries `(VOID, dt)` overlaps used by
    /// the attribute step's V4 FP exclusion. Replaces the prior
    /// `FxHashMap<(u32, u32), u32>` representation: branch-free per-
    /// pixel lookup, cache-resident counts (~`n_gt * n_dt * 4` bytes
    /// — typically ≤ 16 KiB and L1-resident), and a single linear
    /// fold over the parallel label maps.
    pub intersections: DenseIntersections,
}

/// Per-image dense intersection matrix.
///
/// `counts` is `(n_gt + 1) * n_dt` entries in row-major order. Rows
/// `0..n_gt` correspond to actual GT segments (in the same order as
/// `gt_ids`); row `n_gt` is the **VOID row** — `counts[n_gt * n_dt +
/// dt_col]` is the number of pixels where GT == VOID and DT covers
/// the segment at column `dt_col`. The V4 FP-exclusion test reads
/// this row.
///
/// `gt_remap` and `dt_remap` translate raw segment ids to row/col
/// indices. Two backends mirror the per-image distribution: a
/// `Vec<u32>`-backed direct lookup when raw ids fit
/// [`DENSE_LOOKUP_MAX_ID`]; an `FxHashMap`-backed fallback for COCO
/// panoptic and other RGB-encoded id spaces where raw ids exceed
/// the cap.
#[derive(Debug, Clone)]
pub struct DenseIntersections {
    /// Number of GT segments (excluding VOID).
    pub n_gt: u32,
    /// Number of DT segments.
    pub n_dt: u32,
    /// `(n_gt + 1) * n_dt` row-major counts.
    pub counts: Vec<u32>,
    /// `raw GT id -> row`. `PANOPTIC_VOID` maps to `n_gt`; unknown
    /// ids return `u32::MAX`.
    pub gt_remap: SegmentRemap,
    /// `raw DT id -> col`. Unknown ids return `u32::MAX`.
    pub dt_remap: SegmentRemap,
    /// `row -> raw_id` for GT (rows `0..n_gt`; the VOID row at
    /// `n_gt` has no entry here).
    pub gt_ids: Vec<u32>,
    /// `col -> raw_id` for DT.
    pub dt_ids: Vec<u32>,
}

/// Per-image segment-id → index lookup. Two backends:
/// - `Dense`: `Vec<u32>` of size `max_id + 1`, sentinel `u32::MAX`
///   for absent. Hit when `max_id ≤ DENSE_LOOKUP_MAX_ID`.
/// - `Sparse`: `FxHashMap<u32, u32>` for datasets whose encoded id
///   space exceeds the dense cap (the typical case for COCO
///   panoptic, where ids are RGB-packed and exceed 10⁷).
#[derive(Debug, Clone)]
pub enum SegmentRemap {
    /// `id → idx` direct lookup.
    Dense(Vec<u32>),
    /// `id → idx` hashmap fallback.
    Sparse(FxHashMap<u32, u32>),
}

impl SegmentRemap {
    #[inline]
    fn lookup(&self, id: u32) -> u32 {
        match self {
            SegmentRemap::Dense(v) => v.get(id as usize).copied().unwrap_or(u32::MAX),
            SegmentRemap::Sparse(m) => m.get(&id).copied().unwrap_or(u32::MAX),
        }
    }
}

impl DenseIntersections {
    /// Number of pixels where GT has segment `gt_id` and DT has
    /// segment `dt_id`. Returns `0` when either id is unknown.
    /// `gt_id == PANOPTIC_VOID` reads from the VOID row.
    #[inline]
    pub fn count(&self, gt_id: u32, dt_id: u32) -> u32 {
        let gi = self.gt_remap.lookup(gt_id);
        let di = self.dt_remap.lookup(dt_id);
        if gi == u32::MAX || di == u32::MAX {
            return 0;
        }
        let n_dt = self.n_dt as usize;
        self.counts[gi as usize * n_dt + di as usize]
    }
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

    let intersections = build_dense_intersections(gt, dt);
    let n_gt = intersections.n_gt as usize;
    let n_dt = intersections.n_dt as usize;
    let void_row = n_gt;

    // Iterate the dense matrix in (row, col) order. Per quirk **U9**
    // the TP set is invariant under iteration order — once a
    // `(gt, pred)` pair clears `iou > 0.5`, both sides go to
    // `*_matched` and no other candidate can re-match either side.
    // The dense layout makes this deterministic by construction (rows
    // are sorted GT ids; columns are sorted DT ids).
    let mut tp_pairs: Vec<TpPair> = Vec::new();
    let mut gt_matched: Vec<bool> = vec![false; n_gt];
    let mut dt_matched: Vec<bool> = vec![false; n_dt];

    // Range-based loops are intentional: `gi` and `di` index into
    // multiple parallel collections (`gt_matched`, `gt_ids`, the
    // dense `counts` matrix) so iter().enumerate() doesn't simplify.
    #[allow(clippy::needless_range_loop)]
    for gi in 0..n_gt {
        let gt_id = intersections.gt_ids[gi];
        let Some(gt_seg) = gt.segments.get(&gt_id) else {
            continue; // S8: GT segment declared but missing — but rows are
                      // built from `gt.segments`, so this is unreachable.
        };
        if gt_seg.iscrowd {
            continue; // U4: crowd GT cannot be a TP
        }
        #[allow(clippy::needless_range_loop)]
        for di in 0..n_dt {
            if dt_matched[di] {
                continue;
            }
            let intersection = intersections.counts[gi * n_dt + di];
            if intersection == 0 {
                continue;
            }
            let dt_id = intersections.dt_ids[di];
            let Some(dt_seg) = dt.segments.get(&dt_id) else {
                continue;
            };
            if gt_seg.category_id != dt_seg.category_id {
                continue; // U5: category disagreement
            }

            let void_overlap_with_dt = intersections.counts[void_row * n_dt + di];
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
                gt_matched[gi] = true;
                dt_matched[di] = true;
                break; // GT row matched; move to next gi
            }
        }
    }

    // Pin determinism on the outputs (not the matching loop, which
    // is order-independent by U9). The downstream FP/FN attribution
    // walks `unmatched_gt` to build the V3 last-wins / sum-of-overlaps
    // crowd map; sorted ids make strict-mode reproduction stable.
    tp_pairs.sort_unstable_by_key(|p| (p.gt_id, p.dt_id));
    let unmatched_gt: Vec<u32> = intersections
        .gt_ids
        .iter()
        .zip(gt_matched.iter())
        .filter_map(|(&id, &matched)| (!matched).then_some(id))
        .collect();
    let unmatched_dt: Vec<u32> = intersections
        .dt_ids
        .iter()
        .zip(dt_matched.iter())
        .filter_map(|(&id, &matched)| (!matched).then_some(id))
        .collect();

    Ok(PqImageReport {
        image_id,
        tp_pairs,
        unmatched_gt,
        unmatched_dt,
        intersections,
    })
}

/// Build the dense `(gt_row, dt_col) -> intersection_pixels` matrix
/// in one pass over the parallel label maps. The VOID row at index
/// `n_gt` carries the `(VOID, dt)` overlaps used by the attribute
/// step's V4 FP exclusion.
///
/// Quirk **T2** aligned: bit-equal to panopticapi's `np.unique` over
/// the OFFSET-encoded uint64 buffer (no uint64 intermediate). The
/// dense layout additionally avoids a hashmap allocation per pair —
/// for typical panoptic images (n_gt, n_dt ≤ ~64) the matrix is
/// ≤ 16 KiB and L1-resident.
///
/// `gt` and `dt` must have the same label-map length; the
/// orchestrator validates this via [`pq_image_with_id`]'s
/// `ShapeMismatch` check before reaching here.
fn build_dense_intersections(gt: &ImageEntry, dt: &ImageEntry) -> DenseIntersections {
    debug_assert_eq!(gt.label_map.len(), dt.label_map.len());

    // Sort segment ids so rows / cols have a stable order across
    // hashmap rehashes. The matching loop iterates this order.
    let mut gt_ids: Vec<u32> = gt.segments.keys().copied().collect();
    gt_ids.sort_unstable();
    let mut dt_ids: Vec<u32> = dt.segments.keys().copied().collect();
    dt_ids.sort_unstable();

    let n_gt = gt_ids.len();
    let n_dt = dt_ids.len();

    let max_gt_id = gt_ids.iter().copied().max().unwrap_or(0);
    let max_dt_id = dt_ids.iter().copied().max().unwrap_or(0);

    // The +1 row is the VOID row; columns are the DT segments.
    let mut counts: Vec<u32> = vec![0; (n_gt + 1) * n_dt];

    let dense_path = max_gt_id <= DENSE_LOOKUP_MAX_ID && max_dt_id <= DENSE_LOOKUP_MAX_ID;

    let (gt_remap, dt_remap) = if dense_path {
        // Direct-lookup path. `gt_lookup[id]` returns the row;
        // `gt_lookup[VOID] = n_gt` reserves the VOID row. Bounded
        // memory by `DENSE_LOOKUP_MAX_ID` per direction (~4 MiB).
        let gt_lookup_len = (max_gt_id as usize)
            .max(PANOPTIC_VOID as usize)
            .saturating_add(1);
        let mut gt_lookup: Vec<u32> = vec![u32::MAX; gt_lookup_len];
        for (idx, &id) in gt_ids.iter().enumerate() {
            gt_lookup[id as usize] = idx as u32;
        }
        // R3: pixel 0 is VOID. Real GT segments never have id 0, so
        // this slot is safe to reuse for the void row's index.
        gt_lookup[PANOPTIC_VOID as usize] = n_gt as u32;
        let mut dt_lookup: Vec<u32> = vec![u32::MAX; (max_dt_id as usize) + 1];
        for (idx, &id) in dt_ids.iter().enumerate() {
            dt_lookup[id as usize] = idx as u32;
        }

        for (g, d) in gt.label_map.iter().zip(dt.label_map.iter()) {
            let di = dt_lookup.get(*d as usize).copied().unwrap_or(u32::MAX);
            if di == u32::MAX {
                continue;
            }
            let gi = gt_lookup.get(*g as usize).copied().unwrap_or(u32::MAX);
            if gi == u32::MAX {
                continue;
            }
            counts[gi as usize * n_dt + di as usize] += 1;
        }

        (
            SegmentRemap::Dense(gt_lookup),
            SegmentRemap::Dense(dt_lookup),
        )
    } else {
        // Sparse fallback. Real-world COCO panoptic ids are
        // RGB-packed and reach ~16 M; allocating a per-call
        // `Vec<u32; max_id + 1>` would be ~50 MiB of allocator +
        // memset traffic. The `FxHashMap` backend is small (~64
        // entries) and pays a constant-factor lookup cost on the
        // per-pixel hot path.
        let mut gt_map: FxHashMap<u32, u32> =
            FxHashMap::with_capacity_and_hasher(n_gt + 1, Default::default());
        for (idx, &id) in gt_ids.iter().enumerate() {
            gt_map.insert(id, idx as u32);
        }
        gt_map.insert(PANOPTIC_VOID, n_gt as u32);
        let mut dt_map: FxHashMap<u32, u32> =
            FxHashMap::with_capacity_and_hasher(n_dt, Default::default());
        for (idx, &id) in dt_ids.iter().enumerate() {
            dt_map.insert(id, idx as u32);
        }

        for (g, d) in gt.label_map.iter().zip(dt.label_map.iter()) {
            let Some(&di) = dt_map.get(d) else {
                continue;
            };
            let Some(&gi) = gt_map.get(g) else {
                continue;
            };
            counts[gi as usize * n_dt + di as usize] += 1;
        }

        (SegmentRemap::Sparse(gt_map), SegmentRemap::Sparse(dt_map))
    };

    DenseIntersections {
        n_gt: n_gt as u32,
        n_dt: n_dt as u32,
        counts,
        gt_remap,
        dt_remap,
        gt_ids,
        dt_ids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::SegmentInfo;
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
