//! Which `(category, image)` cells can possibly be non-empty.
//!
//! Both evaluate paths walk the full `K x I` cell grid and discover
//! emptiness *inside* the per-cell routine, after two index lookups
//! ([`crate::evaluate::gt_indices_for_cell`] +
//! [`crate::evaluate::raw_dt_indices_for_cell`]). On COCO that waste is
//! invisible — 80 categories x 5000 images is 400 k lookups. It stops
//! being invisible as the category axis grows: Objects365 val is 365
//! categories x 80 000 images = 29.2 M cells of which **626 k (2.1 %)**
//! hold any annotation or detection at all, and LVIS v1 val is worse
//! (1203 categories).
//!
//! This module builds, once, the list of cells worth visiting. A cell is
//! a candidate when the image has at least one GT annotation *or* one
//! detection in that category — exactly the negation of the
//! `gt_indices.is_empty() && raw_dt_indices.is_empty()` early-return
//! both evaluate paths already perform. Skipping a non-candidate is
//! therefore output-identical, not an approximation: the cell would have
//! produced `None`, and `None` is what an unvisited slot already holds.
//!
//! Storage is CSR (`offsets` + `values`) rather than `Vec<Vec<_>>`: one
//! allocation per direction instead of one per row, and the rows stay
//! contiguous for the iteration order each caller needs.
//!
//! - [`OccupiedCells::by_image`] — rows are images, values are category
//!   buckets. The parallel path fans out per image.
//! - [`OccupiedCells::by_category`] — rows are category buckets, values
//!   are images. The sequential path walks `(k, i)` in canonical order.
//!
//! Values within a row are ascending and deduplicated, so callers visit
//! cells in the same order the exhaustive loop did.

use rustc_hash::FxHashMap;

use crate::dataset::{CategoryId, CocoDataset, CocoDetections, EvalDataset, ImageId};

/// Occupied cells in compressed-sparse-row form.
pub(crate) struct OccupiedCells {
    offsets: Vec<u32>,
    values: Vec<u32>,
}

impl OccupiedCells {
    /// Cells of row `row`, ascending. Empty when the row has none.
    #[inline]
    pub(crate) fn row(&self, row: usize) -> &[u32] {
        let start = self.offsets[row] as usize;
        let end = self.offsets[row + 1] as usize;
        &self.values[start..end]
    }
}

/// Position of each category bucket, for mapping an annotation's
/// `category_id` back to its `k`. `None` buckets (the `use_cats=false`
/// collapse) map every category to bucket 0.
///
/// `CocoDataset::from_parts` validates that annotations reference a
/// *known* category but does not reject a `categories` array that
/// repeats an id, so two buckets can carry the same `CategoryId`. The
/// exhaustive loop evaluated both, so both must stay candidates —
/// keeping only one would blank the other's precision row. Repeats are
/// pathological, so they ride a side list and the common lookup stays a
/// single hash probe.
pub(crate) enum BucketIndex {
    Collapsed,
    ById {
        n_buckets: usize,
        first: FxHashMap<CategoryId, u32>,
        repeats: Vec<(CategoryId, u32)>,
    },
}

impl BucketIndex {
    pub(crate) fn new(category_buckets: &[Option<CategoryId>]) -> Self {
        if matches!(category_buckets, [None]) {
            return BucketIndex::Collapsed;
        }
        let mut first = FxHashMap::default();
        first.reserve(category_buckets.len());
        let mut repeats = Vec::new();
        for (k, bucket) in category_buckets.iter().enumerate() {
            if let Some(cat) = bucket {
                let k = u32::try_from(k).unwrap_or(u32::MAX);
                // `or_insert` keeps the earliest bucket as the primary;
                // later ones ride the side list rather than displacing it.
                if *first.entry(*cat).or_insert(k) != k {
                    repeats.push((*cat, k));
                }
            }
        }
        BucketIndex::ById {
            n_buckets: category_buckets.len(),
            first,
            repeats,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        match self {
            BucketIndex::Collapsed => 1,
            BucketIndex::ById { n_buckets, .. } => *n_buckets,
        }
    }

    #[inline]
    fn bucket_of(&self, cat: CategoryId) -> Option<u32> {
        match self {
            // Every category collapses into the single virtual bucket.
            BucketIndex::Collapsed => Some(0),
            // A detection may name a category the GT never declares; it
            // has no bucket and no cell, so it drops out here exactly as
            // it would inside the per-cell lookup.
            BucketIndex::ById { first, .. } => first.get(&cat).copied(),
        }
    }

    /// Extra buckets sharing a `CategoryId` with an earlier one. Empty
    /// for every well-formed dataset.
    #[inline]
    fn repeats(&self) -> &[(CategoryId, u32)] {
        match self {
            BucketIndex::Collapsed => &[],
            BucketIndex::ById { repeats, .. } => repeats,
        }
    }
}

/// Rows are images, values are category buckets.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn by_image(
    images: &[ImageId],
    gt: &CocoDataset,
    dt: &CocoDetections,
    buckets: &BucketIndex,
) -> OccupiedCells {
    let n_buckets = buckets.len();
    let gt_anns = gt.annotations();
    let dt_cats = dt.category_ids();
    let repeats = buckets.repeats();

    let mut seen_stamp = vec![u32::MAX; n_buckets];
    let mut offsets = Vec::with_capacity(images.len() + 1);
    offsets.push(0);
    let mut values: Vec<u32> = Vec::with_capacity(images.len().saturating_mul(8));

    let push = |seen: &mut [u32], vals: &mut Vec<u32>, i: u32, cat: CategoryId| {
        if let Some(k) = buckets.bucket_of(cat) {
            let ku = k as usize;
            if seen[ku] != i {
                seen[ku] = i;
                vals.push(k);
            }
        }
        if !repeats.is_empty() {
            for &(repeat_cat, k) in repeats {
                if repeat_cat == cat {
                    let ku = k as usize;
                    if seen[ku] != i {
                        seen[ku] = i;
                        vals.push(k);
                    }
                }
            }
        }
    };

    for (idx, &image_id) in images.iter().enumerate() {
        let i = u32::try_from(idx).unwrap_or(u32::MAX);
        let row_start = values.len();
        for &j in gt.ann_indices_for_image(image_id) {
            push(&mut seen_stamp, &mut values, i, gt_anns[j].category_id);
        }
        for &j in dt.indices_for_image(image_id) {
            push(&mut seen_stamp, &mut values, i, dt_cats[j]);
        }
        values[row_start..].sort_unstable();
        offsets.push(u32::try_from(values.len()).unwrap_or(u32::MAX));
    }

    OccupiedCells { offsets, values }
}

/// Rows are category buckets, values are image indices (ascending).
#[cfg(test)]
pub(crate) fn by_category(
    images: &[ImageId],
    n_buckets: usize,
    gt: &CocoDataset,
    dt: &CocoDetections,
    buckets: &BucketIndex,
) -> OccupiedCells {
    by_category_subset(images, None, n_buckets, gt, dt, buckets)
}

/// Rows are category buckets, values are image indices restricted to
/// `image_subset` (when `Some`) or `0..images.len()` (when `None`).
pub(crate) fn by_category_subset(
    images: &[ImageId],
    image_subset: Option<&[usize]>,
    n_buckets: usize,
    gt: &CocoDataset,
    dt: &CocoDetections,
    buckets: &BucketIndex,
) -> OccupiedCells {
    let gt_anns = gt.annotations();
    let dt_cats = dt.category_ids();
    let repeats = buckets.repeats();

    let mut seen_stamp = vec![u32::MAX; n_buckets];
    let mut offsets = vec![0_u32; n_buckets + 1];

    let cap_hint = match image_subset {
        Some(sub) => sub.len().saturating_mul(n_buckets.min(16)),
        None => (gt_anns.len() + dt_cats.len()).min(images.len().saturating_mul(n_buckets)),
    };
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(cap_hint);

    let mut visit_cat = |i: u32, cat: CategoryId| {
        if let Some(k) = buckets.bucket_of(cat) {
            let ku = k as usize;
            if ku < n_buckets && seen_stamp[ku] != i {
                seen_stamp[ku] = i;
                offsets[ku + 1] += 1;
                pairs.push((k, i));
            }
        }
        if !repeats.is_empty() {
            for &(repeat_cat, k) in repeats {
                if repeat_cat == cat {
                    let ku = k as usize;
                    if ku < n_buckets && seen_stamp[ku] != i {
                        seen_stamp[ku] = i;
                        offsets[ku + 1] += 1;
                        pairs.push((k, i));
                    }
                }
            }
        }
    };

    match image_subset {
        None => {
            for (idx, &image_id) in images.iter().enumerate() {
                let i = u32::try_from(idx).unwrap_or(u32::MAX);
                for &j in gt.ann_indices_for_image(image_id) {
                    visit_cat(i, gt_anns[j].category_id);
                }
                for &j in dt.indices_for_image(image_id) {
                    visit_cat(i, dt_cats[j]);
                }
            }
        }
        Some(subset) => {
            let mut sorted_cow: Vec<usize> = Vec::new();
            let ordered: &[usize] = if subset.windows(2).all(|w| w[0] < w[1]) {
                subset
            } else {
                sorted_cow.extend_from_slice(subset);
                sorted_cow.sort_unstable();
                sorted_cow.dedup();
                &sorted_cow
            };
            for &idx in ordered {
                if idx >= images.len() {
                    continue;
                }
                let i = u32::try_from(idx).unwrap_or(u32::MAX);
                let image_id = images[idx];
                for &j in gt.ann_indices_for_image(image_id) {
                    visit_cat(i, gt_anns[j].category_id);
                }
                for &j in dt.indices_for_image(image_id) {
                    visit_cat(i, dt_cats[j]);
                }
            }
        }
    }

    for k in 0..n_buckets {
        offsets[k + 1] += offsets[k];
    }
    let mut values = vec![0_u32; pairs.len()];
    let mut cursor = offsets.clone();
    // Because `i` was visited in strictly ascending order and
    // deduplicated via `seen_stamp`, scattering in insertion order
    // produces each category row already sorted ascending with no
    // duplicates.
    for &(k, i) in &pairs {
        let slot = &mut cursor[k as usize];
        values[*slot as usize] = i;
        *slot += 1;
    }

    OccupiedCells { offsets, values }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{AnnId, Bbox, CategoryMeta, CocoAnnotation, DetectionInput, ImageMeta};

    fn img(id: i64) -> ImageMeta {
        ImageMeta {
            id: ImageId(id),
            file_name: None,
            width: 64,
            height: 64,
        }
    }

    fn cat(id: i64) -> CategoryMeta {
        CategoryMeta {
            id: CategoryId(id),
            name: format!("c{id}"),
            supercategory: None,
        }
    }

    fn ann(id: i64, image: i64, category: i64) -> CocoAnnotation {
        CocoAnnotation {
            id: AnnId(id),
            image_id: ImageId(image),
            category_id: CategoryId(category),
            area: 16.0,
            is_crowd: false,
            ignore_flag: None,
            bbox: Bbox {
                x: 0.0,
                y: 0.0,
                w: 4.0,
                h: 4.0,
            },
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
        }
    }

    fn dt_input(image: i64, category: i64) -> DetectionInput {
        DetectionInput {
            id: None,
            image_id: ImageId(image),
            category_id: CategoryId(category),
            score: 0.9,
            bbox: Bbox {
                x: 0.0,
                y: 0.0,
                w: 4.0,
                h: 4.0,
            },
            area: None,
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
        }
    }

    /// GT on (img 1, cat 1) twice and (img 2, cat 2); DT adds
    /// (img 1, cat 2) plus a category the GT never declares.
    fn fixture() -> (
        Vec<ImageId>,
        CocoDataset,
        CocoDetections,
        Vec<Option<CategoryId>>,
    ) {
        let gt = CocoDataset::from_parts(
            vec![img(1), img(2)],
            vec![ann(1, 1, 1), ann(2, 1, 1), ann(3, 2, 2)],
            vec![cat(1), cat(2)],
        )
        .expect("gt");
        let dt = CocoDetections::from_inputs(vec![dt_input(1, 2), dt_input(1, 99)]).expect("dt");
        (
            vec![ImageId(1), ImageId(2)],
            gt,
            dt,
            vec![Some(CategoryId(1)), Some(CategoryId(2))],
        )
    }

    #[test]
    fn rows_are_ascending_deduplicated_and_cover_gt_and_dt() {
        let (images, gt, dt, buckets) = fixture();
        let index = BucketIndex::new(&buckets);

        // Image 1: two GTs in bucket 0 collapse to one entry; the DT
        // adds bucket 1. Image 2: GT in bucket 1 only.
        let by_img = by_image(&images, &gt, &dt, &index);
        assert_eq!(by_img.row(0), &[0, 1]);
        assert_eq!(by_img.row(1), &[1]);

        // Same cells, transposed.
        let by_cat = by_category(&images, buckets.len(), &gt, &dt, &index);
        assert_eq!(by_cat.row(0), &[0]);
        assert_eq!(by_cat.row(1), &[0, 1]);
    }

    #[test]
    fn detection_category_absent_from_gt_has_no_cell() {
        // `dt_input(1, 99)` names a category with no bucket. The
        // evaluate loop could never visit such a cell, so it must not
        // appear as a candidate either.
        let (images, gt, dt, buckets) = fixture();
        let index = BucketIndex::new(&buckets);
        let by_img = by_image(&images, &gt, &dt, &index);
        assert!(by_img.row(0).iter().all(|&k| (k as usize) < buckets.len()));
    }

    #[test]
    fn collapsed_buckets_put_every_category_in_bucket_zero() {
        let (images, gt, dt, _) = fixture();
        let index = BucketIndex::new(&[None]);
        let by_img = by_image(&images, &gt, &dt, &index);
        assert_eq!(by_img.row(0), &[0]);
        assert_eq!(by_img.row(1), &[0]);
    }

    #[test]
    fn image_without_gt_or_dt_has_no_candidate_cells() {
        let gt = CocoDataset::from_parts(vec![img(1), img(7)], vec![ann(1, 1, 1)], vec![cat(1)])
            .expect("gt");
        let dt = CocoDetections::from_inputs(vec![]).expect("dt");
        let images = vec![ImageId(1), ImageId(7)];
        let index = BucketIndex::new(&[Some(CategoryId(1))]);
        let by_img = by_image(&images, &gt, &dt, &index);
        assert_eq!(by_img.row(0), &[0]);
        assert!(by_img.row(1).is_empty());
    }

    /// A GT whose `categories` array repeats an id produces two buckets
    /// with the same `CategoryId`. The exhaustive loop evaluated both,
    /// so both must stay candidates — a bucket map that keeps only the
    /// last would silently blank the earlier category's precision row.
    #[test]
    fn duplicate_category_ids_keep_every_bucket() {
        let gt = CocoDataset::from_parts(vec![img(1)], vec![ann(1, 1, 1)], vec![cat(1), cat(1)])
            .expect("gt");
        let dt = CocoDetections::from_inputs(vec![]).expect("dt");
        let images = vec![ImageId(1)];
        let index = BucketIndex::new(&[Some(CategoryId(1)), Some(CategoryId(1))]);
        let by_img = by_image(&images, &gt, &dt, &index);
        assert_eq!(by_img.row(0), &[0, 1]);
    }

    /// The occupancy index is now the *only* thing deciding which cells
    /// either evaluate path visits, so "parallel matches sequential"
    /// can no longer catch a bug in it — both walks would skip the same
    /// cell. Pin it against a brute-force enumeration of the predicate
    /// the per-cell body actually applies
    /// (`gt_indices.is_empty() && raw_dt_indices.is_empty()`), over a
    /// federated (LVIS-shaped) dataset with crowd, zero-area and
    /// DT-only cells.
    #[test]
    fn matches_brute_force_enumeration_on_a_federated_dataset() {
        let images: Vec<ImageMeta> = (1..=6).map(img).collect();
        let categories: Vec<CategoryMeta> = (1..=5).map(cat).collect();
        let mut anns = Vec::new();
        let mut next = 1_i64;
        for image in 1..=6_i64 {
            for category in 1..=5_i64 {
                // Sparse, irregular coverage: ~a third of the cells.
                if (image * 7 + category * 3) % 3 == 0 {
                    let mut a = ann(next, image, category);
                    a.is_crowd = image % 2 == 0;
                    a.area = if category == 4 { 0.0 } else { 16.0 };
                    anns.push(a);
                    next += 1;
                }
            }
        }
        let gt = CocoDataset::from_parts(images, anns, categories).expect("gt");
        // DT-only cells, plus one category the GT never declares.
        let dt = CocoDetections::from_inputs(vec![
            dt_input(2, 5),
            dt_input(3, 1),
            dt_input(3, 1),
            dt_input(6, 99),
        ])
        .expect("dt");

        let image_ids: Vec<ImageId> = (1..=6).map(ImageId).collect();
        let buckets: Vec<Option<CategoryId>> = (1..=5).map(|c| Some(CategoryId(c))).collect();
        let index = BucketIndex::new(&buckets);
        let by_img = by_image(&image_ids, &gt, &dt, &index);
        let by_cat = by_category(&image_ids, buckets.len(), &gt, &dt, &index);

        for (i, image_id) in image_ids.iter().enumerate() {
            for (k, bucket) in buckets.iter().enumerate() {
                let occupied_by_predicate = !gt
                    .ann_indices_for(*image_id, bucket.expect("cat"))
                    .is_empty()
                    || !dt.indices_for(*image_id, bucket.expect("cat")).is_empty();
                let k32 = u32::try_from(k).expect("k fits");
                let i32_ = u32::try_from(i).expect("i fits");
                assert_eq!(
                    by_img.row(i).contains(&k32),
                    occupied_by_predicate,
                    "by_image disagrees at (image {i}, category {k})"
                );
                assert_eq!(
                    by_cat.row(k).contains(&i32_),
                    occupied_by_predicate,
                    "by_category disagrees at (image {i}, category {k})"
                );
            }
        }
    }
}
