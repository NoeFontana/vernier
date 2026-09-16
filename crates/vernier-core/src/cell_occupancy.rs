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

    /// Build from `(row, column)` pairs already counted per row.
    fn from_pairs(n_rows: usize, mut pairs: Vec<(u32, u32)>) -> Self {
        // Counting sort into CSR, then per-row sort+dedup. Rows are
        // short (a val2017 image holds ~7 categories, an Objects365 one
        // ~8), so the per-row sort is cheap next to the fan-out it saves.
        let mut offsets = vec![0_u32; n_rows + 1];
        for &(row, _) in &pairs {
            offsets[row as usize + 1] += 1;
        }
        for i in 0..n_rows {
            offsets[i + 1] += offsets[i];
        }
        let mut values = vec![0_u32; pairs.len()];
        let mut cursor = offsets.clone();
        for (row, col) in pairs.drain(..) {
            let slot = &mut cursor[row as usize];
            values[*slot as usize] = col;
            *slot += 1;
        }
        let mut compacted: Vec<u32> = Vec::with_capacity(values.len());
        let mut new_offsets = vec![0_u32; n_rows + 1];
        for row in 0..n_rows {
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            let slice = &mut values[start..end];
            slice.sort_unstable();
            let mut last: Option<u32> = None;
            for &v in slice.iter() {
                if last != Some(v) {
                    compacted.push(v);
                    last = Some(v);
                }
            }
            new_offsets[row + 1] = u32::try_from(compacted.len()).unwrap_or(u32::MAX);
        }
        OccupiedCells {
            offsets: new_offsets,
            values: compacted,
        }
    }
}

/// Position of each category bucket, for mapping an annotation's
/// `category_id` back to its `k`. `None` buckets (the `use_cats=false`
/// collapse) map every category to bucket 0.
pub(crate) enum BucketIndex {
    Collapsed,
    ById(FxHashMap<CategoryId, u32>),
}

impl BucketIndex {
    pub(crate) fn new(category_buckets: &[Option<CategoryId>]) -> Self {
        if matches!(category_buckets, [None]) {
            return BucketIndex::Collapsed;
        }
        let mut map = FxHashMap::default();
        map.reserve(category_buckets.len());
        for (k, bucket) in category_buckets.iter().enumerate() {
            if let Some(cat) = bucket {
                map.insert(*cat, u32::try_from(k).unwrap_or(u32::MAX));
            }
        }
        BucketIndex::ById(map)
    }

    #[inline]
    fn bucket_of(&self, cat: CategoryId) -> Option<u32> {
        match self {
            // Every category collapses into the single virtual bucket.
            BucketIndex::Collapsed => Some(0),
            // A detection may name a category the GT never declares; it
            // has no bucket and no cell, so it drops out here exactly as
            // it would inside the per-cell lookup.
            BucketIndex::ById(map) => map.get(&cat).copied(),
        }
    }
}

/// Collect `(image_index, bucket)` pairs for every annotation and
/// detection on `images`.
fn occupied_pairs(
    images: &[ImageId],
    gt: &CocoDataset,
    dt: &CocoDetections,
    buckets: &BucketIndex,
) -> Vec<(u32, u32)> {
    let gt_anns = gt.annotations();
    let dt_anns = dt.detections();
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(gt_anns.len() + dt_anns.len());
    for (i, image_id) in images.iter().enumerate() {
        let i = u32::try_from(i).unwrap_or(u32::MAX);
        for &j in gt.ann_indices_for_image(*image_id) {
            if let Some(k) = buckets.bucket_of(gt_anns[j].category_id) {
                pairs.push((i, k));
            }
        }
        for &j in dt.indices_for_image(*image_id) {
            if let Some(k) = buckets.bucket_of(dt_anns[j].category_id) {
                pairs.push((i, k));
            }
        }
    }
    pairs
}

/// Rows are images, values are category buckets.
pub(crate) fn by_image(
    images: &[ImageId],
    gt: &CocoDataset,
    dt: &CocoDetections,
    buckets: &BucketIndex,
) -> OccupiedCells {
    let pairs = occupied_pairs(images, gt, dt, buckets);
    OccupiedCells::from_pairs(images.len(), pairs)
}

/// Rows are category buckets, values are images.
pub(crate) fn by_category(
    images: &[ImageId],
    n_buckets: usize,
    gt: &CocoDataset,
    dt: &CocoDetections,
    buckets: &BucketIndex,
) -> OccupiedCells {
    let pairs: Vec<(u32, u32)> = occupied_pairs(images, gt, dt, buckets)
        .into_iter()
        .map(|(i, k)| (k, i))
        .collect();
    OccupiedCells::from_pairs(n_buckets, pairs)
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
}
