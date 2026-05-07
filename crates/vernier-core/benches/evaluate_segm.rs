//! Framework-level microbench for [`evaluate_segm`].
//!
//! `SegmIou::compute` itself is exercised by `segm_iou.rs`; the val2017
//! evaluate stage spends the bulk of its wall time in the per-cell
//! framework wrapper:
//!
//! 1. `kernel.build_gt_anns` / `build_dt_anns` allocate a fresh
//!    `Vec<SegmAnn>` per `(image, category)` cell and rasterize
//!    polygons / decode RLE counts into `Rle` buffers
//!    (`evaluate.rs:586-647`).
//! 2. `SegmComputeScratch` is shared across cells through the
//!    `SegmIouCached` kernel, but the per-ann `Rle::area` /
//!    `Rle::bbox` / `decode_fg_offsets_into` walks still run against
//!    the freshly-realized Rle counts each call.
//! 3. The same nine `Vec` gathers, IoU scratch alloc, and
//!    `dt_top_indices_for_cell` allocations the bbox bench surfaces —
//!    bounded by the round of [PR #178][1].
//!
//! Shape parity with `evaluate_bbox.rs` so deltas are directly
//! comparable across paradigms:
//!
//! - **`framework_coco_like`** — sparse multi-category, val2017-shape.
//!   100 images × 80 cats with ~7 GTs/image distributed across cats,
//!   so the median non-empty cell has `G·D = 1` and per-cell setup
//!   dominates. Realistic GT shape (polygons → rasterize) and DT
//!   shape (rectangle polygons → rasterize).
//! - **`framework_dense_mle_5cat`** / **`framework_dense_mle_1cat`** —
//!   the SOTA MLE regime (surveillance, autonomous-driving, dense
//!   single-class). 8 images × {5, 1} cats, 250 GTs and 250 DTs per
//!   image so each surviving cell sits at `G·D ≥ 12,500`. Per-pair
//!   `intersect_area_offsets` dominates; the framework wrapper is
//!   sub-1%.
//!
//! Image size pinned at `IMG_W × IMG_H = 96×96` so rasterization cost
//! stays bounded — large enough to exercise the polygon scanline
//! rasterizer non-trivially, small enough that per-ann RLE walks are
//! O(few-hundred-counts) rather than O(thousands).
//!
//! Run with `just bench` or
//! `cargo bench -p vernier-core --bench evaluate_segm`.
//!
//! [`evaluate_segm`]: vernier_core::evaluate_segm
//! [1]: https://github.com/NoeFontana/vernier/pull/178

#![allow(clippy::unwrap_used)]

use divan::{black_box, Bencher};
use vernier_core::{
    evaluate_segm, iou_thresholds, AnnId, AreaRange, Bbox, CategoryId, CategoryMeta,
    CocoAnnotation, CocoDataset, CocoDetections, DetectionInput, EvaluateParams, ImageId,
    ImageMeta, ParityMode, Segmentation,
};

fn main() {
    divan::main();
}

#[derive(Clone, Copy)]
struct Scenario {
    n_images: usize,
    n_categories: usize,
    gts_per_image: usize,
    dts_per_image: usize,
}

const COCO_LIKE: Scenario = Scenario {
    n_images: 100,
    n_categories: 80,
    gts_per_image: 7,
    dts_per_image: 30,
};

const DENSE_MLE_5CAT: Scenario = Scenario {
    n_images: 8,
    n_categories: 5,
    gts_per_image: 250,
    dts_per_image: 250,
};

const DENSE_MLE_1CAT: Scenario = Scenario {
    n_images: 8,
    n_categories: 1,
    gts_per_image: 250,
    dts_per_image: 250,
};

const IMG_W: f64 = 96.0;
const IMG_H: f64 = 96.0;
const MAX_W: f64 = 24.0;
const MAX_H: f64 = 24.0;

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn unit(state: &mut u64) -> f64 {
    (xorshift(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

/// Synthesize a 4-vertex axis-aligned rectangle polygon.
/// Matches the J2 quirk shape (`pycocotools/coco.py:341`) — what real
/// bbox-only DT files round-trip through.
fn rect_polygon(x: f64, y: f64, w: f64, h: f64) -> Segmentation {
    let polygon = vec![x, y, x, y + h, x + w, y + h, x + w, y];
    Segmentation::Polygons(vec![polygon])
}

fn build_dataset(s: Scenario) -> (CocoDataset, CocoDetections) {
    let mut state: u64 = 0xdead_beef_cafe_babe;

    let images: Vec<ImageMeta> = (0..s.n_images)
        .map(|i| ImageMeta {
            id: ImageId(i as i64 + 1),
            width: IMG_W as u32,
            height: IMG_H as u32,
            file_name: None,
        })
        .collect();

    let categories: Vec<CategoryMeta> = (0..s.n_categories)
        .map(|i| CategoryMeta {
            id: CategoryId(i as i64 + 1),
            name: format!("c{i}"),
            supercategory: None,
        })
        .collect();

    let mut gt_anns = Vec::with_capacity(s.n_images * s.gts_per_image);
    let mut next_gt = 1i64;
    for img in &images {
        for _ in 0..s.gts_per_image {
            let cat = (xorshift(&mut state) as usize) % s.n_categories;
            let w = 6.0 + unit(&mut state) * (MAX_W - 6.0);
            let h = 6.0 + unit(&mut state) * (MAX_H - 6.0);
            let x = unit(&mut state) * (IMG_W - w);
            let y = unit(&mut state) * (IMG_H - h);
            gt_anns.push(CocoAnnotation {
                id: AnnId(next_gt),
                image_id: img.id,
                category_id: CategoryId(cat as i64 + 1),
                area: w * h,
                is_crowd: false,
                ignore_flag: None,
                bbox: Bbox { x, y, w, h },
                segmentation: Some(rect_polygon(x, y, w, h)),
                keypoints: None,
                num_keypoints: None,
            });
            next_gt += 1;
        }
    }

    let mut dt_inputs = Vec::with_capacity(s.n_images * s.dts_per_image);
    for img in &images {
        for _ in 0..s.dts_per_image {
            let cat = (xorshift(&mut state) as usize) % s.n_categories;
            let w = 6.0 + unit(&mut state) * (MAX_W - 6.0);
            let h = 6.0 + unit(&mut state) * (MAX_H - 6.0);
            let x = unit(&mut state) * (IMG_W - w);
            let y = unit(&mut state) * (IMG_H - h);
            let score = unit(&mut state);
            dt_inputs.push(DetectionInput {
                id: None,
                image_id: img.id,
                category_id: CategoryId(cat as i64 + 1),
                score,
                bbox: Bbox { x, y, w, h },
                segmentation: Some(rect_polygon(x, y, w, h)),
                keypoints: None,
                num_keypoints: None,
            });
        }
    }

    let gt = CocoDataset::from_parts(images, gt_anns, categories).unwrap();
    let dt = CocoDetections::from_inputs(dt_inputs).unwrap();
    (gt, dt)
}

fn run(bencher: Bencher, s: Scenario) {
    let (gt, dt) = build_dataset(s);
    let area_ranges = AreaRange::coco_default();
    let iou_thr = iou_thresholds();
    bencher.bench_local(|| {
        let params = EvaluateParams {
            iou_thresholds: iou_thr,
            area_ranges: &area_ranges,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        evaluate_segm(black_box(&gt), black_box(&dt), params, ParityMode::Strict).unwrap()
    });
}

#[divan::bench]
fn framework_coco_like(bencher: Bencher) {
    run(bencher, COCO_LIKE);
}

#[divan::bench]
fn framework_dense_mle_5cat(bencher: Bencher) {
    run(bencher, DENSE_MLE_5CAT);
}

#[divan::bench]
fn framework_dense_mle_1cat(bencher: Bencher) {
    run(bencher, DENSE_MLE_1CAT);
}
