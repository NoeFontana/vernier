//! Framework-level microbench for [`evaluate_boundary`].
//!
//! Mirrors `evaluate_segm.rs` so the same shapes drive both kernels —
//! the boundary path adds per-annotation `boundary_band_segments_into`
//! cost on top of the segm wrapper, so the gap to `evaluate_segm`
//! quantifies the boundary-band contribution at framework scale.
//!
//! Cost centers (see `crates/vernier-core/src/similarity/boundary.rs`):
//!
//! 1. Same per-cell `Vec<SegmAnn>` allocation + RLE realization as
//!    `evaluate_segm` (the kernels share `build_segm_*_anns`).
//! 2. Per-annotation `boundary_band_segments_into` runs unconditionally
//!    on the DT side and on non-crowd GT (O1/O2 suppress the GT band
//!    when `is_crowd=true`). With `BoundaryComputeScratch` already
//!    threaded through the cached kernel path, the lever is the
//!    erosion + segment-decode work itself.
//! 3. Per-pair fold: two `intersect_area_offsets` calls plus the `min`
//!    of mask-IoU and band-IoU.
//!
//! Run with `just bench` or
//! `cargo bench -p vernier-core --bench evaluate_boundary`.
//!
//! [`evaluate_boundary`]: vernier_core::evaluate_boundary

#![allow(clippy::unwrap_used)]

use divan::{black_box, Bencher};
use vernier_core::dataset::{
    AnnId, Bbox, CategoryId, CategoryMeta, CocoAnnotation, DetectionInput, ImageId, ImageMeta,
};
use vernier_core::parity::iou_thresholds;
use vernier_core::segmentation::Segmentation;
use vernier_core::{
    evaluate_boundary, AreaRange, CocoDataset, CocoDetections, EvaluateParams, ParityMode,
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
const DILATION_RATIO: f64 = 0.02;

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn unit(state: &mut u64) -> f64 {
    (xorshift(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

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
        evaluate_boundary(
            black_box(&gt),
            black_box(&dt),
            params,
            ParityMode::Strict,
            DILATION_RATIO,
        )
        .unwrap()
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
