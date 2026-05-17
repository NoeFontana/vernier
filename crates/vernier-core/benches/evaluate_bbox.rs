//! Framework-level microbench for [`evaluate_bbox`].
//!
//! `BboxIou::compute` itself takes ~40 ns at `G·D = 1` (per the divan
//! arms in `bbox_iou.rs`), but the val2017 evaluate stage spends ~25 ms
//! on 14k cells — the lever lives in the per-cell framework wrapper:
//!
//! 1. Nine small `Vec` gathers per cell in `evaluate_with`'s body —
//!    `gt_areas` / `gt_iscrowd` / `gt_base_ignore` / `gt_ids` and the DT
//!    counterparts, plus `kernel.build_gt_anns` / `build_dt_anns`
//!    (`evaluate.rs:871-888`).
//! 2. One `Array2::<f64>::zeros((g, d))` IoU scratch alloc per cell
//!    (`evaluate.rs:890`).
//! 3. Three more allocs in `dt_top_indices_for_cell` (scores,
//!    permutation, result) — `evaluate.rs:1304-1309`.
//! 4. Four area-range repeats of `evaluate_cell` + `match_image`, each
//!    reallocating six small `Vec`s plus three `Array2`s.
//!
//! Total ~77 allocations per `(image, category)` cell — that's the cost
//! center the existing `bbox_iou.rs` arms cannot reach.
//!
//! Two regimes per the `docs/engineering/benchmarking/2026-05-bbox-cdf.md`
//! split:
//!
//! - **`framework_coco_like`** — sparse multi-category, val2017-shape.
//!   100 images × 80 cats with ~7 GTs/image distributed across cats, so
//!   the median non-empty cell has `G·D = 1` and per-cell setup
//!   dominates.
//! - **`framework_dense_mle_5cat`** / **`framework_dense_mle_1cat`** —
//!   the SOTA MLE regime (surveillance, autonomous-driving, dense
//!   single-class). 8 images × {5, 1} cats, 250 GTs and 250 DTs per
//!   image so each surviving cell sits at `G·D ≥ 12,500`. Per-cell
//!   setup is sub-1% of cell cost; the inner kernel + matching loop
//!   dominate.
//!
//! Run with `just bench` or
//! `cargo bench -p vernier-core --bench evaluate_bbox`.
//!
//! [`evaluate_bbox`]: vernier_core::evaluate_bbox

#![allow(clippy::unwrap_used)]

use divan::{black_box, Bencher};
use vernier_core::dataset::{
    AnnId, Bbox, CategoryId, CategoryMeta, CocoAnnotation, DetectionInput, ImageId, ImageMeta,
};
use vernier_core::parity::iou_thresholds;
use vernier_core::{
    evaluate_bbox, evaluate_bbox_parallel, AreaRange, CocoDataset, CocoDetections, EvaluateParams,
    ParityMode,
};

fn main() {
    divan::main();
}

/// Synthetic per-cell shape parameters. Determinism is provided by a
/// fixed-seed xorshift below, so the dataset is identical run-to-run
/// and divan can amortize its setup cost outside the timing window.
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

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn unit(state: &mut u64) -> f64 {
    (xorshift(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

fn build_dataset(s: Scenario) -> (CocoDataset, CocoDetections) {
    let mut state: u64 = 0xdead_beef_cafe_babe;
    let img_w = 640.0;
    let img_h = 480.0;
    let max_w = 80.0;
    let max_h = 80.0;

    let images: Vec<ImageMeta> = (0..s.n_images)
        .map(|i| ImageMeta {
            id: ImageId(i as i64 + 1),
            width: img_w as u32,
            height: img_h as u32,
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
            let w = 20.0 + unit(&mut state) * (max_w - 20.0);
            let h = 20.0 + unit(&mut state) * (max_h - 20.0);
            let x = unit(&mut state) * (img_w - w);
            let y = unit(&mut state) * (img_h - h);
            gt_anns.push(CocoAnnotation {
                id: AnnId(next_gt),
                image_id: img.id,
                category_id: CategoryId(cat as i64 + 1),
                area: w * h,
                is_crowd: false,
                ignore_flag: None,
                bbox: Bbox { x, y, w, h },
                segmentation: None,
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
            let w = 20.0 + unit(&mut state) * (max_w - 20.0);
            let h = 20.0 + unit(&mut state) * (max_h - 20.0);
            let x = unit(&mut state) * (img_w - w);
            let y = unit(&mut state) * (img_h - h);
            let score = unit(&mut state);
            dt_inputs.push(DetectionInput {
                id: None,
                image_id: img.id,
                category_id: CategoryId(cat as i64 + 1),
                score,
                bbox: Bbox { x, y, w, h },
                segmentation: None,
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
        evaluate_bbox(black_box(&gt), black_box(&dt), params, ParityMode::Strict).unwrap()
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

// Parallel sweep arms — pool built once outside the timing window
// (ADR-0047 ~50-200 µs construction cost stays out of the per-iter number).
const THREAD_COUNTS: [usize; 4] = [1, 2, 4, 8];

fn run_parallel(bencher: Bencher, s: Scenario, num_threads: usize) {
    let (gt, dt) = build_dataset(s);
    let area_ranges = AreaRange::coco_default();
    let iou_thr = iou_thresholds();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();
    bencher.bench_local(|| {
        let params = EvaluateParams {
            iou_thresholds: iou_thr,
            area_ranges: &area_ranges,
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        };
        pool.install(|| {
            evaluate_bbox_parallel(black_box(&gt), black_box(&dt), params, ParityMode::Strict)
                .unwrap()
        })
    });
}

#[divan::bench(args = THREAD_COUNTS)]
fn framework_coco_like_parallel(bencher: Bencher, &num_threads: &usize) {
    run_parallel(bencher, COCO_LIKE, num_threads);
}

#[divan::bench(args = THREAD_COUNTS)]
fn framework_dense_mle_5cat_parallel(bencher: Bencher, &num_threads: &usize) {
    run_parallel(bencher, DENSE_MLE_5CAT, num_threads);
}

#[divan::bench(args = THREAD_COUNTS)]
fn framework_dense_mle_1cat_parallel(bencher: Bencher, &num_threads: &usize) {
    run_parallel(bencher, DENSE_MLE_1CAT, num_threads);
}
