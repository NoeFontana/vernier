//! Microbench for [`accumulate`] across the two grid regimes that make
//! its cost profile flip.
//!
//! `accumulate` pays two unrelated costs, and which one dominates is a
//! property of the dataset's shape rather than of its size:
//!
//! 1. **Per-category sorting.** Each `(area range, maxDet)` cell sorts
//!    its concatenated detection stream score-descending. The stream
//!    length is the category's detection count, so this term grows with
//!    `DT / K`.
//! 2. **Grid traversal.** Every `(area range)` pass gathers its cells by
//!    walking `I` slots of the dense `K · A · I` grid, whether or not
//!    they hold anything. This term grows with `K · A · I` and is
//!    memory-bandwidth bound (see ADR-0050's sub-linear scaling note).
//!
//! - **`coco_like`** — 80 categories × 5000 images at ~30 % cell
//!   occupancy, ~6.8k detections per category. Sorting dominates; this
//!   is the arm ADR-0052's single-sort derivation targets.
//! - **`long_tail`** — LVIS-shaped: 1203 categories, sparse cells, ~300
//!   detections per category. The streams are short enough that the
//!   grid walk dominates and the sort is nearly free, so ADR-0052 shows
//!   up only faintly here.
//!
//! Run with `cargo bench -p vernier-core --bench accumulate_shapes`.
//!
//! [`accumulate`]: vernier_core::accumulate::accumulate

#![allow(clippy::unwrap_used)]

use divan::{black_box, Bencher};
use ndarray::Array2;
use vernier_core::accumulate::{accumulate, AccumulateParams, PerImageEval};
use vernier_core::parity::{iou_thresholds, recall_thresholds};
use vernier_core::ParityMode;

fn main() {
    divan::main();
}

/// Synthetic grid shape. A fixed-seed xorshift makes the grid identical
/// run-to-run, so divan can build it outside the timing window and two
/// builds of the crate can be compared arm-for-arm.
#[derive(Clone, Copy)]
struct Shape {
    n_categories: usize,
    n_images: usize,
    /// Fraction of `(category, image)` pairs holding any detection.
    occupancy: f64,
    /// Upper bound on detections per occupied cell (uniform in `1..=d_max`).
    d_max: usize,
}

/// Sorting-dominated: val2017 detection shape.
const COCO_LIKE: Shape = Shape {
    n_categories: 80,
    n_images: 5000,
    occupancy: 0.30,
    d_max: 8,
};

/// Traversal-dominated: LVIS v1 val shape at a quarter of its images,
/// which keeps the 1203-category grid walk without a 760 MB allocation.
const LONG_TAIL: Shape = Shape {
    n_categories: 1203,
    n_images: 5000,
    occupancy: 0.008,
    d_max: 3,
};

const N_AREA_RANGES: usize = 4;
const MAX_DETS: [usize; 3] = [1, 10, 100];

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn unit(state: &mut u64) -> f64 {
    (xorshift(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Build a grid in the `[k][a][i]` layout `accumulate` expects.
///
/// The four area ranges of an occupied `(category, image)` pair share
/// one score vector, mirroring what `evaluate_cell` produces — the
/// precondition ADR-0052's plan reuse checks for.
fn build_grid(shape: Shape) -> Vec<Option<Box<PerImageEval>>> {
    let n_t = iou_thresholds().len();
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut grid: Vec<Option<Box<PerImageEval>>> =
        (0..shape.n_categories * N_AREA_RANGES * shape.n_images)
            .map(|_| None)
            .collect();

    for k in 0..shape.n_categories {
        for i in 0..shape.n_images {
            if unit(&mut state) > shape.occupancy {
                continue;
            }
            let d = 1 + (xorshift(&mut state) as usize % shape.d_max);
            let mut dt_scores: Vec<f64> = (0..d).map(|_| unit(&mut state)).collect();
            // Cells reach `accumulate` score-descending (quirk A4).
            dt_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
            // Deterministic match/ignore texture: some TPs, some FPs,
            // and a sprinkling of ignored DTs so the C7 branch is live.
            let dt_matched = Array2::from_shape_fn((n_t, d), |(t, j)| (t + j) % 3 == 0);
            let dt_ignore = Array2::from_shape_fn((n_t, d), |(_, j)| j % 11 == 0);
            let gt_ignore = vec![false, false, true];
            for a in 0..N_AREA_RANGES {
                grid[k * N_AREA_RANGES * shape.n_images + a * shape.n_images + i] =
                    Some(Box::new(PerImageEval {
                        dt_scores: dt_scores.clone(),
                        dt_matched: dt_matched.clone(),
                        dt_ignore: dt_ignore.clone(),
                        gt_ignore: gt_ignore.clone(),
                    }));
            }
        }
    }
    grid
}

fn run(bencher: Bencher, shape: Shape) {
    let grid = build_grid(shape);
    let params = AccumulateParams {
        iou_thresholds: iou_thresholds(),
        recall_thresholds: recall_thresholds(),
        max_dets: &MAX_DETS,
        n_categories: shape.n_categories,
        n_area_ranges: N_AREA_RANGES,
        n_images: shape.n_images,
    };
    bencher.bench(|| accumulate(black_box(&grid), params, ParityMode::Strict).unwrap());
}

#[divan::bench(sample_count = 5)]
fn coco_like(bencher: Bencher) {
    run(bencher, COCO_LIKE);
}

#[divan::bench(sample_count = 5)]
fn long_tail(bencher: Bencher) {
    run(bencher, LONG_TAIL);
}
