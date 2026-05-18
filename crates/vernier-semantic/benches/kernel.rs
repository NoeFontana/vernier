//! Microbenchmarks for `accumulate_confusion` (ADR-0028 §"Algorithm").
//!
//! Run with `cargo bench -p vernier-semantic --bench kernel`.
//!
//! Measures the per-image confusion-matrix accumulation kernel — the
//! dominant non-decode cost on `coco_val2017_semantic_perfect` (the
//! workload `Semantic — mIoU (val2017)` reports in the bench harness).
//! Each call corresponds to one image's `(gt, dt)` u8 label-map pair
//! folded into a running `ConfusionMatrix`; in production the kernel
//! runs ~5000 times against val2017 inside
//! `vernier_semantic::decode::evaluate_from_pngs`.
//!
//! Shapes follow val2017 panoptic-semantic geometry: 480×640 images,
//! u8 label maps (matches the PNG fused-decode path that drives the
//! workload). Three input distributions exercise the kernel against
//! three meaningfully different cell-touch distributions — per
//! `perf_discipline.md`, the divergence between them is the
//! validation that any optimization we measure is workload-shape and
//! not microbench artifact:
//!
//! * `realistic_perfect` — horizontal stripes, GT == DT. Models the
//!   `_perfect` workload exactly; every contiguous segment hits one
//!   cell repeatedly. Maximum adjacent-pair coherence.
//! * `realistic_jittered` — same with 5 % random DT pixel flips.
//!   Proxy for a real model's output; runs survive but interleaved
//!   with off-diagonal cells.
//! * `uniform_random` — both GT and DT drawn uniformly from
//!   `[0, n_classes)`. No spatial structure; adjacent-pair coherence
//!   is ~1/n. Any sparse-skip / run-length optimization should be
//!   **flat** here; that flatness is the load-bearing validation
//!   signal.

#![allow(clippy::unwrap_used)]

use divan::{black_box, Bencher};
use vernier_semantic::ConfusionMatrix;
use vernier_semantic::kernel::accumulate_confusion;

fn main() {
    divan::main();
}

const HEIGHT: u32 = 480;
const WIDTH: u32 = 640;
const N_PIXELS: usize = (HEIGHT as usize) * (WIDTH as usize);
const N_SEGMENTS: u32 = 50;

/// `(ignore_label, n_classes)` parameter pairs. The two `n_classes`
/// values bracket the production range: Cityscapes (19) is the dense
/// "small confusion matrix" end (3.0 KB total cells); COCO panoptic
/// (133) is the val2017 target (141 KB, L2-resident but bigger than
/// L1). `ignore_label = Some(255)` matches the val2017 panoptic-to-
/// semantic mapping (unlabeled segment_id → 255). `None` exercises
/// the simpler dispatch.
const CASES: &[(Option<u32>, u32)] = &[
    (None, 19),
    (None, 133),
    (Some(255), 19),
    (Some(255), 133),
];

/// Synthetic `(H, W)` label map with `N_SEGMENTS` horizontal stripes
/// in `[0, n_classes)`. Stripe height stays uniform so the segment
/// boundaries land predictably for the run-length cache.
fn build_stripes(n_classes: u32) -> Vec<u8> {
    debug_assert!(n_classes <= 256);
    let stripe = N_PIXELS / N_SEGMENTS as usize;
    (0..N_PIXELS)
        .map(|i| {
            let seg = (i / stripe).min(N_SEGMENTS as usize - 1) as u32;
            (seg % n_classes) as u8
        })
        .collect()
}

/// Sprinkle ~5 % `ignore_label` pixels into `buf`. The deterministic
/// stride (1/20 = 5 %) keeps bench runs reproducible across machines.
fn sprinkle_ignore(buf: &mut [u8], ignore: u8) {
    for slot in buf.iter_mut().step_by(20) {
        *slot = ignore;
    }
}

/// Flip ~5 % of `buf` to a different class (LCG-style deterministic).
/// Avoids per-call RNG state so benches stay reproducible.
fn jitter(buf: &mut [u8], n_classes: u32) {
    let n = n_classes as u8;
    let mut state: u32 = 0x9E3779B1;
    for (i, slot) in buf.iter_mut().enumerate() {
        if i % 20 == 0 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let delta = ((state >> 16) % (n_classes - 1) as u32) as u8 + 1;
            *slot = (*slot).wrapping_add(delta) % n;
        }
    }
}

/// Uniform random `(H, W)` label map drawn from `[0, n_classes)` via
/// a deterministic LCG. Seed varies with `seed` so the GT and DT maps
/// are independent.
fn build_uniform(n_classes: u32, seed: u32) -> Vec<u8> {
    let n = n_classes as u32;
    let mut state = seed;
    (0..N_PIXELS)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            ((state >> 16) % n) as u8
        })
        .collect()
}

/// Build the `(gt, dt)` pair for a given case. Returned buffers feed
/// straight into the kernel without further allocation in the bench
/// hot path.
fn build_perfect(ignore: Option<u32>, n_classes: u32) -> (Vec<u8>, Vec<u8>) {
    let mut gt = build_stripes(n_classes);
    if let Some(ign) = ignore {
        sprinkle_ignore(&mut gt, ign as u8);
    }
    let dt = gt.clone();
    (gt, dt)
}

fn build_jittered(ignore: Option<u32>, n_classes: u32) -> (Vec<u8>, Vec<u8>) {
    let mut gt = build_stripes(n_classes);
    if let Some(ign) = ignore {
        sprinkle_ignore(&mut gt, ign as u8);
    }
    let mut dt = build_stripes(n_classes);
    jitter(&mut dt, n_classes);
    (gt, dt)
}

fn build_random(ignore: Option<u32>, n_classes: u32) -> (Vec<u8>, Vec<u8>) {
    let mut gt = build_uniform(n_classes, 0x1234_5678);
    if let Some(ign) = ignore {
        sprinkle_ignore(&mut gt, ign as u8);
    }
    let dt = build_uniform(n_classes, 0x9ABC_DEF0);
    (gt, dt)
}

#[divan::bench(args = CASES)]
fn realistic_perfect(bencher: Bencher, case: &(Option<u32>, u32)) {
    let (ignore, n_classes) = *case;
    let (gt, dt) = build_perfect(ignore, n_classes);
    bencher
        .with_inputs(|| ConfusionMatrix::zeros(n_classes))
        .bench_local_values(|mut cm| {
            accumulate_confusion(black_box(&gt), black_box(&dt), ignore, &mut cm);
            cm
        });
}

#[divan::bench(args = CASES)]
fn realistic_jittered(bencher: Bencher, case: &(Option<u32>, u32)) {
    let (ignore, n_classes) = *case;
    let (gt, dt) = build_jittered(ignore, n_classes);
    bencher
        .with_inputs(|| ConfusionMatrix::zeros(n_classes))
        .bench_local_values(|mut cm| {
            accumulate_confusion(black_box(&gt), black_box(&dt), ignore, &mut cm);
            cm
        });
}

#[divan::bench(args = CASES)]
fn uniform_random(bencher: Bencher, case: &(Option<u32>, u32)) {
    let (ignore, n_classes) = *case;
    let (gt, dt) = build_random(ignore, n_classes);
    bencher
        .with_inputs(|| ConfusionMatrix::zeros(n_classes))
        .bench_local_values(|mut cm| {
            accumulate_confusion(black_box(&gt), black_box(&dt), ignore, &mut cm);
            cm
        });
}
