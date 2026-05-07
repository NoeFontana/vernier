//! Microbenchmarks for the segm IoU [`Similarity`] kernel.
//!
//! Two cost centers (see `crates/vernier-core/src/similarity/segm.rs`):
//!
//! 1. **bbox prefilter (I1)** — one [`BboxIou::compute`] over the
//!    tight RLE bboxes, zeroing cells that can't overlap.
//! 2. **per-cell intersect** — for cells that survive (1), one
//!    [`Rle::intersect_area`] call plus the union-area math.
//!
//! Three regimes hold shape and grid fixed and vary overlap density:
//!
//! - **production_overlap**: dense overlap; most cells survive (1)
//!   so per-cell intersect dominates. Worst case in practice.
//! - **production_disjoint**: tiled non-overlapping shapes; (1) zeros
//!   most cells so per-cell intersect is amortized down to the
//!   prefilter alone. The gap to `production_overlap` measures (2) in
//!   isolation.
//! - **production_crowd_gt**: GT-side `is_crowd=true`. Segm doesn't
//!   short-circuit crowd rows the way boundary does (no band to
//!   skip), but the denominator switches to `dt_area` per quirk E1.
//!   The gap to `production_overlap` upper-bounds the savings any
//!   future crowd-aware fast path could match.
//!
//! Run with `just bench` or `cargo bench -p vernier-core --bench segm_iou`.
//!
//! [`BboxIou::compute`]: vernier_core::similarity::BboxIou::compute
//! [`Rle::intersect_area`]: vernier_mask::Rle::intersect_area

#![allow(clippy::unwrap_used)]

use divan::{black_box, Bencher};
use ndarray::Array2;
use vernier_core::similarity::{SegmIou, Similarity};

#[path = "common/mod.rs"]
mod common;
use common::{disjoint_rects, overlapping_rects};

fn main() {
    divan::main();
}

/// Per-image grid sizes (gts × dts). Tracks the same band as the
/// boundary bench so segm/boundary deltas are directly comparable —
/// both sit on top of `Rle::intersect_area`, and Step 3 of the
/// 2026-05 perf push touches that shared kernel.
const GRIDS: &[(usize, usize)] = &[(4, 100), (10, 100), (30, 100)];

/// Small-square grids covering the **per-call setup overhead** regime.
/// Both regime extrema (1–30 GT/10 cats, 100–300 GT/100 cats) decompose
/// into many small cells per kernel call. Per
/// `docs/engineering/benchmarking/2026-05-bbox-cdf.md`, val2017 has
/// median `G·D = 1` and ~99% of wall time in cells with `G·D < 256`.
/// At this shape the bbox prefilter (1) is dominant in absolute terms
/// but each call's RLE-side work (2) is also tiny, so the arm
/// captures the per-call setup contribution end-to-end.
const SPARSE_GRIDS: &[(usize, usize)] = &[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (10, 10)];

#[divan::bench(args = GRIDS)]
fn production_overlap(bencher: Bencher, &(g, d): &(usize, usize)) {
    let gts = overlapping_rects(g, false);
    let dts = overlapping_rects(d, false);
    let mut out = Array2::<f64>::zeros((g, d));
    bencher.bench_local(|| {
        SegmIou
            .compute(black_box(&gts), black_box(&dts), &mut out.view_mut())
            .unwrap();
        // Re-zero so each iteration starts from a known initial
        // state. The kernel overwrites every cell today, so this
        // doesn't change the answer; it documents the invariant and
        // stops a future regression from amplifying across iterations.
        out.fill(0.0);
    });
}

#[divan::bench(args = GRIDS)]
fn production_disjoint(bencher: Bencher, &(g, d): &(usize, usize)) {
    let gts = disjoint_rects(g);
    let dts = disjoint_rects(d);
    let mut out = Array2::<f64>::zeros((g, d));
    bencher.bench_local(|| {
        SegmIou
            .compute(black_box(&gts), black_box(&dts), &mut out.view_mut())
            .unwrap();
        out.fill(0.0);
    });
}

#[divan::bench(args = GRIDS)]
fn production_crowd_gt(bencher: Bencher, &(g, d): &(usize, usize)) {
    let gts = overlapping_rects(g, true);
    let dts = overlapping_rects(d, false);
    let mut out = Array2::<f64>::zeros((g, d));
    bencher.bench_local(|| {
        SegmIou
            .compute(black_box(&gts), black_box(&dts), &mut out.view_mut())
            .unwrap();
        out.fill(0.0);
    });
}

#[divan::bench(args = SPARSE_GRIDS)]
fn production_sparse_overlap(bencher: Bencher, &(g, d): &(usize, usize)) {
    let gts = overlapping_rects(g, false);
    let dts = overlapping_rects(d, false);
    let mut out = Array2::<f64>::zeros((g, d));
    bencher.bench_local(|| {
        SegmIou
            .compute(black_box(&gts), black_box(&dts), &mut out.view_mut())
            .unwrap();
        out.fill(0.0);
    });
}

#[divan::bench(args = SPARSE_GRIDS)]
fn production_sparse_disjoint(bencher: Bencher, &(g, d): &(usize, usize)) {
    let gts = disjoint_rects(g);
    let dts = disjoint_rects(d);
    let mut out = Array2::<f64>::zeros((g, d));
    bencher.bench_local(|| {
        SegmIou
            .compute(black_box(&gts), black_box(&dts), &mut out.view_mut())
            .unwrap();
        out.fill(0.0);
    });
}

#[divan::bench(args = SPARSE_GRIDS)]
fn production_sparse_crowd_gt(bencher: Bencher, &(g, d): &(usize, usize)) {
    let gts = overlapping_rects(g, true);
    let dts = overlapping_rects(d, false);
    let mut out = Array2::<f64>::zeros((g, d));
    bencher.bench_local(|| {
        SegmIou
            .compute(black_box(&gts), black_box(&dts), &mut out.view_mut())
            .unwrap();
        out.fill(0.0);
    });
}
