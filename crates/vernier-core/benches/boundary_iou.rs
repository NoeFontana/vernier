//! Microbenchmarks for the boundary IoU [`Similarity`] kernel.
//!
//! The kernel has three cost centers (see comments in
//! `crates/vernier-core/src/similarity/boundary.rs`):
//!
//! 1. **bbox prefilter (I1)** — one [`BboxIou::compute`] call over
//!    the tight RLE bboxes, zeroing cells that can't overlap.
//! 2. **per-annotation band precompute** — one [`boundary_band`]
//!    call per non-crowd annotation; cost is linear in mask area
//!    and runs *unconditionally* per side, so it stays constant
//!    across the overlap/disjoint regimes.
//! 3. **per-cell finishing work** — for cells that survive (1),
//!    two [`Rle::intersect_area`] calls (mask, band) plus the
//!    `min` fold.
//!
//! The three benches below isolate (3)'s contribution by holding
//! shape and grid fixed and varying overlap density:
//!
//! - **production_overlap**: dense overlap; most cells survive (1)
//!   so per-cell finishing dominates. Worst case in practice.
//! - **production_disjoint**: tiled non-overlapping shapes; (1)
//!   zeros most cells so per-cell finishing is amortized down to
//!   the prefilter alone. Band precompute (2) is *not* avoided —
//!   the gap to `production_overlap` measures (3) in isolation.
//! - **production_crowd_gt**: GT-side `is_crowd=true` triggers the
//!   O1/O2 short-circuit, which skips both (3)'s band intersect
//!   and (2)'s GT-side band precompute. The gap to
//!   `production_overlap` upper-bounds the savings any future
//!   work to parallelise (2) could match.
//!
//! Run with `just bench` or `cargo bench -p vernier-core --bench boundary_iou`.
//!
//! [`BboxIou::compute`]: vernier_core::similarity::BboxIou::compute
//! [`boundary_band`]: vernier_mask::ops::boundary_band
//! [`Rle::intersect_area`]: vernier_mask::Rle::intersect_area

#![allow(clippy::unwrap_used)]

use divan::{black_box, Bencher};
use ndarray::Array2;
use vernier_core::similarity::{BoundaryIou, Similarity};

#[path = "common/mod.rs"]
mod common;
use common::{disjoint_rects, overlapping_rects};

fn main() {
    divan::main();
}

/// Per-image grid sizes (gts × dts). Boundary IoU is materially
/// heavier than bbox IoU (per-annotation band precompute + two RLE
/// intersects per surviving cell), so the LVIS-scale (100, 1000)
/// row the bbox bench includes would push individual measurements
/// past divan's default budget. The 4-30 GT × 100 DT band tracks
/// the real COCO eval distribution.
const GRIDS: &[(usize, usize)] = &[(4, 100), (10, 100), (30, 100)];

#[divan::bench(args = GRIDS)]
fn production_overlap(bencher: Bencher, &(g, d): &(usize, usize)) {
    let kernel = BoundaryIou::default();
    let gts = overlapping_rects(g, false);
    let dts = overlapping_rects(d, false);
    let mut out = Array2::<f64>::zeros((g, d));
    bencher.bench_local(|| {
        kernel
            .compute(black_box(&gts), black_box(&dts), &mut out.view_mut())
            .unwrap();
        // Re-zero so each iteration starts from a known initial
        // state. The kernel overwrites every cell today, so this
        // doesn't change the answer; it documents the invariant
        // and stops a future regression from amplifying across
        // iterations.
        out.fill(0.0);
    });
}

#[divan::bench(args = GRIDS)]
fn production_disjoint(bencher: Bencher, &(g, d): &(usize, usize)) {
    let kernel = BoundaryIou::default();
    let gts = disjoint_rects(g);
    let dts = disjoint_rects(d);
    let mut out = Array2::<f64>::zeros((g, d));
    bencher.bench_local(|| {
        kernel
            .compute(black_box(&gts), black_box(&dts), &mut out.view_mut())
            .unwrap();
        out.fill(0.0);
    });
}

#[divan::bench(args = GRIDS)]
fn production_crowd_gt(bencher: Bencher, &(g, d): &(usize, usize)) {
    // O1/O2 short-circuit: crowd GTs skip both the GT-side band
    // precompute and the per-cell band intersect. The savings
    // here vs `production_overlap` upper-bounds what
    // parallelising the band precompute could buy.
    let kernel = BoundaryIou::default();
    let gts = overlapping_rects(g, true);
    let dts = overlapping_rects(d, false);
    let mut out = Array2::<f64>::zeros((g, d));
    bencher.bench_local(|| {
        kernel
            .compute(black_box(&gts), black_box(&dts), &mut out.view_mut())
            .unwrap();
        out.fill(0.0);
    });
}
