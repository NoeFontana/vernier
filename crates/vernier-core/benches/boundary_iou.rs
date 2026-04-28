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
use vernier_core::similarity::{BoundaryIou, SegmAnn, Similarity};
use vernier_mask::Rle;

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

/// Image dimensions. Diagonal ≈ 181 px → at the default 0.02
/// dilation ratio, `round(3.62) = 4` — large enough that a 24×24
/// rect erodes to a non-degenerate 16×16 core and the band is a
/// real frame, not the whole mask. The `production_disjoint`
/// regime uses 8×8 shapes so the small-mask clamp path activates;
/// see the comment on `disjoint_rects` for the implication.
const H: u32 = 128;
const W: u32 = 128;

/// Builds an axis-aligned filled rectangle as an Rle (column-
/// major raster → `from_raster_bytes`, the same primitive the
/// production kernel consumes at the FFI boundary).
fn filled_rect(x0: u32, y0: u32, rw: u32, rh: u32) -> Rle {
    let mut raster = vec![0u8; (H as usize) * (W as usize)];
    for x in x0..x0 + rw {
        for y in y0..y0 + rh {
            raster[(x as usize) * (H as usize) + (y as usize)] = 1;
        }
    }
    Rle::from_raster_bytes(&raster, H, W).unwrap()
}

/// `n` overlapping 24×24 rectangles arranged on coprime strides
/// against the wrap modulus so positions don't repeat across the
/// grid (otherwise the optimizer would CSE band precomputes
/// across duplicate shapes and bias the measurement). Strides 11
/// and 13 are coprime with 104 = `W - 24`, so for the largest
/// configuration (n = 100) every shape is unique.
fn overlapping_rects(n: usize, is_crowd: bool) -> Vec<SegmAnn> {
    let span = (W - 24) as usize;
    (0..n)
        .map(|i| {
            let x0 = ((i * 11) % span) as u32;
            let y0 = ((i * 13) % span) as u32;
            SegmAnn {
                rle: filled_rect(x0, y0, 24, 24),
                is_crowd,
            }
        })
        .collect()
}

/// `n` rectangles tiled on a 12 px grid so every neighbour pair
/// is bbox-disjoint by construction. The 8×8 shape erodes to
/// empty under the default ratio (`round(3.62) = 4`), so the
/// band collapses onto the full mask via the small-mask clamp —
/// fine: the bench's job here is to expose how much per-cell
/// (mask + band) intersect work the I1 prefilter avoids, not to
/// stress the band precompute (`production_overlap` does that).
fn disjoint_rects(n: usize) -> Vec<SegmAnn> {
    let cols = ((n as f64).sqrt().ceil() as u32).max(1);
    (0..n as u32)
        .map(|i| {
            let cx = i % cols;
            let cy = i / cols;
            let x0 = (cx * 12).min(W - 9);
            let y0 = (cy * 12).min(H - 9);
            SegmAnn {
                rle: filled_rect(x0, y0, 8, 8),
                is_crowd: false,
            }
        })
        .collect()
}

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
