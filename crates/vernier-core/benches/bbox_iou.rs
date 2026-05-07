//! Microbenchmarks for the bbox IoU [`Similarity`] kernel.
//!
//! Tracks absolute throughput on representative per-image grid sizes
//! and runs the production kernel (wrapped in `pulp::Arch::dispatch`)
//! alongside a scalar reference living in this file. The delta
//! isolates dispatch overhead from algorithmic changes and gives
//! every later perf-sensitive PR two independent regression
//! baselines.
//!
//! On f64-end-to-end (per ADR-0008) LLVM already auto-vectorizes
//! this loop well, so the production-vs-scalar gap is small or
//! negative on small grids — that is itself useful information.
//!
//! Run with `just bench` or `cargo bench -p vernier-core --bench bbox_iou`.

#![allow(clippy::unwrap_used)]

use divan::{black_box, Bencher};
use ndarray::{Array2, ArrayViewMut2};
use vernier_core::dataset::Bbox;
use vernier_core::similarity::{BboxAnn, BboxIou, Similarity};

fn main() {
    divan::main();
}

/// Per-image grid sizes (gts × dts). Real COCO eval is dominated by
/// 4–30 GTs per image and ≤100 DTs after maxDet truncation; the
/// 100×1000 row stresses an LVIS-scale image.
const GRIDS: &[(usize, usize)] = &[(4, 100), (10, 100), (30, 100), (100, 1000)];

/// Dense-regime grids for single-category-ish workloads (200–300 GT/DT
/// per image, 1–5 categories — surveillance, single-class detection,
/// dense crowd scenes). Per
/// `docs/engineering/benchmarking/2026-05-bbox-cdf.md`, these cells
/// have `G·D ≥ 5,000` so inner-loop work dominates per-call setup; the
/// `production`-vs-`scalar_reference` gap on this row is the headroom
/// any explicit-SIMD pass needs to beat.
const DENSE_GRIDS: &[(usize, usize)] = &[(50, 50), (100, 100), (200, 200), (300, 300)];

fn make_anns(n: usize, offset: f64, is_crowd: bool) -> Vec<BboxAnn> {
    (0..n)
        .map(|i| {
            let f = (i as f64).mul_add(13.0, offset);
            BboxAnn {
                bbox: Bbox {
                    x: f.rem_euclid(800.0),
                    y: ((i as f64).mul_add(17.0, offset)).rem_euclid(800.0),
                    w: 50.0,
                    h: 60.0,
                },
                is_crowd,
            }
        })
        .collect()
}

#[divan::bench(args = GRIDS)]
fn production(bencher: Bencher, &(g, d): &(usize, usize)) {
    let gts = make_anns(g, 0.31, false);
    let dts = make_anns(d, 0.71, false);
    let mut out = Array2::<f64>::zeros((g, d));
    bencher.bench_local(|| {
        BboxIou
            .compute(black_box(&gts), black_box(&dts), &mut out.view_mut())
            .unwrap();
    });
}

#[divan::bench(args = GRIDS)]
fn production_crowd_gt(bencher: Bencher, &(g, d): &(usize, usize)) {
    // Exercises the E1 asymmetric branch: GT.is_crowd=true selects the
    // intersect/dt_area denom, a different inner-loop than the
    // symmetric union form covered by `production`.
    let gts = make_anns(g, 0.31, true);
    let dts = make_anns(d, 0.71, false);
    let mut out = Array2::<f64>::zeros((g, d));
    bencher.bench_local(|| {
        BboxIou
            .compute(black_box(&gts), black_box(&dts), &mut out.view_mut())
            .unwrap();
    });
}

#[divan::bench(args = GRIDS)]
fn scalar_reference(bencher: Bencher, &(g, d): &(usize, usize)) {
    let gts = make_anns(g, 0.31, false);
    let dts = make_anns(d, 0.71, false);
    let mut out = Array2::<f64>::zeros((g, d));
    bencher.bench_local(|| {
        scalar_compute(black_box(&gts), black_box(&dts), &mut out.view_mut());
    });
}

#[divan::bench(args = DENSE_GRIDS)]
fn production_dense(bencher: Bencher, &(g, d): &(usize, usize)) {
    let gts = make_anns(g, 0.31, false);
    let dts = make_anns(d, 0.71, false);
    let mut out = Array2::<f64>::zeros((g, d));
    bencher.bench_local(|| {
        BboxIou
            .compute(black_box(&gts), black_box(&dts), &mut out.view_mut())
            .unwrap();
    });
}

#[divan::bench(args = DENSE_GRIDS)]
fn scalar_reference_dense(bencher: Bencher, &(g, d): &(usize, usize)) {
    let gts = make_anns(g, 0.31, false);
    let dts = make_anns(d, 0.71, false);
    let mut out = Array2::<f64>::zeros((g, d));
    bencher.bench_local(|| {
        scalar_compute(black_box(&gts), black_box(&dts), &mut out.view_mut());
    });
}

/// Scalar reference: same algorithm, no `pulp::Arch::dispatch`.
/// Apples-to-apples on the inner loop so the production-vs-this delta
/// isolates dispatch overhead from the inner-loop work.
fn scalar_compute(gts: &[BboxAnn], dts: &[BboxAnn], out: &mut ArrayViewMut2<'_, f64>) {
    for (g, gt) in gts.iter().enumerate() {
        let gxa = gt.bbox.x;
        let gya = gt.bbox.y;
        let gxb = gxa + gt.bbox.w;
        let gyb = gya + gt.bbox.h;
        let g_area = gt.bbox.w * gt.bbox.h;
        let mut row = out.row_mut(g);
        for (d, dt) in dts.iter().enumerate() {
            let dxa = dt.bbox.x;
            let dya = dt.bbox.y;
            let dxb = dxa + dt.bbox.w;
            let dyb = dya + dt.bbox.h;
            let d_area = dt.bbox.w * dt.bbox.h;
            let iw = (gxb.min(dxb) - gxa.max(dxa)).max(0.0);
            let ih = (gyb.min(dyb) - gya.max(dya)).max(0.0);
            let inter = iw * ih;
            let denom = g_area + d_area - inter;
            row[d] = if denom > 0.0 { inter / denom } else { 0.0 };
        }
    }
}
