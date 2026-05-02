//! Shared rect-mask builders for the segm + boundary IoU benches.
//!
//! Both kernels sit on top of `Rle::intersect_area`, so their benches
//! sweep the same `(g, d)` grid over the same rect-mask shapes —
//! keeping a single source of truth here lets the per-bench files
//! focus on what they actually exercise (cost-center commentary,
//! crowd handling, etc.) instead of restating the geometry.
//!
//! Cargo benches each compile as a separate binary, so this module is
//! pulled in via `#[path = "common/mod.rs"] mod common;` rather than a
//! published crate item.

#![allow(dead_code, unreachable_pub, clippy::unwrap_used)]

use vernier_core::similarity::SegmAnn;
use vernier_mask::Rle;

/// Image dimensions. Diagonal ≈ 181 px → at the boundary kernel's
/// default 0.02 dilation ratio, `round(3.62) = 4` — large enough that
/// a 24×24 rect erodes to a non-degenerate 16×16 core and the band is
/// a real frame. The `disjoint_rects` regime uses 8×8 shapes so the
/// boundary kernel's small-mask clamp path activates; segm has no
/// equivalent path but the smaller footprint keeps the bbox-prefilter
/// behaviour identical across both benches.
pub const H: u32 = 128;
pub const W: u32 = 128;

/// Builds an axis-aligned filled rectangle as an Rle (column-major
/// raster → `from_raster_bytes`, the same primitive the production
/// kernel consumes at the FFI boundary).
pub fn filled_rect(x0: u32, y0: u32, rw: u32, rh: u32) -> Rle {
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
/// grid (otherwise the optimizer would CSE intersect/band work
/// across duplicate shapes and bias the measurement). Strides 11 and
/// 13 are coprime with 104 = `W - 24`, so for `n ≤ 100` every shape
/// is unique.
pub fn overlapping_rects(n: usize, is_crowd: bool) -> Vec<SegmAnn> {
    let span = (W - 24) as usize;
    (0..n)
        .map(|i| {
            let x0 = ((i * 11) % span) as u32;
            let y0 = ((i * 13) % span) as u32;
            SegmAnn {
                rle: filled_rect(x0, y0, 24, 24),
                is_crowd,
                ann_id: i as i64,
            }
        })
        .collect()
}

/// `n` rectangles tiled on a 12 px grid so every neighbour pair is
/// bbox-disjoint by construction. The 8×8 shape erodes to empty
/// under the boundary kernel's default ratio (`round(3.62) = 4`);
/// the boundary bench's prefilter-savings measurement is the
/// motivating use, segm just sees plenty of disjoint pairs.
pub fn disjoint_rects(n: usize) -> Vec<SegmAnn> {
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
                ann_id: i as i64,
            }
        })
        .collect()
}
