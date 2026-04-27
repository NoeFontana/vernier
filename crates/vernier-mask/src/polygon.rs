//! Polygon → RLE rasterizer.
//!
//! Mirrors `rleFrPoly` (`mc:193-235`) and `rleFrBbox` (`mc:180-187`)
//! from `pycocotools-2.0.11/common/maskApi.c`. The whole point of
//! this file is to reproduce the C function's pixel-coverage rule,
//! which is itself defined operationally rather than mathematically.
//!
//! Algorithm in three stages:
//!
//! 1. **Upsample (quirk H3).** Each vertex is mapped to the
//!    supersampled grid via `(int)(5x + 0.5)` — rounds toward zero
//!    after the half-bias, matching the C cast bit-for-bit on
//!    in-range inputs.
//! 2. **Densify.** Each edge is walked along its major axis with
//!    a Bresenham-like loop, producing one supersampled point per
//!    minor-axis unit.
//! 3. **Y-boundary downsample (quirks H4, H5).** A "vertical step"
//!    in the dense buffer (`u[j] != u[j-1]`) becomes one
//!    image-coord boundary crossing. The x-coord is selected
//!    direction-aware (the column the boundary just left) and must
//!    land *exactly* on an integer pixel boundary; the y-coord is
//!    always the smaller of the two endpoints, clamped to `[0, h]`,
//!    and ceiled. The asymmetry is what makes shared polygon edges
//!    cancel cleanly under [`Rle::merge`].
//! 4. **Encode (quirk H6).** Crossings are sorted, differential-
//!    encoded, and zero-length runs are folded back into the
//!    previous run — that's how duplicated edges (a polygon's two
//!    sides crossing the same supersampled cell) cancel out.

use crate::error::{MalformedPolygonReason, MalformedRleReason, MaskError};
use crate::rle::Rle;

/// Supersampling factor. Pinned at `5.0` to match `rleFrPoly`.
const SUPERSAMPLE_SCALE: f64 = 5.0;

/// Minimum vertex count vernier accepts. Quirk **K1** disposition
/// `corrected`: the C code happily rasterizes 1- and 2-vertex
/// inputs to garbage; we reject them.
const MIN_VERTICES: usize = 3;

/// Upper bound on supersampled coordinates. Realistic polygons
/// (coords ≤ image dimension) never approach this; the clamp is
/// purely defensive so that downstream `i32` differences in the
/// densification loop cannot overflow on absurd-but-finite input.
const SUPERSAMPLE_COORD_BOUND: i32 = i32::MAX / 4;

impl Rle {
    /// Rasterizes a closed polygon into an RLE.
    ///
    /// `xy` is a flat slice `[x0, y0, x1, y1, ...]` of pixel
    /// coordinates; the first and last vertex are connected
    /// implicitly (no need to repeat the first vertex).
    ///
    /// Errors per quirk **K1** disposition `corrected` on:
    /// - odd-length `xy` (not a sequence of `(x, y)` pairs),
    /// - fewer than 3 vertices,
    /// - any non-finite (`NaN` / `±∞`) coordinate.
    ///
    /// Returns the empty `h × w` RLE for `h == 0 || w == 0`.
    pub fn from_polygon(xy: &[f64], h: u32, w: u32) -> Result<Self, MaskError> {
        if xy.len() % 2 != 0 {
            return Err(MaskError::MalformedPolygon(
                MalformedPolygonReason::OddCoordinateCount(xy.len()),
            ));
        }
        let k = xy.len() / 2;
        if k < MIN_VERTICES {
            return Err(MaskError::MalformedPolygon(
                MalformedPolygonReason::TooFewVertices(k),
            ));
        }
        for (i, &v) in xy.iter().enumerate() {
            if !v.is_finite() {
                return Err(MaskError::MalformedPolygon(
                    MalformedPolygonReason::NonFiniteCoordinate(i),
                ));
            }
        }
        if h == 0 || w == 0 {
            return Ok(Rle {
                h,
                w,
                counts: vec![],
            });
        }

        // Stage 1: upsample. Closing vertex is appended so each edge
        // can be addressed as `[j, j+1]` for `j ∈ 0..k`.
        let mut sx: Vec<i32> = Vec::with_capacity(k + 1);
        let mut sy: Vec<i32> = Vec::with_capacity(k + 1);
        for j in 0..k {
            sx.push(supersample(xy[2 * j]));
            sy.push(supersample(xy[2 * j + 1]));
        }
        sx.push(sx[0]);
        sy.push(sy[0]);

        // Stage 2: densify. Capacity = sum of major-axis lengths +1
        // per edge — matches the C `m` accumulator in `mc:201`.
        let total_dense: usize = (0..k)
            .map(|j| {
                let dx = (sx[j + 1] - sx[j]).unsigned_abs();
                let dy = (sy[j + 1] - sy[j]).unsigned_abs();
                dx.max(dy) as usize + 1
            })
            .sum();
        let mut u: Vec<i32> = Vec::with_capacity(total_dense);
        let mut v: Vec<i32> = Vec::with_capacity(total_dense);
        for j in 0..k {
            densify_edge(sx[j], sx[j + 1], sy[j], sy[j + 1], &mut u, &mut v);
        }

        // Stage 3: y-boundary downsample. `bx`, `by` collect image-
        // coord crossings; capacity is at worst `u.len()`.
        let mut bx: Vec<u32> = Vec::with_capacity(u.len());
        let mut by: Vec<u32> = Vec::with_capacity(u.len());
        let w_max = f64::from(w - 1);
        let h_max = f64::from(h);
        for j in 1..u.len() {
            if u[j] == u[j - 1] {
                continue;
            }
            // x: cell-boundary the polygon just crossed. Direction-
            // aware, per the C ternary in `mc:219`.
            let xd_super = if u[j] < u[j - 1] { u[j] } else { u[j] - 1 };
            let xd = (f64::from(xd_super) + 0.5) / SUPERSAMPLE_SCALE - 0.5;
            // Quirk H5: only crossings exactly on an integer pixel
            // boundary survive; out-of-image crossings are dropped.
            if xd.floor() != xd || xd < 0.0 || xd > w_max {
                continue;
            }
            // y: minimum of the two endpoints, intentionally ignoring
            // direction — quirk H4. Clamp then `ceil`.
            let yd_super = v[j].min(v[j - 1]);
            let yd = (f64::from(yd_super) + 0.5) / SUPERSAMPLE_SCALE - 0.5;
            let yd = yd.clamp(0.0, h_max).ceil();
            // xd is a non-negative integer ≤ w-1, yd ∈ [0, h]; both
            // fit u32. (h, w themselves are u32.)
            bx.push(xd as u32);
            by.push(yd as u32);
        }

        // Stage 4: build sorted column-major flat indices, append
        // the `h*w` sentinel so the final run extends to mask end.
        let h64 = u64::from(h);
        let w64 = u64::from(w);
        let total_pixels = h64 * w64;
        let mut a: Vec<u64> = Vec::with_capacity(bx.len() + 1);
        for i in 0..bx.len() {
            a.push(u64::from(bx[i]) * h64 + u64::from(by[i]));
        }
        a.push(total_pixels);
        a.sort_unstable();

        // Differential encode in place.
        let mut prev: u64 = 0;
        for slot in a.iter_mut() {
            let cur = *slot;
            *slot = cur - prev;
            prev = cur;
        }

        // Compact zero-length runs. Mirrors `mc:231-233`: a zero
        // gap means two crossings landed on the same flat index, so
        // the next run folds back into the most recent emitted run.
        let mut counts: Vec<u32> = Vec::with_capacity(a.len());
        counts.push(u32_from_u64(a[0])?);
        let mut i = 1usize;
        while i < a.len() {
            let cur = a[i];
            i += 1;
            if cur > 0 {
                counts.push(u32_from_u64(cur)?);
                continue;
            }
            // cur == 0: bridge to the next run if one exists.
            if i < a.len() {
                let bridged = a[i];
                i += 1;
                let last_idx = counts.len() - 1;
                let merged = u64::from(counts[last_idx]) + bridged;
                counts[last_idx] = u32_from_u64(merged)?;
            }
        }

        Ok(Rle { h, w, counts })
    }

    /// Rasterizes an axis-aligned bbox `[x, y, w, h]` into an RLE.
    ///
    /// Mirrors `rleFrBbox` (`mc:180-187`): a bbox is a 4-vertex
    /// polygon traced counter-clockwise from its top-left corner,
    /// so this function is a thin wrapper over [`Self::from_polygon`].
    pub fn from_bbox(bbox: [f64; 4], h: u32, w: u32) -> Result<Self, MaskError> {
        let [x, y, bw, bh] = bbox;
        let xs = x;
        let xe = x + bw;
        let ys = y;
        let ye = y + bh;
        Self::from_polygon(&[xs, ys, xs, ye, xe, ye, xe, ys], h, w)
    }
}

fn supersample(v: f64) -> i32 {
    let raw = (SUPERSAMPLE_SCALE * v + 0.5) as i32;
    raw.clamp(-SUPERSAMPLE_COORD_BOUND, SUPERSAMPLE_COORD_BOUND)
}

fn densify_edge(
    mut xs: i32,
    mut xe: i32,
    mut ys: i32,
    mut ye: i32,
    u: &mut Vec<i32>,
    v: &mut Vec<i32>,
) {
    let dx = (xe - xs).unsigned_abs() as i32;
    let dy = (ys - ye).unsigned_abs() as i32;
    // The C `flip` rotates each edge so densification always walks
    // forward along the major axis; the original direction is
    // restored when stamping `t = flip ? major-d : d`. Without this,
    // shared edges between adjacent polygons would emit boundary
    // points in different orders and fail to cancel under H6
    // compaction.
    let flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
    if flip {
        std::mem::swap(&mut xs, &mut xe);
        std::mem::swap(&mut ys, &mut ye);
    }
    if dx >= dy {
        // Degenerate edge (dx == dy == 0): C computes 0/0 = NaN
        // and casts to int (UB). Vernier emits the single-vertex
        // collapse `(xs, ys)` deterministically.
        let s = if dx == 0 {
            0.0
        } else {
            f64::from(ye - ys) / f64::from(dx)
        };
        for d in 0..=dx {
            let t = if flip { dx - d } else { d };
            u.push(t + xs);
            v.push((f64::from(ys) + s * f64::from(t) + 0.5) as i32);
        }
    } else {
        let s = f64::from(xe - xs) / f64::from(dy);
        for d in 0..=dy {
            let t = if flip { dy - d } else { d };
            v.push(t + ys);
            u.push((f64::from(xs) + s * f64::from(t) + 0.5) as i32);
        }
    }
}

fn u32_from_u64(value: u64) -> Result<u32, MaskError> {
    u32::try_from(value).map_err(|_| MaskError::MalformedRle(MalformedRleReason::U32Overflow))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn rle(h: u32, w: u32, counts: Vec<u32>) -> Rle {
        Rle { h, w, counts }
    }

    #[test]
    fn rejects_odd_coordinate_count() {
        let err = Rle::from_polygon(&[0.0, 0.0, 1.0], 4, 4).unwrap_err();
        assert!(matches!(
            err,
            MaskError::MalformedPolygon(MalformedPolygonReason::OddCoordinateCount(3))
        ));
    }

    #[test]
    fn rejects_fewer_than_three_vertices() {
        // Two-vertex polygon — bbox/poly ambiguity (quirk K1).
        let err = Rle::from_polygon(&[0.0, 0.0, 1.0, 1.0], 4, 4).unwrap_err();
        assert!(matches!(
            err,
            MaskError::MalformedPolygon(MalformedPolygonReason::TooFewVertices(2))
        ));
        // Empty input.
        let err = Rle::from_polygon(&[], 4, 4).unwrap_err();
        assert!(matches!(
            err,
            MaskError::MalformedPolygon(MalformedPolygonReason::TooFewVertices(0))
        ));
    }

    #[test]
    fn rejects_nan_and_infinity() {
        let err = Rle::from_polygon(&[0.0, 0.0, f64::NAN, 1.0, 2.0, 2.0], 4, 4).unwrap_err();
        assert!(matches!(
            err,
            MaskError::MalformedPolygon(MalformedPolygonReason::NonFiniteCoordinate(2))
        ));
        let err = Rle::from_polygon(&[0.0, 0.0, 1.0, f64::INFINITY, 2.0, 2.0], 4, 4).unwrap_err();
        assert!(matches!(
            err,
            MaskError::MalformedPolygon(MalformedPolygonReason::NonFiniteCoordinate(3))
        ));
    }

    #[test]
    fn empty_image_returns_empty_rle() {
        let r = Rle::from_polygon(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 0, 0).unwrap();
        assert_eq!(r, rle(0, 0, vec![]));
        let r = Rle::from_polygon(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 0, 4).unwrap();
        assert_eq!(r, rle(0, 4, vec![]));
        let r = Rle::from_polygon(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 4, 0).unwrap();
        assert_eq!(r, rle(4, 0, vec![]));
    }

    #[test]
    fn axis_aligned_2x2_square_in_4x4_image() {
        // Polygon (0,0)-(2,0)-(2,2)-(0,2): traced by hand below in the
        // module docs. Expected: bg=0, fg=2, bg=2, fg=2, bg=10.
        let r = Rle::from_polygon(&[0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0], 4, 4).unwrap();
        assert_eq!(r.counts, vec![0, 2, 2, 2, 10]);
        // Round-trip through the raster codec to confirm the FG
        // pixel set: (0,0), (0,1), (1,0), (1,1) in column-major.
        let mask = r.to_raster_bytes();
        let mut expected = vec![0u8; 16];
        for x in 0..2 {
            for y in 0..2 {
                expected[x * 4 + y] = 1;
            }
        }
        assert_eq!(mask, expected);
    }

    #[test]
    fn from_bbox_matches_explicit_polygon() {
        let bbox = [0.5, 1.0, 2.0, 3.0];
        let from_bbox = Rle::from_bbox(bbox, 8, 8).unwrap();
        let from_poly = Rle::from_polygon(&[0.5, 1.0, 0.5, 4.0, 2.5, 4.0, 2.5, 1.0], 8, 8).unwrap();
        assert_eq!(from_bbox, from_poly);
    }

    #[test]
    fn polygon_entirely_outside_image_is_empty() {
        // All vertices to the left of the image — every crossing
        // gets dropped by the H5 in-bounds filter.
        let r = Rle::from_polygon(&[-5.0, -5.0, -3.0, -5.0, -3.0, -3.0, -5.0, -3.0], 4, 4).unwrap();
        assert_eq!(r.counts, vec![16]);
        assert_eq!(r.area(), 0);
    }

    #[test]
    fn polygon_clipped_to_image_bounds() {
        // Square partly outside on the right — clipped to a 2x4
        // rectangle covering columns x=2 and x=3 (h=4).
        let r = Rle::from_polygon(&[2.0, 0.0, 6.0, 0.0, 6.0, 4.0, 2.0, 4.0], 4, 4).unwrap();
        // Expected mask: columns 0-1 bg, columns 2-3 fg.
        let mask = r.to_raster_bytes();
        let mut expected = vec![0u8; 16];
        for x in 2..4 {
            for y in 0..4 {
                expected[x * 4 + y] = 1;
            }
        }
        assert_eq!(mask, expected);
    }

    #[test]
    fn degenerate_collinear_polygon_does_not_panic() {
        // Three collinear points: the polygon has zero area. The
        // algorithm should produce a valid (empty or near-empty)
        // RLE without panicking.
        let r = Rle::from_polygon(&[0.0, 0.0, 1.0, 0.0, 2.0, 0.0], 4, 4).unwrap();
        assert_eq!(r.h, 4);
        assert_eq!(r.w, 4);
        let total: u64 = r.counts.iter().map(|&c| u64::from(c)).sum();
        assert_eq!(total, 16);
    }

    proptest! {
        #[test]
        fn random_polygon_round_trips_to_well_formed_rle(
            verts in proptest::collection::vec(0.0_f64..16.0_f64, 6..40).prop_filter(
                "even length",
                |v| v.len() % 2 == 0,
            )
        ) {
            let r = Rle::from_polygon(&verts, 16, 16)?;
            // Counts must sum to h*w, and the mask must round-trip.
            let total: u64 = r.counts.iter().map(|&c| u64::from(c)).sum();
            prop_assert_eq!(total, 16 * 16);
            let bytes = r.to_raster_bytes();
            let r2 = Rle::from_raster_bytes(&bytes, 16, 16)?;
            prop_assert_eq!(r, r2);
        }

        #[test]
        fn bbox_polygon_round_trips_via_raster(
            x in 0.0_f64..16.0_f64,
            y in 0.0_f64..16.0_f64,
            bw in 0.5_f64..16.0_f64,
            bh in 0.5_f64..16.0_f64,
        ) {
            let r = Rle::from_bbox([x, y, bw, bh], 16, 16)?;
            let bytes = r.to_raster_bytes();
            prop_assert_eq!(bytes.len(), 16 * 16);
            let r2 = Rle::from_raster_bytes(&bytes, 16, 16)?;
            prop_assert_eq!(r, r2);
        }
    }
}
