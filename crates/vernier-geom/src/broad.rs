//! Broad phase: which pairs can possibly overlap.
//!
//! ADR-0063's cost model has the broad phase as the only genuinely
//! quadratic term at DOTA scale, so it is the part that gets the SIMD
//! budget. The narrow phase runs on `K = O(G + D)` survivors and stays
//! scalar until a profile says otherwise.
//!
//! # Admissibility
//!
//! A prefilter `F` is *admissible* for a kernel `K` when
//!
//! ```text
//! F(g, d) = reject   =>   bits(K(g, d)) = bits(+0.0)
//! ```
//!
//! — rejecting is only allowed where the kernel would have produced an
//! exact positive zero anyway. Every test here therefore errs toward
//! keeping: comparisons are non-strict, and every bound carries a
//! relative slack that covers its own rounding. A false keep costs a
//! narrow-phase call; a false reject would be a wrong IoU.
//!
//! The DK-strict path does **not** use these filters: its only
//! admissible prefilter is DOTA_devkit's own horizontal-box test with
//! the VOC `+1`, which lives with the DK replica because it is
//! part of the composed oracle rather than an optimization.

use crate::prepared::{Aabb, PreparedRBox};

/// Relative slack applied to every derived bound.
///
/// Sized against the actual error terms rather than picked for looking
/// generous. Both filters compare a separation against a sum of radii,
/// and every quantity in that comparison is a short rounded expression
/// in the *scale* each filter passes in:
///
/// - a half-extent is `|cos| * hw + |sin| * hh` — two products and a
///   sum, so at most 3 ULP of the larger side;
/// - a projection radius is the same shape over a dot product of two
///   unit vectors, so at most 5 ULP once the axis components are
///   counted;
/// - the separation is `|dx * ax + dy * ay|`, two products and a sum
///   over a displacement, so at most 3 ULP of `max(|dx|, |dy|)`.
///
/// Worst case the two sides of the comparison are therefore ~13 ULP
/// apart relative to `max(half-extents, |dx|, |dy|)`, and 16 ULP covers
/// it with room left. That last term is why the scale includes the
/// center separation: bounding the displacement's rounding by the
/// half-extents alone would be unsound for a pair whose centers are far
/// apart relative to their sizes, which is the common case in a sparse
/// cell.
///
/// The budget is what makes the documented contract — a reject implies
/// the kernel would return `bits(+0.0)` — a statement about arithmetic
/// rather than a hope.
pub const SLACK: f64 = 16.0 * f64::EPSILON;

/// Pad appropriate for a pair of envelopes: relative to their own
/// magnitudes, never absolute.
#[inline]
#[must_use]
pub fn pair_slack(a: &Aabb, b: &Aabb) -> f64 {
    let scale = (a.max_x - a.min_x)
        .max(a.max_y - a.min_y)
        .max(b.max_x - b.min_x)
        .max(b.max_y - b.min_y)
        .max(a.min_x.abs().max(b.min_x.abs()))
        .max(a.min_y.abs().max(b.min_y.abs()));
    if scale.is_finite() {
        scale * SLACK
    } else {
        0.0
    }
}

/// Separating-axis test for two rotated rectangles.
///
/// `true` means *provably disjoint*: some axis separates the two boxes
/// by more than the sum of their projected radii plus slack. Pays for
/// itself on thin diagonal objects — ships at a berth, aircraft on an
/// apron — where the axis-aligned envelopes of two 45-degree boxes
/// overlap almost completely while the boxes themselves do not.
#[must_use]
pub fn sat_disjoint(g: &PreparedRBox, d: &PreparedRBox) -> bool {
    let dx = d.raw.cx - g.raw.cx;
    let dy = d.raw.cy - g.raw.cy;
    // The displacement is part of the scale: `dist`'s own rounding is
    // relative to `|dx|`/`|dy|`, not to the half-extents. See [`SLACK`].
    let scale =
        g.hw.abs()
            .max(g.hh.abs())
            .max(d.hw.abs())
            .max(d.hh.abs())
            .max(dx.abs())
            .max(dy.abs());
    let pad = if scale.is_finite() {
        scale * SLACK
    } else {
        0.0
    };

    // Axis family: g's own axes, then d's. Each is a unit vector, so a
    // projection radius is just the half-extents weighted by the axis
    // dot products.
    let axes = [
        (g.cos, g.sin),
        (-g.sin, g.cos),
        (d.cos, d.sin),
        (-d.sin, d.cos),
    ];
    for (ax, ay) in axes {
        let dist = (dx * ax + dy * ay).abs();
        // `|hw|`, `|hh|`: a negative side would shrink the projection
        // radius and could separate a pair that is not separated. See
        // `RotatedBox::aabb_half_extents` for why sign-flipped boxes
        // reach here at all.
        let rg = g.hw.abs() * (g.cos * ax + g.sin * ay).abs()
            + g.hh.abs() * (-g.sin * ax + g.cos * ay).abs();
        let rd = d.hw.abs() * (d.cos * ax + d.sin * ay).abs()
            + d.hh.abs() * (-d.sin * ax + d.cos * ay).abs();
        if dist > rg + rd + pad {
            return true;
        }
    }
    false
}

/// Detection envelopes transposed into structure-of-arrays form.
///
/// The broad phase is the only genuinely quadratic term in the kernel,
/// and it is the part that gets the SIMD budget. Vectorizing it means
/// holding one ground truth's four bounds in registers and sweeping the
/// detections, which needs the detection bounds in four contiguous
/// lanes — an array of [`Aabb`] would make every load a stride-4
/// gather and give most of the win back.
///
/// One allocation, not four: the four columns are slices of a single
/// `4 * D` buffer.
#[derive(Debug, Clone, Default)]
pub struct DtEnvelopes {
    buf: Vec<f64>,
    n: usize,
}

impl DtEnvelopes {
    /// Transpose `aabbs` into columns.
    #[must_use]
    pub fn new(aabbs: &[Aabb]) -> Self {
        let n = aabbs.len();
        let mut buf = Vec::with_capacity(4 * n);
        buf.extend(aabbs.iter().map(|a| a.min_x));
        buf.extend(aabbs.iter().map(|a| a.min_y));
        buf.extend(aabbs.iter().map(|a| a.max_x));
        buf.extend(aabbs.iter().map(|a| a.max_y));
        Self { buf, n }
    }

    /// Number of detections.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Is the set empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// `(min_x, min_y, max_x, max_y)` columns.
    #[must_use]
    fn columns(&self) -> (&[f64], &[f64], &[f64], &[f64]) {
        let n = self.n;
        let (min_x, rest) = self.buf.split_at(n);
        let (min_y, rest) = rest.split_at(n);
        let (max_x, max_y) = rest.split_at(n);
        (min_x, min_y, max_x, max_y)
    }
}

/// Write a `G x D` row-major survivor mask: `1` where the envelopes can
/// overlap, `0` where they provably cannot.
///
/// Returns `false` without touching `out` when `out.len()` is not
/// `gts.len() * dts.len()`.
///
/// Bit-identical to [`Aabb::overlaps`] pair by pair, deliberately: the
/// four comparisons are written in exactly the same associativity, so
/// the vector path and the scalar path reject exactly the same pairs
/// and the admissibility argument is made once. In particular
/// `d.min_x - pad <= g.max_x` is *not* rearranged into
/// `d.min_x <= g.max_x + pad`, which is the same inequality in the
/// reals and a different one in binary64.
#[must_use]
pub fn aabb_overlap_mask(gts: &[Aabb], dts: &DtEnvelopes, pad: f64, out: &mut [u8]) -> bool {
    let Some(total) = gts.len().checked_mul(dts.len()) else {
        return false;
    };
    if out.len() != total {
        return false;
    }
    if total == 0 {
        return true;
    }
    // Same small-cell carve-out as the bbox kernel: below this the
    // `pulp` dispatch boundary costs more than the loop it guards.
    // Bit-identical either way — the body introduces no FMA and no
    // reassociation, so scalar and every SIMD target agree (ADR-0047,
    // "one wheel, one behavior").
    if total < SMALL_CELL_THRESHOLD_OBB {
        mask_inner(gts, dts, pad, out);
    } else {
        arch().dispatch(|| mask_inner(gts, dts, pad, out));
    }
    true
}

/// Below this `G*D` the `pulp` dispatch boundary dominates the loop it
/// guards.
///
/// Provisional: ADR-0063 M5 PR-5.1 re-measures it from the OBB Stage-0
/// histogram on DOTA-v1 / v2 val, the way `SMALL_CELL_THRESHOLD` was
/// measured for bbox. The bbox value is the starting point because the
/// dispatch cost is a property of the boundary, not of the body.
pub const SMALL_CELL_THRESHOLD_OBB: usize = 32;

#[inline(always)]
fn mask_inner(gts: &[Aabb], dts: &DtEnvelopes, pad: f64, out: &mut [u8]) {
    let n = dts.len();
    let (dmin_x, dmin_y, dmax_x, dmax_y) = dts.columns();
    for (g, row) in gts.iter().zip(out.chunks_exact_mut(n)) {
        // Hoisted: the two bounds that depend only on the ground truth.
        let gmin_x = g.min_x - pad;
        let gmin_y = g.min_y - pad;
        let gmax_x = g.max_x;
        let gmax_y = g.max_y;
        for i in 0..n {
            // `&`, not `&&`: no short circuit, so the body is
            // straight-line and the loop vectorizes. All four
            // comparisons are cheap and none of them can fault.
            let hit = (gmin_x <= dmax_x[i])
                & (dmin_x[i] - pad <= gmax_x)
                & (gmin_y <= dmax_y[i])
                & (dmin_y[i] - pad <= gmax_y);
            row[i] = u8::from(hit);
        }
    }
}

/// Process-wide [`pulp::Arch`] cache, as in the bbox kernel: `Arch::new`
/// runs `cpuid`-gated feature detection, which is not something to pay
/// per cell.
fn arch() -> &'static pulp::Arch {
    use std::sync::OnceLock;
    static ARCH: OnceLock<pulp::Arch> = OnceLock::new();
    ARCH.get_or_init(pulp::Arch::new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convention::{Convention, RotatedBox};

    const D2: Convention = Convention::D2;

    fn rb(cx: f64, cy: f64, w: f64, h: f64, t: f64) -> PreparedRBox {
        PreparedRBox::new(
            RotatedBox {
                cx,
                cy,
                w,
                h,
                theta: t,
            },
            D2,
        )
    }

    #[test]
    fn sat_separates_diagonal_neighbours() {
        // Two thin 45-degree boxes side by side: envelopes overlap
        // heavily, boxes do not touch.
        let a = rb(0.0, 0.0, 20.0, 1.0, 45.0);
        let b = rb(6.0, 6.0, 20.0, 1.0, 45.0);
        assert!(a.aabb.overlaps(&b.aabb, 0.0), "envelopes should overlap");
        assert!(sat_disjoint(&a, &b), "SAT should separate them");
    }

    #[test]
    fn sat_keeps_overlapping_pairs() {
        let a = rb(0.0, 0.0, 10.0, 4.0, 30.0);
        let b = rb(1.0, 1.0, 10.0, 4.0, -30.0);
        assert!(!sat_disjoint(&a, &b));
    }

    #[test]
    fn sat_keeps_identical_boxes() {
        let a = rb(100.0, 100.0, 7.0, 3.0, 17.5);
        assert!(!sat_disjoint(&a, &a));
    }

    #[test]
    fn sat_keeps_touching_boxes() {
        let a = rb(0.0, 0.0, 2.0, 2.0, 0.0);
        let b = rb(2.0, 0.0, 2.0, 2.0, 0.0);
        assert!(!sat_disjoint(&a, &b));
    }

    #[test]
    fn mask_shape_is_enforced() {
        let g = [Aabb::new(0.0, 0.0, 1.0, 1.0)];
        let d = DtEnvelopes::new(&[Aabb::new(0.0, 0.0, 1.0, 1.0); 3]);
        let mut out = [0_u8; 2];
        assert!(!aabb_overlap_mask(&g, &d, 0.0, &mut out));
        let mut out = [0_u8; 3];
        assert!(aabb_overlap_mask(&g, &d, 0.0, &mut out));
        assert_eq!(out, [1, 1, 1]);
    }

    #[test]
    fn mask_agrees_with_scalar_on_a_large_cell() {
        let gts: Vec<Aabb> = (0..40)
            .map(|i| {
                let f = f64::from(i);
                Aabb::new(f, f * 0.5, f + 3.0, f * 0.5 + 2.0)
            })
            .collect();
        let dts: Vec<Aabb> = (0..37)
            .map(|i| {
                let f = f64::from(i) * 1.3;
                Aabb::new(f - 1.0, f * 0.4, f + 1.0, f * 0.4 + 3.0)
            })
            .collect();
        let soa = DtEnvelopes::new(&dts);
        let mut simd = vec![0_u8; gts.len() * dts.len()];
        assert!(aabb_overlap_mask(&gts, &soa, 0.0, &mut simd));
        let mut scalar = vec![0_u8; gts.len() * dts.len()];
        mask_inner(&gts, &soa, 0.0, &mut scalar);
        assert_eq!(simd, scalar);
    }

    /// The claim the whole prefilter rests on: the vectorized mask and
    /// `Aabb::overlaps` agree pair for pair, including at a padded
    /// boundary where the two spellings of the comparison could round
    /// apart.
    #[test]
    fn mask_is_bit_equal_to_the_scalar_predicate() {
        let mut rng = 0x243f_6a88_85a3_08d3_u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
        };
        let boxes = |n: usize, f: &mut dyn FnMut() -> f64| -> Vec<Aabb> {
            (0..n)
                .map(|_| {
                    let (x, y) = (f() * 1e4 - 5e3, f() * 1e4 - 5e3);
                    let (w, h) = (f() * 40.0, f() * 40.0);
                    Aabb::new(x, y, x + w, y + h)
                })
                .collect()
        };
        let gts = boxes(23, &mut next);
        let dts = boxes(29, &mut next);
        for pad in [0.0, 1e-12, 16.0 * f64::EPSILON * 5e3, 1.0] {
            let mut mask = vec![0_u8; gts.len() * dts.len()];
            assert!(aabb_overlap_mask(
                &gts,
                &DtEnvelopes::new(&dts),
                pad,
                &mut mask
            ));
            for (gi, g) in gts.iter().enumerate() {
                for (di, d) in dts.iter().enumerate() {
                    assert_eq!(
                        mask[gi * dts.len() + di] == 1,
                        g.overlaps(d, pad),
                        "pair ({gi}, {di}) at pad {pad}"
                    );
                }
            }
        }
    }

    #[test]
    fn empty_cells_are_fine() {
        let mut out: [u8; 0] = [];
        assert!(aabb_overlap_mask(
            &[],
            &DtEnvelopes::new(&[]),
            0.0,
            &mut out
        ));
    }
}
