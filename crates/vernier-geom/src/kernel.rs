//! The canonical (`corrected`) IoU kernels.
//!
//! One f64 kernel serves both geometries. It clips in the **ground
//! truth's own frame**, which is what keeps the error budget
//! independent of where in the image the objects are: the only absolute
//! quantity that enters is `delta = c_d - c_g`, whose rounding is
//! bounded by the pair's diameter `L` rather than by the image extent
//! `X`.
//!
//! # Error bound
//!
//! With `L` the longest side in the pair and `r` the aspect ratio of the
//! box carrying it, the union obeys `U >= L^2 / r`, and
//! `d(IoU)/d(I) <= 2/U`, so
//!
//! ```text
//! |dIoU| <= (2/U)|dI| <= (2r/L^2) * c2 * eps * L^2 = 2*c2*eps*r
//! ```
//!
//! At `eps = 2^-53` and `r = 1000` that is around `1e-13 * c2`,
//! independent of image size. An f32 absolute-frame shoelace at
//! DOTA scale (`X = 2e4`) would round each term by up to
//! `u32 * X^2 ~ 24 px^2` — the area of a small vehicle. D2's
//! pair-midpoint shift exists for exactly this reason; DK gets away
//! without one by being f64 throughout.
//!
//! # Exactness
//!
//! - `IoU(a, a) = 1.0` **bit-exactly** for every non-degenerate `a`.
//!   `delta = 0` and `phi = 0` give corners of exactly `(+-w/2, +-h/2)`;
//!   the clip is a no-op because the classification is inclusive; and
//!   [`crate::clip::Poly::twice_signed_area`]'s fan form returns exactly
//!   `2 * fl(w*h)`.
//! - A pair the broad phase rejects returns exactly `+0.0`, never a
//!   signed zero and never a denormal residue.
//! - When `theta_g` and `theta_d` differ by a multiple of 90 degrees the
//!   detection is *exactly* axis-aligned in the ground truth's frame,
//!   because [`crate::convention::sin_cos_deg`] is exact on the
//!   quadrants. The result then agrees with an axis-aligned bbox IoU to
//!   within the last couple of ULP.

use crate::broad::sat_disjoint;
use crate::clip::{clip_to_centered_box, Poly};
use crate::convention::Convention;
use crate::prepared::PreparedRBox;
use crate::quad::PreparedQuad;

/// Which denominator the ratio uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Denominator {
    /// `I / (A_g + A_d - I)` — the ordinary case.
    Union,
    /// `I / A_d` — quirk **E1**, the crowd-ground-truth asymmetry. A
    /// small detection fully inside a large crowd region scores `1.0`.
    /// The asymmetry lives in the kernel so the matching engine stays
    /// IoU-type-agnostic (ADR-0005).
    CrowdGt,
}

/// Finish the ratio from an intersection area and the two areas.
///
/// Shared by both geometries so the clamping story is told once:
/// `I` is clamped into `[0, min(A_g, A_d)]` before it can reach the
/// denominator, a non-positive denominator yields `0.0` (the **I3**
/// analog — `pycocotools` spells the same guard two different ways for
/// RLE and bbox), and the result is clamped into `[0, 1]`.
///
/// Defensive against arguments the crate's own callers cannot produce:
/// `f64::clamp` asserts `min <= max`, so a negative or `NaN` area would
/// panic. `PreparedRBox` and `PreparedQuad` both clamp their areas at
/// zero, but `ratio` is `pub` in a `pub mod`, and a panic crossing the
/// FFI boundary is not an acceptable answer to a bad argument in a
/// workspace whose lints deny `panic`. The `> 0.0` tests map `NaN` to
/// the zero branch, which also discharges the **OB7** no-`NaN` matrix
/// contract at the one place every ratio passes through.
#[inline]
#[must_use]
pub fn ratio(intersection: f64, area_g: f64, area_d: f64, denom: Denominator) -> f64 {
    let cap = if area_g < area_d { area_g } else { area_d };
    let cap = if cap > 0.0 { cap } else { 0.0 };
    let i = if intersection > 0.0 {
        if intersection < cap {
            intersection
        } else {
            cap
        }
    } else {
        0.0
    };
    let u = match denom {
        Denominator::Union => area_g + area_d - i,
        Denominator::CrowdGt => area_d,
    };
    if u > 0.0 {
        (i / u).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Corners of `d` expressed in `g`'s frame, positively wound.
///
/// `delta` is rotated by `R(-a_g)` and the detection's own half-axes by
/// `R(phi)`, `phi = sigma*kappa*(theta_d - theta_g)`. Every term is
/// `O(L)`; nothing in this function has seen an image coordinate since
/// the subtraction on the first line.
#[must_use]
pub fn detection_in_gt_frame(g: &PreparedRBox, d: &PreparedRBox, conv: Convention) -> Poly {
    let dx = d.raw.cx - g.raw.cx;
    let dy = d.raw.cy - g.raw.cy;
    // R(-a_g) * delta.
    let tx = dx * g.cos + dy * g.sin;
    let ty = -dx * g.sin + dy * g.cos;

    let (sin_p, cos_p) = conv.sin_cos(d.raw.theta - g.raw.theta);
    let (ux, uy) = (cos_p * d.hw, sin_p * d.hw);
    let (vx, vy) = (-sin_p * d.hh, cos_p * d.hh);

    let mut p = Poly::new();
    p.push(tx + ux + vx, ty + uy + vy);
    p.push(tx - ux + vx, ty - uy + vy);
    p.push(tx - ux - vx, ty - uy - vy);
    p.push(tx + ux - vx, ty + uy - vy);
    p
}

/// Intersection area of two rotated boxes, canonical bits.
#[must_use]
pub fn rbox_intersection(g: &PreparedRBox, d: &PreparedRBox, conv: Convention) -> f64 {
    if g.is_degenerate() || d.is_degenerate() {
        return 0.0;
    }
    if !g
        .aabb
        .overlaps(&d.aabb, crate::broad::pair_slack(&g.aabb, &d.aabb))
    {
        return 0.0;
    }
    if sat_disjoint(g, d) {
        return 0.0;
    }
    let subject = detection_in_gt_frame(g, d, conv);
    clip_to_centered_box(&subject, g.hw, g.hh).area()
}

/// IoU of two rotated boxes, canonical bits.
#[must_use]
pub fn rbox_iou(g: &PreparedRBox, d: &PreparedRBox, conv: Convention, denom: Denominator) -> f64 {
    let i = rbox_intersection(g, d, conv);
    if i == 0.0 {
        // Keep the exact `+0.0` the admissibility contract promises
        // rather than routing a zero through a division.
        return 0.0;
    }
    ratio(i, g.area, d.area, denom)
}

/// Clip a convex subject polygon against a convex, positively-wound clip
/// polygon.
#[must_use]
pub fn clip_convex(subject: &Poly, clip: &Poly) -> Poly {
    let mut cur = *subject;
    let n = clip.len();
    for i in 0..n {
        if cur.is_empty() {
            return cur;
        }
        let j = if i + 1 == n { 0 } else { i + 1 };
        let Some((ax, ay)) = clip.vertex(i) else {
            return Poly::new();
        };
        let Some((bx, by)) = clip.vertex(j) else {
            return Poly::new();
        };
        cur = cur.clip_halfplane(ax, ay, bx, by);
    }
    cur
}

/// Intersection area of two validated quads, canonical bits.
///
/// Both operands are decomposed into at most two convex pieces, so this
/// is at most four convex-convex clips. The accumulation order —
/// detection piece outer, ground-truth piece inner — is fixed so the
/// result is reproducible.
#[must_use]
pub fn quad_intersection(g: &PreparedQuad, d: &PreparedQuad) -> f64 {
    let (gx0, gy0, gx1, gy1) = g.aabb;
    let (dx0, dy0, dx1, dy1) = d.aabb;
    if gx0 > dx1 || dx0 > gx1 || gy0 > dy1 || dy0 > gy1 {
        return 0.0;
    }
    let tx = d.cx - g.cx;
    let ty = d.cy - g.cy;

    let mut total = 0.0;
    for dp in d.pieces() {
        // Translate the detection piece into the ground truth's
        // centroid frame. Both operands were stored centroid-relative,
        // so this is the only place an absolute coordinate appears.
        let mut shifted = Poly::new();
        for i in 0..dp.len() {
            if let Some((x, y)) = dp.vertex(i) {
                shifted.push(x + tx, y + ty);
            }
        }
        for gp in g.pieces() {
            total += clip_convex(&shifted, gp).area();
        }
    }
    total
}

/// IoU of two validated quads, canonical bits.
#[must_use]
pub fn quad_iou(g: &PreparedQuad, d: &PreparedQuad, denom: Denominator) -> f64 {
    let i = quad_intersection(g, d);
    if i == 0.0 {
        return 0.0;
    }
    ratio(i, g.area, d.area, denom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convention::{AngleUnit, Convention, RotatedBox, Rotation};

    const D2: Convention = Convention::D2;

    #[test]
    fn ratio_never_panics_on_an_impossible_area() {
        // `f64::clamp` asserts `min <= max`; these are the arguments
        // that would trip it. None can come from a `Prepared*`, which
        // is exactly why the guard has to be tested directly.
        for bad in [-1.0, f64::NAN, f64::NEG_INFINITY] {
            assert_eq!(ratio(1.0, bad, 4.0, Denominator::Union), 0.0, "{bad}");
            assert_eq!(ratio(1.0, 4.0, bad, Denominator::Union), 0.0, "{bad}");
            assert_eq!(ratio(bad, 4.0, 4.0, Denominator::Union), 0.0, "{bad}");
        }
        // And the ordinary case still rounds the way it always did.
        assert_eq!(ratio(2.0, 4.0, 4.0, Denominator::Union), 2.0 / 6.0);
        assert_eq!(ratio(9.0, 4.0, 4.0, Denominator::Union), 1.0);
    }

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

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn self_iou_is_exactly_one() {
        let cases = [
            (0.0, 0.0, 4.0, 2.0, 0.0),
            (1234.5, -987.25, 13.0, 7.0, 37.5),
            (1e5, 1e5, 0.5, 0.25, -163.75),
            (0.0, 0.0, 1e4, 3.0, 45.0),
            (7.0, 9.0, 3.0, 3.0, 90.0),
            (7.0, 9.0, 3.0, 3.0, 180.0),
            (2e4, 2e4, 1.0 / 3.0, 7.0 / 11.0, 21.0),
        ];
        for (cx, cy, w, h, t) in cases {
            let b = rb(cx, cy, w, h, t);
            let got = rbox_iou(&b, &b, D2, Denominator::Union);
            assert_eq!(got, 1.0, "IoU(a, a) for ({cx}, {cy}, {w}, {h}, {t})");
        }
    }

    #[test]
    fn self_iou_is_exactly_one_in_radians() {
        let conv = Convention::new(AngleUnit::Rad, Rotation::ScreenCw);
        for t in [0.0, 0.3, -1.7, 3.0] {
            let b = PreparedRBox::new(
                RotatedBox {
                    cx: 40.0,
                    cy: -12.0,
                    w: 9.0,
                    h: 4.0,
                    theta: t,
                },
                conv,
            );
            assert_eq!(rbox_iou(&b, &b, conv, Denominator::Union), 1.0, "theta={t}");
        }
    }

    #[test]
    fn axis_aligned_half_overlap() {
        let g = rb(0.0, 0.0, 2.0, 2.0, 0.0);
        let d = rb(1.0, 0.0, 2.0, 2.0, 0.0);
        // Intersection 2, union 6.
        assert_eq!(rbox_iou(&g, &d, D2, Denominator::Union), 2.0 / 6.0);
    }

    #[test]
    fn ninety_degree_rotation_is_exact() {
        // A square rotated by 90 degrees is the same square.
        let g = rb(3.0, 5.0, 4.0, 4.0, 0.0);
        let d = rb(3.0, 5.0, 4.0, 4.0, 90.0);
        assert_eq!(rbox_iou(&g, &d, D2, Denominator::Union), 1.0);
        // And a w/h swap plus 90 degrees is the same rectangle.
        let g = rb(3.0, 5.0, 6.0, 2.0, 0.0);
        let d = rb(3.0, 5.0, 2.0, 6.0, 90.0);
        assert_eq!(rbox_iou(&g, &d, D2, Denominator::Union), 1.0);
    }

    #[test]
    fn disjoint_is_positive_zero() {
        let g = rb(0.0, 0.0, 2.0, 2.0, 15.0);
        let d = rb(100.0, 100.0, 2.0, 2.0, 15.0);
        let v = rbox_iou(&g, &d, D2, Denominator::Union);
        assert_eq!(v, 0.0);
        assert!(!v.is_sign_negative());
        assert_eq!(v.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn touching_edges_score_zero() {
        let g = rb(0.0, 0.0, 2.0, 2.0, 0.0);
        let d = rb(2.0, 0.0, 2.0, 2.0, 0.0);
        assert_eq!(rbox_iou(&g, &d, D2, Denominator::Union), 0.0);
    }

    #[test]
    fn degenerate_boxes_score_zero() {
        let g = rb(0.0, 0.0, 0.0, 2.0, 0.0);
        let d = rb(0.0, 0.0, 2.0, 2.0, 0.0);
        assert_eq!(rbox_iou(&g, &d, D2, Denominator::Union), 0.0);
        assert_eq!(rbox_iou(&d, &g, D2, Denominator::Union), 0.0);
    }

    #[test]
    fn crowd_denominator_saturates() {
        let g = rb(0.0, 0.0, 100.0, 100.0, 0.0);
        let d = rb(1.0, 1.0, 2.0, 2.0, 0.0);
        assert_eq!(rbox_iou(&g, &d, D2, Denominator::CrowdGt), 1.0);
        assert!(rbox_iou(&g, &d, D2, Denominator::Union) < 0.001);
    }

    #[test]
    fn rotated_square_diamond_matches_closed_form() {
        // Unit-side square against the same square rotated 45 degrees.
        // Intersection of two congruent squares sharing a center, one
        // rotated by 45 degrees, is a regular octagon of area
        // 2*(sqrt(2) - 1)*s^2.
        let s = 3.0;
        let g = rb(0.0, 0.0, s, s, 0.0);
        let d = rb(0.0, 0.0, s, s, 45.0);
        let want_i = 2.0 * (core::f64::consts::SQRT_2 - 1.0) * s * s;
        let got_i = rbox_intersection(&g, &d, D2);
        assert!(close(got_i, want_i, 1e-12), "{got_i} vs {want_i}");
        let want = want_i / (2.0 * s * s - want_i);
        assert!(close(rbox_iou(&g, &d, D2, Denominator::Union), want, 1e-12));
    }

    #[test]
    fn far_from_origin_is_as_accurate_as_at_origin() {
        let near_g = rb(0.0, 0.0, 30.0, 12.0, 23.0);
        let near_d = rb(4.0, -3.0, 25.0, 16.0, -11.0);
        let off = 2.0e4;
        let far_g = rb(off, off, 30.0, 12.0, 23.0);
        let far_d = rb(off + 4.0, off - 3.0, 25.0, 16.0, -11.0);
        let a = rbox_iou(&near_g, &near_d, D2, Denominator::Union);
        let b = rbox_iou(&far_g, &far_d, D2, Denominator::Union);
        assert!(close(a, b, 1e-12), "{a} vs {b}");
    }

    // --- quads ---

    fn pq(v: [f64; 8]) -> PreparedQuad {
        PreparedQuad::new(&v, false).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn quad_self_iou_is_exactly_one() {
        for v in [
            [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
            [
                1e4,
                1e4,
                1e4 + 3.0,
                1e4 + 1.0,
                1e4 + 2.0,
                1e4 + 5.0,
                1e4 - 1.0,
                1e4 + 3.0,
            ],
            // Non-convex arrowhead: two triangles.
            [0.0, 0.0, 4.0, 0.0, 2.0, 4.0, 2.0, 1.0],
        ] {
            let q = pq(v);
            assert_eq!(quad_iou(&q, &q, Denominator::Union), 1.0, "{v:?}");
        }
    }

    #[test]
    fn quad_matches_rbox_on_a_rectangle_pair() {
        let g = rb(0.0, 0.0, 6.0, 4.0, 0.0);
        let d = rb(2.0, 1.0, 6.0, 4.0, 0.0);
        let gq = pq(g.raw.corners(D2));
        let dq = pq(d.raw.corners(D2));
        let a = rbox_iou(&g, &d, D2, Denominator::Union);
        let b = quad_iou(&gq, &dq, Denominator::Union);
        assert!(close(a, b, 1e-14), "{a} vs {b}");
    }

    #[test]
    fn quad_matches_rbox_on_a_rotated_pair() {
        let g = rb(10.0, -4.0, 9.0, 3.0, 27.5);
        let d = rb(12.0, -2.0, 5.0, 7.0, -40.0);
        let gq = pq(g.raw.corners(D2));
        let dq = pq(d.raw.corners(D2));
        let a = rbox_iou(&g, &d, D2, Denominator::Union);
        let b = quad_iou(&gq, &dq, Denominator::Union);
        assert!(close(a, b, 1e-12), "{a} vs {b}");
    }

    #[test]
    fn quad_disjoint_is_zero() {
        let a = pq([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let b = pq([10.0, 10.0, 11.0, 10.0, 11.0, 11.0, 10.0, 11.0]);
        assert_eq!(quad_iou(&a, &b, Denominator::Union), 0.0);
    }

    #[test]
    fn non_convex_quad_against_a_square() {
        // Arrowhead (area 4) inside a 10x10 square: intersection is the
        // whole arrowhead.
        let arrow = pq([0.0, 0.0, 4.0, 0.0, 2.0, 4.0, 2.0, 1.0]);
        let square = pq([-5.0, -5.0, 5.0, -5.0, 5.0, 5.0, -5.0, 5.0]);
        let i = quad_intersection(&square, &arrow);
        assert!(close(i, arrow.area, 1e-12), "{i} vs {}", arrow.area);
    }

    #[test]
    fn ratio_guards() {
        assert_eq!(ratio(0.0, 0.0, 0.0, Denominator::Union), 0.0);
        assert_eq!(ratio(5.0, 1.0, 1.0, Denominator::Union), 1.0);
        assert_eq!(ratio(-1.0, 2.0, 2.0, Denominator::Union), 0.0);
        assert_eq!(ratio(1.0, 4.0, 0.0, Denominator::CrowdGt), 0.0);
    }
}
