//! Minimum-area enclosing rectangle, by rotating calipers.
//!
//! Used for one diagnostic, and it is worth saying which: **how much a
//! rotated-box label format costs you before the model does anything.**
//! DOTA ground truth is annotated as arbitrary quadrilaterals, but most
//! detectors predict rotated *rectangles*. `IoU(quad, minAreaRect(quad))`
//! is therefore the ceiling on what any rectangle-predicting model could
//! score on that annotation — a per-class number that is a property of
//! the dataset, not of anyone's detector.
//!
//! # Not OpenCV-bit-equal, on purpose
//!
//! `cv2.minAreaRect` is a different implementation with its own
//! conventions (and its own 4.5.1 angle-range change). This function
//! makes no parity claim against it and must never be used to build one:
//! the strict oracle replicas are the only things in this
//! crate that reproduce somebody else's bits. A diagnostic that quietly
//! became a parity surface would be the worst of both.
//!
//! # Why an edge-aligned search is exact
//!
//! The minimum-area enclosing rectangle of a convex polygon always has a
//! side collinear with one of the polygon's edges. So enumerating the
//! hull edges — four of them at most, for a quad — finds the optimum
//! exactly, with no iteration and no tolerance.

use crate::convention::{AngleUnit, Convention, RotatedBox};

/// Convex hull of up to eight points, by monotone chain.
///
/// Returns the hull in positively-wound order. Collinear points are
/// dropped: they cannot be a distinct supporting edge, so keeping them
/// would only add duplicate candidates.
fn convex_hull(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut pts: Vec<(f64, f64)> = points.to_vec();
    // `total_cmp`, not `partial_cmp(..).unwrap_or(Equal)`: the latter
    // is not a total order once a `NaN` is present (it reports
    // `NaN == 0`, `NaN == 2` and `0 < 2`), and Rust's sort is entitled
    // to panic on a comparator that contradicts itself. `min_area_rect`
    // rejects non-finite input before reaching here, so this is the
    // second of two locks on the same door rather than the only one.
    pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut lower: Vec<(f64, f64)> = Vec::with_capacity(pts.len());
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<(f64, f64)> = Vec::with_capacity(pts.len());
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Minimum-area enclosing rectangle of a quad, expressed under `conv`.
///
/// Returns `None` for a degenerate input — fewer than three distinct
/// non-collinear vertices — because there is no rectangle to report and
/// a zero-area answer would read as a real one.
#[must_use]
pub fn min_area_rect(quad: &[f64; 8], conv: Convention) -> Option<RotatedBox> {
    // Non-finite input is rejected here rather than allowed to
    // propagate: every downstream step (`hypot`, the extent scan, the
    // final division) would carry the `NaN` through and hand back a
    // rectangle of five `NaN`s that reads like an answer. The callers
    // in `vernier-ffi` turn `None` into a typed Python error.
    if !quad.iter().all(|v| v.is_finite()) {
        return None;
    }
    let pts: Vec<(f64, f64)> = (0..4).map(|i| (quad[2 * i], quad[2 * i + 1])).collect();
    let hull = convex_hull(&pts);
    if hull.len() < 3 {
        return None;
    }

    // Track only the winning axis; the extents are recomputed from it
    // below so the two can never disagree.
    let mut best: Option<(f64, f64, f64)> = None; // (area, ux, uy)
    for i in 0..hull.len() {
        let a = hull[i];
        let b = hull[(i + 1) % hull.len()];
        let (ex, ey) = (b.0 - a.0, b.1 - a.1);
        let len = ex.hypot(ey);
        if !len.is_finite() || len <= 0.0 {
            continue;
        }
        let (ux, uy) = (ex / len, ey / len);
        let (w, h) = extents(&hull, ux, uy);
        let area = w * h;
        if best.is_none_or(|(best_area, _, _)| area < best_area) {
            best = Some((area, ux, uy));
        }
    }

    let (_, ux, uy) = best?;
    let (vx, vy) = (-uy, ux);
    let (u_lo, u_hi, v_lo, v_hi) = bounds(&hull, ux, uy);
    let (w, h) = (u_hi - u_lo, v_hi - v_lo);
    let (cu, cv) = ((u_lo + u_hi) * 0.5, (v_lo + v_hi) * 0.5);
    // Back to image coordinates: `(u, v)` is orthonormal, so the
    // inverse rotation is its transpose.
    let cx = cu * ux + cv * vx;
    let cy = cu * uy + cv * vy;

    // The box's width axis is `u`, and the width axis is by definition
    // `R(sigma*kappa*theta) * (1, 0)`, so `sigma*kappa*theta =
    // atan2(uy, ux)`. Invert for the caller's convention.
    let radians = uy.atan2(ux) * conv.rotation.sin_sign();
    let theta = match conv.unit {
        AngleUnit::Deg => radians.to_degrees(),
        AngleUnit::Rad => radians,
    };
    Some(RotatedBox {
        cx,
        cy,
        w,
        h,
        theta,
    })
}

/// Projection bounds of `hull` on the orthonormal frame `(u, perp(u))`.
fn bounds(hull: &[(f64, f64)], ux: f64, uy: f64) -> (f64, f64, f64, f64) {
    let (vx, vy) = (-uy, ux);
    let (mut u_lo, mut u_hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut v_lo, mut v_hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(px, py) in hull {
        let pu = px * ux + py * uy;
        let pv = px * vx + py * vy;
        u_lo = u_lo.min(pu);
        u_hi = u_hi.max(pu);
        v_lo = v_lo.min(pv);
        v_hi = v_hi.max(pv);
    }
    (u_lo, u_hi, v_lo, v_hi)
}

/// Extents of `hull` on the frame `(u, perp(u))`.
fn extents(hull: &[(f64, f64)], ux: f64, uy: f64) -> (f64, f64) {
    let (u_lo, u_hi, v_lo, v_hi) = bounds(hull, ux, uy);
    (u_hi - u_lo, v_hi - v_lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_finite_input_is_refused_rather_than_propagated() {
        let conv = Convention::new(AngleUnit::Deg, crate::Rotation::ScreenCcw);
        // A `NaN` would make the hull's comparator intransitive, which
        // Rust's sort may punish with a panic; where it does not, it
        // would return a rectangle of five `NaN`s that reads like an
        // answer.
        let bad = [0.0, 0.0, f64::NAN, 1.0, 2.0, 2.0, 3.0, 3.0];
        assert!(min_area_rect(&bad, conv).is_none());
        let inf = [0.0, 0.0, f64::INFINITY, 1.0, 2.0, 2.0, 3.0, 3.0];
        assert!(min_area_rect(&inf, conv).is_none());
    }
    use crate::kernel::Denominator;
    use crate::{PreparedQuad, PreparedRBox};

    const D2: Convention = Convention::D2;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn axis_aligned_rectangle_is_its_own_min_rect() {
        let q = [0.0, 0.0, 4.0, 0.0, 4.0, 2.0, 0.0, 2.0];
        let r = min_area_rect(&q, D2).expect("non-degenerate");
        assert!(close(r.cx, 2.0, 1e-12) && close(r.cy, 1.0, 1e-12), "{r:?}");
        let (long, short) = if r.w >= r.h { (r.w, r.h) } else { (r.h, r.w) };
        assert!(close(long, 4.0, 1e-12) && close(short, 2.0, 1e-12), "{r:?}");
        assert!(close(r.w * r.h, 8.0, 1e-12));
    }

    #[test]
    fn rotated_rectangle_recovers_its_own_angle() {
        let src = RotatedBox {
            cx: 10.0,
            cy: -5.0,
            w: 9.0,
            h: 3.0,
            theta: 25.0,
        };
        let r = min_area_rect(&src.corners(D2), D2).expect("non-degenerate");
        assert!(
            close(r.cx, src.cx, 1e-9) && close(r.cy, src.cy, 1e-9),
            "{r:?}"
        );
        assert!(close(r.w * r.h, src.w * src.h, 1e-9), "{r:?}");
        // The recovered box is the same *shape*, so its IoU with the
        // original is 1 regardless of which parameterization it landed
        // on.
        let a = PreparedRBox::new(src, D2);
        let b = PreparedRBox::new(r, D2);
        assert!(
            close(crate::rbox_iou(&a, &b, D2, Denominator::Union), 1.0, 1e-9),
            "recovered {r:?} from {src:?}"
        );
    }

    #[test]
    fn a_triangle_quad_is_bounded_not_matched() {
        // Degenerate quad: three distinct vertices. The min rect exists
        // and strictly contains it, so the label ceiling is below 1.
        let q = [0.0, 0.0, 4.0, 0.0, 2.0, 3.0, 2.0, 3.0];
        let r = min_area_rect(&q, D2).expect("non-degenerate");
        assert!(close(r.w * r.h, 12.0, 1e-9), "{r:?}");
        let quad = PreparedQuad::new(&q, true).expect("valid");
        let rect = PreparedQuad::new(&r.corners(D2), false).expect("valid");
        let iou = crate::quad_iou(&rect, &quad, Denominator::Union);
        assert!(close(iou, 0.5, 1e-9), "triangle in its min rect: {iou}");
    }

    #[test]
    fn collinear_input_has_no_rectangle() {
        assert!(min_area_rect(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], D2).is_none());
        assert!(min_area_rect(&[1.0; 8], D2).is_none());
    }

    #[test]
    fn conventions_describe_the_same_rectangle() {
        let q = [0.0, 0.0, 6.0, 2.0, 5.0, 5.0, -1.0, 3.0];
        let cw = Convention::new(AngleUnit::Deg, crate::Rotation::ScreenCw);
        let a = min_area_rect(&q, D2).expect("non-degenerate");
        let b = min_area_rect(&q, cw).expect("non-degenerate");
        assert!(close(a.w * a.h, b.w * b.h, 1e-12));
        let pa = PreparedRBox::new(a, D2);
        let pb = PreparedRBox::new(b, cw);
        assert!(close(
            crate::rbox_iou(&pa, &pa, D2, Denominator::Union),
            crate::rbox_iou(&pb, &pb, cw, Denominator::Union),
            0.0
        ));
        // Same shape under either declaration: compare via the quads.
        let qa = PreparedQuad::new(&a.corners(D2), false).expect("valid");
        let qb = PreparedQuad::new(&b.corners(cw), false).expect("valid");
        assert!(close(
            crate::quad_iou(&qa, &qb, Denominator::Union),
            1.0,
            1e-9
        ));
    }

    #[test]
    fn the_rectangle_encloses_every_vertex() {
        let q = [0.0, 0.0, 6.0, 2.0, 5.0, 5.0, -1.0, 3.0];
        let r = min_area_rect(&q, D2).expect("non-degenerate");
        let quad = PreparedQuad::new(&q, false).expect("valid");
        let rect = PreparedQuad::new(&r.corners(D2), false).expect("valid");
        // The quad is entirely inside, so their intersection is the
        // quad's own area.
        let inter = crate::kernel::quad_intersection(&rect, &quad);
        assert!(close(inter, quad.area, 1e-9), "{inter} vs {}", quad.area);
        assert!(r.w * r.h >= quad.area - 1e-9);
    }
}
