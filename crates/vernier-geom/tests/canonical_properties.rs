#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr
)]

//! Property gates for the canonical kernel (ADR-0063 M2).
//!
//! The canonical kernel has no oracle to be bit-equal to — it exists
//! precisely because both oracles are defective. What it has instead is
//! a list of things that must be true of *any* correct answer, and an
//! independent second opinion.
//!
//! The second opinion is the **DK replica**. That is not circular: DK
//! decomposes both polygons into triangles hinged on the coordinate
//! origin and sums signed areas, while the canonical kernel clips one
//! rectangle against the other in the ground truth's own frame. The two
//! share no code, no intermediate representation and no failure mode, so
//! agreement to `1e-9` on ten thousand random pairs is real evidence.
//! (DK is also f64 throughout, which is why it can serve here and the
//! f32 D2 replica cannot.)

use proptest::prelude::*;
use vernier_geom::clip::{clip_to_centered_box, MAX_VERTICES};
use vernier_geom::kernel::{detection_in_gt_frame, rbox_intersection};
use vernier_geom::{replica, Convention, Denominator, PreparedQuad, PreparedRBox, RotatedBox};

const CONV: Convention = Convention::D2;

fn prep(cx: f64, cy: f64, w: f64, h: f64, t: f64) -> PreparedRBox {
    PreparedRBox::new(
        RotatedBox {
            cx,
            cy,
            w,
            h,
            theta: t,
        },
        CONV,
    )
}

/// Boxes at a scale where a naive f32 kernel would already be losing
/// whole pixels of area, but with sides that keep the aspect ratio
/// within the `r <= 1e3` band ADR-0063's error bound is stated for.
fn box_strategy() -> impl Strategy<Value = (f64, f64, f64, f64, f64)> {
    (
        -500.0_f64..500.0,
        -500.0_f64..500.0,
        0.25_f64..120.0,
        0.25_f64..120.0,
        -360.0_f64..360.0,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4000))]

    /// ADR-0063's headline exactness claim, on arbitrary input.
    #[test]
    fn self_iou_is_exactly_one((cx, cy, w, h, t) in box_strategy()) {
        let b = prep(cx, cy, w, h, t);
        prop_assert_eq!(
            vernier_geom::rbox_iou(&b, &b, CONV, Denominator::Union),
            1.0
        );
    }

    /// The point set is invariant under `theta -> theta + 180` and under
    /// `(w, h, theta) -> (h, w, theta + 90)`, so the IoU must be too.
    ///
    /// Note what is *not* claimed: bit-equality. Two separate things
    /// stand in the way, and both are properties of the reparameterization
    /// rather than defects in the kernel.
    ///
    /// 1. `theta + 90.0` is a rounded addition. For a general `theta`
    ///    the reparameterized angle differs from the exact one by an
    ///    ULP, so the relative angle is no longer an exact quadrant and
    ///    the trigonometry is not exactly related.
    /// 2. Even with exact angles, the quarter-turn maps the detection's
    ///    half-axes onto each other — `u' = v`, `v' = -u` — which
    ///    permutes the corner sequence cyclically. The shoelace is
    ///    summed as a fan from vertex 0, so a different start vertex is
    ///    a different (equally valid) rounding of the same area.
    ///
    /// A few ULP is therefore the honest bound, and it is the bound the
    /// error analysis predicts.
    #[test]
    fn iou_is_invariant_under_the_parameterization_quotient(
        (gx, gy, gw, gh, gt) in box_strategy(),
        (dx, dy, dw, dh, dt) in box_strategy(),
    ) {
        let g = prep(gx, gy, gw, gh, gt);
        let d = prep(dx, dy, dw, dh, dt);
        let base = vernier_geom::rbox_iou(&g, &d, CONV, Denominator::Union);

        // ADR-0063's error bound is `|dIoU| <= 2*c2*eps*r`, absolute
        // and scaled by the pair's aspect ratio `r` — not relative to
        // the IoU. A sliver detection against a fat ground truth has
        // `r` in the hundreds, and that is exactly where the bound is
        // doing work, so the tolerance has to be written the same way
        // the analysis is.
        let ratio = |w: f64, h: f64| (w.max(h) / w.min(h)).max(1.0);
        let r = ratio(gw, gh).max(ratio(dw, dh));
        let tol = 64.0 * f64::EPSILON * r;

        let half_turn = vernier_geom::rbox_iou(
            &g,
            &prep(dx, dy, dw, dh, dt + 180.0),
            CONV,
            Denominator::Union,
        );
        prop_assert!(
            (half_turn - base).abs() <= tol,
            "half turn: {half_turn} vs {base} (tol {tol}, r {r})"
        );

        let swapped = vernier_geom::rbox_iou(
            &g,
            &prep(dx, dy, dh, dw, dt + 90.0),
            CONV,
            Denominator::Union,
        );
        prop_assert!(
            (swapped - base).abs() <= tol,
            "w/h swap: {swapped} vs {base} (tol {tol}, r {r})"
        );
    }

    /// Swapping the two arguments must not change a union-denominator
    /// IoU. It does change which frame the clip happens in, so this is a
    /// genuine check on the geometry rather than a tautology — the two
    /// answers agree to rounding, not bit for bit.
    #[test]
    fn iou_is_symmetric_under_the_union_denominator(
        (gx, gy, gw, gh, gt) in box_strategy(),
        (dx, dy, dw, dh, dt) in box_strategy(),
    ) {
        let g = prep(gx, gy, gw, gh, gt);
        let d = prep(dx, dy, dw, dh, dt);
        let a = vernier_geom::rbox_iou(&g, &d, CONV, Denominator::Union);
        let b = vernier_geom::rbox_iou(&d, &g, CONV, Denominator::Union);
        prop_assert!((a - b).abs() <= 1e-12, "{a} vs {b}");
    }

    #[test]
    fn iou_stays_in_the_unit_interval(
        (gx, gy, gw, gh, gt) in box_strategy(),
        (dx, dy, dw, dh, dt) in box_strategy(),
    ) {
        let g = prep(gx, gy, gw, gh, gt);
        let d = prep(dx, dy, dw, dh, dt);
        let v = vernier_geom::rbox_iou(&g, &d, CONV, Denominator::Union);
        prop_assert!(v.is_finite());
        prop_assert!((0.0..=1.0).contains(&v), "{v}");
    }

    /// Rectangle against rectangle tops out at eight vertices. ADR-0063
    /// sets the gate at twelve, well inside the 32-slot buffer, so a
    /// regression shows up as a test failure rather than as silently
    /// dropped vertices.
    #[test]
    fn clipped_polygons_stay_small(
        (gx, gy, gw, gh, gt) in box_strategy(),
        (dx, dy, dw, dh, dt) in box_strategy(),
    ) {
        let g = prep(gx, gy, gw, gh, gt);
        let d = prep(dx, dy, dw, dh, dt);
        let clipped = clip_to_centered_box(&detection_in_gt_frame(&g, &d, CONV), g.hw, g.hh);
        prop_assert!(clipped.len() <= 12, "{} vertices", clipped.len());
        prop_assert!(clipped.len() < MAX_VERTICES);
    }

    /// Second opinion: DK's origin-hinged triangle fan, on the same
    /// geometry expressed as quads.
    #[test]
    fn canonical_agrees_with_the_dk_oracle(
        (gx, gy, gw, gh, gt) in box_strategy(),
        (dx, dy, dw, dh, dt) in box_strategy(),
    ) {
        let g = prep(gx, gy, gw, gh, gt);
        let d = prep(dx, dy, dw, dh, dt);
        let ours = vernier_geom::rbox_iou(&g, &d, CONV, Denominator::Union);

        let gq = g.raw.corners(CONV);
        let dq = d.raw.corners(CONV);
        let (theirs, _) = replica::dk::iou_poly_raw(&gq, &dq);
        prop_assert!(theirs.is_finite());
        prop_assert!((ours - theirs).abs() <= 1e-9, "{ours} vs dk {theirs}");
    }

    /// The quad path and the rotated-box path are different code —
    /// general half-planes against four axis-aligned stages — and must
    /// still agree when handed the same rectangles.
    #[test]
    fn quad_path_agrees_with_the_rbox_path(
        (gx, gy, gw, gh, gt) in box_strategy(),
        (dx, dy, dw, dh, dt) in box_strategy(),
    ) {
        let g = prep(gx, gy, gw, gh, gt);
        let d = prep(dx, dy, dw, dh, dt);
        let ours = vernier_geom::rbox_iou(&g, &d, CONV, Denominator::Union);

        let gq = PreparedQuad::new(&g.raw.corners(CONV), false);
        let dq = PreparedQuad::new(&d.raw.corners(CONV), false);
        let (Ok(gq), Ok(dq)) = (gq, dq) else {
            // Degenerate corner sets are rejected at ingestion by
            // design; the rbox path returns 0 for them.
            return Ok(());
        };
        let theirs = vernier_geom::quad_iou(&gq, &dq, Denominator::Union);
        prop_assert!((ours - theirs).abs() <= 1e-9, "{ours} vs quad {theirs}");
    }

    /// Admissibility of the canonical broad phase: a pair it rejects
    /// must be one the narrow phase would have scored exactly `+0.0`.
    /// Biased toward near-contact, which is the only regime where the
    /// question is interesting.
    #[test]
    fn broad_phase_only_rejects_genuinely_disjoint_pairs(
        (gx, gy, gw, gh, gt) in box_strategy(),
        (dw, dh, dt) in (0.25_f64..120.0, 0.25_f64..120.0, -360.0_f64..360.0),
        nudge_x in -1.5_f64..1.5,
        nudge_y in -1.5_f64..1.5,
    ) {
        let g = prep(gx, gy, gw, gh, gt);
        // Place the detection just far enough to graze.
        let reach = (gw + gh + dw + dh) * 0.25;
        let d = prep(gx + reach * nudge_x, gy + reach * nudge_y, dw, dh, dt);

        if vernier_geom::broad::sat_disjoint(&g, &d)
            || !g.aabb.overlaps(&d.aabb, vernier_geom::broad::pair_slack(&g.aabb, &d.aabb))
        {
            // Rejected: the unfiltered clip must confirm zero area.
            let clipped =
                clip_to_centered_box(&detection_in_gt_frame(&g, &d, CONV), g.hw, g.hh);
            prop_assert_eq!(clipped.area(), 0.0, "rejected pair had positive area");
        } else {
            // Kept: nothing to prove, but the result must still be sane.
            prop_assert!(rbox_intersection(&g, &d, CONV) >= 0.0);
        }
    }

    /// When the detection's angle differs from the ground truth's by an
    /// exact multiple of 90 degrees, the detection must come out
    /// *exactly* axis-aligned in the ground truth's frame — opposite
    /// edges sharing a coordinate to the bit, not to a tolerance. That
    /// is what `sin_cos_deg`'s quadrant reduction buys, and it is why
    /// the canonical kernel degenerates gracefully into an axis-aligned
    /// bbox IoU instead of almost doing so.
    ///
    /// The base angle is drawn from whole degrees on purpose: for a
    /// general `theta`, `theta + 90.0` rounds, and the *difference* is
    /// then a hair off 90 rather than exactly 90. The claim is about the
    /// relative angle being an exact quadrant, not about how the caller
    /// happened to arrive at it.
    #[test]
    fn quadrant_offsets_reduce_to_axis_aligned_iou(
        (gx, gy, gw, gh) in (-500.0_f64..500.0, -500.0_f64..500.0, 0.25_f64..120.0, 0.25_f64..120.0),
        whole_degrees in -360_i32..=360,
        (dx, dy, dw, dh) in (-500.0_f64..500.0, -500.0_f64..500.0, 0.25_f64..120.0, 0.25_f64..120.0),
        quadrant in 0_i32..4,
    ) {
        let gt = f64::from(whole_degrees);
        let g = prep(gx, gy, gw, gh, gt);
        let d = prep(dx, dy, dw, dh, gt + f64::from(quadrant) * 90.0);
        let poly = detection_in_gt_frame(&g, &d, CONV);
        // In the ground truth's frame the detection must be exactly
        // axis-aligned: opposite edges share a coordinate to the bit.
        let (p0, p1) = (poly.vertex(0).unwrap(), poly.vertex(1).unwrap());
        let (p2, p3) = (poly.vertex(2).unwrap(), poly.vertex(3).unwrap());
        let aligned = (p0.1 == p1.1 && p2.1 == p3.1 && p0.0 == p3.0 && p1.0 == p2.0)
            || (p0.0 == p1.0 && p2.0 == p3.0 && p0.1 == p3.1 && p1.1 == p2.1);
        prop_assert!(aligned, "{p0:?} {p1:?} {p2:?} {p3:?}");
    }
}
