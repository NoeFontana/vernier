//! Orientation error between two rotated boxes.
//!
//! IoU is blind to *how* a detection is wrong: a box that is well
//! localized but 90 degrees out can score the same as one that is
//! roughly placed and correctly oriented. This is the separator.
//!
//! **Status**: a standalone diagnostic, reachable as
//! `vernier.instance.obb.angle_error_deg`. ADR-0063 *anticipates* an
//! `angle_err_deg` column on the `per_pair` result table (ADR-0019),
//! and that column does not exist yet — nothing in `tables.rs` emits
//! it. Said plainly here because a reader wiring up per-pair
//! diagnostics would otherwise go looking for it.
//!
//! # Why it needs a quotient, not a subtraction
//!
//! `theta` is not observable; only the box is. `P(c, w, h, theta)` is
//! invariant under `theta -> theta + pi`, and under
//! `(w, h, theta) -> (h, w, theta + pi/2)`. A raw angular difference
//! would therefore report a 90-degree error for two *identical* boxes
//! written under different conventions — exactly the failure the column
//! exists to detect. Reducing modulo the box's own symmetry removes it.
//!
//! The modulus depends on the shape. An elongated box has a
//! distinguishable long axis, so its symmetry is `pi`; a near-square one
//! does not, and its symmetry is `pi/2`. The changeover is governed by
//! [`NEAR_SQUARE_TAU`].

use crate::convention::Convention;
use crate::prepared::PreparedRBox;

/// Relative side difference below which a box counts as square.
///
/// `|w - h| / max(w, h) < tau`. ADR-0063 proposes `0.05` and flags the
/// value as an open question: it is a reporting threshold, not a
/// parity-carrying constant, so it can move without an oracle
/// conversation — but it moves for everyone at once, which is why it is
/// named here rather than defaulted at each call site.
pub const NEAR_SQUARE_TAU: f64 = 0.05;

/// Direction of the box's long axis, in radians, under `conv`.
fn long_axis(b: &PreparedRBox, conv: Convention) -> f64 {
    let kappa = match conv.unit {
        crate::convention::AngleUnit::Deg => crate::convention::DEG_TO_RAD,
        crate::convention::AngleUnit::Rad => 1.0,
    };
    let a = b.raw.theta * kappa * conv.rotation.sin_sign();
    if b.raw.w >= b.raw.h {
        a
    } else {
        a + core::f64::consts::FRAC_PI_2
    }
}

/// Is either box square enough that its long axis is not meaningful?
fn near_square(b: &PreparedRBox, tau: f64) -> bool {
    let (w, h) = (b.raw.w.abs(), b.raw.h.abs());
    let m = w.max(h);
    if !m.is_finite() || m <= 0.0 {
        return true;
    }
    (w - h).abs() / m < tau
}

/// Orientation error in **degrees**, in `[0, 90]`.
///
/// Symmetric in its arguments, invariant to angle parameterization, and
/// independent of the IoU. Returns `0.0` when either box is degenerate,
/// since an orientation claim about a zero-area box is not meaningful.
#[must_use]
pub fn angle_error_deg(g: &PreparedRBox, d: &PreparedRBox, conv: Convention, tau: f64) -> f64 {
    if g.is_degenerate() || d.is_degenerate() {
        return 0.0;
    }
    let modulus = if near_square(g, tau) || near_square(d, tau) {
        core::f64::consts::FRAC_PI_2
    } else {
        core::f64::consts::PI
    };
    let diff = long_axis(d, conv) - long_axis(g, conv);
    let k = (diff / modulus).round();
    let e = (diff - k * modulus).abs();
    e.to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convention::{AngleUnit, RotatedBox, Rotation};

    const D2: Convention = Convention::D2;

    fn rb(w: f64, h: f64, t: f64) -> PreparedRBox {
        PreparedRBox::new(
            RotatedBox {
                cx: 0.0,
                cy: 0.0,
                w,
                h,
                theta: t,
            },
            D2,
        )
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn identical_boxes_have_zero_error() {
        let b = rb(10.0, 3.0, 37.0);
        assert!(close(angle_error_deg(&b, &b, D2, NEAR_SQUARE_TAU), 0.0));
    }

    #[test]
    fn thirty_degrees_apart() {
        let g = rb(10.0, 3.0, 10.0);
        let d = rb(10.0, 3.0, 40.0);
        assert!(close(angle_error_deg(&g, &d, D2, NEAR_SQUARE_TAU), 30.0));
    }

    #[test]
    fn is_symmetric() {
        let g = rb(10.0, 3.0, 10.0);
        let d = rb(10.0, 3.0, 65.0);
        assert!(close(
            angle_error_deg(&g, &d, D2, NEAR_SQUARE_TAU),
            angle_error_deg(&d, &g, D2, NEAR_SQUARE_TAU)
        ));
    }

    #[test]
    fn invariant_to_the_pi_symmetry() {
        let g = rb(10.0, 3.0, 20.0);
        let d = rb(10.0, 3.0, 200.0);
        assert!(close(angle_error_deg(&g, &d, D2, NEAR_SQUARE_TAU), 0.0));
    }

    #[test]
    fn invariant_to_the_w_h_swap_reparameterization() {
        // The same physical box, written the other way round.
        let g = rb(10.0, 3.0, 20.0);
        let d = rb(3.0, 10.0, 110.0);
        assert!(
            close(angle_error_deg(&g, &d, D2, NEAR_SQUARE_TAU), 0.0),
            "{}",
            angle_error_deg(&g, &d, D2, NEAR_SQUARE_TAU)
        );
    }

    #[test]
    fn near_square_boxes_use_the_quarter_turn_modulus() {
        // A square rotated by 90 degrees is the same square, so the
        // error must be 0 and not 90.
        let g = rb(5.0, 5.0, 0.0);
        let d = rb(5.0, 5.0, 90.0);
        assert!(close(angle_error_deg(&g, &d, D2, NEAR_SQUARE_TAU), 0.0));
        // An elongated pair 90 degrees apart really is 90 degrees off.
        let g = rb(10.0, 2.0, 0.0);
        let d = rb(10.0, 2.0, 90.0);
        assert!(close(angle_error_deg(&g, &d, D2, NEAR_SQUARE_TAU), 90.0));
    }

    #[test]
    fn radians_convention_agrees_with_degrees() {
        let deg = Convention::new(AngleUnit::Deg, Rotation::ScreenCcw);
        let rad = Convention::new(AngleUnit::Rad, Rotation::ScreenCcw);
        let gd = PreparedRBox::new(
            RotatedBox {
                cx: 0.0,
                cy: 0.0,
                w: 8.0,
                h: 2.0,
                theta: 15.0,
            },
            deg,
        );
        let dd = PreparedRBox::new(
            RotatedBox {
                cx: 0.0,
                cy: 0.0,
                w: 8.0,
                h: 2.0,
                theta: 55.0,
            },
            deg,
        );
        let gr = PreparedRBox::new(
            RotatedBox {
                cx: 0.0,
                cy: 0.0,
                w: 8.0,
                h: 2.0,
                theta: 15.0_f64.to_radians(),
            },
            rad,
        );
        let dr = PreparedRBox::new(
            RotatedBox {
                cx: 0.0,
                cy: 0.0,
                w: 8.0,
                h: 2.0,
                theta: 55.0_f64.to_radians(),
            },
            rad,
        );
        let a = angle_error_deg(&gd, &dd, deg, NEAR_SQUARE_TAU);
        let b = angle_error_deg(&gr, &dr, rad, NEAR_SQUARE_TAU);
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        assert!(close(a, 40.0));
    }

    #[test]
    fn degenerate_boxes_report_zero() {
        let g = rb(0.0, 3.0, 10.0);
        let d = rb(10.0, 3.0, 80.0);
        assert_eq!(angle_error_deg(&g, &d, D2, NEAR_SQUARE_TAU), 0.0);
    }

    #[test]
    fn error_never_exceeds_ninety() {
        for t in (0..360).step_by(7) {
            let g = rb(10.0, 3.0, 0.0);
            let d = rb(10.0, 3.0, f64::from(t));
            let e = angle_error_deg(&g, &d, D2, NEAR_SQUARE_TAU);
            assert!((0.0..=90.0 + 1e-9).contains(&e), "t={t} -> {e}");
        }
    }
}
