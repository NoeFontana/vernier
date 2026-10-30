//! Per-box work hoisted out of the `O(G*D)` pair loop.
//!
//! In a `(image, category)` cell with `G` ground truths and `D`
//! detections there are `G*D` pairs but only `G + D` boxes, so every
//! trigonometric call, half-extent and area belongs here. ADR-0063's
//! cost model makes this explicit: the narrow phase runs on `K = O(G+D)`
//! surviving pairs, while preparation is `O(G+D)` by construction and
//! the broad phase is the only genuinely quadratic term.

use crate::convention::{Convention, RotatedBox};

/// An axis-aligned envelope.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    /// Lower `x` bound.
    pub min_x: f64,
    /// Lower `y` bound.
    pub min_y: f64,
    /// Upper `x` bound.
    pub max_x: f64,
    /// Upper `y` bound.
    pub max_y: f64,
}

impl Aabb {
    /// Envelope from explicit bounds.
    #[inline]
    #[must_use]
    pub const fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    /// Envelope from a center and half-extents.
    #[inline]
    #[must_use]
    pub fn centered(cx: f64, cy: f64, ex: f64, ey: f64) -> Self {
        Self::new(cx - ex, cy - ey, cx + ex, cy + ey)
    }

    /// Do the two envelopes overlap, allowing a slack of `pad` on every
    /// side?
    ///
    /// Non-strict (`<=`): a pair that merely touches survives the
    /// prefilter and is resolved by the narrow phase, which returns an
    /// exact `+0.0` for it. Erring toward *keeping* pairs is what makes
    /// the filter admissible — a false reject would be a wrong IoU,
    /// while a false keep only costs time.
    #[inline]
    #[must_use]
    pub fn overlaps(&self, other: &Self, pad: f64) -> bool {
        self.min_x - pad <= other.max_x
            && other.min_x - pad <= self.max_x
            && self.min_y - pad <= other.max_y
            && other.min_y - pad <= self.max_y
    }

    /// COCO-style `w * h` of the envelope, clamped at zero.
    #[inline]
    #[must_use]
    pub fn area(&self) -> f64 {
        (self.max_x - self.min_x).max(0.0) * (self.max_y - self.min_y).max(0.0)
    }

    /// `[x, y, w, h]` — the axis-aligned `bbox` a COCO record would
    /// carry for this geometry.
    ///
    /// A conversion, not a fallback. Nothing derives a *missing* GT
    /// `bbox` from oriented geometry today: `bbox` is required on every
    /// record, by serde on the JSON route and by the required-column
    /// check on the columnar one, and `strict` would have to trust
    /// whatever the annotation file says in any case — exactly as
    /// `pycocotools` does. ADR-0063 leaves the derivation to the
    /// Python and CLI ingest layers; if it ever lands in the loader it
    /// is a `corrected` row and needs its own quirk entry.
    #[inline]
    #[must_use]
    pub fn to_xywh(self) -> [f64; 4] {
        [
            self.min_x,
            self.min_y,
            (self.max_x - self.min_x).max(0.0),
            (self.max_y - self.min_y).max(0.0),
        ]
    }
}

/// A rotated box with its trigonometry, half-extents, area and envelope
/// resolved once.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreparedRBox {
    /// The user's numbers, untouched. `strict` replicas read this.
    pub raw: RotatedBox,
    /// `sin(sigma * kappa * theta)`.
    pub sin: f64,
    /// `cos(sigma * kappa * theta)`.
    pub cos: f64,
    /// `w / 2`, exact.
    pub hw: f64,
    /// `h / 2`, exact.
    pub hh: f64,
    /// `w * h`, clamped at zero.
    pub area: f64,
    /// Axis-aligned envelope.
    pub aabb: Aabb,
}

impl PreparedRBox {
    /// Prepare `rbox` under `conv`.
    #[must_use]
    pub fn new(rbox: RotatedBox, conv: Convention) -> Self {
        let (sin, cos) = conv.sin_cos(rbox.theta);
        let hw = rbox.w * 0.5;
        let hh = rbox.h * 0.5;
        let (ex, ey) = rbox.aabb_half_extents(conv);
        Self {
            raw: rbox,
            sin,
            cos,
            hw,
            hh,
            area: rbox.area(),
            aabb: Aabb::centered(rbox.cx, rbox.cy, ex, ey),
        }
    }

    /// Whether the box has positive area. Zero-extent boxes take the
    /// `IoU = 0` path rather than an error: D2 does the same through its
    /// `area < 1e-14` guard (quirk **OB7**).
    #[inline]
    #[must_use]
    pub fn is_degenerate(&self) -> bool {
        !self.area.is_finite() || self.area <= 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convention::{AngleUnit, Rotation};

    const D2: Convention = Convention::D2;

    #[test]
    fn axis_aligned_preparation() {
        let b = RotatedBox {
            cx: 10.0,
            cy: 20.0,
            w: 4.0,
            h: 6.0,
            theta: 0.0,
        };
        let p = PreparedRBox::new(b, D2);
        assert_eq!((p.sin, p.cos), (0.0, 1.0));
        assert_eq!((p.hw, p.hh), (2.0, 3.0));
        assert_eq!(p.area, 24.0);
        assert_eq!(p.aabb, Aabb::new(8.0, 17.0, 12.0, 23.0));
        assert_eq!(p.aabb.to_xywh(), [8.0, 17.0, 4.0, 6.0]);
    }

    #[test]
    fn rotated_ninety_swaps_extents() {
        let b = RotatedBox {
            cx: 0.0,
            cy: 0.0,
            w: 4.0,
            h: 6.0,
            theta: 90.0,
        };
        let p = PreparedRBox::new(b, D2);
        assert_eq!(p.aabb, Aabb::new(-3.0, -2.0, 3.0, 2.0));
    }

    #[test]
    fn degenerate_extents() {
        for (w, h) in [(0.0, 5.0), (5.0, 0.0), (-1.0, 5.0)] {
            let b = RotatedBox {
                cx: 0.0,
                cy: 0.0,
                w,
                h,
                theta: 17.0,
            };
            assert!(PreparedRBox::new(b, D2).is_degenerate());
        }
    }

    #[test]
    fn radians_convention_is_honored() {
        let conv = Convention::new(AngleUnit::Rad, Rotation::ScreenCw);
        let b = RotatedBox {
            cx: 0.0,
            cy: 0.0,
            w: 2.0,
            h: 2.0,
            theta: core::f64::consts::FRAC_PI_2,
        };
        let p = PreparedRBox::new(b, conv);
        assert!((p.sin - 1.0).abs() < 1e-15);
        assert!(p.cos.abs() < 1e-15);
    }

    #[test]
    fn overlap_is_inclusive_at_touch() {
        let a = Aabb::new(0.0, 0.0, 1.0, 1.0);
        let b = Aabb::new(1.0, 0.0, 2.0, 1.0);
        assert!(a.overlaps(&b, 0.0));
        let c = Aabb::new(1.5, 0.0, 2.0, 1.0);
        assert!(!a.overlaps(&c, 0.0));
        assert!(a.overlaps(&c, 0.5));
    }
}
