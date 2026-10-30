//! Angle conventions: unit (`kappa`) and rotation sign (`sigma`).
//!
//! Per ADR-0063 these are the two parameters an IoU value can depend on,
//! and both are **required** — never inferred, never defaulted. Angle
//! unit and rotation direction are the most common real-world OBB bug,
//! and a wrong guess here produces plausible-looking numbers rather than
//! an error.
//!
//! # The quotient
//!
//! In the pixel frame (`x` right, `y` **down**) a rotated box is
//! center-based:
//!
//! ```text
//! P(c, w, h, theta) = { c + R(sigma*kappa*theta) * (s*w/2, t*h/2) : s, t in [-1, 1] }
//! ```
//!
//! with `R(a) = [[cos a, -sin a], [sin a, cos a]]`. The point set `P` is
//! invariant under `theta -> theta + pi` and under
//! `(w, h, theta) -> (h, w, theta + pi/2)`, so `le90`, `le135`, `oc` and
//! OpenCV's 4.5.1 range change are parameterizations of one quotient:
//! **IoU cannot depend on which one a producer used** (quirk **OB3**).
//! The *parameterization* is not bit-invariant, though — D2 derives
//! corners from `(w, h, theta)`, so swapping `w` and `h` moves last
//! bits. `strict` therefore passes user numbers through untouched; only
//! `corrected` canonicalizes.

use crate::error::{GeomError, ShapeKind};

/// Radians per degree, correctly rounded.
///
/// The **canonical** (`corrected`) kernel uses this. The D2 replica does
/// *not*: detectron2 hard-codes a 12-digit truncation of the same
/// constant, which is pinned separately in
/// `pinned::D2_DEG_TO_RAD` (quirk **OB4**).
pub const DEG_TO_RAD: f64 = core::f64::consts::PI / 180.0;

/// Unit in which `theta` is expressed — `kappa` in ADR-0063.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AngleUnit {
    /// Degrees. `kappa = pi/180`. Multiples of 90 degrees are exact.
    Deg,
    /// Radians. `kappa = 1`.
    Rad,
}

impl AngleUnit {
    /// Stable wire / hash tag. Appended-only, like [`crate::VERSION`]-adjacent
    /// discriminators elsewhere in the workspace.
    pub const fn tag(self) -> u8 {
        match self {
            Self::Deg => 0,
            Self::Rad => 1,
        }
    }

    /// Lowercase name used by the CLI and the Python surface.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Deg => "deg",
            Self::Rad => "rad",
        }
    }
}

/// Direction a positive `theta` turns, seen on screen — `sigma` in
/// ADR-0063.
///
/// The pixel frame has `y` pointing **down**, so the algebraic sign and
/// the visual direction are opposites of what a math-frame reader
/// expects. Naming the variants after what a viewer sees removes that
/// trap from the API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rotation {
    /// `sigma = +1`: positive `theta` turns `+x` toward `+y`, which is
    /// **clockwise on screen** because `y` points down.
    ScreenCw,
    /// `sigma = -1`: positive `theta` turns `+x` toward `-y`,
    /// **counter-clockwise on screen**.
    ///
    /// This is detectron2's convention (quirk **OB2**, pinned by the
    /// `(5, 3, 4, 2, 90)` corner probe).
    ScreenCcw,
}

impl Rotation {
    /// `sigma` as a multiplier on `sin`.
    ///
    /// Only `sin` changes sign under `theta -> -theta`, so applying
    /// `sigma` post-hoc to a `(sin, cos)` pair is exactly equivalent to
    /// negating the angle first — and it keeps the exactness of the
    /// degree reduction, which is defined on the magnitude.
    pub const fn sin_sign(self) -> f64 {
        match self {
            Self::ScreenCw => 1.0,
            Self::ScreenCcw => -1.0,
        }
    }

    /// Stable wire / hash tag.
    pub const fn tag(self) -> u8 {
        match self {
            Self::ScreenCw => 0,
            Self::ScreenCcw => 1,
        }
    }

    /// Lowercase name used by the CLI and the Python surface.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ScreenCw => "screen_cw",
            Self::ScreenCcw => "screen_ccw",
        }
    }
}

/// The `(kappa, sigma)` pair. Both fields are required at construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Convention {
    /// Angle unit, `kappa`.
    pub unit: AngleUnit,
    /// Rotation sign, `sigma`.
    pub rotation: Rotation,
}

impl Convention {
    /// Construct from unit and rotation. There is deliberately no
    /// `Default`.
    pub const fn new(unit: AngleUnit, rotation: Rotation) -> Self {
        Self { unit, rotation }
    }

    /// detectron2's convention: degrees, counter-clockwise on screen.
    ///
    /// Provided as a named constant because it is the one the `D2`
    /// strict oracle is pinned to, not because it is a default.
    pub const D2: Self = Self::new(AngleUnit::Deg, Rotation::ScreenCcw);

    /// `(sin(sigma*kappa*theta), cos(sigma*kappa*theta))`.
    ///
    /// Signed zeros are normalized to `+0.0` so a rotation of `0` and a
    /// rotation of `-0` produce bit-identical geometry.
    #[inline]
    #[must_use]
    pub fn sin_cos(self, theta: f64) -> (f64, f64) {
        let (s, c) = match self.unit {
            AngleUnit::Deg => sin_cos_deg(theta),
            AngleUnit::Rad => normalize_zeros(theta.sin_cos()),
        };
        // `sin(-x) = -sin(x)`, `cos(-x) = cos(x)`.
        normalize_zeros((s * self.rotation.sin_sign(), c))
    }
}

/// `-0.0 -> +0.0`, everything else unchanged.
///
/// `x + 0.0` is exact for every finite `x` and maps `-0.0` to `+0.0`
/// under round-to-nearest, which is the whole trick. ADR-0063's
/// bitwise-equality relation is defined modulo this normalization, so
/// doing it once at the source keeps every downstream comparison a
/// plain `==` on bits.
#[inline]
const fn normalize_zeros(p: (f64, f64)) -> (f64, f64) {
    (p.0 + 0.0, p.1 + 0.0)
}

/// `(sin, cos)` of an angle given in **degrees**, exact on every
/// multiple of 90.
///
/// The reduction is the reason this is not just `(deg *
/// DEG_TO_RAD).sin_cos()`. `deg % 360.0` is IEEE `fmod`, which is
/// *exact* — no rounding at all, at any magnitude. Rounding a large
/// angle through `DEG_TO_RAD` first would spend accuracy before `sin`
/// ever sees the value, and would make `cos(90)` a small non-zero
/// number, which in turn makes an axis-aligned pair of boxes
/// almost-but-not-quite axis-aligned. ADR-0063's exactness properties —
/// `IoU(a, a) = 1.0`, and `theta_g = theta_d (mod 90)` reproducing
/// [`crate::kernel`]'s axis-aligned path — both rest on this.
#[inline]
#[must_use]
pub fn sin_cos_deg(deg: f64) -> (f64, f64) {
    let neg = deg.is_sign_negative();
    let a = deg.abs();
    // Exact: `%` on f64 is `fmod`, not a rounded remainder.
    let m = a % 360.0;
    // `m / 90.0` can only land on an integer when `m` is an exact
    // multiple of 90, and `q * 90.0` is exact for `q in 0..=3`, so
    // `rem` is exact by Sterbenz.
    let q = (m / 90.0).floor();
    let rem = m - q * 90.0;
    let (s0, c0) = if rem == 0.0 {
        (0.0, 1.0)
    } else {
        (rem * DEG_TO_RAD).sin_cos()
    };
    #[allow(clippy::cast_possible_truncation)]
    let (s, c) = match (q as i64) & 3 {
        0 => (s0, c0),
        1 => (c0, -s0),
        2 => (-s0, -c0),
        _ => (-c0, s0),
    };
    normalize_zeros(if neg { (-s, c) } else { (s, c) })
}

/// A rotated box exactly as the user supplied it: `[cx, cy, w, h,
/// theta]`, plus the convention `theta` is read under.
///
/// `strict` never rewrites these numbers — the replicas consume them
/// verbatim, because D2's corner bits are a function of the
/// parameterization and not only of the point set (quirk **OB3**).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotatedBox {
    /// Center `x`.
    pub cx: f64,
    /// Center `y`.
    pub cy: f64,
    /// Extent along the box's own `x` axis.
    pub w: f64,
    /// Extent along the box's own `y` axis.
    pub h: f64,
    /// Rotation, in [`Convention::unit`].
    pub theta: f64,
}

impl RotatedBox {
    /// Read a `[cx, cy, w, h, theta]` payload, rejecting non-finite
    /// values.
    ///
    /// # Errors
    /// [`GeomError::NonFinite`] if any slot is `NaN` or infinite.
    /// Non-positive `w` or `h` is *not* an error: it yields zero area
    /// and therefore zero IoU, matching D2's `area < 1e-14` guard
    /// (quirk **OB7**).
    pub fn from_slice(v: &[f64; 5]) -> Result<Self, GeomError> {
        const FIELDS: [&str; 5] = ["cx", "cy", "w", "h", "theta"];
        for (index, value) in v.iter().enumerate() {
            if !value.is_finite() {
                return Err(GeomError::NonFinite {
                    shape: ShapeKind::RotatedBox,
                    field: FIELDS.get(index).copied().unwrap_or("?"),
                    index,
                });
            }
        }
        Ok(Self {
            cx: v[0],
            cy: v[1],
            w: v[2],
            h: v[3],
            theta: v[4],
        })
    }

    /// The raw payload, in the order the JSON / Arrow surfaces use.
    #[must_use]
    pub const fn to_slice(self) -> [f64; 5] {
        [self.cx, self.cy, self.w, self.h, self.theta]
    }

    /// Area, `w * h`, clamped at zero.
    ///
    /// Matches `pycocotools` `loadRes`'s `bb[2] * bb[3]` on a length-5
    /// `bbox`, so quirk **J3** (DT area is derived, never read) carries
    /// over verbatim (quirk **OB13**).
    #[inline]
    #[must_use]
    pub fn area(self) -> f64 {
        if self.w > 0.0 && self.h > 0.0 {
            self.w * self.h
        } else {
            0.0
        }
    }

    /// The four corners in image coordinates under `conv`, wound
    /// positively in the algebraic `(x, y)` plane.
    ///
    /// These are the **`corrected`** bits: the canonical kernel's own
    /// corner construction, not D2's. `vernier.instance.obb.to_quad`
    /// exposes exactly this; for DK-strict, submit the quads you would
    /// actually submit to DOTA_devkit instead.
    #[must_use]
    pub fn corners(self, conv: Convention) -> [f64; 8] {
        let (sin_a, cos_a) = conv.sin_cos(self.theta);
        let hw = self.w * 0.5;
        let hh = self.h * 0.5;
        // Width axis and height axis of the box, in image coordinates.
        let (ux, uy) = (cos_a * hw, sin_a * hw);
        let (vx, vy) = (-sin_a * hh, cos_a * hh);
        [
            self.cx + ux + vx,
            self.cy + uy + vy,
            self.cx - ux + vx,
            self.cy - uy + vy,
            self.cx - ux - vx,
            self.cy - uy - vy,
            self.cx + ux - vx,
            self.cy + uy - vy,
        ]
    }

    /// Half-extents of the axis-aligned envelope: `(ex, ey)` such that
    /// the tight AABB is `[cx - ex, cx + ex] x [cy - ey, cy + ey]`.
    ///
    /// Takes `|w|` and `|h|`, not `max(w, 0)`, and the difference is a
    /// parity matter rather than a taste one. detectron2 gates on
    /// `w * h < 1e-14`, which a box with *both* sides negative passes:
    /// `get_rotated_vertices` then builds the same rectangle a positive
    /// `(|w|, |h|)` would, and the oracle scores the pair. Clamping
    /// each side independently would collapse that box's envelope to a
    /// point, the padded-AABB prefilter would reject every pair it is
    /// in, and `strict` would return `+0.0` where the oracle returns a
    /// real IoU — a bit-inequality in the one mode whose whole contract
    /// is bit-equality. A prefilter is allowed to be too generous and
    /// not allowed to be too tight, so the envelope widens.
    ///
    /// The canonical kernel is unaffected: [`Self::area`] still clamps,
    /// so a sign-flipped box scores `0.0` under `corrected` either way
    /// and the wider envelope costs one narrow-phase call.
    #[inline]
    #[must_use]
    pub fn aabb_half_extents(self, conv: Convention) -> (f64, f64) {
        let (sin_a, cos_a) = conv.sin_cos(self.theta);
        let hw = self.w.abs() * 0.5;
        let hh = self.h.abs() * 0.5;
        (
            cos_a.abs() * hw + sin_a.abs() * hh,
            sin_a.abs() * hw + cos_a.abs() * hh,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const D2: Convention = Convention::D2;

    #[test]
    fn degree_quadrants_are_exact() {
        let conv = Convention::new(AngleUnit::Deg, Rotation::ScreenCw);
        for (deg, want) in [
            (0.0, (0.0, 1.0)),
            (90.0, (1.0, 0.0)),
            (180.0, (0.0, -1.0)),
            (270.0, (-1.0, 0.0)),
            (360.0, (0.0, 1.0)),
            (-90.0, (-1.0, 0.0)),
            (450.0, (1.0, 0.0)),
            (36_000_090.0, (1.0, 0.0)),
        ] {
            assert_eq!(conv.sin_cos(deg), want, "deg = {deg}");
        }
    }

    #[test]
    fn no_negative_zero_escapes() {
        let conv = Convention::new(AngleUnit::Deg, Rotation::ScreenCcw);
        for deg in [0.0, -0.0, 90.0, 180.0, 270.0, -180.0, -360.0] {
            let (s, c) = conv.sin_cos(deg);
            assert!(!s.is_sign_negative() || s != 0.0, "sin -0.0 at {deg}");
            assert!(!c.is_sign_negative() || c != 0.0, "cos -0.0 at {deg}");
        }
    }

    #[test]
    fn sigma_flips_only_sin() {
        let cw = Convention::new(AngleUnit::Deg, Rotation::ScreenCw);
        let ccw = Convention::new(AngleUnit::Deg, Rotation::ScreenCcw);
        for deg in [13.0, 47.5, 123.25, -77.0] {
            let (s1, c1) = cw.sin_cos(deg);
            let (s2, c2) = ccw.sin_cos(deg);
            assert_eq!(c1, c2);
            assert_eq!(s1, -s2);
        }
    }

    /// ADR-0063 M1 exit gate: the documented detectron2 probe.
    ///
    /// `RotatedBoxes([[5, 3, 4, 2, 90]])` has corners
    /// `(4, 5), (4, 1), (6, 1), (6, 5)` — a fact detectron2's own
    /// `structures` docs state. Reproducing it as a *set* pins
    /// `sigma = ScreenCcw` for the D2 convention (quirk **OB2**).
    #[test]
    fn d2_sigma_probe_5_3_4_2_90() {
        let b = RotatedBox {
            cx: 5.0,
            cy: 3.0,
            w: 4.0,
            h: 2.0,
            theta: 90.0,
        };
        let c = b.corners(D2);
        let mut got: Vec<(f64, f64)> = (0..4).map(|i| (c[2 * i], c[2 * i + 1])).collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        assert_eq!(got, vec![(4.0, 1.0), (4.0, 5.0), (6.0, 1.0), (6.0, 5.0)]);
    }

    #[test]
    fn axis_aligned_corners_are_exact() {
        let b = RotatedBox {
            cx: 10.0,
            cy: 20.0,
            w: 4.0,
            h: 6.0,
            theta: 0.0,
        };
        assert_eq!(
            b.corners(D2),
            [12.0, 23.0, 8.0, 23.0, 8.0, 17.0, 12.0, 17.0]
        );
    }

    #[test]
    fn corners_wind_positively() {
        let b = RotatedBox {
            cx: 0.0,
            cy: 0.0,
            w: 4.0,
            h: 2.0,
            theta: 30.0,
        };
        let c = b.corners(D2);
        let mut twice_area = 0.0_f64;
        for i in 0..4 {
            let j = (i + 1) % 4;
            twice_area += c[2 * i] * c[2 * j + 1] - c[2 * j] * c[2 * i + 1];
        }
        assert!(twice_area > 0.0, "signed area {twice_area}");
    }

    #[test]
    fn non_finite_is_rejected() {
        assert!(matches!(
            RotatedBox::from_slice(&[0.0, 0.0, 1.0, f64::NAN, 0.0]),
            Err(GeomError::NonFinite { index: 3, .. })
        ));
    }

    #[test]
    fn aabb_of_axis_aligned_box_is_tight() {
        let b = RotatedBox {
            cx: 0.0,
            cy: 0.0,
            w: 4.0,
            h: 2.0,
            theta: 0.0,
        };
        assert_eq!(b.aabb_half_extents(D2), (2.0, 1.0));
        let r = RotatedBox { theta: 90.0, ..b };
        assert_eq!(r.aabb_half_extents(D2), (1.0, 2.0));
    }
}
