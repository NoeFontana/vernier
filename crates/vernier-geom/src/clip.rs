//! Sutherland–Hodgman clipping and the shoelace area.
//!
//! This is the canonical (`corrected`) narrow phase. Two clip families
//! are provided: four axis-aligned stages, used when the clip region is
//! a rotated box expressed in its *own* frame, and a general half-plane
//! stage for convex quad pieces.
//!
//! # Why Sutherland–Hodgman and not Green's theorem
//!
//! ADR-0063 rejects the branchless Green / Cyrus–Beck formulation even
//! though it SIMD-izes better. That form classifies each edge of each
//! polygon independently, so a pair of near-coincident edges — perfect
//! detections copied from ground truth, parked cars sharing a boundary —
//! can be counted as "inside" from both sides. A cross term of magnitude
//! `L^2` is then double-counted, which is an `O(1)` error in the IoU,
//! not an `O(eps)` one. Sutherland–Hodgman applies each half-plane once
//! to one evolving polygon, so a misclassification costs `O(eps*L^2)`.
//!
//! # No epsilon guards
//!
//! The intersection parameter is `t = s_P / (s_P - s_Q)` where `s` is
//! the signed distance to the clip line, negative inside. The branch
//! that computes it runs only when `P` and `Q` are classified
//! differently, i.e. `s_P <= 0 < s_Q` or `s_Q <= 0 < s_P`. In both cases
//! `s_P - s_Q` is strictly non-zero *by the classification itself*, so
//! there is no degenerate denominator to guard against and no tolerance
//! to tune.

/// Fixed clip-buffer capacity.
///
/// A convex `m`-gon clipped by a convex `n`-gon has at most `m + n`
/// vertices, so rectangle-against-rectangle tops out at 8. ADR-0063's
/// fuzz gate fails above 12; 32 is the hard buffer bound, chosen so the
/// structure stays a cheap stack value.
pub const MAX_VERTICES: usize = 32;

/// Which coordinate an axis-aligned clip stage acts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// The `x` coordinate.
    X,
    /// The `y` coordinate.
    Y,
}

/// A fixed-capacity polygon in the plane.
///
/// Vertex order is meaningful; the canonical kernel keeps polygons wound
/// positively in the algebraic `(x, y)` plane.
#[derive(Debug, Clone, Copy)]
pub struct Poly {
    xs: [f64; MAX_VERTICES],
    ys: [f64; MAX_VERTICES],
    n: usize,
}

impl Default for Poly {
    fn default() -> Self {
        Self::new()
    }
}

impl Poly {
    /// An empty polygon.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            xs: [0.0; MAX_VERTICES],
            ys: [0.0; MAX_VERTICES],
            n: 0,
        }
    }

    /// Build from a flat `[x0, y0, x1, y1, ...]` slice, truncating at
    /// [`MAX_VERTICES`].
    #[must_use]
    pub fn from_flat(v: &[f64]) -> Self {
        let mut p = Self::new();
        for pair in v.chunks_exact(2) {
            p.push(pair[0], pair[1]);
        }
        p
    }

    /// Number of vertices.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.n
    }

    /// Whether the polygon has no vertices.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Vertex `i`, or `None` when out of range.
    #[inline]
    #[must_use]
    pub fn vertex(&self, i: usize) -> Option<(f64, f64)> {
        (i < self.n).then(|| (self.xs[i], self.ys[i]))
    }

    /// Append a vertex. Silently ignored once [`MAX_VERTICES`] is
    /// reached — the caller's fuzz gate, not a runtime branch, is what
    /// keeps that unreachable.
    #[inline]
    pub fn push(&mut self, x: f64, y: f64) {
        if self.n < MAX_VERTICES {
            self.xs[self.n] = x;
            self.ys[self.n] = y;
            self.n += 1;
        }
    }

    /// Twice the signed area, summed as a fan from vertex 0.
    ///
    /// The fan form — `sum over i of cross(P_i - P_0, P_{i+1} - P_0)` —
    /// is preferred over the raw `x_i*y_{i+1} - x_{i+1}*y_i` shoelace
    /// for two reasons. It subtracts `P_0` first, so every product is
    /// `O(diameter^2)` rather than `O(coordinate^2)`; and it needs
    /// `n - 2` additions instead of `n`, which is what makes the
    /// rectangle case *exact*: a `w x h` rectangle yields two terms of
    /// `fl(w*h)` each, and `fl(w*h) + fl(w*h)` is exact. Summing four
    /// half-terms sequentially instead would round at the third partial
    /// sum and `IoU(a, a) = 1.0` would no longer hold bit-exactly.
    #[inline]
    #[must_use]
    pub fn twice_signed_area(&self) -> f64 {
        if self.n < 3 {
            return 0.0;
        }
        let (x0, y0) = (self.xs[0], self.ys[0]);
        let mut acc = 0.0;
        for i in 1..self.n - 1 {
            let (ax, ay) = (self.xs[i] - x0, self.ys[i] - y0);
            let (bx, by) = (self.xs[i + 1] - x0, self.ys[i + 1] - y0);
            acc += ax * by - bx * ay;
        }
        acc
    }

    /// Signed area (positive when wound positively).
    #[inline]
    #[must_use]
    pub fn signed_area(&self) -> f64 {
        self.twice_signed_area() * 0.5
    }

    /// Unsigned area.
    #[inline]
    #[must_use]
    pub fn area(&self) -> f64 {
        self.signed_area().abs()
    }

    /// Clip to `coord <= bound` (`keep_le`) or `coord >= bound` on
    /// `axis`.
    ///
    /// The surviving coordinate on `axis` is snapped to exactly `bound`
    /// at every generated vertex, so a clipped edge lies *on* the clip
    /// line rather than within a rounding error of it. That is what
    /// makes a second clip stage against the same line idempotent.
    #[must_use]
    pub fn clip_axis(&self, axis: Axis, bound: f64, keep_le: bool) -> Self {
        let mut out = Self::new();
        if self.n == 0 {
            return out;
        }
        let (cs, os) = match axis {
            Axis::X => (&self.xs, &self.ys),
            Axis::Y => (&self.ys, &self.xs),
        };
        let signed = |c: f64| if keep_le { c - bound } else { bound - c };
        for i in 0..self.n {
            let j = if i + 1 == self.n { 0 } else { i + 1 };
            let (sp, sq) = (signed(cs[i]), signed(cs[j]));
            let (p_in, q_in) = (sp <= 0.0, sq <= 0.0);
            if q_in {
                if !p_in {
                    let t = sp / (sp - sq);
                    out.push_axis(axis, bound, os[i] + t * (os[j] - os[i]));
                }
                out.push_axis(axis, cs[j], os[j]);
            } else if p_in {
                let t = sp / (sp - sq);
                out.push_axis(axis, bound, os[i] + t * (os[j] - os[i]));
            }
        }
        out
    }

    /// Clip to the closed half-plane left of the directed line `a -> b`.
    ///
    /// Used for quad pieces, where the clip edges are not axis-aligned
    /// and no coordinate can be snapped.
    #[must_use]
    pub fn clip_halfplane(&self, ax: f64, ay: f64, bx: f64, by: f64) -> Self {
        let mut out = Self::new();
        if self.n == 0 {
            return out;
        }
        let (ex, ey) = (bx - ax, by - ay);
        // `-cross(b - a, p - a)`: negative strictly inside, matching the
        // `s <= 0` convention of `clip_axis`.
        let signed = |x: f64, y: f64| ey * (x - ax) - ex * (y - ay);
        for i in 0..self.n {
            let j = if i + 1 == self.n { 0 } else { i + 1 };
            let (px, py) = (self.xs[i], self.ys[i]);
            let (qx, qy) = (self.xs[j], self.ys[j]);
            let (sp, sq) = (signed(px, py), signed(qx, qy));
            let (p_in, q_in) = (sp <= 0.0, sq <= 0.0);
            if q_in {
                if !p_in {
                    let t = sp / (sp - sq);
                    out.push(px + t * (qx - px), py + t * (qy - py));
                }
                out.push(qx, qy);
            } else if p_in {
                let t = sp / (sp - sq);
                out.push(px + t * (qx - px), py + t * (qy - py));
            }
        }
        out
    }

    #[inline]
    fn push_axis(&mut self, axis: Axis, on_axis: f64, off_axis: f64) {
        match axis {
            Axis::X => self.push(on_axis, off_axis),
            Axis::Y => self.push(off_axis, on_axis),
        }
    }
}

/// Clip `subject` to the axis-aligned rectangle
/// `[-hw, hw] x [-hh, hh]`.
///
/// This is the canonical rotated-box narrow phase: expressing the clip
/// region in its own frame turns four general half-planes into four
/// axis-aligned ones, and removes the image-coordinate magnitude from
/// the error budget entirely (ADR-0063 step 5).
#[must_use]
pub fn clip_to_centered_box(subject: &Poly, hw: f64, hh: f64) -> Poly {
    let p = subject.clip_axis(Axis::X, hw, true);
    if p.is_empty() {
        return p;
    }
    let p = p.clip_axis(Axis::X, -hw, false);
    if p.is_empty() {
        return p;
    }
    let p = p.clip_axis(Axis::Y, hh, true);
    if p.is_empty() {
        return p;
    }
    p.clip_axis(Axis::Y, -hh, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_square() -> Poly {
        Poly::from_flat(&[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    }

    #[test]
    fn area_of_unit_square() {
        assert_eq!(unit_square().area(), 1.0);
    }

    #[test]
    fn rectangle_area_is_exact_via_fan() {
        // A 3 x 7 rectangle centered at the origin, as the canonical
        // kernel builds it: the fan form must return exactly `w * h`.
        for (w, h) in [(3.0, 7.0), (0.1, 0.3), (1e6, 1e-3), (1.0 / 3.0, 7.0 / 11.0)] {
            let (hw, hh) = (w * 0.5, h * 0.5);
            let p = Poly::from_flat(&[hw, hh, -hw, hh, -hw, -hh, hw, -hh]);
            assert_eq!(p.area(), w * h, "w={w} h={h}");
        }
    }

    #[test]
    fn clip_to_self_is_identity_area() {
        let (hw, hh) = (2.0, 1.5);
        let p = Poly::from_flat(&[hw, hh, -hw, hh, -hw, -hh, hw, -hh]);
        let c = clip_to_centered_box(&p, hw, hh);
        assert_eq!(c.len(), 4);
        assert_eq!(c.area(), 4.0 * hw * hh);
    }

    #[test]
    fn half_overlap() {
        // Unit square shifted by 0.5 in x, clipped to [-0.5, 0.5]^2.
        let p = Poly::from_flat(&[0.0, -0.5, 1.0, -0.5, 1.0, 0.5, 0.0, 0.5]);
        let c = clip_to_centered_box(&p, 0.5, 0.5);
        assert_eq!(c.area(), 0.5);
    }

    #[test]
    fn disjoint_clips_to_nothing() {
        let p = Poly::from_flat(&[5.0, 5.0, 6.0, 5.0, 6.0, 6.0, 5.0, 6.0]);
        let c = clip_to_centered_box(&p, 1.0, 1.0);
        assert!(c.area() == 0.0);
    }

    #[test]
    fn clipped_vertices_land_exactly_on_the_bound() {
        let p = Poly::from_flat(&[0.0, 0.0, 3.0, 0.5, 3.0, 2.5, 0.0, 2.0]);
        let c = p.clip_axis(Axis::X, 1.0, true);
        for i in 0..c.len() {
            let (x, _) = c.vertex(i).unwrap_or((f64::NAN, f64::NAN));
            assert!(x <= 1.0);
        }
        assert!((0..c.len()).any(|i| c.vertex(i).map(|v| v.0) == Some(1.0)));
    }

    #[test]
    fn rect_meet_rect_stays_under_eight_vertices() {
        // 45-degree square against an axis-aligned one: the classic
        // 8-gon intersection.
        let s = core::f64::consts::SQRT_2;
        let diamond = Poly::from_flat(&[s, 0.0, 0.0, s, -s, 0.0, 0.0, -s]);
        let c = clip_to_centered_box(&diamond, 1.0, 1.0);
        assert!(c.len() <= 8, "got {}", c.len());
        assert!(c.area() > 3.0 && c.area() < 4.0);
    }

    #[test]
    fn halfplane_clip_matches_axis_clip() {
        let p = unit_square();
        // Left of (1,-1) -> (1, 2) is `x <= 1`... the directed line
        // points +y, so "left" is -x.
        let a = p.clip_halfplane(1.0, -1.0, 1.0, 2.0);
        let b = p.clip_axis(Axis::X, 1.0, true);
        assert_eq!(a.area(), b.area());
    }

    #[test]
    fn empty_polygon_is_inert() {
        let p = Poly::new();
        assert_eq!(p.area(), 0.0);
        assert_eq!(p.clip_axis(Axis::X, 0.0, true).len(), 0);
        assert_eq!(p.clip_halfplane(0.0, 0.0, 1.0, 0.0).len(), 0);
    }
}
