//! Quadrilateral ingestion: orientation, convexity, split and
//! validation.
//!
//! DOTA ground-truth quads are neither rectangles nor guaranteed convex,
//! so the canonical kernel cannot assume either. ADR-0063's ingestion
//! rules:
//!
//! - orient positively by signed area (quirk **OB15** mirrors DK, which
//!   reverses in place when the signed area is negative);
//! - a zero-area quad is a typed error in *both* parity modes, because
//!   DK evaluates `0/0` there and the matching engine forbids `NaN`
//!   (quirk **OB7**);
//! - a self-intersecting quad is a typed error in `corrected` (quirk
//!   **OB16**); the DK replica still accepts it, since reproducing DK
//!   bit-for-bit is the whole point of `strict`.
//!
//! # Why at most two pieces
//!
//! A *simple* quadrilateral has at most one reflex vertex: the interior
//! angles sum to `2*pi`, and two reflex vertices would need more than
//! `2*pi` between them. The diagonal drawn from the reflex vertex is
//! therefore interior, and splitting on it yields two triangles. Convex
//! quads need no split at all.

use crate::clip::Poly;
use crate::error::{GeomError, ShapeKind};

/// Maximum convex pieces a validated quad decomposes into.
pub const MAX_PIECES: usize = 2;

/// Four vertices as supplied, `[x0, y0, x1, y1, x2, y2, x3, y3]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quad {
    /// The raw payload, in submission order.
    pub v: [f64; 8],
}

impl Quad {
    /// Read an `[x0, y0, ..., x3, y3]` payload, rejecting non-finite
    /// values.
    ///
    /// # Errors
    /// [`GeomError::NonFinite`] if any coordinate is `NaN` or infinite.
    pub fn from_slice(v: &[f64; 8]) -> Result<Self, GeomError> {
        for (index, value) in v.iter().enumerate() {
            if !value.is_finite() {
                return Err(GeomError::NonFinite {
                    shape: ShapeKind::Quad,
                    field: if index % 2 == 0 { "x" } else { "y" },
                    index,
                });
            }
        }
        Ok(Self { v: *v })
    }

    /// Vertex `i` of `0..4`.
    #[inline]
    #[must_use]
    pub const fn point(&self, i: usize) -> (f64, f64) {
        (self.v[2 * i], self.v[2 * i + 1])
    }

    /// Twice the signed area, as a fan from vertex 0.
    #[inline]
    #[must_use]
    pub fn twice_signed_area(&self) -> f64 {
        let (x0, y0) = self.point(0);
        let mut acc = 0.0;
        for i in 1..3 {
            let (ax, ay) = (self.v[2 * i] - x0, self.v[2 * i + 1] - y0);
            let (bx, by) = (self.v[2 * i + 2] - x0, self.v[2 * i + 3] - y0);
            acc += ax * by - bx * ay;
        }
        acc
    }

    /// Axis-aligned envelope as `(min_x, min_y, max_x, max_y)`.
    #[must_use]
    pub fn aabb(&self) -> (f64, f64, f64, f64) {
        let (mut nx, mut ny) = self.point(0);
        let (mut xx, mut xy) = (nx, ny);
        for i in 1..4 {
            let (x, y) = self.point(i);
            nx = nx.min(x);
            ny = ny.min(y);
            xx = xx.max(x);
            xy = xy.max(y);
        }
        (nx, ny, xx, xy)
    }
}

/// `cross(b - a, c - a)`.
#[inline]
fn cross3(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// Do the open segments `p1p2` and `p3p4` cross properly?
///
/// Strict on both sides: touching endpoints and collinear overlap are
/// *not* crossings. A quad whose vertices merely repeat is caught by the
/// zero-area check instead.
fn segments_cross(p1: (f64, f64), p2: (f64, f64), p3: (f64, f64), p4: (f64, f64)) -> bool {
    let d1 = cross3(p3, p4, p1);
    let d2 = cross3(p3, p4, p2);
    let d3 = cross3(p1, p2, p3);
    let d4 = cross3(p1, p2, p4);
    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

/// A quad validated, positively wound, and decomposed into convex
/// pieces relative to its own vertex centroid.
///
/// Storing the pieces **centroid-relative** is what makes the pair
/// kernel's translation step cheap and accurate: the only absolute
/// quantity that ever enters a clip is `delta = c_d - c_g`, whose error
/// is bounded by the *pair diameter* rather than by the image
/// coordinate magnitude (ADR-0063 step 2).
#[derive(Debug, Clone, Copy)]
pub struct PreparedQuad {
    /// The raw payload, untouched — `strict` replicas consume this.
    pub raw: [f64; 8],
    /// Vertex centroid, the translation anchor.
    pub cx: f64,
    /// Vertex centroid, the translation anchor.
    pub cy: f64,
    /// Sum of the convex pieces' areas. This is *the* area used by both
    /// the IoU denominator and the dataset-level area bucket, so the two
    /// can never disagree.
    pub area: f64,
    /// Axis-aligned envelope in absolute coordinates,
    /// `(min_x, min_y, max_x, max_y)`.
    pub aabb: (f64, f64, f64, f64),
    pieces: [Poly; MAX_PIECES],
    n_pieces: usize,
    convex: bool,
}

impl PreparedQuad {
    /// Validate and decompose.
    ///
    /// `allow_self_intersection` is `true` only on the DK-strict path,
    /// where the oracle's fan decomposition defines *some* value for a
    /// bowtie and `strict` must reproduce it. The canonical kernel
    /// refuses (quirk **OB16**).
    ///
    /// # Errors
    /// - [`GeomError::NonFinite`] — propagated from [`Quad::from_slice`].
    /// - [`GeomError::ZeroAreaQuad`] — collinear or duplicated vertices.
    /// - [`GeomError::SelfIntersectingQuad`] — edges cross, unless
    ///   `allow_self_intersection`.
    pub fn new(v: &[f64; 8], allow_self_intersection: bool) -> Result<Self, GeomError> {
        let quad = Quad::from_slice(v)?;

        if !allow_self_intersection {
            if segments_cross(quad.point(0), quad.point(1), quad.point(2), quad.point(3)) {
                return Err(GeomError::SelfIntersectingQuad { a: 0, b: 2 });
            }
            if segments_cross(quad.point(1), quad.point(2), quad.point(3), quad.point(0)) {
                return Err(GeomError::SelfIntersectingQuad { a: 1, b: 3 });
            }
        }

        let twice = quad.twice_signed_area();
        if twice == 0.0 {
            return Err(GeomError::ZeroAreaQuad);
        }

        // Orient positively, then re-read the vertices in that order.
        let mut pts = [(0.0, 0.0); 4];
        for (i, slot) in pts.iter_mut().enumerate() {
            *slot = quad.point(if twice < 0.0 { 3 - i } else { i });
        }

        let cx = (pts[0].0 + pts[1].0 + pts[2].0 + pts[3].0) * 0.25;
        let cy = (pts[0].1 + pts[1].1 + pts[2].1 + pts[3].1) * 0.25;
        let local: [(f64, f64); 4] = [
            (pts[0].0 - cx, pts[0].1 - cy),
            (pts[1].0 - cx, pts[1].1 - cy),
            (pts[2].0 - cx, pts[2].1 - cy),
            (pts[3].0 - cx, pts[3].1 - cy),
        ];

        // Reflex vertex, if any. At most one exists for a simple quad;
        // a self-intersecting one (DK-strict) may report two, in which
        // case the first wins and the pieces merely retrace DK's own
        // ambiguity.
        let mut reflex = None;
        for i in 0..4 {
            let prev = local[(i + 3) % 4];
            let cur = local[i];
            let next = local[(i + 1) % 4];
            if cross3(prev, cur, next) < 0.0 {
                reflex = Some(i);
                break;
            }
        }

        let mut pieces = [Poly::new(); MAX_PIECES];
        let n_pieces = match reflex {
            None => {
                let mut p = Poly::new();
                for &(x, y) in &local {
                    p.push(x, y);
                }
                pieces[0] = p;
                1
            }
            Some(r) => {
                let mut t0 = Poly::new();
                let mut t1 = Poly::new();
                for k in 0..3 {
                    let (x, y) = local[(r + k) % 4];
                    t0.push(x, y);
                }
                for k in [0_usize, 2, 3] {
                    let (x, y) = local[(r + k) % 4];
                    t1.push(x, y);
                }
                pieces[0] = t0;
                pieces[1] = t1;
                2
            }
        };

        let mut area = 0.0;
        for piece in pieces.iter().take(n_pieces) {
            area += piece.area();
        }

        Ok(Self {
            raw: *v,
            cx,
            cy,
            area,
            aabb: quad.aabb(),
            pieces,
            n_pieces,
            convex: reflex.is_none(),
        })
    }

    /// The convex pieces, centroid-relative and positively wound.
    #[inline]
    #[must_use]
    pub fn pieces(&self) -> &[Poly] {
        &self.pieces[..self.n_pieces]
    }

    /// Whether the quad needed no split.
    ///
    /// A free accessor: the constructor has to find the reflex vertex
    /// to decompose the quad, so convexity falls out of work already
    /// done. Exposed because a caller reasoning about DOTA ground truth
    /// — frequently neither rectangular nor convex — will want to count
    /// it.
    #[inline]
    #[must_use]
    pub const fn is_convex(&self) -> bool {
        self.convex
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SQ: [f64; 8] = [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0];

    #[test]
    fn convex_square_is_one_piece() {
        let p = PreparedQuad::new(&SQ, false).unwrap();
        assert!(p.is_convex());
        assert_eq!(p.pieces().len(), 1);
        assert_eq!(p.area, 4.0);
        assert_eq!((p.cx, p.cy), (1.0, 1.0));
        assert_eq!(p.aabb, (0.0, 0.0, 2.0, 2.0));
    }

    #[test]
    fn clockwise_input_is_reoriented() {
        let cw = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0];
        let p = PreparedQuad::new(&cw, false).unwrap();
        assert_eq!(p.area, 4.0);
        assert!(p.pieces()[0].signed_area() > 0.0);
        // The raw payload is untouched: strict replicas need submission
        // order preserved.
        assert_eq!(p.raw, cw);
    }

    #[test]
    fn zero_area_is_rejected() {
        let collinear = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        assert_eq!(
            PreparedQuad::new(&collinear, false).err(),
            Some(GeomError::ZeroAreaQuad)
        );
        let degenerate = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(
            PreparedQuad::new(&degenerate, false).err(),
            Some(GeomError::ZeroAreaQuad)
        );
    }

    #[test]
    fn bowtie_is_rejected_in_corrected_but_accepted_for_dk() {
        // Asymmetric: a symmetric bowtie cancels to zero signed
        // area and is caught by the zero-area check first.
        let bowtie = [0.0, 0.0, 4.0, 4.0, 4.0, 0.0, 0.0, 1.0];
        assert!(matches!(
            PreparedQuad::new(&bowtie, false),
            Err(GeomError::SelfIntersectingQuad { .. })
        ));
        assert!(PreparedQuad::new(&bowtie, true).is_ok());
    }

    #[test]
    fn non_convex_quad_splits_into_two_triangles() {
        // An arrowhead: vertex 3 is reflex.
        let arrow = [0.0, 0.0, 4.0, 0.0, 2.0, 4.0, 2.0, 1.0];
        let p = PreparedQuad::new(&arrow, false).unwrap();
        assert!(!p.is_convex());
        assert_eq!(p.pieces().len(), 2);
        for piece in p.pieces() {
            assert_eq!(piece.len(), 3);
            assert!(piece.signed_area() > 0.0, "piece must stay positive");
        }
        // Shoelace of the simple polygon: 0.5 * |16 - 6| = 5, and the
        // two triangles (2 and 3) sum to the same.
        assert_eq!(p.area, 5.0);
    }

    #[test]
    fn non_finite_is_rejected() {
        let mut bad = SQ;
        bad[5] = f64::INFINITY;
        assert!(matches!(
            PreparedQuad::new(&bad, false),
            Err(GeomError::NonFinite { index: 5, .. })
        ));
    }

    #[test]
    fn pieces_are_centroid_relative() {
        let far = [
            1e6,
            1e6,
            1e6 + 2.0,
            1e6,
            1e6 + 2.0,
            1e6 + 2.0,
            1e6,
            1e6 + 2.0,
        ];
        let p = PreparedQuad::new(&far, false).unwrap();
        for i in 0..p.pieces()[0].len() {
            let (x, y) = p.pieces()[0].vertex(i).unwrap();
            assert!(x.abs() <= 1.0 + 1e-9 && y.abs() <= 1.0 + 1e-9, "({x}, {y})");
        }
        assert_eq!(p.area, 4.0);
    }

    #[test]
    fn segments_cross_is_strict_about_touching() {
        assert!(!segments_cross(
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0)
        ));
        assert!(segments_cross(
            (0.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (2.0, 0.0)
        ));
    }
}
