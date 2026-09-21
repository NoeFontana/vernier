//! Op-exact replica of DOTA_devkit's polygon IoU.
//!
//! Upstream: `polyiou.cpp` at commit [`crate::pinned::DK_COMMIT_SHA`],
//! plus the horizontal-box prefilter from `dota_evaluation_task1.py` at
//! the same commit. Neither file is vendored — DOTA_devkit states no
//! license anywhere, so there is no right to redistribute it. The
//! SHA-256 of each is pinned in [`crate::pinned`] and
//! `tests/python/parity_obb/oracle/VENDORING.md` records the position in
//! full. Nothing below is copied text; it is a description of observable
//! behavior, written to be checkable against the hashed originals.
//!
//! # The composed oracle
//!
//! `iou_poly` alone is *not* the oracle. DOTA's task-1 evaluator only
//! ever calls it for ground truths whose horizontal envelope overlaps
//! the detection's, where "overlap" is computed the Pascal VOC way, with
//! a `+1` added to every width and height. So the oracle this crate
//! reproduces is the pair:
//!
//! ```text
//! DK(g, d) = iou_poly(g, d)   if HBB-with-+1 overlap
//!          = 0                otherwise
//! ```
//!
//! That is not a performance detail, it is a *correctness* one. The fan
//! decomposition below sums signed triangle areas that cancel, and on a
//! pair whose envelopes overlap but whose polygons do not, the
//! cancellation is not exact — a small residue survives. Skipping the
//! prefilter would therefore change the matrix, which is why
//! [`hbb_plus_one_overlap`] is part of [`iou`] rather than an
//! optimization layered on top (quirk **OB17**), and why the
//! separating-axis test that the canonical kernel uses is *inadmissible*
//! here.
//!
//! # Fan anchored at the origin
//!
//! `intersectArea` decomposes each polygon into triangles hinged on the
//! **coordinate origin**, not on a polygon vertex or centroid. For DOTA
//! tiles that is thousands of pixels away, so every intermediate is
//! `O(X^2)` where `X` is the image coordinate. f64 absorbs that; the
//! canonical kernel does not need to, because it works in the ground
//! truth's own frame (quirk **OB5**).

use crate::pinned::{DK_EPS, DK_MAXN};

/// Scratch capacity for the clipped triangle. Upstream uses
/// `Point p[10]`; the cut of a triangle by three half-planes cannot
/// exceed six vertices plus the wrap slot.
const TRI_SCRATCH: usize = 16;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct P {
    x: f64,
    y: f64,
}

/// Upstream `sig`: a three-way sign with an **absolute** tolerance.
///
/// Because it is applied to cross products, its effective angular
/// resolution falls off as the square of the coordinate magnitude. Near
/// the origin `1e-8` is a real tolerance; at DOTA tile coordinates it is
/// indistinguishable from an exact-zero test. Reproduced rather than
/// improved: the whole value of `strict` is that it does not
/// second-guess the oracle.
#[inline]
fn sig(d: f64) -> i32 {
    i32::from(d > DK_EPS) - i32::from(d < -DK_EPS)
}

/// Upstream `Point::operator==`, which is `sig`-based and therefore
/// *not* bitwise.
#[inline]
fn peq(a: P, b: P) -> bool {
    sig(a.x - b.x) == 0 && sig(a.y - b.y) == 0
}

/// Upstream `cross(o, a, b)`.
#[inline]
fn cross(o: P, a: P, b: P) -> f64 {
    (a.x - o.x) * (b.y - o.y) - (b.x - o.x) * (a.y - o.y)
}

/// Upstream `area`. Writes the wrap-around slot `ps[n] = ps[0]` as a
/// side effect, which later reads depend on.
///
/// This is the raw cyclic shoelace, summed in index order — not the
/// fan form the canonical kernel uses. The sum order is part of the
/// oracle, which is why reversing a polygon's winding changes the last
/// bits even though it cannot change the magnitude.
fn area(ps: &mut [P], n: usize) -> f64 {
    if n == 0 || n >= ps.len() {
        return 0.0;
    }
    ps[n] = ps[0];
    let mut res = 0.0;
    for i in 0..n {
        res += ps[i].x * ps[i + 1].y - ps[i].y * ps[i + 1].x;
    }
    res / 2.0
}

/// Upstream `lineCross`. Returns 1 when it wrote to `p`, 2 for a
/// fully-degenerate configuration, 0 for parallel lines — and in the
/// latter two cases **leaves `p` untouched**.
fn line_cross(a: P, b: P, c: P, d: P, p: &mut P) -> i32 {
    let s1 = cross(a, b, c);
    let s2 = cross(a, b, d);
    if sig(s1) == 0 && sig(s2) == 0 {
        return 2;
    }
    if sig(s2 - s1) == 0 {
        return 0;
    }
    p.x = (c.x * s2 - d.x * s1) / (s2 - s1);
    p.y = (c.y * s2 - d.y * s1) / (s2 - s1);
    1
}

/// Upstream `polygon_cut`: clip `p` to the left of the directed line
/// `a -> b`, in place.
///
/// # The stale-slot hazard (quirk **OB19**)
///
/// Upstream writes `lineCross(a, b, p[i], p[i+1], pp[m++])` — the
/// post-increment fires whether or not `lineCross` wrote anything. When
/// it returns 0 or 2 the slot keeps whatever was there before: garbage
/// on the first pass over a fresh stack buffer, and the previous cut's
/// leftovers afterwards. That is undefined behavior in the oracle, so
/// there is no "correct" value for `strict` to reproduce.
///
/// This replica models it as faithfully as a defined program can: `pp`
/// is allocated once per triangle-pair call and reused across the three
/// cuts, exactly as upstream's stack array is, and a first-touch slot
/// reads as `(0, 0)`. [`FanTrace::stale_slots`] counts how often the
/// path is taken so a fuzz run can report it rather than assume it never
/// happens.
///
/// Reaching it needs `sig(s1) != sig(s2)` while `|s2 - s1| <= 1e-8` —
/// both cross products within a hair of zero on opposite sides. On
/// integer-coordinate polygons, which is what DOTA annotations are, the
/// cross products are exact integers, so a sign change forces
/// `|s2 - s1| >= 1` and the path is unreachable.
fn polygon_cut(p: &mut [P], n: &mut usize, a: P, b: P, pp: &mut [P], trace: &mut FanTrace) {
    if *n == 0 || *n >= p.len() {
        return;
    }
    let mut m = 0_usize;
    p[*n] = p[0];
    for i in 0..*n {
        if m + 2 > pp.len() {
            break;
        }
        let si = sig(cross(a, b, p[i]));
        if si > 0 {
            pp[m] = p[i];
            m += 1;
        }
        if si != sig(cross(a, b, p[i + 1])) {
            // Preserve-then-restore reproduces the "left untouched"
            // behavior without ever reading uninitialized memory.
            let mut slot = pp[m];
            if line_cross(a, b, p[i], p[i + 1], &mut slot) != 1 {
                trace.stale_slots += 1;
            }
            pp[m] = slot;
            m += 1;
        }
    }
    *n = 0;
    for i in 0..m {
        if i == 0 || !peq(pp[i], pp[i - 1]) {
            if *n >= p.len() - 1 {
                break;
            }
            p[*n] = pp[i];
            *n += 1;
        }
    }
    while *n > 1 && peq(p[*n - 1], p[0]) {
        *n -= 1;
    }
}

/// Diagnostics for a fan evaluation. Not part of the numeric result —
/// it exists so the fuzz suite can report how often the oracle's
/// undefined path was reached instead of quietly assuming it wasn't.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FanTrace {
    /// Times `polygon_cut` consumed a slot `lineCross` did not write.
    pub stale_slots: u64,
}

/// Upstream `intersectArea(Point a, Point b, Point c, Point d)`: the
/// signed intersection area of triangles `(o, a, b)` and `(o, c, d)`,
/// with `o` the coordinate origin.
fn intersect_area_tri(a: P, b: P, c: P, d: P, trace: &mut FanTrace) -> f64 {
    let o = P { x: 0.0, y: 0.0 };
    let s1 = sig(cross(o, a, b));
    let s2 = sig(cross(o, c, d));
    if s1 == 0 || s2 == 0 {
        return 0.0;
    }
    let (a, b) = if s1 == -1 { (b, a) } else { (a, b) };
    let (c, d) = if s2 == -1 { (d, c) } else { (c, d) };

    let mut p = [P::default(); TRI_SCRATCH];
    p[0] = o;
    p[1] = a;
    p[2] = b;
    let mut n = 3_usize;
    // One `pp` per call, shared by the three cuts — the same lifetime
    // the upstream stack array has.
    let mut pp = [P::default(); DK_MAXN];
    polygon_cut(&mut p, &mut n, o, c, &mut pp, trace);
    polygon_cut(&mut p, &mut n, c, d, &mut pp, trace);
    polygon_cut(&mut p, &mut n, d, o, &mut pp, trace);

    let mut res = area(&mut p, n).abs();
    if s1 * s2 == -1 {
        // The one place upstream can produce `-0.0` (quirk **OB14**):
        // `res` is a magnitude, so negating a zero gives a signed zero.
        res = -res;
    }
    res
}

/// Upstream `intersectArea(Point* ps1, int n1, Point* ps2, int n2)`.
///
/// Reverses either polygon **in place** when its signed area is
/// negative (quirk **OB15**) — a mutation the caller then observes,
/// because `iou_poly` recomputes both areas afterwards and the shoelace
/// sums them in index order.
fn intersect_area_poly(
    ps1: &mut [P],
    n1: usize,
    ps2: &mut [P],
    n2: usize,
    trace: &mut FanTrace,
) -> f64 {
    if area(ps1, n1) < 0.0 {
        ps1[..n1].reverse();
    }
    if area(ps2, n2) < 0.0 {
        ps2[..n2].reverse();
    }
    ps1[n1] = ps1[0];
    ps2[n2] = ps2[0];
    let mut res = 0.0;
    for i in 0..n1 {
        for j in 0..n2 {
            res += intersect_area_tri(ps1[i], ps1[i + 1], ps2[j], ps2[j + 1], trace);
        }
    }
    res
}

/// Upstream `iou_poly(p, q)`, with `p` the ground truth and `q` the
/// detection — the order `dota_evaluation_task1.py` calls it in.
///
/// Returns the oracle's raw value, including `NaN` when both quads have
/// zero area (`0/0`, quirk **OB7**) and including any residue on
/// polygon-disjoint pairs (quirk **OB8**). [`iou`] is the composed
/// entry point that applies the prefilter and normalizes; prefer it.
#[must_use]
pub fn iou_poly_raw(gt: &[f64; 8], dt: &[f64; 8]) -> (f64, FanTrace) {
    let mut trace = FanTrace::default();
    let mut ps1 = [P::default(); DK_MAXN];
    let mut ps2 = [P::default(); DK_MAXN];
    for i in 0..4 {
        ps1[i] = P {
            x: gt[2 * i],
            y: gt[2 * i + 1],
        };
        ps2[i] = P {
            x: dt[2 * i],
            y: dt[2 * i + 1],
        };
    }
    let inter = intersect_area_poly(&mut ps1, 4, &mut ps2, 4, &mut trace);
    let union = area(&mut ps1, 4).abs() + area(&mut ps2, 4).abs() - inter;
    (inter / union, trace)
}

/// Axis-aligned envelope of a quad, `(min_x, min_y, max_x, max_y)`.
fn envelope(q: &[f64; 8]) -> (f64, f64, f64, f64) {
    let (mut nx, mut ny) = (q[0], q[1]);
    let (mut xx, mut xy) = (q[0], q[1]);
    for i in 1..4 {
        nx = nx.min(q[2 * i]);
        xx = xx.max(q[2 * i]);
        ny = ny.min(q[2 * i + 1]);
        xy = xy.max(q[2 * i + 1]);
    }
    (nx, ny, xx, xy)
}

/// DOTA task-1's horizontal-box gate, reproduced exactly.
///
/// Pascal VOC's `+1` convention: widths and heights are computed as
/// `max - min + 1`, on the theory that a box spanning pixels 3 through 5
/// is three pixels wide. The gate keeps a ground truth when the
/// resulting envelope IoU is strictly positive.
///
/// The `+1` matters at the margin: two envelopes that merely touch, or
/// that are separated by less than one pixel, still pass. Dropping it —
/// or substituting an exact envelope test — would reject pairs on which
/// the fan decomposition returns a non-zero residue, and the matrix
/// would no longer be bit-equal.
#[must_use]
pub fn hbb_plus_one_overlap(gt: &[f64; 8], dt: &[f64; 8]) -> bool {
    let (gx0, gy0, gx1, gy1) = envelope(gt);
    let (bx0, by0, bx1, by1) = envelope(dt);
    let ixmin = gx0.max(bx0);
    let iymin = gy0.max(by0);
    let ixmax = gx1.min(bx1);
    let iymax = gy1.min(by1);
    let iw = (ixmax - ixmin + 1.0).max(0.0);
    let ih = (iymax - iymin + 1.0).max(0.0);
    let inters = iw * ih;
    let uni =
        (bx1 - bx0 + 1.0) * (by1 - by0 + 1.0) + (gx1 - gx0 + 1.0) * (gy1 - gy0 + 1.0) - inters;
    // `NaN > 0.0` is false, which is also what numpy's mask does.
    inters / uni > 0.0
}

/// The composed DK oracle: prefilter, then `iou_poly`.
///
/// `gt` and `dt` are `[x0, y0, ..., x3, y3]` in submission order; DK
/// consumes vertex order as given (it only reverses to normalize
/// winding), so the caller must not canonicalize first.
///
/// Normalization applied on the way out, and only here: `-0.0` becomes
/// `+0.0` (quirk **OB14**, and ADR-0063's bit-equality relation is
/// defined modulo signed zero anyway), and a non-finite result becomes
/// `+0.0` because ADR-0005's matrix contract forbids `NaN`. The latter
/// is unreachable through vernier's own ingestion, which rejects
/// zero-area quads outright (quirk **OB7**).
#[must_use]
pub fn iou(gt: &[f64; 8], dt: &[f64; 8]) -> f64 {
    if !hbb_plus_one_overlap(gt, dt) {
        return 0.0;
    }
    let (v, _) = iou_poly_raw(gt, dt);
    if v.is_finite() {
        v + 0.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNIT: [f64; 8] = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn upstream_worked_example() {
        // `polyiou.cpp`'s own commented-out `main`: the unit square
        // against the same square shifted by (0.5, 0.5). Intersection
        // 0.25, union 1.75.
        let q = [0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5];
        let (v, _) = iou_poly_raw(&UNIT, &q);
        assert!(close(v, 0.25 / 1.75), "{v}");
    }

    #[test]
    fn identical_quads_score_one() {
        let (v, _) = iou_poly_raw(&UNIT, &UNIT);
        assert!(close(v, 1.0), "{v}");
    }

    #[test]
    fn reversed_winding_is_normalized() {
        let cw = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let (v, _) = iou_poly_raw(&UNIT, &cw);
        assert!(close(v, 1.0), "{v}");
    }

    #[test]
    fn zero_area_yields_nan_raw_and_zero_composed() {
        let degenerate = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let (v, _) = iou_poly_raw(&degenerate, &degenerate);
        assert!(v.is_nan(), "expected the 0/0 path, got {v}");
        assert_eq!(iou(&degenerate, &degenerate), 0.0);
    }

    #[test]
    fn hbb_gate_uses_the_voc_plus_one() {
        let a = UNIT;
        // Envelopes that merely touch: `iw = 1 - 1 + 1 = 1 > 0`, kept.
        // An exact envelope test would reject this pair.
        let touching = [1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 1.0, 1.0];
        assert!(hbb_plus_one_overlap(&a, &touching));
        // A sub-pixel gap is still kept: `iw = 1 - 1.5 + 1 = 0.5`.
        let near = [1.5, 0.0, 2.5, 0.0, 2.5, 1.0, 1.5, 1.0];
        assert!(hbb_plus_one_overlap(&a, &near));
        // A gap of exactly one pixel is where the `+1` runs out:
        // `iw = 1 - 2 + 1 = 0`, which is not strictly positive.
        let gap = [2.0, 0.0, 3.0, 0.0, 3.0, 1.0, 2.0, 1.0];
        assert!(!hbb_plus_one_overlap(&a, &gap));
        assert_eq!(iou(&a, &gap), 0.0);
    }

    #[test]
    fn composed_oracle_is_positive_zero_off_the_gate() {
        let far = [100.0, 100.0, 101.0, 100.0, 101.0, 101.0, 100.0, 101.0];
        let v = iou(&UNIT, &far);
        assert_eq!(v.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn half_overlap() {
        let b = [0.5, 0.0, 1.5, 0.0, 1.5, 1.0, 0.5, 1.0];
        let (v, _) = iou_poly_raw(&UNIT, &b);
        assert!(close(v, 0.5 / 1.5), "{v}");
    }

    #[test]
    fn non_convex_quad_is_handled_by_the_fan() {
        // Arrowhead (area 5) against a 10x10 square that contains it:
        // IoU is area(arrow) / area(square).
        let arrow = [0.0, 0.0, 4.0, 0.0, 2.0, 4.0, 2.0, 1.0];
        let square = [-5.0, -5.0, 5.0, -5.0, 5.0, 5.0, -5.0, 5.0];
        let (v, _) = iou_poly_raw(&square, &arrow);
        assert!(close(v, 5.0 / 100.0), "{v}");
    }

    #[test]
    fn integer_coordinates_never_reach_the_stale_slot() {
        // The stale path needs two cross products within 1e-8 of zero
        // on opposite sides. Integer vertices make every cross product
        // an exact integer, so a sign change forces a gap of at least 1.
        let mut total = FanTrace::default();
        for dx in 0..12_i32 {
            for dy in 0..12_i32 {
                let g = [0.0, 0.0, 10.0, 0.0, 10.0, 6.0, 0.0, 6.0];
                let fx = f64::from(dx);
                let fy = f64::from(dy);
                let d = [fx, fy, fx + 7.0, fy, fx + 7.0, fy + 9.0, fx, fy + 9.0];
                let (_, trace) = iou_poly_raw(&g, &d);
                total.stale_slots += trace.stale_slots;
            }
        }
        assert_eq!(total.stale_slots, 0, "stale-slot path fired on integers");
    }

    #[test]
    fn translation_invariance_is_only_approximate() {
        // The fan hinges on the coordinate origin, so moving a pair far
        // from it costs accuracy. f64 absorbs it; this test records the
        // magnitude rather than asserting it away.
        let g = UNIT;
        let d = [0.25, 0.25, 1.25, 0.25, 1.25, 1.25, 0.25, 1.25];
        let (near, _) = iou_poly_raw(&g, &d);
        let off = 1.0e6;
        let shift = |q: &[f64; 8]| {
            let mut r = *q;
            for i in 0..4 {
                r[2 * i] += off;
                r[2 * i + 1] += off;
            }
            r
        };
        let (far, _) = iou_poly_raw(&shift(&g), &shift(&d));
        assert!((near - far).abs() < 1e-6, "{near} vs {far}");
    }
}
