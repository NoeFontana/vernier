//! Op-exact replica of detectron2's rotated-box IoU.
//!
//! Upstream: `detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_utils.h`
//! at commit [`crate::pinned::D2_COMMIT_SHA`], vendored verbatim under
//! `tests/python/parity_obb/oracle/detectron2/`. Every function below
//! names the upstream function it replicates.
//!
//! # Precision frame (quirk **OB5**)
//!
//! The upstream is a template. `RotatedCOCOeval` instantiates it at
//! `float`, because `boxlist_to_tensor` builds its tensors with
//! `torch.FloatTensor(...)`. So the geometry runs in **f32**, and this
//! replica is written in `f32` with the handful of deliberate f64 steps
//! spelled out:
//!
//! | Upstream line | Type |
//! | --- | --- |
//! | `double theta = box.a * 0.01745329251` | f32 angle widened, f64 multiply |
//! | `(T)cos(theta) * 0.5f` | f64 `cos`, narrowed, then f32 multiply |
//! | `(box1_raw[0] + box2_raw[0]) / 2.0` | f32 add, f64 divide |
//! | `box1.x_ctr = box1_raw[0] - center_shift_x` | f64 subtract, narrowed to f32 |
//! | every `EPS` comparison | f32 operand promoted, compared in f64 |
//!
//! Getting those five rows wrong is the entire difference between a
//! parity claim and a plausible-looking number.
//!
//! # No FMA, ever
//!
//! The upstream's own Graham-scan step carries a comment explaining that
//! `cross_2d()` may be compiled with FMA and therefore fails to return
//! zero for `q1 == q2`, which is why that step compares the two products
//! directly instead. detectron2's shipped `linux-x86_64` wheels target
//! baseline `x86-64`, which has no FMA for the compiler to contract
//! into, so the pinned claim is against the unfused form. Rust never
//! contracts implicitly, so this file matches by construction — and
//! `replica::tests` scans its own source text for `mul_add` so it stays
//! that way.
//!
//! # Argument order is load-bearing
//!
//! `RotatedCOCOeval.computeIoU` calls `pairwise_iou_rotated(dt, gt)`, so
//! the detection is `box1` and the ground truth is `box2`. The kernel is
//! not symmetric at the bit level: `get_intersection_points` emits line
//! crossings as `pts1[i] + vec1[i] * t1`, which is a different rounding
//! than the same point expressed on the other rectangle's edge, and it
//! collects `pts1`-inside-`pts2` before the reverse. [`iou`] therefore
//! takes `(dt, gt)` in that order and the caller must not swap them.

use crate::pinned::{
    D2_AREA_EPS, D2_CROSS_EPS, D2_DEG_TO_RAD, D2_DET_EPS, D2_DIST_EPS, D2_EPS, D2_MAX_POINTS,
};

/// Upstream `Point<float>`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Pt {
    x: f32,
    y: f32,
}

impl Pt {
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self {
            x: self.x - o.x,
            y: self.y - o.y,
        }
    }
    #[inline]
    fn add_scaled(self, v: Self, t: f32) -> Self {
        // Upstream: `pts1[i] + vec1[i] * t1` — the multiply happens
        // first, into a temporary `Point`, and the add follows. Written
        // out so nothing can fuse.
        let sx = v.x * t;
        let sy = v.y * t;
        Self {
            x: self.x + sx,
            y: self.y + sy,
        }
    }
}

/// Upstream `dot_2d<T>`.
#[inline]
fn dot2(a: Pt, b: Pt) -> f32 {
    a.x * b.x + a.y * b.y
}

/// Upstream `cross_2d<T, T>`.
#[inline]
fn cross2(a: Pt, b: Pt) -> f32 {
    a.x * b.y - b.x * a.y
}

/// Upstream `RotatedBox<float>`.
#[derive(Debug, Clone, Copy)]
struct RBox32 {
    x_ctr: f32,
    y_ctr: f32,
    w: f32,
    h: f32,
    a: f32,
}

/// Upstream `get_rotated_vertices`.
///
/// Note that `pts[2]` and `pts[3]` are *reflections* of `pts[0]` and
/// `pts[1]` through the center — `2*c - p`, not an independent
/// construction. That is why the vertex set is exactly centrally
/// symmetric in f32 even though the trigonometry is not exact, and it is
/// one of the places a "cleaner" rewrite would silently lose parity.
fn get_rotated_vertices(b: RBox32) -> [Pt; 4] {
    let theta = f64::from(b.a) * D2_DEG_TO_RAD;
    #[allow(clippy::cast_possible_truncation)]
    let cos_t2 = (theta.cos() as f32) * 0.5_f32;
    #[allow(clippy::cast_possible_truncation)]
    let sin_t2 = (theta.sin() as f32) * 0.5_f32;

    let mut p = [Pt::default(); 4];
    p[0].x = b.x_ctr + sin_t2 * b.h + cos_t2 * b.w;
    p[0].y = b.y_ctr + cos_t2 * b.h - sin_t2 * b.w;
    p[1].x = b.x_ctr - sin_t2 * b.h + cos_t2 * b.w;
    p[1].y = b.y_ctr - cos_t2 * b.h - sin_t2 * b.w;
    p[2].x = 2.0_f32 * b.x_ctr - p[0].x;
    p[2].y = 2.0_f32 * b.y_ctr - p[0].y;
    p[3].x = 2.0_f32 * b.x_ctr - p[1].x;
    p[3].y = 2.0_f32 * b.y_ctr - p[1].y;
    p
}

/// Upstream `get_intersection_points`.
///
/// Collects, in this order: every relaxed line-line crossing of the
/// sixteen edge pairs, then the vertices of rect 1 inside rect 2, then
/// the vertices of rect 2 inside rect 1. Duplicates are expected and
/// harmless — the upstream comment says as much, and the hull that
/// follows dedupes by construction. Order matters because the hull's
/// tie-breaking is a *stable-by-index* exchange sort, so a different
/// collection order can produce a different vertex sequence and
/// therefore different last bits of area.
fn get_intersection_points(pts1: &[Pt; 4], pts2: &[Pt; 4]) -> ([Pt; D2_MAX_POINTS], usize) {
    let mut vec1 = [Pt::default(); 4];
    let mut vec2 = [Pt::default(); 4];
    for i in 0..4 {
        vec1[i] = pts1[(i + 1) % 4].sub(pts1[i]);
        vec2[i] = pts2[(i + 1) % 4].sub(pts2[i]);
    }

    let mut out = [Pt::default(); D2_MAX_POINTS];
    let mut num = 0_usize;

    for i in 0..4 {
        for j in 0..4 {
            let det = cross2(vec2[j], vec1[i]);
            if f64::from(det).abs() <= D2_DET_EPS {
                continue;
            }
            let vec12 = pts2[j].sub(pts1[i]);
            let t1 = cross2(vec2[j], vec12) / det;
            let t2 = cross2(vec1[i], vec12) / det;
            // `EPS` is a `double` and `t1`/`t2` are `float`, so the
            // upstream comparison promotes. `1.0f + EPS` likewise.
            let (t1d, t2d) = (f64::from(t1), f64::from(t2));
            if t1d > -D2_EPS && t1d < 1.0 + D2_EPS && t2d > -D2_EPS && t2d < 1.0 + D2_EPS {
                if num < D2_MAX_POINTS {
                    out[num] = pts1[i].add_scaled(vec1[i], t1);
                }
                num += 1;
            }
        }
    }

    num = collect_inside(pts1, pts2, &vec2, &mut out, num);
    num = collect_inside(pts2, pts1, &vec1, &mut out, num);
    (out, num.min(D2_MAX_POINTS))
}

/// The "vertices of one rectangle inside the other" block, written once
/// because upstream spells it twice with the roles swapped.
///
/// A point is inside `ABCD` iff its projection on `AB` lies within `AB`
/// and its projection on `AD` lies within `AD`. Both tests are relaxed
/// by `EPS` on an **unnormalized** dot product, so the slack in world
/// units is `EPS / |AB|` — it grows as the rectangle gets thinner. That
/// is the term that dominates [`reach`].
fn collect_inside(
    probe: &[Pt; 4],
    host: &[Pt; 4],
    host_vec: &[Pt; 4],
    out: &mut [Pt; D2_MAX_POINTS],
    mut num: usize,
) -> usize {
    let ab = host_vec[0];
    let da = host_vec[3];
    let ab_dot_ab = dot2(ab, ab);
    let ad_dot_ad = dot2(da, da);
    for &p in probe.iter() {
        let ap = p.sub(host[0]);
        let ap_dot_ab = f64::from(dot2(ap, ab));
        let ap_dot_ad = f64::from(-dot2(ap, da));
        if ap_dot_ab > -D2_EPS
            && ap_dot_ad > -D2_EPS
            && ap_dot_ab < f64::from(ab_dot_ab) + D2_EPS
            && ap_dot_ad < f64::from(ad_dot_ad) + D2_EPS
        {
            if num < D2_MAX_POINTS {
                out[num] = p;
            }
            num += 1;
        }
    }
    num
}

/// Upstream `convex_hull_graham` with `shift_to_zero = true`.
///
/// # Why this is reproducible at all
///
/// ADR-0063 flagged the hull as the one step that might sink the D2
/// strict claim: an epsilon-based comparator is not a strict weak
/// ordering, and feeding one to `std::sort` makes the output depend on
/// libstdc++'s introsort internals. The audit settles it — **the
/// `std::sort` call is commented out upstream**. Both the CPU and CUDA
/// paths run the same hand-written `O(n^2)` exchange sort reproduced
/// below, which is fully specified by its own source. There is no
/// standard-library ordering dependency and no libstdc++ version to pin.
fn convex_hull_graham(p: &[Pt; D2_MAX_POINTS], num_in: usize) -> ([Pt; D2_MAX_POINTS], usize) {
    // Step 1: lowest y, ties broken by lowest x.
    let mut t = 0_usize;
    for i in 1..num_in {
        if p[i].y < p[t].y || (p[i].y == p[t].y && p[i].x < p[t].x) {
            t = i;
        }
    }
    let start = p[t];

    // Step 2: shift so the start point is the origin, then swap it to
    // index 0.
    let mut q = [Pt::default(); D2_MAX_POINTS];
    for i in 0..num_in {
        q[i] = p[i].sub(start);
    }
    q.swap(0, t);

    // Step 3: order by angle. An exchange sort, not a comparison sort:
    // for each `i` it walks every later `j` and swaps whenever `j`
    // should precede `i`. Ties in angle (`|cross| < 1e-6`) fall back to
    // squared distance.
    let mut dist = [0.0_f32; D2_MAX_POINTS];
    for i in 0..num_in {
        dist[i] = dot2(q[i], q[i]);
    }
    for i in 1..num_in.saturating_sub(1) {
        for j in i + 1..num_in {
            let cp = f64::from(cross2(q[i], q[j]));
            if cp < -D2_CROSS_EPS || (cp.abs() < D2_CROSS_EPS && dist[i] > dist[j]) {
                q.swap(i, j);
                dist.swap(i, j);
            }
        }
    }
    // Distances are recomputed *after* the sort, because the points at
    // each index are now different ones.
    for i in 0..num_in {
        dist[i] = dot2(q[i], q[i]);
    }

    // Step 4: find the first point that is not coincident with the
    // start.
    let mut k = 1_usize;
    while k < num_in {
        if f64::from(dist[k]) > D2_DIST_EPS {
            break;
        }
        k += 1;
    }
    if k == num_in {
        // The hull collapsed to a point. Upstream writes the unshifted
        // `p[t]` back and returns 1; with `m <= 2` the area is zero
        // either way.
        q[0] = start;
        return (q, 1);
    }

    q[1] = q[k];
    let mut m = 2_usize;
    // Step 5: the scan. The convexity test compares the two products
    // directly rather than calling `cross_2d`, precisely so an FMA
    // cannot make `q1 == q2` produce a non-zero cross.
    for i in k + 1..num_in {
        while m > 1 {
            let q1 = q[i].sub(q[m - 2]);
            let q2 = q[m - 1].sub(q[m - 2]);
            if q1.x * q2.y >= q2.x * q1.y {
                m -= 1;
            } else {
                break;
            }
        }
        if m < D2_MAX_POINTS {
            q[m] = q[i];
            m += 1;
        }
    }
    // Step 6 is skipped: `shift_to_zero = true`, so the hull stays in
    // start-relative coordinates. That is not merely an optimization —
    // the area below is computed on these smaller numbers, and shifting
    // back first would change its last bits.
    (q, m)
}

/// Upstream `polygon_area`.
///
/// Note the `fabs` **inside** the sum: each fan triangle contributes its
/// magnitude, so a hull that is not convex — which the epsilon-relaxed
/// scan can produce — accumulates rather than cancels.
fn polygon_area(q: &[Pt; D2_MAX_POINTS], m: usize) -> f32 {
    if m <= 2 {
        return 0.0;
    }
    let mut area = 0.0_f32;
    for i in 1..m - 1 {
        area += cross2(q[i].sub(q[0]), q[i + 1].sub(q[0])).abs();
    }
    #[allow(clippy::cast_possible_truncation)]
    let half = (f64::from(area) / 2.0) as f32;
    half
}

/// Upstream `rotated_boxes_intersection`.
fn rotated_boxes_intersection(box1: RBox32, box2: RBox32) -> f32 {
    let pts1 = get_rotated_vertices(box1);
    let pts2 = get_rotated_vertices(box2);
    let (pts, num) = get_intersection_points(&pts1, &pts2);
    if num <= 2 {
        return 0.0;
    }
    let (ordered, m) = convex_hull_graham(&pts, num);
    polygon_area(&ordered, m)
}

/// Upstream `single_box_iou_rotated<float>`, in `(box1, box2)` order.
///
/// `box1` is the **detection** and `box2` the **ground truth**; see the
/// module docs on why that is not interchangeable.
///
/// The returned value is whatever the oracle computes, including values
/// outside `[0, 1]` (quirk **OB8**, detectron2#350): `strict` preserves
/// them and the matching engine handles them correctly, since its
/// threshold guard is `min(t, 1 - 1e-10)`. A non-finite result is
/// impossible for inputs that clear the `area >= 1e-14` guard — the
/// denominator is then bounded below by the larger area — but is mapped
/// to `+0.0` anyway, because ADR-0005's matrix contract forbids `NaN`
/// and a silent infinity would corrupt every match in the cell.
#[must_use]
pub fn iou(box1: &[f64; 5], box2: &[f64; 5]) -> f64 {
    #[allow(clippy::cast_possible_truncation)]
    let n = |v: f64| v as f32;
    // `torch.FloatTensor(...)` narrows the JSON f64s to f32 before the
    // kernel ever runs, so the replica must narrow first too.
    let (b1, b2) = (
        [n(box1[0]), n(box1[1]), n(box1[2]), n(box1[3]), n(box1[4])],
        [n(box2[0]), n(box2[1]), n(box2[2]), n(box2[3]), n(box2[4])],
    );

    // Shift both centers to their midpoint. This is the upstream's
    // answer to the f32 error budget: it buys back the dynamic range
    // that an absolute-coordinate shoelace at DOTA scale would spend.
    // Note the mixed types — the sum is f32, the halving is f64, and
    // the result lands back in f32.
    let center_shift_x = f64::from(b1[0] + b2[0]) / 2.0;
    let center_shift_y = f64::from(b1[1] + b2[1]) / 2.0;
    #[allow(clippy::cast_possible_truncation)]
    let shift = |raw: f32, s: f64| (f64::from(raw) - s) as f32;

    let box1 = RBox32 {
        x_ctr: shift(b1[0], center_shift_x),
        y_ctr: shift(b1[1], center_shift_y),
        w: b1[2],
        h: b1[3],
        a: b1[4],
    };
    let box2 = RBox32 {
        x_ctr: shift(b2[0], center_shift_x),
        y_ctr: shift(b2[1], center_shift_y),
        w: b2[2],
        h: b2[3],
        a: b2[4],
    };

    let area1 = box1.w * box1.h;
    let area2 = box2.w * box2.h;
    if f64::from(area1) < D2_AREA_EPS || f64::from(area2) < D2_AREA_EPS {
        return 0.0;
    }

    let intersection = rotated_boxes_intersection(box1, box2);
    let iou = intersection / (area1 + area2 - intersection);
    if iou.is_finite() {
        // Widening f32 -> f64 is exact, and `+ 0.0` normalizes `-0.0`.
        f64::from(iou) + 0.0
    } else {
        0.0
    }
}

/// How far apart two boxes can be and still produce a non-zero D2
/// intersection.
///
/// Every point the oracle collects is either a vertex of one of the two
/// rectangles or a line crossing whose parameter lies within `EPS` of
/// `[0, 1]`, so it sits within `EPS * L` of some edge. The
/// vertex-in-rectangle test adds a second term: its `EPS` slack is on an
/// **unnormalized** projection, worth `EPS / s` in world units for a
/// side of length `s`. The sum, with a safety factor and an f32
/// rounding allowance, is an admissible prefilter margin: outside it,
/// [`iou`] returns exactly `+0.0`.
///
/// Returns infinity — "never reject" — when any side is non-positive,
/// since the `EPS / s` term is then unbounded and the pair is cheap to
/// evaluate anyway.
#[must_use]
pub fn reach(box1: &[f64; 5], box2: &[f64; 5]) -> f64 {
    let longest = box1[2]
        .abs()
        .max(box1[3].abs())
        .max(box2[2].abs())
        .max(box2[3].abs());
    let shortest = box1[2]
        .abs()
        .min(box1[3].abs())
        .min(box2[2].abs())
        .min(box2[3].abs());
    if !shortest.is_finite() || shortest <= 0.0 || !longest.is_finite() {
        return f64::INFINITY;
    }
    let center = box1[0]
        .abs()
        .max(box1[1].abs())
        .max(box2[0].abs())
        .max(box2[1].abs());
    let f32_eps = f64::from(f32::EPSILON);
    4.0 * (D2_EPS * longest + D2_EPS / shortest) + 16.0 * f32_eps * (center + longest)
}

/// Cell-level upper bound on [`reach`], folded one box at a time.
///
/// [`reach`] is a per-pair quantity, and evaluating it inside the
/// `G x D` loop costs a division and a dozen `min`/`max` operations per
/// pair for a number that barely varies across a cell. Every term of it
/// is monotone in the per-box quantities — non-decreasing in the
/// longest side and the center magnitude, non-increasing in the
/// shortest side — so taking those extrema over the whole cell bounds
/// every pair's value from above.
///
/// Bounding from *above* is the admissible direction: a larger margin
/// can only keep pairs that the per-pair margin would also have kept,
/// and the contract only forbids rejecting a pair the oracle would
/// score.
#[derive(Debug, Clone, Copy)]
pub struct ReachBound {
    longest: f64,
    shortest: f64,
    center: f64,
}

impl Default for ReachBound {
    fn default() -> Self {
        Self::new()
    }
}

impl ReachBound {
    /// An empty bound.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            longest: 0.0,
            shortest: f64::INFINITY,
            center: 0.0,
        }
    }

    /// Fold one box, in detectron2 dialect, into the bound.
    pub fn add(&mut self, b: &[f64; 5]) {
        let (w, h) = (b[2].abs(), b[3].abs());
        self.longest = self.longest.max(w).max(h);
        self.shortest = self.shortest.min(w).min(h);
        self.center = self.center.max(b[0].abs()).max(b[1].abs());
    }

    /// The margin, or infinity — "never reject".
    ///
    /// Infinity whenever a box in the cell is degenerate or
    /// non-finite, for the same reason [`reach`] returns it per pair:
    /// the `EPS / s` term is unbounded as a side goes to zero. It also
    /// covers the `NaN` case, which would otherwise make every padded
    /// comparison false and reject the entire cell.
    #[must_use]
    pub fn pad(self) -> f64 {
        if !self.shortest.is_finite()
            || self.shortest <= 0.0
            || !self.longest.is_finite()
            || !self.center.is_finite()
        {
            return f64::INFINITY;
        }
        let f32_eps = f64::from(f32::EPSILON);
        4.0 * (D2_EPS * self.longest + D2_EPS / self.shortest)
            + 16.0 * f32_eps * (self.center + self.longest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reach_bound_dominates_every_pair() {
        let boxes = [
            b(0.0, 0.0, 10.0, 4.0, 0.0),
            b(500.0, -300.0, 3.0, 60.0, 37.0),
            b(-1e4, 2e4, 0.5, 0.5, -90.0),
            b(12.0, 12.0, 120.0, 1.5, 180.0),
        ];
        let mut bound = ReachBound::new();
        for bx in &boxes {
            bound.add(bx);
        }
        let pad = bound.pad();
        assert!(pad.is_finite(), "{pad}");
        for p in &boxes {
            for q in &boxes {
                let exact = reach(p, q);
                assert!(pad >= exact, "pad {pad} < reach {exact}");
            }
        }
    }

    #[test]
    fn a_degenerate_box_disables_the_cell_bound() {
        let mut bound = ReachBound::new();
        bound.add(&b(0.0, 0.0, 10.0, 4.0, 0.0));
        bound.add(&b(0.0, 0.0, 10.0, 0.0, 0.0));
        assert_eq!(bound.pad(), f64::INFINITY);
    }

    fn b(cx: f64, cy: f64, w: f64, h: f64, a: f64) -> [f64; 5] {
        [cx, cy, w, h, a]
    }

    #[test]
    fn identical_boxes_score_one() {
        // f32 arithmetic, so this is "1.0 to f32 resolution", not a
        // bit-exactness claim: D2 makes no such promise and the
        // canonical kernel is where that guarantee lives.
        for a in [0.0, 30.0, 45.0, 90.0, -137.5] {
            let x = b(100.0, 50.0, 20.0, 8.0, a);
            let v = iou(&x, &x);
            assert!((v - 1.0).abs() < 1e-6, "a={a} -> {v}");
        }
    }

    #[test]
    fn axis_aligned_half_overlap() {
        let d = b(0.0, 0.0, 2.0, 2.0, 0.0);
        let g = b(1.0, 0.0, 2.0, 2.0, 0.0);
        let v = iou(&d, &g);
        assert!((v - 1.0 / 3.0).abs() < 1e-6, "{v}");
    }

    #[test]
    fn disjoint_is_exactly_positive_zero() {
        let d = b(0.0, 0.0, 2.0, 2.0, 15.0);
        let g = b(500.0, 500.0, 2.0, 2.0, 15.0);
        let v = iou(&d, &g);
        assert_eq!(v.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn degenerate_area_short_circuits() {
        let d = b(0.0, 0.0, 0.0, 2.0, 0.0);
        let g = b(0.0, 0.0, 2.0, 2.0, 0.0);
        assert_eq!(iou(&d, &g), 0.0);
        assert_eq!(iou(&g, &d), 0.0);
    }

    #[test]
    fn square_against_its_own_forty_five_degree_rotation() {
        let s = 3.0;
        let d = b(0.0, 0.0, s, s, 0.0);
        let g = b(0.0, 0.0, s, s, 45.0);
        let want_i = 2.0 * (core::f64::consts::SQRT_2 - 1.0) * s * s;
        let want = want_i / (2.0 * s * s - want_i);
        let v = iou(&d, &g);
        assert!((v - want).abs() < 1e-5, "{v} vs {want}");
    }

    #[test]
    fn ninety_degree_equivalences_hold_to_f32() {
        let g = b(3.0, 5.0, 6.0, 2.0, 0.0);
        let d = b(3.0, 5.0, 2.0, 6.0, 90.0);
        let v = iou(&d, &g);
        assert!((v - 1.0).abs() < 1e-5, "{v}");
    }

    #[test]
    fn reach_is_finite_for_ordinary_boxes_and_infinite_for_slivers() {
        assert!(reach(&b(0.0, 0.0, 10.0, 5.0, 0.0), &b(1.0, 1.0, 10.0, 5.0, 0.0)).is_finite());
        assert!(reach(&b(0.0, 0.0, 10.0, 0.0, 0.0), &b(1.0, 1.0, 10.0, 5.0, 0.0)).is_infinite());
    }

    /// Admissibility: beyond [`reach`], the oracle is exactly `+0.0`.
    #[test]
    fn beyond_reach_the_oracle_is_zero() {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // `[0, 1)`
            ((state >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
        };
        for _ in 0..20_000 {
            let w1 = 0.5 + rng() * 40.0;
            let h1 = 0.5 + rng() * 40.0;
            let w2 = 0.5 + rng() * 40.0;
            let h2 = 0.5 + rng() * 40.0;
            let a1 = rng() * 360.0 - 180.0;
            let a2 = rng() * 360.0 - 180.0;
            let g = b(0.0, 0.0, w1, h1, a1);
            let margin = reach(&g, &b(0.0, 0.0, w2, h2, a2));
            // Push the detection just past the point where the padded
            // envelopes can still meet.
            let sep = (w1 + h1 + w2 + h2) * 0.5 + margin + 1.0;
            let ang = rng() * core::f64::consts::TAU;
            let d = b(sep * ang.cos(), sep * ang.sin(), w2, h2, a2);
            let v = iou(&d, &g);
            assert_eq!(v, 0.0, "separated pair scored {v}");
        }
    }
}
