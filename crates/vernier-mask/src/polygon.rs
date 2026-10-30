//! Polygon → RLE rasterizer.
//!
//! Mirrors `rleFrPoly` (`mc:193-235`) and `rleFrBbox` (`mc:180-187`)
//! from `pycocotools-2.0.11/common/maskApi.c`. The whole point of
//! this file is to reproduce the C function's pixel-coverage rule,
//! which is itself defined operationally rather than mathematically.
//!
//! Algorithm in three stages:
//!
//! 1. **Upsample (quirk H3).** Each vertex is mapped to the
//!    supersampled grid via `(int)(5x + 0.5)` — rounds toward zero
//!    after the half-bias, matching the C cast bit-for-bit on
//!    in-range inputs.
//! 2. **Densify.** Each edge is walked along its major axis with
//!    a Bresenham-like loop, producing one supersampled point per
//!    minor-axis unit. The C buffers every point; vernier walks each
//!    edge in place and visits only the steps that can survive stage 3
//!    (see `edge_crossings`).
//! 3. **Y-boundary downsample (quirks H4, H5).** A "vertical step"
//!    between consecutive points (`u[j] != u[j-1]`) becomes one
//!    image-coord boundary crossing. The x-coord is selected
//!    direction-aware (the column the boundary just left) and must
//!    land *exactly* on an integer pixel boundary; the y-coord is
//!    always the smaller of the two endpoints, clamped to `[0, h]`,
//!    and ceiled. The asymmetry is what makes shared polygon edges
//!    cancel cleanly under [`Rle::merge`].
//! 4. **Encode (quirk H6).** Crossings are sorted (a counting sort on
//!    the column; see `sorted_flat_indices`), differential-
//!    encoded, and zero-length runs are folded back into the
//!    previous run — that's how duplicated edges (a polygon's two
//!    sides crossing the same supersampled cell) cancel out.

use crate::error::{MalformedPolygonReason, MalformedRleReason, MaskError};
use crate::rle::Rle;

/// Supersampling factor. Pinned at `5` to match `rleFrPoly`; the one
/// source for the f64 upsample and the integer downsample.
const SUPERSAMPLE_SCALE_INT: u32 = 5;
#[allow(clippy::cast_lossless)]
const SUPERSAMPLE_SCALE: f64 = SUPERSAMPLE_SCALE_INT as f64;

/// The downsample maps supersampled `s` to `(s + 0.5) / scale - 0.5 =
/// (s - BOUNDARY_OFFSET) / scale` image units, an integer exactly when
/// `s ≡ BOUNDARY_OFFSET (mod scale)`. Needs an odd scale.
#[allow(clippy::cast_possible_wrap)]
const BOUNDARY_OFFSET: i32 = (SUPERSAMPLE_SCALE_INT as i32 - 1) / 2;
const _: () = assert!(SUPERSAMPLE_SCALE_INT % 2 == 1);

/// Minimum vertex count vernier accepts. Quirk **K1** disposition
/// `corrected`: the C code happily rasterizes 1- and 2-vertex
/// inputs to garbage; we reject them.
const MIN_VERTICES: usize = 3;

/// Upper bound on supersampled coordinates. Realistic polygons
/// (coords ≤ image dimension) never approach this; the clamp is
/// purely defensive so that downstream `i32` differences in the
/// densification loop cannot overflow on absurd-but-finite input.
const SUPERSAMPLE_COORD_BOUND: i32 = i32::MAX / 4;

impl Rle {
    /// Rasterizes a closed polygon into an RLE.
    ///
    /// `xy` is a flat slice `[x0, y0, x1, y1, ...]` of pixel
    /// coordinates; the first and last vertex are connected
    /// implicitly (no need to repeat the first vertex).
    ///
    /// Errors per quirk **K1** disposition `corrected` on:
    /// - odd-length `xy` (not a sequence of `(x, y)` pairs),
    /// - fewer than 3 vertices,
    /// - any non-finite (`NaN` / `±∞`) coordinate.
    ///
    /// Returns the empty `h × w` RLE for `h == 0 || w == 0`.
    pub fn from_polygon(xy: &[f64], h: u32, w: u32) -> Result<Self, MaskError> {
        if xy.len() % 2 != 0 {
            return Err(MaskError::MalformedPolygon(
                MalformedPolygonReason::OddCoordinateCount(xy.len()),
            ));
        }
        let k = xy.len() / 2;
        if k < MIN_VERTICES {
            return Err(MaskError::MalformedPolygon(
                MalformedPolygonReason::TooFewVertices(k),
            ));
        }
        for (i, &v) in xy.iter().enumerate() {
            if !v.is_finite() {
                return Err(MaskError::MalformedPolygon(
                    MalformedPolygonReason::NonFiniteCoordinate(i),
                ));
            }
        }
        if h == 0 || w == 0 {
            return Ok(Rle::empty(h, w));
        }

        // Closing vertex appended so each edge is `[j, j+1]` for `j ∈ 0..k`.
        let mut sx: Vec<i32> = Vec::with_capacity(k + 1);
        let mut sy: Vec<i32> = Vec::with_capacity(k + 1);
        for j in 0..k {
            sx.push(supersample(xy[2 * j]));
            sy.push(supersample(xy[2 * j + 1]));
        }
        sx.push(sx[0]);
        sy.push(sy[0]);

        let mut a = sorted_flat_indices(polygon_crossings(&sx, &sy, h, w), h);
        // Sentinel `h*w` extends the final run to mask end.
        a.push(u64::from(h) * u64::from(w));

        let mut prev: u64 = 0;
        for slot in a.iter_mut() {
            let cur = *slot;
            *slot = cur - prev;
            prev = cur;
        }

        // Compact zero-length runs. Mirrors `mc:231-233`: a zero gap means
        // two crossings landed on the same flat index, so the next run folds
        // back into the most recent emitted run.
        let mut counts: Vec<u32> = Vec::with_capacity(a.len());
        counts.push(
            u32::try_from(a[0])
                .map_err(|_| MaskError::MalformedRle(MalformedRleReason::U32Overflow))?,
        );
        let mut i = 1usize;
        while i < a.len() {
            let cur = a[i];
            i += 1;
            if cur > 0 {
                counts.push(
                    u32::try_from(cur)
                        .map_err(|_| MaskError::MalformedRle(MalformedRleReason::U32Overflow))?,
                );
                continue;
            }
            if i < a.len() {
                let bridged = a[i];
                i += 1;
                let last_idx = counts.len() - 1;
                let merged = u64::from(counts[last_idx]) + bridged;
                counts[last_idx] = u32::try_from(merged)
                    .map_err(|_| MaskError::MalformedRle(MalformedRleReason::U32Overflow))?;
            }
        }

        Ok(Rle::from_counts(h, w, counts))
    }

    /// Rasterizes a multi-polygon segmentation into a single RLE
    /// via union (quirk **K2**: COCO multi-polygon GT — e.g., a
    /// person split across an occluder — is treated as one mask).
    ///
    /// Empty input yields the all-background RLE of shape `(h, w)`
    /// (one run of `h*w` zero pixels, or empty for a 0-area image).
    /// A singleton slice skips the merge round-trip.
    pub fn from_polygons(polys: &[Vec<f64>], h: u32, w: u32) -> Result<Self, MaskError> {
        if let [single] = polys {
            return Self::from_polygon(single, h, w);
        }
        if polys.is_empty() {
            let total = u64::from(h) * u64::from(w);
            if total == 0 {
                return Ok(Rle::empty(h, w));
            }
            let run = u32::try_from(total)
                .map_err(|_| MaskError::MalformedRle(MalformedRleReason::U32Overflow))?;
            return Ok(Rle::from_counts(h, w, vec![run]));
        }
        let rles: Vec<Rle> = polys
            .iter()
            .map(|p| Self::from_polygon(p, h, w))
            .collect::<Result<_, _>>()?;
        Self::merge(&rles, false)
    }

    /// Rasterizes an axis-aligned bbox `[x, y, w, h]` into an RLE.
    ///
    /// Mirrors `rleFrBbox` (`mc:180-187`): a bbox is a 4-vertex
    /// polygon traced counter-clockwise from its top-left corner,
    /// so this function is a thin wrapper over [`Self::from_polygon`].
    pub fn from_bbox(bbox: [f64; 4], h: u32, w: u32) -> Result<Self, MaskError> {
        let [x, y, bw, bh] = bbox;
        let xs = x;
        let xe = x + bw;
        let ys = y;
        let ye = y + bh;
        Self::from_polygon(&[xs, ys, xs, ye, xe, ye, xe, ys], h, w)
    }
}

fn supersample(v: f64) -> i32 {
    let raw = (SUPERSAMPLE_SCALE * v + 0.5) as i32;
    raw.clamp(-SUPERSAMPLE_COORD_BOUND, SUPERSAMPLE_COORD_BOUND)
}

/// Image column of a supersampled boundary crossing, or `None` when the C
/// code drops it (quirk **H5**: off an integer pixel boundary, or outside
/// `[0, w-1]`).
///
/// The predicate both branches of [`edge_crossings`] enumerate directly
/// (kept columns `5c + 2`; see [`boundary_row`] for why the integer form is
/// exact), kept for the tests that tie it to the C's float form. A negative
/// `xd` is dropped either way, so the unsigned form needs no sign fix-up.
#[cfg(test)]
fn boundary_column(xd_super: i32, w: u32) -> Option<u64> {
    let n = u32::try_from(xd_super - BOUNDARY_OFFSET).ok()?;
    let col = n / SUPERSAMPLE_SCALE_INT;
    (n % SUPERSAMPLE_SCALE_INT == 0 && col < w).then_some(u64::from(col))
}

/// Image row of a supersampled boundary crossing: the C's
/// `ceil(clamp((yd_super + 0.5) / 5 - 0.5, 0, h))` (quirk **H4**), i.e.
/// `ceil((yd_super - 2) / 5)` clamped to `[0, h]`, and `clamp` / `ceil`
/// commute because both bounds are integers. A non-positive numerator ceils
/// to at most 0.
///
/// Exact, and so is the column rule both branches of [`edge_crossings`]
/// enumerate (the C keeps `xd = (xd_super + 0.5) / 5 - 0.5` when
/// `floor(xd) == xd`, that is `xd_super ≡ 2 (mod 5)`): for
/// `|xd_super| ≤ i32::MAX / 4 + 1` the f64 evaluation of `(s - 2) / 5` is
/// exact on that residue and stays ≥ 0.2 from an integer off it, so the
/// integer forms are the same predicates, without the `trunc` / `ceil` the
/// x86-64 baseline lowers to libm calls.
fn boundary_row(yd_super: i32, h: u32) -> u64 {
    let n = u32::try_from(yd_super - BOUNDARY_OFFSET).unwrap_or(0);
    let ceil = n.div_ceil(SUPERSAMPLE_SCALE_INT);
    u64::from(ceil.min(h))
}

/// A boundary crossing at image `(col, row)`, packed as `col << 32 | row`.
///
/// Packed order agrees with column-major flat order `col * h + row`: rows
/// run `0..=h`, so the flat index ties only `(c, h)` with `(c + 1, 0)` —
/// equal values, whose relative order the run encoding cannot see.
#[inline]
fn pack(col: u64, row: u64) -> u64 {
    col << 32 | row
}

/// Sorts packed crossings into ascending column-major flat indices, with one
/// slot spare for the end sentinel.
///
/// The crossings of one polygon fall in its column span, and few share a
/// column — two per column for a convex outline. So a counting sort on the
/// column, over the span, scatters each crossing straight to its flat index
/// and leaves disorder only inside a column: `O(n + span)`, where the
/// comparison sort it replaces was half of rasterizing COCO's ground truth.
/// Within a column flat order is row order. Shallow columns are finished by
/// one insertion pass; if any column is deep (a comb or zigzag outline can
/// stack thousands of crossings in one), each column is sorted on its own
/// instead, so no input shape is quadratic. Small or sparse sets, where the
/// counts array would not pay, take the comparison sort.
fn sorted_flat_indices(mut keys: Vec<u64>, h: u32) -> Vec<u64> {
    const MIN_COUNTING: usize = 64;
    /// Deepest column the insertion pass takes.
    const MAX_INSERTION_DEPTH: usize = 16;
    let h = u64::from(h);
    let flat = |k: u64| (k >> 32) * h + (k & u64::from(u32::MAX));
    let (lo, hi) = keys.iter().fold((u64::MAX, 0), |(lo, hi), &k| {
        (lo.min(k >> 32), hi.max(k >> 32))
    });
    let span = usize::try_from(hi.saturating_sub(lo)).map_or(usize::MAX, |s| s + 1);
    if keys.len() < MIN_COUNTING || span > 2 * keys.len() {
        keys.sort_unstable();
        for k in &mut keys {
            *k = flat(*k);
        }
        return keys;
    }
    let column = |k: u64| (k >> 32).abs_diff(lo) as usize;
    // `starts[c]` is where column `lo + c` begins in the output.
    let mut starts = vec![0usize; span + 1];
    for &k in &keys {
        starts[column(k) + 1] += 1;
    }
    let deepest = starts.iter().copied().max().unwrap_or(0);
    for c in 1..=span {
        starts[c] += starts[c - 1];
    }
    let mut out = vec![0u64; keys.len() + 1];
    out.truncate(keys.len());
    for &k in &keys {
        let slot = &mut starts[column(k)];
        out[*slot] = flat(k);
        *slot += 1;
    }
    if deepest <= MAX_INSERTION_DEPTH {
        for i in 1..out.len() {
            let k = out[i];
            let mut j = i;
            while j > 0 && out[j - 1] > k {
                out[j] = out[j - 1];
                j -= 1;
            }
            out[j] = k;
        }
    } else {
        // After the scatter, `starts[c]` is where column `c` ends.
        let mut begin = 0;
        for &end in &starts[..span] {
            out[begin..end].sort_unstable();
            begin = end;
        }
    }
    out
}

/// Boundary crossings (quirks **H4**, **H5**) of the closed supersampled
/// polygon `(sx, sy)` (closing vertex repeated), [packed](pack) and unsorted;
/// one slot is left spare for the end sentinel.
fn polygon_crossings(sx: &[i32], sy: &[i32], h: u32, w: u32) -> Vec<u64> {
    let edges = || sx.windows(2).zip(sy.windows(2));
    // H5 keeps one column in five: at most `dx / 5 + 1` crossings per edge,
    // whichever axis is major.
    let estimate: usize = edges()
        .map(|(x, _)| x[1].abs_diff(x[0]) as usize / SUPERSAMPLE_SCALE_INT as usize + 1)
        .sum();
    let mut out = Vec::with_capacity(estimate + 1);
    for (x, y) in edges() {
        edge_crossings(x[0], x[1], y[0], y[1], h, w, &mut out);
    }
    out
}

/// Appends the boundary crossings of one polygon edge, [packed](pack), to
/// `out`.
///
/// The C densifies every edge into a point buffer (`mc:199-216`) and scans
/// consecutive points for a column change (`mc:217-226`). Every crossing it
/// keeps comes from a pair inside one edge: consecutive edges meet at the
/// same point except where a negative coordinate truncates differently,
/// and such a pair sits left of column 0, which H5 drops. The pairs are
/// unordered for the result (crossings are sorted; `yd` takes the minimum),
/// so each edge is walked in its rotated frame, with the C's own `t`
/// expressions — the same bits — and nothing is buffered.
fn edge_crossings(
    mut xs: i32,
    mut xe: i32,
    mut ys: i32,
    mut ye: i32,
    h: u32,
    w: u32,
    out: &mut Vec<u64>,
) {
    let dx = (xe - xs).unsigned_abs() as i32;
    let dy = (ys - ye).unsigned_abs() as i32;
    // The C `flip` rotates each edge so densification walks forward along
    // the major axis, which is what makes shared edges between adjacent
    // polygons produce identical points and cancel under H6 compaction.
    let flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
    if flip {
        std::mem::swap(&mut xs, &mut xe);
        std::mem::swap(&mut ys, &mut ye);
    }
    if dx >= dy {
        // x-major: point `t` sits at column `xs + t`, so every pair
        // `(t, t + 1)` crosses, at `min = xs + t`. H5 keeps only
        // `xs + t = 5c + 2` with `0 <= c < w`: one pair in five, visited
        // directly. Degenerate edge (dx == dy == 0): no pair.
        let s = if dx == 0 {
            0.0
        } else {
            f64::from(ye - ys) / f64::from(dx)
        };
        let v_at = |t: i64| {
            #[allow(clippy::cast_precision_loss)]
            let t = t as f64;
            (f64::from(ys) + s * t + 0.5) as i32
        };
        let scale = i64::from(SUPERSAMPLE_SCALE_INT);
        let offset = i64::from(BOUNDARY_OFFSET);
        let lo = i64::from(xs).max(offset);
        let hi = (i64::from(xs) + i64::from(dx) - 1).min(scale * (i64::from(w) - 1) + offset);
        let mut x = lo + (offset - lo).rem_euclid(scale);
        while x <= hi {
            let t = x - i64::from(xs);
            let col = (x - offset) / scale;
            let row = boundary_row(v_at(t).min(v_at(t + 1)), h);
            out.push(pack(col.unsigned_abs(), row));
            x += scale;
        }
    } else {
        // y-major: point `t` sits at row `ys + t`, column `u(t)`, and only a
        // move of the column is a crossing. H5 keeps a move only when its
        // smaller column is `k = 5c + 2` with `0 <= c < w`: one in five. The
        // C walks all `dy` steps (most of a COCO rasterization, at five steps
        // per pixel row); here each kept move is found directly. Every
        // operation in `u` is monotone in `t`, so `u` is, and the step at
        // which it passes `k` is a threshold: estimated from the real-valued
        // solution, then settled by evaluating `u` itself, so the result is
        // the C's own arithmetic, not the estimate's.
        let s = f64::from(xe - xs) / f64::from(dy);
        let u_at = |t: i32| (f64::from(xs) + s * f64::from(t) + 0.5) as i32;
        // For the estimate only: the walk below settles `t` exactly.
        let inv_s = 1.0 / s;
        let (u0, u1) = (u_at(0), u_at(dy));
        if u0 == u1 {
            return;
        }
        let rising = u1 > u0;
        // Past `k` means above it when rising, at or below it when falling;
        // false at `t = 0`, true at `t = dy`.
        let past = |t: i32, k: i32| if rising { u_at(t) > k } else { u_at(t) <= k };
        let scale = i64::from(SUPERSAMPLE_SCALE_INT);
        let offset = i64::from(BOUNDARY_OFFSET);
        let lo = i64::from(u0.min(u1)).max(offset);
        let hi = (i64::from(u0.max(u1)) - 1).min(scale * (i64::from(w) - 1) + offset);
        let mut k = lo + (offset - lo).rem_euclid(scale);
        while k <= hi {
            // In range: `k` lies between `u0` and `u1`, both `i32`.
            let k32 = k as i32;
            #[allow(clippy::cast_possible_truncation)] // clamped to `1..=dy`
            let mut t = ((f64::from(k32) + 0.5 - f64::from(xs)) * inv_s)
                .ceil()
                .clamp(1.0, f64::from(dy)) as i32;
            while t > 1 && past(t - 1, k32) {
                t -= 1;
            }
            while !past(t, k32) {
                t += 1;
            }
            // The move at `t` is `u(t - 1) -> u(t)`. Its column is the
            // smaller of the two — the C's choice for a one-column move in
            // either walking direction, and `|s| < 1` keeps moves to one
            // column short of rounding at ~1e8-pixel coordinates. Kept only
            // if that column is `k` itself.
            if u_at(t - 1).min(u_at(t)) == k32 {
                let col = (k - offset) / scale;
                out.push(pack(col.unsigned_abs(), boundary_row(ys + t - 1, h)));
            }
            k += scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// The C's f64 downsample (`mc:219-224`), kept as the oracle for
    /// [`boundary_column`] / [`boundary_row`].
    fn c_boundary(xd_super: i32, yd_super: i32, h: u32, w: u32) -> (Option<u64>, u64) {
        let xd = (f64::from(xd_super) + 0.5) / SUPERSAMPLE_SCALE - 0.5;
        let col = (xd.floor() == xd && xd >= 0.0 && xd <= f64::from(w) - 1.0).then_some(xd as u64);
        let yd = (f64::from(yd_super) + 0.5) / SUPERSAMPLE_SCALE - 0.5;
        (col, yd.clamp(0.0, f64::from(h)).ceil() as u64)
    }

    /// `rleFrPoly`'s densify + downsample (`mc:199-226`) as the C writes
    /// it: every edge densified into a point buffer, every consecutive
    /// pair scanned, the f64 boundary forms. The oracle for
    /// [`polygon_crossings`], which skips the buffer.
    fn c_crossings(sx: &[i32], sy: &[i32], h: u32, w: u32) -> Vec<u64> {
        let (mut u, mut v) = (Vec::new(), Vec::new());
        for j in 0..sx.len() - 1 {
            let (mut xs, mut xe, mut ys, mut ye) = (sx[j], sx[j + 1], sy[j], sy[j + 1]);
            let dx = (xe - xs).abs();
            let dy = (ys - ye).abs();
            let flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
            if flip {
                std::mem::swap(&mut xs, &mut xe);
                std::mem::swap(&mut ys, &mut ye);
            }
            if dx >= dy {
                let s = if dx == 0 {
                    0.0
                } else {
                    f64::from(ye - ys) / f64::from(dx)
                };
                for d in 0..=dx {
                    let t = if flip { dx - d } else { d };
                    u.push(t + xs);
                    v.push((f64::from(ys) + s * f64::from(t) + 0.5) as i32);
                }
            } else {
                let s = f64::from(xe - xs) / f64::from(dy);
                for d in 0..=dy {
                    let t = if flip { dy - d } else { d };
                    v.push(t + ys);
                    u.push((f64::from(xs) + s * f64::from(t) + 0.5) as i32);
                }
            }
        }
        let mut a = Vec::new();
        for j in 1..u.len() {
            if u[j] == u[j - 1] {
                continue;
            }
            let xd_super = if u[j] < u[j - 1] { u[j] } else { u[j] - 1 };
            let (col, row) = c_boundary(xd_super, v[j].min(v[j - 1]), h, w);
            if let Some(col) = col {
                a.push(col * u64::from(h) + row);
            }
        }
        a
    }

    /// Edge-by-edge crossings, sorted, against the C's dense scan.
    fn crossings_match_dense_scan(
        vertices: &[(f64, f64, bool)],
        h: u32,
        w: u32,
    ) -> Result<(), TestCaseError> {
        let mut sx: Vec<i32> = Vec::new();
        let mut sy: Vec<i32> = Vec::new();
        for &(x, y, snap) in vertices {
            let (x, y) = if snap { (x.round(), y.round()) } else { (x, y) };
            sx.push(supersample(x));
            sy.push(supersample(y));
        }
        sx.push(sx[0]);
        sy.push(sy[0]);
        let got = sorted_flat_indices(polygon_crossings(&sx, &sy, h, w), h);
        let mut want = c_crossings(&sx, &sy, h, w);
        want.sort_unstable();
        prop_assert_eq!(got, want);
        Ok(())
    }

    proptest! {
        /// Edge-by-edge crossings are the C's dense scan:
        /// random polygons with vertices left of, above, inside and past
        /// the image (negative coordinates truncate differently at
        /// joins), duplicate vertices, axis-aligned and degenerate edges.
        #[test]
        fn edge_crossings_match_the_dense_scan(
            vertices in proptest::collection::vec(
                (-30.0_f64..90.0, -30.0_f64..90.0, proptest::bool::weighted(0.15)),
                3..14,
            ),
            h in 1_u32..70,
            w in 1_u32..70,
        ) {
            crossings_match_dense_scan(&vertices, h, w)?;
        }

        /// The same at COCO scale: long, steep edges, where the y-major
        /// threshold search replaces hundreds of steps.
        #[test]
        fn edge_crossings_match_the_dense_scan_at_image_scale(
            vertices in proptest::collection::vec(
                (-60.0_f64..760.0, -60.0_f64..760.0, proptest::bool::weighted(0.15)),
                3..30,
            ),
            h in 1_u32..700,
            w in 1_u32..700,
        ) {
            crossings_match_dense_scan(&vertices, h, w)?;
        }

        /// The counting sort orders exactly as a comparison sort of the
        /// flat indices, on dense, sparse, tied and row-`h` crossings.
        #[test]
        fn counting_sort_matches_comparison_sort(
            cells in proptest::collection::vec((0_u64..300, 0_u64..=40), 0..400),
            spread in 1_u64..8,
        ) {
            let h = 40_u32;
            let keys: Vec<u64> = cells.iter().map(|&(c, r)| pack(c * spread, r)).collect();
            let mut want: Vec<u64> = keys
                .iter()
                .map(|&k| (k >> 32) * u64::from(h) + (k & u64::from(u32::MAX)))
                .collect();
            want.sort_unstable();
            prop_assert_eq!(sorted_flat_indices(keys, h), want);
        }

        /// A few deep columns — a comb or zigzag outline — take the
        /// per-column sort and still order exactly.
        #[test]
        fn deep_columns_sort_exactly(
            rows in proptest::collection::vec(0_u64..=500, 64..600),
            cols in 1_u64..4,
        ) {
            let h = 500_u32;
            let keys: Vec<u64> =
                rows.iter().enumerate().map(|(i, &r)| pack(i as u64 % cols, r)).collect();
            let mut want: Vec<u64> = keys
                .iter()
                .map(|&k| (k >> 32) * u64::from(h) + (k & u64::from(u32::MAX)))
                .collect();
            want.sort_unstable();
            prop_assert_eq!(sorted_flat_indices(keys, h), want);
        }
    }

    #[test]
    fn integer_boundary_matches_c_float_form_near_the_image() {
        for s in -2_000..12_000 {
            for (h, w) in [(1, 1), (480, 640), (2_000, 1_000)] {
                let (col, row) = c_boundary(s, s, h, w);
                assert_eq!(boundary_column(s, w), col, "xd_super={s} w={w}");
                assert_eq!(boundary_row(s, h), row, "yd_super={s} h={h}");
            }
        }
    }

    proptest! {
        #[test]
        fn integer_boundary_matches_c_float_form_at_any_coordinate(
            s in -SUPERSAMPLE_COORD_BOUND - 1..=SUPERSAMPLE_COORD_BOUND,
            h in 0_u32..=u32::MAX,
            w in 1_u32..=u32::MAX,
        ) {
            let (col, row) = c_boundary(s, s, h, w);
            prop_assert_eq!(boundary_column(s, w), col);
            prop_assert_eq!(boundary_row(s, h), row);
        }
    }

    #[test]
    fn rejects_odd_coordinate_count() {
        let err = Rle::from_polygon(&[0.0, 0.0, 1.0], 4, 4).unwrap_err();
        assert!(matches!(
            err,
            MaskError::MalformedPolygon(MalformedPolygonReason::OddCoordinateCount(3))
        ));
    }

    #[test]
    fn rejects_fewer_than_three_vertices() {
        // Two-vertex polygon — bbox/poly ambiguity (quirk K1).
        let err = Rle::from_polygon(&[0.0, 0.0, 1.0, 1.0], 4, 4).unwrap_err();
        assert!(matches!(
            err,
            MaskError::MalformedPolygon(MalformedPolygonReason::TooFewVertices(2))
        ));
        // Empty input.
        let err = Rle::from_polygon(&[], 4, 4).unwrap_err();
        assert!(matches!(
            err,
            MaskError::MalformedPolygon(MalformedPolygonReason::TooFewVertices(0))
        ));
    }

    #[test]
    fn rejects_nan_and_infinity() {
        let err = Rle::from_polygon(&[0.0, 0.0, f64::NAN, 1.0, 2.0, 2.0], 4, 4).unwrap_err();
        assert!(matches!(
            err,
            MaskError::MalformedPolygon(MalformedPolygonReason::NonFiniteCoordinate(2))
        ));
        let err = Rle::from_polygon(&[0.0, 0.0, 1.0, f64::INFINITY, 2.0, 2.0], 4, 4).unwrap_err();
        assert!(matches!(
            err,
            MaskError::MalformedPolygon(MalformedPolygonReason::NonFiniteCoordinate(3))
        ));
    }

    #[test]
    fn empty_image_returns_empty_rle() {
        let r = Rle::from_polygon(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 0, 0).unwrap();
        assert_eq!(r, Rle::from_counts(0, 0, vec![]));
        let r = Rle::from_polygon(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 0, 4).unwrap();
        assert_eq!(r, Rle::from_counts(0, 4, vec![]));
        let r = Rle::from_polygon(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 4, 0).unwrap();
        assert_eq!(r, Rle::from_counts(4, 0, vec![]));
    }

    #[test]
    fn axis_aligned_2x2_square_in_4x4_image() {
        // Polygon (0,0)-(2,0)-(2,2)-(0,2): traced by hand below in the
        // module docs. Expected: bg=0, fg=2, bg=2, fg=2, bg=10.
        let r = Rle::from_polygon(&[0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0], 4, 4).unwrap();
        assert_eq!(&r.counts[..], &[0u32, 2, 2, 2, 10][..]);
        // Round-trip through the raster codec to confirm the FG
        // pixel set: (0,0), (0,1), (1,0), (1,1) in column-major.
        let mask = r.to_raster_bytes();
        let mut expected = vec![0u8; 16];
        for x in 0..2 {
            for y in 0..2 {
                expected[x * 4 + y] = 1;
            }
        }
        assert_eq!(mask, expected);
    }

    #[test]
    fn from_bbox_matches_explicit_polygon() {
        let bbox = [0.5, 1.0, 2.0, 3.0];
        let from_bbox = Rle::from_bbox(bbox, 8, 8).unwrap();
        let from_poly = Rle::from_polygon(&[0.5, 1.0, 0.5, 4.0, 2.5, 4.0, 2.5, 1.0], 8, 8).unwrap();
        assert_eq!(from_bbox, from_poly);
    }

    #[test]
    fn polygon_entirely_outside_image_is_empty() {
        // All vertices to the left of the image — every crossing
        // gets dropped by the H5 in-bounds filter.
        let r = Rle::from_polygon(&[-5.0, -5.0, -3.0, -5.0, -3.0, -3.0, -5.0, -3.0], 4, 4).unwrap();
        assert_eq!(&r.counts[..], &[16u32][..]);
        assert_eq!(r.area(), 0);
    }

    #[test]
    fn polygon_clipped_to_image_bounds() {
        // Square partly outside on the right — clipped to a 2x4
        // rectangle covering columns x=2 and x=3 (h=4).
        let r = Rle::from_polygon(&[2.0, 0.0, 6.0, 0.0, 6.0, 4.0, 2.0, 4.0], 4, 4).unwrap();
        // Expected mask: columns 0-1 bg, columns 2-3 fg.
        let mask = r.to_raster_bytes();
        let mut expected = vec![0u8; 16];
        for x in 2..4 {
            for y in 0..4 {
                expected[x * 4 + y] = 1;
            }
        }
        assert_eq!(mask, expected);
    }

    #[test]
    fn from_polygons_empty_yields_all_background_at_requested_shape() {
        let r = Rle::from_polygons(&[], 4, 4).unwrap();
        assert_eq!(r, Rle::from_counts(4, 4, vec![16]));
        assert_eq!(r.area(), 0);
        let r = Rle::from_polygons(&[], 0, 0).unwrap();
        assert_eq!(r, Rle::from_counts(0, 0, vec![]));
    }

    #[test]
    fn from_polygons_singleton_matches_from_polygon() {
        let xy = vec![0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0];
        let one = Rle::from_polygons(std::slice::from_ref(&xy), 4, 4).unwrap();
        let direct = Rle::from_polygon(&xy, 4, 4).unwrap();
        assert_eq!(one, direct);
    }

    #[test]
    fn from_polygons_multi_unions_disjoint_regions_k2() {
        let a = vec![0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0];
        let b = vec![3.0, 0.0, 5.0, 0.0, 5.0, 2.0, 3.0, 2.0];
        let r = Rle::from_polygons(&[a, b], 8, 8).unwrap();
        // Two 2×2 regions → 4 + 4 foreground pixels.
        assert_eq!(r.area(), 8);
    }

    #[test]
    fn degenerate_collinear_polygon_does_not_panic() {
        // Three collinear points: the polygon has zero area. The
        // algorithm should produce a valid (empty or near-empty)
        // RLE without panicking.
        let r = Rle::from_polygon(&[0.0, 0.0, 1.0, 0.0, 2.0, 0.0], 4, 4).unwrap();
        assert_eq!(r.h, 4);
        assert_eq!(r.w, 4);
        let total: u64 = r.counts.iter().map(|&c| u64::from(c)).sum();
        assert_eq!(total, 16);
    }

    proptest! {
        #[test]
        fn random_polygon_round_trips_to_well_formed_rle(
            pairs in proptest::collection::vec((0.0_f64..16.0_f64, 0.0_f64..16.0_f64), 3..20)
        ) {
            let verts: Vec<f64> = pairs.into_iter().flat_map(|(x, y)| [x, y]).collect();
            let r = Rle::from_polygon(&verts, 16, 16)?;
            // Counts must sum to h*w, and the mask must round-trip.
            let total: u64 = r.counts.iter().map(|&c| u64::from(c)).sum();
            prop_assert_eq!(total, 16 * 16);
            let bytes = r.to_raster_bytes();
            let r2 = Rle::from_raster_bytes(&bytes, 16, 16)?;
            prop_assert_eq!(r, r2);
        }

        #[test]
        fn bbox_polygon_round_trips_via_raster(
            x in 0.0_f64..16.0_f64,
            y in 0.0_f64..16.0_f64,
            bw in 0.5_f64..16.0_f64,
            bh in 0.5_f64..16.0_f64,
        ) {
            let r = Rle::from_bbox([x, y, bw, bh], 16, 16)?;
            let bytes = r.to_raster_bytes();
            prop_assert_eq!(bytes.len(), 16 * 16);
            let r2 = Rle::from_raster_bytes(&bytes, 16, 16)?;
            prop_assert_eq!(r, r2);
        }
    }
}
