//! RLE-on-RLE operations: foreground area, tight bounding box,
//! N-way merge (intersection / union), Chebyshev-ball erosion, and
//! the boundary band used by the boundary-IoU metric.
//!
//! Mirrors `rleArea` (`mc:88-91`), `rleToBbox` (`mc:149-178`), and
//! `rleMerge` (`mc:65-86`) from `pycocotools-2.0.11/common/maskApi.c`.
//! The erosion + boundary-band kernels are not part of pycocotools;
//! their oracle is `bowenc0221/boundary-iou-api` (per ADR-0010).
//!
//! Per pycocotools' column-major (Fortran) convention, a flat pixel
//! index `idx` decomposes as `y = idx % h`, `x = idx / h`.

pub mod boundary;
pub mod erode;

pub use boundary::{boundary_band, boundary_band_into, boundary_band_segments_into};
pub use erode::{erode_chebyshev_ball, erode_chebyshev_ball_into, ErodeScratch};

use crate::error::{MalformedRleReason, MaskError};
use crate::rle::Rle;

impl Rle {
    /// Foreground area in pixels. Per quirk **G5**, foreground runs
    /// sit at odd indices.
    ///
    /// Returns `u64` to avoid overflow when summing `u32` runs in
    /// large masks (`h * w` can exceed `u32::MAX` for a `u32`/`u32`
    /// shape).
    pub fn area(&self) -> u64 {
        self.counts
            .iter()
            .skip(1)
            .step_by(2)
            .map(|&c| c as u64)
            .sum()
    }

    /// Tight axis-aligned bounding box of the foreground in
    /// `[x, y, w, h]` integer-pixel form.
    ///
    /// Returns `[0, 0, 0, 0]` if the mask has no foreground (empty
    /// counts, all-background, or all foreground runs zero-length per
    /// quirk **G4**). The C version (`mc:149-178`) leaves its
    /// initial sentinels in place and underflows the width/height in
    /// this corner case; vernier returns an explicit empty bbox
    /// (disposition: `corrected` for safety).
    pub fn bbox(&self) -> [u32; 4] {
        if self.h == 0 || self.w == 0 {
            return [0; 4];
        }
        let h = self.h;
        let w = self.w;
        let h64 = h as u64;
        let mut xs = w;
        let mut ys = h;
        let mut xe: u32 = 0;
        let mut ye: u32 = 0;
        let mut found = false;
        let mut cc: u64 = 0;
        for (j, &len) in self.counts.iter().enumerate() {
            let start = cc;
            cc += len as u64;
            if j % 2 == 0 || len == 0 {
                continue;
            }
            let y_start = (start % h64) as u32;
            let x_start = (start / h64) as u32;
            let y_end = ((cc - 1) % h64) as u32;
            let x_end = ((cc - 1) / h64) as u32;

            xs = xs.min(x_start);
            xe = xe.max(x_end);
            if x_start < x_end {
                // Run spans columns: covers full y range in those cols.
                ys = 0;
                ye = h - 1;
            } else {
                // Same column: y_start <= y_end is guaranteed.
                ys = ys.min(y_start);
                ye = ye.max(y_end);
            }
            found = true;
        }
        if !found {
            return [0; 4];
        }
        [xs, ys, xe - xs + 1, ye - ys + 1]
    }

    /// Foreground intersection area of two RLEs sharing `(h, w)`.
    ///
    /// Equivalent to `Self::merge(&[self.clone(), other.clone()],
    /// true)?.area()` but skips the merged-counts allocation. Used by
    /// the segm-IoU kernel per pair after the bbox prefilter — the
    /// inner sweep mirrors `rleIou` (`mc:33-49`) without materializing
    /// the merged stream.
    ///
    /// Returns [`MaskError::DimensionMismatch`] if `(h, w)` disagree
    /// (quirk **I2** disposition `corrected`: pycocotools' `rleIou`
    /// silently writes a `-1` sentinel here).
    ///
    /// For repeated intersection against the same RLE — e.g. a single
    /// GT vs many DTs — prefer [`SegmentTable::push_from_rle`] +
    /// [`intersect_area_offsets`]: pre-decoding the foreground
    /// segments amortises one walk of `counts` over the whole pair
    /// loop.
    pub fn intersect_area(&self, other: &Rle) -> Result<u64, MaskError> {
        if self.h != other.h || self.w != other.w {
            return Err(MaskError::DimensionMismatch {
                expected: (self.h, self.w),
                got: (other.h, other.w),
            });
        }
        if self.h == 0 || self.w == 0 {
            return Ok(0);
        }
        let a = &self.counts;
        let b = &other.counts;
        if a.is_empty() || b.is_empty() {
            return Ok(0);
        }
        let mut ai = 1usize;
        let mut bi = 1usize;
        let mut ca = u64::from(a[0]);
        let mut cb = u64::from(b[0]);
        let mut va = false;
        let mut vb = false;
        let mut inter: u64 = 0;
        let mut ct: u64 = 1;
        while ct > 0 {
            let c = ca.min(cb);
            if va && vb {
                inter += c;
            }
            ct = 0;
            ca -= c;
            if ca == 0 && ai < a.len() {
                ca = u64::from(a[ai]);
                ai += 1;
                va = !va;
            }
            ct += ca;
            cb -= c;
            if cb == 0 && bi < b.len() {
                cb = u64::from(b[bi]);
                bi += 1;
                vb = !vb;
            }
            ct += cb;
        }
        Ok(inter)
    }

    /// Decodes this RLE's foreground runs into an alternating
    /// `[start_0, end_0, start_1, end_1, …]` flat slice of cumulative
    /// pixel offsets (column-major flat indices, in
    /// `[0, h * w as u64]`).
    ///
    /// Use with [`intersect_area_offsets`] to skip the per-pair RLE
    /// state-machine walk: the caller decodes once per annotation,
    /// then the inner pair loop is a two-pointer sweep over
    /// foreground segments only — typically 10×–100× fewer iterations
    /// than the byte-stream sweep, since most COCO RLE runs are
    /// background.
    ///
    /// Zero-length foreground runs (quirk **G4**) contribute nothing
    /// to the intersect and are dropped on decode. The output length
    /// is therefore even and equals `2 × <number of non-empty fg
    /// runs>`. Empty masks (no fg, or `h * w == 0`) decode to an
    /// empty slice.
    ///
    /// Reuses `out`'s capacity: `clear()` then `push`.
    pub fn decode_fg_offsets_into(&self, out: &mut Vec<u64>) {
        out.clear();
        if self.h == 0 || self.w == 0 {
            return;
        }
        let mut cum: u64 = 0;
        for (j, &c) in self.counts.iter().enumerate() {
            let len = u64::from(c);
            if j % 2 == 1 && len > 0 {
                out.push(cum);
                out.push(cum + len);
            }
            cum += len;
        }
    }

    /// Merges a slice of RLEs into one by intersection (`AND`) or
    /// union (`OR`).
    ///
    /// All inputs must share `(h, w)`; mismatch returns
    /// [`MaskError::DimensionMismatch`] (quirk **H2** disposition:
    /// `corrected` — pycocotools silently emits an empty `0x0` RLE
    /// in this case).
    ///
    /// An empty slice yields an empty `0x0` RLE, matching
    /// pycocotools' `rleMerge` for `n==0`. A singleton slice clones
    /// its only element.
    pub fn merge(rles: &[Rle], intersect: bool) -> Result<Rle, MaskError> {
        let Some(first) = rles.first() else {
            return Ok(Rle {
                h: 0,
                w: 0,
                counts: vec![],
            });
        };
        let (h, w) = (first.h, first.w);
        for r in &rles[1..] {
            if r.h != h || r.w != w {
                return Err(MaskError::DimensionMismatch {
                    expected: (h, w),
                    got: (r.h, r.w),
                });
            }
        }
        if rles.len() == 1 || h == 0 || w == 0 {
            return Ok(first.clone());
        }
        let mut acc = merge_pair(&first.counts, &rles[1].counts, intersect)?;
        for r in &rles[2..] {
            acc = merge_pair(&acc, &r.counts, intersect)?;
        }
        Ok(Rle { h, w, counts: acc })
    }
}

/// CSR-style table of foreground segments for a batch of RLEs.
///
/// Stores all `[start, end, …]` segment pairs in one contiguous
/// `flat` buffer, with `idx[i]..idx[i + 1]` slicing out the segments
/// for batch entry `i`. One `Vec<u64>` allocation amortises across
/// every RLE the table holds — the segm and boundary IoU kernels
/// reuse a single `SegmentTable` per `compute()` call instead of one
/// `Vec<u64>` per annotation.
///
/// Pair-wise intersection on stored segments is via
/// [`intersect_area_offsets`].
#[derive(Debug, Clone)]
pub struct SegmentTable {
    flat: Vec<u64>,
    idx: Vec<usize>,
}

impl Default for SegmentTable {
    fn default() -> Self {
        Self::new()
    }
}

impl SegmentTable {
    /// Empty table with zero rows. Equivalent to [`Self::default`].
    pub fn new() -> Self {
        Self {
            flat: Vec::new(),
            idx: vec![0],
        }
    }

    /// Drops all rows while preserving the buffer capacity.
    pub fn clear(&mut self) {
        self.flat.clear();
        self.idx.clear();
        self.idx.push(0);
    }

    /// Number of rows currently held.
    pub fn len(&self) -> usize {
        self.idx.len() - 1
    }

    /// Returns `true` if no rows are held.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Appends one row by copying an existing `[start, end, …]` slice.
    /// Panics in debug builds if `segments.len()` is odd.
    pub fn push_segments(&mut self, segments: &[u64]) {
        debug_assert!(segments.len() % 2 == 0);
        self.flat.extend_from_slice(segments);
        self.idx.push(self.flat.len());
    }

    /// Appends one row by decoding `rle`'s foreground runs straight
    /// into the flat buffer — same semantics as
    /// [`Rle::decode_fg_offsets_into`] but skips the intermediate
    /// `Vec<u64>` allocation that the per-call scratch would otherwise
    /// need.
    pub fn push_from_rle(&mut self, rle: &Rle) {
        if rle.h != 0 && rle.w != 0 {
            let mut cum: u64 = 0;
            for (j, &c) in rle.counts.iter().enumerate() {
                let len = u64::from(c);
                if j % 2 == 1 && len > 0 {
                    self.flat.push(cum);
                    self.flat.push(cum + len);
                }
                cum += len;
            }
        }
        self.idx.push(self.flat.len());
    }

    /// Appends one row by walking a column-major byte raster: fg-byte
    /// (`!= 0`, per quirk **G6**) transitions become `[start, end]`
    /// cumulative-offset pairs in the flat buffer. Returns the fg-byte
    /// count (== row area) so callers don't need a second pass over
    /// the raster.
    ///
    /// Equivalent to `Rle::from_raster_bytes(raster, h, w)` followed
    /// by `push_from_rle(&band)` plus `band.area()`, but in a single
    /// walk of the raster — no intermediate `Rle::counts` allocation.
    /// Hot-path callers (boundary-band derivation feeding the
    /// boundary-IoU kernel's segment tables) skip the round-trip
    /// entirely.
    pub fn push_from_raster(&mut self, raster: &[u8]) -> u64 {
        let mut area: u64 = 0;
        let mut run_start: u64 = 0;
        let mut in_run = false;
        for (i, &b) in raster.iter().enumerate() {
            let fg = b != 0;
            if fg && !in_run {
                run_start = i as u64;
                in_run = true;
            } else if !fg && in_run {
                let end = i as u64;
                self.flat.push(run_start);
                self.flat.push(end);
                area += end - run_start;
                in_run = false;
            }
        }
        if in_run {
            let end = raster.len() as u64;
            self.flat.push(run_start);
            self.flat.push(end);
            area += end - run_start;
        }
        self.idx.push(self.flat.len());
        area
    }

    /// Borrow row `i` as an alternating `[start, end, …]` slice
    /// suitable for [`intersect_area_offsets`].
    pub fn row(&self, i: usize) -> &[u64] {
        &self.flat[self.idx[i]..self.idx[i + 1]]
    }

    /// Borrow the most recently pushed row. Returns an empty slice
    /// for an empty table — the boundary-IoU cache fill calls this
    /// right after a push, so the empty case is unreachable in
    /// practice but we keep it total.
    pub fn last_row(&self) -> &[u64] {
        if self.is_empty() {
            return &[];
        }
        self.row(self.len() - 1)
    }
}

/// Foreground intersection area between two RLEs given as
/// pre-decoded segment offsets (see [`Rle::decode_fg_offsets_into`]).
///
/// `a` and `b` are alternating `[start, end, start, end, …]` flat
/// pixel offsets — even-length, segments sorted ascending and
/// non-overlapping inside each input. The two-pointer sweep visits
/// only foreground segments, skipping the background runs that
/// dominate `Rle::counts`.
///
/// The pair loop is the per-cell hot path for both the segm and
/// boundary-IoU kernels; per-annotation decode amortises one walk of
/// `counts` over `O(N_pairs)` cells.
///
/// Note: the function does not validate `(h, w)` — callers (segm /
/// boundary kernels) check shape compatibility once per `compute`.
pub fn intersect_area_offsets(a: &[u64], b: &[u64]) -> u64 {
    debug_assert!(a.len() % 2 == 0);
    debug_assert!(b.len() % 2 == 0);
    let (mut i, mut j) = (0usize, 0usize);
    let mut inter: u64 = 0;
    while i + 1 < a.len() && j + 1 < b.len() {
        let a0 = a[i];
        let a1 = a[i + 1];
        let b0 = b[j];
        let b1 = b[j + 1];
        let lo = a0.max(b0);
        let hi = a1.min(b1);
        if hi > lo {
            inter += hi - lo;
        }
        if a1 <= b1 {
            i += 2;
        } else {
            j += 2;
        }
    }
    inter
}

/// Two-pointer scan over a pair of run-length streams, producing the
/// merged run-length stream for `AND` or `OR` semantics. Mirrors the
/// inner loop of `rleMerge` in `mc:75-83`.
///
/// Internally widened to `u64` so accumulating output runs cannot
/// overflow during the sweep; final per-run lengths are checked back
/// down to `u32`.
fn merge_pair(a: &[u32], b: &[u32], intersect: bool) -> Result<Vec<u32>, MaskError> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let mut ai = 1usize;
    let mut bi = 1usize;
    let mut ca = a.first().copied().unwrap_or(0) as u64;
    let mut cb = b.first().copied().unwrap_or(0) as u64;
    let mut va = false;
    let mut vb = false;
    let mut v = false;
    let mut cc: u64 = 0;
    let mut ct: u64 = 1;
    while ct > 0 {
        let c = ca.min(cb);
        cc += c;
        ct = 0;
        ca -= c;
        if ca == 0 && ai < a.len() {
            ca = a[ai] as u64;
            ai += 1;
            va = !va;
        }
        ct += ca;
        cb -= c;
        if cb == 0 && bi < b.len() {
            cb = b[bi] as u64;
            bi += 1;
            vb = !vb;
        }
        ct += cb;
        let vp = v;
        v = if intersect { va && vb } else { va || vb };
        if v != vp || ct == 0 {
            let len = u32::try_from(cc)
                .map_err(|_| MaskError::MalformedRle(MalformedRleReason::U32Overflow))?;
            out.push(len);
            cc = 0;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn rle(h: u32, w: u32, counts: Vec<u32>) -> Rle {
        Rle { h, w, counts }
    }

    #[test]
    fn area_empty_is_zero() {
        assert_eq!(rle(0, 0, vec![]).area(), 0);
        assert_eq!(rle(2, 2, vec![]).area(), 0);
    }

    #[test]
    fn area_all_background_is_zero() {
        assert_eq!(rle(2, 2, vec![4]).area(), 0);
    }

    #[test]
    fn area_all_foreground_is_full() {
        assert_eq!(rle(2, 2, vec![0, 4]).area(), 4);
    }

    #[test]
    fn area_sums_odd_indexed_runs() {
        assert_eq!(rle(10, 10, vec![3, 2, 1, 4, 90]).area(), 6);
    }

    #[test]
    fn bbox_empty_mask() {
        assert_eq!(rle(0, 0, vec![]).bbox(), [0, 0, 0, 0]);
    }

    #[test]
    fn bbox_all_background() {
        assert_eq!(rle(2, 2, vec![4]).bbox(), [0, 0, 0, 0]);
    }

    #[test]
    fn bbox_all_zero_length_foreground() {
        // bg=4, fg=0 — the only foreground run has zero length (G4).
        // C version underflows; vernier returns the empty bbox.
        assert_eq!(rle(2, 2, vec![4, 0]).bbox(), [0, 0, 0, 0]);
    }

    #[test]
    fn bbox_full_image() {
        assert_eq!(rle(2, 3, vec![0, 6]).bbox(), [0, 0, 3, 2]);
    }

    #[test]
    fn bbox_single_pixel_column_major() {
        // 2x3 image, single fg pixel at flat idx 3: y=3%2=1, x=3/2=1.
        assert_eq!(rle(2, 3, vec![3, 1, 2]).bbox(), [1, 1, 1, 1]);
    }

    #[test]
    fn bbox_run_spans_columns() {
        // 2x3 image, fg from idx 1 to idx 4 inclusive (length 4).
        // x_start = 0, x_end = 2 → spans cols → y covers full 0..h-1.
        assert_eq!(rle(2, 3, vec![1, 4, 1]).bbox(), [0, 0, 3, 2]);
    }

    #[test]
    fn bbox_run_within_single_column() {
        // 4x3, fg from idx 5 to 6 (col 1, rows 1 and 2).
        let bb = rle(4, 3, vec![5, 2, 5]).bbox();
        assert_eq!(bb, [1, 1, 1, 2]);
    }

    #[test]
    fn merge_empty_slice_returns_empty_rle() {
        let m = Rle::merge(&[], false).unwrap();
        assert_eq!(m, rle(0, 0, vec![]));
    }

    #[test]
    fn merge_singleton_returns_clone() {
        let r = rle(2, 2, vec![1, 2, 1]);
        assert_eq!(Rle::merge(std::slice::from_ref(&r), false).unwrap(), r);
        assert_eq!(Rle::merge(std::slice::from_ref(&r), true).unwrap(), r);
    }

    #[test]
    fn merge_dimension_mismatch_errors() {
        let a = rle(2, 2, vec![4]);
        let b = rle(3, 3, vec![9]);
        let err = Rle::merge(&[a, b], false).unwrap_err();
        assert!(matches!(
            err,
            MaskError::DimensionMismatch {
                expected: (2, 2),
                got: (3, 3)
            }
        ));
    }

    #[test]
    fn merge_union_two_overlapping() {
        // A: 2x2 mask [1,0,0,0] = [0,1,3].
        // B: 2x2 mask [1,1,0,0] = [0,2,2].
        // Union [1,1,0,0] = [0,2,2].
        let a = rle(2, 2, vec![0, 1, 3]);
        let b = rle(2, 2, vec![0, 2, 2]);
        let u = Rle::merge(&[a, b], false).unwrap();
        assert_eq!(u, rle(2, 2, vec![0, 2, 2]));
    }

    #[test]
    fn merge_intersection_two_overlapping() {
        // Intersection of [1,0,0,0] and [1,1,0,0] = [1,0,0,0] = [0,1,3].
        let a = rle(2, 2, vec![0, 1, 3]);
        let b = rle(2, 2, vec![0, 2, 2]);
        let i = Rle::merge(&[a, b], true).unwrap();
        assert_eq!(i, rle(2, 2, vec![0, 1, 3]));
    }

    #[test]
    fn merge_disjoint_union() {
        // A: [1,0,0,0] = [0,1,3]. B: [0,0,0,1] = [3,1].
        // Union: [1,0,0,1] = [0,1,2,1].
        let a = rle(2, 2, vec![0, 1, 3]);
        let b = rle(2, 2, vec![3, 1]);
        let u = Rle::merge(&[a, b], false).unwrap();
        assert_eq!(u, rle(2, 2, vec![0, 1, 2, 1]));
    }

    #[test]
    fn merge_disjoint_intersection_is_empty_foreground() {
        let a = rle(2, 2, vec![0, 1, 3]);
        let b = rle(2, 2, vec![3, 1]);
        let i = Rle::merge(&[a, b], true).unwrap();
        // No overlap → all background. Sum must equal h*w=4.
        assert_eq!(i.counts.iter().map(|&c| c as u64).sum::<u64>(), 4);
        assert_eq!(i.area(), 0);
    }

    #[test]
    fn merge_three_way_union() {
        let a = rle(2, 2, vec![0, 1, 3]); // [1,0,0,0]
        let b = rle(2, 2, vec![1, 1, 2]); // [0,1,0,0]
        let c = rle(2, 2, vec![3, 1]); //   [0,0,0,1]
        let u = Rle::merge(&[a, b, c], false).unwrap();
        // Union = [1,1,0,1] = [0,2,1,1].
        assert_eq!(u, rle(2, 2, vec![0, 2, 1, 1]));
    }

    #[test]
    fn intersect_area_matches_merge_then_area_for_overlap() {
        let a = rle(2, 2, vec![0, 1, 3]);
        let b = rle(2, 2, vec![0, 2, 2]);
        let via_merge = Rle::merge(&[a.clone(), b.clone()], true).unwrap().area();
        let direct = a.intersect_area(&b).unwrap();
        assert_eq!(direct, via_merge);
        assert_eq!(direct, 1);
    }

    #[test]
    fn intersect_area_disjoint_is_zero() {
        let a = rle(2, 2, vec![0, 1, 3]);
        let b = rle(2, 2, vec![3, 1]);
        assert_eq!(a.intersect_area(&b).unwrap(), 0);
    }

    #[test]
    fn intersect_area_dimension_mismatch_errors() {
        let a = rle(2, 2, vec![4]);
        let b = rle(3, 3, vec![9]);
        let err = a.intersect_area(&b).unwrap_err();
        assert!(matches!(
            err,
            MaskError::DimensionMismatch {
                expected: (2, 2),
                got: (3, 3)
            }
        ));
    }

    #[test]
    fn decode_fg_offsets_drops_zero_length_runs() {
        // counts = [bg=2, fg=0, bg=2] — a zero-length fg run (G4).
        // Decoded segments must be empty.
        let r = rle(2, 2, vec![2, 0, 2]);
        let mut out = Vec::new();
        r.decode_fg_offsets_into(&mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn decode_fg_offsets_layout_is_alternating_start_end() {
        // 2x3 col-major image, fg at flat idx 1..=2 and 4..=4:
        // counts = [bg=1, fg=2, bg=1, fg=1, bg=1].
        let r = rle(2, 3, vec![1, 2, 1, 1, 1]);
        let mut out = Vec::new();
        r.decode_fg_offsets_into(&mut out);
        assert_eq!(out, vec![1, 3, 4, 5]);
    }

    #[test]
    fn intersect_area_offsets_disjoint_segments_is_zero() {
        // a = [0,2], b = [3,5] — no overlap.
        assert_eq!(intersect_area_offsets(&[0, 2], &[3, 5]), 0);
    }

    #[test]
    fn intersect_area_offsets_nested_segment() {
        // a = [0,10], b = [3,5] — b is inside a.
        assert_eq!(intersect_area_offsets(&[0, 10], &[3, 5]), 2);
    }

    #[test]
    fn intersect_area_offsets_multiple_segments() {
        // a = [0,2, 5,7], b = [1,6] — overlap = (1..2) ∪ (5..6) = 2.
        assert_eq!(intersect_area_offsets(&[0, 2, 5, 7], &[1, 6]), 2);
    }

    #[test]
    fn intersect_area_offsets_empty_inputs() {
        assert_eq!(intersect_area_offsets(&[], &[0, 4]), 0);
        assert_eq!(intersect_area_offsets(&[0, 4], &[]), 0);
        assert_eq!(intersect_area_offsets(&[], &[]), 0);
    }

    #[test]
    fn intersect_area_zero_shape_or_empty_counts_is_zero() {
        assert_eq!(
            rle(0, 0, vec![])
                .intersect_area(&rle(0, 0, vec![]))
                .unwrap(),
            0
        );
        assert_eq!(
            rle(2, 2, vec![])
                .intersect_area(&rle(2, 2, vec![0, 4]))
                .unwrap(),
            0
        );
    }

    #[test]
    fn segment_table_push_from_rle_matches_decode_fg_offsets_into() {
        let r1 = rle(2, 2, vec![0, 1, 3]);
        let r2 = rle(2, 2, vec![0, 4]);
        let mut table = SegmentTable::new();
        table.push_from_rle(&r1);
        table.push_from_rle(&r2);
        let mut decoded = Vec::new();
        r1.decode_fg_offsets_into(&mut decoded);
        assert_eq!(table.row(0), decoded.as_slice());
        r2.decode_fg_offsets_into(&mut decoded);
        assert_eq!(table.row(1), decoded.as_slice());
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn segment_table_clear_resets_rows_but_keeps_capacity() {
        let mut table = SegmentTable::new();
        table.push_from_rle(&rle(2, 2, vec![0, 4]));
        let cap_before = table.flat.capacity();
        table.clear();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert!(table.flat.capacity() >= cap_before);
    }

    #[test]
    fn segment_table_push_segments_borrows_existing_slice() {
        let segs: [u64; 4] = [0, 2, 5, 7];
        let mut table = SegmentTable::new();
        table.push_segments(&segs);
        assert_eq!(table.row(0), &segs);
    }

    #[test]
    fn segment_table_push_from_raster_empty_input() {
        let mut table = SegmentTable::new();
        let area = table.push_from_raster(&[]);
        assert_eq!(area, 0);
        assert_eq!(table.row(0), &[] as &[u64]);
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn segment_table_push_from_raster_all_background() {
        let mut table = SegmentTable::new();
        let area = table.push_from_raster(&[0u8; 8]);
        assert_eq!(area, 0);
        assert!(table.row(0).is_empty());
    }

    #[test]
    fn segment_table_push_from_raster_all_foreground() {
        let mut table = SegmentTable::new();
        let area = table.push_from_raster(&[1u8; 8]);
        assert_eq!(area, 8);
        assert_eq!(table.row(0), &[0u64, 8]);
    }

    #[test]
    fn segment_table_push_from_raster_leading_and_trailing_fg() {
        // [1,1,0,1,1,1,0,1] → segments [0,2), [3,6), [7,8); area 6.
        let mut table = SegmentTable::new();
        let area = table.push_from_raster(&[1, 1, 0, 1, 1, 1, 0, 1]);
        assert_eq!(area, 6);
        assert_eq!(table.row(0), &[0u64, 2, 3, 6, 7, 8]);
    }

    #[test]
    fn segment_table_push_from_raster_binarises_per_g6() {
        // Non-zero bytes are foreground (G6 strict).
        let mut table = SegmentTable::new();
        let area = table.push_from_raster(&[0, 2, 255, 0]);
        assert_eq!(area, 2);
        assert_eq!(table.row(0), &[1u64, 3]);
    }

    proptest! {
        #[test]
        fn intersect_area_matches_merge_pair(
            a_bytes in raster_strategy(4, 4),
            b_bytes in raster_strategy(4, 4),
        ) {
            let ra = Rle::from_raster_bytes(&a_bytes, 4, 4)?;
            let rb = Rle::from_raster_bytes(&b_bytes, 4, 4)?;
            let direct = ra.intersect_area(&rb)?;
            let via_merge = Rle::merge(&[ra, rb], true)?.area();
            prop_assert_eq!(direct, via_merge);
        }

        #[test]
        fn intersect_area_offsets_matches_intersect_area(
            a_bytes in raster_strategy(4, 4),
            b_bytes in raster_strategy(4, 4),
        ) {
            let ra = Rle::from_raster_bytes(&a_bytes, 4, 4)?;
            let rb = Rle::from_raster_bytes(&b_bytes, 4, 4)?;
            let scalar = ra.intersect_area(&rb)?;
            let mut a_off = Vec::new();
            let mut b_off = Vec::new();
            ra.decode_fg_offsets_into(&mut a_off);
            rb.decode_fg_offsets_into(&mut b_off);
            let via_offsets = intersect_area_offsets(&a_off, &b_off);
            prop_assert_eq!(via_offsets, scalar);
        }

        #[test]
        fn push_from_raster_matches_from_rle_then_push_from_rle(
            bytes in raster_strategy(4, 5),
        ) {
            let r = Rle::from_raster_bytes(&bytes, 4, 5)?;
            let mut expected = SegmentTable::new();
            expected.push_from_rle(&r);
            let expected_area = r.area();

            let mut actual = SegmentTable::new();
            let actual_area = actual.push_from_raster(&bytes);

            prop_assert_eq!(actual_area, expected_area);
            prop_assert_eq!(actual.row(0), expected.row(0));
        }

        #[test]
        fn push_from_raster_binarises_per_g6(
            bytes in proptest::collection::vec(any::<u8>(), 0..120),
        ) {
            let binarised: Vec<u8> = bytes.iter().map(|&b| u8::from(b != 0)).collect();
            let mut from_raw = SegmentTable::new();
            let from_raw_area = from_raw.push_from_raster(&bytes);
            let mut from_bin = SegmentTable::new();
            let from_bin_area = from_bin.push_from_raster(&binarised);
            prop_assert_eq!(from_raw_area, from_bin_area);
            prop_assert_eq!(from_raw.row(0), from_bin.row(0));
        }
    }

    fn raster_strategy(h: u32, w: u32) -> impl Strategy<Value = Vec<u8>> {
        let total = (h as usize) * (w as usize);
        proptest::collection::vec(0u8..=1, total..=total)
    }

    proptest! {
        #[test]
        fn merge_inclusion_exclusion(
            ba in raster_strategy(4, 5),
            bb in raster_strategy(4, 5),
        ) {
            let a = Rle::from_raster_bytes(&ba, 4, 5)?;
            let b = Rle::from_raster_bytes(&bb, 4, 5)?;
            let u = Rle::merge(&[a.clone(), b.clone()], false)?;
            let i = Rle::merge(&[a.clone(), b.clone()], true)?;
            prop_assert_eq!(u.area() + i.area(), a.area() + b.area());
            prop_assert!(u.area() >= a.area().max(b.area()));
            prop_assert!(i.area() <= a.area().min(b.area()));
        }

        #[test]
        fn merge_union_matches_or(
            ba in raster_strategy(4, 5),
            bb in raster_strategy(4, 5),
        ) {
            let a = Rle::from_raster_bytes(&ba, 4, 5)?;
            let b = Rle::from_raster_bytes(&bb, 4, 5)?;
            let u = Rle::merge(&[a, b], false)?;
            let expected: Vec<u8> = ba.iter().zip(&bb).map(|(x, y)| x | y).collect();
            prop_assert_eq!(u.to_raster_bytes(), expected);
        }

        #[test]
        fn merge_intersect_matches_and(
            ba in raster_strategy(4, 5),
            bb in raster_strategy(4, 5),
        ) {
            let a = Rle::from_raster_bytes(&ba, 4, 5)?;
            let b = Rle::from_raster_bytes(&bb, 4, 5)?;
            let i = Rle::merge(&[a, b], true)?;
            let expected: Vec<u8> = ba.iter().zip(&bb).map(|(x, y)| x & y).collect();
            prop_assert_eq!(i.to_raster_bytes(), expected);
        }
    }
}
