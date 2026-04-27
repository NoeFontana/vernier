//! RLE-on-RLE operations: foreground area, tight bounding box, and
//! N-way merge (intersection / union).
//!
//! Mirrors `rleArea` (`mc:88-91`), `rleToBbox` (`mc:149-178`), and
//! `rleMerge` (`mc:65-86`) from `pycocotools-2.0.11/common/maskApi.c`.
//!
//! Per pycocotools' column-major (Fortran) convention, a flat pixel
//! index `idx` decomposes as `y = idx % h`, `x = idx / h`.

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
        // rles.len() >= 2 by this point.
        let mut acc = merge_pair(&first.counts, &rles[1].counts, intersect)?;
        for r in &rles[2..] {
            acc = merge_pair(&acc, &r.counts, intersect)?;
        }
        Ok(Rle { h, w, counts: acc })
    }
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
