//! Run scanning over label maps.
//!
//! A panoptic label map is piecewise constant: in row-major order a segment
//! covers runs of tens to hundreds of pixels. The per-pixel folds over it
//! (segment areas, the GT × DT intersection matrix) used to bump a counter
//! per pixel — and consecutive pixels bump the *same* counter, so each
//! increment waited on the store of the previous one. These iterators yield
//! maximal runs instead, so a fold does one lookup and one add per run.
//!
//! Runs are found 64 pixels at a time. A block that continues the current
//! run whole — most of a smooth map — is recognized by one constant compare.
//! Any other block gets a boundary mask, built eight lanes at a time (bit `k`
//! where pixel `k` differs from its left neighbour), and its runs are read
//! off the set bits with `trailing_zeros`: a noisy map (salt-and-pepper
//! predictions) pays one bit step per run, with no data-dependent branch
//! per pixel.
//!
//! They are iterators, not callbacks, so a fold's own state (lookup caches)
//! stays in registers.

/// Pixels per boundary mask.
const BLOCK: usize = 64;

/// Bit `k` set where `cur[k] != prev[k]`, for eight lanes: a fixed-size
/// compare-and-pack that lowers to a vector compare and a movemask.
#[inline]
fn lanes_differ(cur: &[u32; 8], prev: &[u32; 8]) -> u8 {
    let mut m = 0u8;
    for k in 0..8 {
        m |= u8::from(cur[k] != prev[k]) << k;
    }
    m
}

/// Bit `k` set where `values[k] != values[k - 1]`, for `k` in `1..len`
/// (`values` holds the block's pixels preceded by one more: `len + 1`
/// values). Whole groups of eight go through [`lanes_differ`].
#[inline]
fn boundary_mask(values: &[u32]) -> u64 {
    let len = values.len() - 1;
    let mut mask = 0u64;
    let mut k = 0;
    while k + 8 <= len {
        let cur: &[u32; 8] = values[k + 1..k + 9].try_into().unwrap_or(&[0; 8]);
        let prev: &[u32; 8] = values[k..k + 8].try_into().unwrap_or(&[0; 8]);
        mask |= u64::from(lanes_differ(cur, prev)) << k;
        k += 8;
    }
    while k < len {
        mask |= u64::from(values[k + 1] != values[k]) << k;
        k += 1;
    }
    mask
}

/// Maximal runs of equal values in `ids`, in order: `(value, len)`.
pub(crate) fn runs(ids: &[u32]) -> Runs<'_> {
    Runs {
        ids,
        scan: Scan::default(),
    }
}

/// See [`runs`].
pub(crate) struct Runs<'a> {
    ids: &'a [u32],
    scan: Scan,
}

impl Iterator for Runs<'_> {
    type Item = (u32, usize);

    #[inline]
    fn next(&mut self) -> Option<(u32, usize)> {
        let ids = self.ids;
        let (start, len) = self.scan.next(ids.len(), |run, from, to| {
            let v = ids[run];
            if ids[from + 1..to].iter().fold(0, |acc, &x| acc | (x ^ v)) == 0 {
                0
            } else {
                boundary_mask(&ids[from..to])
            }
        })?;
        Some((ids[start], len))
    }
}

/// Maximal runs over which both `a` and `b` are constant, over the common
/// prefix of the two slices: `(a_value, b_value, len)`.
pub(crate) fn pair_runs<'a>(a: &'a [u32], b: &'a [u32]) -> PairRuns<'a> {
    let n = a.len().min(b.len());
    PairRuns {
        a: &a[..n],
        b: &b[..n],
        scan: Scan::default(),
    }
}

/// See [`pair_runs`].
pub(crate) struct PairRuns<'a> {
    a: &'a [u32],
    b: &'a [u32],
    scan: Scan,
}

impl Iterator for PairRuns<'_> {
    type Item = (u32, u32, usize);

    #[inline]
    fn next(&mut self) -> Option<(u32, u32, usize)> {
        let (a, b) = (self.a, self.b);
        let (start, len) = self.scan.next(a.len(), |run, from, to| {
            let (u, v) = (a[run], b[run]);
            let uniform = a[from + 1..to]
                .iter()
                .zip(&b[from + 1..to])
                .fold(0, |acc, (&x, &y)| acc | (x ^ u) | (y ^ v))
                == 0;
            if uniform {
                0
            } else {
                boundary_mask(&a[from..to]) | boundary_mask(&b[from..to])
            }
        })?;
        Some((a[start], b[start], len))
    }
}

/// Run-boundary scanner shared by [`Runs`] and [`PairRuns`].
#[derive(Default)]
struct Scan {
    /// Where the current run starts.
    start: usize,
    /// The block `mask` covers compares pixels `base + 1 ..= base + BLOCK`
    /// with their left neighbours.
    base: usize,
    /// Unread boundaries in the current block.
    mask: u64,
    /// Whether `mask` has been computed for `base`.
    loaded: bool,
}

impl Scan {
    /// The next run as `(start, len)`, over `n` pixels. `block(run, from,
    /// to)` returns the boundary mask of `values[from..to]` relative to
    /// `from` (see [`boundary_mask`]): bit `k` marks pixel `from + 1 + k` as
    /// the start of a run. `run` is where the current run starts, so a block
    /// that continues it whole — most of a smooth map — can answer `0` from
    /// one constant compare instead of building the mask.
    #[inline]
    fn next(
        &mut self,
        n: usize,
        block: impl Fn(usize, usize, usize) -> u64,
    ) -> Option<(usize, usize)> {
        loop {
            if self.mask != 0 {
                let end = self.base + 1 + self.mask.trailing_zeros() as usize;
                self.mask &= self.mask - 1;
                let start = std::mem::replace(&mut self.start, end);
                return Some((start, end - start));
            }
            if self.loaded {
                self.base += BLOCK;
            }
            // Block `base` compares pixels `base + 1 ..= base + BLOCK` with
            // their left neighbours; pixel 0 starts the first run.
            if self.base + 1 >= n {
                if self.start < n {
                    let start = std::mem::replace(&mut self.start, n);
                    return Some((start, n - start));
                }
                return None;
            }
            let to = (self.base + BLOCK + 1).min(n);
            self.mask = block(self.start, self.base, to);
            self.loaded = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Values in runs of 1..100, so runs cross block boundaries both ways.
    fn run_pairs() -> impl Strategy<Value = Vec<(u32, u32)>> {
        prop::collection::vec(((0u32..3, 0u32..3), 1usize..100), 0..30).prop_map(|rs| {
            rs.into_iter()
                .flat_map(|(p, len)| std::iter::repeat_n(p, len))
                .collect()
        })
    }

    proptest! {
        /// Runs tile the input exactly and are maximal.
        #[test]
        fn runs_tile_the_input(pairs in run_pairs()) {
            let ids: Vec<u32> = pairs.iter().map(|p| p.0).collect();
            let mut rebuilt = Vec::new();
            let mut prev = None;
            for (v, len) in runs(&ids) {
                assert!(len > 0);
                assert_ne!(prev, Some(v), "adjacent runs share a value");
                prev = Some(v);
                rebuilt.extend(std::iter::repeat_n(v, len));
            }
            prop_assert_eq!(rebuilt, ids);
        }

        #[test]
        fn pair_runs_tile_the_input(
            pairs in run_pairs(),
            extra in prop::collection::vec(0u32..3, 0..5),
        ) {
            let a: Vec<u32> = pairs.iter().map(|p| p.0).collect();
            let mut b: Vec<u32> = pairs.iter().map(|p| p.1).collect();
            b.extend(extra); // longer `b`: only the common prefix counts
            let mut rebuilt = Vec::new();
            let mut prev = None;
            for (u, v, len) in pair_runs(&a, &b) {
                assert!(len > 0);
                assert_ne!(prev, Some((u, v)), "adjacent runs share a pair");
                prev = Some((u, v));
                rebuilt.extend(std::iter::repeat_n((u, v), len));
            }
            prop_assert_eq!(rebuilt, pairs);
        }
    }

    #[test]
    fn a_change_in_b_alone_ends_the_span_mid_chunk() {
        let a = vec![7u32; 24];
        let mut b = vec![1u32; 24];
        b[13] = 2;
        let spans: Vec<_> = pair_runs(&a, &b).collect();
        let total: usize = spans.iter().map(|s| s.2).sum();
        assert_eq!(total, 24);
        assert!(spans.iter().all(|&(_, v, _)| v == 1 || v == 2));
        assert_eq!(
            spans
                .iter()
                .filter(|s| s.1 == 2)
                .map(|s| s.2)
                .sum::<usize>(),
            1
        );
    }

    #[test]
    fn a_long_run_crosses_blocks_whole() {
        let ids = vec![5u32; 1 + 64 * 3];
        assert_eq!(runs(&ids).collect::<Vec<_>>(), vec![(5, 193)]);
        assert_eq!(runs(&[]).count(), 0);
    }
}
