//! Column-major pixel indexing shared by the RLE walks.

/// Column and row of a column-major pixel index, `(idx / h, idx % h)`.
///
/// Every RLE walk needs both for each run it crosses, and a 64-bit
/// division costs more than the rest of the run's work. For an index below
/// 2^32 — every index of every real image — the column is a multiply-high
/// instead (Lemire, Kaser & Kurz 2019, *Faster remainder by direct
/// computation*): with `M = ⌊(2⁶⁴ − 1) / h⌋ + 1`, `idx / h = ⌊M·idx / 2⁶⁴⌋`,
/// exact for every 32-bit `idx` and `h ≥ 2`; the row is then
/// `idx − col·h`. A larger index — which a malformed mask whose runs
/// overshoot `h·w` can reach — and `h = 1` keep the division, so the
/// result is exact for any input. Both branches predict.
#[derive(Clone, Copy)]
pub(crate) struct ColumnMajor {
    h: u64,
    m: u64,
    multiply: bool,
}

impl ColumnMajor {
    pub(crate) fn new(h: u32) -> Self {
        let h = u64::from(h);
        Self {
            h,
            m: (u64::MAX / h.max(1)).wrapping_add(1),
            multiply: h >= 2,
        }
    }

    /// `(idx / h, idx % h)`. `h` must be non-zero.
    #[inline]
    pub(crate) fn col_row(self, idx: u64) -> (u64, u64) {
        if self.multiply && idx <= u64::from(u32::MAX) {
            let col = ((u128::from(self.m) * u128::from(idx)) >> 64) as u64;
            (col, idx - col * self.h)
        } else {
            (idx / self.h, idx % self.h)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ColumnMajor` agrees with `/` and `%` for every height, at both ends
    /// of the 32-bit range and past it (a malformed mask's overshoot).
    #[test]
    fn column_major_matches_division() {
        let heights = (1..=600_u32).chain([
            1023,
            1024,
            4095,
            65_535,
            65_536,
            1 << 20,
            (1 << 31) + 1,
            u32::MAX,
        ]);
        for h in heights {
            let grid = ColumnMajor::new(h);
            let h64 = u64::from(h);
            let probes = [
                0,
                1,
                h64 - 1,
                h64,
                h64 + 1,
                1 << 31,
                u64::from(u32::MAX) - 1,
                u64::from(u32::MAX),
            ];
            let past = [1 << 32, 10_737_418_243, 10_737_418_244, u64::MAX / 3];
            let spread = (0..64_u64).map(|i| u64::from(u32::MAX) / 64 * i);
            for idx in probes.into_iter().chain(past).chain(spread) {
                assert_eq!(grid.col_row(idx), (idx / h64, idx % h64), "h={h} idx={idx}");
            }
        }
    }
}
