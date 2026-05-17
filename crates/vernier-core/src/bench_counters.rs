//! Process-global atomic-counter sets for bench / test instrumentation.
//! Callers stay inside their own `#[cfg]` gate (`bench-timings` /
//! `_test-counter`); this module is the shared inner type.

use std::sync::atomic::{AtomicU64, Ordering};

/// `N` `AtomicU64` counters with `Relaxed` ordering. Suitable as a
/// `static` initializer via [`Self::new`].
pub struct BenchCounterSet<const N: usize> {
    counters: [AtomicU64; N],
}

impl<const N: usize> BenchCounterSet<N> {
    /// Zero-initialized counter set, suitable as a `static` initializer.
    pub const fn new() -> Self {
        Self {
            counters: [const { AtomicU64::new(0) }; N],
        }
    }

    /// Increment counter `idx` by 1.
    #[inline]
    pub fn bump(&self, idx: usize) {
        self.counters[idx].fetch_add(1, Ordering::Relaxed);
    }

    /// Add `value` to counter `idx`. Used for accumulating nanosecond
    /// timings.
    #[inline]
    pub fn add(&self, idx: usize, value: u64) {
        self.counters[idx].fetch_add(value, Ordering::Relaxed);
    }

    /// Read counter `idx` without resetting.
    #[inline]
    pub fn load(&self, idx: usize) -> u64 {
        self.counters[idx].load(Ordering::Relaxed)
    }

    /// Read every counter and reset it to zero. Returns the values in
    /// index order.
    pub fn read_and_reset(&self) -> [u64; N] {
        std::array::from_fn(|i| self.counters[i].swap(0, Ordering::Relaxed))
    }
}

impl<const N: usize> Default for BenchCounterSet<N> {
    fn default() -> Self {
        Self::new()
    }
}
