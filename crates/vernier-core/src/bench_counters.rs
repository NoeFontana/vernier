//! Process-global atomic-counter sets for bench / test instrumentation.
//!
//! Six sites across `vernier-core` and `vernier-ffi` previously hand-
//! rolled `static AtomicU64::new(0)` + `fetch_add(_, Relaxed)` +
//! `swap(0, Relaxed)` boilerplate behind their own feature gate
//! (`bench-timings` for perf-investigation counters, `_test-counter`
//! for ADR-0046 call counters). This module collapses the shared
//! shape into one parameterized type so adding a counter is a single
//! line per call site instead of a static + an inc fn + a no-op stub.
//!
//! The shipped wheel never compiles any of the bench / test features,
//! so the unification cost is zero at runtime: every counter site
//! stays inside its existing `#[cfg]` gate; this module just gives
//! them a shared inner type.

use std::sync::atomic::{AtomicU64, Ordering};

/// `N` `AtomicU64` counters with consistent reset semantics. Use
/// `BenchCounterSet::new()` as a `static` initializer; access via
/// `bump(idx)` / `add(idx, value)` / `load(idx)` / `read_and_reset()`.
///
/// The fixed-`N` array layout means each counter sits in a separate
/// `AtomicU64` (no false sharing risk under contention only if N is
/// small — the counters share a cache line at N ≤ 8, which is the
/// case at every existing site). Per-call `Relaxed` ordering matches
/// every site we replaced.
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
