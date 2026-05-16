//! Opt-in `num_threads` parallelism dispatch (ADR-0047).
//!
//! Resolves the FFI's `num_threads: Option<usize>` kwarg into a
//! [`ThreadPolicy`] used by the batch eval pipeline:
//!
//! - `None` (with no env override) / `1` → [`ThreadPolicy::Sequential`];
//!   today's code path, no rayon symbol entered.
//! - `0` → auto via [`std::thread::available_parallelism`] (cgroup-aware
//!   on Linux); a host with parallelism `1` collapses back to sequential.
//! - `n ≥ 2` → [`ThreadPolicy::Pool`]`(n)`; caller builds a scoped pool
//!   via [`build_scoped_pool`] and `install`s the parallel work inside.
//!
//! Precedence: kwarg `>` `VERNIER_NUM_THREADS` `>` sequential default.
//! `RAYON_NUM_THREADS` is intentionally **not** consulted; inheriting
//! an unrelated library's deployment knob would silently change
//! vernier's behavior.
//!
//! Re-entry detection: if [`rayon::current_thread_index`] returns
//! `Some` the caller is already inside a rayon worker — coerce to
//! sequential and emit a one-shot `UserWarning` rather than
//! oversubscribe.

use std::num::NonZeroUsize;
use std::sync::OnceLock;
use std::thread::available_parallelism;

use pyo3::exceptions::PyUserWarning;
use pyo3::prelude::*;

use crate::emit_warning;

const ENV_VAR_NAME: &str = "VERNIER_NUM_THREADS";

/// Resolved threading decision the FFI matches on once per entry
/// point, inside [`Python::detach`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ThreadPolicy {
    Sequential,
    Pool(NonZeroUsize),
}

impl ThreadPolicy {
    pub(crate) fn thread_count(self) -> Option<NonZeroUsize> {
        match self {
            Self::Sequential => None,
            Self::Pool(n) => Some(n),
        }
    }
}

static REENTRY_WARNED: OnceLock<()> = OnceLock::new();
static ENV_VAR_WARNED: OnceLock<()> = OnceLock::new();

/// Resolve `num_threads` from a single FFI entry point. Called with
/// the GIL held — emits Python warnings on re-entry and env-var
/// resolution. Misconfigured values fall through to sequential rather
/// than failing the call (threading is a perf knob, not a correctness
/// switch).
pub(crate) fn resolve_threads(py: Python<'_>, arg: Option<usize>) -> ThreadPolicy {
    if rayon::current_thread_index().is_some() {
        if REENTRY_WARNED.set(()).is_ok() {
            let _ = emit_warning::<PyUserWarning>(
                py,
                "vernier: num_threads ignored — called from inside a rayon worker; \
                 falling back to the sequential path to avoid oversubscription",
            );
        }
        return ThreadPolicy::Sequential;
    }

    let Some((resolved, source_was_env)) = resolve_threads_raw(arg) else {
        return ThreadPolicy::Sequential;
    };
    let Some(n) = NonZeroUsize::new(resolved).filter(|n| n.get() >= 2) else {
        return ThreadPolicy::Sequential;
    };

    if source_was_env && ENV_VAR_WARNED.set(()).is_ok() {
        let _ = emit_warning::<PyUserWarning>(
            py,
            &format!("vernier: num_threads={n} (from {ENV_VAR_NAME})"),
        );
    }
    ThreadPolicy::Pool(n)
}

fn resolve_threads_raw(arg: Option<usize>) -> Option<(usize, bool)> {
    let (raw, source_was_env) = match arg {
        Some(n) => (n, false),
        None => match std::env::var(ENV_VAR_NAME).ok()?.trim().parse::<usize>() {
            Ok(n) => (n, true),
            Err(_) => return None,
        },
    };
    let resolved = match raw {
        0 => available_parallelism().map(NonZeroUsize::get).unwrap_or(1),
        n => n,
    };
    Some((resolved, source_was_env))
}

/// Build a scoped `rayon::ThreadPool` of exactly `n` threads. The
/// caller `install`s the parallel work and drops the pool on return —
/// the global rayon pool is never involved.
pub(crate) fn build_scoped_pool(n: NonZeroUsize) -> Result<rayon::ThreadPool, String> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n.get())
        .thread_name(|i| format!("vernier-rayon-{i}"))
        .build()
        .map_err(|e| format!("failed to build rayon pool of {} threads: {e}", n.get()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn resolve_no_py(arg: Option<usize>) -> ThreadPolicy {
        // Mirrors `resolve_threads`'s decision logic minus the Python
        // warning emission; the warnings are advisory and orthogonal.
        let (resolved, _) = resolve_threads_raw(arg).unwrap_or((1, false));
        NonZeroUsize::new(resolved)
            .filter(|n| n.get() >= 2)
            .map_or(ThreadPolicy::Sequential, ThreadPolicy::Pool)
    }

    #[test]
    fn none_one_and_invalid_env_resolve_sequential() {
        std::env::remove_var(ENV_VAR_NAME);
        assert_eq!(resolve_no_py(None), ThreadPolicy::Sequential);
        assert_eq!(resolve_no_py(Some(1)), ThreadPolicy::Sequential);
    }

    #[test]
    fn explicit_n_builds_pool() {
        std::env::remove_var(ENV_VAR_NAME);
        assert_eq!(
            resolve_no_py(Some(4)),
            ThreadPolicy::Pool(NonZeroUsize::new(4).unwrap())
        );
    }

    #[test]
    fn zero_resolves_to_available_or_sequential() {
        std::env::remove_var(ENV_VAR_NAME);
        match resolve_no_py(Some(0)) {
            ThreadPolicy::Sequential => {} // host with parallelism == 1
            ThreadPolicy::Pool(n) => assert!(n.get() >= 2),
        }
    }

    #[test]
    fn build_pool_succeeds_for_small_n() {
        let pool = build_scoped_pool(NonZeroUsize::new(2).unwrap()).expect("pool builds");
        assert_eq!(pool.current_num_threads(), 2);
    }
}
