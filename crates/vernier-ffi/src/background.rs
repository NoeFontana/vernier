//! Background evaluator (ADR-0014). Single dedicated worker thread +
//! bounded `mpsc::sync_channel` that wraps [`StreamingEvaluator`]. The
//! worker owns the evaluator → satisfies the single-writer rule by
//! construction.
//!
//! This module is pure orchestration: it composes around the
//! `vernier-core` streaming surface and exposes a thread-safe
//! `submit` / `snapshot` / `finalize` triple. The PyO3 surface that
//! wraps this lives in `lib.rs` (Phase F of the ADR-0014 rollout).
//!
//! ## Single-writer guarantee
//!
//! `StreamingEvaluator::update_parsed` takes `&mut self`. We enforce
//! that at runtime by handing the evaluator to the worker thread on
//! spawn and never moving it out — every write goes through the
//! [`std::sync::mpsc::sync_channel`]. The wrapper itself is `Sync`
//! because every public method only touches the sender, the
//! `Arc<BackgroundState>`, or the worker `JoinHandle` mutex.
//!
//! ## Backpressure
//!
//! The channel is bounded by `BackgroundConfig::queue_capacity` (default
//! 8). [`BackgroundEvaluator::submit_blocking`] blocks until a slot is
//! free; [`BackgroundEvaluator::submit_timeout`] either tries once
//! (`Duration::ZERO`) or waits up to `timeout`, returning [`QueueFull`]
//! if the worker doesn't drain in time.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use vernier_core::evaluate::EvalKernel;
use vernier_core::stream::{ParsedDetections, SnapshotWithCells, StreamingEvaluator};
use vernier_core::tables::{Tables, TablesConfig, TablesRequest};
use vernier_core::{EvalError, Summary};

/// Configuration knobs for the background worker.
///
/// Defaults target a training-loop persona: small bounded queue (so
/// excess submits surface backpressure quickly), no pinned affinity,
/// nice +5 (best-effort lower priority than the trainer), and a 5s
/// shutdown grace period.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BackgroundConfig {
    /// Capacity of the submit channel. Submits beyond this depth block
    /// (or raise `QueueFull` with a non-`None` timeout).
    pub(crate) queue_capacity: usize,
    /// Optional CPU core to pin the worker to. `None` leaves it
    /// schedulable on any core.
    pub(crate) worker_affinity: Option<usize>,
    /// POSIX-style nice value: higher → lower priority. Clamped to
    /// `[-20, 19]`. Best-effort; failures are reported via
    /// `BackgroundState::scheduling_outcome`.
    pub(crate) worker_nice: i32,
    /// Maximum time `shutdown()` waits for the worker to exit cleanly
    /// before giving up.
    pub(crate) shutdown_timeout: Duration,
    /// ADR-0047 Stage B: inner parallelism inside the worker. `None`
    /// is exactly today's behavior — one worker, sequential
    /// `update_parsed`, no rayon pool built. `Some(n)` builds a scoped
    /// `rayon::ThreadPool` of `n` threads at worker spawn, owned by
    /// the worker thread, and routes each `update_parsed` through
    /// `pool.install(...)`. The pool drops when the worker exits.
    pub(crate) num_threads: Option<std::num::NonZeroUsize>,
}

impl Default for BackgroundConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 8,
            worker_affinity: None,
            worker_nice: 5,
            shutdown_timeout: Duration::from_secs(5),
            num_threads: None,
        }
    }
}

/// Wire protocol between the FFI surface and the background worker.
///
/// Per ADR-0035, `BackgroundEvaluator`'s public surface is
/// submit / finalize / finalize_with_tables / finalize_to_partial.
/// ADR-0018 Unit 6 adds `finalize_with_cells` for the calibration
/// summarizer. The worker handles those five request shapes plus
/// cooperative shutdown; mid-flight snapshot variants were removed
/// alongside the public method.
pub(crate) enum WorkerMessage<K: EvalKernel + Send + 'static> {
    /// Hand a parsed detection batch to the worker. The worker either
    /// applies it or stashes the resulting [`EvalError`] in
    /// `BackgroundState::last_error` for the next FFI entry to surface.
    Update(ParsedDetections<K>),
    /// Drain and finalize. The worker replies and exits the loop.
    Finalize {
        reply: SyncSender<Result<Summary, EvalError>>,
    },
    /// Finalize variant that also builds the requested result tables.
    /// Worker exits after sending the reply.
    FinalizeWithTables {
        reply: SyncSender<Result<(Summary, Tables), EvalError>>,
        request: TablesRequest,
        config: TablesConfig,
    },
    /// ADR-0018 Unit 6: finalize variant that also hands back the
    /// per-image cell store needed by the calibration summarizer. The
    /// summary axis is bit-identical to [`Self::Finalize`]; the cells
    /// payload mirrors the canonical
    /// [`StreamingEvaluator::snapshot_with_cells`] return shape. Worker
    /// exits after sending the reply.
    FinalizeWithCells {
        reply: SyncSender<Result<SnapshotWithCells, EvalError>>,
    },
    /// ADR-0031: drain and serialize as a partial blob. Worker exits
    /// after sending the reply.
    FinalizeToPartial {
        reply: SyncSender<Result<Vec<u8>, EvalError>>,
    },
    /// Cooperative shutdown — the worker exits without finalizing.
    Shutdown,
    /// Test-only: panic the worker so the FFI panic-recovery path can
    /// be exercised. Gated behind the `test-poison` Cargo feature.
    #[cfg(feature = "test-poison")]
    Poison,
}

/// Atomics + cells the FFI side reads without touching the worker.
///
/// All counters are `Release`-stored by the worker after a successful
/// update and `Acquire`-loaded by the FFI accessors; no fenced
/// snapshotting is attempted (counters are advisory, not load-bearing).
pub(crate) struct BackgroundState {
    /// Mirrors `StreamingEvaluator::images_seen()` after each successful
    /// update. Updated only on the worker thread.
    pub(crate) images_seen: AtomicUsize,
    /// Mirrors `StreamingEvaluator::detections_seen()`.
    pub(crate) detections_seen: AtomicUsize,
    /// Number of `Update` messages currently sitting in the channel
    /// (incremented by the FFI just before the send, decremented by the
    /// worker on receive). Approximate — useful for backpressure
    /// telemetry, not for control flow.
    pub(crate) queue_depth: AtomicUsize,
    /// Mirrors `StreamingEvaluator::memory_used_bytes()`.
    pub(crate) memory_used_bytes: AtomicUsize,
    /// Worker stashes recoverable update errors here (e.g. budget
    /// breaches). The FFI surface checks this on every entry, raises if
    /// `Some`, and clears it.
    pub(crate) last_error: Mutex<Option<EvalError>>,
    /// Result of the worker's best-effort scheduling adjustment
    /// (nice + affinity). The FFI reads this once after `spawn` and
    /// emits a single `UserWarning` on `Err`.
    pub(crate) scheduling_outcome: Mutex<Option<Result<(), String>>>,
}

impl BackgroundState {
    fn new() -> Self {
        Self {
            images_seen: AtomicUsize::new(0),
            detections_seen: AtomicUsize::new(0),
            queue_depth: AtomicUsize::new(0),
            memory_used_bytes: AtomicUsize::new(0),
            last_error: Mutex::new(None),
            scheduling_outcome: Mutex::new(None),
        }
    }
}

/// Thread-safe wrapper around a [`StreamingEvaluator`] running on a
/// dedicated worker thread.
///
/// Construct via [`Self::spawn`]. `submit_*` posts a parsed batch;
/// `snapshot` / `finalize` block on a reply channel. Public accessor
/// methods read counters from the shared [`BackgroundState`] without
/// going through the channel.
pub(crate) struct BackgroundEvaluator<K: EvalKernel + Send + 'static> {
    sender: SyncSender<WorkerMessage<K>>,
    /// Hold under `Mutex` so `finalize` and `shutdown` (both `self`-by-
    /// value) can take the handle out of an `Arc<Mutex<...>>` from the
    /// FFI side without `unsafe`.
    worker: Mutex<Option<JoinHandle<Result<(), EvalError>>>>,
    config: BackgroundConfig,
    pub(crate) state: Arc<BackgroundState>,
    /// Opt-in `submit_blocking` / `submit_timeout` latency sample
    /// accumulator (B5 — BackgroundEvaluator p99 latency cell).
    /// `None` when disabled (the default); the only always-on cost is
    /// one `Option::is_some` check per submit. Each sample is the
    /// wall-time of the FFI-to-`mpsc::send` leg in nanoseconds.
    latency_samples: Option<Mutex<Vec<u64>>>,
}

/// Returned by [`BackgroundEvaluator::submit_timeout`] when the queue
/// stays full for the requested duration. Carries the configured
/// capacity and the timeout that elapsed, so the FFI can attach both
/// to the `QueueFullError` exception.
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueFull {
    pub(crate) queue_capacity: usize,
    pub(crate) timeout: Duration,
}

/// Internal error type for [`BackgroundEvaluator::submit_timeout`].
/// Disambiguates the two failure modes (queue exhaustion vs. anything
/// else, mainly worker-died). The FFI matches on this to map to the
/// right Python exception.
pub(crate) enum SubmitError {
    /// Wraps a stashed or in-flight [`EvalError`] (worker died, or a
    /// previous update left an error to surface).
    Eval(EvalError),
    /// The queue stayed full for the requested duration.
    Full(QueueFull),
}

impl From<EvalError> for SubmitError {
    fn from(e: EvalError) -> Self {
        Self::Eval(e)
    }
}

impl<K: EvalKernel + Send + 'static> BackgroundEvaluator<K> {
    /// Hand the evaluator to a fresh worker thread named
    /// `vernier-bg-worker` and return the wrapper. Latency-sample
    /// accumulation is off by default; see
    /// [`Self::spawn_with_options`] to opt in.
    ///
    /// The worker applies its scheduling preferences (nice, affinity)
    /// before pulling its first message; the result is stashed in
    /// `state.scheduling_outcome` for the FFI to read once.
    #[allow(
        dead_code,
        reason = "kept for symmetry with BackgroundCore::spawn; FFI calls spawn_with_options directly"
    )]
    pub(crate) fn spawn(
        evaluator: StreamingEvaluator<K>,
        config: BackgroundConfig,
    ) -> Result<Self, EvalError> {
        Self::spawn_with_options(evaluator, config, false)
    }

    /// Variant of [`Self::spawn`] with a per-instance opt-in for
    /// latency-sample accumulation (B5 — BackgroundEvaluator p99
    /// latency cell). When `record_latency_samples` is `true`, every
    /// successful `submit_blocking` / `submit_timeout` records the
    /// wall-time of the channel-send leg in nanoseconds; readers
    /// consume samples via [`Self::latency_samples_drain`]. When
    /// `false`, the accumulator stays unallocated and the only cost
    /// is one `Option::is_some` check per submit.
    pub(crate) fn spawn_with_options(
        evaluator: StreamingEvaluator<K>,
        config: BackgroundConfig,
        record_latency_samples: bool,
    ) -> Result<Self, EvalError> {
        let (sender, receiver) = sync_channel::<WorkerMessage<K>>(config.queue_capacity);
        let state = Arc::new(BackgroundState::new());
        // Seed the visible counters so accessors don't lie before the
        // first update lands. `memory_used_bytes` is 0 on a fresh
        // evaluator; same for images / detections.
        state
            .memory_used_bytes
            .store(evaluator.memory_used_bytes(), Ordering::Release);

        let worker_state = Arc::clone(&state);
        let worker_config = config;
        let handle = thread::Builder::new()
            .name("vernier-bg-worker".to_string())
            .spawn(move || worker_loop::<K>(evaluator, receiver, worker_state, worker_config))
            .map_err(|e| EvalError::InvalidConfig {
                detail: format!("failed to spawn background worker: {e}"),
            })?;

        let latency_samples = if record_latency_samples {
            Some(Mutex::new(Vec::new()))
        } else {
            None
        };

        Ok(Self {
            sender,
            worker: Mutex::new(Some(handle)),
            config,
            state: Arc::clone(&state),
            latency_samples,
        })
    }

    /// Block until the channel has space, then post the batch. Returns
    /// the stashed `last_error` if the worker reported one between this
    /// and the previous call (and clears it).
    pub(crate) fn submit_blocking(&self, parsed: ParsedDetections<K>) -> Result<(), EvalError> {
        self.take_last_error()?;
        let start = self.latency_start();
        self.state.queue_depth.fetch_add(1, Ordering::AcqRel);
        match self.sender.send(WorkerMessage::Update(parsed)) {
            Ok(()) => {
                self.record_latency_sample(start);
                Ok(())
            }
            Err(_) => {
                // Worker is gone; roll back the advisory queue counter
                // and report the disconnect to the caller.
                self.state.queue_depth.fetch_sub(1, Ordering::AcqRel);
                Err(EvalError::InvalidConfig {
                    detail: "background worker is no longer accepting submissions".to_string(),
                })
            }
        }
    }

    /// Try to post a batch with a bounded wait. `Duration::ZERO`
    /// performs a single non-blocking attempt; otherwise waits up to
    /// `timeout`. Block-indefinitely is `submit_blocking`.
    pub(crate) fn submit_timeout(
        &self,
        parsed: ParsedDetections<K>,
        timeout: Duration,
    ) -> Result<(), SubmitError> {
        self.take_last_error().map_err(SubmitError::Eval)?;
        let start = self.latency_start();
        self.state.queue_depth.fetch_add(1, Ordering::AcqRel);
        let result = self.send_with_timeout(WorkerMessage::Update(parsed), timeout);
        match &result {
            Ok(()) => self.record_latency_sample(start),
            Err(_) => {
                self.state.queue_depth.fetch_sub(1, Ordering::AcqRel);
            }
        }
        result
    }

    /// Bounded `try_send` loop. `SyncSender::send_timeout` is unstable
    /// on stable Rust (`std_internals`), so we poll. The poll cadence
    /// (`POLL_INTERVAL`) is short enough that submit latency stays
    /// dominated by the worker, not this loop, but long enough that
    /// busy-waiting doesn't burn a core.
    fn send_with_timeout(
        &self,
        msg: WorkerMessage<K>,
        timeout: Duration,
    ) -> Result<(), SubmitError> {
        const POLL_INTERVAL: Duration = Duration::from_millis(1);
        let deadline = Instant::now().checked_add(timeout);
        let mut payload = msg;
        loop {
            match self.sender.try_send(payload) {
                Ok(()) => return Ok(()),
                Err(TrySendError::Full(returned)) => {
                    if timeout.is_zero() {
                        return Err(SubmitError::Full(QueueFull {
                            queue_capacity: self.config.queue_capacity,
                            timeout,
                        }));
                    }
                    let now = Instant::now();
                    let past_deadline = match deadline {
                        Some(d) => now >= d,
                        // Overflow: the requested timeout is essentially
                        // infinite — just keep polling.
                        None => false,
                    };
                    if past_deadline {
                        return Err(SubmitError::Full(QueueFull {
                            queue_capacity: self.config.queue_capacity,
                            timeout,
                        }));
                    }
                    payload = returned;
                    let remaining = match deadline {
                        Some(d) => d.saturating_duration_since(now),
                        None => POLL_INTERVAL,
                    };
                    thread::sleep(POLL_INTERVAL.min(remaining));
                }
                Err(TrySendError::Disconnected(_)) => {
                    return Err(SubmitError::Eval(EvalError::InvalidConfig {
                        detail: "background worker is no longer accepting submissions".to_string(),
                    }));
                }
            }
        }
    }

    /// Send a finalize-shaped `WorkerMessage` through the channel,
    /// wait for the typed reply, then join the worker thread.
    /// Consumes `self`. The worker prefers its own error over the
    /// finalize reply on a clean panic-free exit.
    fn dispatch_finalize<R, F>(self, build_msg: F) -> Result<R, EvalError>
    where
        R: Send + 'static,
        F: FnOnce(SyncSender<Result<R, EvalError>>) -> WorkerMessage<K>,
    {
        self.take_last_error()?;
        let (reply_tx, reply_rx) = sync_channel::<Result<R, EvalError>>(1);
        self.sender
            .send(build_msg(reply_tx))
            .map_err(|_| EvalError::InvalidConfig {
                detail: "background worker is no longer accepting finalize requests".to_string(),
            })?;
        let result = match reply_rx.recv() {
            Ok(r) => r,
            Err(_) => Err(EvalError::InvalidConfig {
                detail: "background worker dropped finalize reply channel".to_string(),
            }),
        };
        let join_result = match self.take_worker() {
            Some(handle) => handle.join(),
            None => return result,
        };
        match join_result {
            Ok(Ok(())) => result,
            Ok(Err(worker_err)) => Err(worker_err),
            Err(payload) => Err(EvalError::InvalidConfig {
                detail: format!("background worker panicked: {payload:?}"),
            }),
        }
    }

    /// ADR-0031: drain the queue, serialize the evaluator's final
    /// state as a partial blob, and join the worker. Consumes `self`.
    pub(crate) fn finalize_to_partial(self) -> Result<Vec<u8>, EvalError> {
        self.dispatch_finalize(|reply| WorkerMessage::FinalizeToPartial { reply })
    }

    /// Drain the queue, finalize the evaluator, and join the worker.
    ///
    /// Consumes `self` — the wrapper is unusable afterwards. The FFI
    /// layer holds a `Mutex<Option<BackgroundEvaluator<K>>>` so it can
    /// `take()` to satisfy the `self`-by-value signature.
    pub(crate) fn finalize(self) -> Result<Summary, EvalError> {
        self.dispatch_finalize(|reply| WorkerMessage::Finalize { reply })
    }

    /// ADR-0018 Unit 6: finalize variant that also returns the
    /// per-image cell store needed by the calibration summarizer. The
    /// returned [`SnapshotWithCells::summary`] is bit-identical to what
    /// [`Self::finalize`] would have produced for the same evaluator
    /// state — this variant only adds cell retention.
    pub(crate) fn finalize_with_cells(self) -> Result<SnapshotWithCells, EvalError> {
        self.dispatch_finalize(|reply| WorkerMessage::FinalizeWithCells { reply })
    }

    /// Tables-aware finalize.
    pub(crate) fn finalize_with_tables(
        self,
        request: TablesRequest,
        config: TablesConfig,
    ) -> Result<(Summary, Tables), EvalError> {
        self.dispatch_finalize(|reply| WorkerMessage::FinalizeWithTables {
            reply,
            request,
            config,
        })
    }

    /// Best-effort cooperative shutdown. Sends `Shutdown`, then polls
    /// `JoinHandle::is_finished` up to `config.shutdown_timeout`. Any
    /// worker output is discarded — this is the path used by `__exit__`
    /// and `__del__` on the Python side.
    pub(crate) fn shutdown(self) {
        // Best-effort: ignore the error if the worker is already gone.
        let _ = self.sender.send(WorkerMessage::Shutdown);
        let handle = match self.take_worker() {
            Some(h) => h,
            None => return,
        };
        let deadline = Instant::now() + self.config.shutdown_timeout;
        // `JoinHandle` has no native timeout-join; poll `is_finished`.
        loop {
            if handle.is_finished() {
                let _ = handle.join();
                return;
            }
            if Instant::now() >= deadline {
                // Drop the handle without joining; the worker will
                // outlive this wrapper, but the channel is closed so
                // it will exit on its next `recv`.
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    /// Mirror of `StreamingEvaluator::images_seen()`. Approximate
    /// (advisory) — updated post-success by the worker.
    pub(crate) fn images_seen(&self) -> usize {
        self.state.images_seen.load(Ordering::Acquire)
    }

    /// Mirror of `StreamingEvaluator::detections_seen()`. Approximate.
    pub(crate) fn detections_seen(&self) -> usize {
        self.state.detections_seen.load(Ordering::Acquire)
    }

    /// Approximate count of `Update` messages waiting in the channel.
    pub(crate) fn queue_depth(&self) -> usize {
        self.state.queue_depth.load(Ordering::Acquire)
    }

    /// Mirror of `StreamingEvaluator::memory_used_bytes()`. Approximate.
    pub(crate) fn memory_used_bytes(&self) -> usize {
        self.state.memory_used_bytes.load(Ordering::Acquire)
    }

    /// Take the result of the worker's startup scheduling adjustment
    /// (best-effort nice/affinity). FFI calls this once after spawn;
    /// subsequent calls return `None`.
    pub(crate) fn take_scheduling_outcome(&self) -> Option<Result<(), String>> {
        match self.state.scheduling_outcome.lock() {
            Ok(mut guard) => guard.take(),
            Err(_) => None,
        }
    }

    /// Test-only: post a `Poison` message that panics the worker.
    /// Exists only with the `test-poison` Cargo feature; the FFI exposes
    /// a hidden `_inject_poison_for_tests` accessor on top of this.
    #[cfg(feature = "test-poison")]
    pub(crate) fn _inject_poison_for_tests(&self) -> Result<(), EvalError> {
        self.sender
            .send(WorkerMessage::Poison)
            .map_err(|_| EvalError::InvalidConfig {
                detail: "background worker is no longer accepting submissions".to_string(),
            })
    }

    /// Drain and replace the per-instance submit-latency sample buffer
    /// (B5). Each sample is the wall-time of the FFI-to-`mpsc::send`
    /// leg in nanoseconds; the inner `Vec` is left freshly empty so
    /// subsequent submits keep accumulating. Returns an empty `Vec`
    /// when the wrapper was constructed without
    /// `record_latency_samples`.
    pub(crate) fn latency_samples_drain(&self) -> Vec<u64> {
        match self.latency_samples.as_ref() {
            Some(slot) => match slot.lock() {
                Ok(mut guard) => std::mem::take(&mut *guard),
                Err(_) => Vec::new(),
            },
            None => Vec::new(),
        }
    }

    /// `Some(Instant::now())` when latency capture is on; `None`
    /// otherwise. Inlined into the submit hot path.
    fn latency_start(&self) -> Option<Instant> {
        if self.latency_samples.is_some() {
            Some(Instant::now())
        } else {
            None
        }
    }

    /// Push the elapsed nanoseconds since `start` if both the
    /// accumulator is enabled and `start` was captured. Mutex
    /// contention is bounded — held only to push a single `u64`.
    fn record_latency_sample(&self, start: Option<Instant>) {
        let (slot, started) = match (self.latency_samples.as_ref(), start) {
            (Some(slot), Some(t0)) => (slot, t0),
            _ => return,
        };
        // `Duration::as_nanos` is `u128`; saturate on overflow rather
        // than wrap so a runaway tail stays observable instead of
        // looking artificially fast.
        let ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        if let Ok(mut guard) = slot.lock() {
            guard.push(ns);
        }
    }

    /// Drain `state.last_error` and surface it. The contract: errors
    /// land near the calling code, not in the worker, so we read-and-
    /// clear on every public entry.
    fn take_last_error(&self) -> Result<(), EvalError> {
        match self.state.last_error.lock() {
            Ok(mut guard) => match guard.take() {
                Some(e) => Err(e),
                None => Ok(()),
            },
            Err(_) => Err(EvalError::InvalidConfig {
                detail: "background evaluator's error slot is poisoned".to_string(),
            }),
        }
    }

    /// Lift the worker handle out of the mutex. Used by `finalize` and
    /// `shutdown` (both `self`-by-value); never returns `Some` more
    /// than once.
    fn take_worker(&self) -> Option<JoinHandle<Result<(), EvalError>>> {
        match self.worker.lock() {
            Ok(mut guard) => guard.take(),
            Err(_) => None,
        }
    }
}

/// The worker loop. Owns the [`StreamingEvaluator`] outright; every
/// write goes through `&mut evaluator`, which is the structural enforcer
/// of the single-writer rule.
///
/// On `WorkerMessage::Update`, recoverable [`EvalError`]s are stashed in
/// `state.last_error` so the next FFI entry surfaces them — the loop
/// stays alive (a budget breach should not poison the channel). On
/// `WorkerMessage::Finalize` the worker replies and returns; on
/// `Shutdown` or a closed channel it returns immediately.
//
// `rx` and `state` are intentionally passed by value: this function is
// the entry point of a spawned thread, so the values must be owned here
// (the spawning thread cannot keep them alive on this thread's behalf).
#[allow(clippy::needless_pass_by_value)]
fn worker_loop<K: EvalKernel + Send + Sync + 'static>(
    mut evaluator: StreamingEvaluator<K>,
    rx: Receiver<WorkerMessage<K>>,
    state: Arc<BackgroundState>,
    config: BackgroundConfig,
) -> Result<(), EvalError> {
    let outcome = crate::thread_sched::apply_scheduling(config.worker_nice, config.worker_affinity);
    if let Ok(mut guard) = state.scheduling_outcome.lock() {
        *guard = Some(outcome);
    }

    // ADR-0047 Stage B: build the per-worker `rayon::ThreadPool` once,
    // here, owned by this thread. The pool drops naturally when this
    // function returns. On non-Linux platforms the pool threads do not
    // inherit `worker_nice` (Linux process-attribute semantics do not
    // apply); the `start_handler` re-applies it explicitly. Affinity is
    // intentionally **not** propagated — pinning all N pool threads to
    // one CPU defeats the purpose (ADR-0047 §"`BackgroundEvaluator`
    // shape").
    let pool: Option<rayon::ThreadPool> = match config.num_threads {
        None => None,
        Some(n) => {
            let nice = config.worker_nice;
            let result = rayon::ThreadPoolBuilder::new()
                .num_threads(n.get())
                .thread_name(|i| format!("vernier-bg-rayon-{i}"))
                .start_handler(move |_idx| {
                    // Best-effort; failures here are unobservable but
                    // not load-bearing — they only matter on platforms
                    // where Linux's `setpriority` process-wide nice
                    // doesn't carry across to spawned threads.
                    let _ = crate::thread_sched::set_thread_nice(nice);
                })
                .build();
            match result {
                Ok(p) => Some(p),
                Err(e) => {
                    // Best-effort: stash and continue sequentially.
                    // Mirrors how `apply_scheduling` failures are
                    // surfaced — the worker stays alive and the FFI
                    // emits a single `UserWarning` next time it reads
                    // the scheduling outcome.
                    if let Ok(mut guard) = state.scheduling_outcome.lock() {
                        // Overwrite only if the prior outcome was Ok;
                        // a previous Err is more informative.
                        let prior_ok = matches!(*guard, Some(Ok(())));
                        if guard.is_none() || prior_ok {
                            *guard = Some(Err(format!(
                                "failed to build rayon pool of {} threads: {e}",
                                n.get()
                            )));
                        }
                    }
                    None
                }
            }
        }
    };

    loop {
        match rx.recv() {
            Ok(WorkerMessage::Update(parsed)) => {
                state.queue_depth.fetch_sub(1, Ordering::AcqRel);
                let result = match pool.as_ref() {
                    Some(p) => p.install(|| evaluator.update_parsed_parallel(parsed)),
                    None => evaluator.update_parsed(parsed),
                };
                match result {
                    Ok(_report) => {
                        state
                            .images_seen
                            .store(evaluator.images_seen(), Ordering::Release);
                        state
                            .detections_seen
                            .store(evaluator.detections_seen(), Ordering::Release);
                        state
                            .memory_used_bytes
                            .store(evaluator.memory_used_bytes(), Ordering::Release);
                    }
                    Err(e) => {
                        if let Ok(mut guard) = state.last_error.lock() {
                            *guard = Some(e);
                        }
                        // Recoverable — keep the loop alive so a fresh
                        // submit can land after the FFI surfaces and
                        // clears the error.
                    }
                }
            }
            Ok(WorkerMessage::Finalize { reply }) => {
                // `finalize()` consumes the evaluator. The local binding
                // owns it outright, so the move out of the loop body is
                // unconditional — reaching this arm always exits the loop.
                let s = evaluator.finalize();
                let _ = reply.send(s);
                return Ok(());
            }
            Ok(WorkerMessage::FinalizeWithTables {
                reply,
                request,
                config,
            }) => {
                let s = evaluator.finalize_with_tables(request, &config);
                let _ = reply.send(s);
                return Ok(());
            }
            Ok(WorkerMessage::FinalizeWithCells { reply }) => {
                let s = evaluator.snapshot_with_cells();
                let _ = reply.send(s);
                return Ok(());
            }
            Ok(WorkerMessage::FinalizeToPartial { reply }) => {
                let p = evaluator.finalize_to_partial();
                let _ = reply.send(p);
                return Ok(());
            }
            Ok(WorkerMessage::Shutdown) | Err(RecvError) => return Ok(()),
            #[cfg(feature = "test-poison")]
            Ok(WorkerMessage::Poison) => {
                // Test-only path. The panic propagates to
                // `JoinHandle::join` so the FFI can recover and surface
                // a `RuntimeError`. Allowed clippy::panic only here.
                #[allow(clippy::panic)]
                {
                    panic!("test-only worker panic");
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    //! End-to-end smoke tests for the FFI background worker (the
    //! `StreamingEvaluator<K>`-flavored sibling of
    //! `crate::background_streaming`). Today this covers the ADR-0018
    //! Unit 6 `finalize_with_cells` round-trip — the only worker
    //! message the generic core doesn't already exercise.
    use super::*;
    use vernier_core::dataset::{AnnId, Bbox, CategoryId, ImageId};
    use vernier_core::dataset::{CategoryMeta, CocoAnnotation, CocoDataset, ImageMeta};
    use vernier_core::evaluate::{AreaRange, OwnedEvaluateParams};
    use vernier_core::parity::iou_thresholds;
    use vernier_core::similarity::BboxIou;
    use vernier_core::stream::{MemoryBudget, ParsedDetections, StreamingEvaluator};
    use vernier_core::ParityMode;

    fn tiny_dataset() -> CocoDataset {
        let images = vec![ImageMeta {
            id: ImageId(1),
            width: 100,
            height: 100,
            file_name: None,
        }];
        let cats = vec![CategoryMeta {
            id: CategoryId(1),
            name: "thing".into(),
            supercategory: None,
        }];
        let anns = vec![CocoAnnotation {
            id: AnnId(1),
            image_id: ImageId(1),
            category_id: CategoryId(1),
            area: 100.0,
            is_crowd: false,
            ignore_flag: None,
            bbox: Bbox {
                x: 0.0,
                y: 0.0,
                w: 10.0,
                h: 10.0,
            },
            segmentation: None,
            keypoints: None,
            num_keypoints: None,
        }];
        CocoDataset::from_parts(images, anns, cats).unwrap()
    }

    fn default_params() -> OwnedEvaluateParams {
        OwnedEvaluateParams {
            iou_thresholds: iou_thresholds().to_vec(),
            area_ranges: AreaRange::coco_default().to_vec(),
            max_dets_per_image: 100,
            use_cats: true,
            retain_iou: false,
        }
    }

    fn spawn_test_worker() -> BackgroundEvaluator<BboxIou> {
        let ev = StreamingEvaluator::new(
            tiny_dataset(),
            BboxIou,
            default_params(),
            ParityMode::Strict,
            MemoryBudget::auto_default(),
        )
        .unwrap();
        BackgroundEvaluator::spawn(ev, BackgroundConfig::default()).unwrap()
    }

    /// ADR-0018 Unit 6 smoke test: update → finalize_with_cells round-
    #[test]
    fn finalize_with_cells_round_trips_through_worker() {
        let bg = spawn_test_worker();
        let parsed = ParsedDetections::<BboxIou>::from_json_bytes(
            br#"[{"image_id": 1, "category_id": 1, "score": 0.9, "bbox": [0, 0, 10, 10]}]"#,
        )
        .unwrap();
        bg.submit_blocking(parsed).unwrap();

        let bundle = bg.finalize_with_cells().unwrap();
        assert_eq!(bundle.summary.lines.len(), 12);
        assert_eq!(bundle.n_categories, 1);
        assert_eq!(bundle.n_area_ranges, AreaRange::coco_default().len());
        assert!(!bundle.iou_thresholds.is_empty());
        assert!(matches!(bundle.parity_mode, ParityMode::Strict));
        assert!(bundle.eval_imgs.iter().any(|c| c.is_some()));
    }

    #[test]
    fn finalize_with_cells_empty_evaluator_still_returns_dense_store() {
        let bg = spawn_test_worker();
        let bundle = bg.finalize_with_cells().unwrap();
        assert_eq!(bundle.summary.lines.len(), 12);
        // 1 category × 4 area ranges × 1 image.
        assert_eq!(bundle.eval_imgs.len(), 4);
    }
}
