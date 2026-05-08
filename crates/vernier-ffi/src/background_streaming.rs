//! Paradigm-generic background evaluator core (ADR-0014 + ADR-0032).
//!
//! Sibling to [`crate::background`] which wraps the instance
//! `StreamingEvaluator<K: EvalKernel>`. That module carries instance-
//! specific concerns — `ParsedDetections<K>` parsing, tables-aware
//! snapshot/finalize variants, `snapshot_running` — that don't
//! generalize cleanly across paradigms. This module is the leaner
//! generic for paradigms whose streaming surface is just
//! `apply_update / snapshot / finalize / *_to_partial`: today,
//! semantic and panoptic.
//!
//! ## Single-writer guarantee
//!
//! [`BackgroundCore::spawn`] hands the evaluator to a worker thread on
//! construction and never moves it out. Every write goes through
//! [`std::sync::mpsc::sync_channel`]; the wrapper itself is `Sync`
//! because every public method only touches the sender, the
//! `Arc<BackgroundState>`, or the worker `JoinHandle` mutex.
//!
//! ## Backpressure
//!
//! Same shape as [`crate::background`]: bounded channel of
//! `BackgroundConfig::queue_capacity`; [`BackgroundCore::submit_blocking`]
//! waits indefinitely; [`BackgroundCore::submit_timeout`] either tries
//! once (`Duration::ZERO`) or waits up to `timeout`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::background::{BackgroundConfig, QueueFull};

/// Paradigm-side seam: a streaming evaluator that can run on a
/// background worker. The implementor owns its own state shape; this
/// trait names only the operations the worker dispatches and the two
/// constructors `BackgroundCore` needs to surface "the worker is
/// gone" / "already finalized" as the paradigm's typed error.
///
/// `Update` is the per-image payload sent over the channel — already-
/// parsed off the GIL so the worker thread does no Python work.
/// `Summary` is the value returned from `snapshot` / `finalize`.
/// `Error` is the paradigm's typed error; the FFI shim maps it to a
/// `PyErr` at the call site (each paradigm has its own
/// `*_error_to_pyerr` helper).
pub(crate) trait BackgroundCapable: Send + 'static {
    /// Per-image payload sent over the channel. Owned data —
    /// constructed on the FFI thread before the GIL is dropped.
    type Update: Send + 'static;
    /// Snapshot / finalize return value.
    type Summary: Send + 'static;
    /// Paradigm-typed error.
    type Error: Send + 'static;

    /// Apply one update to the evaluator. Recoverable errors are
    /// stashed in the wrapper's `last_error` slot; the worker stays
    /// alive so a fresh submit can land after the FFI surfaces the
    /// error.
    fn apply_update(&mut self, update: Self::Update) -> Result<(), Self::Error>;

    /// Snapshot the current state. Non-consuming.
    fn snapshot(&self) -> Result<Self::Summary, Self::Error>;

    /// Drain and finalize. Consumes the evaluator.
    fn finalize(self) -> Result<Self::Summary, Self::Error>;

    /// Serialize the current state as an opaque partial blob
    /// (ADR-0032). Non-consuming.
    fn snapshot_to_partial(&self) -> Result<Vec<u8>, Self::Error>;

    /// Consuming variant of [`Self::snapshot_to_partial`].
    fn finalize_to_partial(self) -> Result<Vec<u8>, Self::Error>;

    /// Mirrored into [`BackgroundState::images_seen`] after each
    /// successful update so the FFI accessor reads it without
    /// crossing the channel.
    fn images_seen(&self) -> usize;

    /// Mirrored into [`BackgroundState::memory_used_bytes`] after each
    /// successful update. Default 0 for paradigms with no per-image
    /// memory growth (semantic's confusion matrix is a fixed
    /// `n_classes × n_classes` allocation).
    fn memory_used_bytes(&self) -> usize {
        0
    }

    /// Construct the typed error returned when the worker channel is
    /// closed (worker died, or `submit` raced with `shutdown`).
    /// Returning a paradigm-typed error keeps the Python `kind`
    /// attribute meaningful — see [`vernier_partial::PartialFormatErrorKind::Internal`]
    /// for the wire-format tag the FFI surfaces.
    fn worker_disconnected() -> Self::Error;

    /// Construct the typed error returned when an op runs after
    /// `finalize` / `__exit__` has consumed the worker.
    fn already_finalized() -> Self::Error;
}

/// Wire protocol between the FFI surface and the worker. Generic over
/// the implementor's `Update`, `Summary`, and `Error` types.
enum WorkerMessage<E: BackgroundCapable> {
    Update(E::Update),
    Snapshot {
        reply: SyncSender<Result<E::Summary, E::Error>>,
    },
    Finalize {
        reply: SyncSender<Result<E::Summary, E::Error>>,
    },
    SnapshotToPartial {
        reply: SyncSender<Result<Vec<u8>, E::Error>>,
    },
    FinalizeToPartial {
        reply: SyncSender<Result<Vec<u8>, E::Error>>,
    },
    Shutdown,
    /// Test-only: panic the worker so the FFI panic-recovery path can
    /// be exercised. Gated behind the `test-poison` Cargo feature.
    #[cfg(feature = "test-poison")]
    Poison,
}

/// Shared atomics + cells. The FFI accessors read these without
/// touching the worker; the worker `Release`-stores after each
/// successful update so reads are causally consistent with the most
/// recent observed state.
pub(crate) struct BackgroundState<E: BackgroundCapable> {
    images_seen: AtomicUsize,
    queue_depth: AtomicUsize,
    memory_used_bytes: AtomicUsize,
    /// Stash for recoverable update errors. Surfaced and cleared on
    /// the next FFI entry.
    last_error: Mutex<Option<E::Error>>,
    /// Worker's startup scheduling result; the FFI reads it once and
    /// emits a single `UserWarning` on `Err`.
    scheduling_outcome: Mutex<Option<Result<(), String>>>,
}

impl<E: BackgroundCapable> BackgroundState<E> {
    fn new() -> Self {
        Self {
            images_seen: AtomicUsize::new(0),
            queue_depth: AtomicUsize::new(0),
            memory_used_bytes: AtomicUsize::new(0),
            last_error: Mutex::new(None),
            scheduling_outcome: Mutex::new(None),
        }
    }
}

/// Internal error type for [`BackgroundCore::submit_timeout`].
/// Disambiguates queue exhaustion (`Full`) from "anything else"
/// (`Eval`, mainly worker-died or a stashed update error).
pub(crate) enum SubmitError<Err> {
    Eval(Err),
    Full(QueueFull),
}

impl<Err> From<Err> for SubmitError<Err> {
    fn from(e: Err) -> Self {
        Self::Eval(e)
    }
}

/// Generic background-evaluator wrapper. Owns the channel sender,
/// the worker `JoinHandle`, and the shared `BackgroundState`. `E` is
/// fully constrained by `sender` (`SyncSender<WorkerMessage<E>>`) and
/// `state` (`Arc<BackgroundState<E>>`), so no `PhantomData` is
/// required.
pub(crate) struct BackgroundCore<E: BackgroundCapable> {
    sender: SyncSender<WorkerMessage<E>>,
    /// Hold under `Mutex` so `finalize` and `shutdown` (both `self`-by-
    /// value) can take the handle out from an `Arc<Mutex<...>>` in
    /// the FFI side without `unsafe`.
    worker: Mutex<Option<JoinHandle<()>>>,
    config: BackgroundConfig,
    state: Arc<BackgroundState<E>>,
    /// Opt-in `submit_blocking` / `submit_timeout` latency sample
    /// accumulator (B5 — BackgroundEvaluator p99 latency cell).
    /// `None` when disabled (the default); always-on cost is one
    /// `is_some()` check per submit. Each sample is the wall time from
    /// the FFI entry to the channel send completing, in nanoseconds.
    latency_samples: Option<Mutex<Vec<u64>>>,
}

impl<E: BackgroundCapable> BackgroundCore<E> {
    /// Hand the evaluator to a fresh worker thread named
    /// `vernier-bg-stream-worker` and return the wrapper. Latency
    /// sample accumulation is off by default; see
    /// [`Self::spawn_with_options`] to opt in.
    pub(crate) fn spawn(evaluator: E, config: BackgroundConfig) -> std::io::Result<Self> {
        Self::spawn_with_options(evaluator, config, false)
    }

    /// Variant of [`Self::spawn`] with a per-instance opt-in for
    /// latency-sample accumulation (B5). When `record_latency_samples`
    /// is `true`, every successful `submit_blocking` /
    /// `submit_timeout` records the wall-time of the channel-send leg
    /// in nanoseconds; readers consume samples via
    /// [`Self::latency_samples_drain`]. When `false`, the
    /// accumulator stays unallocated and the only cost is one
    /// `Option::is_some` check per submit.
    pub(crate) fn spawn_with_options(
        evaluator: E,
        config: BackgroundConfig,
        record_latency_samples: bool,
    ) -> std::io::Result<Self> {
        let (sender, receiver) = sync_channel::<WorkerMessage<E>>(config.queue_capacity);
        let state = Arc::new(BackgroundState::<E>::new());
        // Seed counters so the FFI accessors don't lie before the
        // first update lands.
        state
            .images_seen
            .store(evaluator.images_seen(), Ordering::Release);
        state
            .memory_used_bytes
            .store(evaluator.memory_used_bytes(), Ordering::Release);

        let worker_state = Arc::clone(&state);
        let worker_config = config;
        let handle = thread::Builder::new()
            .name("vernier-bg-stream-worker".to_string())
            .spawn(move || worker_loop::<E>(evaluator, receiver, worker_state, worker_config))?;

        let latency_samples = if record_latency_samples {
            Some(Mutex::new(Vec::new()))
        } else {
            None
        };

        Ok(Self {
            sender,
            worker: Mutex::new(Some(handle)),
            config,
            state,
            latency_samples,
        })
    }

    /// Hand the worker an update; block until the channel has space.
    pub(crate) fn submit_blocking(&self, update: E::Update) -> Result<(), E::Error> {
        if let Some(err) = self.take_last_error() {
            return Err(err);
        }
        let start = self.latency_start();
        self.state.queue_depth.fetch_add(1, Ordering::AcqRel);
        match self.sender.send(WorkerMessage::Update(update)) {
            Ok(()) => {
                self.record_latency_sample(start);
                Ok(())
            }
            Err(_) => {
                self.state.queue_depth.fetch_sub(1, Ordering::AcqRel);
                Err(E::worker_disconnected())
            }
        }
    }

    /// Try to post an update with a bounded wait. `Duration::ZERO`
    /// is non-blocking; otherwise waits up to `timeout` polling
    /// `try_send`.
    pub(crate) fn submit_timeout(
        &self,
        update: E::Update,
        timeout: Duration,
    ) -> Result<(), SubmitError<E::Error>> {
        if let Some(err) = self.take_last_error() {
            return Err(SubmitError::Eval(err));
        }
        let start = self.latency_start();
        self.state.queue_depth.fetch_add(1, Ordering::AcqRel);
        let result = self.send_with_timeout(WorkerMessage::Update(update), timeout);
        match &result {
            Ok(()) => self.record_latency_sample(start),
            Err(_) => {
                self.state.queue_depth.fetch_sub(1, Ordering::AcqRel);
            }
        }
        result
    }

    /// Bounded `try_send` loop. `SyncSender::send_timeout` is
    /// unstable on stable Rust, so we poll. Same cadence as the
    /// instance background.
    fn send_with_timeout(
        &self,
        msg: WorkerMessage<E>,
        timeout: Duration,
    ) -> Result<(), SubmitError<E::Error>> {
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
                    return Err(SubmitError::Eval(E::worker_disconnected()));
                }
            }
        }
    }

    /// Request a snapshot. Blocks on the worker reply.
    pub(crate) fn snapshot(&self) -> Result<E::Summary, E::Error> {
        if let Some(err) = self.take_last_error() {
            return Err(err);
        }
        let (reply_tx, reply_rx) = sync_channel(1);
        if self
            .sender
            .send(WorkerMessage::Snapshot { reply: reply_tx })
            .is_err()
        {
            return Err(E::worker_disconnected());
        }
        reply_rx
            .recv()
            .unwrap_or_else(|_| Err(E::worker_disconnected()))
    }

    /// Drain and finalize. Consumes the wrapper.
    pub(crate) fn finalize(self) -> Result<E::Summary, E::Error> {
        if let Some(err) = self.take_last_error() {
            return Err(err);
        }
        let (reply_tx, reply_rx) = sync_channel(1);
        if self
            .sender
            .send(WorkerMessage::Finalize { reply: reply_tx })
            .is_err()
        {
            return Err(E::worker_disconnected());
        }
        let summary = reply_rx
            .recv()
            .unwrap_or_else(|_| Err(E::worker_disconnected()));
        self.join_after_consuming();
        summary
    }

    /// Snapshot to a partial blob (ADR-0032). Non-consuming.
    pub(crate) fn snapshot_to_partial(&self) -> Result<Vec<u8>, E::Error> {
        if let Some(err) = self.take_last_error() {
            return Err(err);
        }
        let (reply_tx, reply_rx) = sync_channel(1);
        if self
            .sender
            .send(WorkerMessage::SnapshotToPartial { reply: reply_tx })
            .is_err()
        {
            return Err(E::worker_disconnected());
        }
        reply_rx
            .recv()
            .unwrap_or_else(|_| Err(E::worker_disconnected()))
    }

    /// Finalize to a partial blob (ADR-0032). Consumes the wrapper.
    pub(crate) fn finalize_to_partial(self) -> Result<Vec<u8>, E::Error> {
        if let Some(err) = self.take_last_error() {
            return Err(err);
        }
        let (reply_tx, reply_rx) = sync_channel(1);
        if self
            .sender
            .send(WorkerMessage::FinalizeToPartial { reply: reply_tx })
            .is_err()
        {
            return Err(E::worker_disconnected());
        }
        let blob = reply_rx
            .recv()
            .unwrap_or_else(|_| Err(E::worker_disconnected()));
        self.join_after_consuming();
        blob
    }

    /// Best-effort cooperative shutdown. Sends `Shutdown`, then polls
    /// `JoinHandle::is_finished` up to `config.shutdown_timeout`. Any
    /// worker output is discarded — used by `__exit__` and `__del__`.
    pub(crate) fn shutdown(self) {
        let _ = self.sender.send(WorkerMessage::Shutdown);
        let handle = match self.take_worker() {
            Some(h) => h,
            None => return,
        };
        let deadline = Instant::now() + self.config.shutdown_timeout;
        loop {
            if handle.is_finished() {
                let _ = handle.join();
                return;
            }
            if Instant::now() >= deadline {
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    /// Approximate count of `update` messages currently in the channel.
    pub(crate) fn queue_depth(&self) -> usize {
        self.state.queue_depth.load(Ordering::Acquire)
    }

    /// Mirror of the evaluator's `images_seen()`. Advisory.
    pub(crate) fn images_seen(&self) -> usize {
        self.state.images_seen.load(Ordering::Acquire)
    }

    /// Mirror of the evaluator's `memory_used_bytes()`. Advisory.
    /// Mirrors the instance-background API surface so paradigms that
    /// later wire up a `memory_used_bytes` getter on their pyclass
    /// don't need to reach into private state.
    #[allow(
        dead_code,
        reason = "exposed for forward-compat with future paradigm getters"
    )]
    pub(crate) fn memory_used_bytes(&self) -> usize {
        self.state.memory_used_bytes.load(Ordering::Acquire)
    }

    /// Take the result of the worker's startup scheduling adjustment.
    /// FFI calls this once after spawn; subsequent calls return `None`.
    pub(crate) fn take_scheduling_outcome(&self) -> Option<Result<(), String>> {
        self.state.scheduling_outcome.lock().ok()?.take()
    }

    /// Read-and-clear the stashed update error.
    fn take_last_error(&self) -> Option<E::Error> {
        self.state.last_error.lock().ok()?.take()
    }

    /// Drain and replace the per-instance submit-latency sample buffer
    /// (B5). Returns the accumulated samples (each is the wall-time of
    /// the FFI-to-`mpsc::send` leg in nanoseconds) and leaves the inner
    /// `Vec` freshly empty so subsequent submits keep accumulating.
    /// Returns an empty `Vec` when the wrapper was constructed without
    /// `record_latency_samples`.
    ///
    /// Exposed pub(crate) so the panoptic / semantic FFI shims can
    /// wire it through their pyclasses if/when they need a
    /// per-paradigm `drain_latency_samples_ns()`. Today the
    /// instance evaluator routes its own fork (see
    /// [`crate::background::BackgroundEvaluator::latency_samples_drain`])
    /// because B5's saturation workload uses the bbox kernel.
    #[allow(
        dead_code,
        reason = "exposed for forward-compat with panoptic / semantic latency-cell wiring"
    )]
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
        // `Instant::elapsed` returns `Duration::from_secs(0)` when the
        // monotonic clock is non-monotonic on the platform; saturating
        // to `u64::MAX` keeps a runaway tail observable rather than
        // wrapping into "fast" measurements.
        let ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        if let Ok(mut guard) = slot.lock() {
            guard.push(ns);
        }
    }

    fn take_worker(&self) -> Option<JoinHandle<()>> {
        self.worker.lock().ok()?.take()
    }

    /// Join the worker after a consuming op (`finalize` /
    /// `finalize_to_partial`). Worker panics are not surfaced here —
    /// the worker's `Result` is `()` and the reply channel already
    /// carried any typed error to the caller.
    fn join_after_consuming(&self) {
        if let Some(handle) = self.take_worker() {
            let _ = handle.join();
        }
    }

    /// Test-only: panic the worker.
    #[cfg(feature = "test-poison")]
    pub(crate) fn _inject_poison_for_tests(&self) -> Result<(), E::Error> {
        self.sender
            .send(WorkerMessage::Poison)
            .map_err(|_| E::worker_disconnected())
    }
}

/// Owning wrapper around a `BackgroundCore<E>` that models its
/// post-finalize state. Both paradigm pyclasses store
/// `Mutex<BackgroundLifecycle<E>>` so `submit` / `snapshot` / etc.
/// see the same active-or-finalized story without each paradigm
/// re-implementing the enum + dispatch table.
///
/// `take_and_*` methods consume the `BackgroundCore` (it has
/// `self`-by-value finalize methods); subsequent ops resolve to
/// `E::already_finalized()`.
pub(crate) struct BackgroundLifecycle<E: BackgroundCapable> {
    core: Option<BackgroundCore<E>>,
}

impl<E: BackgroundCapable> BackgroundLifecycle<E> {
    pub(crate) fn new(core: BackgroundCore<E>) -> Self {
        Self { core: Some(core) }
    }

    /// Borrow the active core. `Err(already_finalized)` if the
    /// wrapper has already been consumed.
    pub(crate) fn active(&self) -> Result<&BackgroundCore<E>, E::Error> {
        self.core.as_ref().ok_or_else(E::already_finalized)
    }

    /// Drain, finalize, and join the worker. Replaces the core with
    /// `None` so subsequent ops resolve to `already_finalized`.
    pub(crate) fn take_and_finalize(&mut self) -> Result<E::Summary, E::Error> {
        match self.core.take() {
            Some(core) => core.finalize(),
            None => Err(E::already_finalized()),
        }
    }

    /// Drain, serialize the final state to a partial blob, and join
    /// the worker.
    pub(crate) fn take_and_finalize_to_partial(&mut self) -> Result<Vec<u8>, E::Error> {
        match self.core.take() {
            Some(core) => core.finalize_to_partial(),
            None => Err(E::already_finalized()),
        }
    }

    /// Best-effort cooperative shutdown. Drops the core regardless of
    /// the worker's response.
    pub(crate) fn shutdown(&mut self) {
        if let Some(core) = self.core.take() {
            core.shutdown();
        }
    }

    /// Forwarder for the FFI's one-shot scheduling-outcome read.
    pub(crate) fn take_scheduling_outcome(&self) -> Option<Result<(), String>> {
        self.core.as_ref()?.take_scheduling_outcome()
    }

    /// Forwarder for [`BackgroundCore::latency_samples_drain`] (B5).
    /// Returns an empty `Vec` when the wrapper has been finalized or
    /// was constructed without `record_latency_samples`.
    #[allow(
        dead_code,
        reason = "exposed for forward-compat with panoptic / semantic latency-cell wiring"
    )]
    pub(crate) fn latency_samples_drain(&self) -> Vec<u64> {
        match self.core.as_ref() {
            Some(core) => core.latency_samples_drain(),
            None => Vec::new(),
        }
    }
}

/// Worker entry point. Owns the evaluator outright; the only way to
/// mutate state is `WorkerMessage::Update`, which is the structural
/// enforcer of the single-writer rule.
//
// `rx`, `state`, and `config` are intentionally passed by value: this
// is a thread entry point, so the values must be owned here.
#[allow(clippy::needless_pass_by_value)]
fn worker_loop<E: BackgroundCapable>(
    mut evaluator: E,
    rx: Receiver<WorkerMessage<E>>,
    state: Arc<BackgroundState<E>>,
    config: BackgroundConfig,
) {
    let outcome = crate::thread_sched::apply_scheduling(config.worker_nice, config.worker_affinity);
    if let Ok(mut guard) = state.scheduling_outcome.lock() {
        *guard = Some(outcome);
    }

    loop {
        match rx.recv() {
            Ok(WorkerMessage::Update(update)) => {
                state.queue_depth.fetch_sub(1, Ordering::AcqRel);
                match evaluator.apply_update(update) {
                    Ok(()) => {
                        state
                            .images_seen
                            .store(evaluator.images_seen(), Ordering::Release);
                        state
                            .memory_used_bytes
                            .store(evaluator.memory_used_bytes(), Ordering::Release);
                    }
                    Err(e) => {
                        if let Ok(mut guard) = state.last_error.lock() {
                            *guard = Some(e);
                        }
                    }
                }
            }
            Ok(WorkerMessage::Snapshot { reply }) => {
                let s = evaluator.snapshot();
                let _ = reply.send(s);
            }
            Ok(WorkerMessage::Finalize { reply }) => {
                let s = evaluator.finalize();
                let _ = reply.send(s);
                return;
            }
            Ok(WorkerMessage::SnapshotToPartial { reply }) => {
                let p = evaluator.snapshot_to_partial();
                let _ = reply.send(p);
            }
            Ok(WorkerMessage::FinalizeToPartial { reply }) => {
                let p = evaluator.finalize_to_partial();
                let _ = reply.send(p);
                return;
            }
            Ok(WorkerMessage::Shutdown) | Err(RecvError) => return,
            #[cfg(feature = "test-poison")]
            Ok(WorkerMessage::Poison) => {
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
    //! Latency-sample accumulator unit tests (B5). Exercise the
    //! generic `BackgroundCore` opt-in path without the instance
    //! `StreamingEvaluator<K>` parsing pipeline.

    use super::*;

    /// Minimal `BackgroundCapable` impl: counts `apply_update` calls
    /// in an in-memory u64. Sufficient to drive `submit_blocking` for
    /// the latency hook test.
    #[derive(Default)]
    struct CountingEvaluator {
        applied: usize,
    }

    impl BackgroundCapable for CountingEvaluator {
        type Update = ();
        type Summary = usize;
        type Error = String;

        fn apply_update(&mut self, _: ()) -> Result<(), String> {
            self.applied += 1;
            Ok(())
        }

        fn snapshot(&self) -> Result<usize, String> {
            Ok(self.applied)
        }

        fn finalize(self) -> Result<usize, String> {
            Ok(self.applied)
        }

        fn snapshot_to_partial(&self) -> Result<Vec<u8>, String> {
            Ok(Vec::new())
        }

        fn finalize_to_partial(self) -> Result<Vec<u8>, String> {
            Ok(Vec::new())
        }

        fn images_seen(&self) -> usize {
            self.applied
        }

        fn worker_disconnected() -> String {
            "worker_disconnected".to_string()
        }

        fn already_finalized() -> String {
            "already_finalized".to_string()
        }
    }

    fn config() -> BackgroundConfig {
        BackgroundConfig {
            queue_capacity: 4,
            worker_affinity: None,
            worker_nice: 0,
            shutdown_timeout: Duration::from_millis(200),
        }
    }

    #[test]
    fn latency_samples_disabled_by_default_returns_empty() {
        let core = match BackgroundCore::spawn(CountingEvaluator::default(), config()) {
            Ok(c) => c,
            Err(e) => panic!("spawn worker: {e}"),
        };
        for _ in 0..8u32 {
            if let Err(e) = core.submit_blocking(()) {
                panic!("submit: {e}");
            }
        }
        let drained = core.latency_samples_drain();
        assert!(
            drained.is_empty(),
            "default-off accumulator must return empty"
        );
        core.shutdown();
    }

    #[test]
    fn latency_samples_opt_in_records_one_per_submit() {
        let core = match BackgroundCore::spawn_with_options(
            CountingEvaluator::default(),
            config(),
            true,
        ) {
            Ok(c) => c,
            Err(e) => panic!("spawn worker: {e}"),
        };
        let n = 16usize;
        for _ in 0..n {
            if let Err(e) = core.submit_blocking(()) {
                panic!("submit: {e}");
            }
        }
        let drained = core.latency_samples_drain();
        assert_eq!(drained.len(), n, "one sample per successful submit");
        // Subsequent drain returns empty (the buffer was reset).
        assert!(core.latency_samples_drain().is_empty());
        core.shutdown();
    }
}
