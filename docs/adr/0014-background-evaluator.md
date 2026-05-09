# ADR-0014: `BackgroundEvaluator` — single-worker, bounded-queue async wrapper around `StreamingEvaluator`

- **Status:** accepted (amended by [ADR-0035](0035-api-surface-consolidation.md))
  — the public surface is trimmed to ``submit`` / ``finalize`` /
  ``finalize_with_tables`` / ``finalize_to_partial`` / context manager.
  ``snapshot``, ``snapshot(peek=True)``, and the non-finalize
  ``to_partial`` are removed. The single-worker resource discipline is
  unchanged.
- **Date:** 2026-04-28
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

ADR-0006 promised a `BackgroundEvaluator` for Phase 3, deferred to
Phase 5 along with the rest of the streaming story. ADR-0013 lands
the synchronous half: `StreamingEvaluator` is single-writer, mutable,
and runs `match_image` + `accumulate()` on the calling thread. This
ADR is the asynchronous half — the wrapper that takes
`StreamingEvaluator` off the training thread.

The motivating shape is the training-loop persona. A robotics training
rig running every few hundred steps of validation eval inline is paying
two costs the trainer cannot afford:

1. **Wall-clock.** Even at vernier's per-image speeds, a 5k-image
   validation pass takes seconds. Multiplied by the per-epoch eval
   frequency the training-loop user actually wants (every N steps,
   not every epoch), it dominates.
2. **CPU contention.** The training process is already saturating
   cores on data loading, augmentation, and loss computation. Eval
   competes for those cores. Naive thread-spawning makes this worse,
   not better.

The `BackgroundEvaluator` exists to take eval off the critical path
of training without making the contention problem worse. The contract
is: `submit(detections)` returns in milliseconds (JSON parse +
channel post, no matching); the eval work happens on a single,
bounded-resource worker; the trainer never blocks on eval unless the
user explicitly asked it to.

The risk shape this ADR has to manage is different from ADR-0013.
ADR-0013 was a numerical-determinism ADR: the question was "do
streaming snapshots equal batch results?" This ADR is a
**resource-budget ADR**: the question is "what guarantees does
the wrapper give about its CPU, memory, and queue footprint, such
that a trainer can adopt it without measuring resource impact for
every workload?" The right defaults are conservative on resource
consumption and noisy when the budget is exceeded, exactly mirroring
the memory-budget rule from ADR-0013.

ADR-0006 framed the type as having `submit(image_id, gt, dt)` and
`result()` async semantics. Two things have changed since: ADR-0013
established that detections are submitted as batched bytes (not
per-image), and the `StreamingEvaluator`'s `update()` /
`snapshot()` / `finalize()` shape is what the wrapper composes
around. The signatures here are updated to match.

## Decision drivers

- **Eval is less important than training.** The BackgroundEvaluator
  exists because training-loop time is precious; if eval falls behind,
  eval is the loss leader, not the trainer. Defaults must encode this
  asymmetry — anywhere a trade-off exists between training throughput
  and eval throughput, the design picks training.
- **Bound the CPU footprint at one core.** A trainer that adopts
  `BackgroundEvaluator` must not see eval grow from "one process
  saturating one core" to "eval contending with the data loader for
  N cores." Single-worker, no internal parallelism inside the worker.
- **Bound the memory footprint at the streaming budget plus the
  queue.** ADR-0013's `memory_budget_bytes` already caps the
  streaming evaluator. The queue adds a bounded buffer of
  not-yet-processed batches; that bound has to be small and
  configurable.
- **No new top-level dep.** `std::sync::mpsc::sync_channel` is in
  the standard library and is sufficient. ADR-0001 §"Add or remove
  a top-level dependency" applies to crossbeam-channel,
  crossbeam-queue, tokio, and rayon equivalently — none of them
  cross the bar against `std::sync::mpsc` for what this ADR needs.
- **Compose, don't fork.** The wrapper owns a `StreamingEvaluator`
  and dispatches `update()` / `snapshot()` / `finalize()` calls to
  it from the worker thread. ADR-0013's parity contract carries
  through unchanged. ADR-0005's no-edits invariant carries through
  unchanged.
- **Public API.** Per ADR-0001 §"Affect the public API" and
  §"Cross the FFI boundary."

## Considered options

The ADR has four orthogonal axes. Each axis is decided independently;
the chosen design is the combination of one option per axis.

### Axis A — Worker count

1. **Single dedicated worker.** One worker thread per evaluator
   instance, lifetime tied to the evaluator. Bounded CPU footprint;
   trivial scheduling.
2. **Pool of N workers.** Configurable pool size, default = number
   of cores. Best per-call throughput, worst contention with the
   training process.
3. **Thread-on-demand.** Spawn a worker per `submit()`, join at
   drain. No persistent thread; better for sporadic eval. Per-spawn
   overhead is high enough to offset benefits.

### Axis B — Backpressure on full queue

1. **Block submit forever.** `submit()` waits until the worker drains
   one slot. Predictable; trainer becomes a regulator on eval. But
   training stalls silently when eval is slow.
2. **Raise immediately.** `submit()` raises `QueueFullError` on a
   full queue. Loud; user must handle. Forces explicit policy.
3. **Drop oldest.** `submit()` displaces the oldest queued batch.
   Never blocks; silently invalidates eval results.
4. **Configurable timeout.** `submit(timeout=...)`. Caller picks
   policy per submit.

### Axis C — Snapshot semantics

1. **Drain-then-snapshot.** Wait for the worker to process the queue,
   then snapshot. Result is stats over all submitted detections.
2. **Peek-current-state.** Snapshot whatever the worker has
   processed; ignore queued-but-unprocessed batches. Fast but
   produces stats over an arbitrary submitted subset.
3. **Both, with a flag.** Default is drain; `peek=True` is the fast
   path. Mirrors ADR-0013's `running=True` precedent.

### Axis D — Where JSON parse runs

1. **On the training thread.** `submit()` parses bytes into the
   internal detection structure, then queues the structure. Trainer
   pays parse cost (milliseconds per batch); worker is pure compute.
2. **On the worker thread.** `submit()` queues raw bytes, returns
   immediately. Trainer pays only allocation + queue post; worker
   does parse + match + accumulate.

## Decision outcome

The design is the combination **A1 + B4 + C3 + D1**: single dedicated
worker, configurable submit timeout, drain-by-default with peek flag,
JSON parse on the submitting thread.

The reasoning per axis:

- **A1 (single worker).** This is the load-bearing decision. Eval is
  not allowed to grow past one CPU core regardless of workload. A
  pool is a follow-up if and when a real production deployment shows
  one worker is the bottleneck *after* the worker's own optimizations
  have been exhausted; it is not a v0.1 default. The same `mpsc`
  channel used here trivially extends to N consumers when that day
  arrives, so we are not foreclosing the design.
- **B4 (configurable timeout, default `None` = block forever).** Per
  the principal-engineer review of this question: blocking is the
  correct default because silent data loss in eval invalidates
  quality gates, and a quietly-stalled trainer is at least
  *visible* (training throughput drops, the user notices). The user
  who wants non-blocking semantics passes `timeout=0.0` and handles
  `QueueFullError`; the user who wants graceful degradation passes
  `timeout=2.0` and handles the timeout. Drop-oldest is not on the
  menu.
- **C3 (drain by default, peek opt-in).** Mirrors ADR-0013's
  `running=True` precedent. Drain is correct; peek is a training
  signal. The user who logs `peek=True` to a TensorBoard scalar gets
  fast feedback without claiming parity; the CI quality gate runs
  with the default and gets the deterministic answer.
- **D1 (parse on submit thread).** The training thread's natural
  latency at batch boundaries (waiting for the next data-loader
  batch, GPU sync, etc.) is the right place to amortize JSON parse.
  Putting it on the worker would make `submit()` hold the bytes
  longer, increasing allocator pressure on the hot path, and would
  let parse errors surface only when the user calls `result()` —
  far from where they were caused. Parse-on-submit fails fast,
  fails loudly, and uses cycles that would otherwise be idle.

### Rust core surface

A new module `crates/vernier-core/src/background.rs` defines:

```rust
pub struct BackgroundEvaluator<K: EvalKernel> {
    sender: SyncSender<WorkerMessage<K>>,
    worker: Option<JoinHandle<Result<(), EvalError>>>,
    snapshot_chan: SnapshotChannel,
    config: BackgroundConfig,
    state: Arc<BackgroundState>,   // counters, last error, finalize sentinel
}

enum WorkerMessage<K: EvalKernel> {
    Update(ParsedDetections<K>),
    Snapshot { reply: SyncSender<Result<Summary, EvalError>>, peek: bool },
    Checkpoint { reply: SyncSender<Result<Vec<u8>, EvalError>> },
    Finalize { reply: SyncSender<Result<Summary, EvalError>> },
    Shutdown,
}

#[derive(Clone, Copy)]
pub struct BackgroundConfig {
    pub queue_capacity: usize,         // default 8
    pub worker_affinity: Option<usize>, // default None
    pub worker_nice: i32,              // default +5 (lower priority than trainer)
    pub memory_budget_bytes: Option<usize>,  // forwarded to StreamingEvaluator
}
```

The worker thread runs:

```rust
fn worker_loop<K: EvalKernel>(
    mut evaluator: StreamingEvaluator<K>,
    rx: Receiver<WorkerMessage<K>>,
    state: Arc<BackgroundState>,
    config: BackgroundConfig,
) -> Result<(), EvalError> {
    apply_affinity_and_nice(&config);
    while let Ok(msg) = rx.recv() {
        match msg {
            WorkerMessage::Update(parsed) => {
                let report = evaluator.update_parsed(parsed)?;
                state.update_counters(&report);
            }
            WorkerMessage::Snapshot { reply, peek } => {
                let summary = if peek {
                    evaluator.snapshot_running()
                } else {
                    evaluator.snapshot()
                };
                let _ = reply.send(summary);
            }
            WorkerMessage::Checkpoint { reply } => {
                let _ = reply.send(Ok(evaluator.checkpoint()));
            }
            WorkerMessage::Finalize { reply } => {
                // finalize consumes — replace evaluator with a sentinel,
                // send the summary, exit the loop
                let summary = evaluator.finalize();
                let _ = reply.send(summary);
                return Ok(());
            }
            WorkerMessage::Shutdown => return Ok(()),
        }
    }
    Ok(())
}
```

Three structural points:

- **The worker owns the `StreamingEvaluator`.** Per ADR-0013's
  single-writer rule, the worker thread becomes the owner thread on
  the first `update()`. The wrapper forces this by construction —
  no `update()` ever runs anywhere else, so the owner-thread invariant
  is satisfied without runtime checks.
- **Snapshot and checkpoint are messages, not direct calls.** They
  travel through the same channel as updates so the worker handles
  them in submission order. A snapshot submitted *after* updates U1,
  U2, U3 sees the state after U3, never some other order. This is
  the in-order property the streaming evaluator's parity contract
  implicitly assumed; the background wrapper makes it explicit.
- **`finalize` consumes the wrapper.** When the worker processes
  `Finalize`, it calls the underlying `StreamingEvaluator::finalize`
  (which consumes that), sends the summary back, and returns from
  the loop. The Python wrapper joins the worker thread and refuses
  subsequent calls.

`update_parsed` is a new internal method on `StreamingEvaluator`
that takes already-parsed detections (skipping the JSON step). The
public `update(bytes)` on `StreamingEvaluator` calls the parser and
then `update_parsed` internally. The split exists so the
background wrapper can parse on the submit thread and pass the
parsed structure across the channel. `update_parsed` is *not*
exposed at the FFI boundary.

### Python surface

```python
class BackgroundEvaluator:
    def __init__(
        self,
        ground_truth: bytes,
        *,
        iou_type: IouType = "bbox",
        parity_mode: ParityMode = "corrected",
        max_dets: tuple[int, ...] = (1, 10, 100),
        use_cats: bool = True,
        # background-specific
        queue_capacity: int = 8,
        worker_affinity: int | None = None,
        worker_nice: int = 5,
        memory_budget_bytes: int | None = None,
    ) -> None: ...

    def submit(
        self,
        detections: bytes,
        *,
        timeout: float | None = None,   # None = block forever
    ) -> None: ...

    def snapshot(self, *, peek: bool = False) -> Summary: ...

    def finalize(self) -> Summary: ...

    def checkpoint(self) -> bytes: ...

    @classmethod
    def restore(cls, ground_truth: bytes, state: bytes, **kwargs) -> BackgroundEvaluator: ...

    # Context manager — guarantees worker shutdown on scope exit
    def __enter__(self) -> BackgroundEvaluator: ...
    def __exit__(self, *exc) -> None: ...

    # Counters reflect what the worker has *processed*, not what was submitted.
    @property
    def images_seen(self) -> int: ...
    @property
    def detections_seen(self) -> int: ...
    @property
    def queue_depth(self) -> int: ...      # currently queued, not yet processed
    @property
    def memory_used_bytes(self) -> int: ...
```

A new exception:

```python
class QueueFullError(RuntimeError):
    queue_capacity: int
    timeout: float | None
```

Three deliberate API choices:

- **Context-manager support.** Background workers are a resource the
  Python user can leak. `__enter__` / `__exit__` pair guarantees the
  worker shuts down cleanly on scope exit, including on exception.
  Without this, a crash in the training loop leaves a zombie worker
  thread until process exit. Context-manager use is the documented
  default in the how-to guide; bare `__init__` works but is a
  footgun.
- **`submit()` returns `None`, not a future.** The background
  evaluator is intentionally fire-and-forget for `update`s — the user
  observes results via `snapshot` / `finalize`, not by awaiting per-
  submit completions. A future-per-submit API would force users to
  track futures across the training loop, which is the kind of
  bookkeeping the wrapper exists to remove. `snapshot` and
  `finalize` are synchronous waits because *those* are the points
  where the user actually wants the answer.
- **Counters reflect processed, not submitted.** `images_seen` is
  what the worker has finished, not what's in flight. Submission
  counters are unnecessary — the user already tracked what they
  submitted. The wrapper's counters answer the question "is the
  worker keeping up?", which is the production-relevant signal.
  `queue_depth` exposes the in-flight count for the same reason.

The detection payload is bytes — JSON, exactly as
`StreamingEvaluator.update` accepts. Per the D1 decision, parse runs
on the submit thread before the channel post; the worker receives
the parsed structure, not bytes. This is invisible to the Python
user, who sees only `submit(bytes)`.

### Threading and the GIL

`submit()` parses JSON with the GIL held (the JSON parser is in
Rust but operates on a `&[u8]` borrowed from the Python `bytes`),
then drops the GIL via `Python::detach` for the channel post. The
post is `try_send` with the configured timeout; it does not block
holding the GIL.

`snapshot(peek=True)` posts the snapshot message and waits on the
reply channel — both with GIL dropped. The worker is single-threaded
compute (per the broader CPU-budget rule above) and runs
`evaluator.snapshot_running()` directly when it processes the
message. Wall-clock cost of the round trip is one queue post + the
worker draining whatever messages are ahead of it + one reply
channel post. The worker's queue is FIFO, so a `peek=True` snapshot
*still* waits behind already-submitted updates — the difference is
that the snapshot itself is fast once the worker reaches it.

`snapshot()` (drain) is the same shape: post the message, wait for
reply. The worker drains the queue ahead of the snapshot before
producing it. There is no separate "drain" message; the in-order
property of the channel does the drain implicitly.

`finalize()` posts the finalize message, waits for reply, joins the
worker thread. If the worker has died (panicked or hit an
unrecoverable error), `finalize` re-raises the worker's last error.

### Worker scheduling

The worker thread is created with two non-default scheduling hints:

- **`worker_nice`** (default +5 on Unix, `BELOW_NORMAL_PRIORITY_CLASS`
  equivalent on Windows). Eval should not preempt training. The
  `nice` value is configurable for users on real-time-critical
  systems where any priority inversion is unacceptable; the default
  encodes the asymmetry.
- **`worker_affinity`** (default `None` — let the OS schedule). On a
  known-topology rig, the user can pin the worker to a specific
  core and the trainer/data loader to others. This is a tuning knob
  for the MLOps integrator persona; the default is unpinned because
  premature pinning is worse than unpinned on most workloads. We
  document the knob and a recipe in the how-to guide.

Both knobs are best-effort. If the platform or the environment
denies the syscall (containerized deployment without
`SYS_NICE`, for instance), the worker falls back to default
scheduling and emits a warning *once*, with the syscall error
attached. We do not fail construction — denied affinity / nice is
common in production and not a reason to refuse to run.

### Backpressure

The channel is `sync_channel(queue_capacity)` — a bounded MPSC
that blocks the sender on full. The Python wrapper's `submit`
handles three timeout cases:

- **`timeout=None`** (default). `try_send` in a loop with no
  timeout — equivalent to `send`. The training thread blocks until
  the worker drains a slot. This is the safe default; eval is
  always correct, and a stalled trainer is a visible signal that
  eval is the bottleneck.
- **`timeout=0.0`**. `try_send` once. On full, raises
  `QueueFullError` with the current queue capacity. The user
  handles the error explicitly — drop the batch, log, raise the
  capacity, or whatever their policy is.
- **`timeout=t > 0`**. `try_send` with a deadline. On timeout,
  `QueueFullError`. Same handling.

Drop-oldest semantics are explicitly not supported. If the user
wants drop-oldest, they implement it on top of `timeout=0.0` and
their own ring buffer. We do not bake silent data loss into the
default API.

The default `queue_capacity=8` is small intentionally. The intent
is to bound queueing latency: with eight queued batches, the user
sees backpressure within a small number of training steps if eval
is slow, rather than building up a many-second backlog before the
problem becomes visible. Users who want to absorb bursty submission
can raise this; users who want immediate feedback on eval slowness
can lower it.

### Error propagation

The worker can fail in two ways:

- **Recoverable error** during `update_parsed` (e.g., a malformed
  detection that survived the JSON parser, or
  `OutOfBudgetError` from the underlying streaming evaluator).
  The worker stores the error in `Arc<BackgroundState>` and
  *continues running*. The next `submit` / `snapshot` /
  `finalize` reads the stored error and re-raises it on the
  caller's thread. This means a broken batch produces an exception
  near where the caller was when they submitted it, not pages
  later.
- **Worker panic.** The worker's `JoinHandle::join` returns an
  error. The wrapper marks the worker dead, drops the channel, and
  re-raises a `RuntimeError` describing the panic on the next
  Python call. The wrapper does *not* attempt to restart the
  worker — a panic is a bug, and silent restart would mask it.

Both paths preserve the calling-thread guarantee: the user always
sees errors on the thread that called the wrapper, not on the
worker thread (which is hidden from them by design).

### Lifecycle and shutdown

Three shutdown paths:

- **Normal `finalize`.** Worker processes the `Finalize` message,
  returns the summary, exits cleanly. The wrapper joins.
- **Context-manager exit without `finalize`.** `__exit__` posts
  `Shutdown`, joins the worker, drops any queued messages. The
  evaluator state is lost; the user gets no summary. Documented
  as such — context-manager exit on the *abnormal* path is for
  cleanup, not for results. If they want results on early exit,
  they call `snapshot` before leaving the scope.
- **Object dropped without `__exit__`.** `__del__` posts
  `Shutdown` and joins with a small timeout (default 5s). On
  timeout, the worker is detached and the process leaks one
  thread until exit. We log a warning. This is the footgun case
  the context-manager path is designed to prevent; we make it
  noisy rather than clean because clean would let users skip
  `__exit__` indefinitely.

### Parity harness

The background-evaluator parity tests build on the streaming
fixtures from ADR-0013:

- **Async-equals-sync.** For every fixture in `ALL_FIXTURES`,
  evaluate via `BackgroundEvaluator.submit(...) → finalize()`
  and via `StreamingEvaluator.update(...) → finalize()`. Stats
  must be bit-equal. (This validates that the channel, the
  worker thread, and the JSON parse split do not perturb the
  result.)
- **Drain snapshot equals streaming snapshot.** Submit half the
  detections, `snapshot()`; assert bit-equality with
  `StreamingEvaluator.snapshot()` after the same half-submission.
- **Peek snapshot is fast and biased.** Submit a stream with a
  full queue ahead of the snapshot; `snapshot(peek=True)`
  completes within the documented latency budget; result differs
  from drain-snapshot by a measurable amount (regression guard,
  not a correctness claim).
- **Backpressure raises.** `queue_capacity=2`, submit three
  payloads with `timeout=0.0`, assert third raises
  `QueueFullError` with capacity in the error.
- **Backpressure blocks.** Same setup with `timeout=None`,
  assert the third submit blocks until a slow worker drains
  one slot, then succeeds.
- **Worker panic re-raises on calling thread.** Inject a poison
  payload (via a test-only hook) that panics the worker; assert
  the next Python call raises `RuntimeError` carrying the panic
  payload, *not* on the worker thread.
- **Context-manager early exit.** `with BackgroundEvaluator(...)
  as ev: ev.submit(...); raise ValueError`. On exit, worker is
  joined, no zombie thread. Process-level thread count returns
  to baseline within the join timeout.
- **Affinity / nice are best-effort.** In a container without
  `SYS_NICE`, construction succeeds and emits exactly one warning.

### What this ADR explicitly does not decide

- **Worker pool.** Single-worker is the v0.1 commitment. A pool is
  a follow-up *only* if a production workload shows one worker is
  the documented bottleneck, *and* the worker's own optimizations
  (D1 parse-on-submit, future internal speedups) have been
  exhausted. Speculative pool support is a no.
- **`merge` API across instances.** Out of scope for this ADR; same
  follow-up as ADR-0013's merge note. Sharded eval across processes
  combines via `StreamingEvaluator.merge`, which the
  background wrapper inherits trivially once it ships.
- **Async/await Python interface.** `submit` returns `None`, not a
  coroutine. Users who want `asyncio` integration wrap the wrapper
  in `loop.run_in_executor(...)`. We do not bake an async-runtime
  choice into vernier — that's a `pyo3-asyncio` decision per
  ADR-0006 §"Considered options" (Option 4), which was rejected for
  the same reasons.
- **Cross-process worker.** A worker in a different process (via
  `multiprocessing` or otherwise) would let eval escape the GIL
  and the trainer's process address space entirely. Real value
  in some deployments, but it's a different shape (queue becomes
  a pipe, error propagation becomes pickling, lifecycle becomes
  process management). Out of scope; if a user wants it, they
  build it on top of `StreamingEvaluator` directly.

### Consequences

- **Positive.** The training-loop persona gets eval off their
  critical path with one line of setup. CPU footprint is bounded
  at one core by default. Memory footprint is bounded by ADR-0013's
  budget plus `queue_capacity` batches' worth of parsed
  detections. The default backpressure is correct and visible.
  The default snapshot semantics are correct. The wrapper composes
  on top of the locked spine without touching `matching.rs`,
  `accumulate.rs`, or `stream.rs`. Phase 5 ships with both
  synchronous and asynchronous streaming evaluators, and they share
  one parity story.
- **Negative.** Single-worker means the background evaluator is
  not faster than `StreamingEvaluator` on per-eval throughput —
  it just gets out of the way of the trainer. A user with a CPU
  budget to spare and a real need for higher eval throughput is
  not served by this ADR; they wait for a follow-up that
  introduces a pool. Worker `nice`/affinity are best-effort and
  silently degrade in container environments — production users
  in containerized deployments may see eval scheduling that
  doesn't match the documentation. The bounded queue with
  default capacity 8 is a heuristic; some workloads will need
  tuning. Context-manager use is strongly recommended but not
  enforced.
- **Neutral.** `BackgroundEvaluator` and `StreamingEvaluator`
  cover overlapping use cases. Documentation positions
  `StreamingEvaluator` as the default for offline / batch /
  CI eval and `BackgroundEvaluator` as the answer for
  in-training eval; users who don't need the wrapper don't pay
  for the worker thread.

## Pros and cons of the options

### Axis A — Worker count

**A1 (chosen) — single dedicated worker**
- 👍 Bounded CPU footprint. Predictable scheduling. No new dep.
- 👍 Trivially extends to a pool later via the same channel shape.
- 👎 Per-eval throughput capped at one core's worth of compute.

**A2 — pool of N workers**
- 👍 Higher per-eval throughput on big-CPU rigs.
- 👎 Contends with training process for cores. Loses the
  CPU-budget property the ADR is built around. Requires per-worker
  evaluator state and merge logic.

**A3 — thread-on-demand**
- 👍 Zero overhead when not evaluating.
- 👎 Per-spawn cost dominates for small batches. The whole point
  of the wrapper is steady-state low-overhead eval.

### Axis B — Backpressure

**B4 (chosen) — configurable timeout, default `None`**
- 👍 Defaults to correctness (eval never silently drops). Explicit
  opt-in for non-blocking semantics. Three modes (block / fail /
  wait) cover the realistic policies.
- 👎 Default blocks the trainer when eval is slow. Visible, not
  silent.

**B1 — block forever, no timeout knob**
- 👍 Simplest API.
- 👎 Forces every user into the same policy.

**B2 — raise immediately**
- 👍 Forces explicit policy.
- 👎 Default is hostile; every user has to handle queue-full
  even if they'd be fine waiting briefly.

**B3 — drop oldest**
- 👍 Trainer never blocks.
- 👎 Silent data loss invalidates eval results. Hard no for any
  CI / quality-gate use case.

### Axis C — Snapshot semantics

**C3 (chosen) — drain by default, peek opt-in**
- 👍 Correct default; fast path available. Mirrors ADR-0013's
  `running=True` precedent.
- 👎 Two semantics to document.

**C1 — drain only**
- 👍 One semantic.
- 👎 Mid-training peek is expensive; users skip eval logging
  to avoid the cost.

**C2 — peek only**
- 👍 Mid-training peek is fast.
- 👎 No correct snapshot; users have to build one themselves
  via `finalize`.

### Axis D — JSON parse location

**D1 (chosen) — parse on submit thread**
- 👍 Fails fast on parse errors (caller sees them at the
  submit, not pages later). Worker stays pure compute. Submit
  thread's natural latency at batch boundaries amortizes the
  parse cost.
- 👎 Submit cost scales with batch size — a 10k-detection
  batch costs more to submit than a 100-detection one.

**D2 — parse on worker thread**
- 👍 Constant submit cost.
- 👎 Parse errors surface at `result()`, far from the
  cause. Worker thread's CPU budget is now split between parse
  and compute — the very split the single-worker rule was meant
  to keep tight.

## Links and references

- ADR-0001 — Record architecture decisions (§"Affect the public API",
  §"Cross the FFI boundary", §"Add or remove a top-level dependency",
  §"Change the threading model").
- ADR-0002 — Three-tier parity model. Inherited unchanged through
  the streaming wrapper.
- ADR-0005 — Lock the `Similarity` trait and matching-engine API for
  Phases 1–3. Untouched: the background wrapper composes around
  `StreamingEvaluator`, which composes around the locked spine.
- ADR-0006 — Threading model. Promised `BackgroundEvaluator` as a
  Phase 3 deliverable. This ADR is its delivery, with the API shape
  updated to match ADR-0013's `submit(bytes)` over
  `submit(image_id, gt, dt)`.
- ADR-0013 — Streaming evaluator. The wrapper here owns one of these
  per instance; all parity, memory-budget, and detection-id
  contracts inherit unchanged.
- `crates/vernier-core/src/stream.rs` — the `StreamingEvaluator` the
  worker thread owns. The new `update_parsed` private method is the
  fast path the wrapper uses.
- `docs/explanation/possible-extensions.md` — the Phase 5 capability
  ranking that motivated streaming and the in-training-eval
  use case this ADR addresses.
