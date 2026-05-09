# ADR-0013: Streaming evaluator — store per-image evals, fold on snapshot and finalize

- **Status:** superseded by [ADR-0035](0035-api-surface-consolidation.md)
  (the public ``StreamingEvaluator`` class is demoted to ``vernier._impl``
  and the DDP entry points move to ``Evaluator``; the streaming substrate
  itself continues to exist below the FFI). The ``snapshot(running=True)``
  and ``checkpoint``/``restore`` sections of this ADR are no longer
  load-bearing.
- **Original status:** accepted
- **Date:** 2026-04-28
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

Phase 5 ships a `StreamingEvaluator` whose purpose is to evaluate
detection metrics *during* a training run, on predictions that arrive
image-by-image (or batch-by-batch), without holding the full detection
corpus in memory and without blocking the training loop on a synchronous
end-of-epoch eval pass. The capability is described in
`docs/explanation/possible-extensions.md` as the load-bearing piece of
Phase 5 — the unlock that makes per-image DataFrames, mid-training
calibration peeks, and crash-recoverable eval state simultaneously
cheap.

The design space is constrained on three sides.

1. **The matching engine and accumulator are locked.** ADR-0005's
   architectural test for Phases 1–3 is "does Phase 2 / Phase 3 require
   edits to `matching.rs` or `accumulate.rs`?". The same test applies
   to Phase 5: streaming must not edit either module. The streaming
   evaluator is a new orchestration layer above the spine, not a fork
   of it.
2. **The accumulator is a single-shot fold over a flat grid.**
   `accumulate()` consumes a `[K * A * I]`-shaped slice of
   `PerImageEval` and produces the `(T, R, K, A, M)` precision tensor
   in one pass. The merged-stream sort across images in each `(K, A,
   M)` cell is a stable mergesort (quirk **A1**, strict). The fold
   *can* be re-entered with new images appended, but the merged sort
   has to be redone — incremental TP/FP cumulatives are not
   composable in a way that preserves A1.
3. **The drop-in is batch.** ADR-0007's `patch_pycocotools` reproduces
   the synchronous `evaluate()` → `accumulate()` → `summarize()`
   shape, because that is the surface every downstream is calibrated
   against. Streaming is extended-API only; the drop-in doesn't gain a
   `submit()` method.

The naive design — "make `Evaluator` mutable and append predictions
between calls" — fails all three. It pulls accumulation into the hot
path, breaks the immutability contract that ADR-0006 leaned on for
thread safety, and silently changes the batch evaluator's parity
behavior depending on whether the caller appended once or many times.

ADR-0006 promised a `BackgroundEvaluator` in Phase 3 with `submit()` /
`result()` async semantics. That work was deferred to Phase 5 along
with the rest of the streaming story. This ADR is the synchronous
half: it defines the state model, the determinism contract, and the
public Python surface for the foreground streaming evaluator.
`BackgroundEvaluator` will land in a later ADR as a worker-thread
wrapper around the type defined here.

## Decision drivers

- **ADR-0005 invariant.** No edits to `crates/vernier-core/src/matching.rs`
  or `crates/vernier-core/src/accumulate.rs`. The streaming evaluator
  is a new module that *calls* those, exactly as the batch orchestrator
  in `evaluate.rs` does.
- **ADR-0006 threading.** `update()` and `snapshot()` drop the GIL on
  every non-trivial entry. The Rust core stays single-threaded internally
  in Phase 5; the worker-thread story belongs to a follow-up ADR.
- **ADR-0002 parity contract.** `finalize()` in strict mode must produce
  bit-identical stats to `Evaluator(...).evaluate(gt, dt_concatenation)`
  for any sequence of `update()` calls covering the same detection set.
  Snapshots in the middle of the stream are a new property the three-tier
  model does not address; this ADR adds one.
- **Memory budget.** Steady-state memory must be proportional to the
  total detection count × T (the IoU-threshold ladder), not to the
  cumulative size of the detection JSON. A 100k-image / ~30-DT-per-image
  workload should fit in ~1 GB of evaluator state — large but tractable;
  the JSON it replaces is ~10×–20× larger.
- **Snapshot cost.** A snapshot every N training steps must be cheap
  enough that an N=100 cadence over a 24-hour training run does not
  visibly slow training. Target: <100 ms for a 100k-image workload at
  half-stream, sub-second at full stream.
- **Crash recovery.** Evaluator state is a checkpointable artifact.
  After an OOM kill or preemption, the user reloads the state and
  continues submitting predictions; `finalize()` is unaffected.
- **No new top-level dep.** Per ADR-0001 §"Add or remove a top-level
  dependency", introducing rayon, tokio, or crossbeam-channel is its own
  ADR. This ADR adds none of them.
- **Public Python API.** Per ADR-0001 §"Affect the public API". Crosses
  the FFI boundary per ADR-0001 §"Cross the FFI boundary".

## Considered options

1. **Eager cumulative accumulation.** Each `update()` extends the
   running TP / FP cumulative arrays in every `(K, A, M)` cell.
   `snapshot()` reads the current cumulatives and integrates the PR
   curve. Cheap snapshot, but breaks A1: the merged-stream sort has
   to globally reorder DTs by score, and a DT submitted late can land
   anywhere in the cumulative — including ahead of one submitted
   earlier. Maintaining "the cumulative the batch run would have
   produced" requires re-sorting on every update or accepting that
   `finalize()` ≠ batch. Both are unacceptable.
2. **Fully lazy: store `PerImageEval`, fold on snapshot and finalize.**
   `update()` runs `match_image` and stores one `PerImageEval` per
   `(K, A, image_id)`, exactly as the batch orchestrator does. Nothing
   accumulates between calls. `snapshot()` calls `accumulate()` on the
   current store. `finalize()` calls `accumulate()` on the final store.
   Reuses the locked spine unchanged. Snapshots are bit-identical to
   "batch evaluation over the currently-submitted subset". Snapshot
   cost grows linearly with the stream — at 100k images, the fold is
   the same cost as the entire batch `accumulate()`.
3. **Hybrid: per-(K, A, image) eager, cross-image deferred.** Within
   each cell, do the per-image work eagerly during `update()`
   (matching, ignore-flag folding, score-desc sort *within the image*).
   Across images, defer the merged sort and cumulative accumulation
   until `snapshot()` / `finalize()`. The hot path is what it would
   have been under option 2 — the per-image work is identical. The
   snapshot path saves nothing measurable, because the merged sort
   *is* the cost of `accumulate()`. This is option 2 with extra
   bookkeeping; rejected.
4. **Background worker (the ADR-0006 `BackgroundEvaluator` shape now).**
   `update()` posts the GTs/DTs onto a `crossbeam-channel`, returns
   immediately; a worker thread runs `match_image` and writes to the
   `PerImageEval` store. `snapshot()` waits for queue drain. Adds a
   top-level dep, defers ADR-0001 discipline, and front-loads complexity
   we cannot justify before measuring single-threaded latency.
5. **Re-run batch every snapshot.** Each `snapshot()` and `finalize()`
   re-runs the full pipeline from raw GT/DT JSON. Trivial to implement,
   trivially correct, but defeats the purpose: the user is paying the
   batch cost on every snapshot, *plus* the cost of accumulating raw
   predictions in memory. The whole point of streaming is to amortize
   matching across the stream so it doesn't dominate end-of-epoch.

## Decision outcome

Chosen option: **Option 2 — fully lazy. Each `update()` materializes
`PerImageEval` cells via the unchanged `match_image` + B7 fold pipeline
and appends them to an internal store. `snapshot()` and `finalize()`
both call `accumulate()` over the current store.**

Option 3's incremental optimizations are deferred to a follow-up ADR
once we have profile data showing snapshot cost is actually a problem
in a real training loop. Option 4's worker-thread design lands as
`BackgroundEvaluator` in a follow-up that takes the type defined here
and wraps it.

### Rust core surface

A new module `crates/vernier-core/src/stream.rs` defines:

```rust
pub struct StreamingEvaluator<K: EvalKernel> {
    dataset: CocoDataset,
    params: EvaluateParams,
    grid_meta: EvalGridMeta,        // K, A, image-id index
    cells: PerImageEvalStore,       // sparse, keyed by (cat_idx, area_idx, image_idx)
    seen: HashSet<i64>,             // image ids that have received an update()
    n_detections: usize,            // monotonically increasing
    parity_mode: ParityMode,
    _kernel: PhantomData<K>,
}

impl<K: EvalKernel> StreamingEvaluator<K> {
    pub fn new(dataset: CocoDataset, params: EvaluateParams) -> Result<Self, EvalError>;

    pub fn update(
        &mut self,
        detections: CocoDetections,
    ) -> Result<UpdateReport, EvalError>;

    pub fn snapshot(&self) -> Result<Summary, EvalError>;

    pub fn finalize(self) -> Result<Summary, EvalError>;

    // Crash-recovery surface
    pub fn checkpoint(&self) -> Vec<u8>;
    pub fn restore(bytes: &[u8]) -> Result<Self, EvalError>;

    // Inspection — cheap, GIL-bound, no compute
    pub fn images_seen(&self) -> usize { self.seen.len() }
    pub fn detections_seen(&self) -> usize { self.n_detections }
    pub fn images_pending(&self) -> usize { /* gt - seen */ }
}
```

Two non-obvious points about this shape:

- **`finalize` consumes self.** Once finalized, no further `update`s
  are accepted. This catches the "I called `finalize`, then submitted
  a late batch, my numbers are wrong" footgun at the type system. Users
  who want a non-consuming `finalize` semantics keep the streaming
  evaluator alive and call `snapshot` — the cost is identical, the
  guarantee is the same.
- **`update` is `&mut self`.** It is *not* thread-safe to call
  concurrently. The Python wrapper makes this explicit (see below).
  Concurrent reads (`snapshot`) during a write (`update`) are
  rejected at compile time by the borrow checker; the FFI wrapper
  enforces the same at runtime.

`UpdateReport` carries diagnostics — number of new detections matched,
number of duplicate-image-id rejections, number of out-of-area DTs
flipped to ignore via B7 — and is what training loops can log to
TensorBoard between snapshots.

### Determinism contract

`finalize()` is bit-identical to a batch run over the same detection
set, in every parity mode. This is a hard invariant, asserted in the
parity harness:

```python
ev = StreamingEvaluator(gt, parity_mode="strict")
for image_id, dets in stream:
    ev.update(dets)
final = ev.finalize()

reference = Evaluator(parity_mode="strict").evaluate(gt, all_dets)
assert final.stats == reference.stats   # bit-equal
```

The invariant holds because `accumulate()` over the final store is
*the same call* the batch orchestrator makes — same inputs, same
ordering rules, same code path. The store is built incrementally but
read in one pass.

`snapshot()` mid-stream is a different beast. Its contract is:

> A snapshot is bit-identical to a batch evaluation over (the original
> GT dataset, the union of all detections submitted via `update()` so
> far) — *provided* the same detections were submitted in the same
> order. Reordering `update()` calls can produce snapshot stats that
> differ in the last few ULP at the boundaries between IoU-threshold
> tiebreaks, even when the final state of the store is identical.

This is a new property the three-tier ADR-0002 model does not name:
the existing tiers describe vernier vs. pycocotools, while this is
vernier vs. itself across stream orderings. We call it
**stream-order sensitivity**, and document it explicitly:

| Surface              | Stream-order invariant? | Bit-equals batch? |
| -------------------- | ----------------------- | ----------------- |
| `finalize()`         | yes                     | yes               |
| `snapshot()` default | no (boundary ULPs only) | yes-on-subset     |
| `snapshot(running=True)` | no                  | no — running PR   |

The default `snapshot()` calls `accumulate()` on the current store and
is bit-identical to a batch run over the submitted subset, with one
caveat: if two DTs have exactly equal scores and ended up in the store
in different orders depending on how the user batched their `update()`
calls, the merged-stream stable-sort tiebreak (A1) goes the other way.
The visible delta is bounded by `4 * f64::EPSILON` (the ADR-0004
aligned-tolerance) on the precision tensor.

In strict mode, even this tiebreak ambiguity is resolved: the FFI
wrapper assigns each detection a stream-position index at `update()`
time, and the score-desc sort breaks ties by `(score, stream_position)`
deterministically. Two snapshots from two streams that submit the
same `(image_id, detection)` pairs in the same order produce
bit-identical stats. The `stream_position` is the cost of strict
ordering — six bytes per detection — and is dropped in
`corrected` mode where input-position tiebreaks aren't reproduced
anyway.

`snapshot(running=True)` is a fast path discussed below.

### Detection identifiers

Pycocotools' `loadRes` assigns sequential `id` values to detections at
load time, in the order they appear in the JSON. The batch evaluator
inherits that numbering and threads it through `evalImgs` (`dtIds`,
`dtMatches` carrying matched DT ids). The streaming evaluator does
not have a single load step — `loadRes` runs once per `update()` call,
which means naive id assignment would have batch 2's ids collide with
batch 1's.

Detection ids are **not load-bearing for the stats**. They appear in
`evalImgs` as match indicators (non-zero in `dtMatches`) and as
identifiers when downstream tools post-hoc inspect which DT matched
which GT. They never enter the precision/recall calculation.

The streaming evaluator owns the id namespace: at `update()` time it
assigns ids monotonically from a global counter, starting from 1, in
submission order. The user does not preassign them. We do not try to
reproduce pycocotools' file-order numbering, which depends on a JSON
layout the streaming user no longer has.

The honest contract: **streaming `dtIds` are an internal sequence and
are not bit-equal to batch `dtIds` for the same detection set.**
Stats parity holds; structural `evalImgs` parity on the id columns
does not. This is consistent with the stream-order-sensitivity
property above — `dtIds` reflect submission order, which the
streaming evaluator does not claim invariance over.

Where this leaks into user-visible API is the Phase 5 Week 2
per-detection DataFrame. We document there that detection ids
round-trip *within* a streaming run (the user can join `per_detection`
against their own predictions table by id) but are not stable
between batch and streaming evaluation of the same data. A user who
needs that stability uses the batch evaluator. The training-loop
persona, who is the streaming evaluator's primary consumer, does not.

### Fast snapshot mode (`running=True`)

For training loops that snapshot every step or every few steps, the
full fold becomes the dominant eval cost. The fast path keeps a running
PR-curve approximation that is updated incrementally:

- After each `update()`, for each `(K, A, M)` cell touched, append the
  new image's TP/FP contributions to a running cumulative.
- `snapshot(running=True)` reads the cumulatives and integrates without
  re-sorting cross-image.

This produces a "running mAP" that is *not* equivalent to "batch mAP
over the current subset". A new high-score detection submitted late
appears at the end of the cumulative rather than the start, which is
where the global score-desc sort would have put it. The mAP this
yields is biased — typically optimistic for monotonically improving
models, since high-confidence late detections inflate late-stage
precision.

The contract is: `snapshot(running=True)` is a *training signal*, not
a *parity claim*. It is appropriate for "is the model getting better?"
but inappropriate for "does my model release pass the quality gate?".
The default snapshot path, or `finalize()`, answers the latter.

We ship this mode in 0.5.0 because the cost difference is real (2–3
orders of magnitude on snapshot for >50k images), and the use case is
real (training-loop logging). We document its limitations on every
public surface that exposes it.

### Memory model

Steady-state memory is bounded by:

```
mem ≈ Σ_image (Σ_cell |D_cell| · (8 + 2T))     bytes
    + |GT| · (1 + a few cell-keys)              bytes
```

For a 100k-image / 80-category / 4-area / `T=10` / ~30-DT-per-image
workload (typical COCO-scale training-time eval), this works out to
~600 MB of evaluator state — the input JSON for the same workload is
8–15 GB. The compression factor is the elimination of all string keys
(category names, image filenames), all bbox coordinates beyond what
the kernel already cached, and the JSON parsing scaffolding.

The streaming evaluator enforces a **two-stage memory budget**:

- **Soft warning at 80%** of budget. Emitted via Python's `warnings`
  module with a dedicated `vernier.MemoryBudgetWarning` subclass so
  users can either filter it out (interactive notebook use) or
  escalate to an error in CI (`warnings.simplefilter("error",
  MemoryBudgetWarning)`). Emitted at most once per evaluator
  instance.
- **Hard error at 100%** of budget. `update()` raises
  `vernier.OutOfBudgetError` carrying a structured breakdown: cells
  store size, per-image counts, score and match-flag array sizes.
  The breakdown is what tells the user whether to shard
  (recommended), reduce `iou_thresholds` (parity-breaking), lower
  `max_dets[-1]` (parity-breaking), or raise the budget explicitly.

The default budget is `min(8 GiB, system_total * 0.5)`, where
`system_total` is read once at evaluator construction (not on every
update; we cache the value and don't react to subsequent system
memory pressure). The lower bound means the same code works on a
developer's 16 GB laptop and a 256 GB CI machine without per-environment
tuning. The upper bound caps the runaway case where vernier eats half
a workstation because nobody set the budget explicitly.

Auto-scaling has one real downside worth naming: same workload,
different machine, different behavior. For the MLOps-integrator
persona, this is unacceptable in CI. The how-to guide for that
persona prescribes setting `memory_budget_bytes` to a fixed value at
construction; the same value travels with the pipeline and the same
failure mode reproduces deterministically. We document this rather
than make the default conservative enough that LVIS users break on
day one.

`UpdateReport` carries the current evaluator footprint in bytes, so
training loops can log memory growth alongside detections-seen. The
budget is an evaluator-construction parameter; it cannot be raised
mid-run (raising it on the fly would let the evaluator drift past
the limit between checkpoints, which defeats the predictability
goal).

### Crash recovery

`checkpoint()` returns a `Vec<u8>` containing:

- A magic number + version byte (so future format changes can refuse
  old checkpoints loudly).
- The `EvaluateParams` (so restore validates the dataset against the
  same params).
- The dataset hash (so restore catches "you changed the GT between
  checkpoint and restore").
- The serialized `cells` store, encoded compactly via `rkyv` (the
  zero-copy framework already in `vernier-core`'s dep tree for the
  RLE codec).
- The `seen` set and detection counter.

`restore()` validates the dataset hash against the current
`CocoDataset` and refuses to load if they differ. Common case at scale
is "the trainer crashed and resumed from the same checkpoint" — the
GT is unchanged, the streaming state is recovered, and the user
continues submitting from where the trainer left off.

The format is private. `checkpoint()` emits an opaque blob; we make no
backward-compatibility promises across vernier minor versions until
the streaming surface stabilizes. Documented as such.

### Python surface

```python
class StreamingEvaluator:
    def __init__(
        self,
        ground_truth: bytes,
        *,
        iou_type: IouType = "bbox",
        parity_mode: ParityMode = "corrected",
        max_dets: tuple[int, ...] = (1, 10, 100),
        use_cats: bool = True,
        memory_budget_bytes: int | None = None,  # None → min(8 GiB, system/2)
    ) -> None: ...

    def update(self, detections: bytes) -> UpdateReport: ...

    def snapshot(self, *, running: bool = False) -> Summary: ...

    def finalize(self) -> Summary: ...

    def checkpoint(self) -> bytes: ...

    @classmethod
    def restore(cls, ground_truth: bytes, state: bytes) -> StreamingEvaluator: ...

    @property
    def images_seen(self) -> int: ...
    @property
    def detections_seen(self) -> int: ...
    @property
    def images_pending(self) -> int: ...
    @property
    def memory_used_bytes(self) -> int: ...
    @property
    def memory_budget_bytes(self) -> int: ...
```

Two new exceptions ship in `vernier`:

```python
class MemoryBudgetWarning(UserWarning): ...

class OutOfBudgetError(RuntimeError):
    used_bytes: int
    budget_bytes: int
    breakdown: dict[str, int]   # "cells_store", "scores", "match_flags", ...
```

Three deliberate departures from `Evaluator`'s shape (per ADR-0006):

- **Mutable.** `update()` mutates the evaluator in place. `Evaluator`
  is frozen because it has no notion of "between calls"; `StreamingEvaluator`
  exists *because* it does. The two types are siblings, not subtypes.
- **Single-writer.** See *Concurrency model* below for the full rule;
  in short, the first `update()` call captures the owner thread, and
  subsequent `update()` from any other thread raises `RuntimeError`.
- **`finalize()` consumes the instance.** Mirrors the Rust shape.
  Subsequent calls on a finalized instance raise `RuntimeError("already
  finalized")`. Users who want to keep going call `snapshot()` instead.

The detections payload is bytes in the same shape as `Evaluator.evaluate`
accepts — a JSON array of detection records. We intentionally do *not*
expose a "submit one detection at a time" API at the FFI layer: the
fixed cost of crossing the FFI boundary makes per-detection submission
2–3 orders of magnitude slower than per-batch. Convenience helpers in
the Python wrapper batch up records over a configurable buffer (default:
1024 records) before crossing.

### Threading and the GIL

`update()`, `snapshot()`, `finalize()`, `checkpoint()`, and `restore()`
all wrap their compute body in `Python::detach` per ADR-0006. The
wrapped closures touch only owned Rust data — the dataset and the
cells store — so the GIL-drop is mechanical.

Concurrent Python callers using *separate* `StreamingEvaluator`
instances (one per training loop, say) interleave perfectly: the GIL
is dropped during compute, each Rust evaluator owns its own state,
no shared mutable state crosses the FFI boundary. Concurrent callers
on the *same* instance serialize on the internal lock; this is the
documented contract.

`BackgroundEvaluator` (the ADR-0006 promise) becomes a thin Rust
wrapper: a worker thread, an `mpsc::SyncSender<Detections>` channel,
a `StreamingEvaluator` owned by the worker. The Python wrapper has
`submit()` / `result()` async semantics that ADR-0006 already
specified. Nothing about the type defined in this ADR forecloses
that design — the worker just owns one of these and calls `update()`
on it. We do not implement `BackgroundEvaluator` in this ADR; we
verify the shape composes.

### Concurrency model

The streaming evaluator is **single-writer**. The first `update()`
call on an instance captures `thread::current().id()` as the
*owner thread*; subsequent `update()` calls from any other thread
raise `RuntimeError("StreamingEvaluator is single-writer; submitted
from <tid>, owned by <tid>")`. The owner is captured at first update
rather than at construction, so the common
construct-on-main-thread-then-update-on-eval-thread pattern works
without ceremony.

Read-only operations (`snapshot()`, `checkpoint()`, properties) are
unrestricted. They acquire the internal lock blocking, so concurrent
reads succeed and reads during an in-flight `update()` block until
the update completes. There is no "see partial update" race.

`finalize()` is a write — it requires the owner thread.

This restriction is deliberate. The realistic streaming use cases are
single-writer:

- A training loop submits batch by batch from one thread.
- A CI eval pass submits image by image, also from one thread.
- A robotics replay pipeline submits frame by frame from one thread.

The cases that *look* like they need multi-writer support are actually
either "many sources of predictions feeding one consumer thread that
submits in order" (use a queue, the consumer is the owner) or "many
evaluators that get merged at the end" (out of scope here, future
`merge` API). Genuine multi-writer streaming runs into the
stream-order-sensitivity wall: strict-mode tiebreaks via
`stream_position` only deterministically order *within* a single
`update()` call, not across them, and forcing global ordering on
concurrent submitters defeats the value of concurrency. We do not
ship that complexity; we refuse it at the API.

Users who genuinely need cross-thread submission build the wrapper
themselves: an `mpsc` channel, one consumer thread that owns the
evaluator, producers posting `Detections` payloads. This is the
`BackgroundEvaluator` shape from ADR-0006, and we ship it as a
follow-up. The single-writer rule on the underlying type is what
makes that wrapper trivial — the worker is by definition the owner
thread, and any `update()` from outside the worker is a programming
error that surfaces immediately rather than corrupting state silently.

### Parity harness

A new fixture set in `tests/python/parity/streaming/` covers:

- **Finalize-equals-batch.** For every fixture in `ALL_FIXTURES`,
  shard the detections randomly into 1, 4, 16, and N batches; submit
  via `update()`; `finalize()`; assert bit-equality of `stats` against
  `Evaluator(...).evaluate(gt, dt)`.
- **Order invariance of `finalize()`.** Same shard, three random
  permutations of update order; assert `finalize()` outputs are
  bit-equal across permutations in strict mode (per the
  `stream_position` rule above).
- **Snapshot equals batch on subset.** Submit half the detections;
  `snapshot()`; compare to `Evaluator(...).evaluate(gt, half_dt)`;
  assert bit-equality up to the 4-ULP aligned tolerance, exact in
  strict mode.
- **Running-mode does not equal subset.** `snapshot(running=True)`
  on the same half-stream; assert it differs from the batch-on-subset
  result by a measurable amount (the test pins the expected delta
  for a known fixture; this is a regression guard, not a correctness
  claim, for the running mode).
- **Checkpoint round-trip.** `checkpoint()` → `restore()` → continue
  → `finalize()` equals an uninterrupted run.
- **Crash mid-stream.** A fixture that submits half a stream, drops
  the evaluator, restores from checkpoint, finishes; asserts
  bit-equality against the uninterrupted run.
- **Memory budget hard cap.** A fixture with a deliberately small
  `memory_budget_bytes` that submits until `OutOfBudgetError` fires;
  asserts the breakdown is non-empty and the error happens *before*
  the system OOMs (the budget is the gate, not the kernel).
- **Owner-thread enforcement.** A fixture that submits from thread A,
  then attempts an `update()` from thread B; asserts `RuntimeError`
  and asserts thread A can keep submitting unaffected.

The harness extends the existing parity infrastructure rather than
forking it (this is *not* the boundary-IoU isolation case from
ADR-0010 — the streaming evaluator is parity-equivalent to the batch
one, by construction). New file, same disposition table.

### What this ADR explicitly does *not* decide

- **Async `BackgroundEvaluator`.** Deferred to a follow-up ADR. The
  type defined here composes into the worker design described in
  ADR-0006, but a top-level threading dep (rayon, crossbeam-channel,
  or `std::sync::mpsc` with a runtime-spawned thread) requires its
  own ADR-0001 trigger.
- **Distributed / multi-process aggregation.** The Possible Extensions
  document explicitly takes this off the table for Phase 5: users
  shard manually across processes and combine via
  `StreamingEvaluator.merge(other)` if they need it. The merge
  operation is itself out of scope for this ADR; it's a small
  follow-up that lands once we see real demand.
- **Pluggable storage backend for the cells store.** The `rkyv`
  in-memory representation is a private implementation detail. If
  someone later wants `lance` or `parquet` spillover for >100M-image
  workloads, that's a follow-up, not a v0.1 streaming feature.
- **Streaming versions of the extended-API tables (per-image,
  per-class DataFrames).** Those are Phase 5 Week 2 work that builds
  on the per-image cell store this ADR introduces. They are a pure
  addition: the cells store already has the data; the DataFrame
  surface is a fold over it. Out of scope here, in scope next week.

### Consequences

- **Positive.** The training-loop user gets the headline streaming
  capability without forcing any design changes to the Phase 1–3
  spine. ADR-0005's invariant survives Phase 5. `finalize()` is
  bit-identical to batch in strict mode — the parity story extends
  cleanly. The cells store becomes the load-bearing primitive that
  Phase 5 Week 2 (DataFrames), Week 3 (TIDE), and Week 4
  (calibration) all consume; all four features share one underlying
  data structure rather than each rebuilding it. Crash recovery is
  free given the in-memory store is already serializable.
  `BackgroundEvaluator` becomes a thin wrapper rather than a redesign.
- **Negative.** Snapshot cost grows with stream length; at full
  stream, a snapshot is the same wall-clock cost as a batch
  `accumulate()`. Users who snapshot every step on a 1M-image
  workload will eat that cost. The fast path (`running=True`) is the
  mitigation but is not a parity claim and has to be documented at
  every surface. Memory grows with the stream: ~600 MB at 100k
  images is the realistic working-set figure, ~6 GB at 1M images,
  past which the user shards across processes. Strict-mode
  stream-position tiebreaks add ~6 bytes per detection — small but
  non-zero.
- **Neutral.** The streaming evaluator is mutable; the batch
  `Evaluator` is frozen. The two types are siblings, neither
  subtype of the other. Users learning the API see two evaluators
  with two shapes; the difference is justified by the difference in
  use case. Documentation positions them as complementary rather
  than redundant.

## Pros and cons of the options

### Option 1 — eager cumulative accumulation

- 👍 Snapshot cost is constant in the stream length (just integrates
  the running cumulatives).
- 👎 Breaks A1: late high-score DTs go to the wrong place in the
  cumulative. `finalize()` would not match batch.
- 👎 Forces all parity-related logic into a new code path, which is
  exactly what ADR-0005 says we don't do.

### Option 2 (chosen) — fully lazy, fold on snapshot/finalize

- 👍 Reuses `accumulate()` unchanged. ADR-0005 invariant holds.
- 👍 `finalize()` is bit-identical to batch. Strict-mode parity story
  is one-line: "the streaming evaluator passes the same parity harness
  as the batch one".
- 👍 Cells store doubles as the foundation for per-image DataFrames,
  TIDE, calibration, and breakdowns (Phase 5 Weeks 2–5).
- 👎 Snapshot cost is O(stream-length). Large workloads + frequent
  snapshots = real cost. Mitigated by the running-mode fast path.

### Option 3 — hybrid eager/deferred

- 👍 Hot path identical to option 2.
- 👎 Snapshot path saves nothing measurable; the merged sort is the
  cost of `accumulate()`, and that's still done at snapshot time.
- 👎 Adds bookkeeping (per-cell partial state) that is read once and
  thrown away.

### Option 4 — background worker now

- 👍 Best per-update latency (returns immediately).
- 👎 Adds `crossbeam-channel` as a top-level dep without an ADR-0001
  conversation.
- 👎 Snapshot has to wait for queue drain — cost is moved, not
  removed.
- 👎 Can land later as a wrapper around option 2 with no rework.

### Option 5 — re-run batch every snapshot

- 👍 Trivially correct.
- 👎 Defeats the purpose: the user is paying batch cost on every
  snapshot *and* holding raw detection JSON in memory. This is the
  pycocotools status quo and is what the streaming evaluator exists
  to fix.

## Links and references

- ADR-0001 — Record architecture decisions (§"Affect the public API",
  §"Cross the FFI boundary", §"Add or remove a top-level dependency").
- ADR-0002 — Three-tier parity model. The streaming evaluator inherits
  the existing tiers; this ADR adds *stream-order sensitivity* as an
  orthogonal property of the snapshot surface.
- ADR-0005 — Lock the `Similarity` trait and matching-engine API for
  Phases 1–3. The architectural test ("no edits to `matching.rs` or
  `accumulate.rs`") is satisfied by the chosen option.
- ADR-0006 — Threading model. `BackgroundEvaluator` was promised in
  Phase 3; its synchronous predecessor lands here, and the worker-thread
  wrapper composes around the type defined here.
- ADR-0007 — `patch_pycocotools` policy. Streaming is extended-API only;
  the drop-in remains the synchronous batch shape.
- ADR-0004 — Numerical layout policy. The 4-ULP aligned tolerance is
  reused as the snapshot-vs-batch-on-subset bound in non-strict mode.
- ADR-0011 — Discriminated kernel config. The `K: EvalKernel` generic
  threaded through `StreamingEvaluator` is the same kernel-config value
  type defined there; streaming inherits the kernel-coupled defaults
  (e.g. ADR-0012's keypoint `max_dets`) without restating them.
- `crates/vernier-core/src/evaluate.rs` — the batch orchestrator that
  the streaming evaluator parallels (and reuses for the per-image work).
- `crates/vernier-core/src/accumulate.rs` — the fold the streaming
  evaluator calls unchanged on snapshot/finalize.
- `docs/explanation/possible-extensions.md` — the Phase 5 capability
  ranking that motivates this ADR's place in the schedule.
