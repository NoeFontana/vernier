# How to evaluate on a background thread

`BackgroundEvaluator` runs the evaluation kernel on a worker thread;
`submit()` enqueues a batch and returns immediately, so the calling
thread (typically a training loop) does not stall waiting for the
matching kernel to finish.

## Submit and finalize

```python
from pathlib import Path
import json
from vernier.instance import Bbox, CocoDataset, Evaluator

gt = CocoDataset.from_json(Path("instances_val2017.json").read_bytes())
evaluator = Evaluator(iou=Bbox())

with evaluator.background(gt) as bg:
    for images, _ in val_loader:
        detections = model(images)
        bg.submit(json.dumps(detections).encode())
    summary = bg.finalize()
print("final AP:", summary.stats[0])
```

`Evaluator.background(gt)` carries the evaluator's `iou` /
`parity_mode` / `max_dets` / `use_cats` / `cast_inputs` onto the
worker thread; passing a `CocoDataset` reuses the parsed-once GT and
its per-kernel derivation cache (ADR-0020) — meaningful on segm,
load-bearing on boundary IoU. If your harness already holds GT JSON
bytes and you do not reuse the dataset, you can also construct
directly: `BackgroundEvaluator(gt_bytes, iou_type="bbox")`.

The context-manager form drains the worker queue and joins the
thread on exit. Without it, call `evaluator.finalize()` directly
(which also drains and joins).

- **`submit(detections, *, timeout=None)`** enqueues a batch (either
  loadRes-shaped JSON bytes or a `Detections` dict / sequence of
  dicts — see [array ingest](array-ingest.md)); returns immediately on
  success or raises `QueueFullError` if the queue is at capacity
  (default queue size is set at construction via `queue_capacity=`).
- **`finalize()`** drains the queue, finishes evaluation, and shuts
  the worker down. The returned Summary is canonical. Subsequent
  calls raise.
- **`finalize_with_tables(...)`** is the tables-aware variant; same
  drain-and-join semantics.
- **`finalize_to_partial()`** drains, serializes the worker's final
  state as a partial blob, and shuts down. Combine with
  `Evaluator.from_partials(...)` on the head rank for distributed
  evaluation — see [`distributed-eval.md`](distributed-eval.md).

## When to use it vs `Evaluator.evaluate`

| Scenario | Pick |
|---|---|
| End-of-epoch evaluation only. | `Evaluator.evaluate(gt, dt)`. Simplest path; no thread to manage. |
| Each `evaluate(...)` call adds visible latency to the training loop. | `BackgroundEvaluator`. Frees the calling thread; same kernel. |
| Multi-rank distributed eval. | `Evaluator.evaluate_to_partial` per rank + `Evaluator.from_partials` on the head, or the `BackgroundEvaluator` variant if eval is in-loop. See [`distributed-eval.md`](distributed-eval.md). |

In a typical PyTorch training loop on a single GPU, the GPU is the
bottleneck and a plain `Evaluator.evaluate` call at end-of-epoch is
fine. `BackgroundEvaluator` is the right choice when the validation
batch size is large enough that JSON-encoding and matching show up
in the profiler.

## Queue capacity and back-pressure

The queue is bounded. If `submit` is called faster than the worker
can drain, the queue fills and `submit` raises `QueueFullError`
(or blocks until `timeout` expires when `timeout=` is set):

```python
try:
    evaluator.submit(detections, timeout=0.5)
except QueueFullError as e:
    log_metrics(step, dropped_batch=True, queue_capacity=e.queue_capacity)
```

Sizing: the queue absorbs bursts where `submit` runs faster than
the worker drains. `2-4` is the safe default; raise it only if the
profiler shows `submit` blocking on a full queue under your actual
batch cadence.

## Memory budget

`BackgroundEvaluator` honors the `memory_budget_bytes=` knob.
Exceeding the budget surfaces as an `OutOfBudgetError` from the
calling thread on the next `submit`, not silently from the worker.

## Inner parallelism with `num_threads`

The default `BackgroundEvaluator` runs a single dedicated worker
thread — ADR-0014's resource discipline, picked to keep eval from
preempting the trainer. That default is the right choice when the
training loop is competing for the same N cores.

For the *dedicated-validation-pass* persona — `val_loader` driving
inference (typically GPU-bound) with no trainer in the same loop —
the single-core bound is the bottleneck. Opt in with
`num_threads=N` (ADR-0047):

```python
# Training-loop validation: keep the trainer's cores free.
ev.background(gt)                       # default: one worker, no rayon pool

# Dedicated val-loader: spend the idle CPUs.
ev.background(gt, num_threads=8)        # per-worker rayon pool, drain-batched
```

Under the hood the worker builds a scoped `rayon::ThreadPool` of
`N` threads, drains pending submissions from the channel into a
batch, and dispatches them through the paradigm's parallel kernel
(`update_parsed_parallel` for instance / semantic / panoptic) under
`pool.install`. The pool is owned by the worker and drops when
the worker exits — no global state leaks between
`BackgroundEvaluator` instances.

`num_threads=None` or `1` keeps the pre-0.0.5 single-image worker
path byte-for-byte. The submit-blocking semantics, queue capacity,
back-pressure, and memory-budget contract are all preserved.
`VERNIER_NUM_THREADS` is honored as the env-var equivalent.

Strict-mode bit-equality across thread counts is guaranteed by
construction — see [ADR-0047](../adr/0047-threading-model.md) §"Strict
mode" for the per-paradigm reasoning (instance / panoptic re-sort by
`image_id` before the f64 fold; semantic accumulation is u64-additive).

The panoptic background path is special-cased for end-to-end
performance: when `num_threads > 1`, `submit_png` ships raw PNG
bytes through the channel as zero-copy
`pyo3::pybacked::PyBackedBytes` so libpng decode runs in parallel
inside the worker pool. Single-threaded `submit_png` keeps inline
decode (producer/consumer overlap against the worker), matching
today's wall time exactly.

## See also

- [ADR-0014](../adr/0014-background-evaluator.md) — the worker-thread
  resource discipline (single worker, bounded queue, GIL drop).
- [ADR-0035](../adr/0035-api-surface-consolidation.md) — why the
  public surface is `submit` / `finalize` / `finalize_to_partial`
  with no snapshot path.
- [`tutorials/first-evaluation.md`](../tutorials/first-evaluation.md) —
  the in-loop walkthrough (Path B), end-to-end on COCO val2017.
- [ADR-0047](../adr/0047-threading-model.md) — opt-in
  `num_threads` parallelism, the trainer-vs-val-loader persona
  split, and the strict-mode bit-equality contract.
- [`distributed-eval.md`](distributed-eval.md) — multi-rank pattern.
