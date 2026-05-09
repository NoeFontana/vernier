# How to evaluate on a background thread

`BackgroundEvaluator` runs the evaluation kernel on a worker thread;
`submit()` enqueues a batch and returns immediately, so the calling
thread (typically a training loop) does not stall waiting for the
matching kernel to finish.

## Submit and finalize

```python
from pathlib import Path
import json
from vernier.instance import BackgroundEvaluator

gt_bytes = Path("instances_val2017.json").read_bytes()

with BackgroundEvaluator(gt_bytes, iou_type="bbox") as evaluator:
    for images, _ in val_loader:
        detections = model(images)
        evaluator.submit(json.dumps(detections).encode())
    summary = evaluator.finalize()
print("final AP:", summary.stats[0])
```

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

`BackgroundEvaluator` is a single-finalize contract by design (per
ADR-0035): it does not expose a mid-epoch snapshot path. If you need
a periodic AP readout during validation, accumulate detections and
call `Evaluator.evaluate(gt, accumulated_dt)` every N steps — the
recipe in [`tutorials/training-loop.md`](../tutorials/training-loop.md)
shows the pattern.

## When to use it vs `Evaluator.evaluate`

| Scenario | Pick |
|---|---|
| End-of-epoch evaluation only. | `Evaluator.evaluate(gt, dt)`. Simplest path; no thread to manage. |
| Each `evaluate(...)` call adds visible latency to the training loop. | `BackgroundEvaluator`. Frees the calling thread; same kernel. |
| You log metrics every N steps. | `Evaluator.evaluate(gt, accumulated_dt)` from the loop, or `BackgroundEvaluator` plus an end-of-epoch finalize — see the training-loop tutorial. |
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

## See also

- [ADR-0014](../adr/0014-background-evaluator.md) — the worker-thread
  resource discipline (single worker, bounded queue, GIL drop).
- [ADR-0035](../adr/0035-api-surface-consolidation.md) — why the
  public surface is `submit` / `finalize` / `finalize_to_partial`
  with no snapshot path.
- [`tutorials/training-loop.md`](../tutorials/training-loop.md) —
  end-to-end training-loop recipe with logger integration.
- [`distributed-eval.md`](distributed-eval.md) — multi-rank pattern.
