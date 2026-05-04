# How to evaluate on a background thread

`BackgroundEvaluator` runs the same kernel as `StreamingEvaluator`
on a worker thread; `submit()` enqueues a batch and returns
immediately, so the calling thread (typically a training loop) does
not stall waiting for the matching kernel to finish.

## Submit and snapshot

```python
from pathlib import Path
import json
from vernier.instance import BackgroundEvaluator

gt_bytes = Path("instances_val2017.json").read_bytes()

with BackgroundEvaluator(gt_bytes, iou_type="bbox") as evaluator:
    for step, (images, _) in enumerate(val_loader):
        detections = model(images)
        evaluator.submit(json.dumps(detections).encode())

        if step % 100 == 0:
            running = evaluator.snapshot(peek=True)
            log_metrics(step, ap=running.stats[0])

    summary = evaluator.finalize()
print("final AP:", summary.stats[0])
```

The context-manager form drains the worker queue and joins the
thread on exit. Without it, call `evaluator.finalize()` (which
also drains and joins).

- **`submit(bytes, *, timeout=None)`** enqueues a batch; returns
  immediately on success or raises `QueueFullError` if the queue
  is at capacity (default queue size is set at construction via
  `queue_capacity=`).
- **`snapshot(peek=True)`** returns the current Summary against
  everything consumed so far without blocking the worker;
  `peek=False` waits for outstanding batches to drain first.
- **`finalize()`** drains the queue, finishes evaluation, and
  shuts the worker down. The returned Summary is canonical.

## When to use it vs `StreamingEvaluator`

| Scenario | Pick |
|---|---|
| The kernel does not stall the training loop. | `StreamingEvaluator`. Simpler; no thread to manage. |
| Each `update(...)` call adds visible latency. | `BackgroundEvaluator`. Frees the calling thread. |
| You need exact per-step timing of the kernel. | `StreamingEvaluator`. Background timing depends on queue scheduling. |
| You log metrics every N steps, not every step. | Either. `StreamingEvaluator` is the lower-overhead default. |

In a typical PyTorch training loop on a single GPU, the GPU is the
bottleneck and `StreamingEvaluator` adds negligible CPU stall time.
`BackgroundEvaluator` is the right choice when the validation
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

`BackgroundEvaluator` honors the same `memory_budget_bytes=` knob
as `StreamingEvaluator`. Exceeding the budget surfaces as an
`OutOfBudgetError` from the calling thread on the next `submit`,
not silently from the worker.

## See also

- [ADR-0013](../adr/0013-streaming-evaluator.md) — the streaming
  surface design; explains the snapshot/finalize split and the
  parity contract for streaming-vs-batch numbers.
- [`tutorials/training-loop.md`](../tutorials/training-loop.md) —
  the streaming counterpart, with the W&B / TensorBoard logger
  sketch. Same Summary shape; same call style minus the worker
  thread.
