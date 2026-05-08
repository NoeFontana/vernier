# Logging AP during training

Most users want `Evaluator(...).evaluate(...)` at end-of-epoch — that
path is fine, has no streaming overhead, and is the default in every
migration guide. This tutorial covers the smaller case where you
specifically need a *running* AP mid-epoch: a long validation pass you
want to log every N steps, or a smoke check that fires before the last
batch lands. `StreamingEvaluator`
([ADR-0013](../adr/0013-streaming-evaluator.md)) accepts predictions in
batches and yields a running summary at any point.

If your need is multi-rank distributed eval (rank-local + gather), that
is a separate topic — see
[`how-to/distributed-eval.md`](../how-to/distributed-eval.md).

## Construct it

`StreamingEvaluator` lives at `vernier.instance.StreamingEvaluator`:

```python
from pathlib import Path
from vernier.instance import StreamingEvaluator

gt_bytes = Path("instances_val2017.json").read_bytes()
evaluator = StreamingEvaluator(
    gt_bytes,
    iou_type="bbox",
    parity_mode="corrected",
)
```

The constructor takes the GT JSON once. Subsequent calls feed it
detections; the GT side is parsed and held internally for the lifetime
of the evaluator.

## Feed batches and log periodically

```python
import json

for step, (images, targets) in enumerate(val_loader):
    detections = model(images)  # list of dicts: {image_id, category_id, bbox, score}
    evaluator.update(json.dumps(detections).encode())

    if step % 100 == 0:
        snapshot = evaluator.snapshot(running=True)
        log_metrics(step, ap=snapshot.stats[0], ap50=snapshot.stats[1])

final = evaluator.finalize()
log_metrics(step, ap=final.stats[0], ap50=final.stats[1], final=True)
```

Three things to know:

- **`update(bytes)`** appends a batch of detections. The argument is the
  same JSON shape `evaluate(...)` accepts — a list of detection dicts,
  encoded as bytes. Returns a small dict with running counts (images
  and detections seen so far).
- **`snapshot(running=True)`** computes a Summary against everything
  consumed so far without finalizing — cheap to call from the logging
  hook. `running=False` is the post-finalize shape (matches what
  `finalize()` returns); use it after the last `update` if you want
  canonical numbers without closing the evaluator.
- **`finalize()`** produces the canonical end-of-epoch Summary and
  closes the evaluator (no further `update` calls accepted).

## Logger integration

The Summary maps cleanly into key/value loggers (W&B, TensorBoard,
MLflow, comet, plain stdout). Pull from `summary.stats` by index or
parse `summary.pretty_lines()`:

```python
def log_metrics(step, **stats):
    # Example: replace this body with your logger of choice.
    # wandb.log({f"val/{k}": v for k, v in stats.items()}, step=step)
    # writer.add_scalar("val/ap", stats["ap"], step)
    print(step, stats)
```

The example deliberately does not import a specific logger — vernier
takes no dependency on W&B, TensorBoard, or any training framework.
Wire your logger of choice; the contract is `stats[i]` being a plain
Python `float`.

## Memory budget

Streaming holds the matched-pair grid in memory; on full COCO val2017
that is ~hundreds of MB at peak. The constructor accepts a
`memory_budget_bytes=` knob; exceeding the budget raises a typed
`OutOfBudgetError` rather than silently spilling. Memory state is
queryable:

```python
print(evaluator.memory_used_bytes, "/", evaluator.memory_budget_bytes)
print(evaluator.images_seen, evaluator.detections_seen)
```

Running on COCO val2017 with the default kernel and no budget override
keeps under 1 GB; LVIS-scale workloads warrant the explicit budget.

## When to use `BackgroundEvaluator` instead

`StreamingEvaluator` runs in the same thread as `update()` calls. For
training loops where the kernel work measurably stalls the main thread,
`BackgroundEvaluator` runs the same kernel on a worker thread;
`submit(detections)` enqueues and returns immediately. Recipe:
[`../how-to/background-evaluator.md`](../how-to/background-evaluator.md).
