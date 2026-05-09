# Logging AP during training

Most users want `Evaluator(...).evaluate(...)` at end-of-epoch — that
path is fine, has no streaming overhead, and is the default in every
migration guide. This tutorial covers the smaller case where you
specifically need a periodic AP readout mid-epoch: a long validation
pass you want to log every N steps, or a smoke check that fires before
the last batch lands. `BackgroundEvaluator`
([ADR-0014](../adr/0014-background-evaluator.md)) runs the kernel on a
worker thread; `submit(detections)` enqueues and returns immediately,
keeping the training loop unblocked.

If your need is multi-rank distributed eval (rank-local + gather), that
is a separate topic — see
[`how-to/distributed-eval.md`](../how-to/distributed-eval.md).

## Construct it

`BackgroundEvaluator` lives at `vernier.instance.BackgroundEvaluator`:

```python
from pathlib import Path
from vernier.instance import BackgroundEvaluator

gt_bytes = Path("instances_val2017.json").read_bytes()
evaluator = BackgroundEvaluator(
    gt_bytes,
    iou_type="bbox",
    parity_mode="corrected",
)
```

The constructor takes the GT JSON once. Subsequent `submit` calls feed
it detections from the training thread; the worker thread does the
kernel work without holding the GIL.

## Feed batches and finalize at end-of-epoch

```python
import json
from vernier.instance import BackgroundEvaluator, Evaluator, Bbox

with BackgroundEvaluator(gt_bytes, iou_type="bbox") as evaluator:
    for images, targets in val_loader:
        detections = model(images)  # list of dicts: {image_id, category_id, bbox, score}
        evaluator.submit(json.dumps(detections).encode())
    final = evaluator.finalize()

log_metrics(ap=final.stats[0], ap50=final.stats[1])
```

Two things to know:

- **`submit(detections)`** enqueues a batch on the worker thread. The
  argument is the same JSON shape `evaluate(...)` accepts — a list of
  detection dicts, encoded as bytes. Returns immediately; the queue
  defaults to capacity 8 batches.
- **`finalize()`** drains the queue, joins the worker, and produces the
  canonical end-of-epoch Summary. Subsequent calls raise.

## Mid-epoch readout

`BackgroundEvaluator` is a single-finalize contract — it does not
expose a mid-epoch snapshot path. If you need a periodic AP readout
during validation, accumulate detections in a list and call
`Evaluator.evaluate(gt, accumulated_dt)` every N steps:

```python
from vernier.instance import Evaluator, Bbox

batch_evaluator = Evaluator(iou=Bbox(), parity_mode="corrected")
accumulated = []

for step, (images, targets) in enumerate(val_loader):
    detections = model(images)
    accumulated.extend(detections)

    if step % 100 == 0:
        snapshot = batch_evaluator.evaluate(gt_bytes, json.dumps(accumulated).encode())
        log_metrics(step, ap=snapshot.stats[0], ap50=snapshot.stats[1])

final = batch_evaluator.evaluate(gt_bytes, json.dumps(accumulated).encode())
log_metrics(step, ap=final.stats[0], ap50=final.stats[1], final=True)
```

This is unbiased (it's the same code path as end-of-epoch) but pays
the full re-eval cost every N steps — fine for COCO val2017 every
100 steps, probably not for LVIS-scale every 10 steps.

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

The background worker holds the matched-pair grid in memory; on full
COCO val2017 that is ~hundreds of MB at peak. The constructor accepts
a `memory_budget_bytes=` knob; exceeding the budget surfaces a typed
`OutOfBudgetError` from `submit()` rather than silently spilling.

Running on COCO val2017 with the default kernel and no budget override
keeps under 1 GB; LVIS-scale workloads warrant the explicit budget.
