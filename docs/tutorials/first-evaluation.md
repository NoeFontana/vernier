# Your first evaluation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NoeFontana/vernier/blob/main/docs/tutorials/notebooks/colab_smoke.ipynb)

This tutorial runs vernier end-to-end on COCO val2017 in the two
shapes most users start with:

- **Path A — one-shot evaluation from JSON.** Predictions are already
  serialized to a file. Used for end-of-epoch checkpoints, CI gates,
  and post-training inspection.
- **Path B — during a training loop.** A `val_loader` produces
  predictions step by step; the kernel runs on a worker thread so the
  training thread does not stall.

Both paths produce the same 12-line `pycocotools`-shaped summary —
pick the one that matches your harness.

## Prerequisites

```sh
pip install vernier
```

You also need two JSON files:

- **Ground truth**: `instances_val2017.json` from
  [cocodataset.org/#download](https://cocodataset.org/#download)
  ("2017 Train/Val annotations [241MB]"). Extract `annotations/`
  from the archive; we want the `instances_val2017.json` member,
  ~7000 categories × 5000 images.
- **Predictions**: a detector's output in the COCO results format
  (`[{"image_id":..., "category_id":..., "bbox":..., "score":...}, ...]`).
  Any model trained on COCO works; if you do not have one handy,
  the COCO test server's
  [example detection submissions](https://cocodataset.org/#detection-eval)
  page links to public per-image JSON outputs from past challenges.

These files are not redistributable through this site (COCO terms of
use); download them locally before continuing.

## Path A — one-shot evaluation from JSON

<!-- needs-coco -->
```python
from pathlib import Path
from vernier.instance import Bbox, CocoDataset, Evaluator

gt_bytes = Path("instances_val2017.json").read_bytes()
dt_bytes = Path("detections.json").read_bytes()

dataset = CocoDataset.from_json(gt_bytes)
summary = Evaluator(iou=Bbox()).evaluate(dataset, dt_bytes)

for line in summary.pretty_lines():
    print(line)
```

That is the full evaluation. On a 2024-era laptop it completes in a
few seconds; the bottleneck is JSON parsing on the GT side, not the
matching kernel. `CocoDataset.from_json(gt_bytes)` parses the GT
once so subsequent `evaluate(...)` calls reuse the cache
([ADR-0020](../adr/0020-parsed-once-dataset-handle.md)).

## Path B — during a training loop

`BackgroundEvaluator`
([ADR-0014](../adr/0014-background-evaluator.md)) runs the matching
kernel on a worker thread; `submit(detections)` enqueues a batch and
returns immediately, so the training thread is not blocked.
`Evaluator.background(gt)` is the recommended way to construct one —
it carries the evaluator's `iou` / `parity_mode` configuration onto
the worker thread, and accepts a `CocoDataset` so the GT-side
derivation cache (ADR-0020) is reused across every epoch's
`submit()` rounds.

```python
from pathlib import Path
import json
from vernier.instance import Bbox, CocoDataset, Evaluator

gt = CocoDataset.from_json(Path("instances_val2017.json").read_bytes())
evaluator = Evaluator(iou=Bbox())

for epoch in range(num_epochs):
    with evaluator.background(gt) as bg:
        for images, _ in val_loader:
            detections = model(images)  # list[{image_id, category_id, bbox, score}]
            bg.submit(json.dumps(detections).encode())
        summary = bg.finalize()
    print(f"epoch {epoch}: AP =", summary.stats[0], "AP50 =", summary.stats[1])
```

For boundary IoU (`Evaluator(iou=Boundary())`) the `CocoDataset` path
collapses the per-epoch GT-band derivation cost (~36k Chebyshev
erodes on val2017) from O(epochs) to O(1); for bbox/keypoints the
benefit is just the one-shot JSON parse. If you don't reuse the GT
across epochs, pass the bytes directly to
`BackgroundEvaluator(gt_bytes, iou_type="bbox")` and the constructor
takes the same one-shot path it always has.

`submit(detections)` accepts the same loadRes-shaped JSON bytes that
`evaluate(...)` accepts and blocks if the bounded worker queue is
full; pass `timeout=` to surface back-pressure as a typed
`QueueFullError` instead. The full API surface — queue sizing,
memory budget, array-form ingest, multi-rank — lives in
[`how-to/background-evaluator.md`](../how-to/background-evaluator.md).

## What the output means

`pretty_lines()` returns the 12-line block pycocotools'
`COCOeval.summarize()` prints to stdout. Each line maps to one cell
of `summary.stats`:

| Index | Stat | Reads as |
|---|---|---|
| 0 | `AP` | mean over IoU thresholds 0.50:0.05:0.95, all categories, area="all", maxDets=100 |
| 1 | `AP50` | same fold, IoU=0.50 only |
| 2 | `AP75` | same fold, IoU=0.75 only |
| 3-5 | `APs / APm / APl` | small / medium / large area buckets |
| 6-8 | `AR1 / AR10 / AR100` | average recall at 1 / 10 / 100 maxDets |
| 9-11 | `ARs / ARm / ARl` | per-area-bucket AR at maxDets=100 |

The full per-stat definitions live in
[`reference/coco-summary-stats.md`](../reference/coco-summary-stats.md).
Empty buckets surface as `-1.0` (quirk **C5**); see
[`migrate/from-pycocotools.md`](../migrate/from-pycocotools.md#sentinels-empty-buckets-are-10)
for the cross-codebase comparison.

## Where to go next

- **Tune the evaluator.** `parity_mode`, `max_dets`, `use_cats`,
  `cast_inputs`, the GT-handle reuse pattern, and how they compose:
  [`how-to/configure-evaluator.md`](../how-to/configure-evaluator.md).
  Custom IoU / recall / area grids (ADR-0040) live in
  [`how-to/custom-evaluation-grids.md`](../how-to/custom-evaluation-grids.md).
- **Drill into the numbers.** Pass `tables="all"` to
  `evaluate(...)` to get per-image / per-class / per-detection /
  per-pair polars DataFrames. Recipe:
  [`how-to/result-tables.md`](../how-to/result-tables.md).
- **Switch the IoU kernel.** `iou=Segm()` for instance-mask AP,
  `iou=Boundary()` for boundary IoU (ADR-0010), `iou=Keypoints()`
  for OKS (ADR-0012). Recipes:
  [`how-to/boundary-iou.md`](../how-to/boundary-iou.md),
  [`how-to/keypoints-oks.md`](../how-to/keypoints-oks.md).
- **Tune `BackgroundEvaluator`.** Queue capacity, memory budget,
  array-form ingest, and the back-pressure contract:
  [`how-to/background-evaluator.md`](../how-to/background-evaluator.md).
- **Run multi-rank.** `Evaluator.evaluate_to_partial` per rank +
  `Evaluator.from_partials` on the head:
  [`how-to/distributed-eval.md`](../how-to/distributed-eval.md).
- **Migrate from other tools.** The TL;DR table in
  [`migrate/from-pycocotools.md`](../migrate/from-pycocotools.md)
  maps the `COCOeval(...).evaluate().accumulate().summarize()`
  call sequence onto vernier's `Evaluator(...).evaluate(...)`.
- **Diagnose a regression.** TIDE decomposes the gap between
  measured AP and a perfect-matching upper bound into six
  interpretable bins. Tutorial:
  [`debugging-with-tide.md`](debugging-with-tide.md).

## What this tutorial does NOT cover

- **Building a Dataset from arrays.** This tutorial assumes COCO
  JSON in, COCO JSON out. For programmatic dataset construction
  (no JSON), see the `Dataset.from_arrays` constructors documented
  on the API reference page.
- **Strict pycocotools parity.** The default `parity_mode` is
  `"corrected"` — vernier applies its documented quirk fixes (see
  [`engineering/pycocotools-quirks.md`](../engineering/pycocotools-quirks.md)).
  For bit-exact pycocotools numbers, pass `parity_mode="strict"`.
  ADR-0002 has the strict-vs-corrected rationale.
- **Custom IoU kernels or category folds.** vernier's kernels are
  the `Bbox` / `Segm` / `Boundary` / `Keypoints` discriminated
  union (ADR-0011). Adding a new kernel is an ADR-level decision,
  not a configuration knob.
