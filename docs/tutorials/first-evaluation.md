# Your first evaluation

This tutorial runs vernier end-to-end on COCO val2017 and prints the
12-line pycocotools-shaped summary. By the end you will have a
working evaluation pipeline, know how to read the output, and know
where to go next.

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

## Run it

<!-- needs-coco -->
```python
from pathlib import Path
from vernier.instance import Bbox, Dataset, Evaluator

gt_bytes = Path("instances_val2017.json").read_bytes()
dt_bytes = Path("detections.json").read_bytes()

dataset = Dataset.from_json(gt_bytes)
summary = Evaluator(iou=Bbox()).evaluate(dataset, dt_bytes)

for line in summary.pretty_lines():
    print(line)
print("AP =", summary.stats[0])
```

That's the full evaluation. On a 2024-era laptop this completes in
a few seconds; the bottleneck is JSON parsing on the GT side, not
the matching kernel.

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

- **Drill into the numbers.** Pass `tables="all"` to
  `evaluate(...)` to get per-image / per-class / per-detection /
  per-pair polars DataFrames. Recipe:
  [`how-to/result-tables.md`](../how-to/result-tables.md).
- **Switch the IoU kernel.** `iou=Segm()` for instance-mask AP,
  `iou=Boundary()` for boundary IoU (ADR-0010), `iou=Keypoints()`
  for OKS (ADR-0012). Recipes:
  [`how-to/boundary-iou.md`](../how-to/boundary-iou.md),
  [`how-to/keypoints-oks.md`](../how-to/keypoints-oks.md).
- **Run continuously during training.** `StreamingEvaluator`
  accepts predictions in batches and yields a running summary —
  you can log AP every N steps without re-evaluating from scratch.
  Tutorial: [`training-loop.md`](training-loop.md).
- **Migrate from a competing tool.** The TL;DR table in
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
  ADR-0002 has the three-tier rationale.
- **Custom IoU kernels or category folds.** vernier's kernels are
  the `Bbox` / `Segm` / `Boundary` / `Keypoints` discriminated
  union (ADR-0011). Adding a new kernel is an ADR-level decision,
  not a configuration knob.
