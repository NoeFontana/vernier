# Why `per_image` does not ship an AP column

The `per_image` result table introduced by [ADR-0019](../adr/0019-result-tables.md)
deliberately omits `ap` and `ap_50` columns. Pycocotools-cli and
faster-coco-eval both expose a per-image AP value; vernier does not.
This page records why — long enough that an issue asking "where is
per-image AP?" can be closed by a link.

## AP integrates a curve that a single image cannot draw

Average precision is the area under a precision-recall curve, swept
over a sorted list of detections. The shape of the curve is the
contribution. A *dataset-level* PR curve has thousands of points and
is a meaningful signal.

A *single-image* PR curve almost never does. Most images contribute
fewer than ten detections and one or two ground truths. The resulting
curve is a step function with one or two points, and its area is
either 0.0 or 1.0 in the overwhelming majority of cells. The number
is precisely defined; it just is not comparable across images,
because images with three detections and images with three hundred
produce the same statistic via incompatible curves.

## Faster-coco-eval and pycocotools-cli surface this anyway

Both tools compute per-image AP via the same `accumulate` machinery
used for the dataset summary, applied to one image's detections. The
output is a one-cell number that users plot, sort, and act on. The
most common artifact is a histogram piled at exactly 0.0 and 1.0
with an uninformative tail. Users then set thresholds, mine "hard
images", and chase phantom regressions on a metric that has no
statistical meaning at that granularity. We do not ship the column
because we do not want to encode a default that misleads.

## What the project ships instead

The `per_image` schema (see ADR-0019 §`per_image`) reports raw counts —
`tp_at_50`, `fp_at_50`, `fn_at_50` and their `_75` siblings — plus
`tp_mean_iou`. These are well-defined per-image, comparable across
images of different sizes, and direct inputs to the failure-mining
queries the table exists to enable. Users who genuinely want
per-image PR curves have the data: `per_pair` carries IoU values
keyed by `(detection_id, ground_truth_id, image_id)`, and the curve
for a single image at IoU 0.50 is a 5-line polars expression:

```python
import polars as pl

pairs = result.per_pair.filter(pl.col("image_id") == 42)
matches = pairs.filter(pl.col("iou") >= 0.5)
# Sort by score elsewhere (per_detection) to draw the curve.
```

This is the same composition we ask of users elsewhere: vernier
ships the data primitives; user-defined metrics layer on top.

## The dataset-level analog

Even the dataset-level pycocotools layout has the same problem in
miniature. Quirk **C5** (see
[`docs/engineering/pycocotools-quirks.md`](../engineering/pycocotools-quirks.md))
records that a category with no ground truths is initialized to `-1`
and *filtered out* of the mean rather than counted as zero. Without
that filter, an empty category would drag the headline AP to
nonsense. The `per_image` AP omission is the same discipline applied
one level deeper.

## See also

- [ADR-0019](../adr/0019-result-tables.md) §`per_image` — schema
  decision, including the explicit list of columns omitted.
- [`docs/engineering/pycocotools-quirks.md`](../engineering/pycocotools-quirks.md)
  §C5 — the absent-category sentinel that motivates the same
  discipline at dataset scope.
- [`docs/how-to/result-tables.md`](../how-to/result-tables.md) —
  recipes that use `per_image` counts and `per_pair` IoUs in lieu of
  a per-image AP column.
