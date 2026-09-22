# How to compute COCO metrics inside a training loop

**If you evaluate from files, you need none of this.** A COCO ground
truth on disk and a detections JSON go straight into
[`Evaluator`](../reference/python/instance.md), and
[Ingest detection arrays](array-ingest.md) covers the case where the
ground truth is a file but the detections are already tensors.

This page is for the other case: **both sides live in memory**, as one
record per image or as columns a metric object accumulated, and there is
no COCO file anywhere. `vernier.adapters` converts that state into
vernier's evaluation inputs, so the trainer does not have to.

vernier imports no framework to do it. Arrays are read through DLPack or
`numpy.asarray`, and a device tensor is moved by duck-typed `.detach()` /
`.cpu()` — so torch, JAX and plain numpy all work, and none of them is a
dependency. ADR-0063 is the record; ADR-0031 is the promise it keeps.

## A framework-free loop

`coco_metrics` is the one-call form. Give it one record per image on each
side:

```python
from vernier.adapters import coco_metrics

predictions, targets = [], []
for images, batch_targets in val_loader:
    out = model(images)
    for boxes, scores, labels in zip(out.boxes, out.scores, out.labels):
        predictions.append({"boxes": boxes, "scores": scores, "labels": labels})
    for target in batch_targets:
        targets.append({"boxes": target["boxes"], "labels": target["labels"]})

metrics = coco_metrics(predictions, targets, box_format="xyxy")
print(metrics["map"], metrics["map_50"], metrics["mar_100"])
```

`predictions[i]` and `targets[i]` describe the same image. The keys are
the conventional ones the wider ecosystem logs — `map`, `map_50`,
`map_75`, `map_small` / `medium` / `large`, `mar_{n}` at each entry of
`max_dets`, and `mar_small` / `medium` / `large`. Values are Python
floats and numpy arrays; `torch.as_tensor(...)` converts them zero-copy
if you want tensors.

`box_format` is **never** auto-detected. An `(N, 4)` array is ambiguous
between `xywh` and `xyxy`, and guessing wrong yields plausible, wrong AP
rather than an error.

## When the state is already concatenated

A TorchMetrics-shaped metric does not hold per-image records — it holds
one big column per field plus the per-image counts. Splitting that back
into records only to have vernier concatenate it again costs a
Python-level pass per image per field, and at validation scale it is the
most expensive thing on the path.

`coco_inputs_from_columns` takes the columns whole:

```python
import numpy as np
from vernier.adapters import coco_inputs_from_columns
from vernier.instance import Evaluator, Bbox

detections = {
    "boxes": np.concatenate(per_image_boxes),
    "scores": np.concatenate(per_image_scores),
    "labels": np.concatenate(per_image_labels),
    "counts": np.array([len(s) for s in per_image_scores]),
}
targets = {
    "boxes": np.concatenate(gt_boxes),
    "labels": np.concatenate(gt_labels),
    "area": np.concatenate(gt_areas),
    "counts": np.array([len(l) for l in gt_labels]),
}

ground_truth, dt = coco_inputs_from_columns(detections, targets)
summary = Evaluator(iou=Bbox()).evaluate(ground_truth, dt)
print(summary.stats[0])
```

`counts` is what assigns rows to images — entry `i` is image `i`'s row
count — so the columns themselves need no grouping. Both `counts` columns
have one entry per image, and each must sum to its side's row count.

It is the same conversion as `coco_inputs`: both spellings land on one
builder, and the test suite pins them to the identical
`CocoDataset.dataset_hash`. Use whichever matches the state you already
have; do not synthesize the other.

## Segmentation

Pass masks and a `segm` grid can read the inputs. Either spelling is
accepted, because the two populations differ — a metric object's state at
`compute()` time is usually already RLE, while a plain loop holds
bitmasks:

```python
targets.append({
    "boxes": target["boxes"],
    "labels": target["labels"],
    "masks": target["masks"],          # (N, H, W) bitmasks
})
# ...or, already encoded:
targets.append({
    "boxes": target["boxes"],
    "labels": target["labels"],
    "rles": [{"size": (h, w), "counts": counts}, ...],
})

metrics = coco_metrics(predictions, targets, iou_type=("bbox", "segm"))
print(metrics["bbox_map"], metrics["segm_map"])
```

**There is no `iou_type` on the builders** — the masks decide what the
inputs can be read as, and the IoU type is named once, where a kernel is
actually chosen. One conversion therefore serves both passes of the run
above.

`boxes` are optional when masks are present: no segm kernel reads them,
so an instance-segmentation pipeline need not materialize boxes it does
not have. A *bbox* pass does read them, so `coco_metrics` refuses
`iou_type="bbox"` for records that omit them rather than reporting the
zero column's score.

## Beyond AP

`coco_inputs` and `coco_inputs_from_columns` return the
`(CocoDataset, DetectionsInput)` pair rather than a metric, so the same
conversion drives [`Evaluator`](../reference/python/instance.md), every
`evaluate_*_grid` / `evaluate_*_summary` entry point,
[custom grids](custom-evaluation-grids.md),
[calibration](calibration.md) and the
[distributed path](distributed-eval.md):

```python
from vernier.adapters import coco_inputs
from vernier.instance import Evaluator, Bbox

ground_truth, dt = coco_inputs(predictions, targets, box_format="xyxy")
summary = Evaluator(iou=Bbox()).evaluate(ground_truth, dt)
print(summary.stats[0])
```

`coco_metrics` is just the convenience wrapper over this pair.

TIDE, LRP, the confusion matrix and the `tables=` / `manifest=` paths do
**not** accept these inputs yet — each asks for GT JSON bytes, a
limitation that predates this route. For those, keep using a COCO file.

## What it decides for you

These are behaviour, not parameters, because each has exactly one correct
answer — and each is silent when got wrong:

- every image gets an `images` entry, including one with no annotations;
- annotation ids start at **1**; COCOeval's results are wrong from 0;
- `iscrowd` is widened to `int64`, since vernier reads any non-zero as a
  crowd and a `uint8` column wraps 256 to 0;
- image sizes resolve from the image's own first mask, else the size its
  detections carry, else `0x0` — which nothing reads;
- a `(1, 0)`-shaped empty box array is normalized to `(0, 4)` before both
  concatenation and per-image counting.

`area` is the one that is a parameter, because `"supplied"` is a
legitimate choice for a caller whose areas are authoritative. The default
`"auto"` takes a positive supplied area and otherwise falls back **per
element** — to the mask's area whenever the annotation carries a mask,
and to the box's when it does not. A framework that recorded areas for
some annotations and zeros for the rest is the common case, not a corner
one.

The mask wins **under either IoU type**, because a COCO annotation has
one `area` and it is the segmentation's: `COCOeval` with
`iouType="bbox"` buckets by that same field rather than recomputing
`w * h`. Deriving the box area for a bbox pass is invisible until an
object's two areas straddle `32²` or `96²`, and then it moves AP between
the small and medium buckets against every pycocotools-shaped evaluator.

## What it refuses

Per ADR-0057, an input that cannot be read correctly is an error, never a
repair. Expect a raise — not a plausible number — for an unknown
`box_format` / `iou_type` / `area`; a missing required field (an image
genuinely without annotations passes an explicit empty array); a
transposed `(4, N)` box array; a fractional class label or `iscrowd`; a
mask column covering only some of a side's rows; duplicate or misaligned
`image_id`s; and a `max_dets` ladder that does not increase.

## Dtypes

Unlike the rest of vernier's array surface, this route converts dtypes by
default: a model emits `float32`, and refusing it would restore the glue
the route exists to delete. Pass `cast_inputs=False` to get ADR-0004's
strict `f64` boundary back.
