# Migrating from `mmsegmentation` to vernier

vernier ships first-class semantic-segmentation evaluation under
`vernier.semantic` (ADR-0028). This guide is the user-facing migration
path from `mmsegmentation`'s `IoUMetric`. Audience: anyone moving an
existing semantic-segmentation evaluation pipeline onto vernier.

> **Status note.** vernier's semantic surface is functional and
> tested against hand-computed fixtures, but the strict-mode parity
> claim against a vendored `mmsegmentation` oracle is a follow-up
> (PR-B6). Until that lands, `parity_mode="strict"` produces
> mmsegmentation-shaped numbers but is not yet pinned to a frozen
> commit SHA. The
> [`docs/engineering/sem-seg-quirks.md`](../engineering/sem-seg-quirks.md)
> survey enumerates the `(quirk, oracle) → mode` cells the eventual
> harness verifies.

## TL;DR — what to change

| `mmsegmentation` | vernier |
|---|---|
| `IoUMetric(iou_metrics=['mIoU', 'mAcc'], num_classes=19, ignore_index=255)` constructed inside an evaluation loop | `vernier.semantic.Evaluator(parity_mode='corrected')` constructed once |
| `metric.process(data_batch, data_samples)` per batch | `evaluator.evaluate(gt: Dataset, dt: Predictions)` for batch, or `evaluator.stream(...)` + `update(image_id, gt, dt)` for streaming |
| `metric.compute_metrics(results)` returns `dict[str, float]` | `Summary` with field accessors (`miou`, `fwiou`, `pixel_accuracy`, `mean_accuracy`) plus `confusion_matrix` and `per_class()` |
| `np.bincount(gt * n_classes + pred, ...).reshape(n_classes, n_classes)` per image | Internal — confusion matrix is a first-class output via `summary.confusion_matrix.counts()` |
| `reduce_zero_label=True` flag | `Evaluator(label_remap={0: 255, 1: 0, 2: 1, ...})` (the equivalent dict; ADE20K's preset bakes `ignore_label=0` instead) |

`vernier.semantic.Evaluator` is **not** a variant of
`vernier.instance.Evaluator` — it's a sibling class because
semantic-mIoU does not share the AP fold (no per-detection matching,
no score gradient, no T/R/A/M axes; the kernel is a per-image
confusion-matrix accumulator). Choose the class that matches your
metric.

## Per-dataset presets

The presets bake the canonical `(n_classes, ignore_label)` constants
so the user only passes the PNG paths:

```python
from vernier.semantic import Dataset, Predictions, Evaluator

# Cityscapes 19-class evaluation (n_classes=19, ignore_label=255):
gt = Dataset.cityscapes({1: "munich_000000_gtFine_labelTrainIds.png", ...})
dt = Predictions.from_files({1: "predictions/munich_000000.png", ...})
summary = Evaluator(parity_mode="strict").evaluate(gt, dt)
print(f"mIoU: {summary.miou:.4f}")
```

| Preset | n_classes | ignore_label |
|---|---|---|
| `Dataset.cityscapes(png_paths)` | 19 | 255 |
| `Dataset.ade20k(png_paths)` | 150 | 0 (class 0 is "other/unlabeled") |
| `Dataset.pascal_voc(png_paths)` | 21 | 255 (void at object boundary) |

PNGs are loaded via lazy-imported Pillow (single-channel only —
multi-channel RGB-encoded panoptic PNGs are rejected with a typed
message pointing at `vernier.panoptic.Dataset`). For pre-decoded
arrays, use `Dataset.from_arrays(label_maps, n_classes,
ignore_label=...)` — uint8 / uint16 / uint32 dtypes are accepted and
upcast at the FFI boundary.

## Streaming

Confusion matrices are additively aggregable, so streaming is a thin
wrapper:

```python
ev = vernier.semantic.Evaluator(parity_mode="strict").stream(
    n_classes=19, ignore_label=255,
)
for image_id, gt_arr, dt_arr in dataloader:
    ev.update(image_id, gt_arr, dt_arr)
summary = ev.finalize()
```

Streaming `finalize()` is **bit-equal to batch `evaluate(...)`** on
f64 outputs over the same images (pinned by
`tests/python/test_semantic_streaming.py::test_streaming_finalize_bit_equals_batch_evaluate`).

There is no fast-vs-running snapshot mode (per ADR-0013 §"Fast
snapshot mode") — `snapshot()` is constant-time relative to image
count because it folds over the fixed-size `(n_classes, n_classes)`
matrix, not over a per-image cell store. The instance streaming
evaluator needs the distinction; semantic doesn't.

## NaN vs 0.0 for zero-support classes (quirk **AL2**)

For a class with `TP_c + FP_c + FN_c == 0` (never seen, never
predicted), per-class IoU is undefined. The three oracles disagree:
mmsegmentation returns `nan`; cityscapesScripts returns `0.0`; the
ADE20K reference returns `nan`.

| `parity_mode` | Per-class IoU for zero-support | mIoU includes zero-support? |
|---|---|---|
| `"strict"` | `f64::NAN` (matches mmsegmentation `np.nanmean`) | No (quirk **AL3**) |
| `"corrected"` (default) | `0.0` (matches cityscapesScripts shape) | No (quirk **AL3**) |

**Either way, the mean (mIoU and mAcc) excludes zero-support classes
from the average** — same convention as panopticapi (W2) and LVIS
(AB3). A class never seen contributes neither to the mean nor to its
denominator.

Plotting the per-class IoU table? Filter out NaN (or zero, depending
on mode) before downstream comparisons or you'll get a misleading
"zero" bar for missing-from-fixture classes.

## Confusion matrix as a first-class output

ADR-0028 §F1 promotes the confusion matrix to a first-class output:
downstream calibration, error decomposition, and model-diff tools
consume it directly without re-running the kernel.

```python
cm = summary.confusion_matrix
print(f"shape: {cm.n_classes} × {cm.n_classes}")
print(f"total evaluated pixels: {cm.total}")
print(f"trace (correct pixels): {cm.trace}")
counts: np.ndarray = cm.counts()  # (N, N) numpy uint64
print(f"row 0 (gt=0): {counts[0]}")
print(f"col 5 (pred=5): {counts[:, 5]}")
```

## Binary-mask predictions (AV / robotics)

Some semantic-segmentation models emit one binary mask per class
(typical for free-space / drivable-surface / lane-segmentation
heads). vernier provides an explicit merge step:

```python
# masks[image_id] is shape (n_classes, H, W) uint8.
dt = Predictions.from_binary_masks(
    {image_id: masks_3d},
    merge="argmax",        # or "first" / "highest_class_id"
    unlabeled_class=255,   # written into all-zero-mask pixels
)
```

The `merge` selector is required to be explicit per quirk **AN3**:
`argmax` ties go to the lowest class id (numpy default tiebreak);
`first` is order-dependent; `highest_class_id` is
convention-dependent. Pixels where every mask is zero are written as
`unlabeled_class` (per quirk **AN4**) — typical use is to set this
to the dataset's `ignore_label`.

## What's deferred

- **Boundary semantic IoU** — different metric (per-class
  Chebyshev-eroded boundary bands; Cheng et al. 2021 §6 default
  `dilation_ratio = 0.01`). ADR-0030 follow-up.
- **Per-image mIoU** — the metric is not well-defined (a class
  absent from one image has undefined IoU for that image) and
  biases comparisons toward easy images. Same rationale as
  [`docs/explanation/why-no-per-image-ap.md`](../explanation/why-no-per-image-ap.md).
  The per-image confusion matrix is exposed for callers who want to
  derive their own per-image metric.
- **3D / BEV semantic segmentation** — same shape (per-pixel class
  assignment, per-class IoU) but the data model is voxel- or
  BEV-grid-shaped. Out of scope for the per-image surface.
- **Class-incremental / open-set evaluation** — research problem
  with no standard metric set yet; revisit when the field
  consolidates.

## See also

- [ADR-0028](../adr/0028-sem-seg.md) — design record for the
  semantic-segmentation surface.
- [ADR-0029](../adr/0029-namespace.md) — per-paradigm namespace
  restructure (why `vernier.semantic` is a sibling submodule, not a
  flat-root `SemanticEvaluator`).
- [`docs/engineering/sem-seg-quirks.md`](../engineering/sem-seg-quirks.md)
  — `(quirk_id, oracle) → mode` disposition table for
  mmsegmentation / cityscapesScripts / Pascal-VOC / ADE20K.
- [Why three evaluators?](../explanation/three-paradigms.md) — when
  to use which paradigm.
