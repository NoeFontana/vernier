# How to evaluate keypoints with OKS

OKS (Object Keypoint Similarity) is the COCO keypoints equivalent
of IoU for boxes. ADR-0012 pins the per-category sigmas surface;
this guide is the recipe.

## Default COCO sigmas (17 keypoints)

```python
from pathlib import Path
from vernier.instance import Dataset, Evaluator, Keypoints

gt_bytes = Path("person_keypoints_val2017.json").read_bytes()
dt_bytes = Path("keypoints_predictions.json").read_bytes()

dataset = Dataset.from_json(gt_bytes)
summary = Evaluator(iou=Keypoints()).evaluate(dataset, dt_bytes)
print(summary.stats[0])  # keypoints AP
```

`Keypoints()` with no arguments uses the COCO 17-keypoint sigmas
baked into the kernel — the same constants `pycocotools` uses on
`person_keypoints_val2017`.

## Per-category sigmas

For datasets with multiple keypoint taxonomies (animal pose,
vehicle parts, custom categories), pass `sigmas=` as
`{category_id: [sigma_per_kp]}`:

```python
summary = Evaluator(
    iou=Keypoints(sigmas={
        1: [0.026, 0.025, 0.025, 0.035, 0.035,  # head
            0.079, 0.079, 0.072, 0.072, 0.062, 0.062,  # arms
            0.107, 0.107, 0.087, 0.087, 0.089, 0.089],  # legs
        2: [0.04] * 12,  # custom 12-keypoint category
    }),
).evaluate(dataset, dt_bytes)
```

The list length must match the keypoint count for that category in
GT. A mismatch raises a typed `ValueError` at evaluator construction
(not at evaluate-time) so misconfiguration surfaces early.

## TIDE on keypoints

`Keypoints(...)` raises `NotImplementedError` from
`error_decomposition` per ADR-0024: COCO keypoints is single-class,
which makes the Cls and Both bins structurally empty, and OKS is
not IoU so the `(t_b, t_f)` phase diagram does not carve the same
error geometry. Per-keypoint OKS contribution analysis is the
right TIDE-shaped tool here; it is planned as a separate capability.
The deferral note in
[`tutorials/debugging-with-tide.md`](../tutorials/debugging-with-tide.md#what-tide-does-not-do)
is the user-visible reference.

## CLI flag

```sh
vernier eval --iou keypoints --gt person_keypoints_val2017.json --dt preds.json
```

For per-category sigmas via the CLI, point at a JSON sigmas file:

```sh
vernier eval --iou keypoints --sigmas custom_sigmas.json --gt ... --dt ...
```

The full CLI recipe with the sigmas file format lives in
[`how-to/cli-eval.md`](cli-eval.md#evaluate-keypoints-with-custom-sigmas).

## See also

- [ADR-0012](../adr/0012-oks-keypoints-surface.md) — per-category
  sigmas, OKS-vs-IoU decision, validation rule.
- [ADR-0024](../adr/0024-tide-keypoints-deferred.md) — why TIDE on
  keypoints (OKS) is deferred; what the right replacement looks
  like.
- [`reference/coco-summary-stats.md`](../reference/coco-summary-stats.md)
  — what the 10 keypoints summary entries mean (no `AR1` row;
  keypoints uses a 3-area-bucket grid per quirk **D5**).
