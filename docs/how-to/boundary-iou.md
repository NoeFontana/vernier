# How to evaluate with boundary IoU

Boundary IoU (ADR-0010) measures detection quality in the band
around a mask's edge — the part of the geometry that mIoU often
under-penalizes for thin objects. vernier's boundary kernel is
parity-pinned against the
[`bowenc0221/boundary-iou-api`](https://github.com/bowenc0221/boundary-iou-api)
oracle.

## Default COCO boundary IoU

```python
from pathlib import Path
from vernier.instance import Boundary, Dataset, Evaluator

gt_bytes = Path("instances_val2017.json").read_bytes()
dt_bytes = Path("detections.json").read_bytes()

dataset = Dataset.from_json(gt_bytes)
summary = Evaluator(iou=Boundary()).evaluate(dataset, dt_bytes)
print(summary.stats[0])  # boundary AP
```

`Boundary()` defaults to `dilation_ratio=0.02` (the COCO
convention; same value `bowenc0221/boundary-iou-api` ships).

## LVIS-flavored boundary IoU

LVIS-scale workloads use `dilation_ratio=0.008` (boundary-iou-quirks
**Q1**, ADR-0026 quirk **AE3**). Pass it explicitly:

```python
summary = Evaluator(iou=Boundary(dilation_ratio=0.008)).evaluate(dataset, dt_bytes)
```

The migration guide
[`migrate/from-lvis-api.md`](../migrate/from-lvis-api.md#boundary-iou-on-lvis)
covers the LVIS path end-to-end (federated trim, `max_dets=300`).

## Boundary AP and TIDE

The TIDE decomposition's `t_b` background threshold is per-kernel
(ADR-0022): boundary IoU concentrates at lower values than bbox IoU
on the same overlap geometry, so the boundary-kernel default is
`t_b=0.05` (vs `0.1` for bbox / segm). The tutorial
[`tutorials/debugging-with-tide.md`](../tutorials/debugging-with-tide.md#choosing-thresholds)
explains why.

## CLI flag

The CLI takes the kernel as `--iou boundary`:

```sh
vernier eval --iou boundary --gt instances_val2017.json --dt detections.json
```

`--dilation-ratio` overrides the default (`0.02`); pass `0.008`
for the LVIS convention. Recipe in
[`how-to/cli-eval.md`](cli-eval.md#evaluate-boundary-iou).

## See also

- [ADR-0010](../adr/0010-boundary-iou-isolated-subsystem.md) — why
  boundary IoU is an isolated subsystem with a vendored oracle.
- [`docs/engineering/boundary-iou-quirks.md`](../engineering/boundary-iou-quirks.md)
  — the quirks survey for the `bowenc0221` oracle.
- [Migrating from `lvis-api`](../migrate/from-lvis-api.md#boundary-iou-on-lvis)
  — the LVIS-side migration; uses the `0.008` dilation default.
