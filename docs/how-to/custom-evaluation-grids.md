# How to evaluate on a custom IoU / recall / area grid

`vernier.instance.Evaluator` exposes three fields per ADR-0040 that
override the canonical pycocotools-shaped evaluation grid:

| Field | What it sweeps | Default (when `None`) |
|---|---|---|
| `iou_thresholds: tuple[float, ...] \| None` | The matching IoU ladder | `(0.50, 0.55, …, 0.95)` (10 points) |
| `recall_thresholds: tuple[float, ...] \| None` | The AP-integration recall samples | `(0.0, 0.01, …, 1.0)` (101 points) |
| `area_ranges: Breakdown \| None` | The `(all / small / medium / large)` area buckets | COCO 4-bucket detection / 3-bucket keypoints |

Use them for sensitivity analysis (does AP@0.5 disagree with
AP@0.75?), domain area buckets (robotics distance ranges, medical
size ranges), or denser recall sampling near `R=1.0` for
extreme-recall regimes.

## The headline footgun

The canonical 12-stat / 10-stat / 13-stat summary plans address
hardcoded slot indices in the `(T, R, K, A, M)` accumulator tensor
— `AP_S` is *the second area-bucket entry of the all-IoU slice at
maxDet=100*, not "the small-area slot". Custom grids break that
index assumption silently.

So `Evaluator.evaluate(gt, dt)` raises `IncompatibleSummaryPlan`
the moment any of the three fields is set. The exception names the
remediation: call `evaluate_tables(gt, dt)` instead, which returns
an `EvalResult` whose per-row tables carry explicit labels.

```python
from vernier.instance import Bbox, Evaluator, IncompatibleSummaryPlan

ev = Evaluator(iou=Bbox(), iou_thresholds=(0.5, 0.6, 0.7, 0.8, 0.9))
try:
    ev.evaluate(gt, dt)
except IncompatibleSummaryPlan as e:
    print(e.field)        # "iou_thresholds"
    print(e.plan)         # "COCO 12-stat / keypoints 10-stat / LVIS 13-stat"
    print(e.remediation)  # "...Use evaluate_tables(...) for tabular output..."
```

The fix is a one-line swap to `evaluate_tables`:

```python
result = Evaluator(iou=Bbox(), iou_thresholds=(0.5, 0.6, 0.7, 0.8, 0.9)) \
    .evaluate_tables(gt, dt)

# `result.summary` is None on a custom-grid run — slot indices don't generalise.
# `result.per_class`, `result.per_image`, etc. carry explicit labels.
print(result.per_class)
```

## Sensitivity analysis on IoU thresholds

A four-point ladder bracketing the COCO defaults — useful for
"is AP@0.5 telling me the same thing as AP@0.9?":

```python
ev = Evaluator(iou=Bbox(), iou_thresholds=(0.50, 0.75, 0.85, 0.95))
result = ev.evaluate_tables(gt, dt)
```

Each `iou_thresholds` value must lie in `[0.0, 1.0]`, be finite,
and the tuple must be non-empty, sorted ascending, and free of
duplicates. Validation runs at construction time and raises
`InvalidInstanceParams` with a remediation pointer:

```python
from vernier.instance import InvalidInstanceParams

try:
    Evaluator(iou=Bbox(), iou_thresholds=(0.5, 0.5))
except InvalidInstanceParams as e:
    print(e.field, e.value, e.remediation)
```

## Domain-specific area buckets

`area_ranges` takes a `Breakdown` built with
`Breakdown.from_ranges(axis, buckets)`. Each bucket is a
`(label, lo, hi)` tuple with `lo <= hi`, both finite and non-negative.
Use `1e10` for the unbounded upper end (the COCO sentinel):

```python
from vernier.instance import Bbox, Breakdown, Evaluator

# Robotics distance buckets (m²; "all" + three domain buckets, mirrors
# the canonical 4-bucket count so per_class / per_image tables build).
area = Breakdown.from_ranges(
    "distance",
    [
        ("all", 0.0, 1e10),
        ("near", 0.0, 25.0),       # 0-5 m
        ("mid", 25.0, 225.0),      # 5-15 m
        ("far", 225.0, 1e10),      # 15 m+
    ],
)
ev = Evaluator(iou=Bbox(), area_ranges=area)
result = ev.evaluate_tables(gt, dt)
```

Bucket-count constraint: the canonical `per_class` and `per_image`
table schemas pin a 4-bucket layout (`all` + three domain buckets)
inherited from pycocotools. Bucket counts other than 4 still
evaluate, but `per_class` / `per_image` will raise from the table
builder; reach for `per_pair` / `per_detection` instead, which are
label-keyed and tolerate any bucket count.

Bucket-boundary semantics follow pycocotools quirks **D6/D7** —
both bounds are inclusive (`lo <= area <= hi`), so an annotation
with area exactly equal to a boundary lands in *both* adjacent
buckets. Use distinct boundaries if that's not what you want.

The `area_ranges` field accepts only the range variant of
`Breakdown`. Class-group breakdowns belong on the
`vernier.semantic` / `vernier.panoptic` `class_grouping` parameter,
not here; passing one raises `InvalidInstanceParams`.

## Denser recall sampling

For extreme-recall regimes (autonomous-driving safety, medical
screening), the canonical 101-point linspace is too sparse near
`R=1.0`. Densify the tail:

```python
import numpy as np

dense_tail = tuple(np.concatenate([
    np.linspace(0.0, 0.9, 91),
    np.linspace(0.901, 1.0, 100),  # 10× density on the last decile
]).round(4).tolist())

ev = Evaluator(iou=Bbox(), recall_thresholds=dense_tail)
result = ev.evaluate_tables(gt, dt)
```

Same validation rule as `iou_thresholds`: non-empty, sorted, no
duplicates, every element finite and in `[0.0, 1.0]`.

## Composing custom grids with the rest of the evaluator

The grid axes are orthogonal to the rest of the `Evaluator`
configuration:

```python
from vernier.instance import Boundary, Breakdown, Evaluator

ev = Evaluator(
    iou=Boundary(dilation_ratio=0.02),
    parity_mode="corrected",
    max_dets=(100, 300),
    iou_thresholds=(0.5, 0.75, 0.9),
    area_ranges=Breakdown.from_ranges(
        "area",
        [("all", 0.0, 1e10), ("tiny", 0.0, 64.0), ("small", 64.0, 1024.0)],
    ),
)
result = ev.evaluate_tables(gt, dt)
```

`max_dets`, `parity_mode`, `iou`, `use_cats`, `cast_inputs`, and
the GT-handle path (`evaluate_tables(dataset, dt)`) all behave
identically to canonical-grid runs.

## Distributed eval and `BackgroundEvaluator`

The wire format and worker-thread plumbing for ADR-0040 grids are a
follow-up to the streaming substrate. Until that lands, custom-grid
configurations are batch-only:

- `Evaluator.evaluate_to_partial(gt, dt, rank_id=...)` raises
  `InvalidInstanceParams` with a pointer to `evaluate_tables`.
- `BackgroundEvaluator` accepts only canonical grids today; build a
  single-rank custom-grid run and call `evaluate_tables(...)` on
  the gathered detections instead.

## See also

- [ADR-0040](../adr/0040-user-parametrizable-instance-evaluation-grid.md)
  — the surface ratification.
- [ADR-0019](../adr/0019-result-tables.md) — the result-tables surface
  custom grids route through.
- [ADR-0016](../adr/0016-generalized-breakdown-axis.md) — the
  value-typed `Breakdown` used for `area_ranges`.
- [How to use result tables](result-tables.md) — consuming the
  `EvalResult` polars DataFrames.
- [How to configure the evaluator](configure-evaluator.md) — the
  rest of the evaluator's surface.
