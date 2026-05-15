# How to evaluate model calibration

Calibration (ADR-0018) reports whether a detector's confidence scores
match its empirical correctness. vernier ships ECE / MCE plus a
per-bin reliability table for the detection family (bbox / segm /
boundary / keypoints), folded over the ADR-0013 per-image cell store
so re-folding with different params is matching-free.

## Default ECE / MCE

```python
from pathlib import Path
from vernier.instance import Bbox, Evaluator

gt_bytes = Path("instances_val2017.json").read_bytes()
dt_bytes = Path("detections.json").read_bytes()

result = Evaluator(iou=Bbox()).evaluate(gt_bytes, dt_bytes, calibration=True)
cal = result.calibration()

print(cal.ece, cal.mce, cal.n_detections, cal.effective_n_bins)
print(cal.reliability)  # polars DataFrame, one row per bin
```

`calibration=True` retains the per-image cell store on the result;
without it `result.calibration(...)` raises `RuntimeError`. The
reliability table has 9 columns: `bin_id`, `score_lo`, `score_hi`,
`mean_score`, `accuracy`, `count`, `gap`, `ci_lo`, `ci_hi`. Zero-count
bins emit `NaN` for the float columns (kernel convention R2).

## Re-fold without re-matching

`result.calibration(...)` accepts seven keyword-only knobs; folding
with different values does not re-run the matching engine — only the
histogram pass.

```python
cal_15 = result.calibration(n_bins=15)
cal_30 = result.calibration(n_bins=30)
cal_equal = result.calibration(binning="equal_width")
cal_strict = result.calibration(min_score=0.0)         # include all scores
cal_iou75 = result.calibration(iou=0.75)               # re-fold at a different T-axis
```

`iou` resolves to the kernel's pinned T-axis index under `PARITY_EPS`;
values that don't land on a threshold the evaluator was configured
for raise `ValueError`. The defaults (`iou=0.5`, `n_bins=15`,
`binning="quantile"`, `min_score=0.05`, `confidence="wilson"`,
`per_class=False`, `per_class_aggregation="macro"`) are
parity-pinned per the quirks survey
[`calibration-quirks.md`](../engineering/calibration-quirks.md) (P1,
P4, P5, P6).

## Per-class breakdown

```python
cal = result.calibration(per_class=True)
print(cal.per_class)         # polars: class_id, ece, mce, n
print(cal.worst(5))          # top-5 worst-calibrated classes by ECE
```

The per-class table is only built when `per_class=True`; accessing
`.per_class` without it raises `RuntimeError`. The marginal
`cal.ece` / `cal.mce` still come from the
`per_class_aggregation="macro"` rollup (unweighted mean across
classes — quirk P6). Pass `per_class_aggregation="micro"` to pool
detections globally instead.

## Streaming during training

Pair `BackgroundEvaluator` with `StreamingSnapshot` to fold the cell
store off-thread:

```python
from vernier.calibration import StreamingSnapshot
from vernier.instance import Bbox, Evaluator

ev = Evaluator(iou=Bbox())
bg = ev.background(gt_bytes)

for batch in epoch_outputs():           # whatever your loop yields
    bg.submit(batch.detections)

snap = StreamingSnapshot.from_background(bg)
print(snap.summary.stats[0])             # bit-equal to the batch path
print(snap.calibration().ece)            # same fold over retained cells
```

`StreamingSnapshot.from_background(bg)` consumes the worker via
`finalize_with_cells()` (one-shot — a second `finalize_*` call on the
same worker raises). The returned snapshot holds the canonical
`Summary` *and* the cell handle; `.calibration(...)` accepts the same
kwargs as the batch surface.

## Arrow surface

`cal.reliability` and `cal.per_class` are polars `DataFrame` views
over Arrow `RecordBatch`es backed by the ADR-0019 PyCapsule pattern.
Table names: `calibration_reliability` (9 columns above) and
`calibration_per_class` (`class_id`, `ece`, `mce`, `n`). The
underlying batches are reachable as `cal._reliability_batch` /
`cal._per_class_batch` for zero-copy consumers.

## What's not supported yet

- **Panoptic / semantic calibration.** Each is gated on a data-model
  prerequisite per the ADR-0018 per-paradigm shape map. Today
  `calibration=` only appears on `vernier.instance.Evaluator.evaluate`.
- **Clopper-Pearson CIs.** `confidence="clopper_pearson"` is a typed
  knob in the signature but documented Phase-2; today the parity-pinned
  default `"wilson"` is the only flavor with kernel-level tests.

## See also

- [ADR-0018](../adr/0018-calibration.md) — design rationale,
  per-paradigm shape map, deferral table.
- [`calibration-and-its-limits.md`](../explanation/calibration-and-its-limits.md)
  — what calibration answers, what it does not, when to use it
  alongside AP / oLRP.
- [`docs/engineering/calibration-quirks.md`](../engineering/calibration-quirks.md)
  — the P1–P10 quirks survey and three-tier dispositions.
- [`how-to/background-evaluator.md`](background-evaluator.md) — the
  general `BackgroundEvaluator` recipe `StreamingSnapshot` plugs into.
