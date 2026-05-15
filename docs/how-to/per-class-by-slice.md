# Per-class metrics broken down by manifest slice

`Evaluator.evaluate(tables=...)` and `Evaluator.evaluate(manifest=...)`
each ship the half a user typically needs:

- `tables=("per_class",)` produces a per-class table (one row per
  category, columns are `ap_s` / `ap_m` / `ap_l` / etc.).
- `manifest=...` produces a slices table (one row per `(axis, value)`
  cell, columns are the same metric set).

**Their cross product — per-class metrics broken down by manifest slice
— is a deliberate non-feature.** Combining the two kwargs raises
`ValueError`:

```python
ev.evaluate(gt, dt, tables=("per_class",), manifest=meta)
# ValueError: tables= and manifest= cannot be combined; see
# docs/how-to/per-class-by-slice.md for the recipe.
```

If you actually need the cross product (AP for category 17 *within*
`weather=fog`), assemble it client-side. The pattern is one
`Evaluator.evaluate(tables=...)` call per slice over a filtered
detection set:

```python
import json
import polars as pl
from vernier.instance import Bbox, Evaluator

gt_bytes = open("gt.json", "rb").read()
dt_records = json.load(open("dt.json"))
manifest = json.load(open("weather.json"))  # key_kind="image_id"

ev = Evaluator(iou=Bbox(), parity_mode="strict")

# 1) Build {image_id -> {axis -> value}} from the manifest.
assignments: dict[int, dict[str, str]] = {}
for row in manifest["rows"]:
    image_id = int(row["key"])
    assignments[image_id] = {k: v for k, v in row.items() if k != "key"}

# 2) For each (axis, value) cell the manifest reaches, run per_class
#    over the slice's detections.
per_slice_per_class: dict[tuple[str, str], pl.DataFrame] = {}
for axis in {a for row in assignments.values() for a in row}:
    values = {row[axis] for row in assignments.values() if axis in row}
    for value in values:
        slice_image_ids = {
            iid for iid, axes in assignments.items()
            if axes.get(axis) == value
        }
        slice_dt = [r for r in dt_records if r["image_id"] in slice_image_ids]
        result = ev.evaluate(gt_bytes, json.dumps(slice_dt).encode(), tables=("per_class",))
        per_slice_per_class[(axis, value)] = result.per_class

# 3) Stack into one long table for analysis.
long = pl.concat([
    df.with_columns(pl.lit(axis).alias("axis"), pl.lit(value).alias("value"))
    for (axis, value), df in per_slice_per_class.items()
], how="diagonal")
```

The pattern runs the matching engine `N` times (one per slice) — C1 in
ADR-0046's terminology, not the C3 axiom the partitioned `result.slices`
path honors. For most workflows this is fine because per-class queries
aren't latency-sensitive (they're an offline analysis step, not part
of training loop telemetry).

## Why not a built-in `result.per_class_slices`?

A built-in would need to retain each slice's `Accumulated` tensor (one
per category × area × maxDets cell × slice) across the partition
orchestrator and surface a paradigm-specific Arrow builder per slice.
For the niche of users who actually need the cross product, the
recipe above is one screen of code and runs in the same time as the
built-in would (the per-slice eval inside vernier *is* the dominant
cost).

If your workflow needs the cross product on **every** evaluate call
(e.g. for a regression-tracking bucket on a corruption sweep), please
open an issue with the use case — the design space is shipped enough
that a focused FFI surface drop-in is feasible if there's demand.
