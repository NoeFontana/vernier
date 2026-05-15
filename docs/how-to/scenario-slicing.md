# How to slice an evaluation by scenario

vernier's slice-and-aggregate surface (ADR-0046) takes a *partition
manifest* — a tiny CSV/JSON table mapping image ids to scenario axes
like `weather` or `time_of_day` — and emits one row of headline
metrics per `(axis, value)` cell on top of the usual overall summary.
The matching engine still runs once; slicing is post-hoc.

The reference for the manifest format itself lives at
[`docs/reference/manifest-schema.md`](../reference/manifest-schema.md);
this page is the recipe.

## Slice a single evaluation by an axis

```python
from pathlib import Path
from vernier.instance import Bbox, Evaluator

manifest = {
    "manifest_version": "1",
    "key_kind": "image_id",
    "rows": [
        {"key": 1, "weather": "clear", "time_of_day": "day"},
        {"key": 2, "weather": "clear", "time_of_day": "night"},
        {"key": 3, "weather": "fog",   "time_of_day": "day"},
        {"key": 4, "weather": "fog",   "time_of_day": "night"},
    ],
}

gt_bytes = Path("instances_val2017.json").read_bytes()
dt_bytes = Path("detections.json").read_bytes()

result = Evaluator(iou=Bbox()).evaluate(gt_bytes, dt_bytes, manifest=manifest)

print(result.stats[0])   # overall AP — bit-equal to the un-partitioned path
print(result.slices)     # polars DataFrame: one row per (axis, value) cell
```

The `overall` headline on `result.stats` / `result.summary` is
bit-equal to the un-partitioned call — that is ADR-0046's
load-bearing parity claim. The new surface is `result.slices`, a
polars `DataFrame` with one row per cell. With two axes × two
values + an `__unassigned__` bucket per axis, the example above
emits 6 rows.

`manifest=` accepts a dict, a JSON path, a `.csv` path, or any
object that implements the Arrow C Stream PyCapsule protocol (a
polars / pyarrow DataFrame works directly).

## Joint cells with `cross_axes`

The default is per-axis marginals. To opt into joint cells over a
tuple, pass `cross_axes=`:

```python
result = Evaluator(iou=Bbox()).evaluate(
    gt_bytes,
    dt_bytes,
    manifest=manifest,
    cross_axes=[["weather", "time_of_day"]],
)
# Now result.slices also contains rows like
#   ("weather::time_of_day", "fog::night")
```

Joint cells live on a `::`-joined axis name with `::`-joined values.
Single-axis crosses are rejected — that is just a marginal, already
emitted. A 256-slice cap (`SLICES_CAP`) guards against typo-driven
combinatorial explosions; over that, split into multiple runs.

## Panoptic and semantic share the same manifest

The C3 axiom in ADR-0046 — fan-out matching once, reuse across slices
— is paradigm-wide. Same manifest, same `evaluate(..., manifest=...)`
kwarg:

```python
import vernier.panoptic as pq
import vernier.semantic as sem

pq.Evaluator(parity_mode="corrected").evaluate(gt, dt, manifest=manifest)
sem.Evaluator(parity_mode="corrected").evaluate(gt, dt, manifest=manifest)
```

The result shape on each paradigm follows the paradigm's metric
ladder (PQ / SQ / RQ for panoptic; mIoU for semantic), but the
slicing surface — `result.slices`, `result.overall_n_images`,
`result.overall_n_detections` — is the same.

## CLI form

```sh
vernier eval --gt gt.json --dt dt.json --iou-type bbox \
    --manifest weather.json \
    --emit json=run.json
```

With `--manifest`, the CLI emits a v2 JSON envelope under
`"version": "2"` carrying both `overall` and a `slices[]` array.
Un-partitioned `vernier eval` keeps emitting v1 verbatim — that is
the load-bearing byte-stability contract of ADR-0046.

`--cross AXIS_A,AXIS_B` is the joint-cell flag (repeatable). The
schema is documented in
[`docs/reference/cli-output-schema.md`](../reference/cli-output-schema.md).

oLRP composes: pass `--metric olrp --manifest manifest.json` for
per-slice oLRP decomposition (`oLRP_Loc` / `oLRP_FP` / `oLRP_FN` +
per-class `tau`).

## Aggregating across corruption runs

The fan-in companion to `--manifest` is `vernier aggregate`. Run the
evaluator N times against different corruption variants of the same
detection set, stamp each run with a `--label`, then aggregate them
against a `key_kind="result"` manifest:

```sh
vernier eval --gt gt.json --dt dt_clean.json  --label clean --emit json=clean.json
vernier eval --gt gt.json --dt dt_fog.json    --label fog   --emit json=fog.json
vernier eval --gt gt.json --dt dt_snow.json   --label snow  --emit json=snow.json

vernier aggregate clean.json fog.json snow.json \
    --manifest corruptions.json \
    --baseline clean \
    --emit json=corruption_table.json
```

`corruptions.json`:

```json
{
  "manifest_version": "1",
  "key_kind": "result",
  "rows": [
    {"key": "clean", "weather": "clear"},
    {"key": "fog",   "weather": "fog"},
    {"key": "snow",  "weather": "snow"}
  ]
}
```

The output carries `<metric>` columns (mPC — mean across runs) plus
`<metric>__rpc` columns when `--baseline` is set (rPC — ratio to the
baseline cell). The naming mirrors Michaelis et al.
(NeurIPS-W 2019)'s corruption-benchmark convention.

Python mirror:

```python
from vernier.aggregate import aggregate

table = aggregate(
    ["clean.json", "fog.json", "snow.json"],
    manifest=corruptions_dict,
    baseline="clean",
)
# table is a pyarrow.RecordBatch — wrap in polars/pandas as needed
```

The full output schema is documented in
[`docs/reference/aggregate-schema.md`](../reference/aggregate-schema.md).

## What you can't do (deliberately)

`tables=("per_class",) + manifest=...` raises `ValueError`. The
per-class × per-slice cross product is a deliberate non-feature; the
client-side recipe lives in
[`per-class-by-slice.md`](per-class-by-slice.md).

## See also

- [ADR-0046](../adr/0046-slice-and-aggregate.md) — design rationale,
  C1/C2/C3 axiom analysis, `vernier aggregate` verb rationale.
- [`docs/reference/manifest-schema.md`](../reference/manifest-schema.md)
  — manifest fields, `key_kind` discriminator, `__unassigned__`
  semantics, CSV form.
- [`docs/reference/aggregate-schema.md`](../reference/aggregate-schema.md)
  — `aggregate_version: "1"` output schema, mPC/rPC semantics,
  metric alias table.
- [`docs/reference/cli-output-schema.md`](../reference/cli-output-schema.md)
  — the v2 partitioned-eval JSON envelope.
- [`per-class-by-slice.md`](per-class-by-slice.md) — the explicitly
  non-shipped cross product, recoverable in one screen of Python.
