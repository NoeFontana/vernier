# How to use result tables

Recipe-style guide for the per-image / per-class / per-detection / per-pair
result tables shipped by [ADR-0019](../adr/0019-result-tables.md). Each
recipe is a short heading, a copy-paste snippet, and one paragraph of
context. The schema is pinned in ADR-0019 §"Schemas"; the rationale for
the deliberately absent per-image AP column lives in
[`why-no-per-image-ap.md`](../explanation/why-no-per-image-ap.md). Tables
are the extended-API surface; the `patch_pycocotools` drop-in (ADR-0007)
does not gain them.

## Quick start

```sh
pip install 'vernier[tables]'
```

```python
from vernier.instance import Evaluator

with open("instances_val2017.json", "rb") as f:
    gt = f.read()
with open("detections.json", "rb") as f:
    dt = f.read()

result = Evaluator().evaluate(gt, dt, tables="all")
print(result.stats[0])       # headline AP, same as Summary.stats
print(result.per_image)      # polars.DataFrame
print(result.per_class)      # polars.DataFrame
print(result.per_detection)  # polars.DataFrame
print(result.per_pair)       # polars.DataFrame
```

`tables="all"` requests every table the active build supports. The
return type widens from `Summary` to `EvalResult`; `result.stats` is a
pass-through that preserves the existing summary-shape API. polars is
imported lazily on the first DataFrame attribute access — `import
vernier` does not pay the cost. Pass a tuple
(`tables=("per_image", "per_class")`) when you only want a subset.

## Find the worst-performing images

```python
import polars as pl

worst = (
    result.per_image
    .with_columns((pl.col("fp_at_50") + pl.col("fn_at_50")).alias("err_at_50"))
    .sort("err_at_50", descending=True)
    .head(20)
    .select("image_id", "n_gt", "n_dt", "tp_at_50", "fp_at_50", "fn_at_50")
)
print(worst)
```

`per_image` carries TP/FP/FN counts at IoU 0.50 and 0.75. Sorting by
`fp_at_50 + fn_at_50` surfaces the images contributing most error at
the loose threshold; swap in `_at_75` for the strict threshold. The
table deliberately omits per-image AP; see
[`why-no-per-image-ap.md`](../explanation/why-no-per-image-ap.md).

## Per-class drill-down

```python
weak_classes = (
    result.per_class
    .filter(pl.col("ap") < 0.3)
    .sort("ap")
    .select("category_id", "category_name", "ap", "ap50", "n_gt", "n_dt")
)
print(weak_classes)
```

`per_class` mirrors the 12 stats `summarize_detection` reports per
category, plus support counts (`n_gt`, `n_dt`). Cells with no GTs in a
category surface as Arrow nulls — the `-1` sentinel pycocotools uses
internally (quirk **C5**) is translated rather than leaked.

## Hardest matched detections

```python
hard_tps = (
    result.per_detection
    .filter((pl.col("match_status_at_50") == "tp") & (pl.col("best_iou") < 0.6))
    .sort("best_iou")
    .select("detection_id", "image_id", "category_id", "score", "best_iou")
)
print(hard_tps.head(50))
```

`per_detection` carries one row per detection with its match status
at IoU 0.50, the matched GT id when applicable, and `best_iou` (the
maximum IoU to any same-class GT in the same image). Filtering to
true positives with low `best_iou` surfaces the detections that
*just barely* matched — the natural mining target for hard-example
fine-tuning. `best_iou` is null when there were no same-class GTs in
the image.

## Confusion-pair analysis (within-class)

```python
from vernier.instance import Evaluator, TablesConfig

result = Evaluator().evaluate(
    gt, dt,
    tables=("per_pair",),
    tables_config=TablesConfig(per_pair_iou_floor=0.0),
)

per_class_pairs = (
    result.per_pair
    .group_by("category_id")
    .agg(
        pl.len().alias("n_pairs"),
        pl.col("iou").mean().alias("mean_iou"),
        pl.col("iou").quantile(0.1).alias("p10_iou"),
    )
    .sort("p10_iou")
)
print(per_class_pairs)
```

`per_pair` is **class-restricted in v0.5**: every row has the same
`category_id` for DT and GT, matching the matching engine's
behavior. Cross-class confusion (a DT of class A overlapping a GT of
class B) is on the 2026-Q3 roadmap and is explicitly deferred per
ADR-0019 §"What this ADR explicitly does *not* decide". Within-class
mining (low-IoU true positives, near-miss pairs) works today; pair
the `per_pair` IoU stream with `per_detection.score` joined on
`detection_id` to reconstruct the per-class PR shape.

## Export to Parquet for offline analysis

```python
result.per_pair.write_parquet("pairs.parquet")
result.per_detection.write_parquet("detections.parquet")
result.per_image.write_parquet("images.parquet")
result.per_class.write_parquet("classes.parquet")
```

Parquet emit is polars-native, not a vernier method. ADR-0019
§"What this ADR explicitly does *not* decide" deliberately punts
on-disk emit to the user's DataFrame library — we ship the Arrow
PyCapsule producer and let the consumer pick the file format.

## Convert to pandas

```python
df = result.per_image.to_pandas()
print(df.describe())
```

`to_pandas()` is built into polars; we do not reimplement it. The
same applies for duckdb (`duckdb.from_arrow(...)`) and pyarrow
(`pyarrow.table(...)`) — every ADR-0019 result table is a
PyCapsule-backed Arrow `RecordBatch` underneath, and any
PyCapsule-aware consumer reads it zero-copy.

## Memory budget for `per_pair`

```python
from vernier.instance import Evaluator, TablesConfig

result = Evaluator().evaluate(
    gt, dt,
    tables=("per_pair",),
    tables_config=TablesConfig(
        per_pair_iou_floor=0.3,      # default 0.1; raise to keep fewer rows
        per_pair_max_rows=5_000_000, # default 10_000_000; lower to fail faster
    ),
)
```

`per_pair` row count is the dominant memory cost. For COCO val2017
the default `iou_floor=0.1` keeps it under ~1 M rows (~40 MB Arrow);
LVIS-scale workloads with the same floor can exceed hundreds of
millions of rows, which is why the cap exists. Exceeding
`per_pair_max_rows` raises an exception rather than silently
truncating — `ValueError` in v0.5 (the typed `PerPairOverflowError`
shipped on the public surface in Week 2.4 of the phased landing
plan; check the ADR for the current state).

## What's not in the table

The v0.5 `EvalResult` deliberately omits the following surfaces; each
is documented at its own anchor:

- **Per-image `ap` / `ap_50` columns.** PR curves from a single image
  are degenerate. See
  [`why-no-per-image-ap.md`](../explanation/why-no-per-image-ap.md).
- **Cross-class `per_pair` confusion.** Pairs are class-restricted in
  v0.5; cross-class confusion lands in 2026-Q3 per ADR-0019 §"What
  this ADR explicitly does *not* decide".
- **Chunked Arrow IPC for very-large `per_pair`.** v0.5 emits a single
  `RecordBatch` per table; workloads beyond ~10 M `per_pair` rows
  motivate a follow-up ADR for streaming IPC. Until then, raise
  `per_pair_iou_floor` or fail loud via `per_pair_max_rows`.
- **On-disk Parquet emit as a vernier method.** Free via polars
  (`df.write_parquet(...)`); we ship the data, the user picks the
  serializer.
- **`tables=` on the `COCOeval` drop-in.** Out of scope per ADR-0007;
  the drop-in is the migration shim, not the extended API.

## See also

- [`docs/adr/0019-result-tables.md`](../adr/0019-result-tables.md) —
  the table design, schemas, and the zero-overhead-by-default
  contract on the unparameterized `evaluate()` path.
- [`docs/explanation/why-no-per-image-ap.md`](../explanation/why-no-per-image-ap.md)
  — rationale for the per-image AP omission.
- [`docs/adr/0007-patch-pycocotools-policy.md`](../adr/0007-patch-pycocotools-policy.md)
  — why the `COCOeval` drop-in stays narrow.
- [`docs/adr/0014-background-evaluator.md`](../adr/0014-background-evaluator.md)
  — `BackgroundEvaluator.finalize_with_tables(...)` shape and the
  detection-id caveat that propagates into `per_detection`.
