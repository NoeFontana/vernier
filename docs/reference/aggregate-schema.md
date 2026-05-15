# `vernier aggregate --emit json` output schema (`aggregate_version: "1"`)

The fan-in companion to `vernier eval --manifest` (ADR-0046). Reads N result documents (v1 or v2), joins each to a `key_kind=result` manifest by `--label` (falling back to the result file path's basename when a result has no label), groups runs by `(axis, value)`, and emits a comparative per-slice table.

The output lives in its own version namespace — `aggregate_version`, not `version` — because the verb and consumer differ from `vernier eval`'s output. Coupling their version axes would force lockstep bumps the lifecycles do not warrant.

## Accepted input shapes

Both the Rust CLI (`vernier aggregate`) and the Python entry point (`vernier.aggregate`) accept the same input shapes:

- **CLI v2 JSON envelope.** The on-disk format `vernier eval --manifest --emit json=…` produces. Each `overall` / slice entry carries:
  - `lines: [{metric, iou_threshold, iou_threshold_label, area, max_dets, value}, …]` — one row per stat in plan order.
  - `stats: [v1, v2, …]` — the same values as `lines[].value`, kept for pycocotools-style positional access. `stats` is **redundant** with `lines` by construction; the aggregator reads `lines` and discards the positional `stats` array.
- **Dict-keyed `stats` shape.** A v2 doc whose `stats` field is a `{name: value}` map. Used by Python-side fixtures and by callers that already key metrics by their final column name. Each key is the column name verbatim; no further canonicalization is applied.
- **Arrow `RecordBatch` / `Table`** (Python entry point only). The canonical wide slice-table shape with columns `axis`, `value`, plus metric Float64 columns. Column names are whatever the producer emitted (e.g. the in-process `vernier.instance.Evaluator` schema in `crates/vernier-ffi/src/tables.rs`).

CLI v2 JSON output is a **first-class input** to `vernier.aggregate`: piping `vernier eval --manifest out.json` and then calling `vernier.aggregate([out.json], manifest=…)` is a supported flow.

## Top-level fields

| Field               | Type                                  | Notes                                                                                                                                                       |
|---------------------|---------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `aggregate_version` | string                                | Schema version pin. `"1"` today. Surfaces first so a downstream tool can sniff compatibility without parsing the rest.                                       |
| `baseline`          | string or `null`                      | The `--baseline` value passed on the command line. `null` when omitted; the document then carries no `*__rpc` columns.                                       |
| `metrics`           | array of strings                      | Metric column names, in the order each row's `metrics` map declares them. When `baseline` is set, each metric is paired with its `<metric>__rpc` sibling.    |
| `rows`              | array of [`Row` objects](#rows-subfields) | One row per `(axis, value)` cell, in canonical order (axis ascending, value ascending, `__unassigned__` last). |

Object keys appear in the order shown above (ADR-0015 §"Output determinism" — fixed key order, not insertion order).

## `rows[]` subfields

| Field      | Type    | Notes                                                                                                                |
|------------|---------|----------------------------------------------------------------------------------------------------------------------|
| `axis`     | string  | The manifest axis name. For joint cells, the `::`-joined tuple (`weather::time_of_day`).                              |
| `value`    | string  | The categorical level. For joint cells, the `::`-joined value tuple (`fog::night`). `__unassigned__` for unassigned. |
| `n_runs`   | integer | Number of joined result documents that contributed to this cell.                                                     |
| `metrics`  | object  | `{metric_name -> mean across runs}`. Insertion order matches the top-level `metrics` array, so column ordering is wire-stable. |

## mPC / rPC semantics

The aggregator emits two flavors of column depending on whether `--baseline` is set:

- **mPC** (mean Performance under Corruption). Each `<metric>` column is the simple arithmetic mean of `<metric>` across the runs joined to that `(axis, value)` cell. Runs missing the metric (or carrying a `null` for it) are dropped from the mean; runs without any joinable manifest row are skipped upstream with a stderr warning. With three corruption runs at `(weather=fog) -> ap=0.30 / 0.40 / 0.50`, the `(weather, fog)` row's `ap` column resolves to `0.40`.
- **rPC** (relative Performance under Corruption). When `--baseline VALUE` is set, each metric column gets a sibling `<metric>__rpc` column equal to `mean(<metric>_at_this_slice) / mean(<metric>_at_baseline_slice)` on the same axis. The baseline row resolves to `1.0` by construction; other slices are < 1.0 for metrics that degrade under corruption and > 1.0 if performance improves. Computed per-axis: the `weather=fog` row divides by the `weather=clean` baseline mean, not by some other axis's baseline. A baseline cell with mean `0.0` or non-finite resolves the rPC to `null` (division-by-zero is not a silent NaN here).

The two-column shape — `<metric>` plus `<metric>__rpc` — is what Michaelis et al. (NeurIPS-W 2019) call the mCE / rCE pair on the COCO-C corruption benchmark; vernier ships the AP-side mirror with the same naming convention so the tables drop in.

## `--baseline` semantics

`--baseline VALUE` activates the rPC pass. The argument is a *slice value* (e.g. `clean`), not an axis name — the aggregator looks up that value on every axis and treats the matching row as the baseline for that axis. A baseline value present on one axis but not another is fine: rPC columns for the absent axis resolve to `null`.

`--baseline` is the *only* way to get the rPC columns; without it the document carries only mPC. This is deliberate — the relative-reduction view requires an explicit reference, and an inferred default ("the first slice value alphabetically", say) would silently change which run is the reference when a manifest row is added.

## Metric column selection

`--metric NAME` is repeatable and selects which metric columns the document carries. Without it, every numeric column that appears on at least one joined result is included, in stable order.

The aggregator exposes two naming conventions for metric columns:

- **Aliases.** Pycocotools' standard table positions get short nicknames: `ap`, `ap50`, `ap75`, `ap_small`, `ap_medium`, `ap_large`, `ar_1`, `ar_10`, `ar_100`, `ar_small`, `ar_medium`, `ar_large`. Each alias resolves to the corresponding `(metric, iou_label, area, max_dets)` tuple on the result's `lines` array.
- **Canonical.** Every line is also exposed under the full canonical name `<metric>_<iou_label>_<area>_<max_dets>` (e.g. `AP_0.50:0.95_all_100`). Use this when an alias does not apply (custom area ranges, non-default `max_dets`).

Both naming conventions resolve to the same column value, so `--metric ap` and `--metric AP_0.50:0.95_all_100` are equivalent on a default-shaped detection result.

### Canonical mapping table

The CLI's `lines[]` entries deterministically map to the column names below. The Rust source of truth is `crates/vernier-cli/src/commands/aggregate.rs::position_alias`; `vernier.aggregate` mirrors it in `python/vernier/aggregate.py::_position_alias` so the two paths agree byte-for-byte on the same v2 JSON document.

| `metric` | `iou_threshold_label` | `area`   | `max_dets` | Alias        | Canonical                     |
|----------|-----------------------|----------|------------|--------------|-------------------------------|
| `AP`     | `0.50:0.95`           | `all`    | any        | `ap`         | `AP_0.50:0.95_all_<max_dets>` |
| `AP`     | `0.50`                | `all`    | any        | `ap50`       | `AP_0.50_all_<max_dets>`      |
| `AP`     | `0.75`                | `all`    | any        | `ap75`       | `AP_0.75_all_<max_dets>`      |
| `AP`     | `0.50:0.95`           | `small`  | any        | `ap_small`   | `AP_0.50:0.95_small_<max_dets>` |
| `AP`     | `0.50:0.95`           | `medium` | any        | `ap_medium`  | `AP_0.50:0.95_medium_<max_dets>` |
| `AP`     | `0.50:0.95`           | `large`  | any        | `ap_large`   | `AP_0.50:0.95_large_<max_dets>` |
| `AR`     | `0.50:0.95`           | `all`    | `1`        | `ar_1`       | `AR_0.50:0.95_all_1`          |
| `AR`     | `0.50:0.95`           | `all`    | `10`       | `ar_10`      | `AR_0.50:0.95_all_10`         |
| `AR`     | `0.50:0.95`           | `all`    | `100`      | `ar_100`     | `AR_0.50:0.95_all_100`        |
| `AR`     | `0.50:0.95`           | `small`  | any        | `ar_small`   | `AR_0.50:0.95_small_<max_dets>` |
| `AR`     | `0.50:0.95`           | `medium` | any        | `ar_medium`  | `AR_0.50:0.95_medium_<max_dets>` |
| `AR`     | `0.50:0.95`           | `large`  | any        | `ar_large`   | `AR_0.50:0.95_large_<max_dets>` |

A line whose tuple does not match any row above is exposed **only** under its canonical name. This applies to custom IoU thresholds, custom area ranges, and non-default `max_dets` ladders.

Both surfaces use the **same column names** for the 12 canonical detection slots — the alias table above is authoritative. The Arrow path (`vernier.instance.Evaluator().slices`, sourced from `crates/vernier-ffi/src/tables.rs::slices_instance_ap_schema`) emits the spelled-out pycocotools-style labels (`ap_small`, `ap_medium`, `ap_large`, `ar_1`, `ar_10`, `ar_100`, `ar_small`, `ar_medium`, `ar_large`) so a `vernier.aggregate` call that mixes Arrow `RecordBatch` inputs with CLI v2 JSON inputs (the `vernier eval --manifest` envelope) yields a single non-null column per metric — no wide-union with mostly-null cells. The label set matches pycocotools' text-table convention (`area=small`, `maxDets=  1`) emitted by `Summary::pretty_lines`.

## Worked example

```json
{
  "aggregate_version": "1",
  "baseline": "clean",
  "metrics": ["ap", "ap__rpc"],
  "rows": [
    {"axis": "weather", "value": "clean", "n_runs": 1, "metrics": {"ap": 0.80, "ap__rpc": 1.0}},
    {"axis": "weather", "value": "fog",   "n_runs": 1, "metrics": {"ap": 0.40, "ap__rpc": 0.5}},
    {"axis": "weather", "value": "noise", "n_runs": 1, "metrics": {"ap": 0.20, "ap__rpc": 0.25}}
  ]
}
```

## Determinism

The byte-determinism rules ADR-0015 pins for `vernier eval --emit json` apply verbatim here: fixed key order, no timestamps, no environment leakage, round-trip-safe float formatting, atomic file writes. The output bytes for a given input set (manifest + globbed results + flag set) are stable across runs, machines, and elapsed time.

## See also

- [`docs/adr/0046-slice-and-aggregate.md`](../adr/0046-slice-and-aggregate.md) — the design source of truth.
- [`docs/reference/manifest-schema.md`](manifest-schema.md) — the `manifest_version: "1"` reference (the input side of this verb).
- [`docs/reference/cli-output-schema.md`](cli-output-schema.md) — the v2 partitioned-eval schema (the consumer-side of `--results`).
