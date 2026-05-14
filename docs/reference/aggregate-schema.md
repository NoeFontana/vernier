# `vernier aggregate --emit json` output schema (`aggregate_version: "1"`)

The fan-in companion to `vernier eval --manifest` (ADR-0046). Reads N result documents (v1 or v2), joins each to a `key_kind=result` manifest by `--label` (falling back to the result file path's basename when a result has no label), groups runs by `(axis, value)`, and emits a comparative per-slice table.

The output lives in its own version namespace — `aggregate_version`, not `version` — because the verb and consumer differ from `vernier eval`'s output. Coupling their version axes would force lockstep bumps the lifecycles do not warrant.

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
