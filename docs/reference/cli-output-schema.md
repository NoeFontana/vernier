# `vernier eval --emit json` output schema

The JSON formatter shipped by `vernier-cli` (ADR-0015) writes a single document per invocation. The shape is versioned independently of the `vernier-cli` package version: the document carries a `version` field whose value is the *schema* version, currently `"1"`. This page is the field-by-field reference for that schema.

The schema is a stability contract surface for the CLI — additive changes (new fields, new metric rows in known IoU types) keep the same `version` string; renames or removals require a schema-version bump (`"1"` → `"2"`). The `vernier-cli` package itself is shipping under 0.0.x patches today, but the schema version is decoupled from the package version on purpose: `"version": "1"` documents archived against any 0.0.x release stay consumable across all subsequent 0.0.x patches. ADR-0015 §"Versioning and stability commitments" pins this commitment.

## Top-level fields

| Field         | Type                                                | Notes                                                                                                                                                       |
|---------------|-----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `version`     | string                                              | Schema version. `"1"` for everything `vernier-cli` emits today. Bumps on breaking shape changes; see *Schema versioning* below.                              |
| `iou_type`    | string, one of `bbox` / `segm` / `boundary` / `keypoints` | Mirrors the `--iou-type` flag and the `IouKind` variant the kernel ran. Required: `--iou-type` has no default.                                         |
| `parity_mode` | string, one of `strict` / `corrected`               | The kernel-resolved parity mode. `--parity-mode aligned` collapses to `strict` per ADR-0002 (aligned-tier changes are output-equivalent), so the JSON only ever carries the two distinct kernel modes. Default `strict` (ADR-0015).                                                                                              |
| `max_dets`    | array of non-negative integers                      | The `M`-axis of the accumulator the eval ran on. Defaults resolve via the kernel-canonical path (ADR-0012): `[1, 10, 100]` for det-family, `[20]` for kp.   |
| `use_cats`    | boolean                                             | Mirrors `--use-cats` / `--no-use-cats`. Default `true`.                                                                                                      |
| `lines`       | array of [`Line` objects](#lines-subfields)         | Per-row stat plan output. Length is fixed per IoU type: 12 for bbox / segm / boundary, 10 for keypoints.                                                    |
| `stats`       | array of floats                                     | The `value` column of `lines`, extracted as a flat array. Same length and order as `lines`. See [`stats` and `lines` correspondence](#stats-and-lines-correspondence) below. |

Object keys appear in the order shown above. The CLI commits to a fixed key order, not insertion order — see ADR-0015 §"Output determinism".

## `lines[]` subfields

Each entry of `lines` is an object with the following fields:

| Field                  | Type                  | Notes                                                                                                                                                            |
|------------------------|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `metric`               | string, `AP` or `AR`  | Which kernel-output tensor this row reads (`precision` or `recall`).                                                                                              |
| `iou_threshold`        | float or `null`       | The IoU threshold this row pins. `null` means the row averages across the IoU ladder (the `0.50:0.95` rows). Floats are emitted as the threshold value (e.g. `0.5`, `0.75`). |
| `iou_threshold_label`  | string                | Human-readable label, used by the text formatter and surfaced here for parsing convenience. `"0.50"`, `"0.75"`, or `"0.50:0.95"`.                                |
| `area`                 | string                | Area-bucket label. `"all"`, `"small"`, `"medium"`, or `"large"` for the canonical COCO grid. Custom `AreaRng` callers see their label here verbatim.            |
| `max_dets`             | non-negative integer  | The `M`-axis cap this row resolved to. Always one of the entries in the top-level `max_dets` array.                                                              |
| `value`                | float                 | The metric value. Sentinel-filtered (cells equal to `-1` are excluded from the mean, per the `summarize_detection` rule documented in [`coco-summary-stats.md`](coco-summary-stats.md)). |

Within a `Line`, the field order is the order shown above.

## `stats` and `lines` correspondence

`stats[i]` is `lines[i].value` for every `i`. Both arrays have identical length and identical order. The canonical mapping from index `i` to `(metric, iou_threshold, area, max_dets)` for the bbox / segm / boundary plan is the 12-stat table in [`docs/reference/coco-summary-stats.md`](coco-summary-stats.md). The keypoints plan is 10 entries with `max_dets = [20]` and a 3-bucket area axis (`all`, `medium`, `large` — the `small` bucket is dropped per quirk D5); ADR-0012 §"Decision outcome" pins the kp plan order.

The redundancy is deliberate: tools that already index into pycocotools' `eval.stats` array port to `summary["stats"]` with one line of code. Tools that prefer addressed access (`row["metric"] == "AP" and row["area"] == "small"`) read from `summary["lines"]`.

## Worked examples

The numbers below are illustrative, not pinned. They match the canonical COCO plans in shape and order; the precision values are plausible for a strong detector on COCO val2017.

### `iou_type = "bbox"` (12 lines)

```json
{
  "version": "1",
  "iou_type": "bbox",
  "parity_mode": "strict",
  "max_dets": [1, 10, 100],
  "use_cats": true,
  "lines": [
    {"metric": "AP", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "all",    "max_dets": 100, "value": 0.527},
    {"metric": "AP", "iou_threshold": 0.5,  "iou_threshold_label": "0.50",      "area": "all",    "max_dets": 100, "value": 0.728},
    {"metric": "AP", "iou_threshold": 0.75, "iou_threshold_label": "0.75",      "area": "all",    "max_dets": 100, "value": 0.581},
    {"metric": "AP", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "small",  "max_dets": 100, "value": 0.341},
    {"metric": "AP", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "medium", "max_dets": 100, "value": 0.566},
    {"metric": "AP", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "large",  "max_dets": 100, "value": 0.683},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "all",    "max_dets": 1,   "value": 0.392},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "all",    "max_dets": 10,  "value": 0.624},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "all",    "max_dets": 100, "value": 0.661},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "small",  "max_dets": 100, "value": 0.471},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "medium", "max_dets": 100, "value": 0.706},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "large",  "max_dets": 100, "value": 0.812}
  ],
  "stats": [0.527, 0.728, 0.581, 0.341, 0.566, 0.683, 0.392, 0.624, 0.661, 0.471, 0.706, 0.812]
}
```

### `iou_type = "segm"` (12 lines)

The shape is identical to the bbox example. Only `iou_type` and the `value` column change.

```json
{
  "version": "1",
  "iou_type": "segm",
  "parity_mode": "strict",
  "max_dets": [1, 10, 100],
  "use_cats": true,
  "lines": [
    {"metric": "AP", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "all",    "max_dets": 100, "value": 0.451},
    {"metric": "AP", "iou_threshold": 0.5,  "iou_threshold_label": "0.50",      "area": "all",    "max_dets": 100, "value": 0.687},
    {"metric": "AP", "iou_threshold": 0.75, "iou_threshold_label": "0.75",      "area": "all",    "max_dets": 100, "value": 0.486},
    {"metric": "AP", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "small",  "max_dets": 100, "value": 0.255},
    {"metric": "AP", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "medium", "max_dets": 100, "value": 0.484},
    {"metric": "AP", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "large",  "max_dets": 100, "value": 0.622},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "all",    "max_dets": 1,   "value": 0.354},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "all",    "max_dets": 10,  "value": 0.561},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "all",    "max_dets": 100, "value": 0.589},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "small",  "max_dets": 100, "value": 0.382},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "medium", "max_dets": 100, "value": 0.633},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "large",  "max_dets": 100, "value": 0.751}
  ],
  "stats": [0.451, 0.687, 0.486, 0.255, 0.484, 0.622, 0.354, 0.561, 0.589, 0.382, 0.633, 0.751]
}
```

### `iou_type = "boundary"` (12 lines)

Same plan shape as `segm`. The numerical values differ because the kernel evaluates `min(mask_iou, boundary_iou)` per ADR-0010, not raw mask IoU.

```json
{
  "version": "1",
  "iou_type": "boundary",
  "parity_mode": "strict",
  "max_dets": [1, 10, 100],
  "use_cats": true,
  "lines": [
    {"metric": "AP", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "all",    "max_dets": 100, "value": 0.398},
    {"metric": "AP", "iou_threshold": 0.5,  "iou_threshold_label": "0.50",      "area": "all",    "max_dets": 100, "value": 0.612},
    {"metric": "AP", "iou_threshold": 0.75, "iou_threshold_label": "0.75",      "area": "all",    "max_dets": 100, "value": 0.421},
    {"metric": "AP", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "small",  "max_dets": 100, "value": 0.231},
    {"metric": "AP", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "medium", "max_dets": 100, "value": 0.426},
    {"metric": "AP", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "large",  "max_dets": 100, "value": 0.553},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "all",    "max_dets": 1,   "value": 0.317},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "all",    "max_dets": 10,  "value": 0.503},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "all",    "max_dets": 100, "value": 0.527},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "small",  "max_dets": 100, "value": 0.341},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "medium", "max_dets": 100, "value": 0.564},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "large",  "max_dets": 100, "value": 0.689}
  ],
  "stats": [0.398, 0.612, 0.421, 0.231, 0.426, 0.553, 0.317, 0.503, 0.527, 0.341, 0.564, 0.689]
}
```

### `iou_type = "keypoints"` (10 lines)

Per ADR-0012 / quirk D5, the keypoint summary drops the `small` area bucket and the multi-cap `AR` rows. `max_dets` resolves to `[20]` by default. The plan order is `AP / AP@.50 / AP@.75 / AP_M / AP_L / AR / AR@.50 / AR@.75 / AR_M / AR_L`.

```json
{
  "version": "1",
  "iou_type": "keypoints",
  "parity_mode": "strict",
  "max_dets": [20],
  "use_cats": true,
  "lines": [
    {"metric": "AP", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "all",    "max_dets": 20, "value": 0.642},
    {"metric": "AP", "iou_threshold": 0.5,  "iou_threshold_label": "0.50",      "area": "all",    "max_dets": 20, "value": 0.864},
    {"metric": "AP", "iou_threshold": 0.75, "iou_threshold_label": "0.75",      "area": "all",    "max_dets": 20, "value": 0.708},
    {"metric": "AP", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "medium", "max_dets": 20, "value": 0.598},
    {"metric": "AP", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "large",  "max_dets": 20, "value": 0.711},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "all",    "max_dets": 20, "value": 0.706},
    {"metric": "AR", "iou_threshold": 0.5,  "iou_threshold_label": "0.50",      "area": "all",    "max_dets": 20, "value": 0.901},
    {"metric": "AR", "iou_threshold": 0.75, "iou_threshold_label": "0.75",      "area": "all",    "max_dets": 20, "value": 0.764},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "medium", "max_dets": 20, "value": 0.659},
    {"metric": "AR", "iou_threshold": null, "iou_threshold_label": "0.50:0.95", "area": "large",  "max_dets": 20, "value": 0.768}
  ],
  "stats": [0.642, 0.864, 0.708, 0.598, 0.711, 0.706, 0.901, 0.764, 0.659, 0.768]
}
```

## Determinism guarantees

ADR-0015 §"Output determinism" pins the following invariants for the JSON formatter; every consumer that archives or diffs the output relies on them:

- **Fixed key order.** Object keys are emitted in the schema-defined order documented above (top-level and per-`Line`), not insertion order and not sorted alphabetically. The `lines` array is in plan order — the same order `Summary::pretty_lines()` produces — never sorted by metric name. Reordering keys in a future PR is a schema-version bump.
- **No timestamps.** No `generated_at`, `started_at`, `wall_clock_seconds`, or any other field whose value changes between two runs of the same input. The eval inputs (GT, DT, flags) define the result's identity.
- **No environment leakage.** No host, user, working directory, or vernier-build-metadata fields. The schema's `version` is the contract surface; the *commit* of vernier that produced the file lives in `Cargo.lock` / the release tag, not in the document.
- **Round-trip-safe float formatting.** `value` fields use Rust's default `{}` for `f64`, which is the shortest representation that round-trips through `f64::from_str`. Stable across the workspace MSRV (rustc 1.83+) and across platforms.
- **Atomic file writes.** When `--emit json=PATH` is used, the file is written to `PATH.tmp.<pid>`, fsynced, and renamed atomically. A concurrent reader either sees the previous contents in full or the new contents in full — never a half-written file.

The contract: byte-equal output for byte-equal input, across runs, machines, and elapsed time. The one well-defined exception is the `version` field, which only changes on schema revisions.

## Schema versioning

`version` tracks the *schema*, not the `vernier-cli` package version. The CLI commits to:

- **Backward-compatible additions** ship under the same `version` value. New fields appended to the top level or to `lines[]` for an existing IoU type do not bump `version`.
- **Renames, removals, or shape changes** bump `version` to the next integer string. The older shape remains the CLI's default until a future release switches the default; the per-formatter knob `--emit json[,version=N]` (ADR-0015 §"Formatter: JSON") is the planned opt-in for newer shapes ahead of that switch. Today the only shipped schema is `"1"` and `--emit json,version=N` has no other accepted values.
- **Older schemas remain consumable** for one schema generation past the default switch. A pipeline that pins `vernier-cli` to a specific 0.0.x patch and stores its output for replay can read the same bytes back through every subsequent 0.0.x patch that keeps `"version": "1"` as the default, and through the first generation after the default flips. The generation after that, parsers may stop accepting `"1"`.

The schema version is decoupled from the package version on purpose — `vernier-cli` ships under 0.0.x patches today (no SemVer guarantees on the package), but `"version": "1"` is a hard contract: reshaping it outside the discipline above breaks the regression-archive use case the JSON formatter is built around.

## Schema v2 — partitioned output (`--manifest`)

ADR-0046 adds a fan-out lane: when `vernier eval --manifest PATH` is passed, the binary emits a *partitioned* JSON document under `"version": "2"`. Un-partitioned eval keeps emitting v1 verbatim — that is the load-bearing byte-stability contract of ADR-0046.

> **Rule:** without `--manifest`, the output is byte-for-byte identical to the v1 document above. The v2 envelope is **only** emitted when a manifest drives the eval.

### v2 top-level fields

| Field         | Type                                  | Notes                                                                                                                                                       |
|---------------|---------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `version`     | string                                | `"2"` for partitioned output.                                                                                                                               |
| `label`       | string or `null`                      | `--label` value stamped on this run. `null` when omitted. `vernier aggregate` joins by this field when present.                                            |
| `iou_type`    | string                                | As v1.                                                                                                                                                       |
| `parity_mode` | string                                | As v1.                                                                                                                                                       |
| `max_dets`    | array of non-negative integers        | As v1.                                                                                                                                                       |
| `use_cats`    | boolean                               | As v1.                                                                                                                                                       |
| `overall`     | [`Overall` object](#overall-subfields) | The un-partitioned summary. Bit-identical to a v1 document over the same `(GT, DT)` pair — load-bearing parity contract per ADR-0046.               |
| `slices`      | array of [`Slice` objects](#slices-subfields) | One entry per partition slice, in canonical order (axis ascending, value ascending, `__unassigned__` last; joint cells follow the marginals).        |

### `overall` subfields

| Field           | Type                       | Notes                                                                          |
|-----------------|----------------------------|--------------------------------------------------------------------------------|
| `lines`         | array of `Line` objects    | Same shape and order as v1's `lines`.                                          |
| `stats`         | array of floats            | Same shape and order as v1's `stats`.                                          |
| `n_images`      | non-negative integer       | Number of dataset images behind the overall summary.                           |
| `n_detections`  | non-negative integer       | Total detection count behind the overall summary (a=0 "all" bucket counted once per detection). |

### `slices[]` subfields

| Field           | Type                       | Notes                                                                                                                |
|-----------------|----------------------------|----------------------------------------------------------------------------------------------------------------------|
| `axis`          | string                     | The manifest axis name. For joint cells, the `::`-joined tuple (`weather::time_of_day`).                              |
| `value`         | string                     | The categorical level. For joint cells, the `::`-joined value tuple (`fog::night`). `__unassigned__` for unassigned. |
| `n_images`      | non-negative integer       | Number of dataset images in this slice.                                                                              |
| `n_detections`  | non-negative integer       | Detection count in this slice (same accounting as `overall.n_detections`).                                            |
| `lines`         | array of `Line` objects    | Same shape and order as v1's `lines`, evaluated over this slice's image subset.                                       |
| `stats`         | array of floats            | Same shape and order as v1's `stats`, evaluated over this slice's image subset.                                       |

### Determinism guarantees (v2)

The v1 determinism rules carry through verbatim. Additionally:

- **Slice order is canonical**, not insertion order: `(axis ascending, value ascending, __unassigned__ last)` for marginals; joint cells follow the marginals in the same canonical order applied to the joined `axis` / `value` strings.
- **`overall` is bit-identical to a v1 run** on the same `(GT, DT)` pair — the partitioned dispatch invokes the same `accumulate + summarize` calls over the un-filtered grid.

### Worked example (v2)

```json
{
  "version": "2",
  "label": "run_2026_05_14",
  "iou_type": "bbox",
  "parity_mode": "strict",
  "max_dets": [1, 10, 100],
  "use_cats": true,
  "overall": {
    "lines": [/* 12 entries, same shape as v1 */],
    "stats": [0.527, 0.728, 0.581, 0.341, 0.566, 0.683, 0.392, 0.624, 0.661, 0.471, 0.706, 0.812],
    "n_images": 5000,
    "n_detections": 31_500
  },
  "slices": [
    {"axis": "weather", "value": "clear",          "n_images": 3700, "n_detections": 22_900, "lines": [/* ... */], "stats": [/* 12 */]},
    {"axis": "weather", "value": "fog",            "n_images": 1100, "n_detections":  7_300, "lines": [/* ... */], "stats": [/* 12 */]},
    {"axis": "weather", "value": "__unassigned__", "n_images":  200, "n_detections":  1_300, "lines": [/* ... */], "stats": [/* 12 */]}
  ]
}
```

See [`docs/reference/manifest-schema.md`](manifest-schema.md) for the manifest input and [`docs/reference/aggregate-schema.md`](aggregate-schema.md) for the `vernier aggregate` companion output.

## Schema v2 — partitioned LRP output (`--metric olrp --manifest`)

ADR-0046 (with ADR-0043 / ADR-0044 / ADR-0045 for the LRP semantics) extends the partition fan-out to the LRP / oLRP metric. When `--metric olrp --manifest PATH` is supplied, the binary emits a *partitioned LRP* JSON document under `"version": "2"`, distinguished from the AP v2 envelope by a `"metric": "olrp"` discriminator.

> **Rule:** `--metric olrp` without `--manifest` keeps emitting the un-partitioned LRP shape under `"version": "1"`. The partitioned-LRP envelope is **only** emitted when both `--metric olrp` and `--manifest` are present.

### LRP v2 top-level fields

| Field         | Type                       | Notes                                                                                                                |
|---------------|----------------------------|----------------------------------------------------------------------------------------------------------------------|
| `version`     | string                     | `"2"` for partitioned output.                                                                                        |
| `metric`      | string                     | `"olrp"` — discriminator for parsers that switch on `(version, metric)`.                                              |
| `label`       | string or `null`           | `--label` value stamped on this run. `null` when omitted.                                                            |
| `iou_type`    | string                     | As v1.                                                                                                                |
| `parity_mode` | string                     | As v1.                                                                                                                |
| `use_cats`    | boolean                    | As v1.                                                                                                                |
| `overall`     | [`LrpOverall` object](#lrp-overall-subfields) | The un-partitioned LRP block. Bit-identical to a single `optimal_lrp_*` call over the same `(GT, DT)`. |
| `slices`      | array of [`LrpSlice` objects](#lrp-slices-subfields) | One entry per partition slice, in the same canonical order as the AP v2 envelope.            |

Note: the LRP envelope does **not** carry the AP `max_dets` top-level field. LRP runs at a single `max_dets_per_image` rung (the top of the resolved ladder); the value is implicit in the kernel-canonical defaults and is recorded inside the resolved `config` for documentation rather than as a wire-level array.

### `LrpOverall` subfields

| Field             | Type                       | Notes                                                                                                |
|-------------------|----------------------------|------------------------------------------------------------------------------------------------------|
| `olrp`            | float                      | Mean per-class oLRP across classes with at least one positive GT (per ADR-0043).                     |
| `olrp_loc`        | float                      | Mean per-class `oLRP_Loc` across classes with at least one TP at the optimal tau.                     |
| `olrp_fp`         | float                      | Mean per-class `oLRP_FP` (same denominator as `olrp_loc`).                                            |
| `olrp_fn`         | float                      | Mean per-class `oLRP_FN` (same denominator as `olrp`).                                                |
| `n_empty_classes` | non-negative integer       | Number of classes with no positive GTs (excluded from the headline means).                            |
| `n_images`        | non-negative integer       | Number of dataset images behind the overall report.                                                    |
| `n_detections`    | non-negative integer       | Total detection count behind the overall report.                                                       |
| `config`          | [`LrpConfig` object](#lrpconfig-subfields) | Resolved configuration (kernel, `tp_threshold`, `tau_grid_len`) — every report self-describes per ADR-0044. |

### `LrpSlice` subfields

| Field             | Type                       | Notes                                                                                                                |
|-------------------|----------------------------|----------------------------------------------------------------------------------------------------------------------|
| `axis`            | string                     | The manifest axis name. For joint cells, the `::`-joined tuple.                                                       |
| `value`           | string                     | The categorical level. For joint cells, the `::`-joined value tuple. `__unassigned__` for unassigned.                |
| `n_images`        | non-negative integer       | Number of dataset images in this slice.                                                                              |
| `n_detections`    | non-negative integer       | Detection count in this slice.                                                                                       |
| `olrp`            | float                      | Per-slice headline oLRP, computed by restricting the LRP decompose walk to this slice's image set.                   |
| `olrp_loc`        | float                      | Per-slice `oLRP_Loc`.                                                                                                 |
| `olrp_fp`         | float                      | Per-slice `oLRP_FP`.                                                                                                  |
| `olrp_fn`         | float                      | Per-slice `oLRP_FN`.                                                                                                  |
| `n_empty_classes` | non-negative integer       | Per-slice count of classes with no positive GTs (a slice may be more sparsely class-covered than the overall dataset). |

### `LrpConfig` subfields

| Field           | Type                  | Notes                                                                                                                |
|-----------------|-----------------------|----------------------------------------------------------------------------------------------------------------------|
| `tp_threshold`  | float                 | IoU/OKS floor for TP — resolved per ADR-0044 (`0.5` for every kernel by default).                                   |
| `tau_grid_len`  | non-negative integer  | Length of the confidence-threshold grid the LRP search ran on (101 for the canonical default grid).                   |
| `kernel`        | string                | Canonical kernel name (`"bbox"` / `"segm"` / `"boundary"` / `"keypoints"`).                                          |

### `per_class` is omitted by default

The un-partitioned LRP v1 envelope ships a `per_class` array (one row per category) carrying the deployable `tau` plus the four per-class decomposition fields. The partitioned LRP v2 envelope **omits `per_class` from both `overall` and `slices`** by default, for two reasons:

- **Size at LVIS scale.** A 1203-category dataset crossed with 8 slices would balloon the document by ~10k per-class rows — the bulk of which downstream `vernier aggregate` flows do not consume.
- **Wrong surface for the partitioned use-case.** The partitioned document is the comparative table — the headline numbers per slice are the reason to slice. Per-class detail is the un-partitioned `--metric olrp` run's job; users who want both spawn both.

A future `--per-class` opt-in flag is anticipated if a workload ever needs the per-class table embedded in the partitioned envelope.

### Determinism guarantees (LRP v2)

The v1 / AP-v2 determinism rules carry through verbatim: fixed key order, no timestamps, no environment leakage, round-trip-safe float formatting, atomic file writes. Additionally:

- **Slice order is canonical**, same rule as the AP v2 envelope: `(axis ascending, value ascending, __unassigned__ last)` for marginals; joint cells follow the marginals.
- **`overall` is bit-identical** to a v1 LRP run on the same `(GT, DT)` pair — the partitioned LRP dispatch invokes the same matching engine once and runs the decompose walk over the un-filtered image set for `overall`.

### Worked example (LRP v2)

```json
{
  "version": "2",
  "metric": "olrp",
  "label": "run_2026_05_14",
  "iou_type": "bbox",
  "parity_mode": "strict",
  "use_cats": true,
  "overall": {
    "olrp": 0.412,
    "olrp_loc": 0.183,
    "olrp_fp": 0.092,
    "olrp_fn": 0.137,
    "n_empty_classes": 0,
    "n_images": 5000,
    "n_detections": 31_500,
    "config": {"tp_threshold": 0.5, "tau_grid_len": 101, "kernel": "bbox"}
  },
  "slices": [
    {"axis": "weather", "value": "clear",          "n_images": 3700, "n_detections": 22_900, "olrp": 0.398, "olrp_loc": 0.177, "olrp_fp": 0.086, "olrp_fn": 0.135, "n_empty_classes": 0},
    {"axis": "weather", "value": "fog",            "n_images": 1100, "n_detections":  7_300, "olrp": 0.487, "olrp_loc": 0.221, "olrp_fp": 0.121, "olrp_fn": 0.145, "n_empty_classes": 2},
    {"axis": "weather", "value": "__unassigned__", "n_images":  200, "n_detections":  1_300, "olrp": 0.453, "olrp_loc": 0.198, "olrp_fp": 0.114, "olrp_fn": 0.141, "n_empty_classes": 4}
  ]
}
```

## See also

- [`docs/adr/0015-vernier-cli.md`](../adr/0015-vernier-cli.md) — the source of truth for the CLI surface, the formatter abstraction, and the determinism contract.
- [`docs/adr/0046-slice-and-aggregate.md`](../adr/0046-slice-and-aggregate.md) — partition manifest and the v1 → v2 schema bump.
- [`docs/reference/coco-summary-stats.md`](coco-summary-stats.md) — the `stats[i]` ↔ slice mapping for the bbox / segm 12-stat plan.
- [`docs/adr/0012-oks-keypoints-surface.md`](../adr/0012-oks-keypoints-surface.md) — keypoints plan order and kernel-canonical `max_dets = [20]`.
- [`docs/adr/0010-boundary-iou-isolated-subsystem.md`](../adr/0010-boundary-iou-isolated-subsystem.md) — boundary IoU metric definition consumed by `iou_type = "boundary"`.
- [`docs/adr/0043-lrp-oracle-and-namespace.md`](../adr/0043-lrp-oracle-and-namespace.md) / [`docs/adr/0044-lrp-thresholds-and-tau-grid.md`](../adr/0044-lrp-thresholds-and-tau-grid.md) / [`docs/adr/0045-lrp-keypoints-shipped.md`](../adr/0045-lrp-keypoints-shipped.md) — the LRP / oLRP semantics, per-kernel defaults, and keypoints support consumed by `--metric olrp`.
