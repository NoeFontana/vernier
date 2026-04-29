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

## See also

- [`docs/adr/0015-vernier-cli.md`](../adr/0015-vernier-cli.md) — the source of truth for the CLI surface, the formatter abstraction, and the determinism contract.
- [`docs/reference/coco-summary-stats.md`](coco-summary-stats.md) — the `stats[i]` ↔ slice mapping for the bbox / segm 12-stat plan.
- [`docs/adr/0012-oks-keypoints-surface.md`](../adr/0012-oks-keypoints-surface.md) — keypoints plan order and kernel-canonical `max_dets = [20]`.
- [`docs/adr/0010-boundary-iou-isolated-subsystem.md`](../adr/0010-boundary-iou-isolated-subsystem.md) — boundary IoU metric definition consumed by `iou_type = "boundary"`.
