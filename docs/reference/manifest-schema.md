# Partition manifest schema (`manifest_version: "1"`)

ADR-0046 introduces the *partition manifest*, the single artifact that drives both `vernier eval --manifest` (per-image scenario slicing) and `vernier aggregate` (cross-run corruption / regression fan-in). One schema, two consumers — the `key_kind` field is the discriminator.

This page is the field-by-field reference. See [`docs/adr/0046-slice-and-aggregate.md`](../adr/0046-slice-and-aggregate.md) for the design rationale.

## Canonical wire format (JSON)

```json
{
  "manifest_version": "1",
  "key_kind": "image_id",
  "rows": [
    {"key": 100, "weather": "fog",   "time_of_day": "night"},
    {"key": 101, "weather": "clear", "time_of_day": "day"}
  ]
}
```

| Field              | Type                                  | Notes                                                                                                                                                       |
|--------------------|---------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `manifest_version` | string                                | Schema version pin. `"1"` is the only accepted value today.                                                                                                  |
| `key_kind`         | string, `"image_id"` or `"result"`    | Discriminator. `"image_id"` → consumed by `vernier eval --manifest`. `"result"` → consumed by `vernier aggregate`.                                          |
| `rows`             | array of objects                      | One row per key. Each row carries the primary `key` column plus one cell per **axis** (every other column header).                                          |

### `rows[]` shape

Each entry of `rows` is an object whose keys are the column names. The first column **must** be `key`; every other column is an *axis* (`weather`, `time_of_day`, etc.) whose values are categorical strings.

| Cell type        | Required shape                                                                                |
|------------------|-----------------------------------------------------------------------------------------------|
| `key` (when `key_kind="image_id"`) | JSON integer. Matches the COCO `images[].id` field. Stringified integers (`"100"`) are also accepted. |
| `key` (when `key_kind="result"`)   | JSON string. The run label produced by `vernier eval --label NAME`.                  |
| axis cells       | JSON string. Numeric axis values are **rejected** (numeric slicing is the `Breakdown` axis, ADR-0016, not this surface). |

#### Mandatory axes-on-every-row rule

Manifests must be **rectangular**: every row carries the same axis-column set. Ragged manifests (some rows missing an axis cell) are rejected with a typed config error. A missing cell is ambiguous — *is the image unassigned to that value, or did the user forget the column?* — and forcing the user to fill it makes the answer explicit.

#### `__unassigned__` rule

Dataset images that no manifest row covers land in an explicit `__unassigned__` slice on every axis, never silently dropped. The slice is materialized in every emitted document so the partition is exhaustive — adding a manifest row that mentions a new value never changes the totals of the other values on the same axis.

#### Unknown-key warnings

Rows whose `key` is absent from the live dataset / result set emit a *warning* on stderr (CLI) or a Python `warnings.warn` (Python lane) and are skipped. The manifest is allowed to drift from the dataset without being invalid — a stale spreadsheet that still mentions retired image ids is annoying, not catastrophic.

## CSV form

Robotics attribute tables are spreadsheet-native; the CSV form is accepted on both verbs and converted to the canonical JSON shape at parse time.

```csv
key,weather,time_of_day
100,fog,night
101,clear,day
```

- The first column header **must** be `key`. The parser dispatches on the file extension (`.csv` → CSV path, `.json` or no extension → JSON path).
- Every other column is an axis. Every cell is a categorical string value. The header row supplies the axis names verbatim — `weather`, `time_of_day`, etc.
- `key_kind` is not in the file; the calling verb declares it (`vernier eval --manifest` declares `image_id`, `vernier aggregate` declares `result`).
- BOM (`EF BB BF`) and Windows line endings (`\r\n`) are tolerated. RFC-4180 quoted fields have their surrounding quotes stripped.
- Ragged rows, empty cells in an axis column, and non-integer `key` cells (under `key_kind=image_id`) are rejected at parse time.

## Cross-product (`--cross`) syntax

The CLI emits per-axis marginals by default. To opt into joint cells over a tuple of axes, pass `--cross AXIS_A,AXIS_B`:

```bash
vernier eval --gt gt.json --dt dt.json --iou-type bbox \
  --manifest weather_time.json \
  --cross weather,time_of_day
```

Each `--cross` flag declares one tuple of at least two axes (single-axis crosses are rejected — that is just a marginal, already emitted). The flag is repeatable for multiple cross-products. The joint slice's axis name is the `::`-joined tuple (`weather::time_of_day`); its value is the `::`-joined value tuple (`fog::night`). A joint `__unassigned__` bucket collects dataset images that no joint cell of the tuple covered, mirroring the marginal rule.

A 256-slice cap (`SLICES_CAP`) guards against typo-driven combinatorial explosions; partitions over that limit must split into multiple runs.

## Determinism

Slice output order is `(axis ascending, value ascending)`, with `__unassigned__` sorted last on each axis. Joint cells follow the marginals, in the same canonical order applied to the joined axis / value strings. The CLI's byte-determinism contract (ADR-0015 §"Output determinism") extends to every partitioned output the manifest drives — stable key order, stable float formatting, atomic file writes.

## See also

- [`docs/adr/0046-slice-and-aggregate.md`](../adr/0046-slice-and-aggregate.md) — the source of truth for the partition / aggregate design.
- [`docs/reference/cli-output-schema.md`](cli-output-schema.md) — the v2 envelope that partitioned eval emits.
- [`docs/reference/aggregate-schema.md`](aggregate-schema.md) — the `vernier aggregate` JSON schema.
