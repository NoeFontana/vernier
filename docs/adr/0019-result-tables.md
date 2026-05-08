# ADR-0019: Result tables — opt-in, Arrow-backed, zero-overhead by default

- **Status:** accepted
- **Date:** 2026-05-02
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors
- **Target landing:** Phase 5, Week 2 (per `docs/explanation/possible-extensions.md`)
- **Maps to:** §3 of `Possible_Extensions`

## Context and problem statement

Phase 5 Week 2 ships per-image, per-class, per-detection, and per-pair
result tables. The capability is described in
`docs/explanation/possible-extensions.md` as the foundational primitive
that hard-example mining, active learning, dashboards, and model-diff
tools all build on top of. ADR-0013 §"What this ADR explicitly does
*not* decide" reserved the design to a follow-up; this is that
follow-up.

The user-visible problem is unchanged from the Possible Extensions
write-up. Pycocotools surfaces summary stats and a
`evalImgs[(catId, areaRng, imgId)] -> dict` structure with a layout
shaped by the 2014 implementation. To answer "which images is my
model failing on?" or "which class is dragging the average down?",
every team writes the same 30 lines of indexing code. Vernier's
extended API has the opportunity to retire that work entirely by
exposing the data as DataFrames, but doing so naively introduces
three risks the project is explicitly built to avoid:

1. **Default-path cost.** A user calling
   `Evaluator(...).evaluate(gt, dt)` to get a `Summary` and shipping
   it to a CI artifact must not pay the cost of building a polars
   DataFrame they never read. The parity baseline for the headline
   performance claim — "5–10× faster than pycocotools" — is the bare
   summary path. Tables cannot regress it.
1. **Top-level dependency drift.** ADR-0018 (calibration) declined to
   pull `polars-rs` into `vernier-ffi` because doing so is a
   top-level dependency change requiring its own ADR-0001 trigger.
   That decision applies equally here. A naive "let Rust build polars
   DataFrames" design forces the dep without addressing the
   discipline.
1. **Schema sprawl.** Every column shipped is a forward-compatibility
   commitment. Faster-coco-eval's `extended_metrics` accumulates
   columns over time and ends up with semantically overlapping
   fields (macro vs micro precision, NaN-filtered vs not). The
   "fewer columns we're confident in" rule cited in the Possible
   Extensions write-up is the correct discipline; we encode it
   explicitly so we don't drift from it.

The architecture already has the data the cheap tables need.
`EvalImageMeta` carries `image_id`, `category_id`, `dt_ids`,
`gt_ids`, and the `dt_matches` / `gt_matches` arrays (per ADR-0005's
locked spine). `Accumulated` carries the `(T, R, K, A, M)` precision
tensor and the `(T, K, A, M)` recall tensor. `StreamingEvaluator`'s
cells store (ADR-0013) keeps the same per-image arrays alive across
updates. The per-image and per-class tables are folds over data that
is already produced in the default path — the question is how to
expose them without making the default path pay for the exposure.

The per-detection and per-pair tables are different. They need data
that is produced inside `evaluate_cell` (the IoU matrix in
particular) and currently discarded after matching. Surfacing them
requires a retention flag that survives the matching/accumulator
boundary.

## Decision drivers

- **Opt-in, zero-overhead by default.** A caller who does not ask for
  a table pays no cost — no extra Rust allocations, no extra Python
  imports, no extra threads, no extra memory in the cells store, no
  extra columns retained across the FFI. This is the headline driver
  and the constraint everything else negotiates around.
- **ADR-0005 invariant.** No edits to
  `crates/vernier-core/src/matching.rs` or
  `crates/vernier-core/src/accumulate.rs`. Tables are a new
  orchestration layer above the spine, the same way streaming was.
- **ADR-0001 §"Add or remove a top-level dependency".** Crosses the
  bar — adding `arrow-rs` to the workspace is a top-level dep
  change. We do it deliberately, with this ADR.
- **ADR-0006 threading.** Table construction runs under `py.detach`
  (GIL dropped). No Python objects touched inside the closure.
- **ADR-0007 drop-in surface.** The `COCOeval` drop-in does not gain
  a `.per_image` accessor. Tables are extended-API only; the drop-in
  remains the synchronous batch shape its consumers are calibrated
  against.
- **ADR-0013 streaming compatibility.** Tables are a fold over the
  per-image cells store. `StreamingEvaluator` and
  `BackgroundEvaluator` get the same `EvalResult` surface their
  batch sibling does, and the determinism contract carries through
  unchanged.
- **Public API surface.** Per ADR-0001 §"Affect the public API" and
  §"Cross the FFI boundary". The new types (`EvalResult`,
  `TablesRequest`) and the `tables=...` keyword on `evaluate()` /
  `finalize()` are public.
- **Honest schema.** Some columns the Possible Extensions write-up
  named — notably "per-image AP" — are not well-defined within the
  COCO framework. The schema does not include them. We err on the
  side of fewer columns we can defend than more columns that drift
  the user toward a bad metric.

## Considered options

The decision has two coupled axes: **how the DataFrame is built**
(the cross-language path) and **how it is opted into** (the
activation model). Each axis is decided independently; the chosen
design is the combination of one option per axis.

### Axis A — Cross-language path

1. **JSON dump from Rust → pandas in Python.** Rust serializes a
   list-of-dicts; Python builds a pandas DataFrame from it. Trivial
   to implement, no new Rust dep. Round-trips a per-detection
   table through JSON parsing — 10 to 50× slower than the columnar
   alternatives. Pandas-only.
1. **`polars-rs` end-to-end.** `vernier-ffi` constructs a
   `polars::DataFrame` and returns it via the `pyo3-polars` bridge.
   Best-in-class single-language ergonomics. Adds the entire polars
   query engine to every wheel — ~12 MB compressed, ~50 MB
   uncompressed — including for users who never touch a table.
   Crosses ADR-0001 hard, with a much larger blast radius than
   `arrow-rs`.
1. **`arrow-rs` (subset) + Arrow PyCapsule Interface.**
   `vernier-ffi` constructs Arrow `RecordBatch`es using the
   minimal `arrow-array` / `arrow-buffer` / `arrow-schema` crates,
   exposes them as PyCapsule objects implementing the Arrow C
   Data Interface. Python's polars / pandas / pyarrow / duckdb all
   consume Arrow PyCapsules zero-copy. Wheel size impact:
   ~3 MB compressed. polars becomes a Python-side optional dep
   (lazy-imported on first DataFrame access).
1. **Numpy structured arrays.** Rust returns numpy structured
   arrays via the `numpy` crate already in the workspace. No new
   dep. Strings (category names) require `object`-dtype arrays
   which break vectorization. Composes poorly with polars / pandas
   columnar ops downstream — the very ops users want.

### Axis B — Activation model

1. **Always built, hidden behind lazy properties.** Every
   `evaluate()` call builds and stores the DataFrames on the
   result; the user pays the cost whether they read them or not.
   Simple but violates the headline driver.
1. **Built lazily on first attribute access from a result that's
   always returned.** `evaluate()` always returns `EvalResult`
   (changes the return type); `EvalResult.per_image` is a
   `cached_property` that builds on read. Cheap when not read,
   but it's a breaking API change — every caller of the existing
   `Evaluator(...).evaluate(...).stats` has to migrate.
1. **Opt-in via a `tables=` keyword that switches the return
   type.** `evaluate()` with no `tables=` argument returns
   `Summary` (existing shape, unchanged). `evaluate(tables=...)`
   returns `EvalResult`. The opt-in is explicit at the call site;
   the default path is bit-identical to what ships today.
1. **Separate method (`evaluate_full`).** Same shape as option 3
   but two methods. Easier to type-narrow, but doubles the public
   surface for every entry point we ship (batch, streaming,
   background — six methods instead of three keywords).

## Decision outcome

**Axis A → option 3.** `arrow-rs` (subset) joins the workspace
dependencies. Tables cross the FFI as Arrow PyCapsules. polars is
a Python-side import, lazy on first DataFrame access, declared as
a `tables` extra (`pip install vernier[tables]`).

**Axis B → option 3.** `tables=` keyword on `evaluate()` /
`finalize()` / `snapshot()`. Default `tables=None` returns
`Summary` (existing shape, unchanged).
`tables=("per_image", "per_class")` returns an `EvalResult` whose
`summary` field is the existing `Summary`. The shape of the keyword
is a tuple of strings; an alias `tables="all"` is provided for the
common case.

The headline consequence of these two decisions together: a user
who does not pass `tables=` pays no DataFrame cost, no `arrow-rs`
allocation cost, and no polars import cost. The default path is
the same code, the same allocations, and the same return type as
the 0.0.1 release.

### What "zero-overhead by default" means concretely

Bench-anchored, not aspirational:

- The default `evaluate()` path (without `tables=`) does not
  construct a single Arrow array, does not allocate a single byte
  in the `tables.rs` module, does not retain a single per-cell
  IoU matrix beyond what the matching engine already discards,
  and does not import polars or pyarrow on either side of the
  FFI.
- The `EvalResult` type does not exist on the default path —
  `evaluate()` with `tables=None` returns the same `PySummary`
  it has always returned.
- The per-cell retention flags (`retain_iou`, `retain_geometry`)
  default to `false` in `EvaluateParams`. Activating them is a
  conscious caller choice that propagates from the keyword on
  `evaluate()` to the spine via the existing params plumbing —
  no new global state, no new fields on `EvalGrid`, no new
  branches inside `match_image`.

These claims are pinned by a benchmark in `tests/python/bench/`
that asserts `evaluate(gt, dt)` (default args) is within 1% of
the `vernier-core` baseline measured at 0.0.1 freeze. The
benchmark fails if anyone — including this ADR's
implementation — silently makes the default path wider.

### Rust core surface

A new module `crates/vernier-core/src/tables.rs`:

```rust
/// Which tables to compute. Empty set = "no tables, return Summary only".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TablesRequest {
    pub per_image: bool,
    pub per_class: bool,
    pub per_detection: bool,
    pub per_pair: bool,
}

impl TablesRequest {
    pub const NONE: Self = Self { per_image: false, per_class: false,
                                  per_detection: false, per_pair: false };
    pub const CHEAP: Self = Self { per_image: true, per_class: true,
                                   per_detection: false, per_pair: false };
    pub const ALL: Self = Self { per_image: true, per_class: true,
                                 per_detection: true, per_pair: true };

    pub fn requires_iou_retention(&self) -> bool {
        self.per_pair || self.per_detection
    }
}

/// Configuration knobs for the expensive tables. Inert when the
/// corresponding bool in TablesRequest is false.
#[derive(Debug, Clone)]
pub struct TablesConfig {
    /// IoU floor for `per_pair`. Pairs with `iou < iou_floor` are
    /// dropped from the table.
    pub per_pair_iou_floor: f64,           // default 0.1
    /// Hard cap on per_pair row count. Exceeding it is an error,
    /// not a silent truncation.
    pub per_pair_max_rows: usize,          // default 10_000_000
    /// Whether to include bbox geometry columns in per_detection.
    /// Off by default — most callers don't need them and the cost
    /// is non-trivial for D in the millions.
    pub per_detection_with_geometry: bool, // default false
}

/// Concrete row types — ndarray-friendly columnar buffers, not
/// row-of-struct. Keeps the Arrow conversion in vernier-ffi a
/// straight column-by-column move.
pub struct PerImageTable {
    pub image_id: Vec<i64>,
    pub n_gt: Vec<u32>,
    pub n_dt: Vec<u32>,
    pub tp_at_50: Vec<u32>,
    pub fp_at_50: Vec<u32>,
    pub fn_at_50: Vec<u32>,
    pub tp_at_75: Vec<u32>,
    pub fp_at_75: Vec<u32>,
    pub fn_at_75: Vec<u32>,
    pub tp_mean_iou: Vec<u32>,    // mean across the T-axis
}

pub struct PerClassTable {
    pub category_id: Vec<i64>,
    pub category_name: Vec<String>,
    pub ap: Vec<f64>,
    pub ap50: Vec<f64>,
    pub ap75: Vec<f64>,
    pub ap_s: Vec<f64>,
    pub ap_m: Vec<f64>,
    pub ap_l: Vec<f64>,
    pub ar_max_1: Vec<f64>,
    pub ar_max_10: Vec<f64>,
    pub ar_max_100: Vec<f64>,
    pub n_gt: Vec<u32>,
    pub n_dt: Vec<u32>,
}

pub struct PerDetectionTable {
    pub detection_id: Vec<i64>,
    pub image_id: Vec<i64>,
    pub category_id: Vec<i64>,
    pub score: Vec<f64>,
    pub area: Vec<f64>,
    /// "tp" | "fp" | "ignored", encoded as Arrow dictionary.
    pub match_status_at_50: Vec<MatchStatus>,
    pub matched_gt_id_at_50: Vec<Option<i64>>,
    /// IoU to the best-overlapping GT regardless of class. None if
    /// retain_iou=false. Always None for "fp" rows where there was
    /// no GT in the image.
    pub best_iou: Vec<Option<f64>>,
    /// Geometry columns. Empty Vec when per_detection_with_geometry
    /// is false; populated otherwise.
    pub bbox: Option<BboxColumns>,
}

pub struct PerPairTable {
    pub detection_id: Vec<i64>,
    pub ground_truth_id: Vec<i64>,
    pub image_id: Vec<i64>,
    pub category_id_dt: Vec<i64>,
    pub category_id_gt: Vec<i64>,
    pub iou: Vec<f64>,
}

/// Build the requested tables from the locked-spine outputs.
/// Empty `request` returns empty tables without touching the grid.
pub fn build_tables(
    grid: &EvalGrid,
    accum: &Accumulated,
    dataset: &CocoDataset,
    retained_ious: Option<&RetainedIous>,
    request: TablesRequest,
    config: &TablesConfig,
) -> Result<Tables, EvalError>;
```

Two non-obvious points about this shape:

- **No business logic creeps into `vernier-ffi`.** `tables.rs` is
  pure Rust over types `vernier-core` already exposes. The FFI
  crate adds an Arrow conversion layer; the FFI conversion is
  mechanical (one column at a time, each column a `Vec` of
  primitives or strings).
- **`RetainedIous` is a new opt-in artifact.** When
  `request.requires_iou_retention()` is true, the orchestrator
  asks the existing `evaluate_with` machinery to keep the IoU
  matrix for each cell. Default is to discard, exactly as today.
  The retention flag rides on `EvaluateParams` as a new optional
  field; the spine reads it once in `evaluate_with` and either
  copies or drops the matrix at the end of each cell. No edits
  to `match_image` or `accumulate`.

### FFI surface

A new module `crates/vernier-ffi/src/tables.rs` exposes one
PyO3 entry point per table:

```rust
#[pyfunction]
fn per_image_to_arrow_pycapsule(grid: &PyEvalGrid, accum: &PyAccumulated)
    -> PyResult<Bound<'_, PyAny>>;

#[pyfunction]
fn per_class_to_arrow_pycapsule(grid: &PyEvalGrid, accum: &PyAccumulated,
                                dataset: &PyCocoDataset)
    -> PyResult<Bound<'_, PyAny>>;

// per_detection and per_pair require RetainedIous; signature carries it.
```

Each entry point follows the same shape: take the locked-spine
outputs, build the corresponding `*Table` via `vernier_core::tables`
under `py.detach`, then convert column-by-column into an Arrow
`RecordBatch`, then return a `PyCapsule` exposing the Arrow C
Data Interface (`__arrow_c_array__`) per the
[Arrow PyCapsule Interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html)
spec.

The `arrow` crate's `arrow-pyarrow` feature is *not* used —
that path requires pyarrow at the FFI boundary, which would
make pyarrow a hard runtime dep of the wheel. The PyCapsule
interface is the protocol-only path: any consumer that
implements `from_arrow(...)` or accepts a PyCapsule (polars,
pandas ≥ 2.2, duckdb, pyarrow itself) reads the data zero-copy
without a transitive Rust binding.

### Python surface

The main change is one new keyword argument and one new return
type. Both batch and streaming/background evaluators inherit
the same shape:

```python
# python/vernier/_types.py
TableName = Literal["per_image", "per_class", "per_detection", "per_pair"]

@dataclass(frozen=True)
class TablesConfig:
    per_pair_iou_floor: float = 0.1
    per_pair_max_rows: int = 10_000_000
    per_detection_with_geometry: bool = False

@dataclass(frozen=True)
class EvalResult:
    """Returned only when `tables=...` is non-None on evaluate()."""
    summary: Summary

    # Lazy properties — polars is imported on first access.
    @cached_property
    def per_image(self) -> "polars.DataFrame": ...
    @cached_property
    def per_class(self) -> "polars.DataFrame": ...
    @cached_property
    def per_detection(self) -> "polars.DataFrame": ...
    @cached_property
    def per_pair(self) -> "polars.DataFrame": ...

    # Convenience pass-through; preserves the existing
    # `result.stats` shape callers may already use on Summary.
    @property
    def stats(self) -> list[float]:
        return self.summary.stats
```

`Evaluator` gains one keyword:

```python
class Evaluator:
    @overload
    def evaluate(self, gt: bytes, dt: bytes,
                 *, tables: None = ...) -> Summary: ...
    @overload
    def evaluate(self, gt: bytes, dt: bytes,
                 *, tables: tuple[TableName, ...] | Literal["all"],
                 tables_config: TablesConfig = ...) -> EvalResult: ...
```

`StreamingEvaluator` and `BackgroundEvaluator` gain matching
keywords on `finalize()` and `snapshot()`. The overloaded return
type lets existing callers keep their `Summary` typing without
edits; new callers who pass `tables=` get the wider type
automatically.

Three deliberate API choices:

- **Lazy polars import.** `from vernier import Evaluator` does
  not import polars. The first attribute access on `EvalResult`
  imports it, raising a structured `ImportError` with the
  install command (`pip install vernier[tables]`) if it isn't
  present. Users who only ever call `evaluate(tables=None)`
  never trigger the import.
- **`to_pandas()` is not a vernier method.** Polars already
  exposes `df.to_pandas()`; we don't reimplement it. Pandas
  goes from "first-class" in the Possible Extensions doc to
  "one method call away on every result", which is the same
  thing operationally and keeps the surface narrower.
- **No string-to-DataFrame coercion at the keyword.** `tables=`
  takes a tuple, not a single string (except the literal
  `"all"` alias). This is consistent with how `max_dets=` is
  shaped on `Evaluator` and avoids the "did you mean a tuple
  of one or a string?" footgun.

### Schemas

Each table has a pinned column list, pinned types, and pinned
semantics. Schemas are versioned via a `_schema_version`
constant attached to the Arrow `RecordBatch` metadata; the CLI
JSON output schema (per ADR-0015) gets a sibling `tables_schema`
section that mirrors the same versions.

#### `per_image`

One row per image in the GT dataset, summed across categories,
with the COCO `all` area range. Columns the Possible Extensions
doc named that we *don't* ship in v0.5:

- **`AP` / `AP50`.** Per-image AP is not well-defined in the
  COCO framework: AP integrates a PR curve, and a PR curve from
  a single image is degenerate (most images have ≤ 1 detection
  at any given recall point). Faster-coco-eval and
  pycocotools-Cli both expose per-image AP and both produce
  numerically misleading values that users then plot and act
  on. We intentionally omit them; users who want them can
  compute their own from the per_pair table without us
  encoding a bad default. `docs/explanation/why-no-per-image-ap.md`
  is shipped alongside this ADR to record the reasoning.

| Column        | Type | Semantics                                                 |
|---------------|------|-----------------------------------------------------------|
| `image_id`    | i64  | COCO image id.                                            |
| `n_gt`        | u32  | Non-ignore GT count (summed across categories, area=all). |
| `n_dt`        | u32  | DT count, post-area-filter, pre-maxDets.                  |
| `tp_at_50`    | u32  | TPs at IoU=0.50 (matched, not ignored).                   |
| `fp_at_50`    | u32  | FPs at IoU=0.50 (unmatched, not ignored).                 |
| `fn_at_50`    | u32  | `n_gt - tp_at_50`.                                        |
| `tp_at_75`    | u32  | TPs at IoU=0.75.                                          |
| `fp_at_75`    | u32  | FPs at IoU=0.75.                                          |
| `fn_at_75`    | u32  | `n_gt - tp_at_75`.                                        |
| `tp_mean_iou` | u32  | Mean across the T-axis: `floor(mean(tp_t))`.              |

10 columns.

#### `per_class`

One row per category. Mirrors `summarize_detection`'s 12-stat
layout per category, plus support counts. Cells with no GTs
inherit pycocotools' `-1` sentinel (quirk **C5**) and are
emitted as Arrow nulls.

| Column          | Type   | Semantics                              |
|-----------------|--------|----------------------------------------|
| `category_id`   | i64    | COCO category id.                      |
| `category_name` | string | From `CategoryMeta::name`.             |
| `ap`            | f64    | AP@.50:.95, area=all, maxDets=last.    |
| `ap50`          | f64    | AP@.50, area=all, maxDets=last.        |
| `ap75`          | f64    | AP@.75, area=all, maxDets=last.        |
| `ap_s`          | f64    | AP@.50:.95, area=small.                |
| `ap_m`          | f64    | AP@.50:.95, area=medium.               |
| `ap_l`          | f64    | AP@.50:.95, area=large.                |
| `ar_max_1`      | f64    | AR with maxDets=1.                     |
| `ar_max_10`     | f64    | AR with maxDets=10.                    |
| `ar_max_100`    | f64    | AR with maxDets=100.                   |
| `n_gt`          | u32    | Non-ignore GT count for this category. |
| `n_dt`          | u32    | DT count for this category.            |

13 columns.

#### `per_detection`

One row per detection. Opt-in via
`tables=("per_detection", ...)`. Geometry columns gated
behind `per_detection_with_geometry=True`.

| Column                 | Type                    | Semantics                                                                                                                  |
|------------------------|-------------------------|----------------------------------------------------------------------------------------------------------------------------|
| `detection_id`         | i64                     | DT id (pycocotools 1-based for batch; stream-position for streaming, per ADR-0013).                                        |
| `image_id`             | i64                     |                                                                                                                            |
| `category_id`          | i64                     | DT's claimed category.                                                                                                     |
| `score`                | f64                     | Confidence.                                                                                                                |
| `area`                 | f64                     | DT area, kernel-defined.                                                                                                   |
| `match_status_at_50`   | dict<utf8>              | `"tp"` / `"fp"` / `"ignored"`.                                                                                             |
| `matched_gt_id_at_50`  | i64 nullable            | GT id at IoU=0.50, null on FP.                                                                                             |
| `best_iou`             | f64 nullable            | Max IoU to any same-class GT in the same image. Null if `retain_iou=false` or if there are no same-class GTs in the image. |
| `bbox_xywh` (optional) | fixed_size_list<f64, 4> | Only when `per_detection_with_geometry=True`.                                                                              |

8 always-present columns + 1 optional geometry column.

#### `per_pair`

Every (DT, GT) pair where IoU ≥ `iou_floor` (default 0.1). Opt-in
via `tables=("per_pair", ...)`. Always class-restricted: pairs
across categories are excluded (matching the matching engine's
behavior).

| Column            | Type | Semantics                                        |
|-------------------|------|--------------------------------------------------|
| `detection_id`    | i64  |                                                  |
| `ground_truth_id` | i64  |                                                  |
| `image_id`        | i64  |                                                  |
| `category_id`     | i64  | Same for DT and GT — pairs are class-restricted. |
| `iou`             | f64  | Raw IoU as the kernel produces it.               |

5 columns.

A row count exceeding `per_pair_max_rows` raises
`PerPairOverflowError` — a new exception in `vernier`. We do
not silently truncate; users with genuinely massive pair
counts raise the cap explicitly with `tables_config=...`.

### Streaming and background integration

`StreamingEvaluator.finalize(tables=...)` and
`StreamingEvaluator.snapshot(tables=...)` accept the same keyword
as `Evaluator.evaluate(tables=...)` and return either `Summary`
(default) or `EvalResult` (with `tables`). The cells store
already holds the per-image arrays the cheap tables need; no
new state is added for `per_image` or `per_class`.

The wrinkle is `per_detection` and `per_pair`. Both require
the IoU matrix (or, for `per_detection`, the per-DT max IoU
column). Streaming has to know about retention at evaluator
construction time, before any `update()` runs — once the
matrix is discarded after a cell, we can't reconstruct it
later from the cells store.

The contract:

```python
ev = StreamingEvaluator(gt, retain_iou=True)         # opt-in at construction
for batch in train_loader:
    ev.update(predictions(batch))
result = ev.finalize(tables="all")                   # works
```

```python
ev = StreamingEvaluator(gt)                          # default, no retention
for batch in train_loader:
    ev.update(predictions(batch))
result = ev.finalize(tables=("per_image", "per_class"))  # works — cheap tables
result = ev.finalize(tables=("per_pair",))           # raises ValueError
```

The `ValueError` is structured: it names the missing flag and
suggests the construction-site fix. We deliberately reject at
`finalize()` rather than at `update()` because the user may
genuinely want the cheap tables on a non-retaining stream;
catching the misuse at the moment of demand is the
least-surprising behavior.

`BackgroundEvaluator` (per ADR-0014) inherits the same shape —
the worker thread builds the tables before sending the result
back, GIL is dropped during construction, and the `peek=True`
snapshot fast path emits the `summary` field without populating
the DataFrames at all (consistent with ADR-0014's running-mode
contract).

### Memory and performance contract

For COCO val2017 (5 000 images, 80 categories, ~37 k detections
across the largest models we benchmark):

| Table           | Rows  | Cols | Memory (Arrow) | Build time |
|-----------------|-------|------|----------------|------------|
| `per_image`     | 5 000 | 10   | ~250 KB        | < 5 ms     |
| `per_class`     | 80    | 13   | ~10 KB         | < 1 ms     |
| `per_detection` | 37 k  | 8    | ~3 MB          | ~30 ms     |
| `per_pair`      | ~1 M  | 5    | ~40 MB         | ~200 ms    |

These figures pin the benchmarks; the build-time numbers are
asserted in CI with a 2× tolerance for runner variance. The
`per_pair` row count above assumes `iou_floor=0.1`; lower
floors blow this up quickly, which is why the hard cap exists.

For LVIS-scale workloads (1 200 categories, ~250 k detections),
the cheap tables stay sub-megabyte; `per_detection` reaches
~25 MB; `per_pair` at default `iou_floor` can hit hundreds of
millions of rows, which is precisely the case
`per_pair_max_rows` exists to fail loudly on. We do not auto-
chunk in v0.5; the follow-up ADR for chunked Arrow IPC emit
covers that case if it materializes.

### Parity and determinism contract

Tables are deterministic functions of:

- The cells store (per-image cell arrays, ADR-0013).
- The accumulator output (precision/recall tensors, locked spine).
- The dataset metadata (category names, image ids).
- The retained IoU matrices (when applicable).

In **strict mode**, two runs over the same input produce
bit-identical Arrow buffers (ignoring buffer addresses, which
are runtime values and not part of the data contract). In
**aligned** and **corrected** modes, the underlying tensors
match within the existing 4-ULP tolerance from ADR-0004; the
tables inherit that tolerance verbatim.

Stream-order sensitivity (ADR-0013) carries through unchanged.
`finalize(tables=...)` is bit-equal to
`Evaluator.evaluate(..., tables=...)` over the same detection
set. Mid-stream `snapshot(tables=...)` is bit-equal to a batch
evaluation of the submitted subset, with the same boundary-
ULP caveat the Summary-only snapshot already has. The
`per_detection` table's `detection_id` column inherits the
streaming-vs-batch numbering disagreement noted in ADR-0013
§"Detection identifiers"; we document it on the column rather
than try to paper over it.

`per_pair` is **not** affected by stream order even in non-
strict mode: the IoU values are computed inside `evaluate_cell`
from the raw GT/DT geometry, and matching does not reorder
them.

### Test plan

The harness extends the existing parity infrastructure rather
than forking it.

- **Schema golden test.** Each table's schema (column names,
  types, nullability) is committed to
  `tests/python/tables/schemas/<table>.json`. The test asserts
  the live Arrow schema against the golden; bumping the schema
  is a deliberate edit to that file, gated by review.
- **Cross-language round-trip.** Build a table in Rust → emit
  Arrow PyCapsule → consume in polars → consume in pandas (via
  `polars.to_pandas()`) → consume in duckdb (via
  `duckdb.from_arrow(...)`). All four reads agree on row
  count, column names, and per-cell values.
- **Zero-overhead benchmark.** `evaluate(gt, dt)` (default,
  no `tables=`) wall-clock and allocation count must be within
  1% of a 0.0.1-frozen baseline. The benchmark is in
  `tests/python/bench/test_zero_overhead_default_path.py` and
  runs in CI on the same dedicated benchmark runner the
  pulp-vs-scalar test uses.
- **Streaming/batch table equality.** Build a table from
  `Evaluator.evaluate(..., tables="all")`; build the same
  table from `StreamingEvaluator(retain_iou=True)` driven
  with the same detections in submission order;
  `polars.assert_frame_equal` with check_exact=True under
  strict mode.
- **Retention-required error path.** Construct a streaming
  evaluator without `retain_iou=True`, feed it some
  detections, call `finalize(tables=("per_pair",))`. Assert
  `ValueError` with the message naming `retain_iou`.
- **Per-pair cap.** Construct a fixture with a tiny
  `iou_floor` and `per_pair_max_rows=100`. Submit a workload
  that produces > 100 pairs; assert `PerPairOverflowError`.
- **Per-image AP omission.** Assert the schema does not
  contain `ap` or `ap_50` columns. Asserts the project's
  position is encoded in the test, not just the docs.
- **Lazy polars import.** Import `vernier`, run an
  `evaluate(tables=None)` cycle, assert `polars` is not in
  `sys.modules`. Then access `result.per_image` on a
  `tables="all"` run; assert it is.
- **`patch_pycocotools` shim is unaffected.** `COCOeval`
  drop-in does not gain `tables=`; the shim's surface is the
  pycocotools surface, period (per ADR-0007).

A new fixture class lives at `tests/python/tables/`, parallel
to `tests/python/parity/` rather than nested inside it — the
DataFrame surface is parity-equivalent by construction (same
underlying tensors, different presentation), so the parity
harness does not need to re-validate it.

### Crate dependency

`arrow-rs` joins the workspace dependencies, with feature
selection pinned to the minimum surface we actually use:

```toml
# Cargo.toml workspace.dependencies
arrow-array  = { version = "55", default-features = false }
arrow-buffer = { version = "55", default-features = false }
arrow-schema = { version = "55", default-features = false }
```

We deliberately do not depend on the `arrow` umbrella crate,
on `arrow-ipc`, on `arrow-csv`, or on `arrow-pyarrow`. The
PyCapsule path needs only the array/buffer/schema crates; the
others are an order of magnitude more code, more compile time,
and more wheel size for capabilities we don't expose.

`pyo3-stub-gen` (already in the workspace) generates the
PyCapsule-returning entry points correctly; no new stub
infrastructure is required.

### What this ADR explicitly does *not* decide

- **`vernier-viz`.** Plotting (PR curves, F1-confidence
  curves, confusion-matrix heatmaps) is explicitly punted to
  a separate optional package. The Possible Extensions doc
  argues for this and we agree: visualization choices are
  personal, the data is the durable contribution, and
  matplotlib is not a dependency we want to drag into the
  core wheel. If a `vernier[viz]` extra ships, it lands as
  its own ADR.
- **Chunked Arrow IPC for very-large `per_detection` /
  `per_pair`.** v0.5 emits one `RecordBatch` per table.
  Workloads beyond ~10 M `per_pair` rows that genuinely need
  larger tables motivate a chunked emit + streaming
  IPC writer, which is a follow-up.
- **On-disk Parquet emit.** Free for the user via polars
  (`result.per_pair.write_parquet(...)`); we do not need a
  vernier method.
- **DuckDB / pyarrow integration helpers.** Free for the user
  via the PyCapsule interface; we ship no helper, we ship a
  protocol-conforming primitive.
- **Per-image AP / AP50.** Deliberate omission, documented
  separately in `docs/explanation/why-no-per-image-ap.md`.
- **Custom user-defined columns.** A future ADR may add a
  `vernier-tables-extras` plugin point. Out of scope here.
- **Tables on the `COCOeval` drop-in.** Out of scope per
  ADR-0007. The drop-in is the migration shim, not the
  extended API.

### Phased landing plan

1. **Week 2.1 — `arrow-rs` adoption.** Workspace dep ADR (this
   one) merges; arrow-rs lands in workspace deps; one
   trivially-tested `per_class` table builds end-to-end. CI
   wheel-size delta is measured and pinned.
1. **Week 2.2 — `per_image` and `per_class` end-to-end.**
   Schema goldens, cross-language round-trip, zero-overhead
   benchmark.
1. **Week 2.3 — IoU retention plumbing.** New
   `EvaluateParams::retain_iou`; spine reads it; cells store
   stores it; streaming evaluator's construction-time
   `retain_iou=True` flag.
1. **Week 2.4 — `per_detection` and `per_pair`.** Schema,
   tests, overflow path.
1. **Week 2.5 — streaming integration + docs.**
   `finalize(tables=...)` on `StreamingEvaluator` and
   `BackgroundEvaluator`. How-to guide in
   `docs/how-to/result-tables.md`.

Each step is mostly self-contained and can be skipped or
deferred without breaking the others. If Week 2 slips, the
release ships fewer tables rather than slipping the date.

### Consequences

- **Positive.** The Possible Extensions Week 2 commitment
  ships in roughly 400 lines of new logic (300 in
  `vernier-core`, 100 in `vernier-ffi`) plus 200 lines of
  Python wrapper. Users who only want a `Summary` see no
  change in the API, no change in the wheel runtime, and no
  change in the import surface. Users who want tables get
  polars, pandas, duckdb, or pyarrow zero-copy via one
  protocol — we ship the data, the user picks the consumer.
  Streaming, background, and batch share one
  `EvalResult` surface; Phase 5's other features (TIDE,
  calibration) get a foundational primitive that Phase 5
  Weeks 3–4 build on top of rather than reimplementing.
- **Negative.** Adds `arrow-rs` (subset) to the workspace —
  a real top-level dep with its own release cadence, its own
  bug surface, and ~3 MB of wheel weight. Users who pip-install
  vernier on a slow link feel that. polars-as-extra means
  some users who expected polars-by-default will hit a
  delayed `ImportError` the first time they touch a
  DataFrame — we mitigate with a structured, install-command-
  carrying error, but the friction is real. The Arrow
  PyCapsule interface is younger than pickle-based interop;
  we depend on polars / pandas / duckdb keeping their
  consumer code stable. Per-image AP omission will surprise
  some users who expect it and will produce GitHub issues
  asking for it; the explanation doc absorbs the conversation.
- **Neutral.** `EvalResult` and `Summary` are sibling types,
  not subtype-related. Documentation positions
  `Summary` as the answer for "I want stats" and
  `EvalResult` as the answer for "I want to introspect."
  Users learning the API see two return types, not one with
  optional fields; the difference is justified by the
  difference in cost.

## Pros and cons of the options

### Axis A — Cross-language path

**A3 (chosen) — `arrow-rs` (subset) + Arrow PyCapsule Interface**

- 👍 Zero-copy across the FFI; one protocol, four consumers
  (polars, pandas, duckdb, pyarrow).
- 👍 ~3 MB wheel impact; ~1 s of compile time per crate.
- 👍 `arrow-rs` is load-bearing for half the data ecosystem
  (DataFusion, Polars, DuckDB-rs); maintenance is not at risk.
- 👎 PyCapsule consumer support requires polars ≥ 1.0,
  pandas ≥ 2.2, duckdb ≥ 1.0. v0.5 ships matching minimum
  versions in the `tables` extra.

**A1 — JSON dump → pandas**

- 👍 Trivial to implement; no new dep.
- 👎 10–50× slower per-detection. Pandas-only.
- 👎 Round-trip through JSON loses type fidelity (i64 → object
  for null-bearing columns).

**A2 — `polars-rs` end-to-end**

- 👍 Best single-language ergonomics; one dep, one type.
- 👎 ~50 MB wheel impact (uncompressed). Pulled in for users
  who never touch a table.
- 👎 polars's ABI evolves quickly; pinning the workspace to
  it is a maintenance commitment we don't need.

**A4 — numpy structured arrays**

- 👍 Smallest possible new surface; numpy is already in the
  workspace.
- 👎 Object-dtype for strings is not vectorized; downstream
  joins materialize.
- 👎 Composes poorly with polars / pandas / duckdb; users
  re-wrap immediately.

### Axis B — Activation model

**B3 (chosen) — `tables=` keyword switching the return type**

- 👍 Default path is bit-identical to 0.0.1. Zero overhead by
  construction.
- 👍 Single keyword on every entry point; no method
  duplication.
- 👍 Overloads narrow the return type per call site.
- 👎 `tables=` is one more thing to teach. Mitigated by
  examples in the how-to guide.

**B1 — always built**

- 👍 Simplest API; no keyword.
- 👎 Violates the headline driver. Eats per-call cost users
  did not opt into.

**B2 — always-returned `EvalResult` with lazy properties**

- 👍 Cheap when not read.
- 👎 Breaking API change to every existing caller; we are not
  willing to break the headline `evaluate(...).stats` shape
  for the cleaner type.

**B4 — separate `evaluate_full` method**

- 👍 Easier to type-narrow than overloads.
- 👎 Doubles the public-method count for every entry point
  (batch, streaming, background, drop-in if we extended it).

## Links and references

- ADR-0001 — Record architecture decisions (§"Affect the
  public API", §"Cross the FFI boundary", §"Add or remove a
  top-level dependency"). This ADR triggers all three.
- ADR-0002 — Three-tier parity model. Tables inherit the
  existing tiers; they introduce no new parity properties.
- ADR-0004 — Numerical layout policy. The 4-ULP tolerance
  this ADR inherits for aligned/corrected modes.
- ADR-0005 — Lock the `Similarity` trait and matching-engine
  API for Phases 1–3. Tables are an orchestration layer
  above the spine; the architectural test is satisfied.
- ADR-0006 — Threading model. Table construction runs under
  `py.detach`; lazy polars import means the GIL hold for
  Arrow conversion is bounded by row count, not by polars
  startup.
- ADR-0007 — `patch_pycocotools` policy. Drop-in does not
  gain tables; tables are extended-API only.
- ADR-0013 — Streaming evaluator. Cells store is the
  substrate; `finalize(tables=...)` and
  `snapshot(tables=...)` inherit its determinism contract
  verbatim. §"Detection identifiers" is the source of the
  streaming-vs-batch `detection_id` caveat.
- ADR-0014 — Background evaluator. Worker thread builds
  tables; `peek=True` snapshot does not.
- ADR-0015 — `vernier-cli`. The CLI gains a `--tables` flag
  in a follow-up; the JSON output schema (per
  `docs/reference/cli-output-schema.md`) gains a
  `tables_schema` section behind the same flag.
- ADR-0018 — Calibration metrics. The "no polars-rs in FFI"
  position established there is the precedent this ADR
  carries forward (subject to a deliberate revisit: arrow-rs
  joins, polars-rs does not).
- `crates/vernier-core/src/evaluate.rs` — `EvalImageMeta`,
  `EvalGrid`, `EvaluateParams`, and `evaluate_with` are the
  load-bearing inputs to `tables.rs`; `evaluate_cell` is the
  internal site at which IoU retention is evaluated.
- `crates/vernier-core/src/accumulate.rs` — `Accumulated`
  is the load-bearing input to `per_class`.
- `crates/vernier-core/src/stream.rs` — `PerImageEvalStore`
  is the streaming substrate that makes
  `finalize(tables=...)` free.
- `crates/vernier-core/src/dataset.rs` — `CategoryMeta::name`
  feeds the `per_class.category_name` column.
- `docs/engineering/pycocotools-quirks.md` §C5 — the `-1`
  sentinel for absent-category cells, surfaced as Arrow
  nulls in `per_class`.
- `docs/explanation/possible-extensions.md` §3 — the Phase 5
  capability ranking that motivates this ADR's place in the
  schedule.
- [Arrow PyCapsule Interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html)
  — the cross-language protocol this ADR adopts.
