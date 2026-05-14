# ADR-0046: Slice-and-aggregate — partition manifest, `vernier eval --manifest`, and the `vernier aggregate` verb

- **Status:** accepted
- **Date:** 2026-05-14
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

Two capabilities were tracked separately in the metrics strategy:

- **Scenario slicing** (v2 strategy Priority 5) — per-image / per-instance
  categorical breakdowns: AP by `weather`, `time_of_day`, `sensor`,
  `scene_tag`; mIoU in rain vs clear; recall by occlusion bucket.
- **Robustness / corruption decomposition** (v2 strategy Priority 7) —
  mPC / rPC (mean / relative Performance under Corruption, Michaelis
  et al. NeurIPS-W 2019) over corrupted-input evaluation runs.

A design pass shows these are the **fan-out and fan-in halves of one
pipeline**, sharing one artifact — a *partition manifest*. Scenario
slicing partitions one `(GT, DT)` pair by a per-image categorical
attribute into N sub-evaluations; corruption decomposition aggregates
N already-computed evaluations. The shared back-half — "produce a
labelled, comparative, per-slice table" — is the `vernier aggregate`
verb that ADR-0015 explicitly deferred. This ADR ratifies the whole
pipeline as one design.

### What constrains the design

- **ADR-0005** locks the matching engine and `accumulate.rs`. No new
  accumulator tensor axis; the spine may be *invoked* over subsets but
  not *modified*.
- **ADR-0015** names `vernier aggregate` as the deferred verb for
  "aggregation across many runs … consumes the initial-surface JSON
  output", commits the CLI to byte-determinism, and rules
  prediction-running explicitly out of scope ("not a prediction
  runner").
- **ADR-0016** makes `Breakdown` `f64`-keyed / class-id-keyed,
  *explicitly rejected* the per-annotation closure (its Option 4
  — labels and bounds vanish at the FFI boundary, and it inverts the
  data-flow ownership), and named categorical keys as needing "another
  ADR".
- **ADR-0018 / ADR-0019** settled the output-format posture: result
  tables are Arrow `RecordBatch`es exposed via the Arrow PyCapsule
  Interface; `arrow-rs` is a workspace dep and `polars-rs` is not;
  on-disk Parquet emit is *not* a vernier method ("we ship the data,
  the user picks the serializer"). This ADR inherits that posture
  rather than re-deciding it.
- **ADR-0026** (LVIS) established the load-bearing precedent that
  subset evaluation is "a category-subset filter at summarize time" —
  the locked spine is invoked over a subset, never modified.
- **ADR-0039** anticipated that the CLI JSON schema "bumps to v2 only
  when a `Breakdown` is actually emitted in CLI output."

### The problem

Provide categorical scenario slicing and cross-run aggregation with
(a) the best downstream UX — no dataset-JSON surgery, spreadsheet-native
inputs, Arrow-native results, one mental model; (b) performance that
scales to LVIS-class datasets crossed with many slices; (c) zero
modification to the ADR-0005-locked spine; (d) no gratuitous new
permanent public dataset surface and no new Rust dependency; (e) the
right surface for each user — the CLI for CI / language-agnostic
scripting, the Python API for the DataFrame-native analysis workflow.

## Decision drivers

- **One mental model.** "Slice by anything, compare slices" should be
  one tool, not three. The scenario table, the corruption sweep, and
  (future) the regression-vs-baseline diff are the same shape; a
  bespoke `--slice` filter that does not share the aggregate back-half
  hands the user three incompatible tools for one need.
- **Spine is untouchable.** ADR-0005. Categorical slicing must be an
  orchestration-level fan-out, never a new tensor axis.
- **Performance at LVIS scale × many slices.** Matching is the
  expensive step; the accumulator tensor is large. The design must do
  matching once and must not hold N large accumulators live.
- **No new permanent dataset surface for a per-evaluation concern.**
  Attribute tags are a property of *an evaluation*, not of *a
  dataset*; baking them into the COCO dataset model is the wrong
  lifetime.
- **Vendor-transcribable.** The manifest must be mechanically
  producible from a FiftyOne `ViewExpression` export, a Kolena
  stratification, or a pandas `DataFrame` — per the v2 strategy §3
  Segment C positioning (vernier *consumes* vendor slices, does not
  compete with the triage UX).
- **Inherit, don't re-decide, the output posture.** ADR-0018 /
  ADR-0019 already settled Arrow-via-PyCapsule, no `polars-rs` in FFI,
  no vendor Parquet method. The slice / aggregate outputs are tabular;
  they are ADR-0019-shaped tables, not a new format question.
- **Surface fits the user.** The advanced slice-and-aggregate
  workflow is DataFrame-native end to end (metadata in, results out);
  the CLI's "skip the interpreter" rationale does not apply to a user
  already in Python. The CLI must stay the CI / language-agnostic
  lane; the Python API must be the ergonomic analysis lane.
- **Determinism.** ADR-0015's byte-determinism commitments
  (stable key order, no timestamps, atomic writes) extend to every
  new JSON output this ADR introduces; the Arrow path inherits
  ADR-0019's *value*-determinism contract (strict-mode bit-identical
  buffers, 4-ULP aligned / corrected) rather than a byte contract.

## Considered options

Decided across five axes.

### A. Where do per-image attribute tags live?

- **A1.** Extend the COCO dataset model with an `attributes` map on
  images / annotations. New permanent public dataset surface;
  ADR-0001-significant.
- **A2.** Sidecar **partition manifest** — a separate file mapping a
  key to `{axis: value}` pairs. No dataset-schema change. The manifest
  *is* the attribute model, and it is also the exact artifact the
  fan-in verb needs.
- **A3.** Categorical `Breakdown` — the ADR-0016 successor; force
  categorical keys into the accumulator A-axis.

### B. One verb or two?

- **B1.** One `vernier aggregate` with two modes (evaluate-and-aggregate;
  aggregate-only).
- **B2.** Two verbs: `vernier eval --manifest` (partition fan-out, one
  job: evaluate) + `vernier aggregate` (fan-in over result documents,
  one job: aggregate).

### C. Partition-and-evaluate performance model

- **C1.** Partition into N datasets, run `vernier eval` N times.
- **C2.** One matching pass; hold N accumulators live; fold each
  image's per-image result into the accumulators of every slice it
  belongs to.
- **C3.** One matching pass; persist compact per-image match results
  (reusing the per-image cell store the streaming evaluator, ADR-0013,
  already maintains); then N accumulation+summarize passes with **one
  accumulator live at a time**, each pass filtering the cell store to
  a slice's image set — the ADR-0026 subset-at-summarize-time pattern.

### D. Manifest file format

- **D1.** JSON-records only.
- **D2.** JSON-records (canonical) **+ CSV** accepted and converted at
  parse time — robotics attribute tables are spreadsheet-native.

### E. Cross-product cells

- **E1.** Always emit the full cross-product of all manifest axes.
- **E2.** Per-axis marginals by default; cross-product opt-in via
  `--cross axisA,axisB`.

### F. Output data model and serialization

- **F1.** JSON-only — both `vernier eval --manifest` and `vernier
  aggregate` emit only the versioned JSON document.
- **F2.** Arrow-native data model, inheriting ADR-0019: the
  slice and aggregate results are Arrow `RecordBatch`es exposed
  through the Arrow PyCapsule Interface; the CLI additionally emits a
  JSON projection. vernier takes **no `parquet` crate dependency** —
  Parquet is the user's `polars` / `pyarrow` writing the Arrow handle
  vernier hands them.
- **F3.** vernier takes a `parquet` crate dependency and the CLI emits
  Parquet directly.

### G. Primary surface for the advanced workflow

- **G1.** CLI-primary — the manifest is always a file, results are
  always JSON, the Python API is a thin wrapper.
- **G2.** Two deliberate lanes: the **Python API is the rich,
  ergonomic surface** for the slice-and-aggregate workflow
  (manifest constructible from a DataFrame, results Arrow-native,
  `vernier.aggregate` over in-memory result objects); the **CLI is the
  scriptable, language-agnostic lane** (file-based manifests, JSON
  output) for CI and non-Python shops.

## Decision outcome

Chosen: **A2 + B2 + C3 + D2 + E2 + F2 + G2.**

### A2 — the partition manifest is the attribute model

A sidecar manifest sidesteps a new permanent public dataset surface
entirely: no ADR-0001 trigger, no COCO-JSON surgery for the user. It
is also the precise artifact `vernier aggregate` consumes for fan-in,
so **one schema serves scenario slicing, corruption decomposition, and
future regression diffing**. A1 adds permanent surface for a concern
that is inherently per-evaluation. A3 forces categorical keys into the
ADR-0005-locked accumulator and re-opens the axis ADR-0016 explicitly
deferred; the orchestration-level partition is a strictly better
answer because it **composes with** the existing numeric `Breakdown`
rather than competing with it (see *Composition with `Breakdown`*
below). **A2 renders the ADR-0016 categorical-`Breakdown` successor
unnecessary.**

#### Manifest schema (`manifest_version: "1"`)

A versioned table, one row per key:

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

- `key_kind` is `"image_id"` (consumed by `vernier eval --manifest`)
  or `"result"` (consumed by `vernier aggregate`). One schema, the
  discriminator selects the consumer.
- Remaining columns are **axis names**; cells are categorical string
  values. Numeric slicing is *not* this mechanism's job — that is the
  `Breakdown` axis (ADR-0016), which composes (below).
- **CSV form** (D2): first column `key`, header row supplies axis
  names, `key_kind` passed as a flag. Converted to the canonical JSON
  shape at parse time.
- **Unassigned keys.** Dataset images absent from the manifest land in
  an explicit `__unassigned__` slice on every axis — never silently
  dropped (the "no silent data loss" discipline from the panoptic /
  semantic quirks surveys).
- **Unknown keys.** Manifest rows whose key is absent from the dataset
  / result set produce a typed warning on stderr and are skipped.
- **Determinism.** Slice output order is `(axis name ascending, then
  value ascending)`, with `__unassigned__` sorted last on each axis.

### B2 — two verbs, one job each

`vernier eval --manifest` evaluates (now optionally partitioned);
`vernier aggregate` aggregates pre-existing result documents. This
matches ADR-0015's framing of `aggregate` verbatim ("consumes the
initial-surface JSON output"). B1's single mega-verb that both
evaluates and aggregates is misnamed and conflates responsibilities.

*The two verbs below are the **CLI lane** — file-based manifests, JSON
output, deterministic, language-agnostic. The Arrow-native Python lane
is described under F2 / G2 below.*

#### `vernier eval --manifest` (CLI lane)

```
vernier eval --gt gt.json --dt dt.json --iou-type bbox \
  --manifest weather.json [--cross weather,time_of_day] \
  [--label run_2026_05_14] [--emit json=result.json]
```

- Produces a result document carrying a `slices` array — each entry is
  `{axis, value, lines, stats}` — **plus an `overall` entry** that is
  bit-identical to today's un-partitioned single-eval output. The
  existing output is a strict subset of the partitioned output.
- **Schema bump v1 → v2.** ADR-0039 anticipated this: a partitioned
  result emits per-slice rows, which is "a `Breakdown` actually
  emitted in CLI output". Un-partitioned `vernier eval` stays
  `"version": "1"` verbatim; the `--json-schema-version` opt-in from
  ADR-0015 governs the transition.
- `--label` (new, additive `vernier eval` flag) stamps a `label` field
  into the result document so `vernier aggregate` can join to a
  `key_kind: "result"` manifest by label rather than by file path.

#### `vernier aggregate` (CLI lane)

```
vernier aggregate --manifest corruptions.json --results 'runs/*.json' \
  [--baseline clean] [--metric AP] [--emit json=summary.json]
```

- Fan-in: read N result documents, join to the manifest by `label`
  (falling back to file path when a result has no `--label`), group by
  axis value, emit a comparative per-slice table.
- **mPC** = mean of the chosen metric over the non-baseline slices of
  an axis. **rPC** = mPC / (baseline-slice metric). The relative
  reductions are emitted **only** when `--baseline` names a slice
  value (e.g. `--baseline clean`); without it, the comparative table
  is emitted without the relative columns.
- New output schema (`aggregate_version: "1"`), inheriting all
  ADR-0015 byte-determinism commitments.
- This verb is also the future home for **cross-run regression
  diffing** (compare two result documents / two manifests). Named here
  as a future consumer; its comparison semantics are its own ADR.

### C3 — one matching pass, persist per-image results, N cheap summarize passes

Matching is the expensive `O(images × GT × DT)` step — it must run
**once**. The accumulator tensor is large: ≈78 MB per accumulator at
COCO scale (`T·R·K·A·M` ≈ 10·101·80·4·3 f64 cells), ≈1.2 GB at
LVIS scale (1203 categories). Holding N of them live (C2) is fine for
COCO × a few slices but **fatal for LVIS × many slices** (N=20 →
≈23 GB). C1 (N full eval runs) re-matches every image once per slice
*membership* — acceptable for a single disjoint partition, but
quadratic-ish for the multi-axis cross-tabulated slicing a robotics
regression suite actually wants.

C3 reuses the per-image cell store the streaming evaluator (ADR-0013)
already maintains: run the matching pass once to populate the cell
store, then run N accumulation+summarize passes, **one accumulator
live at a time**, each filtering the cell store to a slice's image
set. Compact per-image match results are `O(total detections)` —
megabytes, not gigabytes. The result: **1× matching, 1× per-image-result
memory, 1× accumulator memory, N cheap accumulation passes.**

C3 stays ADR-0005-safe: it *invokes* the locked matching engine and
accumulator without modifying them — exactly the ADR-0026
subset-at-summarize-time precedent, with the subset axis being images
rather than categories.

### D2 — JSON canonical, CSV accepted

Robotics attribute tables are spreadsheet-native; JSON-records-only
adds friction for the primary user. JSON-records stays canonical
(ADR-0015 JSON discipline); CSV is accepted and converted at parse
time. Both are pure data with no schema-version coupling beyond
`manifest_version`.

### E2 — marginals by default, cross-product opt-in

The cross-product cell count is the product of axis cardinalities — it
explodes. Per-axis marginal slices are what the user reads first;
`--cross weather,time_of_day` opts into the joint cells when the user
specifically wants them. E1 would make a 3-axis manifest emit
dozens-to-hundreds of cells by default.

### F2 — Arrow-native data model, no `parquet` dependency

This is not a fresh decision — it is **inheriting ADR-0019**. Result
tables (`per_class`, `per_image`, `per_detection`, `per_pair`) are
already Arrow `RecordBatch`es exposed through the Arrow PyCapsule
Interface (`__arrow_c_array__`), consumed zero-copy by polars / pandas
/ duckdb / pyarrow. ADR-0019 explicitly settled that **on-disk Parquet
emit is not a vernier method** — "we ship the data, the user picks the
serializer" — and ADR-0018 + ADR-0019 lock the dependency posture:
`arrow-rs` is a workspace dep, `polars-rs` is not, and the
`arrow-pyarrow` feature is not used (it would make pyarrow a hard
runtime dep of the wheel; PyCapsule is the protocol-only path).

The slice-and-aggregate outputs are tabular by nature — a `slices`
result is one row per `(axis, value)` cell with the stat columns
wide; an `aggregate` result is one row per `(run, axis, value)` with
mPC / rPC columns. Both are exactly ADR-0019-shaped tables. So:

- The slice and aggregate results get **Arrow `RecordBatch` schemas**,
  pinned as goldens under `tests/python/tables/schemas/` exactly like
  the existing per-class / per-image schemas. The Arrow schema
  metadata carries a `vernier.schema_version` key — the Arrow-native
  equivalent of the JSON `version` field.
- vernier takes **no `parquet` crate dependency** and ships **no
  `.write_parquet()` method**. The user calls `.write_parquet()` on
  the polars frame they got for free from the PyCapsule handle. This
  is F3 rejected: a `parquet` crate dep is heavy, and Parquet
  byte-determinism (page sizing, dictionary-encoding decisions,
  compression, the embedded `created_by` string) is fragile across
  `arrow` / `parquet` crate versions — a bad thing to put a contract
  on. The *data* is deterministic (ADR-0019's strict-mode bit-identical
  buffers / 4-ULP aligned-corrected contract carries over verbatim);
  the *Parquet bytes* are the user's serializer's problem.
- The **CLI keeps JSON** as its deterministic contract surface
  (ADR-0015). CLI Arrow-IPC output (`--emit arrow=...`) is a possible
  additive formatter later — Arrow IPC needs only `arrow-rs`, already
  a workspace dep, though it would add `arrow-rs` to `vernier-cli`'s
  dep set — but it is **not** MVP and would be explicitly exempt from
  byte-determinism. Parquet-from-CLI is never in scope.

Parquet remains the right *archival* format for a regression-tracking
bucket — it is just produced by the user's `polars` / `pyarrow` /
`duckdb` on the Arrow handle vernier hands them, not by vernier. For a
non-Python CLI shop, Parquet conversion of the JSON output is a
documented one-line `duckdb` recipe, not a vernier feature —
consistent with ADR-0015 already deferring "Parquet / Arrow output …
to a separate tool."

### G2 — Python API is the rich surface; CLI is the scriptable lane

The slice-and-aggregate *advanced* workflow is: (1) construct a
partition — usually from per-image metadata the user **already has in
a DataFrame**; (2) run partitioned eval; (3) pull results **into a
DataFrame**; (4) plot / compare / append to a regression store. Steps
1, 3, and 4 are DataFrame-native. Forcing them through the CLI means a
filesystem round-trip at both ends — write a manifest JSON only to
hand it back to a subprocess, write a result JSON only to re-read and
`json_normalize` it. The CLI's reason to exist (ADR-0015: "skip the
Python interpreter") **does not apply to this user** — they are
already in Python, already holding a DataFrame.

So the two lanes are deliberate and neither is "primary" in the
abstract — they serve different users — but **for this workflow the
Python API is favored**:

- **Python lane (rich).** Each paradigm's `Evaluator.evaluate(...)`
  gains a `manifest=` parameter that accepts a file path, a dict,
  **or any object exposing the Arrow PyCapsule interface** — a polars
  or pandas DataFrame of image metadata passes straight in, no sidecar
  file. The returned `EvalResult` gains a `.slices` property: an Arrow
  `RecordBatch`, same PyCapsule pattern as ADR-0019's existing tables.
  A top-level `vernier.aggregate(results, manifest, baseline=...)`
  takes in-memory result objects (or Arrow tables) and returns an
  Arrow table. The user does `pl.from_arrow(result.slices)` →
  `.write_parquet(...)` — zero-copy in, one-line Parquet out.
- **CLI lane (scriptable).** File-based manifests (JSON / CSV), JSON
  output, byte-deterministic. This is the CI lane ("run partitioned
  eval nightly, dump JSON, an aggregation job globs them") and the
  language-agnostic lane (a Rust / Go perception shop). Deliberately
  the *minimum viable* projection — no manifest DSL, no `--emit
  parquet`. Richness lives in Python; the CLI stays lean (the same
  spirit as ADR-0015 keeping `clap` out of `vernier-core`).

`vernier.aggregate` is cross-paradigm (it aggregates instance,
panoptic, or semantic results), so it lives at the top level, not
under a paradigm submodule — consistent with ADR-0029's namespace
logic. G1 (CLI-primary, thin Python wrapper) is rejected because it
inverts the ergonomics for the exact user the feature is for.

### Python API surface

```python
from vernier.instance import Evaluator
import polars as pl

# Manifest from a DataFrame the user already has — no sidecar file.
meta = pl.read_parquet("image_metadata.parquet")  # image_id, weather, time_of_day, ...
result = Evaluator().evaluate(gt, dt, manifest=meta)   # manifest= accepts PyCapsule objects

slices = pl.from_arrow(result.slices)   # zero-copy Arrow RecordBatch → polars
slices.write_parquet("run_2026_05_14_slices.parquet")  # user's polars, not vernier

# Fan-in over many in-memory results (or Arrow tables, or JSON paths).
summary = vernier.aggregate(
    results=[r_clean, r_fog, r_noise],
    manifest=corruption_meta,
    baseline="clean",        # enables rPC
)
pl.from_arrow(summary).write_parquet("robustness.parquet")
```

The CLI lane is the same pipeline expressed for scripts: `vernier eval
--manifest m.json` → JSON v2, `vernier aggregate --results 'runs/*.json'`
→ JSON. Same verbs, same manifest schema, same metrics — two
serializations of one pipeline.

### Composition with `Breakdown`

The manifest partition is **orchestration-level and categorical**; the
`Breakdown` axis (ADR-0016) is **accumulator-level and numeric /
class-id keyed**. They compose cleanly:

```
vernier eval --manifest weather.json --area-ranges <breakdown>
```

gives, *per weather slice*, the full S/M/L (or custom) area breakdown.
This is the concrete reason A2 renders the categorical-`Breakdown`
successor unnecessary: the orchestration-level partition is the
categorical answer, and it *layers over* the numeric axis rather than
replacing it.

## Consequences

### Positive

- One artifact — the partition manifest — serves scenario slicing,
  corruption decomposition, and (future) regression diffing. One
  mental model, one schema to learn.
- No new permanent public dataset surface; no COCO-JSON surgery for
  users. The attribute model is a sidecar by construction.
- ADR-0016's deferred categorical-`Breakdown` successor is rendered
  unnecessary — the orchestration-level partition is the better
  answer and composes with the shipped numeric axis.
- The ADR-0005 spine is untouched: C3 invokes the locked matching
  engine and accumulator over image subsets (the ADR-0026 pattern),
  never modifies them.
- The manifest is vendor-transcribable: FiftyOne / Kolena / pandas →
  manifest is a mechanical transcription.
- Performance scales: 1× matching, 1× accumulator memory, even for
  LVIS-class datasets crossed with many slices.
- **No new Rust dependency.** F2 inherits the ADR-0019 Arrow posture
  wholesale — `arrow-rs` is already a workspace dep; no `parquet`
  crate, no `polars-rs` in FFI. The slice / aggregate tables are just
  more ADR-0019-shaped `RecordBatch`es.
- **Zero-copy in and out on the Python lane.** Manifest accepts any
  Arrow-PyCapsule object (a polars / pandas DataFrame passes straight
  in); results are PyCapsule `RecordBatch`es. The advanced user never
  touches the filesystem unless they choose to — and Parquet is their
  one-line `.write_parquet()`, on their serializer, not vernier's.

### Negative

- The CLI JSON schema bumps to v2 for *partitioned* eval output.
  Anticipated by ADR-0039, but still a real consumer-facing change;
  mitigated by un-partitioned output staying v1 verbatim and the
  `--json-schema-version` opt-in.
- Three schema surfaces to version and maintain: the two JSON schemas
  (`manifest_version`, `aggregate_version`) and the Arrow schemas for
  the `slices` / `aggregate` tables (golden-pinned per ADR-0019,
  carrying `vernier.schema_version` in Arrow metadata).
- C3 introduces an internal per-image-results materialization stage
  that the single-shot eval path does not have — more moving parts,
  even though it reuses the ADR-0013 cell store.
- `--label` is a new (additive) flag on `vernier eval`.
- Two lanes (CLI JSON, Python Arrow) mean the pipeline has two
  serializations to keep behaviourally in lockstep — the same
  CLI-vs-API duality ADR-0019 already lives with, but real overhead.

### Neutral

- `vernier aggregate` consumes pre-computed result documents; the user
  is responsible for producing the N corrupted-input result documents.
  vernier never generates corruptions — consistent with ADR-0015 "not
  a prediction runner". This is a deliberate scope line, not a gap.
- Parquet is the right archival format for a regression-tracking
  bucket — but it is produced by the user's `polars` / `duckdb` on the
  Arrow handle, never by vernier. For non-Python CLI shops, JSON →
  Parquet is a documented `duckdb` one-liner.

## Pros and cons of the options

**A1 (dataset-model attributes)** — Pro: attributes travel with the
dataset. Con: permanent public surface for a per-evaluation concern;
ADR-0001-significant; every downstream dataset consumer inherits the
field. **A3 (categorical `Breakdown`)** — Pro: one slicing concept.
Con: touches the ADR-0005-locked accumulator; re-opens the axis
ADR-0016 deferred; competes with rather than composes with the numeric
axis. **A2 (chosen)** — Pro: no new dataset surface, one artifact for
the whole pipeline, vendor-transcribable. Con: a sidecar file the user
must keep in sync with the dataset.

**B1 (one mega-verb)** — Pro: one command to learn. Con: a verb named
`aggregate` that also evaluates is misnamed; conflates two
responsibilities. **B2 (chosen)** — Pro: each verb has one job;
matches ADR-0015's framing. Con: users learn two verbs.

**C1 (N eval runs)** — Pro: trivial, reuses `vernier eval` unchanged.
Con: re-matches per slice membership; quadratic-ish for multi-axis
slicing. **C2 (N live accumulators)** — Pro: single pass, simple.
Con: N × accumulator memory; fatal at LVIS scale × many slices.
**C3 (chosen)** — Pro: 1× matching, 1× accumulator memory, scales.
Con: a per-image-results materialization stage.

**D1 (JSON only)** — Pro: one parser. Con: friction for the
spreadsheet-native primary user. **D2 (chosen)** — Pro: meets the user
where their data already is. Con: a CSV adapter to maintain.

**E1 (full cross-product)** — Pro: nothing to opt into. Con: cell
count explodes by default. **E2 (chosen)** — Pro: readable default,
joint cells on demand. Con: `--cross` is one more flag.

**F1 (JSON-only)** — Pro: one output path. Con: forces the
analysis user through `json_normalize`; ignores that the outputs are
tabular and ADR-0019 already established the Arrow path. **F3
(`parquet` crate dep, CLI emits Parquet)** — Pro: Parquet straight
from the CLI. Con: heavy new Rust dep; Parquet byte-determinism is
fragile across crate versions — a bad contract surface. **F2
(chosen)** — Pro: inherits ADR-0019 wholesale, no new dep, zero-copy
to the user's DataFrame, Parquet is one user-side line. Con: a third
schema surface (Arrow schemas) to golden-pin.

**G1 (CLI-primary, thin Python wrapper)** — Pro: one surface to
document. Con: inverts the ergonomics for the exact user the feature
serves — a filesystem round-trip at both ends of a DataFrame-native
workflow. **G2 (chosen)** — Pro: each lane fits its user (Python =
interactive analysis, CLI = CI / language-agnostic); matches
ADR-0019's existing CLI-vs-API split. Con: two serializations of one
pipeline to keep in lockstep.

## What this ADR explicitly does *not* decide

- **A dataset-attribute model.** The sidecar manifest is deliberately
  a sidecar. Promoting attributes into the dataset model is a separate
  decision if it is ever wanted; this ADR does not foreclose it but
  does not need it.
- **Corruption generators.** vernier consumes corrupted-input
  *predictions*; generating COCO-C / Cityscapes-C / Pascal-C images is
  an inference-pipeline concern, out of scope per ADR-0015.
- **A `parquet` crate dependency or a `.write_parquet()` method.**
  F2 decides vernier ships Arrow `RecordBatch`es (the ADR-0019
  posture) and the user serializes — Parquet, Feather, whatever — with
  their own `polars` / `pyarrow` / `duckdb`. vernier never takes the
  `parquet` crate.
- **CLI Arrow-IPC output (`--emit arrow=...`).** Possible additive
  formatter later (needs `arrow-rs` in `vernier-cli`, which is not
  there today); not MVP, and would be explicitly exempt from
  byte-determinism. Parquet-from-CLI is never in scope.
- **Parquet dataset management** — partitioning, appending, schema
  evolution of a growing regression-tracking bucket. That is the
  user's data-lake concern; vernier hands out per-run Arrow tables and
  documents a recipe, it does not manage a dataset.
- **`vernier-viz` / plotting.** Inherited from ADR-0019's deferral:
  the data is the durable contribution; plotting is a separate
  optional package if it ever ships.
- **Chunked Arrow IPC for very large `slices` / `aggregate` tables.**
  Inherits ADR-0019's "one `RecordBatch` per table" posture; a
  streaming-IPC follow-up covers it if a workload ever needs it.
- **Cross-run regression diffing semantics.** Named as a future
  `vernier aggregate` consumer; what counts as a regression, and the
  tolerance model, is its own ADR.
- **The categorical-`Breakdown` successor (ADR-0016).** Rendered
  unnecessary by the orchestration-level partition. If a future axis
  genuinely needs categorical keys *inside* the accumulator tensor,
  that remains a separate ADR — but there is no current need.
- **Streaming `vernier aggregate`.** NDJSON envelopes over a stream of
  result documents stay deferred per ADR-0015's format discussion.

## Links and references

- ADR-0001 — significance gate (new public surfaces, schema bumps).
- ADR-0002 — parity model; every slice inherits the run's parity mode.
- ADR-0005 — locked matching engine / accumulator; C3 invokes, never
  modifies.
- ADR-0013 — streaming evaluator; its per-image cell store is the
  materialization C3 reuses.
- ADR-0015 — `vernier` CLI; `vernier aggregate` is its deferred verb,
  and its byte-determinism commitments extend to every output here.
- ADR-0016 — `Breakdown`; categorical keys deferred (rendered
  unnecessary here), numeric axis is the composition target.
- ADR-0018 — established "no `polars-rs` in FFI"; F2 carries that
  dependency posture forward.
- ADR-0019 — result tables; the Arrow-`RecordBatch`-via-PyCapsule
  pattern and the "we ship the data, the user picks the serializer"
  position that F2 inherits wholesale. The `slices` / `aggregate`
  tables are new ADR-0019-shaped tables, not new infrastructure.
- ADR-0026 — LVIS; the subset-evaluation-at-summarize-time precedent
  C3 follows.
- ADR-0039 — `Breakdown` Python lift; anticipated the CLI schema
  v1 → v2 bump this ADR triggers for partitioned output.
- `docs/how-to/result-tables.md` — the Arrow PyCapsule consumer
  pattern (`pl.from_arrow`, `to_pandas`, `duckdb.from_arrow`) the
  Python lane reuses verbatim.
- `docs/reference/cli-output-schema.md` — gains a v2 entry for the
  `slices` / `overall` shape; `manifest_version` and
  `aggregate_version` get their own reference pages; the `slices` /
  `aggregate` Arrow schemas are golden-pinned under
  `tests/python/tables/schemas/`.
