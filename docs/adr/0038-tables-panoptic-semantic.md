# ADR-0038: Result tables for panoptic and semantic — per-class only, sibling result types

- **Status:** accepted
- **Date:** 2026-05-09
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors
- **Amends:** ADR-0019 §"What this ADR explicitly does *not* decide"
- **Related:** ADR-0025 (panoptic API), ADR-0028 (sem-seg), ADR-0029 (namespace), ADR-0036 (mmsegmentation oracle), ADR-0037 (semantic kernel class-id)

## Context and problem statement

ADR-0019 shipped result tables for the instance-detection paradigm — four tables (`per_image`, `per_class`, `per_detection`, `per_pair`) exposed via `Evaluator.evaluate(tables=...)`, returning an `EvalResult` whose DataFrame fields are lazy polars views over Arrow PyCapsules. The capability is the foundational primitive for hard-example mining, dashboards, model-diff tools, and the calibration / TIDE work that follows in Phase 5.

ADR-0019 explicitly deferred the same surface for the other paradigms:

> Tables on the `COCOeval` drop-in. Out of scope per ADR-0007. The drop-in is the migration shim, not the extended API. *(implicit deferral for panoptic and semantic noted at landing review)*

ADR-0025 (panoptic) and ADR-0028 (sem-seg) each closed with the same line: "Arrow result tables for PQ / for sem-seg. Each is a separate follow-up; ADR-0019's table surface applies cleanly when those land."

The user-visible problem is the same one ADR-0019 named, restated for the other paradigms:

- **Panoptic.** "Which class is dragging PQ down?" requires walking `summary.per_class: BTreeMap<CategoryId, ClassPanopticStats>` by hand. Every team writes the same indexing code; some users have already filed issues asking for a DataFrame view.
- **Semantic.** "Per-class IoU + TP/FP/FN pixels" is the canonical mmseg / ADE20K output shape. The vendored oracle (ADR-0036) emits exactly that table; vernier today exposes the data but not the table.

The data both questions need is **already retained at summarize time**. The deferral in ADR-0019 was a scoping decision made before the panoptic and semantic implementations had stabilized; with both shipped (ADR-0025 accepted, ADR-0028 accepted, ADR-0037 generalizing the kernel), the deferral now stands in the way of a no-cost extension. This ADR closes that deferral.

What this ADR is *not*: a generalization. The four ADR-0019 tables do not generalize cleanly. `per_detection` and `per_pair` are artifacts of the IoU-curve matching flow (instance + LVIS + future keypoints); they have no semantic counterpart in PQ matching (one-to-one over IoU > 0.5) or in semantic confusion (no detections at all). Pretending they do — by, say, defining a paradigm-agnostic `tables=("per_pair",)` that returns an empty frame for panoptic — would be worse than honest column pinning.

## Decision drivers

- **ADR-0019's zero-overhead invariant carries through unchanged.** Every paradigm's `evaluate()` without a `tables=` keyword pays no DataFrame cost, no Arrow allocation, no polars import. The bench at `tests/python/bench/test_zero_overhead_default_path.py` extends to cover panoptic and semantic; the 1% regression budget is the same.
- **ADR-0029 namespace split is load-bearing.** Each paradigm has its own `Summary` type (`vernier.instance.Summary`, `vernier.panoptic.Summary`, `vernier.semantic.Summary`). Result types follow the same split. No shared base class, no `Union` return types — just three sibling `EvalResult` types, one per paradigm.
- **ADR-0001 §"Affect the public API."** New types (`vernier.panoptic.EvalResult`, `vernier.semantic.EvalResult`) and the `tables=` keyword on two new entry points are public-API additions. Crosses the bar; this ADR is the trigger.
- **No new top-level deps.** `arrow-rs` (subset) is already in the workspace per ADR-0019; polars stays a Python-side optional. This ADR adds zero workspace deps.
- **ADR-0037 fused decode+fold path is sacred.** The semantic `evaluate_from_pngs` entry point (`crates/vernier-semantic/src/decode.rs`) is the headline performance result of the most recent perf round (`project_panoptic_semantic_perf_round.md`). Any per-image retention design that materializes a per-image confusion matrix in that path is rejected on sight. This is the reason `per_image` for semantic is deferred again rather than shipped here.
- **Honest schema.** "Per-class only, for now" is the disposition. Adding `per_image` is a real design choice; cramming both into one ADR would dilute the discussion of the per-image cost model and lock in choices we don't yet need to make.

## Considered options

The decision has three coupled axes.

### Axis A — Which tables to ship

1. **Both `per_class` and `per_image` for both paradigms.** Maximalist. Forces a per-image retention design for semantic that conflicts with ADR-0037, and a parallel-accumulator decision for panoptic that the streaming layer (`crates/vernier-panoptic/src/stream.rs:53`) has only half-answered. Two open questions for one ADR.
1. **`per_class` only.** Ships the cheap data already retained at summarize time. Defers `per_image` to a separate follow-up that can argue for the retention cost on its own merits. One open question, closed.
1. **`per_class` only for semantic; `per_class` + `per_image` for panoptic.** Splits the asymmetry. Tempting because panoptic's `retain_per_image_deltas` (already in `stream.rs:53`) gets us most of the way to per-image. But shipping `per_image` for one paradigm and not the other is a worse user surface than shipping it for neither.

### Axis B — Result type shape

1. **One shared `EvalResult` with paradigm-tagged fields.** Add `summary: Summary | PanopticSummary | SemanticSummary` and let the user narrow. Removes the type-level guarantee that `result.per_detection` is even meaningful for the paradigm in hand.
1. **Three sibling types, no inheritance.** `vernier.instance.EvalResult`, `vernier.panoptic.EvalResult`, `vernier.semantic.EvalResult`. Each carries its paradigm's `Summary`. Each carries only the table batches that apply to its paradigm. Mirrors ADR-0029.
1. **Shared base class with paradigm-specific subclasses.** Adds an inheritance hierarchy for the two methods (`__init__`, attribute access pattern) that genuinely repeat. Pyright's strict mode handles structural subtyping fine without it; the inheritance edge would be carried for one helper.

### Axis C — Where the per-paradigm `TableName` literal lives

1. **Unified `vernier._types.TableName = Literal["per_image", "per_class", "per_detection", "per_pair"]`.** Allows panoptic / semantic callers to ask for tables their paradigm doesn't produce, with no compile-time error. Type-narrowing must be at the entry point, not the literal.
1. **Per-paradigm literals.** `vernier.panoptic.TableName = Literal["per_class"]`. `vernier.semantic.TableName = Literal["per_class"]`. Pyright catches `Evaluator.evaluate(tables=("per_pair",))` as a type error at the call site.

## Decision outcome

- **Axis A → option 2.** `per_class` only, both paradigms. `per_image` is a separate follow-up.
- **Axis B → option 2.** Three sibling `EvalResult` types under `vernier.instance`, `vernier.panoptic`, `vernier.semantic`. No shared base class.
- **Axis C → option 2.** Per-paradigm `TableName` literals, `Literal["per_class"]` for both new paradigms. Widens additively if and when `per_image` lands.

The combined consequence: a panoptic or semantic user who does not pass `tables=` sees no change. A user who passes `tables=("per_class",)` gets a polars DataFrame whose columns are pinned by an Arrow schema golden checked into `tests/python/tables/<paradigm>/schemas/per_class.json`.

### What "zero-overhead by default" means concretely (panoptic + semantic)

Same standard as ADR-0019, applied per-paradigm:

- `vernier.panoptic.Evaluator.evaluate(gt, dt)` (no `tables=`) does not construct a single Arrow array, does not allocate a single byte in the new `crates/vernier-panoptic/src/tables.rs`, and does not import polars on either side of the FFI. Returns `vernier.panoptic.Summary`, bit-identical to today.
- `vernier.semantic.Evaluator.evaluate(gt, dt)` (no `tables=`) preserves the ADR-0037 fused decode+fold path with no per-image retention. Returns `vernier.semantic.Summary`, bit-identical to today.
- The bench at `tests/python/bench/test_zero_overhead_default_path.py` gains parallel cases for panoptic and semantic. Same 1% regression budget.

### Rust core surface — panoptic

A new module `crates/vernier-panoptic/src/tables.rs`:

```rust
pub struct PerClassTable {
    pub category_id: Vec<i64>,
    pub pq: Vec<f64>,
    pub sq: Vec<f64>,
    pub rq: Vec<f64>,
    pub n_tp: Vec<u64>,
    pub n_fp: Vec<u64>,
    pub n_fn: Vec<u64>,
    pub iou_sum: Vec<f64>,
}

pub fn build_per_class(summary: &PanopticSummary) -> PerClassTable;
```

`category_name` is omitted: vernier does not carry dataset metadata (ADR-0029 keeps name resolution at the user). Users who want names can left-join the polars DataFrame against their own category map.

`iou_sum` is currently dropped at `crates/vernier-panoptic/src/summarize.rs:110` — it is computed and divided through into `sq` then discarded. This ADR threads it into a new `ClassPanopticStats.iou_sum: f64` field (one f64 per category, retained across the existing summary path). The cost is zero in the default path (the value is computed regardless); the addition is a public-API change to `ClassPanopticStats`.

### Rust core surface — semantic

A new module `crates/vernier-semantic/src/tables.rs`:

```rust
pub struct PerClassTable {
    pub category_id: Vec<i64>,
    pub iou: Vec<f64>,
    pub accuracy: Vec<f64>,        // recall: TP / (TP + FN)
    pub precision: Vec<f64>,       // TP / (TP + FP)
    pub n_gt_pixels: Vec<u64>,
    pub n_dt_pixels: Vec<u64>,
    pub tp_pixels: Vec<u64>,
    pub fp_pixels: Vec<u64>,
    pub fn_pixels: Vec<u64>,
}

pub fn build_per_class(
    summary: &SemanticSummary,
    confusion: &ConfusionMatrix,
) -> PerClassTable;
```

`tp_pixels` is read from the confusion matrix diagonal (one u64 lookup per category). `fp_pixels = n_dt_pixels - tp_pixels`, `fn_pixels = n_gt_pixels - tp_pixels`. All three derive at table-build time, with no change to the summary or to the streaming wire body (`crates/vernier-semantic/src/distributed.rs:91-99`).

The 9 columns match the standard mmseg / ADE20K table shape (the vendored oracle in ADR-0036 emits the same set), reducing migration friction for users coming from those tools.

### FFI surface

Two new modules, mirroring `crates/vernier-ffi/src/tables.rs`:

- `crates/vernier-ffi/src/panoptic_tables.rs::panoptic_per_class_to_arrow_pycapsule(py, summary) -> ArrowRecordBatchPy`
- `crates/vernier-ffi/src/semantic_tables.rs::semantic_per_class_to_arrow_pycapsule(py, summary, confusion) -> ArrowRecordBatchPy`

The PyCapsule plumbing (`make_capsule`, `wrap_batch`, `ArrowRecordBatchPy`) currently lives at `crates/vernier-ffi/src/tables.rs:77-93`. As part of this ADR, that plumbing is extracted into `crates/vernier-ffi/src/arrow_helpers.rs` so all three table modules import the same primitives. This is the only refactor of existing code in the implementation.

Both new entry points run under `py.detach` (GIL dropped), per ADR-0006 and the ADR-0019 precedent.

### Python surface

```python
# python/vernier/panoptic/__init__.py

TableName = Literal["per_class"]

@dataclass(frozen=True)
class EvalResult:
    summary: Summary  # = vernier.panoptic.Summary (PanopticSummary)
    _per_class_batch: object | None = field(default=None, repr=False)

    @cached_property
    def per_class(self) -> "pl.DataFrame":
        return _arrow_to_dataframe(self._per_class_batch, "per_class")

class Evaluator:
    @overload
    def evaluate(self, gt, dt, *, tables: None = None) -> Summary: ...
    @overload
    def evaluate(
        self, gt, dt,
        *, tables: Literal["all"] | tuple[TableName, ...],
    ) -> EvalResult: ...
    def evaluate(self, gt, dt, *, tables=None): ...
```

Semantic mirrors this shape. The `_arrow_to_dataframe` helper (`python/vernier/_types.py:126-150`) is extracted into a new `python/vernier/_tables.py` so all three paradigm modules import it without pulling each other's types.

`tables="all"` expands to `("per_class",)` for panoptic and semantic — additive when `per_image` lands, no breaking change.

### What this ADR explicitly does *not* decide

- **`per_image` for either paradigm.** Deferred to a follow-up. The two paradigms have different cost shapes:
  - **Panoptic** can reuse `stream.rs::retain_per_image_deltas` (already opt-in for distributed merge per ADR-0032) by propagating the flag from `tables=("per_image", ...)` through the batch path. The follow-up ADR decides whether to wire that flag through, add a parallel per-image accumulator, or both.
  - **Semantic** has a hard conflict with ADR-0037. The fused decode+fold path is the headline result of the most recent perf round (`project_panoptic_semantic_perf_round.md`); a per-image confusion matrix in that path is rejected. A reasonable narrower option — per-image scalar mIoU + (n_gt, n_dt, n_tp) totals only — exists but warrants its own ADR.
- **`category_name` column.** Vernier does not carry category-name metadata (ADR-0029). Adding name resolution would mean either a new ingest field (changes ADR-0020 dataset shape) or a Python-side join helper (a new public method). Neither is in scope.
- **Keypoints tables.** ADR-0012 is accepted but the keypoints crate is not on disk. When the kernel lands, keypoints reuses the **instance** machinery (it is an OKS variant of the same AP-fold) and inherits all four ADR-0019 tables (`per_image`, `per_class`, `per_detection`, `per_pair`) as a free side effect of the AP-fold reuse, with `category_id` resolved against the OKS sigma map. No new tables work is required when keypoints lands; this ADR records that expectation so the keypoints ADR-0012 acceptance does not re-derive it.
- **Custom user-defined columns / Parquet emit / vernier-viz.** All deferred per ADR-0019 §"What this ADR explicitly does *not* decide."

### Phased landing plan

1. **Step 1 — shared FFI helper extraction.** Move `make_capsule`, `wrap_batch`, `ArrowRecordBatchPy` from `crates/vernier-ffi/src/tables.rs` into `crates/vernier-ffi/src/arrow_helpers.rs`. Mechanical refactor; existing instance tables tests are the regression check.
1. **Step 2 — panoptic per_class end-to-end.** New `tables.rs` in `vernier-panoptic`; `iou_sum` retained on `ClassPanopticStats`; FFI shim; Python `Evaluator.evaluate(tables=...)`; schema golden; smoke test; lazy-import test.
1. **Step 3 — semantic per_class end-to-end.** Same surface, mirroring the standard mmseg shape.
1. **Step 4 — zero-overhead bench coverage.** Extend the existing default-path bench to assert the panoptic and semantic `evaluate()` paths stay within 1% of their pre-change baseline at val2017-jittered, `--mode release`.
1. **Step 5 — documentation.** How-to entry under `docs/how-to/result-tables.md`; the existing instance-tables guide picks up two paragraphs and a `vernier.semantic` example.

Each step is self-contained and can be skipped or deferred without breaking the others, same discipline as ADR-0019.

### Consequences

- **Positive.** Two canonical user questions ("which class is dragging PQ down?" / "per-class IoU table for sem-seg") are answered with one polars DataFrame instead of 30 lines of indexing. The capability lands in roughly 200 lines of new logic across two crates plus Python wrappers, with one mechanical refactor (`arrow_helpers.rs`). The default `evaluate()` path stays bit-identical for both paradigms. The shared FFI helper extraction is a small API-surface win independently — three tables modules now import one shared primitive instead of duplicating it.
- **Negative.** Four new public types (`vernier.panoptic.EvalResult`, `vernier.panoptic.TableName`, `vernier.semantic.EvalResult`, `vernier.semantic.TableName`) widen the API surface. The asymmetry — instance has four tables, panoptic and semantic have one each — is real and visible; users learning the API see "tables look different per paradigm." The `category_name` omission will surprise some users who expect mmseg-style tables to carry names; the explanation lives in the how-to guide. `ClassPanopticStats` gains an `iou_sum` field — a public-API addition that requires the same care as any pre-1.0 surface change.
- **Neutral.** `vernier.instance.EvalResult` is unchanged in shape. The shared `_types.EvalResult` symbol stays where it is; the canonical path becomes `vernier.instance.EvalResult` via re-export, but no breaking move is required for this ADR. The `tables="all"` literal expanding to different sets per paradigm is honest about the asymmetry rather than papering over it.

## Pros and cons of the options

### Axis A — Which tables to ship

**A2 (chosen) — `per_class` only**

- 👍 Ships only data already retained at summarize time. No retention flag, no accumulator change beyond `iou_sum`.
- 👍 One open design question (per_image cost model) closed in a follow-up that can give it the attention it needs.
- 👎 Users asking for per-image PQ get "soon, but not yet." Some will be disappointed.

**A1 — Both `per_class` and `per_image`**

- 👍 One ADR, one PR, two paradigms fully covered.
- 👎 Forces a semantic per-image design that conflicts with ADR-0037; either we break the fused decode+fold path or we ship a narrower per-image surface than the panoptic one (asymmetric in a worse way).
- 👎 Doubles the implementation scope; doubles the surface area of mistakes.

**A3 — Asymmetric per-image (panoptic only)**

- 👍 Maximizes panoptic's "data is half-retained anyway" advantage.
- 👎 Worse user-facing asymmetry. "Per-image works for panoptic but not semantic" is harder to teach than "per-image is in the next release."

### Axis B — Result type shape

**B2 (chosen) — Three sibling types**

- 👍 Mirrors ADR-0029's per-paradigm `Summary` split exactly.
- 👍 `vernier.panoptic.EvalResult.per_detection` does not exist as a type-level error — pyright catches the mistake.
- 👍 No inheritance edge to maintain across paradigms.
- 👎 Three classes with one common helper (`_arrow_to_dataframe`). The repetition is ~5 lines per class.

**B1 — Shared `EvalResult` with paradigm-tagged fields**

- 👍 One class.
- 👎 `result.per_detection` autocompletes for panoptic users, returning `None` or raising. Type system can't help.
- 👎 Couples instance, panoptic, and semantic into one type — every future paradigm change touches one struct.

**B3 — Shared base class**

- 👍 One inheritance edge captures the `_arrow_to_dataframe` helper.
- 👎 Adds a hierarchy for ~5 lines of shared code; pyright's structural subtyping handles the same case without inheritance.

### Axis C — `TableName` literal

**C2 (chosen) — Per-paradigm literals**

- 👍 `vernier.panoptic.Evaluator.evaluate(tables=("per_pair",))` is a type error at the call site.
- 👍 Widens additively when `per_image` lands.
- 👎 Three literals to keep in sync with the entry-point overloads.

**C1 — Unified literal**

- 👍 One symbol.
- 👎 Defers all "is this table valid for this paradigm?" enforcement to runtime. The type system has the answer; there's no reason to discard it.

## Links and references

- **ADR-0019** — instance-detection tables, the surface this ADR extends.
- **ADR-0025** — panoptic API; closes with "Arrow result tables for PQ" deferral that this ADR resolves.
- **ADR-0028** — semantic-segmentation; same deferral.
- **ADR-0029** — namespace split that this ADR's per-paradigm result types mirror.
- **ADR-0036** — vendored mmsegmentation oracle; defines the column shape semantic per-class table targets.
- **ADR-0037** — generalized semantic kernel class-id; the fused decode+fold contract this ADR will not break.
- `python/vernier/_types.py:69-150` — current `EvalResult` and `_arrow_to_dataframe`.
- `crates/vernier-ffi/src/tables.rs:77-93` — current PyCapsule plumbing, target for the `arrow_helpers.rs` extraction.
- `crates/vernier-panoptic/src/summarize.rs:31-110` — `ClassPanopticStats` and the `iou_sum` drop site.
- `crates/vernier-semantic/src/summarize.rs:62-113` — `ClassSemanticStats`, the data already retained per-class.
