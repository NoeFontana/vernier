# ADR-0041: User-parametrizable semantic evaluation

- **Status:** proposed
- **Date:** 2026-05-09
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

> Renumbered from the user's implementation plan (which reserved
> ADR-0037). 0037 is taken (generalize semantic kernel class-id).
> This ADR ratifies the cross-paradigm conventions in ADR-0039 for the
> semantic kernel.

## Context and problem statement

`vernier.semantic.Evaluator` (`python/vernier/semantic/__init__.py`)
is a frozen dataclass whose user-facing surface today is:
`parity_mode`, `n_classes`, `ignore_label`. Two parameters that real
semantic-segmentation users want are *not* surfaced:

- **Class filtering** — restrict mIoU averaging to a class subset.
  Common asks: "mIoU over the 19 evaluation classes" (Cityscapes
  drops 14 of its 33 raw labels at evaluation time), "mIoU over the
  flat / vehicle classes only" for ablation studies, "mIoU
  excluding rare classes that drag the mean down."
- **Class grouping** — hierarchical mIoU over named class subsets.
  Cityscapes' canonical evaluation reports per-category mIoU over
  `flat / human / vehicle / construction / object / nature / sky`
  groups (the `categoryEval` mode of `cityscapesScripts`). Today
  vernier produces a flat per-class breakdown; reproducing
  Cityscapes' grouped output requires the user to fold the dict.

Semantic's math is per-pixel confusion matrix (`accumulate_confusion`
in `crates/vernier-semantic/src/kernel.rs`, generalized over
class-id width per the proposed ADR-0037). Unlike the instance kernel
there is no IoU ladder (the per-class IoU is a single value), no
recall ladder, and no area buckets — semantic doesn't fold the same
`(T, R, K, A, M)` tensor.

The constraint, as always, is parity: strict mode against
`cityscapesScripts` and `mmsegmentation` must continue producing
byte-identical four-scalar summaries (`miou`, `mean_accuracy`,
`pixel_accuracy`, `frequency_weighted_iou`) on every existing
fixture. The new fields are additive opt-in.

## Decision drivers

- **Cross-paradigm conventions (ADR-0039) apply verbatim.** Sentinel
  resolution (`None` → kernel-canonical), construction-time
  validation, per-paradigm `params_hash` extension, drop-in shim
  non-exposure (vacuous on semantic — there's no shim today).
- **Reuse the LVIS `CategoryFilter` shape, don't reinvent it.**
  ADR-0026 shipped `CategoryFilter { All, Frequency(Frequency),
  ByIds(BTreeSet<CategoryId>) }`. Two of three variants apply to
  semantic; `Frequency` does not (semantic has no LVIS-style
  rare/common/frequent class notion). The user's plan called the
  semantic field `class_filter: ClassFilter` and the panoptic field
  `category_filter: CategoryFilter` — this ADR resolves the naming
  inconsistency by keeping the type shared (`CategoryFilter`) and
  using paradigm-natural field names (`class_filter` here,
  `category_filter` in ADR-0042).
- **`Frequency` is not a `Breakdown` axis.** ADR-0026 lines 178–182
  pin the distinction: *"Frequency keys a category by enum and selects
  a subset of K. Forcing the round trip through f64 encodes a sum
  type that doesn't generalize to non-numeric axes."* Semantic
  `class_grouping` is a `Breakdown` over class IDs (a class-id-keyed
  partition of K), distinct from the LVIS `Frequency` enum.
- **Named-field summary composes with filtering.** `SemanticSummary`
  has named per-class fields (`per_class: dict[int, ClassSemanticStats]`,
  shipped via commit 82ba92f / ADR-0038). Filtering and grouping
  don't break the index assumption that bites instance — the math
  composes cleanly. No `IncompatibleSummaryPlan` rule applies.
- **Class-id partition, not overlap.** For per-group mIoU to be
  well-defined, class groups must be a partition (every class belongs
  to at most one group). Overlapping groups produce ambiguous
  per-group means; validation rejects them at construction.

## Considered options

1. **Status quo + post-hoc dict folding.** Users who want
   class-grouped mIoU fold `summary.per_class` themselves; users who
   want filtered mIoU re-run with a subset GT/DT.
2. **Surface `class_filter` only; defer grouping.** Filter without
   grouping covers ablation studies; defer Cityscapes-style
   hierarchical mIoU to a follow-up.
3. **Surface both `class_filter` and `class_grouping` together.**
   Filter and grouping compose (`class_filter=ByGrouping(name)`
   selects one specific group's mean from the configured grouping).

## Decision outcome

Chosen option: **3 — surface both fields together.**

### New fields

```python
@dataclass(frozen=True, slots=True)
class Evaluator:
    parity_mode: ParityMode = "corrected"
    n_classes: int                                       # required (existing)
    ignore_label: int = 255                              # existing
    class_filter: CategoryFilter | None = None           # NEW
    class_grouping: Breakdown | None = None              # NEW
```

`CategoryFilter` is the type from ADR-0026, re-exported here. The
semantic field is named `class_filter` because the surrounding
vocabulary is "classes" (`n_classes`, `ignore_label`, `per_class`),
not "categories." The type is shared with LVIS instance and with
ADR-0042's panoptic `category_filter`.

The `CategoryFilter` enum gains a fourth variant:

```rust
pub enum CategoryFilter {
    All,
    Frequency(Frequency),       // LVIS-only; semantic / panoptic ignore
    ByIds(BTreeSet<CategoryId>),
    ByGrouping(Cow<'static, str>),  // NEW — selects a named group from
                                    // the configured class_grouping
}
```

This **extends** the shipped enum; ADR-0026 stays `accepted`
unchanged. `ByGrouping` is only valid when `class_grouping` is set
(validation enforces this). LVIS users continue to consume `All`,
`Frequency(_)`, `ByIds(_)`; semantic and panoptic users opt into
`ByIds(_)` and `ByGrouping(_)`. Construction-time validation rejects
`Frequency(_)` on semantic / panoptic with an explicit
`InvalidSemanticParams` / `InvalidPanopticParams` pointing at this
ADR (the LVIS frequency notion does not transfer).

Sentinel resolution per ADR-0039:

| Field | `None` resolves to |
|---|---|
| `class_filter` | `CategoryFilter::All` (every class with class_id < n_classes contributes) |
| `class_grouping` | unset; per-group fields on `SemanticSummary` are `None` |

### Validation contract

Construction-time validation in `Evaluator.__post_init__`, raising
`InvalidSemanticParams` (ADR-0039 hierarchy):

- `class_filter`:
  - `Frequency(_)` is rejected with a pointer to ADR-0026's
    `Frequency`-vs-`Breakdown` distinction.
  - `ByIds(ids)`: every id in `[0, n_classes)`, no duplicates,
    non-empty.
  - `ByGrouping(name)`: only valid when `class_grouping is not None`;
    `name` must match one of the configured group labels.
- `class_grouping` (`Breakdown` constructed via
  `Breakdown.from_class_groups({group_name: [class_ids], ...})`):
  - non-empty;
  - every class id in `[0, n_classes)`;
  - no class id appears in two groups (partition discipline);
  - no duplicate group labels.

### `SemanticSummary` extension

`SemanticSummary` gains an optional named field:

```python
class SemanticSummary:
    miou: float
    mean_accuracy: float
    pixel_accuracy: float
    frequency_weighted_iou: float
    per_class: dict[int, ClassSemanticStats]
    per_group: dict[str, GroupSemanticStats] | None  # NEW
    # ... existing fields ...
```

`per_group` is `None` when `class_grouping is None` and a populated
dict otherwise. Each `GroupSemanticStats` carries the group's
`miou`, `mean_accuracy`, member class ids, and aggregated
confusion-matrix counts.

The four headline scalars (`miou`, `mean_accuracy`, `pixel_accuracy`,
`frequency_weighted_iou`) reflect the `class_filter` scope when set:
`miou` becomes the unweighted mean over filtered classes only.
Strict-mode parity is preserved because the default
`class_filter=None → CategoryFilter::All` produces the same scalar
as today.

Schema versioning per ADR-0019: the optional `per_group` field is a
non-breaking addition under the existing `_schema_version` discipline.
Consumers reading `summary.per_class` without inspecting `per_group`
continue working unchanged.

### Distributed-eval `params_hash` extension

Per ADR-0039: `SemanticParams` resolved-params struct fingerprints
the four shared fields plus the resolved `class_filter` and
`class_grouping`. `class_filter=None` and `class_filter=CategoryFilter::All`
hash identically (both resolve to "every class"). Diverging hashes
across ranks fire `PartialParamsMismatch` per ADR-0032.

### Drop-in shim non-exposure

There is no semantic drop-in shim today (vernier.semantic does not
mirror a single canonical Python tool — `cityscapesScripts` and
`mmsegmentation` are separate oracles). The non-exposure rule from
ADR-0039 applies vacuously; if a future shim lands, it inherits the
rule.

### Consequences

- **Positive.** Cityscapes-style hierarchical mIoU becomes a
  first-class output. Class-subset mIoU for ablation studies stops
  needing post-hoc folding. The `CategoryFilter` extension to
  `ByGrouping(name)` generalizes naturally to ADR-0042 (panoptic).
  Strict-mode parity against `cityscapesScripts` and
  `mmsegmentation` is unchanged on every existing fixture.
- **Negative.** Two more public fields and one new
  `CategoryFilter` variant. The `Frequency(_)` rejection on semantic
  is a small footgun for LVIS users who try to apply the same enum
  shape across paradigms — the validation message has to do the
  migration work. The `ByGrouping(name)` validation requires
  `class_grouping` to be set first; the doc page (`docs/how-to/semantic-grouping.md`,
  scoped to a follow-up PR) needs a worked Cityscapes example.
- **Neutral.** No kernel changes. ADR-0006 immutable-config
  invariant survives. The `n_classes` and `ignore_label` semantics
  are unchanged.

## Pros and cons of the options

### Option 1 (status quo + post-hoc folding)

- 👍 Zero new public surface; users solve their problem with a
  three-line dict comprehension.
- 👎 Cityscapes' canonical group hierarchy is a moving target if
  every consumer re-derives it. The grouped output is metric-defining
  for that benchmark; not surfacing it forces every Cityscapes user
  to reinvent the same fold.

### Option 2 (filter only; defer grouping)

- 👍 Smaller surface today; `class_filter` covers ablation studies
  (the common-case ask). Grouping ships when demand materializes.
- 👎 The `class_filter=ByGrouping(name)` composition is the natural
  user surface — splitting it across two ADRs forces a second
  surface change (not just an extension) when grouping lands. The
  Cityscapes use case is concrete today, not speculative.

### Option 3 (chosen — both fields together)

- 👍 The `class_filter`/`class_grouping` interaction is designed
  once, validated once, hashed once. The `CategoryFilter::ByGrouping`
  extension lands once and serves semantic and panoptic identically.
- 👎 Slightly more validation surface than option 2; the `ByGrouping`
  variant has the dependency on `class_grouping` being set, which is
  the only inter-field validation rule on this `Evaluator`.

## Links and references

- ADR-0001 — gates significance.
- ADR-0002 — three-tier parity model.
- ADR-0006 — frozen `Evaluator` discipline (preserved).
- ADR-0019 — result tables; `_schema_version` discipline keeps the
  `per_group` addition non-breaking.
- ADR-0026 — LVIS support; `CategoryFilter` enum is the type extended
  here. ADR-0026 stays `accepted` ungoverned; this ADR adds the
  fourth variant via cross-reference.
- ADR-0028 — semantic segmentation as a sibling crate.
- ADR-0029 — namespace restructure.
- ADR-0031 / ADR-0032 — distributed-eval `params_hash` (extension
  ratified in ADR-0039).
- ADR-0037 — generalize the semantic kernel over class-id type
  (orthogonal; supports any class-id width including the wider
  `n_classes` ranges this ADR's filtering operates on).
- ADR-0038 — per_class result tables for panoptic and semantic;
  the `per_class` shape this ADR composes with.
- ADR-0039 — cross-paradigm conventions (this ADR ratifies them).
- ADR-0042 — panoptic counterpart (consumes the `CategoryFilter::ByGrouping`
  extension introduced here).
