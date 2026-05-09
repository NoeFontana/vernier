# ADR-0042: User-parametrizable panoptic evaluation

- **Status:** accepted
- **Date:** 2026-05-09
- **Accepted on:** 2026-05-09
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

> Renumbered from the user's implementation plan (which reserved
> ADR-0038). 0038 is taken (per_class result tables for panoptic and
> semantic). This ADR ratifies the cross-paradigm conventions in
> ADR-0039 for the panoptic kernel.

## Context and problem statement

`vernier.panoptic.Evaluator` (`python/vernier/panoptic/__init__.py`)
is a frozen dataclass whose user-facing surface today is essentially
just `parity_mode`. Four parameters real panoptic users want are not
surfaced:

- **`pq_iou_threshold`** — the IoU threshold above which a (DT, GT)
  pair is a true positive. panopticapi's canonical 0.5 (Kirillov et
  al. 2019, Eq. 1) is hardcoded in the kernel. Safety-critical
  evaluation (autonomous driving, medical) wants tighter thresholds
  (0.7, 0.8); ablations want a sweep.
- **`category_filter`** — restrict mean PQ to a category subset. Same
  asks as semantic: per-supercategory PQ, ablation subsets, mean PQ
  excluding a noisy class.
- **`class_grouping`** — hierarchical PQ over named class subsets,
  parallel to ADR-0041's semantic surface. Cityscapes-panoptic wants
  per-category PQ over its `flat / human / vehicle / construction /
  object / nature / sky` groups.
- **`stuff_thing_partition`** — explicit override of the GT-derived
  stuff/thing split that drives `PQ_St` and `PQ_Th` sub-metrics. Some
  datasets carry inconsistent or missing `isthing` flags on category
  metadata; users want to specify the partition explicitly without
  rewriting the dataset.

Panoptic's math is one-to-one matching at IoU > 0.5 with categorical
agreement (ADR-0025), feeding a per-category `(TP, FP, FN, TP_IoU_sum)`
quad that produces PQ = SQ × RQ. No T/R/A/M tensor; no score-ranked
greedy matcher. Filtering and grouping compose with the named-field
`PanopticSummary` exactly the way they compose with the semantic
summary (ADR-0041). Thresholding is the structurally novel knob.

The constraint, as always, is parity: strict mode against
panopticapi must continue producing byte-identical PQ / SQ / RQ /
per-category breakdown on every existing fixture. The new fields are
additive opt-in.

## Decision drivers

- **Cross-paradigm conventions (ADR-0039) apply verbatim.** Sentinel
  resolution, construction-time validation, per-paradigm
  `params_hash` extension, drop-in shim non-exposure (vacuous on
  panoptic — there's no shim today).
- **Reuse the `CategoryFilter` shape extended in ADR-0041.**
  `ByGrouping(name)` is the variant ADR-0041 introduced; ADR-0042
  consumes it without re-introducing.
- **`Frequency` is not a panoptic concept.** Same argument as
  ADR-0041: panoptic has no LVIS-style rare/common/frequent class
  notion. `Frequency(_)` is rejected with a typed
  `InvalidPanopticParams` pointing at ADR-0026.
- **PQ's threshold is single, not a ladder.** PQ doesn't sweep IoU
  like AP does — the metric is defined at one threshold. The user
  surface is a single `float`, not a tuple.
- **Threshold sensitivity is not monotonic in the obvious direction.**
  Lowering `pq_iou_threshold` below 0.5 reads as "loosen matching"
  but PQ's denominator includes FPs and FNs; pairs that were FPs at
  threshold 0.5 may become TPs at 0.4, *and* the SQ contribution of
  a low-IoU pair drags the score down. PQ can move in unintuitive
  directions across thresholds. The doc page (`docs/how-to/panoptic-threshold.md`,
  scoped to a follow-up PR) carries a sensitivity-analysis chart.
- **Stuff/thing partition is dataset-derived today.** `PanopticDataset`
  reads `categories[i].isthing` from `segments_info` metadata. An
  explicit override is additive: when present it wins; when absent
  the dataset-derived split applies (ADR-0025's existing behavior).

## Considered options

1. **Status quo + canonical-only evaluation.** Users who want
   non-canonical thresholds re-run with patched panopticapi.
2. **Surface the four fields independently.** Each is its own
   `Evaluator` field with its own validation, no inter-field rules.
3. **Surface the four fields with `category_filter`/`class_grouping`
   composing per ADR-0041.** Single shared `CategoryFilter` enum
   across semantic / panoptic; `ByGrouping(name)` requires
   `class_grouping` set.

## Decision outcome

Chosen option: **3 — surface the four fields with shared filter type
and composition rules.**

### New fields

```python
@dataclass(frozen=True, slots=True)
class Evaluator:
    parity_mode: ParityMode = "corrected"
    pq_iou_threshold: float | None = None                # NEW
    category_filter: CategoryFilter | None = None        # NEW
    class_grouping: Breakdown | None = None              # NEW
    stuff_thing_partition: StuffThingPartition | None = None  # NEW
```

`StuffThingPartition` is a new value type:

```python
@dataclass(frozen=True, slots=True)
class StuffThingPartition:
    stuff: frozenset[int]   # category ids
    things: frozenset[int]  # category ids
```

Sentinel resolution per ADR-0039:

| Field | `None` resolves to |
|---|---|
| `pq_iou_threshold` | `0.5` (panopticapi canonical, Kirillov et al. 2019 Eq. 1) |
| `category_filter` | `CategoryFilter::All` |
| `class_grouping` | unset; per-group fields on `PanopticSummary` are `None` |
| `stuff_thing_partition` | derived from the dataset's `categories[i].isthing` metadata (existing ADR-0025 behavior) |

### Validation contract

Construction-time validation in `Evaluator.__post_init__`, raising
`InvalidPanopticParams` (ADR-0039 hierarchy):

- `pq_iou_threshold`: finite; `0.0 < t <= 1.0`. Strict-zero
  matches everything (every overlap is a TP); rejected as a footgun.
- `category_filter`:
  - `Frequency(_)` rejected with pointer to ADR-0026.
  - `ByIds(ids)`: every id corresponds to a category in the
    evaluator's expected dataset (validated at evaluate time, not
    construction — the dataset isn't bound to the Evaluator); no
    duplicates, non-empty.
  - `ByGrouping(name)`: only valid when `class_grouping is not None`;
    name matches one of the group labels.
- `class_grouping` (`Breakdown.from_class_groups(...)`):
  - non-empty;
  - no class id appears in two groups (partition);
  - no duplicate group labels.
- `stuff_thing_partition`:
  - `stuff` and `things` disjoint (no category in both);
  - both sets non-empty;
  - validated against the dataset at evaluate time (every id present
    in `segments_info` categories), not at construction.

### `PanopticSummary` extension

`PanopticSummary` gains an optional `per_group` field, parallel to
ADR-0041's semantic counterpart:

```python
class PanopticSummary:
    pq: float
    sq: float
    rq: float
    pq_st: float       # stuff-only PQ (existing)
    pq_th: float       # things-only PQ (existing)
    per_class: dict[int, ClassPanopticStats]
    per_group: dict[str, GroupPanopticStats] | None  # NEW
    # ... existing fields ...
```

`per_group` is `None` when `class_grouping is None` and a populated
dict otherwise. The four headline scalars (`pq`, `sq`, `rq`, plus
`pq_st`/`pq_th`) reflect `category_filter` scope when set; strict-mode
parity is preserved because `category_filter=None → All` produces the
same scalars as today.

`pq_st` and `pq_th` reflect `stuff_thing_partition` when set; default
behavior (dataset-derived `isthing`) is unchanged.

Schema versioning per ADR-0019: the optional `per_group` field is a
non-breaking addition under the existing `_schema_version` discipline.

### Distributed-eval `params_hash` extension

Per ADR-0039: `PanopticParams` resolved-params struct fingerprints
the four resolved fields. `pq_iou_threshold=None` and
`pq_iou_threshold=Some(0.5)` hash identically (both resolve to 0.5).
`stuff_thing_partition=None` hashes deterministically over the empty
override sentinel; the dataset-derived partition is part of the GT
fingerprint already covered by ADR-0032.

### Drop-in shim non-exposure

There is no panoptic drop-in shim today. The non-exposure rule from
ADR-0039 applies vacuously; if a future shim lands, it inherits the
rule.

### Threshold sensitivity caveat (doc-track)

The doc page `docs/how-to/panoptic-threshold.md` (scoped to a
follow-up PR) carries:

- A worked sensitivity chart of PQ as a function of `pq_iou_threshold`
  on a representative public dataset (COCO panoptic val).
- An explanation of the FN/FP coupling: lowering the threshold shifts
  pairs from FP into TP (raises the TP count) but also shifts the SQ
  contribution downward (lower-IoU TPs); the two effects can cancel
  or compound. PQ is *not* monotonic in the threshold across all
  models or datasets.
- Recommended use: sensitivity analysis around 0.5 (e.g.,
  `[0.4, 0.5, 0.6, 0.7]`), not single-point arbitrary thresholds.

The validation accepts `(0.0, 1.0]` to support this analysis without
encoding "we know your use case better than you do." The doc carries
the intent.

### Consequences

- **Positive.** Safety-critical evaluation gets a tighter
  `pq_iou_threshold`. Cityscapes-panoptic group hierarchy becomes a
  first-class output. Datasets with inconsistent `isthing` metadata
  get an explicit override path. Strict-mode parity against
  panopticapi is unchanged on every existing fixture.
- **Negative.** Four new public fields and one new public value type
  (`StuffThingPartition`). The threshold sensitivity caveat is a
  real footgun if users skip the doc page; the validation accepts a
  range that the doc tells them is unsafe to use blindly. The
  `Frequency(_)` rejection inherited from ADR-0041 is a small
  cross-paradigm gotcha.
- **Neutral.** No kernel changes (the threshold flows through the
  matching engine as a single comparator constant). ADR-0006
  immutable-config invariant survives. ADR-0025's sibling-crate
  separation is unchanged.

## Pros and cons of the options

### Option 1 (status quo + canonical-only)

- 👍 Zero new public surface; users wanting non-canonical thresholds
  patch panopticapi themselves.
- 👎 Patching panopticapi means giving up vernier's parity claim and
  performance on the same workload. Sensitivity analysis is a
  legitimate research / safety-eval need that the kernel could
  cheaply support.

### Option 2 (four independent fields, no inter-field rules)

- 👍 Simplest validation; each field is independent.
- 👎 `category_filter=ByGrouping(name)` without `class_grouping` set
  is a runtime error waiting to happen. Naming a single shared
  filter type across semantic/panoptic but not enforcing the
  composition rule produces drift between the two paradigm ADRs.

### Option 3 (chosen — shared type, composed rules)

- 👍 The `CategoryFilter::ByGrouping` extension lands once
  (ADR-0041) and serves both paradigms identically. Composition is
  validated at construction, not at evaluate time. The
  `StuffThingPartition` value type matches the named-field shape of
  the rest of `PanopticSummary` — symmetric with the input.
- 👎 Inter-field validation (`ByGrouping` requires `class_grouping`)
  is the only such rule on this `Evaluator`; it's a small precedent
  for future fields to follow.

## Links and references

- ADR-0001 — gates significance.
- ADR-0002 — three-tier parity model.
- ADR-0006 — frozen `Evaluator` discipline (preserved).
- ADR-0011 — discriminated kernel config; orthogonal (panoptic is
  a sibling crate, not an `IouKind` variant).
- ADR-0019 — result tables; `_schema_version` discipline keeps the
  `per_group` addition non-breaking.
- ADR-0025 — panoptic-quality evaluation as a sibling crate
  (ADR-0042 amends the public Python `Evaluator` surface; the kernel
  and crate boundary are unchanged).
- ADR-0026 — LVIS support; `CategoryFilter` enum reused (with
  ADR-0041's `ByGrouping` extension).
- ADR-0029 — namespace restructure.
- ADR-0031 / ADR-0032 — distributed-eval `params_hash` (extension
  ratified in ADR-0039).
- ADR-0038 — per_class result tables for panoptic and semantic;
  the `per_class` shape this ADR composes with.
- ADR-0039 — cross-paradigm conventions (this ADR ratifies them).
- ADR-0041 — semantic counterpart (introduced the
  `CategoryFilter::ByGrouping` extension this ADR consumes).
- Kirillov, He, Girshick, Rother, Dollár. *Panoptic Segmentation.*
  CVPR 2019. arXiv:1801.00868. Eq. 1 pins the IoU > 0.5 matching
  rule the threshold canonicalization preserves.
- panopticapi 0.1 — `panopticapi/evaluation.py` — strict-mode oracle.
