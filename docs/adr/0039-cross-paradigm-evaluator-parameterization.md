# ADR-0039: Cross-paradigm conventions for user-parametrizable evaluation

- **Status:** proposed
- **Date:** 2026-05-09
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

> The original implementation plan reserved ADR-0036/0037/0038 for this
> work and a "0036-A" appendix for the shared base. Those numbers are
> taken (vendor mmsegmentation IoUMetric / generalize semantic kernel
> class-id / per_class result tables, all shipped). The repo also does
> not use `-A` appendix suffixes — multi-part decisions are documented
> as separate ADRs with cross-references. This ADR is the renumbered
> shared base; ADR-0040, ADR-0041, and ADR-0042 are the per-paradigm
> siblings that ratify it by reference.

## Context and problem statement

Three paradigm-specific `Evaluator` classes — `vernier.instance.Evaluator`,
`vernier.semantic.Evaluator`, `vernier.panoptic.Evaluator` — are each
growing user-facing parameter knobs. Instance is queued to surface
custom IoU thresholds, recall thresholds, and area ranges (lifting the
ADR-0016 deferral). Semantic is queued to surface class filters and
class grouping. Panoptic is queued to surface `pq_iou_threshold`, a
category filter, class grouping, and a stuff/thing partition override.

Each paradigm has its own math, its own summary shape, and its own
parity oracle (`pycocotools==2.0.11`, `lvis==0.5.3`,
`cityscapesScripts`, `mmsegmentation`, `panopticapi==0.1`). Letting
each paradigm pick its own conventions for how user parameters are
defaulted, validated, fingerprinted for distributed-eval, and surfaced
on the drop-in shims would produce three not-quite-compatible APIs at
0.1.0 — the moment surface choices freeze for a deprecation cycle.

We are still on 0.0.x (release-pace memo). This ADR ratifies the
shared discipline so the three paradigm ADRs land against a frozen
contract instead of negotiating it three times in parallel. The
discipline is the same one ADR-0012 already established for
`max_dets` on the instance kernel; this ADR generalizes it.

## Decision drivers

- **Parity-default invariant.** Existing reference-oracle parity
  (ADR-0002 strict tier; pycocotools / lvis / cityscapesScripts /
  panopticapi) must survive every new knob unchanged. Users opting
  into custom parameters opt out of parity by construction — never
  the reverse.
- **One rule, three paradigms.** ADR-0012's `max_dets` sentinel
  (`None` → kernel-canonical) is the precedent. Generalizing it to
  every new field across every paradigm produces a single rule users
  learn once. Mirrors `python/vernier/instance/__init__.py:218–228`
  (`_resolve_max_dets`).
- **Validation at construction, not evaluation.** Misconfiguration
  surfaces when the `Evaluator` is built — fast, with a typed error
  pointing at the relevant ADR — not 200 ms into `evaluate()` after
  the user has already paid the GT-parsing cost.
- **Distributed-eval `params_hash` continuity.** ADR-0031 / ADR-0032
  rely on a `params_hash` (`crates/vernier-core/src/evaluate.rs`,
  `EvaluateParams::params_hash`) to refuse cross-rank merges with
  divergent configurations. Today it fingerprints a single struct
  with no paradigm discrimination. Surfacing new knobs means each
  paradigm gets its own resolved-params type and its own hash; the
  cross-paradigm-merge refusal already enforced by ADR-0032 stays
  structural.
- **Drop-in shim non-exposure.** ADR-0007 pins the
  `patch_pycocotools` shim's surface to pycocotools' actual API.
  pycocotools doesn't expose any of the new knobs — surfacing them
  on the shim would invent a non-pycocotools API and defeat the
  point of the shim.
- **One canonical `Breakdown` lift to Python.** ADR-0016 explicitly
  deferred Python exposure: *"Surface stays internal until it's not.
  The PR exposing this in Python (CrowdPose-driven) is a separate
  ADR."* (lines 55–58). Two of the three paradigm ADRs (instance for
  area ranges, semantic / panoptic for class grouping) need it.
  Designing the FFI shape three times is drift waiting to happen.

## Considered options

1. **Status quo.** Each paradigm ADR negotiates conventions
   independently. Drift after 0.1.0 is the new owner's problem.
2. **Three sibling paradigm ADRs with copy-pasted conventions.**
   Each ADR repeats the sentinel rule, validation rule,
   `params_hash` extension, and shim non-exposure rule.
3. **Cross-paradigm ADR + three sibling ADRs.** Conventions land
   once in a shared ADR; per-paradigm ADRs ratify by reference and
   focus on paradigm-specific math. Mirrors the project's existing
   pattern of ADR-0029 (namespace restructure) referenced by 0025
   (panoptic), 0026 (LVIS), and 0028 (semantic).

## Decision outcome

Chosen option: **3 — cross-paradigm ADR plus three sibling ADRs.**

The shared conventions, ratified once here:

### Sentinel resolution

Every new user-facing parameter on every `Evaluator` is typed as
`T | None` with a `None` default. `None` resolves at dispatch to the
kernel-canonical value that produces bit-exact reference-oracle
output. Explicit values are passed through verbatim after validation.

This generalizes the precedent at `python/vernier/instance/__init__.py:218–228`
where `Evaluator.max_dets: tuple[int, ...] | None = None` resolves
through `_resolve_max_dets()` to a per-`IouKind` table
(`_KERNEL_MAX_DETS`). The same pattern applies to `iou_thresholds`,
`recall_thresholds`, `area_ranges`, `class_filter`, `class_grouping`,
`pq_iou_threshold`, `category_filter`, and `stuff_thing_partition`
across the three paradigm ADRs.

The two-line user-facing rule, repeated verbatim in the migration
guides: **"`None` is parity. Any other value opts out."**

### Validation contract

A typed exception hierarchy in `vernier._types`, re-exported from
each paradigm namespace:

```python
class InvalidEvalParams(ValueError):
    """Base for paradigm-specific construction errors."""

class InvalidInstanceParams(InvalidEvalParams): ...
class InvalidSemanticParams(InvalidEvalParams): ...
class InvalidPanopticParams(InvalidEvalParams): ...
```

Each subclass carries the offending field name, the offending value,
and a remediation pointer (the relevant ADR or doc page). Validation
runs during `Evaluator.__post_init__` (today's `@dataclass(frozen=True,
slots=True)` shape supports this naturally — see ADR-0006 and
`python/vernier/instance/__init__.py:184–216` for the precedent).
Construction-time validation means `evaluate()` cannot fail on a
misconfigured `Evaluator`; only on bad data.

### `params_hash` extension protocol

Today `EvaluateParams::params_hash()` fingerprints
`iou_thresholds`, `area_ranges`, `max_dets_per_image`, `use_cats`,
`retain_iou` via rkyv → blake3 with no paradigm discrimination. The
new convention:

1. Each paradigm gets its own resolved-params struct on the Rust
   side (`InstanceParams`, `SemanticParams`, `PanopticParams`).
2. Each paradigm fingerprints its struct independently. The hash
   covers post-sentinel resolved values, not raw `Option<T>`s — so
   `iou_thresholds=None` and `iou_thresholds=Some(linspace(0.5, 0.95, 10))`
   produce the same hash (they evaluate the same).
3. Cross-paradigm merge stays structurally rejected: ADR-0032's
   `vernier-partial` envelope already carries a `paradigm` enum and
   refuses heterogeneous merges. No additional protocol needed.
4. `PartialParamsMismatch` (defined at
   `crates/vernier-partial/src/error.rs:45–50`, re-exported from each
   paradigm's `__all__`) continues to fire when two ranks of the same
   paradigm carry diverging hashes.

The migration is mechanical: split the existing single struct into
three paradigm-specific structs, route each paradigm through its own
`params_hash` call. No wire-format change (ADR-0031 `FORMAT_VERSION`
stays at 2). Per-paradigm regression tests (one per sibling ADR) pin
that ranks with diverging post-resolution parameters refuse to merge.

### `Breakdown` lift to Python

ADR-0016 deferred Python exposure of `Breakdown` to a follow-up ADR.
This ADR is that follow-up.

The Rust shape (`vernier_core::breakdown::Breakdown { axis, buckets:
Vec<Bucket> }`) lifts to Python as one type with two factories:

```python
class Breakdown:
    @classmethod
    def from_ranges(cls, axis: str, buckets: Sequence[tuple[str, float, float]]) -> Breakdown: ...
        # f64-keyed; instance area ranges, future depth/occlusion axes.
    @classmethod
    def from_class_groups(cls, axis: str, groups: Mapping[str, Sequence[int]]) -> Breakdown: ...
        # class-id-keyed; semantic / panoptic class grouping.
```

Internal storage differs (`Vec<Bucket>` for f64-keyed continues
unchanged; class-id-keyed gains a parallel storage path), but the
Python user surface is one type. The closed-on-both-ends inclusion
semantics from ADR-0016 (quirk D6) carry over to the f64 factory; the
class-id factory rejects overlap (no class appears in two groups).

Schema continuity: ADR-0015's CLI JSON output schema bumps to v2 only
when a `Breakdown` is actually emitted in CLI output. Today that
happens only via the result-tables surface (ADR-0019), which already
carries bucket labels in its column schema. No breaking change to the
v1 default.

### Drop-in shim non-exposure

The new knobs do not appear on `vernier._compat.COCOeval` (the
`patch_pycocotools` shim) or any future drop-in shim. pycocotools
exposes none of them — the shim mirrors pycocotools, by ADR-0007.

Implementation cost worth flagging: today the shim's `_Params` class
at `python/vernier/_compat.py:133–172` does **not** define
`__setattr__` or `__getattr__`. Unknown attributes are silently
accepted via stock Python object behavior. The convention this ADR
ratifies — surfacing `AttributeError` with a pointer to the native
`Evaluator` and the relevant ADR — requires an explicit `__setattr__`
override on `_Params`. The override is a one-screen change but it is
not free; the per-paradigm ADRs cite this implementation cost rather
than treating the rule as a free assertion.

### Consequences

- **Positive.** Three paradigm ADRs land against a frozen shared
  contract. One sentinel rule, one validation hierarchy, one Python
  `Breakdown` shape. Distributed-eval cross-rank discipline survives
  the surface expansion. Drop-in shims do not invent non-pycocotools
  APIs.
- **Negative.** Lifting `Breakdown` to Python means the wheel ABI
  and the result-tables CLI schema gain a new public type — its
  shape is now under the same evolution discipline as the rest of
  `vernier.instance`. The construction-time validation requires every
  paradigm's `Evaluator.__post_init__` to grow a per-field check;
  the validation code is paradigm-specific and not shareable beyond
  the exception hierarchy.
- **Neutral.** No kernel changes. No FFI version bump. No change to
  `vernier-partial` `FORMAT_VERSION`. ADR-0006 immutable-config
  invariant survives — `Evaluator` stays a frozen dataclass.

## Pros and cons of the options

### Option 1 (status quo)

- 👍 No coordination cost; each paradigm ADR ships when its author
  is ready.
- 👎 Drift across paradigms is structural after 0.1.0. Three
  not-quite-compatible APIs are the kind of mistake a deprecation
  cycle can't undo cheaply. `Breakdown` Python shape gets designed
  three times.

### Option 2 (sibling ADRs with copy-pasted conventions)

- 👍 No new ADR file; no extra cross-reference burden.
- 👎 The shared rules (sentinel, validation, params_hash protocol,
  shim non-exposure, `Breakdown` lift) drift the moment one paradigm
  ADR amends one of them and the others don't follow. The "shared
  contract" is ceremonial, not load-bearing.

### Option 3 (chosen — cross-paradigm ADR + sibling ADRs)

- 👍 The shared contract has one home; amendments to the contract
  amend one ADR, not three. Matches the project's existing pattern
  (ADR-0029 referenced by 0025/0026/0028). Per-paradigm ADRs focus
  on paradigm-specific math without re-deriving conventions.
- 👎 One additional ADR file in the index. The sibling ADRs have to
  cross-reference this one explicitly.

## Links and references

- ADR-0001 — gates significance.
- ADR-0002 — three-tier parity model (vocabulary reused).
- ADR-0006 — immutable evaluator config (preserved).
- ADR-0007 — `patch_pycocotools` shim policy (extended to non-exposure
  of new knobs).
- ADR-0011 — discriminated kernel config (`IouKind`; orthogonal to
  the grid parameters).
- ADR-0012 — OKS keypoints surface; established the
  `max_dets: tuple[int, ...] | None = None` precedent this ADR
  generalizes.
- ADR-0015 — CLI JSON output schema (v1 preserved; v2 follow-up
  scoped to `Breakdown` emission).
- ADR-0016 — generalized `Breakdown` axis. This ADR lifts the
  deferral in §"Surface stays internal until it's not" (lines 55–58).
- ADR-0019 — result tables (downstream consumer of `Breakdown` labels).
- ADR-0026 — LVIS support (`CategoryFilter` enum precedent reused
  and extended in ADR-0041 / ADR-0042).
- ADR-0029 — namespace restructure (the cross-paradigm referencing
  pattern this ADR mirrors).
- ADR-0031 — distributed eval / partial wire format
  (`FORMAT_VERSION = 2` preserved; `params_hash` extended).
- ADR-0032 — distributed eval across paradigms (cross-paradigm-merge
  refusal preserved).
- ADR-0040 / ADR-0041 / ADR-0042 — paradigm-specific ratifications
  of this contract.
