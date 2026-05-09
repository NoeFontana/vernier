# ADR-0040: User-parametrizable instance evaluation grid

- **Status:** accepted
- **Date:** 2026-05-09
- **Accepted on:** 2026-05-09
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

> Renumbered from the user's implementation plan (which reserved
> ADR-0036). 0036 is taken (vendor mmsegmentation IoUMetric). This ADR
> ratifies the cross-paradigm conventions in ADR-0039 for the instance
> kernel; ADR-0041 and ADR-0042 do the same for semantic and panoptic.

## Context and problem statement

`vernier.instance.Evaluator` is a frozen dataclass at
`python/vernier/instance/__init__.py:184–216` with five user-facing
fields: `iou`, `parity_mode`, `max_dets`, `use_cats`, `cast_inputs`.
Three pycocotools-derived parameters that govern the evaluation grid
are *not* surfaced today:

- `iou_thresholds` — pycocotools' `[0.50, 0.55, …, 0.95]` ladder.
  Hardcoded in the kernel.
- `recall_thresholds` — pycocotools' `[0.0, 0.01, …, 1.0]` 101-point
  ladder for AP integration.
- `area_ranges` — pycocotools' `(all, small, medium, large)`
  buckets, partly addressed by the value-typed `Breakdown` shipped
  in ADR-0016 but not yet exposed at the FFI boundary.

ADR-0012 explicitly deferred the area-range surface (Option 8 in its
considered options): *"The right shape for area_rng is a `Breakdown`
object, not a tuple of tuples — shipping an interim shape commits us
to a breaking change when the Breakdown ADR lands."* That ADR has now
landed (ADR-0016) and ADR-0039 ratifies the Python lift. The
deferral is ready to be lifted.

Real users want three things the canonical defaults can't give them:

1. **Sensitivity analysis on IoU thresholds** — model selection at
   IoU=0.5 vs IoU=0.75 disagrees, and "is the gap a localization
   problem or a classification problem?" requires sweeping the ladder.
2. **Domain area buckets** — robotics evaluations want distance ranges
   (0–5 m / 5–15 m / 15–30 m / 30+ m); medical imaging wants size
   ranges that don't match COCO's `32²` / `96²` boundaries.
3. **Recall-threshold sampling** — extreme-recall regimes
   (autonomous-driving safety, medical screening) want a denser
   sampling near `R=1.0` than the canonical 101 evenly-spaced points.

The constraint is parity: pycocotools-strict mode must continue
producing byte-identical numbers on every existing fixture. The
canonical defaults (`linspace(0.5, 0.95, 10)`, `linspace(0.0, 1.0, 101)`,
COCO 4-bucket / 3-bucket area layouts) stay the parity baseline.
This ADR opens the surface for explicit overrides without disturbing
the default.

## Decision drivers

- **Cross-paradigm conventions (ADR-0039) are non-negotiable.** Sentinel
  resolution (`None` → kernel-canonical), construction-time validation,
  per-paradigm `params_hash` extension, drop-in shim non-exposure all
  apply verbatim.
- **Lift ADR-0012's Option 8 deferral, not redesign it.** The
  `Breakdown` shape from ADR-0016 plus the Python factory from
  ADR-0039 (`Breakdown.from_ranges(...)`) is the FFI surface for
  `area_ranges`. No new bucket type.
- **Strict-parity contract is non-negotiable.** Every fixture in
  `tests/python/parity/`, every val2017 parity smoke, and every LVIS
  parity row must continue producing byte-identical output under
  the default. The new fields are additive opt-in.
- **Boundary IoU inherits the surface unchanged.** Boundary's
  `dilation_ratio` lives on the `IouKind::Boundary` discriminated
  variant per ADR-0011. The new grid parameters are orthogonal —
  `Evaluator(iou=Boundary(dilation_ratio=0.02), area_ranges=...)`
  composes naturally.
- **The 12-stat / 10-stat / 13-stat summary plans don't generalize.**
  Pycocotools' summary plan (`coco_detection_default()`,
  `coco_keypoints_default()`, `lvis_default()`) keys directly on
  hardcoded slot indices in the `(T, R, K, A, M)` tensor — `AP_S` is
  *the second area-bucket entry of the all-IoU-thresholds slice at
  maxDet=100*, not "the small-area slot". Custom grids break the
  index assumption silently.

## Considered options

1. **Status quo + per-grid summary plan branches.** Leave the surface
   alone; ship `Evaluator.evaluate_with_grid(...)` that returns a
   custom-shaped summary.
2. **Surface the three fields and synthesize per-grid stat plans.**
   Detect the user's grid; build a `StatRequest` plan that maps slot
   indices into custom labels.
3. **Surface the three fields; raise `IncompatibleSummaryPlan` from
   `evaluate()` when any is set explicitly; route custom-grid users to
   `evaluate_tables()` (ADR-0019).** Tables already carry bucket
   labels; consumers key off labels, not indices.
4. **Surface the three fields; allow `evaluate()` to return a
   `Summary` whose `stats` array shape varies with the grid.** The
   shape becomes a function of construction.

## Decision outcome

Chosen option: **3 — surface the three fields, raise on summary
incompatibility, route to result tables.**

### New fields

```python
@dataclass(frozen=True, slots=True)
class Evaluator:
    iou: IouKind = field(default_factory=Bbox)
    parity_mode: ParityMode = "corrected"
    max_dets: tuple[int, ...] | None = None
    iou_thresholds: tuple[float, ...] | None = None       # NEW
    recall_thresholds: tuple[float, ...] | None = None    # NEW
    area_ranges: Breakdown | None = None                  # NEW
    use_cats: bool = True
    cast_inputs: bool = False
```

Sentinel resolution per ADR-0039:

| Field | `None` resolves to | Explicit override |
|---|---|---|
| `iou_thresholds` | `linspace(0.5, 0.95, 10)` (kernel canonical) | sorted, deduplicated, validated, used verbatim |
| `recall_thresholds` | `linspace(0.0, 1.0, 101)` | sorted, deduplicated, validated, used verbatim |
| `area_ranges` | `Breakdown::coco_area_det()` for det kernels; `Breakdown::coco_area_keypoints()` for OKS | user `Breakdown` validated, used verbatim |

`max_dets=None` already resolves per ADR-0012 — that precedent stays.
The keypoints kernel (ADR-0012) keeps its 3-bucket area default;
boundary (ADR-0010) inherits whatever the underlying instance kernel
resolves to.

### Validation contract

Construction-time validation in `Evaluator.__post_init__`, raising
`InvalidInstanceParams` (ADR-0039 hierarchy):

- `iou_thresholds`: non-empty; every element finite and in `[0.0, 1.0]`;
  sorted ascending; no duplicates.
- `recall_thresholds`: non-empty; every element finite and in `[0.0, 1.0]`;
  sorted ascending; no duplicates.
- `area_ranges` (`Breakdown`): non-empty; every bucket `lo <= hi`;
  every `lo` and `hi` finite and `>= 0`; no duplicate labels; no
  duplicate `(index, label)` pairs.

Each error carries field name, offending value, and a remediation
pointer (this ADR + the migration guide).

### `evaluate()` summary-incompatibility rule

If any of `iou_thresholds`, `recall_thresholds`, or `area_ranges` is
explicitly set (i.e., not `None`), `Evaluator.evaluate(gt, dt)` raises
a typed `IncompatibleSummaryPlan` exception pointing at
`Evaluator.evaluate_tables(gt, dt, tables="all")` (ADR-0019). The
exception message:

```
IncompatibleSummaryPlan: the canonical 12-stat / 10-stat / 13-stat
summary plans are keyed on hardcoded slot indices in the (T, R, K,
A, M) tensor — AP_S is "the second area-bucket entry of the all-IoU
slice at maxDet=100", not "the small-area slot". Your custom grid
breaks this index assumption. Use evaluate_tables(...) for tabular
output that carries explicit labels per row, or remove the custom
grid (set the field back to None) to use evaluate().
See ADR-0040 §"Decision outcome".
```

The result-tables surface (ADR-0019) already carries bucket labels in
its column schema (per-class `category_id`, per-image `image_id`,
per-pair `iou`, etc.). Adding the user's own bucket labels to the
per-image and per-pair tables is a leaf change; consumers key off
labels, not slot positions.

### Distributed-eval `params_hash` extension

Per ADR-0039: a paradigm-specific `InstanceParams` resolved-params
struct fingerprinted via rkyv → blake3. The hash covers post-sentinel
resolved values (so `iou_thresholds=None` and the canonical ladder
hash identically). `PartialParamsMismatch` continues to fire when two
ranks of the instance paradigm have diverging hashes; cross-paradigm
merge stays structurally rejected by ADR-0032's `vernier-partial`
envelope.

### Drop-in shim non-exposure

Per ADR-0039 §"Drop-in shim non-exposure": `vernier._compat.COCOeval`
does not expose any of the three new fields. The `_Params` class at
`python/vernier/_compat.py:133–172` gains an explicit `__setattr__`
override that surfaces `AttributeError` on `iou_thresholds`,
`recall_thresholds`, `area_ranges`, with a pointer to
`vernier.instance.Evaluator` and this ADR.

### Supersession framing

This ADR amends the relevant clauses of **ADR-0012** that pinned
`max_dets`-style sentinel resolution as a single-field pattern; the
pattern now generalizes per ADR-0039 to three additional fields on
the same `Evaluator`. ADR-0012's keypoints kernel-coupling rule
(`max_dets=None` resolves through the kernel) is preserved verbatim
and now extends to `area_ranges=None` (resolves to the kernel's
canonical `Breakdown`). ADR-0012 stays `accepted`.

This ADR also lifts the **Option 8 deferral** in ADR-0012 by
ratifying `area_ranges: Breakdown | None`. The "the right shape is a
`Breakdown`, not a tuple of tuples" rationale ADR-0012 gave for
deferring is now satisfied by ADR-0016's value type and ADR-0039's
Python factories.

### Consequences

- **Positive.** Sensitivity analysis, domain area buckets, and
  custom recall sampling become first-class. Every existing parity
  fixture, every val2017 parity smoke, every LVIS parity row passes
  unchanged. The result-tables surface (ADR-0019) gains real users
  for the per-image / per-pair output it already produces. The
  pycocotools shim (ADR-0007) stays a faithful pycocotools mirror.
- **Negative.** Three more public fields on
  `vernier.instance.Evaluator` mean three more validation paths,
  three more documentation rows, three more `params_hash` inputs.
  The `IncompatibleSummaryPlan` exception is a new failure mode —
  users who reach for `area_ranges` expecting `AP_S/M/L` to "just
  work" hit it. The error message has to do the migration work; the
  doc page (`docs/how-to/instance-grid-parameters.md`, scoped to a
  follow-up PR) needs a worked example.
- **Neutral.** No kernel changes. No FFI version bump. ADR-0006
  immutable-config invariant survives. The pyright-strict
  exhaustiveness check on `IouKind` matches still passes.

## Pros and cons of the options

### Option 1 (status quo + per-grid evaluate variant)

- 👍 Zero changes to `Evaluator`'s field list.
- 👎 Two parallel entry points (`evaluate` / `evaluate_with_grid`) for
  what is the same matching kernel; doubles the API surface for
  every paradigm-specific knob added later. ADR-0035 just removed
  three classes from this paradigm to consolidate the entry points;
  this option re-introduces parallel entry points.

### Option 2 (synthesize per-grid summary plans)

- 👍 `evaluate()` keeps working for every input.
- 👎 The 12-stat / 10-stat / 13-stat plan slot indices are pinned to
  pycocotools / lvis-api documentation. Synthesizing custom plans
  silently produces stat tuples whose indices the user has to look up
  — and there's no canonical reference for what `AP_S` "should mean"
  on a custom 7-bucket area split. Quirks survey honesty (ADR-0002
  three-tier) calls this out: misleading labels are worse than
  missing ones.

### Option 3 (chosen — IncompatibleSummaryPlan + tables route)

- 👍 The canonical summary stays bit-exact under defaults. Custom
  grids route to a surface (ADR-0019 tables) that already carries
  labels. The rule is one sentence: "custom grid → tables, not
  stats." Users can still get a single-number aggregate by folding
  the per-pair table themselves.
- 👎 Users have to rewrite their reporting code to consume tables
  instead of `summary.stats`. The error happens at evaluate time, not
  construction time — slightly later than the rest of the validation.
  We accept the late surface because the canonical-grid path is the
  common case and shouldn't pay a runtime check.

### Option 4 (variable-shape Summary)

- 👍 Single entry point; no exception.
- 👎 `Summary.stats` shape becomes a function of construction. Every
  consumer of `summary.stats[i]` now has to introspect the grid to
  know what `i` means. Breaks `pretty_lines()`, breaks the CLI JSON
  schema (ADR-0015), breaks every existing `summary.stats[0]` call
  site silently.

## Links and references

- ADR-0001 — gates significance.
- ADR-0002 — three-tier parity model (strict baseline preserved).
- ADR-0005 — `Similarity` trait + matching-engine API lock (untouched).
- ADR-0006 — frozen `Evaluator` discipline (preserved).
- ADR-0007 — `patch_pycocotools` shim (extended to non-exposure of new
  knobs per ADR-0039).
- ADR-0010 — boundary IoU isolated subsystem; inherits this surface
  via `IouKind::Boundary`.
- ADR-0011 — discriminated kernel config (`IouKind`); orthogonal to
  the grid parameters.
- ADR-0012 — OKS keypoints surface; this ADR amends the `max_dets`
  sentinel pattern to three additional fields and lifts Option 8's
  `area_rng` deferral.
- ADR-0015 — CLI JSON schema (v1 preserved; per-grid output will
  flow through ADR-0019's tables surface).
- ADR-0016 — generalized `Breakdown` axis (the type
  `area_ranges: Breakdown | None` consumes; Python factory ratified
  in ADR-0039).
- ADR-0019 — result tables; the canonical home for custom-grid
  output.
- ADR-0026 — LVIS support; `lvis_default()` 13-stat plan inherits the
  same incompatibility rule.
- ADR-0029 — namespace restructure (`vernier.instance` namespace).
- ADR-0031 / ADR-0032 — distributed-eval `params_hash` (extension
  ratified in ADR-0039).
- ADR-0039 — cross-paradigm conventions (this ADR ratifies them).
- `docs/engineering/pycocotools-quirks.md` — D4 / D5 / D6 area-bucket
  semantics preserved by ADR-0016's closed-on-both-ends rule.
