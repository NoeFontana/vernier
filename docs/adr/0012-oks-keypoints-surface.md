# ADR-0012: OKS keypoints public surface and kernel-coupled defaults

- **Status:** accepted
- **Date:** 2026-04-28
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

## Context and problem statement

Phase 3 kicks off with COCO val2017 keypoints parity. The Rust core has a
`Similarity` trait per ADR-0005 and a discriminated kernel-config union
per ADR-0011, so the *shape* of a new kernel is largely settled —
`Keypoints` becomes another `IouKind` variant; `OksSimilarity` becomes
another `Similarity` impl. What is *not* settled, and what this ADR
pins, is two interlocking questions:

1. **Sigmas shape.** How are the per-keypoint OKS standard deviations
   exposed to the user, given that pycocotools hardcodes a 17-element
   COCO-person table (`cocoeval.py:523`) and downstream forks
   (CrowdPose, MMPose) monkey-patch it post-construction?
2. **Kernel-coupled summary defaults.** How do the *other* fields that
   pycocotools' `setKpParams()` rewrites — `maxDets = [20]` (vs
   `[1, 10, 100]` for det), `areaRng` as a 3-entry array that drops the
   `small` bucket — survive the move from a mutable `Params` bag to
   a frozen, value-typed `Evaluator`?

The two questions look separable but aren't. `setKpParams()` rewrites
`kpt_oks_sigmas`, `maxDets`, and `areaRng` *atomically* — the
"keypoints" identity in pycocotools is the union of those three, not
just the kernel choice. Vernier's discriminated union per ADR-0011
makes the kernel a value rather than mutable state, but the *coupling*
that `setKpParams()` encodes is real. A user who selects keypoints and
leaves the eval grid at the bbox/segm defaults is silently asking for
pycocotools-divergent numbers — the headline parity claim ("default
behavior matches pycocotools") collapses unless the coupling is
preserved on this side of the FFI too.

Quirk **F1** in `docs/engineering/pycocotools-quirks.md` flags
sigmas-as-`corrected` precisely because faithful reproduction would
foreclose multi-category keypoints (LVIS-Pose, CrowdPose+COCO-person
mixed datasets, future robotics fixtures). The other keypoint quirks
(F2–F5, D2, D5, L8) have natural homes in the same document.

## Decision drivers

- **Match pycocotools-canonical defaults out of the box on every
  kernel.** `Evaluator(iou=Keypoints())` should produce bit-identical
  numbers to a fresh pycocotools `COCOeval(iouType="keypoints")` on
  COCO-person, with no migration setup. Whatever the right call for
  sigmas is, the rest of the eval grid (max_dets, area_rng) cannot
  silently disagree.
- **F1 disposition is `corrected` per the quirks survey.** Per-category
  sigmas must be expressible without monkey-patching.
- **Native API and `patch_pycocotools` shim agree by construction.** The
  shim mirrors `setKpParams` semantics; the native API must reach the
  same answer for the same input. Two paths, one outcome.
- **No `Evaluator` parameter sprawl.** ADR-0011's discriminated union is
  the home for kernel-local params. `Keypoints` carries its sigmas;
  shared summary fields stay shared but resolve from the kernel where
  the kernel canonicalizes them.
- **Frozen `Evaluator` (ADR-0006).** Resolution happens at dispatch via
  sentinel fields, not via post-construction mutation.
- **`pyright --strict` exhaustiveness.** Adding `Keypoints` to `IouKind`
  must surface every existing `match self.iou:` site that lacks a
  `Keypoints()` arm.
- **Mirror the Rust `EvalIouType` enum.** FFI dispatch is a single
  match arm.
- **Pre-1.0 freedom.** No deprecation cycle is owed.

## Considered options

### For sigmas shape

1. **Single global sigmas tuple** mirroring `Params.kpt_oks_sigmas`.
   Single skeleton, no per-category override.
2. **Per-category sigmas mapping keyed on category_id**, with the empty
   mapping meaning "use COCO-person for every category".
3. **Per-category sigmas via callable.** Maximum flexibility; opaque to
   introspection.
4. **`Params`-style mutable bag.** Matches pycocotools shape literally;
   conflicts with ADR-0006.

### For kernel-coupled summary defaults

5. **Sentinel default + per-kernel resolution at dispatch.** `max_dets:
   tuple[int, ...] | None = None`; `None` resolves to the kernel-
   canonical default (`(1, 10, 100)` for Bbox/Segm/Boundary, `(20,)`
   for Keypoints) inside `evaluate()`. Explicit values always win.
6. **User-explicit defaults.** User sets `max_dets=(20,)` for keypoints;
   library never auto-switches. (This was the original draft of this
   ADR; reversed during review.)
7. **Per-variant fields.** Push `max_dets` onto each `IouKind` variant.
   Each kernel carries its canonical default; users override per
   variant.
8. **Surface `area_rng` as a public Evaluator field now**, paired with
   `max_dets`. Surfaces both `setKpParams`-rewritten dimensions in this
   ADR.

## Decision outcome

Chosen options: **2** for sigmas (per-category mapping) and **5** for
summary defaults (sentinel + per-kernel resolution). `area_rng` stays
internal in this ADR and is re-litigated when the future Breakdown ADR
lands.

### Public Python surface

```python
@dataclass(frozen=True, slots=True)
class Keypoints:
    """Object-keypoint similarity (OKS) kernel selector (ADR-0012).

    ``sigmas`` overrides the per-keypoint standard deviation used in the
    OKS formula on a per-category basis. The empty default means every
    category uses the 17-element COCO-person table — pycocotools-
    equivalent behavior on single-category-person datasets.
    """

    sigmas: Mapping[int, tuple[float, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Evaluator:
    iou: IouKind = field(default_factory=Bbox)
    parity_mode: ParityMode = "corrected"
    max_dets: tuple[int, ...] | None = None     # was tuple[int, ...] = (1, 10, 100)
    use_cats: bool = True
```

`max_dets` flips from `(1, 10, 100)` to `None`. `None` is a sentinel
meaning "use the kernel-canonical default for whichever `iou` is
selected"; an explicit value always wins.

### Dispatch resolution

```python
_KERNEL_MAX_DETS: dict[type[IouKind], tuple[int, ...]] = {
    Bbox: (1, 10, 100),
    Segm: (1, 10, 100),
    Boundary: (1, 10, 100),
    Keypoints: (20,),
}

def _resolve_max_dets(self) -> list[int]:
    if self.max_dets is not None:
        return list(self.max_dets)
    return list(_KERNEL_MAX_DETS[type(self.iou)])
```

Resolution lives at the dispatch site, next to the `match self.iou:` it
informs — single source of truth for kernel-canonical params, exhaustive
under `pyright --strict` (a new variant added without a `_KERNEL_MAX_DETS`
entry trips the table lookup at construction time on first use; we add a
`pytest` parametrized over `IouKind.__args__` to fail fast in CI).

The `Evaluator.evaluate` arm for keypoints:

```python
case Keypoints(sigmas=sig):
    return evaluate_keypoints_summary(
        gt, dt, self.parity_mode, self._resolve_max_dets(),
        self.use_cats, dict(sig),
    )
```

Behavior across the four kernels under default construction:

| Construction                              | Resolved `max_dets` | Equivalent pycocotools call             |
| ----------------------------------------- | ------------------- | --------------------------------------- |
| `Evaluator()` / `Evaluator(iou=Bbox())`   | `[1, 10, 100]`      | `COCOeval(..., iouType="bbox")` default |
| `Evaluator(iou=Segm())`                   | `[1, 10, 100]`      | `iouType="segm"` default                |
| `Evaluator(iou=Boundary())`               | `[1, 10, 100]`      | `iouType="boundary"` default (ADR-0010) |
| `Evaluator(iou=Keypoints())`              | `[20]`              | `iouType="keypoints"` default (post-`setKpParams`) |
| `Evaluator(iou=Keypoints(), max_dets=(1, 10, 100))` | `[1, 10, 100]` | `params.maxDets = [1, 10, 100]` after `setKpParams` (intentional override) |

### Rust FFI

`evaluate_keypoints_summary(..., sigmas: HashMap<i64, Vec<f64>>)` is a
new pyfunction. Internally, the keypoints kernel uses pycocotools' kp
`area_rng` 3-entry array `[(0, 1e10), (1024, 9216), (9216, 1e10)]` — no
`small` bucket (quirk **D5** strict). `area_rng` is not a public
`Evaluator` field today; the future Breakdown ADR (CrowdPose-triggered)
surfaces it with the same sentinel-resolution shape used here for
`max_dets`.

### Implementation note: `Evaluator.with_options`

`max_dets` becoming nullable means `with_options(max_dets=None)` is
ambiguous between "leave unchanged" and "reset to kernel-canonical."
The implementation threads a private sentinel
(`_UNSET`-style singleton) through `with_options` so that `None` means
"reset" and the absence of the keyword means "leave unchanged." This is
helper-method scope; the public `Evaluator` constructor is unaffected.

### Quirk dispositions (ratified)

| Quirk | Disposition | Implementation note |
| ----- | ----------- | ------------------- |
| **F1** | `corrected` | Per-category sigmas mapping; empty = COCO-person for all. |
| **F2** | `aligned`   | OKS area uses `gt.area + f64::EPSILON` (divide-by-zero guard). |
| **F3** | `strict`    | DT with no visible keypoints falls back to a 2× bbox surrogate metric, exactly per pycocotools. |
| **F4** | `strict`    | OKS bbox-expansion `[bb.x - bb.w, bb.x + 2*bb.w]` asymmetric — preserved. |
| **F5** | `aligned`   | `compute_oks` returns a correctly-shaped 0-row/col array when either side is empty (pycocotools returns `[]`; downstream consumers treat both identically). |
| **D2** | `strict`    | GT with 0 visible keypoints is implicit ignore — OR'd into `_ignore` as pycocotools. |
| **D5** | `strict`    | Keypoints kernel internally uses pycocotools' kp `area_rng` 3-entry array `[(0, 1e10), (1024, 9216), (9216, 1e10)]` — no `small` bucket. Internal today; surfaced by the future Breakdown ADR. |
| **L8** | `aligned`   | `Params.kpt_oks_sigmas` leaks across iou-types in pycocotools because `Params` is one mutable bag. Vernier's discriminated union forecloses the leak by construction. The same applies to the kernel-coupled `max_dets` and `area_rng` defaults — they live where they belong (per-kernel resolution at dispatch) rather than as stale state. |

### Consequences

- **Positive:** Default behavior is bit-identical to pycocotools on
  single-category COCO-person across the *full* eval grid (sigmas,
  max_dets, area_rng). The headline parity claim is actually true, not
  just true-for-sigmas.
- **Positive:** One mental rule across the API: "field unset →
  kernel-canonical default." `patch_pycocotools` and the native
  `Evaluator` API agree by construction, eliminating a class of "two
  paths, two answers" migration footguns.
- **Positive:** F1 corrected disposition shipped without monkey-patching;
  multi-skeleton evaluations (CrowdPose + COCO-person, robotics
  fixtures) Just Work.
- **Positive:** No `Evaluator` field churn beyond `max_dets` widening to
  nullable — pre-1.0 callers absorb the change mechanically; ADR-0011
  already taught them the migration shape.
- **Negative:** `max_dets` widens to `tuple[int, ...] | None`. Callers
  who introspect `evaluator.max_dets` see `None` rather than
  `(1, 10, 100)` until they call `evaluate()` or set it explicitly.
  Documented in the docstring.
- **Negative:** A small per-kernel default table (`_KERNEL_MAX_DETS`)
  lives near the dispatch site. Adding a new kernel without an entry
  fails fast; the cost is one dictionary line per kernel.
- **Negative:** `with_options(max_dets=None)` is no longer a no-op and
  requires a sentinel singleton in the helper to distinguish "leave"
  from "reset." Implementation detail; doesn't escape into the public
  API.
- **Neutral:** F4's asymmetric bbox expansion stays `strict`; could be
  promoted to `corrected` later without breaking the public surface.

## Out of scope

- **Generalized `Breakdown` mechanism.** The Phase-3 plan calls for an
  ADR that replaces hardcoded small/medium/large area ranges with a
  generic `Breakdown { name, key_fn, ranges }`. CrowdPose's `crowdIndex`
  is the trigger. Until then, `area_rng` stays internal and the
  keypoints kernel uses pycocotools' 3-entry kp default per D5.
- **Pose-PCK / OKS-AP10K / non-COCO keypoint datasets.** `EvalDataset`
  extensions, separate ADR if they perturb sigmas semantics.

## Pros and cons of the options

### For sigmas shape

#### Option 1 — single global sigmas tuple

- 👍 Smallest surface change; one tuple, no mapping.
- 👍 Trivial pycocotools-equivalence on COCO-person.
- 👎 Forecloses multi-category keypoints. F1 stays `strict` rather than
  `corrected`; downstream forks keep monkey-patching.

#### Option 2 — per-category mapping (chosen)

- 👍 F1 `corrected` shipped without monkey-patching.
- 👍 Default (empty mapping) is byte-identical to pycocotools on
  COCO-person.
- 👍 Mirrors the natural Rust `HashMap<i64, Vec<f64>>` shape; FFI is
  mechanical.
- 👎 Slightly wordier construction for the COCO-person common case
  (`Keypoints()` works, but multi-category users build a mapping).

#### Option 3 — sigmas as a callable

- 👍 Maximum flexibility (e.g., interpolated sigmas, learned schedules).
- 👎 Opaque to introspection / serialization. Cannot mirror across the
  FFI without serializing the closure.
- 👎 Speculative complexity; no Phase-3 use case.

#### Option 4 — pycocotools-style mutable bag

- 👍 Matches `Params.kpt_oks_sigmas` shape literally.
- 👎 Conflicts with ADR-0006's frozen `Evaluator` discipline.
- 👎 Recreates the L8 leak (kpt_oks_sigmas surviving an `iouType` flip).

### For kernel-coupled summary defaults

#### Option 5 — sentinel default + per-kernel resolution (chosen)

- 👍 Pycocotools-canonical defaults out of the box on every kernel.
- 👍 One uniform rule across the API: "field unset → kernel-canonical."
- 👍 Explicit values always win — overrides survive without surprise.
- 👍 `patch_pycocotools` and native API agree by construction.
- 👍 Minimal API surface delta — one field's type widens.
- 👎 Nullable user-facing type for `max_dets`.
- 👎 One small dispatch table to maintain (mitigated by `pyright`
  exhaustive `match` and a CI-checked exhaustiveness fixture).

#### Option 6 — user-explicit defaults

- 👍 Keeps `max_dets` simple `tuple[int, ...]`; no nullable type.
- 👍 Kernel choice strictly orthogonal to summary config.
- 👎 Silent footgun: `Evaluator(iou=Keypoints())` produces
  pycocotools-divergent numbers without obvious cause.
- 👎 Native `Evaluator` and `patch_pycocotools` disagree on the same
  input — same library, two paths, two answers.
- 👎 The "orthogonality" the option preserves is fictional anyway —
  `parity_mode`, `max_dets`, and `use_cats` are *all* parity-contract
  fields; `iou` was never uniquely coupled.

#### Option 7 — per-variant fields

- 👍 Each kernel carries its canonical default explicitly on the variant.
- 👎 ADR-0011 explicitly avoids parameter sprawl across variants;
  duplicates `max_dets` on every kernel.
- 👎 Fragments the user-facing override path — bbox callers who want
  `max_dets=(1, 10, 100, 1000)` must override on the variant rather
  than at the Evaluator level.

#### Option 8 — surface `area_rng` as a public Evaluator field now

- 👍 Pairs both `setKpParams`-rewritten dimensions in one ADR.
- 👍 Lets users override area_rng for their domain today.
- 👎 Expands the public surface beyond what this ADR's primary scope
  (sigmas) justifies.
- 👎 The right shape for area_rng is a `Breakdown` object, not a tuple
  of tuples — shipping an interim shape commits us to a breaking
  change when the Breakdown ADR lands.
- 👎 Defer.

## Links and references

- ADR-0001 §"Affect the public API" — change requires an ADR.
- ADR-0002 — three-tier parity model that frames F1 corrected.
- ADR-0005 — locks `Similarity` and matching-engine APIs; `OksSimilarity`
  slots in.
- ADR-0006 — frozen `Evaluator` discipline.
- ADR-0007 — `patch_pycocotools` shim; resolves the same kernel-canonical
  defaults on behalf of migrating users.
- ADR-0011 — discriminated kernel config; `Keypoints` is a new variant.
- `docs/engineering/pycocotools-quirks.md` — F1, F2, F3, F4, F5, D2, D5, L8.
- pycocotools 2.0.11 `cocoeval.py:215-235, 502-525` — `setKpParams` and
  the OKS reference.
