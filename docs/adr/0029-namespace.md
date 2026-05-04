# ADR-0029: Restructure the public API into per-paradigm submodules

- **Status:** accepted
- **Date:** 2026-05-03
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

Through 0.0.1 the public Python surface lives flat under
`vernier`: `Evaluator`, `COCOeval`, `Bbox`, `Segm`, `Boundary`,
`Keypoints`, `Summary`, `EvalResult`, `patch_pycocotools`,
`StreamingEvaluator`, `BackgroundEvaluator`. ADR-0025 adds
`PanopticEvaluator`, `PanopticDataset`, `PanopticPredictions`,
`PanopticSummary`, `SegmentInfo`. ADR-0026 extends `CocoDataset`
with federated fields and adds `Frequency`, `CategoryFilter`,
`summarize.lvis_default()`. ADR-0028 adds `SemanticEvaluator`,
`SemanticDataset`, `SemanticPredictions`, `SemanticSummary`,
`ClassSemanticStats`, `ConfusionMatrix`, plus the per-dataset
constructors `cityscapes()`, `ade20k()`, `pascal_voc()`.

If all of those land flat under `vernier`, the import root
becomes a flat enumeration of every type in the project, with
collisions waiting to happen — `Evaluator` is unambiguous today,
`SemanticEvaluator` and `PanopticEvaluator` are explicitly
disambiguated, and once the three pipelines accumulate sibling
types (Dataset / Predictions / Summary / Stats / streaming
variants), the flat root tells the wrong story about which
types compose with which. ADR-0028 §"Negative consequences"
explicitly flagged this: "the `SemanticEvaluator` /
`PanopticEvaluator` / `Evaluator` triple is a real ergonomic
cost; the namespace question (B4) is deferred to a follow-up
but worth flagging."

This is that follow-up. The question this ADR answers is:
*does the public Python surface restructure into per-paradigm
submodules before 0.1.0, and if so, what's the migration
shape for the small flat-import population?*

The decision is bounded by two facts. First, **pre-1.0 freedom
is real but not unlimited**. ADR-0011 established the precedent
that breaking changes inside the 0.0.x line are acceptable when
the future shape is materially better; a deprecation cycle is
not owed before 0.1.0. The same precedent applies here.
Second, **the moment to restructure is before the surface
multiplies**. ADR-0028 is the inflection point: the semantic
crate triples the type count and triples the documentation
surface for "how do I evaluate". Restructuring after the
flat-root population grows from one paradigm to three is more
expensive than restructuring now, before semantic and panoptic
land.

This ADR triggers ADR-0001 §"Affect the public API" (every
public name moves), §"Set a project-wide convention" (the
naming/layout convention is the load-bearing decision), and
implicitly §"Cross the FFI boundary" (no FFI symbol changes,
but the Python wrappers around the FFI symbols are reorganized).

## Decision drivers

- **Pre-1.0 freedom is the right time.** Breaking imports inside
  0.0.x is inexpensive; breaking them post-0.1.0 is a
  deprecation cycle. ADR-0011 set the precedent for "do the
  breaking change before 0.1.0 freezes the surface."
- **The flat root tells the wrong story.** A user reading
  `from vernier import Evaluator, PanopticEvaluator,
  SemanticEvaluator, COCOeval, ConfusionMatrix, Frequency` has
  no signal that the first three are mutually-exclusive
  evaluation paradigms and the rest are kernel-internal. A
  per-paradigm submodule makes that obvious.
- **Three is the trigger, not two.** With instance + panoptic,
  flat names disambiguate fine (`Evaluator` vs.
  `PanopticEvaluator`). With instance + panoptic + semantic +
  whatever 4th paradigm appears, the disambiguation suffix
  becomes the dominant pattern in every type name. The
  submodule shape scales; the suffix shape doesn't.
- **No `__init__.py`-injected re-exports of the old names.**
  ADR-0007's `patch_pycocotools` is a sanctioned monkey-patch
  with a real cost-benefit story (migration from a third-party
  library). The flat-root namespace re-export has no
  third-party migration story to justify it; it just lets
  users avoid updating imports. Pre-1.0 freedom means we don't
  owe them that.
- **`patch_pycocotools` and `COCOeval` stay where they are.**
  Those are the migration shim for users coming from
  pycocotools, and pycocotools is a single library with a
  single import root. Dropping `vernier.COCOeval` to
  `vernier.instance.COCOeval` would break the
  drop-in-replacement claim in ADR-0007 ("you only changed the
  import"). The flat-root keeps exactly two symbols:
  `COCOeval` and `patch_pycocotools`.
- **The Rust side is unaffected.** Crate names stay
  `vernier-core`, `vernier-mask`, `vernier-panoptic`,
  `vernier-semantic`. This ADR is a Python-surface
  reorganization only; the FFI symbols and the `vernier._core`
  extension module are unchanged. ADR-0009's leaf-direction
  discipline carries through verbatim.
- **Mirror what users already type.** The community standard
  for multi-paradigm Python libraries is `library.submodule.X`
  (sklearn, scipy, statsmodels, torch, jax). Users are
  pre-trained for the shape; deviating from it has to clear a
  high bar.

## Considered options

### Axis A — Top-level structure

1. **Flat root, with disambiguation suffixes**
   (`vernier.Evaluator`, `vernier.PanopticEvaluator`,
   `vernier.SemanticEvaluator`). Status quo extended.
2. **Per-paradigm submodules**
   (`vernier.instance`, `vernier.panoptic`,
   `vernier.semantic`), each exposing its own `Evaluator`,
   `Dataset`, `Predictions`, `Summary`. Shared types
   (`ParityMode`, `Breakdown`, `CategoryMeta`) at the root.
3. **Per-paradigm submodules with deep aliases at the root**
   (B with `vernier.PanopticEvaluator =
   vernier.panoptic.Evaluator`, etc.). Both paths work; users
   pick.

### Axis B — Migration shim for the flat-root population

1. **No shim.** Old imports break; users update.
2. **Re-export with `DeprecationWarning`.** Old imports work
   for one minor; warnings point to the new path.
3. **`vernier.compat.flat_root` import.** `from
   vernier.compat.flat_root import *` restores the old layout;
   off by default, opt-in for users who want one bisect-cycle
   of grace period.

### Axis C — `Evaluator` shape inside `vernier.instance`

1. **Single `vernier.instance.Evaluator` class** — same as
   today's `vernier.Evaluator`, just relocated.
2. **Split per-IoU-kind classes** —
   `vernier.instance.BboxEvaluator`,
   `vernier.instance.SegmEvaluator`, etc.

### Axis D — Where shared types live

1. **Top-level**: `vernier.ParityMode`, `vernier.Breakdown`,
   `vernier.CategoryMeta`, `vernier.ImageId`. Importable as
   `from vernier import ParityMode`.
2. **`vernier.types`** submodule for everything cross-cutting.
3. **Duplicated** in each paradigm submodule
   (`vernier.instance.ParityMode`,
   `vernier.panoptic.ParityMode`).

## Decision outcome

Chosen: **A2 + B1 + C1 + D1.**

### Top-level structure (A2)

The Python surface restructures into three per-paradigm
submodules:

```
vernier/
├── __init__.py              # shared types + COCOeval / patch_pycocotools
├── instance/
│   └── __init__.py          # Evaluator, COCOeval-equivalent types,
│                            # Bbox, Segm, Boundary, Keypoints, IouKind,
│                            # Summary, EvalResult, StreamingEvaluator,
│                            # BackgroundEvaluator, CocoDataset,
│                            # CocoDetections, Frequency, CategoryFilter
├── panoptic/
│   └── __init__.py          # Evaluator, Dataset, Predictions, Summary,
│                            # SegmentInfo, ClassPanopticStats
├── semantic/
│   └── __init__.py          # Evaluator, Dataset, Predictions, Summary,
│                            # ClassSemanticStats, ConfusionMatrix
└── summarize/               # already exists; gains lvis_default(),
                             # coco_panoptic_default(), cityscapes_default()
```

The top-level `vernier` namespace keeps exactly:

- **Shared types**: `ParityMode`, `Breakdown`, `Bucket`,
  `CategoryMeta`, `ImageId`, `CategoryId`, `Frequency`. Every
  paradigm reads these; duplicating them per submodule would
  be confusing.
- **The pycocotools migration shim**: `COCOeval` and
  `patch_pycocotools`. ADR-0007 commits to "drop-in for
  pycocotools imports"; preserving that means the symbol
  lives where pycocotools-shaped imports expect it.
  (`vernier.COCOeval` shadows `pycocotools.cocoeval.COCOeval`
  via `patch_pycocotools` on the user's request.) The shim
  internally constructs an `instance.Evaluator` — the public
  symbol is at the root, the implementation is inside the
  `instance` submodule.
- **The version string** `__version__`.

Everything else moves into a paradigm submodule.

### `vernier.instance.Evaluator` keeps the existing surface (C1)

Inside `vernier.instance`, the `Evaluator` class is the same
class today's `vernier.Evaluator` is — same constructor, same
`evaluate()` signature, same `IouKind` discriminated union,
same defaults. The class is *moved*, not split. Splitting per
IoU kind (`BboxEvaluator`, `SegmEvaluator`, …) would multiply
the type count for no gain — `IouKind` already discriminates
what they would. C1 over C2.

### Shared types live at the top level (D1)

`ParityMode`, `Breakdown`, `Bucket`, `CategoryMeta`, `ImageId`,
`CategoryId`, and `Frequency` are imported as
`from vernier import ParityMode`. They are shared across
paradigms; duplicating them per submodule would force users to
think about which paradigm's `ParityMode` they want, when the
answer is always "the only one that exists." A `vernier.types`
submodule (D2) is the alternative — slightly more discoverable
in IDE completion, slightly more typing per import. D1 wins on
ergonomics; D2 is the right shape if the shared-types list ever
grows past ~10 entries (which it shouldn't for an evaluation
library).

### No flat-root re-exports (B1)

The old flat-root names are removed. Users update their
imports:

| Before                              | After                                  |
|-------------------------------------|----------------------------------------|
| `vernier.Evaluator`                 | `vernier.instance.Evaluator`           |
| `vernier.Bbox`                      | `vernier.instance.Bbox`                |
| `vernier.Segm`                      | `vernier.instance.Segm`                |
| `vernier.Boundary`                  | `vernier.instance.Boundary`            |
| `vernier.Keypoints`                 | `vernier.instance.Keypoints`           |
| `vernier.IouKind`                   | `vernier.instance.IouKind`             |
| `vernier.Summary`                   | `vernier.instance.Summary`             |
| `vernier.EvalResult`                | `vernier.instance.EvalResult`          |
| `vernier.TableName`                 | `vernier.instance.TableName`           |
| `vernier.TablesConfig`              | `vernier.instance.TablesConfig`        |
| `vernier.StreamingEvaluator`        | `vernier.instance.StreamingEvaluator`  |
| `vernier.BackgroundEvaluator`       | `vernier.instance.BackgroundEvaluator` |
| `vernier.Dataset`                   | `vernier.instance.Dataset`             |
| `vernier.MemoryBudgetWarning`       | `vernier.instance.MemoryBudgetWarning` |
| `vernier.OutOfBudgetError`          | `vernier.instance.OutOfBudgetError`    |
| `vernier.QueueFullError`            | `vernier.instance.QueueFullError`      |
| `vernier.confusion_matrix`          | `vernier.instance.confusion_matrix`    |
| `vernier.error_decomposition`       | `vernier.instance.error_decomposition` |
| `vernier.fp_iou_histogram`          | `vernier.instance.fp_iou_histogram`    |
| `vernier.FpIouHistogram`            | `vernier.instance.FpIouHistogram`      |
| `vernier.TideConfig`                | `vernier.instance.TideConfig`          |
| `vernier.TideReport`                | `vernier.instance.TideReport`          |
| `vernier.PanopticEvaluator`         | `vernier.panoptic.Evaluator`           |
| `vernier.PanopticDataset`           | `vernier.panoptic.Dataset`             |
| `vernier.PanopticPredictions`       | `vernier.panoptic.Predictions`         |
| `vernier.PanopticSummary`           | `vernier.panoptic.Summary`             |
| `vernier.ClassPanopticStats`        | `vernier.panoptic.ClassPanopticStats`  |
| `vernier.ParityMode`                | `vernier.ParityMode` (unchanged)       |
| `vernier.Frequency`                 | `vernier.Frequency` (unchanged, shared)|
| `vernier.COCOeval`                  | `vernier.COCOeval` (unchanged)         |
| `vernier.patch_pycocotools`         | `vernier.patch_pycocotools`            |

Three deferrals shape the table:

- **No semantic-* rows.** `SemanticEvaluator`, `SemanticDataset`,
  `SemanticPredictions`, `SemanticSummary`, `ConfusionMatrix` land
  in `vernier.semantic` *as the implementation does*, under
  ADR-0028. Pre-listing them here would commit this ADR to types
  that don't exist yet.
- **`Dataset` → `CocoDataset` rename deferred.** The FFI pyclass
  is named `Dataset` today; renaming the constructor is a separate
  source-compatibility break, scoped to a follow-up 0.0.x patch.
  This ADR moves the symbol as `Dataset` into `vernier.instance`.
- **No aspirational shared-type exports.** `Breakdown`, `Bucket`,
  `CategoryMeta`, `ImageId`, `CategoryId`, `CocoDetections`, and
  `CategoryFilter` were considered for the root surface in earlier
  drafts but are not currently Python-exposed types; surfacing them
  preemptively violates KISS. They land at the root if and when a
  consumer (e.g., ADR-0028's `Breakdown`-aware semantic mIoU)
  needs them.

The `python/vernier/summarize/` submodule sketched in the layout
diagram above is similarly deferred: it is created when the first
cross-paradigm helper needs a home (planned alongside ADR-0028's
result-tables follow-up), not as an empty placeholder.

Pre-1.0 freedom + the small flat-root population (we have no
external users on 0.0.x making this a "real" breaking change
yet) makes B1 the right call. The migration is a single
search-and-replace per moved symbol; a sed script in the
release notes covers most users in two minutes.

The `DeprecationWarning` shim (B2) and the opt-in
`vernier.compat.flat_root` (B3) are both real options if user
demand materializes after 0.1.0 ships. Neither is in the
initial restructure, because the right time to find out
whether they're needed is after the new shape lands. If the
GitHub Discussions tab fills up with import errors in the week
post-0.1.0, B2 lands as a 0.1.x patch.

### Three things this ADR explicitly preserves

- **`COCOeval` at the root.** ADR-0007's drop-in claim
  ("change one import line and your eval code runs") is
  load-bearing for adoption. Moving `COCOeval` to
  `vernier.instance.COCOeval` would cost a bullet point in
  every migration discussion. The cost-benefit there points
  hard at "leave it where pycocotools-aware users expect it."
- **`patch_pycocotools` at the root.** Same reasoning.
- **The `vernier._core` extension module.** Unchanged. The
  Rust side does not know about Python submodule layout.

## What this ADR explicitly does *not* decide

- **Renaming `instance` to something else.** "instance"
  matches the ADR-0025 / ADR-0028 vocabulary
  ("instance segmentation" vs. "panoptic segmentation"
  vs. "semantic segmentation"). Alternatives considered:
  `vernier.detection` (narrower; excludes keypoints which the
  AP fold already handles), `vernier.coco` (couples the
  paradigm name to a dataset). Neither improves on `instance`.
- **A `vernier.tracking` submodule** for video / tracking
  metrics. `Possible_Extensions` flags 3D / BEV / tracking as
  separate-package material; this ADR makes room for the
  submodule shape if a future ADR brings tracking in-tree, but
  takes no position on whether tracking belongs in vernier at
  all.
- **The `vernier-cli` flag layout.** ADR-0015's CLI surface
  is unchanged by this ADR — the CLI is a process boundary,
  not a Python import surface. `vernier eval --kernel
  panoptic ...` continues to work; the CLI dispatch happens
  in `vernier-cli` Rust code.
- **Re-exporting Rust types in Python.** Each paradigm
  submodule re-exports the Rust-backed types it owns. No new
  cross-language exports vs. the flat-root surface; just a
  different organizational layer above them.
- **A `vernier.utils` submodule.** No utility functions
  warrant a new submodule today. If one accumulates, a
  future ADR adds `vernier.utils`.
- **Whether 0.1.0 ships with all three paradigms or in
  stages.** This ADR commits to the *layout* the three
  paradigms land in, not the *order*. ADR-0025 and ADR-0028
  set the implementation order; 0.1.0's release manager picks
  which paradigms ship in which patch.

## Consequences

- **Positive.** The three-paradigm surface is legible at the
  import-root level. A user typing `vernier.<TAB>` sees
  `instance`, `panoptic`, `semantic`, `summarize`, plus the
  shared types — one tier of cognitive load instead of a
  flat list of 25+ symbols. Documentation organization
  mirrors the namespace organization (per-paradigm tutorials,
  per-paradigm reference pages); the user's mental model and
  the docs structure match. The shape scales: a fourth or
  fifth paradigm (semantic-3D, tracking, etc.) drops into a
  new submodule without disturbing the existing three.
  Migration cost is one search-and-replace per symbol, done
  before any external user has built infrastructure on the
  flat root.
- **Negative.** Existing 0.0.x integration tests, examples,
  and notebooks all need updates. The pycocotools migration
  shim sits asymmetrically at the root rather than in
  `vernier.instance`; the asymmetry is documented but real.
  The "shared types at the top level" choice (D1) means
  `vernier.ParityMode` lives next to `vernier.COCOeval`,
  which is not the cleanest separation — a future migration
  to `vernier.types` is plausible. `vernier.instance` is
  longer to type than `vernier`; ergonomics tradeoff is
  small but noticeable in interactive use.
- **Neutral.** The Rust side is unchanged. The FFI is
  unchanged. The CLI is unchanged. No new dependencies.
  ADR-0011's `IouKind` discriminated union is unchanged; it
  just lives at `vernier.instance.IouKind` instead of
  `vernier.IouKind`.

## Pros and cons of the options

### A. Top-level structure

- **A1 flat root with suffixes.** 👍 minimal change. 👎
  doesn't scale past three paradigms; flat root accumulates
  every type in the project.
- **A2 per-paradigm submodules (chosen).** 👍 scales; matches
  community standard; mirrors documentation structure. 👎
  breaking change for 0.0.x callers.
- **A3 submodules + flat aliases.** 👍 both paths work. 👎
  documentation has to teach both; `vernier.PanopticEvaluator
  is vernier.panoptic.Evaluator` is the kind of detail that
  surfaces in confused stack traces; either path is the
  "wrong" one for some users.

### B. Migration shim

- **B1 no shim (chosen).** 👍 forces clean migration; no
  deprecation surface to maintain. 👎 hard break for early
  adopters. Mitigated by the small flat-root population at
  0.0.x and a sed script in the release notes.
- **B2 DeprecationWarning re-exports.** 👍 graceful
  migration. 👎 every old name has to be re-exported with a
  warning shim; cleanup itself is a future breaking change.
- **B3 opt-in `vernier.compat.flat_root`.** 👍 grace period
  for users who need one. 👎 zero usage signal; we don't know
  if anyone needs it. Worth doing reactively if demand
  appears post-0.1.0.

### C. Evaluator shape

- **C1 single relocated Evaluator (chosen).** 👍 zero
  semantic change. 👎 none meaningful.
- **C2 split per IoU kind.** 👍 each evaluator class has
  one job. 👎 `IouKind` already discriminates them; this
  duplicates the discrimination at the type-name level for
  no gain.

### D. Shared types

- **D1 top-level (chosen).** 👍 ergonomic; one import for
  the cross-cutting types. 👎 asymmetric (`COCOeval` and
  `ParityMode` are siblings at the root for different
  reasons).
- **D2 `vernier.types` submodule.** 👍 cleaner separation.
  👎 extra typing; doesn't scale to less-than-10 shared
  types.
- **D3 duplicated per paradigm.** 👍 no asymmetry. 👎
  three `ParityMode`s that are the same enum is confusing
  on its own; impossible to type "parity mode of any
  paradigm" without a Union.

## Links and references

- ADR-0001 — Record architecture decisions (this ADR
  triggers §"Affect the public API" and §"Set a project-wide
  convention").
- ADR-0007 — `patch_pycocotools` policy. The drop-in claim
  is the reason `COCOeval` and `patch_pycocotools` stay at
  the root rather than relocating to `vernier.instance`.
- ADR-0009 — `vernier-mask` as a pure-Rust leaf crate.
  Unaffected; the Rust side carries no submodule layout.
- ADR-0011 — Discriminated kernel config. Unaffected;
  `IouKind` keeps its shape, just lives at
  `vernier.instance.IouKind`.
- ADR-0015 — `vernier-cli`. Unaffected by Python-surface
  reorganization.
- ADR-0019 — Result tables. `EvalResult` follows
  `Evaluator` into `vernier.instance`.
- ADR-0025 — Panoptic-quality evaluation. The "negative
  consequences" §"three evaluator classes" is the trigger
  for this ADR.
- ADR-0026 — LVIS federated evaluation. `Frequency`,
  `CategoryFilter`, `summarize.lvis_default()` move into
  `vernier.instance` and `vernier.summarize` respectively.
- ADR-0028 — Semantic evaluation. Explicitly defers the
  namespace question to this follow-up; B4 in that ADR's
  considered options.
- scikit-learn, scipy, statsmodels, torch, jax — community
  precedent for `library.submodule.X` namespacing.
