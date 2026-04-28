# ADR-0012: OKS keypoints public surface and quirk dispositions

- **Status:** proposed
- **Date:** 2026-04-28
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

## Context and problem statement

Phase 3 kicks off with COCO val2017 keypoints parity. The Rust core has a
`Similarity` trait per ADR-0005 and a discriminated kernel-config union per
ADR-0011, so the *shape* of a new kernel is largely settled — `Keypoints`
becomes another `IouKind` variant, `OksSimilarity` becomes another
`Similarity` impl. What's not settled, and what this ADR pins, is the
content of that variant: how object-keypoint similarity (OKS) parameters
are exposed to the user, and which pycocotools quirks the implementation
matches bit-for-bit versus corrects.

The thorny one is OKS sigmas. `pycocotools` hardcodes a 17-element table
keyed on the COCO person skeleton (`cocoeval.py:523`); there is no
mechanism to override per category in the official API. Downstream forks
(CrowdPose, MMPose) monkey-patch `Params.kpt_oks_sigmas` after instantiation
— which works because pycocotools' `Params` object is a mutable bag, but
which collapses if a user evaluates two categories with different
skeletons in one pass. Quirk **F1** in `docs/engineering/pycocotools-quirks.md`
flags this as `corrected` precisely because faithful reproduction would
foreclose multi-category keypoints (LVIS-Pose, CrowdPose+COCO-Person mixed
datasets, future robotics fixtures).

The remaining keypoint quirks (F2–F5, D2, D5, L8) are mostly already
dispositioned in the survey and have a forward home; this ADR ratifies
them in the same place as F1 so a single document tells the keypoints
story.

## Decision drivers

- **F1 disposition is `corrected` per the quirks survey.** Per-category
  sigmas must be expressible without monkey-patching.
- **No `Evaluator` parameter sprawl.** ADR-0011's discriminated union is
  the home for kernel-local params. `Keypoints` carries its own sigmas
  field; nothing else moves.
- **Default behavior matches pycocotools COCO-person.** A user who
  constructs `Keypoints()` (no arguments) gets the 17-sigma COCO person
  skeleton applied to every category — identical numeric output to
  `pycocotools` on a single-category-`person` evaluation.
- **`pyright --strict` exhaustiveness.** Adding `Keypoints` to `IouKind`
  must surface every existing `match self.iou:` site that lacks a
  `Keypoints()` arm.
- **Mirror the Rust `EvalIouType` enum.** The Rust side gains a
  `Keypoints { sigmas: ... }` variant; the FFI dispatch is a single
  `match` arm.
- **Pre-1.0 freedom.** No deprecation cycle is owed. `Keypoints` is
  net-new.

## Considered options

1. **Single global sigmas tuple** mirroring pycocotools'
   `Params.kpt_oks_sigmas`. `Keypoints(sigmas: tuple[float, ...] =
   COCO_PERSON_SIGMAS)`. Single skeleton, no per-category override.
2. **Per-category sigmas mapping keyed on category_id**, with the empty
   mapping meaning "use COCO-person for every category".
   `Keypoints(sigmas: Mapping[int, tuple[float, ...]] = {})`.
3. **Per-category sigmas via callable.** `Keypoints(sigmas: Callable[[int],
   tuple[float, ...]])`. Maximum flexibility; opaque to introspection.
4. **`Params`-style mutable bag.** Match pycocotools shape literally;
   mutate after construction. Rejected by ADR-0006's frozen `Evaluator`
   discipline.

## Decision outcome

Chosen option: **2 — per-category sigmas as a `Mapping[int, tuple[float,
...]]`**, keyed on `category_id`. The empty mapping is the default and
means "every category uses the COCO-person 17-sigma skeleton" — the same
numeric output as pycocotools on COCO-person. Categories present in the
mapping use their override; categories absent fall back to COCO-person.

Public Python surface:

```python
@dataclass(frozen=True, slots=True)
class Keypoints:
    """Object-keypoint similarity (OKS) kernel selector (ADR-0012).

    ``sigmas`` overrides the per-keypoint standard deviation used in the
    OKS formula on a per-category basis. The empty default means every
    category uses the 17-element COCO-person table; pycocotools-equivalent
    behavior on single-category-person datasets.
    """

    sigmas: Mapping[int, tuple[float, ...]] = field(default_factory=dict)
```

`Evaluator.evaluate` gains a `match` arm:

```python
case Keypoints(sigmas=sig):
    return evaluate_keypoints_summary(
        gt, dt, self.parity_mode, max_dets_list, self.use_cats, dict(sig)
    )
```

The Rust FFI gains `evaluate_keypoints_summary(..., sigmas: HashMap<i64,
Vec<f64>>)`. Keypoints in COCO are passed as the standard 17-triplet
`[x_i, y_i, v_i, ...]` flat array on each annotation; the `Annotation`
enum gains a `Keypoints { keypoints: Vec<f64>, num_keypoints: u32, area:
f64, bbox: Bbox }` variant. Dispatch on the GT enum at load time, not at
matching time — `OksSimilarity::Annotation = OksAnn`, same shape as the
existing kernels.

`max_dets` is **not** auto-switched when `iou=Keypoints`. The pycocotools
default for keypoints is `[20]`, vs `[1, 10, 100]` for bbox/segm; we
document `max_dets=(20,)` as the recommended override and let the user
set it explicitly. Auto-switching couples kernel choice to a "summary
config" field that is otherwise kernel-agnostic; that coupling has no
analogue elsewhere in the API and breaks the invariant that `iou` is the
only field whose change is a parity-contract change.

### Quirk dispositions (ratified)

| Quirk | Disposition | Implementation note |
| ----- | ----------- | ------------------- |
| **F1** | `corrected` | Per-category sigmas mapping; empty = COCO-person for all. Above. |
| **F2** | `aligned` | OKS area uses `gt.area + f64::EPSILON` (the divide-by-zero guard). |
| **F3** | `strict` | DT with no visible keypoints falls back to a 2× bbox surrogate metric, exactly per pycocotools. |
| **F4** | `strict` | OKS bbox-expansion `[bb.x - bb.w, bb.x + 2*bb.w]` asymmetric — preserved. |
| **F5** | `aligned` | `compute_oks` returns a correctly-shaped 0-row/col array when either side is empty (pycocotools returns `[]`; the downstream consumer treats both identically). |
| **D2** | `strict` | GT with 0 visible keypoints is implicit ignore — same OR'd into `_ignore` as pycocotools. |
| **D5** | not applicable | Default `areaRng` for keypoints in pycocotools drops the "small" bucket. In vernier, area ranges are a future `Breakdown` ADR's concern (see "Out of scope"); for COCO val2017 keypoints we hardcode the same `(0, 1024^2), (32^2, 96^2), (96^2, 1e10^2)` triple as bbox/segm and document that the keypoints summarizer iterates over the medium/large entries only when `iou=Keypoints`, matching pycocotools' summary tables. |
| **L8** | `aligned` | `Params.kpt_oks_sigmas` leaks across iou-types in pycocotools because `Params` is one mutable bag. Vernier's discriminated union per ADR-0011 forecloses the leak by construction — `sigmas` lives only on `Keypoints`. |

### Consequences

- **Positive:** F1 corrected disposition shipped without monkey-patching;
  multi-skeleton evaluations (CrowdPose + COCO-person, future robotics
  fixtures) Just Work.
- **Positive:** Default behavior is bit-identical to pycocotools on
  single-category COCO-person — the headline parity claim holds without
  the user having to know about sigmas.
- **Positive:** ADR-0011's `Evaluator` shape is preserved; no new fields.
- **Negative:** `Mapping[int, tuple[float, ...]]` is wordier than
  pycocotools' bare tuple. The cost buys multi-category support.
- **Negative:** The user must remember to set `max_dets=(20,)` for the
  pycocotools-canonical keypoints summary. We document this on the
  `Keypoints` docstring and the `patch_pycocotools` shim handles it
  transparently for migrating users (per ADR-0007).
- **Neutral:** F4's asymmetric bbox expansion is a `strict` reproduction
  of a pycocotools wart (extends `bb.w` left, `2*bb.w` right). We could
  promote it to `corrected` later without breaking the public surface;
  this ADR locks the default at `strict` for headline-parity continuity.

## Out of scope

- **Generalized `Breakdown` mechanism.** The Phase-3 plan calls for an
  ADR that replaces hardcoded small/medium/large area ranges with a
  generic `Breakdown { name, key_fn, ranges }`. CrowdPose's `crowdIndex`
  is the primary motivation; for COCO val2017 keypoints, the existing
  hardcoded ranges suffice. Deferred to a later ADR with CrowdPose as
  the trigger.
- **Pose-PCK / OKS-AP10K / non-COCO keypoint datasets.** Ratifies COCO
  keypoints only; non-COCO datasets are an `EvalDataset` extension and
  carry their own ADR if they perturb sigmas semantics.

## Pros and cons of the options

### Option 1 — single global sigmas tuple

- 👍 Smallest surface change; one tuple, no mapping.
- 👍 Trivial pycocotools-equivalence on COCO-person.
- 👎 Forecloses multi-category keypoints. F1 stays `strict` rather than
  `corrected`; downstream forks keep monkey-patching.
- 👎 Diverges from the survey's pre-existing F1 disposition.

### Option 2 — per-category mapping (chosen)

- 👍 F1 `corrected` shipped without monkey-patching.
- 👍 Default (empty mapping) is byte-identical to pycocotools on
  COCO-person.
- 👍 Mirrors the natural Rust `HashMap<i64, Vec<f64>>` shape; FFI is
  mechanical.
- 👎 Slightly wordier construction for the COCO-person common case
  (`Keypoints()` works, but multi-category users build a mapping).

### Option 3 — sigmas as a callable

- 👍 Maximum flexibility (e.g., interpolated sigmas, learned schedules).
- 👎 Opaque to introspection / serialization. Cannot be mirrored across
  the FFI without serializing the closure.
- 👎 Speculative complexity; no Phase-3 use case.

### Option 4 — pycocotools-style mutable bag

- 👍 Matches `Params.kpt_oks_sigmas` shape literally.
- 👎 Conflicts with ADR-0006's frozen `Evaluator` discipline.
- 👎 Recreates the L8 leak (kpt_oks_sigmas surviving an `iouType` flip).

## Links and references

- ADR-0001 §"Affect the public API" — change requires an ADR.
- ADR-0002 — three-tier parity model that frames F1 corrected.
- ADR-0005 — locks `Similarity` trait; `OksSimilarity` slots in.
- ADR-0006 — frozen `Evaluator` discipline.
- ADR-0007 — `patch_pycocotools` carries the `iouType="keypoints"`
  default-overrides on behalf of migrating users.
- ADR-0011 — discriminated kernel config; `Keypoints` is a new variant.
- `docs/engineering/pycocotools-quirks.md` — F1, F2, F3, F4, F5, D2, D5, L8.
- pycocotools 2.0.11 `cocoeval.py:215-235, 502-525` — the OKS reference
  implementation and `setKpParams`.
