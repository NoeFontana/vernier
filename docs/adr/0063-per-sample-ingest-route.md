# ADR-0063: An ingest route for training loops

- **Status:** proposed
- **Date:** 2026-09-21
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

Integrating vernier into a training loop currently costs the caller a
conversion layer. The reference case is roboflow/rf-detr, which took
roughly 250 lines in `src/rfdetr/training/coco_map.py` that do nothing
but turn TorchMetrics' per-image metric state into the inputs ADR-0057
and ADR-0060 defined.

Nothing in those 250 lines is specific to that project. Every trainer that
accumulates predictions and targets per image and evaluates at
epoch end has to write the same code, and the rules it encodes are
mostly silent when violated rather than loud:

- an `images` entry is needed for *every* image, including ones with no
  annotations;
- annotation ids must start at 1 — COCOeval's results are wrong from 0;
- `area` falls back **per element** to the mask's area when the
  annotation has a mask and to the box's only when it does not —
  independent of the IoU type, because a COCO ground truth has one
  `area` per annotation and it is the segmentation's;
- `iscrowd` must be widened to `int64`: vernier reads any non-zero value
  as a crowd, so a `uint8` column wraps 256 to 0 and quietly evaluates a
  crowd annotation as a normal one;
- image sizes resolve the way `with_mask_image_sizes` (ADR-0055)
  resolves them — the image's own first mask, else the size its
  detections carry, else the `0x0` nothing reads;
- TorchMetrics' `_fix_empty_tensors` reshapes an empty per-image box
  tensor to `(1, 0)` rather than `(0, 4)` (it is avoiding a DDP
  all-reduce hang), which breaks both concatenation and per-image
  counting unless the caller normalizes it;
- the fastest detection route differs by IoU type: the `(N, 7)` matrix
  carries bbox state with no Python object per detection, while `segm`
  needs the columnar route because it is the only one that carries a
  mask *and* keeps the arrays whole.

A user who gets the `iscrowd` width wrong does not get an exception.
They get slightly wrong AP, on the annotations that matter most.

vernier already has every piece this conversion targets — columnar
ground truth (ADR-0060), three detection routes (ADR-0030, ADR-0057),
and DLPack ingest on every array argument, which is what lets a torch
CPU tensor cross the boundary with no `import torch` anywhere in the
package. What is missing is the route that lands framework-shaped
per-image records on them.

### Out of scope

- Any import of torch, torchmetrics, lightning, jax or tensorflow,
  **including behind an extra**. ADR-0031 settled this ("no torch
  dependency, ever") and ADR-0055 restated it for TorchMetrics
  specifically ("a consumer we satisfy, not a dependency we take").
  This ADR does not reopen it; `tests/python/test_no_framework_imports.py`
  makes it a CI gate instead of prose.
- A stateful `update()` / `compute()` accumulator. See
  §"Considered options", option 4.
- The pycocotools-shaped `COCO` and `mask` surfaces. Separable, and
  argued in their own ADR.
- Callbacks, loggers, EMA and checkpoint metrics. `docs/comparison.md`
  §"What vernier doesn't do (yet)" calls this downstream-framework
  territory and that stays true.

## Decision drivers

- **ADR-0001 §"Affect the public API"** — every Python entry point is
  public surface, so this is an ADR-level change even though the
  underlying routes already exist.
- **ADR-0030 §"Extend, do not fork"** — "Two ingest paths producing
  identical results is the maintenance ceiling. Three would not be."
  Whatever this adds must reduce to the existing representations before
  any downstream code runs, or it becomes a second place for parity to
  drift.
- **ADR-0057 §"Validation must refuse, never repair"** — a route that
  guesses is worse than one that rejects.
- **Framework-agnostic by construction, not by discipline.** The route
  must work for TorchMetrics, Lightning, Ultralytics, MMDetection,
  Detectron2 and a hand-written loop without naming any of them.
  Duck-typing on DLPack is how that is achieved; `isinstance` against a
  framework type is how it would be lost.
- **The rules above are vernier's knowledge, not the caller's.** Each
  one exists because of a pycocotools quirk vernier already
  dispositions. Leaving them in a docstring means every consumer
  reimplements them and some get them wrong.

## Considered options

1. **Status quo** — document the conversion in a how-to page and let
   each trainer write it.
2. **A per-sample ingest route returning metrics** — one function taking
   per-image records and returning the twelve COCO stats.
3. **An ingest route returning vernier's inputs** — functions taking
   the caller's state and returning `(CocoDataset, DetectionsInput)`,
   with a thin metric wrapper over them.
4. **A stateful `update()` / `compute()` accumulator** — mirror the
   metric protocol trainers already speak.
5. **A `vernier.adapters.torchmetrics` module** that imports
   torchmetrics and subclasses `CocoBackend`.

## Decision outcome

Chosen option: **option 3 — a per-sample ingest route returning
vernier's inputs**, with `coco_metrics` as a convenience wrapper over
it.

Option 3 is the honest description of the change: it is a fourth way to
construct the inputs the other three routes construct, which is a route,
not a subsystem. It also composes. Returning the
`(CocoDataset, DetectionsInput)` pair means TIDE, LRP, result tables,
calibration, custom grids (ADR-0040) and the partitioned/DDP path all
work against per-image records for free, whereas option 2 would serve AP
and leave every other surface needing its own adapter.

### Surface

Published through `vernier.adapters` — the canonical path ADR-0055
established for exactly this kind of helper — with the input shapes as
`TypedDict`s in `vernier._array_types` beside the existing `Detections`.

**Two spellings, one conversion.** `coco_inputs` takes per-image
records; `coco_inputs_from_columns` takes the same state already
concatenated, plus a `counts` column carrying the image structure. Both
land on one internal builder, so they cannot drift — a test pins the two
to the identical `dataset_hash`.

The second spelling is not sugar. A caller whose state is already
columnar — which a TorchMetrics-shaped metric's is — would otherwise
split it into per-image records only to have vernier concatenate it
again, paying a Python-level pass per image per field. Measured on a
COCO-val-shaped run (5000 images x 300 detections), the per-record
spelling costs 0.414 s of conversion against 0.051 s for the columnar
one, and 765k Python calls against 61k. That is the difference between
this route being a regression against hand-rolled glue and being
slightly faster than it.

```python
def coco_inputs(
    predictions: Sequence[Prediction],
    targets: Sequence[Target],
    *,
    iou_type: Literal["bbox", "segm"] = "bbox",
    box_format: Literal["xywh", "xyxy", "cxcywh"] = "xywh",
    categories: Sequence[int] | Sequence[GtCategory] | None = None,
    area: Literal["auto", "supplied", "box", "mask"] = "auto",
    cast_inputs: bool = True,
) -> tuple[CocoDataset, DetectionsInput]: ...

def coco_metrics(predictions, targets, *, ...) -> dict[str, float | NDArray[np.float64]]: ...

def gt_image_sizes(gt_rles, dt_rles=None, supplied=None) -> tuple[NDArray[np.int64], ...]: ...

def coco_inputs_from_columns(
    detections: DetectionColumns,
    targets: TargetColumns,
    *,
    iou_type=..., box_format=..., categories=..., area=..., image_ids=None, cast_inputs=True,
) -> tuple[CocoDataset, DetectionsInput]: ...
```

`Prediction` and `Target` accept **both** `masks` (a bitmask array) and
`rles` (pre-encoded), because the two populations differ: TorchMetrics'
state at `compute()` time is already RLE, while a plain loop,
Ultralytics and Detectron2 hold bitmasks. Every value is duck-typed —
anything exporting `__dlpack__` or convertible by `np.asarray`.

**`coco_inputs` returns the pair rather than splitting into two
functions**, because three of its decisions cannot be made from one side
alone:

- a class that appears only in predictions must still become a category,
  so `categories=None` resolves to the union of both sides;
- image sizes are `with_mask_image_sizes(gt, detection_image_sizes(dt))`,
  which reads both;
- the detection route choice depends on the IoU type the ground truth
  was built for.

`gt_image_sizes` is the columnar mirror of the already-published
`detection_image_sizes`. ADR-0055 explicitly left this to the caller —
"The `cocoDt`-less array-grid caller still builds the mapping itself" —
and that is the single rule rf-detr most visibly reimplemented.

### What is a parameter, and what is not

The rules listed in §"Context" are **not** parameters. Every one of them
has exactly one correct answer, and exposing a knob would mean shipping
a way to be silently wrong. `area` is the sole exception and it is a
parameter only because `"supplied"` is a legitimate choice for a caller
whose areas are authoritative; `"auto"` is the per-element fallback that
matches what COCOeval does.

**`"auto"` reads the mask under `bbox` too.** `COCOeval` with
`iouType="bbox"` buckets by the annotation's `area` field, which in a
COCO file is the segmentation's area — it never recomputes `w * h`. So
a mask, when the record carries one, decides the area whatever IoU type
is being evaluated. Deriving the box area per pass instead is invisible
until an object's two areas straddle `32**2` or `96**2`, and then it
moves AP between the small and medium buckets against every
pycocotools-shaped evaluator, with no error anywhere. This is why masks
are worth passing even for a bbox-only run: they are not evaluated, but
they are read.

`box_format` **must not auto-detect.** A `(N, 4)` array is
indistinguishable between `xywh` and `xyxy` in general, and a wrong
guess produces plausible, wrong AP rather than an error. The default is
`"xywh"` because that is vernier-native and COCO-native.

### `cast_inputs=True` on this route only

ADR-0004 pins `f64` at the boundary and ADR-0030 refuses `f32` rather
than silently promoting it, because a silent promotion surfaces later as
parity drift. That reasoning holds for `Detections`, where the caller
chose the array's dtype.

It does not hold here. Framework tensors are `float32` / `int32` by
construction — that is what a model emits — so requiring `f64` on this
route means every caller writes the `.double()` calls this route exists
to delete. The default is therefore `cast_inputs=True`, and the
one-shot `UserWarning` the columnar path emits is suppressed on this
route: it is correct advice for a caller who picked the dtype and pure
noise for one who received it from a forward pass.

This is a deliberate, recorded divergence, confined to one function.

### Return shape

`coco_inputs` returns vernier's own types. `coco_metrics` returns a
plain `dict` of Python floats and numpy arrays keyed by the conventional
COCO metric names (`map`, `map_50`, `map_75`, `map_small`, `map_medium`,
`map_large`, `mar_{max_dets[0..2]}`, `mar_small`, `mar_medium`,
`mar_large`, plus `*_per_class` under `class_metrics`).

vernier does **not** return another library's tensor type. The caller
writes `torch.as_tensor(...)`, which is one line and zero-copy off
numpy. That is the whole of what "plugged in" costs, and it is the
reason no framework import is needed.

### Consequences

- **Positive.** The rules in §"Context" are written once, in the project
  that owns the quirks they descend from, and tested against the parity
  oracle rather than rediscovered per trainer. Any framework with
  per-image predictions and targets — named or not, existing or not —
  reaches every vernier surface through one call. rf-detr's ~250 lines
  collapse to roughly 35.
- **Negative.** Four `TypedDict`s and four functions of new public
  surface that must be kept working. The bet is on the second consumer;
  the first one already had equivalent code, and adopting this is worth
  it to that consumer only because the columnar spelling makes it free. `cast_inputs=True` is also a second dtype policy in the
  codebase, and "which route silently casts?" is now a question a reader
  can ask.
- **Negative.** `Prediction`/`Target` accept two mask spellings, so the
  validation matrix for `segm` doubles.
- **Neutral.** The metric key names are a borrowed vocabulary — vernier's
  own names are `AP`, `AP@.50`, `AR_1` (`docs/reference/coco-summary-stats.md`).
  The borrow is confined to `coco_metrics` and is what makes its output
  directly loggable.

## Pros and cons of the options

### Option 1 — status quo

- 👍 Zero new public surface; zero maintenance.
- 👎 Leaves seven silent-failure rules to be reimplemented per consumer.
  One of them (`iscrowd` width) is wrong-by-default in numpy, since
  `uint8` is the natural dtype for a boolean-ish column.
- 👎 The cost is paid again by every integration, and the failures it
  invites are silent ones.

### Option 2 — route returning metrics

- 👍 Smallest possible call site for the AP case.
- 👎 Serves AP only. TIDE, LRP, tables, calibration and custom grids
  would each need their own per-sample entry point, or callers fall back
  to writing the conversion anyway.
- 👎 Makes the metric vocabulary the route's contract rather than a
  detail of one wrapper.

### Option 3 — route returning vernier's inputs (chosen)

- 👍 One conversion, every downstream surface.
- 👍 Reduces to the existing representations, satisfying ADR-0030
  §"Extend, do not fork"; nothing downstream can tell which route built
  its inputs.
- 👍 `coco_metrics` still gives the short call site option 2 wanted.
- 👎 Two-step for the common case (though the wrapper hides it).
- 👎 Two input spellings to document and keep in step, where one would
  be simpler to explain. The shared builder and its equivalence test are
  what make that safe; without them this would be the fork ADR-0030
  warned about.
- 👎 Exposes `CocoDataset` and `DetectionsInput` in a signature aimed at
  users who may not have met either.

### Option 4 — stateful accumulator

- 👍 Matches the `update()` / `compute()` protocol trainers already
  speak, so the call site is familiar.
- 👎 TorchMetrics owns its own state, including DDP synchronization. A
  second accumulator inside it would be redundant at best and divergent
  at worst.
- 👎 `BackgroundEvaluator` fixes ground truth at construction
  (ADR-0014), so it is not the substrate — a trainer has both sides
  streaming, and building on it would mean buffering targets anyway.
  Buffering is then all the accumulator does, which the caller's own
  lists already do.
- 👎 A new lifecycle (reset, merge across ranks, partial finalize) to
  specify and test, for a shape nobody has asked for yet.

Revisit if a trainer that does not already accumulate asks for it; the
route chosen here is the substrate it would be built on.

### Option 5 — `vernier.adapters.torchmetrics` importing torchmetrics

- 👍 Smallest possible call site for TorchMetrics users specifically.
- 👎 Contradicts ADR-0031 and ADR-0055 by taking the dependency both
  rule out.
- 👎 Couples a vernier release to torchmetrics' private metric-state
  attribute names, which are not API.
- 👎 Binds the work to one framework, when the conversion it automates
  is common to all of them.

## Links and references

- Related ADRs: [0004](0004-numerical-layout.md) (f64 boundary, diverged
  from here), [0007](0007-patch-pycocotools-policy.md),
  [0014](0014-background-evaluator.md),
  [0030](0030-buffer-protocol.md) (DLPack ingest; "extend, do not fork"),
  [0031](0031-dist-eval.md) ("no torch dependency, ever"),
  [0040](0040-user-parametrizable-instance-evaluation-grid.md),
  [0055](0055-drop-in-params-surface-and-coco-json-adapters.md)
  (`vernier.adapters` as the canonical path; the normalizers this
  extends), [0057](0057-python-detection-ingest-routes.md),
  [0060](0060-python-ground-truth-array-ingest.md)
- External: `rfdetr.training.coco_map` — the per-trainer conversion
  this route replaces, and the reference for the rules in §"Context".
