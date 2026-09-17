# ADR-0055: Finish the `COCOeval` drop-in's mutable surface, and publish the COCO-JSON normalizers

- **Status:** proposed
- **Date:** 2026-09-17
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

ADR-0007 made `vernier.COCOeval` the migration path: "downstream eval
code that imports `pycocotools.cocoeval.COCOeval` runs unchanged once
the symbol is swapped." The README states the same promise as "change
one import, or none."

TorchMetrics' `MeanAveragePrecision` is the widest real consumer of
that surface, and it is the one the promise fails against. Its COCO
backend (`torchmetrics/detection/helpers.py`) does four things the
drop-in did not support:

1. Rebinds `params.iouThrs` / `recThrs` / `maxDets` on every
   `compute()`. Fixed by the preceding commit on this branch.
2. Reads `coco_eval.ious` whenever `extended_summary=True`. The shim
   had no `ious` attribute at all — `AttributeError`.
3. Assigns `params.catIds = [class_id]` and re-runs the whole
   evaluate/accumulate/summarize cycle once per class when
   `class_metrics=True`, reusing a single evaluator. The shim raised
   `NotImplementedError("catIds subsetting")`, which made
   `class_metrics=True` dead.
4. Assembles `cocoGt` / `cocoDt` by assigning `dataset` and calling
   `createIndex()`, never `loadRes`. Those datasets omit image sizes
   `COCOeval` never reads and carry `bytes` RLE counts, both of which
   the shim now normalizes.

Item 4's normalizers are not shim-specific. RF-DETR's vernier backend
(`src/rfdetr/training/coco_map.py`) drives the array grid directly,
bypasses the shim, and therefore re-implements all three of them by
hand — the image-size fill, the `bytes`-counts JSON hook and the
NumPy-scalar hook — against the same TorchMetrics-shaped input. The
size fill in particular encodes a non-obvious rule (fill only where
`annToRLE` would not have been called) that took a careful reading of
`_prepare` to get right, and it is now written twice, in two repos,
with two slightly different answers.

A fifth item is *not* in scope and is recorded here so it is not
mistaken for an oversight: the shim still rejects `params.imgIds`
subsetting. No consumer we know of mutates it, and unlike `catIds` it
changes the `I` axis that `accumulate` normalizes over.

## Decision drivers

- ADR-0007 §"Decision outcome" — the drop-in exists so downstream code
  runs unchanged. A surface that raises on the second-most-common
  consumer's default configuration does not meet that bar.
- ADR-0001 §"Affect the public API" — every Python entry point is
  public surface, so publishing the normalizers is an ADR-level move
  even though the code already exists.
- ADR-0002 — whatever is published must carry a parity disposition.
  `ious` is `strict`: it mirrors `computeIoU` / `computeOks` output
  including quirk **F5**'s bare `[]`.
- Cost of the unused path. The opt-in `retain_meta` work exists
  precisely so a caller that never reads `evalImgs` does not pay for
  the per-cell bookkeeping. `ious` retention is strictly more
  expensive (an `O(G x D)` matrix per cell), so it cannot be
  unconditional either.
- Do not borrow API names from competing tools.

## Considered options

### For `catIds` subsetting

1. **Filter at the shim boundary** — restrict `categories` and
   `annotations` on both sides before serializing, which is what
   `COCOeval._prepare` itself does.
2. **A category filter in the Rust evaluator.** A new
   `EvaluateParams` field, threaded through every grid entry point.
3. **Slice the `K` axis of the accumulated tensors.** Evaluate once,
   summarize a sub-range.

### For `ious` and `evalImgs` cost

1. **Retain unconditionally.** What the shim did for `retain_meta`.
2. **Re-evaluate on first read, and cache.** Retention flags widen
   monotonically; the second read of either attribute is free.
3. **A constructor knob.** `COCOeval(..., retain_ious=True)`.

## Decision outcome

**`catIds`: option 1, filter at the shim boundary.** It is not an
approximation — `_prepare` loads exactly
`getAnnIds(imgIds=p.imgIds, catIds=p.catIds)`, so a subset `catIds`
evaluates a dataset that contains nothing else, which is what the
filter builds. Option 2 would put a pycocotools-shaped concern in the
core for one consumer; option 3 diverges the moment `useCats=0` is
set, because the collapse then pools a different set of annotations.
A requested category the dataset never declares is kept as an empty
category rather than dropped, because pycocotools evaluates it to a
row of `-1`s and a shorter `K` axis would silently renumber the rest.

**`ious` / `evalImgs`: option 2, lazy with a widening cache.**
`retain_meta` flips to `False` on the default cycle, and both
attributes re-run the per-image pass with what they need on first
access. A caller that reads both pays two extra passes over the
object's lifetime, not one per read. Option 3 was rejected because a
drop-in cannot grow constructor arguments the class it replaces does
not have — that is exactly the kind of divergence ADR-0007 set out to
avoid.

The FFI grows `EvalGrid.ious()`, returning
`{(image_id, category_id): ndarray}` in pycocotools' `(D, G)`
orientation (the core stores `(G, D)`), and only for pairs that have
both a detection and a ground truth. `evaluate_keypoints_grid` gains
the `retain_iou` flag the other three grids already had, so the OKS
matrix is reachable the same way.

### The published normalizers

`vernier.adapters` gains four functions, which is the canonical import
path for all of them:

```python
from vernier.adapters import (
    with_placeholder_image_sizes,  # bbox / keypoints: fill every gap with 0
    with_mask_image_sizes,         # segm / boundary: fill only where annToRLE would not have run
    coco_json_default,             # json.dumps hook: bytes RLE counts, NumPy scalars
    to_coco_json,                  # the dumps + encode one-liner
)
```

They live in a private `vernier._coco_json` module so `vernier._compat`
can import them without a cycle through the `adapters` package; the
re-export is the only public path, and nothing is exposed under two
names. `with_mask_image_sizes` takes the detection side as a
`{image_id: (height, width) | None}` mapping rather than a second COCO
dataset, so a caller that has detection RLEs but no `cocoDt` — the
array-grid case — can use it. The mapping's *key set* carries the
"a detection points at this image" bit the fill rule turns on, and
`None` marks an image the detection side cannot size either, which is
where pycocotools raises `KeyError` and vernier's schema error has to
stand rather than a `0x0` fill scoring silently.

The names describe what the function does to the dataset. None of them
is borrowed from `pycocotools`, `faster-coco-eval` or `hotcoco`.

### What this deliberately does not do

- **`params.imgIds` subsetting** stays rejected, loudly.
- **`params.areaRng` mutation** stays rejected; ADR-0040 puts custom
  area ranges on `vernier.instance.Evaluator`.
- **No `vernier[torchmetrics]` extra, no TorchMetrics import.** The
  library is a consumer we satisfy, not a dependency we take. It
  appears only in the `real-models` test extra, driving
  `tests/python/test_compat_torchmetrics.py`, which runs
  `MeanAveragePrecision` patched and unpatched and requires the two
  result dictionaries to be equal.

## Consequences

- **Positive.** `MeanAveragePrecision(class_metrics=True,
  extended_summary=True)` runs under `patched_pycocotools` and agrees
  with pycocotools element for element. The default
  evaluate/accumulate/summarize cycle stops allocating per-cell
  metadata nothing reads. RF-DETR can delete its hand-copied
  normalizers and take the maintained ones.
- **Negative.** Two public helpers whose shape is dictated by
  TorchMetrics' dataset layout. They are documented as what they are —
  conversions for a `pycocotools`-shaped dictionary — and their
  contract is `_prepare`'s behaviour, which is pinned.
- **Negative.** Reading `evalImgs` or `ious` now costs a second
  evaluation pass. Documented on both properties.
- **Neutral.** Exposing `ious` makes two pre-existing divergences
  observable for the first time. Neither is introduced here and
  neither changes a score on the datasets under test:
  - On arbitrary decimal coordinates, a retained IoU can sit one ULP
    off pycocotools'. The kernel arithmetic is bit-identical (the
    array-ingest path reproduces `bbIou` exactly on the same boxes);
    the drift is `serde_json`'s number parser, the same root cause as
    the `dtScores` drift, closed by ADR-0054's `float_roundtrip`.
  - Under `useCats=0`, pycocotools concatenates a cell's ground truths
    category by category while vernier keeps them in annotation order,
    so the collapsed matrix agrees up to a permutation of its `G`
    axis. `evalImgs[...]["gtIds"]` already carried this.
