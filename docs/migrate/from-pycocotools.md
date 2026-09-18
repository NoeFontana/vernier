# Migrating from `pycocotools` to vernier

vernier reproduces `pycocotools==2.0.11`'s evaluation semantics
bit-for-bit in strict parity mode, on every architecture it ships
(see [Bit-for-bit, and against which build](#bit-for-bit-and-against-which-build)
for the one scoping caveat). ADR-0002 (parity model) and
ADR-0007 (drop-in policy) are the design records; this guide is the
user-facing migration path. Audience: anyone moving an existing
`COCOeval`-based evaluation pipeline onto vernier.

## TL;DR — what to change

The public surface lives under `vernier.instance` (per ADR-0029),
plus the pycocotools-shaped shim re-exported at the root:

```python
from vernier import COCOeval, patch_pycocotools          # shim path
from vernier.instance import Bbox, CocoDataset, Evaluator    # native path
```

| `pycocotools` | vernier (shim) | vernier (native) |
|---|---|---|
| `from pycocotools.cocoeval import COCOeval` | `from vernier import COCOeval` | `from vernier.instance import Evaluator` |
| `cocoEval = COCOeval(coco_gt, coco_dt, iouType="bbox")` | same call shape, vernier subclass | `evaluator = Evaluator(iou=Bbox(), parity_mode="strict")` |
| `cocoEval.evaluate(); cocoEval.accumulate(); cocoEval.summarize()` | same three calls | `summary = evaluator.evaluate(dataset, dt_bytes)` |
| `cocoEval.stats` (12-entry numpy array) | same `.stats` array | `summary.stats` (12-entry `list[float]`, same order) |
| `print(cocoEval)` (the `summarize()` stdout) | same stdout in strict mode | `for line in summary.pretty_lines(): print(line)` |

The `COCOeval` shim is a drop-in: existing pycocotools-based code
runs unchanged once the symbol is swapped. The native `Evaluator`
surface is the ergonomic path forward — it returns a typed `Summary`
instead of mutating instance attributes, and it exposes the per-image
/ per-class / per-detection / per-pair tables documented in
[`how-to/result-tables.md`](../how-to/result-tables.md).

## Drop-in via `patch_pycocotools`

Existing scripts that already `import pycocotools.cocoeval` need not
edit the import. `patch_pycocotools` swaps the class in-place:

```python
from vernier import patch_pycocotools

unpatch = patch_pycocotools(parity_mode="strict")
try:
    # Existing pycocotools code runs unchanged; COCOeval is now vernier's.
    from pycocotools.cocoeval import COCOeval
    cocoEval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    cocoEval.evaluate(); cocoEval.accumulate(); cocoEval.summarize()
finally:
    unpatch()
```

The context-manager form is `patched_pycocotools()` and nests
correctly. `patch_pycocotools` defaults to `parity_mode="strict"`
because migration intent is bit-exactness with pycocotools; the
native `Evaluator` constructor defaults to `parity_mode="corrected"`
because new code does not need pycocotools' historical quirks.
ADR-0002 documents the two dispositions (`strict` / `corrected`);
ADR-0007 §"Behavior" pins the helper's default. The patch raises
`ImportError` if pycocotools is not installed, rather than silently
no-oping.

## Pytest integration

To run an unmodified pycocotools-based test suite (mmdetection,
ultralytics, detectron2, etc.) under vernier, drop a five-line
`conftest.py` at the test root:

```python
# conftest.py
import pytest
from vernier import patch_pycocotools

@pytest.fixture(autouse=True, scope="session")
def _vernier_strict():
    unpatch = patch_pycocotools(parity_mode="strict")
    yield
    unpatch()
```

`autouse=True` and `scope="session"` are the only non-obvious bits.
Session scope guarantees the patch fires once, before any test
module is collected and imported — which is the window pycocotools-
shaped imports need (see Troubleshooting below). Autouse means no
test has to opt in by parameter.

The same pattern translates to any framework with setup / teardown
hooks: `unittest.TestCase.setUpClass` / `tearDownClass`, a
`nose`-style module-level fixture, or a script-level `try` /
`finally` around the run. There is no `vernier[pytest]` extra and
no plugin entry point on purpose — the shim is the whole product;
making it more invisible would just slow the migration to the
native `Evaluator` API.

## Troubleshooting: my patch had no effect

Symptom: you called `patch_pycocotools()` (or installed the
`conftest.py` fixture above), the call ran without error, but the
numbers your suite produces are byte-identical to a run *without*
vernier.

Cause: a module that imports `pycocotools.cocoeval` was loaded
*before* `patch_pycocotools` fired. Python's `from … import name`
binds `name` to whatever object the source module exposes at that
moment — patching `sys.modules["pycocotools.cocoeval"].COCOeval`
later does not retroactively rewrite already-bound names. The
patch is live for any *subsequent* `from pycocotools.cocoeval import
COCOeval`, but the downstream test module captured the original
class at its own import time and that binding wins.

The rule: `patch_pycocotools()` must run before any module that
imports `pycocotools.cocoeval`. In practice that means one of:

- A session-scoped `conftest.py` fixture (above) — pytest collects
  and runs `conftest.py` before importing test modules.
- A direct call at the top of a top-level script, before the first
  `import` of a module that pulls in pycocotools.
- The context-manager form (`patched_pycocotools()`) wrapping the
  invocation that triggers the eval-using imports.

The patch is intentionally not an import side-effect of `import
vernier` (ADR-0007 §"Discoverability"): a silent rewrite would make
unexpected score differences untraceable. The cost of that policy is
that ordering is the user's responsibility, hence this section.

## `params`: what the shim honors, and what it rejects

The shim mirrors `pycocotools.cocoeval.Params` as a plain mutable
namespace, so downstream code that pokes at `cocoEval.params` keeps
working. Where vernier cannot reproduce a mutation it raises
`NotImplementedError` from `evaluate()` rather than ignoring it — a
divergence you can see beats one you cannot.

| `params` field | Mutable? | Notes |
|---|---|---|
| `iouThrs` | yes | Any ladder. Evaluated as given, including a ladder round-tripped through float32 (what `torch.linspace(...).tolist()` yields). |
| `recThrs` | yes | Any ladder. |
| `maxDets` | yes | Sorted ascending at `accumulate()` time, as pycocotools does (quirk **A2**). The largest entry caps the per-cell detections. |
| `catIds` | yes | Subsetting supported. Mirrors `_prepare`: only annotations in the selected categories are evaluated. A category the dataset never declares evaluates to a row of `-1`s, as upstream. |
| `useCats` | yes | `0` runs the collapsed single-bucket fold (quirk **L4**). |
| `kpt_oks_sigmas` | yes | `iouType="keypoints"` only; applied to every category (quirk **F1**). |
| `imgIds` | **no** | Subsetting raises. Evaluate the full dataset, or slice the result with [`vernier.aggregate`](../how-to/per-class-by-slice.md). |
| `areaRng` | **no** | Raises. Custom area ranges live on `vernier.instance.Evaluator(area_ranges=...)` per ADR-0040. |
| `useSegm` | **no** | Any non-`None` assignment raises. pycocotools deprecated it years ago but still honors it, silently overriding `iouType`; vernier drops the honor path (quirk **L3**). Pass `iouType=` instead. |

Assigning a *snake_case* `Evaluator` field name (`iou_thresholds`,
`recall_thresholds`, `area_ranges`) on `params` raises `AttributeError`
immediately, with a pointer to the native surface. The camelCase
pycocotools names above still mutate normally.

### Reading results back off the instance

`evaluate()` populates the grid; `accumulate()` populates `eval`;
`summarize()` populates `stats`. Two further attributes exist and are
built the first time they are read, because they cost a second
evaluation pass and most callers never touch them:

- **`evalImgs`** — the flat `[k][a][i]` list of per-image dicts
  (`dtIds`, `gtIds`, `dtMatches`, `gtMatches`, …).
- **`ious`** — `{(imgId, catId): matrix}`, one entry per pair in
  `imgIds x catIds`, with `catId == -1` under `useCats=0`. Each matrix
  is `(detections, ground truths)` with detections score-descending and
  truncated to `max(maxDets)`. A pair with nothing on one side is a
  bare `[]`, not an empty array — quirk **F5**, and what
  `maskUtils.iou` returns.

Both are also assignable, so code that overwrites them keeps working.
Two caveats on `ious`, neither of which moves a score:

- On coordinates that are arbitrary decimals a retained IoU can land
  one ULP off pycocotools'. The kernel arithmetic is bit-identical;
  the drift is in vernier's JSON number parser and is closed by
  ADR-0054.
- Under `useCats=0` the matrix agrees up to a permutation of its
  ground-truth axis: pycocotools concatenates a cell's ground truths
  category by category, vernier keeps them in annotation order.

### `COCOeval` on an in-memory `COCO`

TorchMetrics and similar callers build `cocoGt` / `cocoDt` by assigning
`coco.dataset` and calling `createIndex()`, never `loadRes`. The shim
handles that the way `COCOeval` does — detection `area` is read off the
object rather than derived (quirk **J3**), `bytes` RLE counts are
accepted (quirk **K3**), and a missing image `width` / `height` is
filled wherever `annToRLE` would never have looked at it.

If you assemble such a dataset yourself and drive a vernier grid
*directly* rather than through the shim, the same conversions are
published:

```python
from vernier.adapters import (
    detection_image_sizes,
    to_coco_json,
    with_mask_image_sizes,
    with_placeholder_image_sizes,
)

# bbox / keypoints: no kernel reads an image size, so fill every gap.
gt = with_placeholder_image_sizes(coco_gt.dataset)

# segm / boundary: fill only where `annToRLE` would never have looked,
# which needs the sizes the *detection* side knows.
gt = with_mask_image_sizes(coco_gt.dataset, detection_image_sizes(coco_dt.dataset))

gt_bytes = to_coco_json(gt)
dt_bytes = to_coco_json(coco_dt.dataset["annotations"])
```

`with_mask_image_sizes` takes a `{image_id: (height, width) | None}`
mapping rather than a second dataset, so a caller with detection RLEs
but no `cocoDt` object can build it from the first detection mask's
`size` per image instead of calling `detection_image_sizes`.

ADR-0055 is the record for both the `params` surface above and these
helpers.

## Worked example

Native `Evaluator` form, end-to-end:

```python
from pathlib import Path
from vernier.instance import Bbox, CocoDataset, Evaluator

gt_bytes = Path("instances_val2017.json").read_bytes()
dt_bytes = Path("detections.json").read_bytes()

dataset = CocoDataset.from_json(gt_bytes)
summary = Evaluator(iou=Bbox(), parity_mode="strict").evaluate(dataset, dt_bytes)

print(summary.stats[0])              # AP, e.g. 0.347
for line in summary.pretty_lines():  # the pycocotools-shaped 12-line block
    print(line)
```

The 12-entry `summary.stats` vector matches pycocotools'
`cocoEval.stats` position-for-position (`AP, AP50, AP75, APs, APm,
APl, AR1, AR10, AR100, ARs, ARm, ARl`). Switch to `Segm()` for
instance-mask IoU, `Boundary()` for boundary IoU (ADR-0010), or
`Keypoints()` for OKS (ADR-0012).

## Sentinels: empty buckets are `-1.0`

`pycocotools` initializes `precision`, `recall`, and `scores` to
`-1` and filters with `s[s>-1]` before averaging (quirk **C5** in
[`docs/engineering/pycocotools-quirks.md`](../engineering/pycocotools-quirks.md)).
A category with no GTs disappears from the average — it is not
counted as zero. vernier reproduces the sentinel byte-for-byte in
strict mode, so `summary.stats[i] == -1.0` for empty buckets.

If you cross-compare with LVIS (also `-1.0`, quirk **AF6**) or
panoptic (vernier returns `0.0` for the corrected `EmptyCategory`
case, quirk **W6**), the parallel sentinel table in
[`from-lvis-api.md`](from-lvis-api.md#sentinels-1-vs-0-vs-nan) is
the cross-codebase reference.

## What does NOT carry over

- **Per-image AP.** `pycocotools-cli` and `faster-coco-eval` both
  expose a per-image AP value; vernier does not, by design. PR
  curves from a single image are degenerate. See
  [`why-no-per-image-ap.md`](../explanation/why-no-per-image-ap.md)
  for the rationale and the polars recipe to reconstruct it from
  raw counts when genuinely needed.
- **`useCats=False` cross-class matching.** vernier's `Evaluator`
  takes the per-category fold as the contract. The pycocotools
  `useCats=0` path runs in the shim under `parity_mode="strict"`
  for migration, but the native surface does not surface that
  switch — cross-class confusion analysis lands in `per_pair`
  (ADR-0019) on a separate roadmap.
- **In-place mutation of `COCOeval` instance attributes.** The
  native `Evaluator.evaluate(...)` returns a typed `Summary` instead
  of mutating `cocoEval.eval` / `cocoEval.evalImgs` /
  `cocoEval.stats`. Code that reaches into those attributes should
  migrate to `summary.stats` (the 12-entry vector) and the per-image /
  per-class / per-detection / per-pair tables surfaced via
  `Evaluator(...).evaluate(gt, dt, tables="all")` (ADR-0019). The
  `evalImgs` flat-cube layout is not exposed on the native surface;
  the shim under `parity_mode="strict"` still produces it.

## Pinned pycocotools version

vernier's strict-mode parity is keyed to `pycocotools==2.0.11`
(pinned exactly in `pyproject.toml`). Bumping that pin is an
ADR-level decision per CLAUDE.md §"Parity contract" — every quirk
vernier reproduces is keyed to this version, and the parity harness
double-runs reference and candidate at exactly this SHA.

### Bit-for-bit, and against which build

"Bit-for-bit" is measured against `pycocotools==2.0.11` **as published
on PyPI** — the wheel `pyproject.toml` pins and the parity harness
installs. That is the reference on x86-64 and on aarch64 alike, and
vernier matches it on both.

The scoping matters for one narrow case. pycocotools' IoU kernel is C,
and a C compiler is allowed to fuse a multiply and an add into a single
rounding step. The published builds do not (checked by disassembly on
five arm64 binaries, covering the Linux glibc and musl wheels, the
macOS universal2 arm64 slice and the Windows arm64 build), but a
pycocotools you compile yourself —
`pip install --no-binary pycocotools`, a distro package, a conda-forge
build — goes through your compiler, not the wheel builder's. On ARM,
where the fused instruction is always available, such a build can round
the union differently from the published one.

If that happens, expect vernier and your locally built pycocotools to
disagree by 1-2 units in the last place of the IoU, on roughly one box
pair in a million. It is far below any reporting precision and it is
not a divergence vernier introduces: vernier's output is byte-identical
on x86-64 and ARM by design, because a result that depended on the CPU
that produced it could not be cached, shipped between machines, or
combined across a heterogeneous cluster. Install the published wheel
and the question does not arise.

Design record: [ADR-0056](../adr/0056-pin-no-fp-contraction-for-bbox-iou.md);
quirk **I7** in the [quirks survey](../engineering/pycocotools-quirks.md).

## Whole-dataset parity smoke

The `tests/python/parity/test_parity.py` suite double-runs vernier
and pycocotools on a fixture corpus and diffs every intermediate
(`evalImgs`, `eval`, `stats`). The COCO val2017 smoke at
`tests/python/parity/test_coco_val.py` is env-gated and pins the
12-entry summary bit-equal against `COCOeval` on the full dataset.
Run with:

```sh
just test-parity                                              # the fast fixture suite
VERNIER_COCO_GT_PATH=... VERNIER_COCO_DT_PATH=... just test-coco-val  # the val2017 smoke
```

The val2017 GT and a public-detector predictions JSON are downloaded
under the COCO terms of use and never committed to the repo;
`tools/fetch-coco-val.sh` is the canonical setup helper.

## See also

- [ADR-0002](../adr/0002-three-tier-parity-model.md) — strict / corrected
  parity tiers (the `aligned` tier was folded into `strict` by the
  2026-05-10 amendment; the filename keeps the original title).
- [ADR-0007](../adr/0007-patch-pycocotools-policy.md) — why
  `patch_pycocotools` (verb names the mechanism), not
  `init_as_pycocotools` (faster-coco-eval's borrowed shape).
- [`docs/engineering/pycocotools-quirks.md`](../engineering/pycocotools-quirks.md)
  — the disposition table for every pycocotools quirk vernier had
  to reckon with. Cite quirks by ID (e.g. **C5**, **B1**, **D1**)
  in issues and PRs.
- [Migrating from `faster-coco-eval`](from-faster-coco-eval.md) —
  if your starting point is faster-coco-eval rather than vanilla
  pycocotools.
