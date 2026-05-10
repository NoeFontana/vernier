# Migrating from `pycocotools` to vernier

vernier reproduces `pycocotools==2.0.11`'s evaluation semantics
bit-for-bit in strict parity mode. ADR-0002 (parity model) and
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

- [ADR-0002](../adr/0002-three-tier-parity-model.md) — strict / aligned /
  corrected parity tiers.
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
