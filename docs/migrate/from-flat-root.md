# Migrating from `vernier 0.0.x` flat-root imports

ADR-0029 restructures the public Python surface into per-paradigm
submodules. Pre-1.0, vernier ships breaking import changes inside
the 0.0.x line (per ADR-0011 precedent). This page is the migration
recipe for downstream code that imported from the flat root.

> **Pre-1.0 freedom note.** No flat-root re-exports, no
> `DeprecationWarning` shim. `from vernier import Evaluator` raises
> `ImportError` outright. The migration is mechanical: a
> search-and-replace per moved symbol.

## Sed recipe

The 38 in-tree importer files migrated via
`tools/migrate_imports.py` — the same script ships in this release
for external 0.0.x users:

```bash
python tools/migrate_imports.py --tree path/to/your/code
```

The script rewrites:

- `from vernier import X` lines (split per-symbol across the new
  submodules)
- `vernier.X` qualified references (rewritten to
  `vernier.instance.X` / `vernier.panoptic.X` for moved symbols)

It is idempotent: running twice produces the same output as running
once. Multi-line wrapped imports
(`from vernier import (\n    X,\n    Y,\n)`) are not handled — flatten
them first.

## What moved where

### `vernier.instance` (AP fold)

| Before | After |
|---|---|
| `vernier.Evaluator` | `vernier.instance.Evaluator` |
| `vernier.Bbox` / `Segm` / `Boundary` / `Keypoints` | `vernier.instance.Bbox` / `Segm` / `Boundary` / `Keypoints` |
| `vernier.IouKind` | `vernier.instance.IouKind` |
| `vernier.Summary` | `vernier.instance.Summary` |
| `vernier.EvalResult` | `vernier.instance.EvalResult` |
| `vernier.TableName` / `TablesConfig` | `vernier.instance.TableName` / `TablesConfig` |
| `vernier.StreamingEvaluator` | `vernier.instance.StreamingEvaluator` |
| `vernier.BackgroundEvaluator` | `vernier.instance.BackgroundEvaluator` |
| `vernier.Dataset` | `vernier.instance.Dataset` |
| `vernier.MemoryBudgetWarning` / `OutOfBudgetError` / `QueueFullError` | `vernier.instance.{...}` |
| `vernier.confusion_matrix` / `error_decomposition` / `fp_iou_histogram` | `vernier.instance.{...}` |
| `vernier.FpIouHistogram` / `TideConfig` / `TideReport` | `vernier.instance.{...}` |

### `vernier.panoptic` (panoptic-quality)

The `Panoptic` prefix is dropped from the unqualified type names:

| Before | After |
|---|---|
| `vernier.PanopticEvaluator` | `vernier.panoptic.Evaluator` |
| `vernier.PanopticDataset` | `vernier.panoptic.Dataset` |
| `vernier.PanopticPredictions` | `vernier.panoptic.Predictions` |
| `vernier.PanopticSummary` | `vernier.panoptic.Summary` |
| `vernier.ClassPanopticStats` | `vernier.panoptic.ClassPanopticStats` (kept; doesn't conflict) |

### `vernier.semantic` (semantic-segmentation, ADR-0028)

New surface — no migration needed (these types didn't exist at
0.0.x flat root). See
[Migrating from `mmsegmentation`](from-mmsegmentation.md) for the
per-dataset preset story.

### Stays at the root

ADR-0029 keeps two categories at `vernier`:

- **Cross-paradigm shared types**: `ParityMode`, `Frequency`,
  `__version__`, `version`.
- **The pycocotools migration shim**: `COCOeval`,
  `patch_pycocotools`. ADR-0007's drop-in claim ("change one
  import line and your eval code runs") is load-bearing for
  adoption; moving these to `vernier.instance` would cost a bullet
  point in every downstream migration discussion.

```python
# These imports are unchanged:
from vernier import COCOeval, patch_pycocotools, ParityMode, Frequency, version
```

## A common gotcha

`Dataset` exists in both `vernier.instance` and `vernier.panoptic`
(and `vernier.semantic`). The unprefixed name reflects the paradigm:

```python
from vernier.instance import Dataset as InstanceDataset
from vernier.panoptic import Dataset as PanopticDataset
from vernier.semantic import Dataset as SemanticDataset
```

If you mix paradigms in one file, alias them. The
`migrate_imports.py` script does **not** perform this aliasing —
only one of the three Dataset types existed under the flat root
(`vernier.Dataset` was the instance one), so there's no ambiguity to
preserve.

## See also

- [ADR-0029](../adr/0029-namespace.md) — design record.
- [Three paradigms: instance, panoptic, semantic](../explanation/three-paradigms.md)
  — when to reach for which submodule.
