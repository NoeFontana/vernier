"""COCO panoptic val2017 workload module (ADR-0033 §B1).

Resolves panoptic workload IDs into the
:class:`bench.workloads.PanopticWorkload` shape. Today registers one
concrete workload:

- ``coco_panoptic_val2017_perfect`` — GT-as-DT smoke. Every DT segment
  perfectly matches its GT counterpart; the comparator should produce
  ``PQ=SQ=RQ=1.0`` for every present category. Sub-minute on COCO
  panoptic val (5000 imgs); useful as a bench warm-up cell rather
  than a realistic detector benchmark.

The GT cache is provisioned by :mod:`panoptic_val_cache` (see
``tools/panoptic_val_cache/``); this module never downloads — it
only resolves cached paths and reports a clear error when the cache
is unprovisioned.

Also writes a sibling ``categories.json`` (extracted from the GT JSON's
``categories`` array) into the cache directory so the runner argspec
can pass ``--categories-json`` as a separate path. The categories
file is idempotent (rewritten only if missing).

The real-prediction follow-up (Mask2Former dump) is deferred to
Stage 3 — see the docstring TODO at module footer.
"""

from __future__ import annotations

import json
from pathlib import Path

from panoptic_val_cache import cache_root, ensure_gt, ensure_perfect_dt

PERFECT_WORKLOAD_ID: str = "coco_panoptic_val2017_perfect"

_CATEGORIES_FILENAME = "categories.json"


def _ensure_categories_json(gt_json_path: Path) -> Path:
    """Extract the ``categories`` array from the GT JSON into a
    sibling ``categories.json`` so the runner argspec can pass it as
    a separate path. Idempotent: skips if the file already exists.

    panopticapi's ``pq_compute_single_core`` reads categories from
    the GT JSON internally; vernier's :meth:`Dataset.from_arrays`
    takes them as a separate JSON byte string. Splitting the file
    once here keeps both runners' argspec uniform.
    """
    cache_dir = gt_json_path.parent
    cats_path = cache_dir / _CATEGORIES_FILENAME
    if cats_path.is_file():
        return cats_path
    with gt_json_path.open() as f:
        data = json.load(f)
    categories = data.get("categories", [])
    cats_path.write_text(json.dumps(categories))
    return cats_path


def perfect_workload_paths() -> tuple[Path, Path, Path, Path, Path]:
    """Return ``(gt_png_dir, gt_json, dt_png_dir, dt_json,
    categories_json)`` for the perfect-DT smoke.

    Raises ``FileNotFoundError`` (with an actionable hint) when the
    cache isn't provisioned. Never downloads — that's the
    :mod:`panoptic_val_cache` CLI's job.
    """
    cache = cache_root()
    gt_json = cache / "panoptic_val2017.json"
    if not gt_json.is_file():
        raise FileNotFoundError(
            f"COCO panoptic val2017 GT not cached at {cache}. "
            f"Provision via `python -m panoptic_val_cache` (downloads "
            f"~250MB JSON + ~3GB PNGs from images.cocodataset.org). "
            f"Per project memory: never commit dataset bytes — the "
            f"cache stays under the user data dir, not the repo."
        )
    # Cache is present; resolve all paths via the canonical helpers.
    # ``ensure_gt`` and ``ensure_perfect_dt`` are idempotent skips
    # when the artifacts already exist on disk.
    gt_json, gt_png_dir = ensure_gt()
    dt_json, dt_png_dir = ensure_perfect_dt()
    cats_json = _ensure_categories_json(gt_json)
    return gt_png_dir, gt_json, dt_png_dir, dt_json, cats_json


# TODO(stage-3 / S3-A): coco_panoptic_val2017_mask2former_<version> via
# tools/real_predictions_cache/panoptic.py:ensure_mask2former().
