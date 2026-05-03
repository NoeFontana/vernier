"""Shared env-var contract + cache layout for the COCO panoptic val2017 parity smoke.

Mirrors :mod:`coco_val_paths` and :mod:`lvis_val_paths` for the
panoptic rollout (ADR-0025 PR-6). The smoke gates on
``VERNIER_PANOPTIC_GT_PATH`` / ``VERNIER_PANOPTIC_GT_PNG_DIR`` /
``VERNIER_PANOPTIC_DT_PATH`` / ``VERNIER_PANOPTIC_DT_PNG_DIR`` and
falls through to the canonical :mod:`panoptic_val_cache` cache when
all four are unset.

A subsample knob (``VERNIER_PANOPTIC_VAL_SAMPLE_IMAGES``) lets the
smoke run on a bounded prefix of the val set. Defaults to 100
images. Negative means "full corpus" (no subsample); zero is treated
as the default 100 — a "no images" smoke would silently exercise
nothing.

Q6 closure procedure (multi-process tolerance pinning, ``PANOPTIC_PARITY_EPS``):

1. Run :func:`panoptic_val_cache.ensure_perfect_dt` (or any real DT) on full val.
2. Run :func:`panopticapi.evaluation.pq_compute_single_core` with ``proc_id=0``
   over the matched annotation set; capture per-category PQ rows.
3. Run :func:`panopticapi.evaluation.pq_compute_multi_core` for ``cpu_count``
   in ``{2, 4, 8}`` (override via ``multiprocessing.cpu_count`` monkey patch).
4. Compute max ULP distance per category between single-core and each
   multi-core trace; the max across the matrix is the new
   ``PANOPTIC_PARITY_EPS``.
5. Update ``crates/vernier-panoptic/src/parity.rs`` and the
   "Parity tolerance" section of
   ``tests/python/parity_panoptic/oracle/VENDORING.md`` atomically.

Steps 2-4 of the procedure are scripted as
``tools/panoptic_val_cache.py --measure-eps`` (lands in a follow-up
PR; the val data is the load-bearing input). Until that lands, the
placeholder ``1e-9`` is what guards aligned-mode comparisons; strict
mode demands bit-equality vs ``pq_compute_single_core`` regardless.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from panoptic_val_cache import GT_JSON_FILENAME, GT_PNG_DIRNAME, cache_root

GT_ENV = "VERNIER_PANOPTIC_GT_PATH"
GT_PNG_DIR_ENV = "VERNIER_PANOPTIC_GT_PNG_DIR"
DT_ENV = "VERNIER_PANOPTIC_DT_PATH"
DT_PNG_DIR_ENV = "VERNIER_PANOPTIC_DT_PNG_DIR"
SAMPLE_IMAGES_ENV = "VERNIER_PANOPTIC_VAL_SAMPLE_IMAGES"
DEFAULT_SAMPLE_IMAGES = 100


def sample_image_count() -> int:
    raw = os.environ.get(SAMPLE_IMAGES_ENV)
    if raw is None or raw == "":
        return DEFAULT_SAMPLE_IMAGES
    try:
        n = int(raw)
    except ValueError:
        return DEFAULT_SAMPLE_IMAGES
    if n == 0:
        return DEFAULT_SAMPLE_IMAGES
    return n


def _from_env_file(env: str) -> Path | None:
    value = os.environ.get(env)
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_file():
        return None
    return path


def _from_env_dir(env: str) -> Path | None:
    value = os.environ.get(env)
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_dir():
        return None
    return path


def require_artifacts() -> tuple[Path, Path, Path, Path]:
    """Locate the cached panoptic GT + DT, or skip the test.

    Returns ``(gt_json, gt_png_dir, dt_json, dt_png_dir)``. The DT
    path falls through to the perfect-DT synthesized by
    :func:`panoptic_val_cache.ensure_perfect_dt` (a self-comparison
    that should produce ``PQ=1.0``).
    """
    cache = cache_root()
    gt_json = _from_env_file(GT_ENV) or cache / GT_JSON_FILENAME
    gt_png_dir = _from_env_dir(GT_PNG_DIR_ENV) or cache / GT_PNG_DIRNAME
    dt_json = _from_env_file(DT_ENV) or cache / "perfect_dt.json"
    dt_png_dir = _from_env_dir(DT_PNG_DIR_ENV) or cache / "perfect_dt_pngs"

    missing: list[str] = []
    if not gt_json.is_file():
        missing.append(f"GT JSON: {gt_json}")
    if not gt_png_dir.is_dir():
        missing.append(f"GT PNG dir: {gt_png_dir}")
    if not dt_json.is_file():
        missing.append(f"DT JSON: {dt_json}")
    if not dt_png_dir.is_dir():
        missing.append(f"DT PNG dir: {dt_png_dir}")

    if missing:
        pytest.skip(
            "panoptic val2017 cache not provisioned (run "
            "`python -m panoptic_val_cache`). Missing:\n  - " + "\n  - ".join(missing)
        )
    return gt_json, gt_png_dir, dt_json, dt_png_dir
