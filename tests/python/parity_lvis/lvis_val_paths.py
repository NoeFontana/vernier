"""Shared env-var contract + cache layout for the LVIS v1 val parity smoke.

Mirrors :mod:`coco_val_paths` for the LVIS rollout (ADR-0026 PR-6).
The smoke gates on ``VERNIER_LVIS_GT_PATH`` and
``VERNIER_LVIS_DT_PATH`` (plus segm flavor) and falls through to the
canonical :mod:`lvis_val_cache` cache when both are unset.

A subsample knob (``VERNIER_LVIS_VAL_SAMPLE_IMAGES``) lets the smoke
run on a bounded prefix of the val set. Defaults to 1000 images
(the dense ``Vec<Option<PerImageEval>>`` orchestrator grid allocates
~232 bytes per cell * 1203 cats * 4 areas * n_images, so full val
peaks above 22 GB resident — out of reach on a 16 GB box). Explicit
override to ``-1`` runs the full corpus; ``0`` is treated as 1000.
The follow-up perf push (sparse cell storage) is tracked outside
this PR.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from lvis_val_cache import GT_FILENAME, cache_root

GT_ENV = "VERNIER_LVIS_GT_PATH"
DT_ENV = "VERNIER_LVIS_DT_PATH"
DT_SEGM_ENV = "VERNIER_LVIS_DT_SEGM_PATH"
SAMPLE_IMAGES_ENV = "VERNIER_LVIS_VAL_SAMPLE_IMAGES"
DEFAULT_SAMPLE_IMAGES = 1000


def sample_image_count() -> int:
    """How many val images the smoke should evaluate.

    Reads ``VERNIER_LVIS_VAL_SAMPLE_IMAGES`` (decimal int). Negative
    means "full corpus" (no subsample). Zero is treated as the
    default 1000 — we never want a "no images" smoke that silently
    exercises nothing.
    """
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


def _from_env(env: str) -> Path | None:
    value = os.environ.get(env)
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_file():
        return None
    return path


def require_perfect_dt_artifacts(dt_filename: str) -> tuple[Path, Path]:
    """Locate the cached LVIS GT and a perfect-DT JSON, or skip the test.

    Resolution order:

    1. ``VERNIER_LVIS_GT_PATH`` + the matching DT env var (caller picks
       which one).
    2. The :func:`lvis_val_cache.cache_root` directory, where
       ``python -m lvis_val_cache`` writes them.

    Skipping when both are missing keeps ``just test`` green on a
    clean checkout — the smoke is opt-in.
    """
    cache = cache_root()
    gt = _from_env(GT_ENV) or cache / GT_FILENAME
    dt_env_value = _from_env(DT_ENV) if dt_filename == "perfect_dt.json" else _from_env(DT_SEGM_ENV)
    dt = dt_env_value or cache / dt_filename
    if not gt.is_file() or not dt.is_file():
        pytest.skip(
            f"run `python -m lvis_val_cache` to populate {cache}; "
            f'see ADR-0026 §"Parity strategy" for the GT URL and SHA256.'
        )
    return gt, dt
