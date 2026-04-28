"""Whole-dataset boundary-IoU parity smoke against the bowenc0221 oracle on COCO val2017.

Mirrors ``tests/python/parity/test_coco_val.py`` for the boundary track. The
reference is the vendored ``bowenc0221/boundary-iou-api`` oracle (per
ADR-0010), not pycocotools — boundary IoU isn't a stock pycocotools
``iouType``. The smoke reads its inputs from the same env vars the
bbox/segm smoke uses so a single ``./tools/fetch-coco-val.sh`` populates
all three tracks.

Skipped by default: enabling these requires the COCO val2017 annotations
(license-restricted, ~245 MB) and a segmentation predictions JSON.
Neither is committed. See ``docs/engineering/coco-val-parity.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ..coco_val_paths import DT_SEGM_ENV, GT_ENV, require_env_path, require_perfect_dt_artifacts
from .e2e_harness import assert_snapshots_equal, snapshot


def _assert_parity(gt: Path, dt: Path) -> None:
    ref = snapshot("oracle", gt, dt)
    cand = snapshot("vernier", gt, dt)
    assert_snapshots_equal(ref, cand)


@pytest.mark.parity_boundary
@pytest.mark.coco_val
def test_coco_val2017_boundary_parity() -> None:
    """Real detector segm predictions: bring your own JSON via env vars."""
    _assert_parity(require_env_path(GT_ENV), require_env_path(DT_SEGM_ENV))


@pytest.mark.parity_boundary
@pytest.mark.coco_val
def test_coco_val2017_boundary_parity_perfect_dt() -> None:
    """Synthesised perfect-DT smoke: every det carries GT polygons."""
    _assert_parity(*require_perfect_dt_artifacts("perfect_dt_segm.json"))
