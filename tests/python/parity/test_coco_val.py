"""Whole-dataset parity smoke tests against pycocotools on COCO val2017.

Five hand-crafted fixtures (``test_parity.py``) cover known quirks but
do not exercise the long tail of real-world inputs. The tests here run
the full COCO val2017 ground truth and assert every snapshot field —
eval-image dicts, precision/recall/scores tensors, the 12-element stats
vector — matches pycocotools bit-for-bit in strict parity mode.

Skipped by default: enabling them requires the COCO val2017
annotations (license-restricted, ~245 MB) and a detections JSON.
Neither is committed. See ``docs/engineering/coco-val-parity.md`` for
the env-var contract and the ``tools/fetch-coco-val.sh`` helper that
populates ``.cache/coco-val2017/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ..coco_val_paths import (
    DT_ENV,
    DT_SEGM_ENV,
    GT_ENV,
    require_env_path,
    require_perfect_dt_artifacts,
)
from .harness import IouType, assert_snapshots_equal, snapshot


def _assert_parity(gt: Path, dt: Path, iou_type: IouType) -> None:
    ref = snapshot("pycocotools", gt, dt, iou_type)
    cand = snapshot("vernier", gt, dt, iou_type)
    assert_snapshots_equal(ref, cand)


@pytest.mark.parity
@pytest.mark.coco_val
def test_coco_val2017_bbox_parity() -> None:
    """Real detector predictions: bring your own JSON via env vars."""
    _assert_parity(require_env_path(GT_ENV), require_env_path(DT_ENV), "bbox")


@pytest.mark.parity
@pytest.mark.coco_val
def test_coco_val2017_bbox_parity_perfect_dt() -> None:
    """Synthesised perfect-DT smoke: scale-only check (AP is trivially 1.0)."""
    _assert_parity(*require_perfect_dt_artifacts("perfect_dt.json"), "bbox")


@pytest.mark.parity
@pytest.mark.coco_val
def test_coco_val2017_segm_parity() -> None:
    """Real detector predictions: bring your own segm JSON via env vars."""
    _assert_parity(require_env_path(GT_ENV), require_env_path(DT_SEGM_ENV), "segm")


@pytest.mark.parity
@pytest.mark.coco_val
def test_coco_val2017_segm_parity_perfect_dt() -> None:
    """Synthesised perfect-DT smoke for segm: every det carries GT polygons."""
    _assert_parity(*require_perfect_dt_artifacts("perfect_dt_segm.json"), "segm")
