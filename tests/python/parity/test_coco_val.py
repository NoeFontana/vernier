"""Whole-dataset parity smoke tests against pycocotools on COCO val2017.

Five hand-crafted fixtures (``test_parity.py``) cover known quirks but
do not exercise the long tail of real-world inputs. The tests here run
the full COCO val2017 ground truth and assert every snapshot field —
eval-image dicts, precision/recall/scores tensors, the 12-element det
(or 10-element keypoints, per ADR-0012) stats vector — matches
pycocotools bit-for-bit in strict parity mode. Bbox, segm, and
keypoints all dispatch through the harness; the ``parity_boundary``
sibling tree covers boundary IoU against its own oracle.

Skipped by default: enabling them requires the COCO val2017
annotations (license-restricted, ~245 MB) and a detections JSON.
Neither is committed. See ``docs/engineering/coco-val-parity.md`` for
the env-var contract and the ``tools/fetch-coco-val.sh`` helper that
populates ``.cache/coco-val2017/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vernier import Evaluator, Keypoints

from ..coco_val_paths import (
    DT_ENV,
    DT_KEYPOINTS_ENV,
    DT_SEGM_ENV,
    GT_ENV,
    GT_KEYPOINTS_ENV,
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


@pytest.mark.parity
@pytest.mark.coco_val
def test_coco_val2017_keypoints_parity() -> None:
    """Real keypoint detector predictions: bring your own JSON via env vars.

    Per ADR-0012 the OKS kernel surfaces a 10-stat summary (re-indexed
    A-axis, no ``_S`` row — quirk D5). The vernier ``Evaluator`` runs
    in strict mode and we assert byte-identical equality against
    pycocotools' ``keypoints`` summary.
    """
    gt = require_env_path(GT_KEYPOINTS_ENV)
    dt = require_env_path(DT_KEYPOINTS_ENV)
    cand = Evaluator(iou=Keypoints(), parity_mode="strict").evaluate(
        gt.read_bytes(), dt.read_bytes()
    )
    cand_stats = np.asarray(cand.stats, dtype=np.float64)
    ref_stats = snapshot("pycocotools", gt, dt, "keypoints").stats
    assert cand_stats.shape == (10,), f"expected 10-stat kp summary, got {cand_stats.shape}"
    np.testing.assert_array_equal(cand_stats, ref_stats, err_msg="kp stats")
