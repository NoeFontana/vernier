"""Tests for the pycocotools drop-in (`vernier._compat.PycocotoolsCOCOeval`).

The drop-in mirrors the pycocotools state machine
(``evaluate`` → ``accumulate`` → ``summarize``); these tests verify the
shapes downstream code depends on, the ``Params`` mutability surface,
and the ADR-0007 default of ``parity_mode="strict"``.

Bit-for-bit numerical parity vs pycocotools is exercised by
``tests/python/parity/test_parity.py`` once the harness routes through
the drop-in (Phase 1 PR-3a / PR-3c). This file focuses on shape and
state-machine compliance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pycocotools.coco import COCO

from vernier import COCOeval
from vernier._compat import PycocotoolsCOCOeval

FIXTURES = Path(__file__).parent / "parity" / "fixtures"


@pytest.fixture(scope="module")
def perfect_match_coco() -> tuple[COCO, COCO]:
    gt = COCO(str(FIXTURES / "perfect_match" / "gt.json"))
    dt = gt.loadRes(str(FIXTURES / "perfect_match" / "dt.json"))
    return gt, dt


def test_public_alias_points_to_drop_in() -> None:
    assert COCOeval is PycocotoolsCOCOeval


def test_default_parity_mode_is_strict() -> None:
    # Per ADR-0007: default is "strict" because the drop-in is the
    # migration path from pycocotools, where bit-exact behavior is the
    # expected baseline.
    assert PycocotoolsCOCOeval.DEFAULT_PARITY_MODE == "strict"


def test_state_machine_populates_pycocotools_attrs(
    perfect_match_coco: tuple[COCO, COCO],
) -> None:
    gt, dt = perfect_match_coco
    e = COCOeval(gt, dt, iouType="bbox")

    e.evaluate()
    assert isinstance(e.evalImgs, list)
    populated = [x for x in e.evalImgs if x is not None]
    assert populated, "evaluate() should populate evalImgs"

    e.accumulate()
    assert set(e.eval.keys()) >= {"params", "counts", "date", "precision", "recall", "scores"}
    assert isinstance(e.eval["precision"], np.ndarray)
    assert e.eval["precision"].shape[2:] == (1, 4, 3)  # K, A, M

    e.summarize()
    assert e.stats.shape == (12,)
    assert e.stats[0] == pytest.approx(1.0)


def test_eval_dict_date_is_iso_like(perfect_match_coco: tuple[COCO, COCO]) -> None:
    gt, dt = perfect_match_coco
    e = COCOeval(gt, dt, iouType="bbox")
    e.evaluate()
    e.accumulate()
    # Pycocotools format: "%Y-%m-%d %H:%M:%S" — 19 chars, two spaces.
    assert isinstance(e.eval["date"], str)
    assert len(e.eval["date"]) == 19


def test_summarize_strict_mode_prints(
    perfect_match_coco: tuple[COCO, COCO],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Quirk L5: strict mode preserves pycocotools' stdout side-effect.
    gt, dt = perfect_match_coco
    e = COCOeval(gt, dt, iouType="bbox")
    e.evaluate()
    e.accumulate()
    capsys.readouterr()
    e.summarize()
    out = capsys.readouterr().out
    assert "Average Precision" in out
    assert "Average Recall" in out


def test_summarize_corrected_mode_is_silent(
    perfect_match_coco: tuple[COCO, COCO],
    capsys: pytest.CaptureFixture[str],
) -> None:
    gt, dt = perfect_match_coco
    e = COCOeval(gt, dt, iouType="bbox", parity_mode="corrected")
    e.evaluate()
    e.accumulate()
    capsys.readouterr()
    e.summarize()
    assert capsys.readouterr().out == ""


def test_iou_type_keypoints_not_yet_supported(perfect_match_coco: tuple[COCO, COCO]) -> None:
    gt, dt = perfect_match_coco
    with pytest.raises(NotImplementedError, match="keypoints"):
        COCOeval(gt, dt, iouType="keypoints")


def test_constructor_default_iou_type_matches_pycocotools() -> None:
    # pycocotools.cocoeval.COCOeval()'s third positional default is
    # iouType="segm"; the drop-in mirrors it so existing user code
    # constructed without a positional iouType lands on the same path.
    e = COCOeval()
    assert e.params.iouType == "segm"


def test_evaluate_without_inputs_raises() -> None:
    e = COCOeval(iouType="bbox")
    with pytest.raises(RuntimeError, match="cocoGt"):
        e.evaluate()


def test_accumulate_before_evaluate_raises(
    perfect_match_coco: tuple[COCO, COCO],
) -> None:
    gt, dt = perfect_match_coco
    e = COCOeval(gt, dt, iouType="bbox")
    with pytest.raises(RuntimeError, match="evaluate"):
        e.accumulate()


def test_summarize_before_accumulate_raises(
    perfect_match_coco: tuple[COCO, COCO],
) -> None:
    gt, dt = perfect_match_coco
    e = COCOeval(gt, dt, iouType="bbox")
    e.evaluate()
    with pytest.raises(RuntimeError, match="accumulate"):
        e.summarize()


def test_params_default_grid_matches_pycocotools(
    perfect_match_coco: tuple[COCO, COCO],
) -> None:
    gt, dt = perfect_match_coco
    e = COCOeval(gt, dt, iouType="bbox")
    assert list(e.params.maxDets) == [1, 10, 100]
    assert e.params.useCats == 1
    assert e.params.iouThrs.shape == (10,)
    assert e.params.recThrs.shape == (101,)
    assert e.params.areaRngLbl == ["all", "small", "medium", "large"]
    assert e.params.imgIds == sorted(gt.getImgIds())
    assert e.params.catIds == sorted(gt.getCatIds())


def test_params_max_dets_mutation_propagates(
    perfect_match_coco: tuple[COCO, COCO],
) -> None:
    # Downstream code mutates params.maxDets after construction; the
    # accumulator must consume the mutated list. The bbox summary
    # template requires 100 in the M axis (it's the AR_100 bucket), so
    # the mutation must keep that — we add a fourth threshold to widen
    # the precision tensor.
    gt, dt = perfect_match_coco
    e = COCOeval(gt, dt, iouType="bbox")
    e.params.maxDets = [1, 10, 100, 200]
    e.evaluate()
    e.accumulate()
    assert e.eval["precision"].shape[-1] == 4


def test_unsupported_iou_thrs_mutation_raises(
    perfect_match_coco: tuple[COCO, COCO],
) -> None:
    gt, dt = perfect_match_coco
    e = COCOeval(gt, dt, iouType="bbox")
    e.params.iouThrs = np.array([0.5, 0.75])
    with pytest.raises(NotImplementedError, match="iouThrs"):
        e.evaluate()


def test_unsupported_area_rng_mutation_raises(
    perfect_match_coco: tuple[COCO, COCO],
) -> None:
    gt, dt = perfect_match_coco
    e = COCOeval(gt, dt, iouType="bbox")
    e.params.areaRng = [[0, 1e10]]
    with pytest.raises(NotImplementedError, match="areaRng"):
        e.evaluate()


def test_unsupported_img_ids_subsetting_raises(
    perfect_match_coco: tuple[COCO, COCO],
) -> None:
    gt, dt = perfect_match_coco
    e = COCOeval(gt, dt, iouType="bbox")
    e.params.imgIds = []
    with pytest.raises(NotImplementedError, match="imgIds"):
        e.evaluate()
