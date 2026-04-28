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


@pytest.fixture(scope="module")
def perfect_match_segm_coco() -> tuple[COCO, COCO]:
    gt = COCO(str(FIXTURES / "perfect_match_segm" / "gt.json"))
    dt = gt.loadRes(str(FIXTURES / "perfect_match_segm" / "dt.json"))
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


def test_boundary_iou_type_runs_end_to_end(
    perfect_match_segm_coco: tuple[COCO, COCO],
) -> None:
    gt, dt = perfect_match_segm_coco
    e = COCOeval(gt, dt, iouType="boundary")
    e.evaluate()
    e.accumulate()
    e.summarize()
    assert e.stats.shape == (12,)
    assert e.stats[0] == pytest.approx(1.0)


def test_boundary_dilation_ratio_propagates(
    perfect_match_segm_coco: tuple[COCO, COCO],
) -> None:
    # `dilation_ratio` reaches the kernel: a degenerate band radius (very
    # small ratio) collapses the boundary mask to zero pixels and the
    # `min(mask_iou, boundary_iou)` composition pulls AP below 1.
    gt, dt = perfect_match_segm_coco
    tight = COCOeval(gt, dt, iouType="boundary", dilation_ratio=1e-6)
    tight.evaluate()
    tight.accumulate()
    tight.summarize()
    assert tight.stats[0] < 1.0


def test_boundary_dilation_ratio_validation_propagates(
    perfect_match_segm_coco: tuple[COCO, COCO],
) -> None:
    # Validation lives at the FFI boundary; the shim doesn't shadow it.
    gt, dt = perfect_match_segm_coco
    e = COCOeval(gt, dt, iouType="boundary", dilation_ratio=-0.1)
    with pytest.raises(ValueError, match="dilation_ratio"):
        e.evaluate()


def test_dilation_ratio_ignored_for_bbox(
    perfect_match_coco: tuple[COCO, COCO],
) -> None:
    # Mirrors bowenc0221's silent-accept behavior: passing
    # `dilation_ratio` with a non-boundary iouType is a no-op.
    gt, dt = perfect_match_coco
    e = COCOeval(gt, dt, iouType="bbox", dilation_ratio=0.5)
    e.evaluate()
    e.accumulate()
    e.summarize()
    assert e.stats[0] == pytest.approx(1.0)


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


def test_accumulate_normalizes_max_dets_ascending(
    perfect_match_coco: tuple[COCO, COCO],
) -> None:
    # Quirk A2 (aligned): pycocotools' cocoeval.py:137 opens
    # accumulate() with `p.maxDets = sorted(p.maxDets)`. The drop-in
    # mirrors that — feeding `[100, 1, 10]` must produce the same M-axis
    # layout (and therefore the same `stats` vector) as the canonical
    # `[1, 10, 100]`, and `params.maxDets` itself must be normalized in
    # place so downstream introspection sees the sorted ladder.
    gt, dt = perfect_match_coco

    canonical = COCOeval(gt, dt, iouType="bbox")
    canonical.params.maxDets = [1, 10, 100]
    canonical.evaluate()
    canonical.accumulate()
    canonical.summarize()

    permuted = COCOeval(gt, dt, iouType="bbox")
    permuted.params.maxDets = [100, 1, 10]
    permuted.evaluate()
    permuted.accumulate()
    permuted.summarize()

    assert permuted.params.maxDets == [1, 10, 100]
    np.testing.assert_array_equal(permuted.stats, canonical.stats)


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


def test_default_use_segm_is_none_and_does_not_raise(
    perfect_match_coco: tuple[COCO, COCO],
) -> None:
    # Quirk L3 (corrected): vernier rejects an *assigned* useSegm but
    # the default sentinel of None must remain a no-op so untouched
    # downstream code keeps working.
    gt, dt = perfect_match_coco
    e = COCOeval(gt, dt, iouType="bbox")
    assert e.params.useSegm is None
    e.evaluate()  # should not raise


@pytest.mark.parametrize("use_segm_value", [0, 1])
def test_assigned_use_segm_raises_with_l3_message(
    perfect_match_coco: tuple[COCO, COCO],
    use_segm_value: int,
) -> None:
    # Quirk L3 (corrected): pycocotools deprecated useSegm years ago
    # but kept honoring it (overriding iouType, printing a warning).
    # Vernier drops the honor path entirely — any non-None assignment
    # must raise with a message that names useSegm, points users at
    # iouType, and cites the L3 quirk so the error is grep-able back
    # to the disposition table.
    gt, dt = perfect_match_coco
    e = COCOeval(gt, dt, iouType="bbox")
    e.params.useSegm = use_segm_value
    with pytest.raises(NotImplementedError) as excinfo:
        e.evaluate()
    msg = str(excinfo.value)
    assert "useSegm" in msg
    assert "iouType" in msg
    assert "L3" in msg
