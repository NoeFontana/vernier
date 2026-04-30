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

import json
from pathlib import Path

import numpy as np
import pytest
from pycocotools.coco import COCO

import vernier
from vernier import COCOeval
from vernier._compat import PycocotoolsCOCOeval

FIXTURES = Path(__file__).parent / "parity" / "fixtures"

# Synthetic 17-keypoint perfect-match fixture (mirrors the inline shape
# used by ``tests/python/test_evaluator.py``). One image, one person GT
# with all 17 keypoints visible; one DT with byte-identical coordinates.
# AP @ default sigmas collapses to 1.0 — the perfect-prediction sentinel
# that drives the parity assertions below.
_KP_COORDS: tuple[tuple[float, float], ...] = (
    (10.0, 10.0),
    (12.0, 8.0),
    (8.0, 8.0),
    (14.0, 9.0),
    (6.0, 9.0),
    (16.0, 20.0),
    (4.0, 20.0),
    (18.0, 30.0),
    (2.0, 30.0),
    (20.0, 40.0),
    (0.0, 40.0),
    (14.0, 50.0),
    (6.0, 50.0),
    (16.0, 65.0),
    (4.0, 65.0),
    (18.0, 80.0),
    (2.0, 80.0),
)


def _flatten_kp(coords: tuple[tuple[float, float], ...], visibility: int = 2) -> list[float]:
    flat: list[float] = []
    for x, y in coords:
        flat.extend((x, y, float(visibility)))
    return flat


def _kp_gt_dict() -> dict[str, object]:
    return {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                # bbox area 3200 lands in the 'large' kp area bucket
                # (>96^2 == 9216 is 'large'; 32^2..96^2 is 'medium';
                # this annotation is in 'medium').
                "bbox": [0, 0, 40, 80],
                "area": 3200,
                "iscrowd": 0,
                "num_keypoints": 17,
                "keypoints": _flatten_kp(_KP_COORDS),
            },
        ],
        "categories": [{"id": 1, "name": "person"}],
    }


def _kp_dt_list() -> list[dict[str, object]]:
    return [
        {
            "image_id": 1,
            "category_id": 1,
            "score": 0.99,
            "bbox": [0, 0, 40, 80],
            "keypoints": _flatten_kp(_KP_COORDS),
        },
    ]


@pytest.fixture
def perfect_match_kp_coco(tmp_path: Path) -> tuple[COCO, COCO]:
    gt_path = tmp_path / "kp_gt.json"
    dt_path = tmp_path / "kp_dt.json"
    gt_path.write_text(json.dumps(_kp_gt_dict()))
    dt_path.write_text(json.dumps(_kp_dt_list()))
    gt = COCO(str(gt_path))
    dt = gt.loadRes(str(dt_path))
    return gt, dt


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


def test_iou_type_keypoints_constructs(perfect_match_kp_coco: tuple[COCO, COCO]) -> None:
    # The shim accepts iouType="keypoints" without raising — the
    # explicit Phase-3 rejection has been replaced by an OKS dispatch.
    gt, dt = perfect_match_kp_coco
    e = COCOeval(gt, dt, iouType="keypoints")
    assert e.params.iouType == "keypoints"


def test_keypoints_default_param_grid_matches_pycocotools(
    perfect_match_kp_coco: tuple[COCO, COCO],
) -> None:
    # Mirrors `setKpParams` in pycocotools: kp drops the small bucket
    # (quirk D5), pins the ladder to [20], and exposes a default
    # COCO-person 17-sigma table on `params.kpt_oks_sigmas` (quirk F1).
    gt, dt = perfect_match_kp_coco
    e = COCOeval(gt, dt, iouType="keypoints")
    assert e.params.maxDets == [20]
    assert e.params.areaRng == [[0, 1e5**2], [32**2, 96**2], [96**2, 1e5**2]]
    assert e.params.areaRngLbl == ["all", "medium", "large"]
    assert e.params.kpt_oks_sigmas.shape == (17,)
    # Spot-check the COCO-person sigma table (nose / left-eye).
    np.testing.assert_allclose(e.params.kpt_oks_sigmas[:2], [0.026, 0.025])


def test_keypoints_evaluate_end_to_end(
    perfect_match_kp_coco: tuple[COCO, COCO],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Perfect prediction collapses AP and AR to 1.0; the kp summary plan
    # produces 10 stats (vs. 12 for detection).
    gt, dt = perfect_match_kp_coco
    e = COCOeval(gt, dt, iouType="keypoints")
    e.evaluate()
    e.accumulate()
    capsys.readouterr()
    e.summarize()
    assert e.stats.shape == (10,)
    assert e.stats[0] == pytest.approx(1.0)  # AP @ all
    assert e.stats[5] == pytest.approx(1.0)  # AR @ all


def test_keypoints_shim_matches_pycocotools_strict(
    perfect_match_kp_coco: tuple[COCO, COCO],
) -> None:
    # Strict mode is bit-exact with pycocotools. Build a fresh
    # pycocotools-native evaluator on the same GT/DT and assert the
    # 10-stat vector matches array-equal (not allclose).
    from pycocotools.cocoeval import COCOeval as PycocoEval

    gt, dt = perfect_match_kp_coco
    ref = PycocoEval(gt, dt, iouType="keypoints")
    ref.evaluate()
    ref.accumulate()
    ref.summarize()

    shim = COCOeval(gt, dt, iouType="keypoints", parity_mode="strict")
    shim.evaluate()
    shim.accumulate()
    shim.summarize()

    np.testing.assert_array_equal(shim.stats, ref.stats)


def test_keypoints_shim_matches_evaluator_api(
    perfect_match_kp_coco: tuple[COCO, COCO],
) -> None:
    # The shim's keypoints path dispatches to the same Rust kernel as
    # `vernier.Evaluator(iou=Keypoints())`; on a shared GT/DT pair the
    # summary statistics agree element-wise.
    gt, dt = perfect_match_kp_coco
    shim = COCOeval(gt, dt, iouType="keypoints", parity_mode="strict")
    shim.evaluate()
    shim.accumulate()
    shim.summarize()

    gt_bytes = json.dumps(_kp_gt_dict()).encode()
    dt_bytes = json.dumps(_kp_dt_list()).encode()
    direct = vernier.Evaluator(iou=vernier.Keypoints(), parity_mode="strict").evaluate(
        gt_bytes, dt_bytes
    )
    np.testing.assert_array_equal(shim.stats, np.asarray(direct.stats, dtype=np.float64))


def test_keypoints_custom_sigmas_propagates_to_kernel(
    perfect_match_kp_coco: tuple[COCO, COCO],
) -> None:
    # Quirk F1: pycocotools stores a single 17-tuple on the params
    # object; vernier fans it out across every GT category id at the
    # FFI boundary. Tightening sigmas by ~30x squeezes OKS toward an
    # indicator function — a deliberately-shifted DT that the default
    # sigmas tolerate (AP=1.0) collapses under the tight sigmas.
    gt, _ = perfect_match_kp_coco

    # Shift each predicted x by 5 px so the prediction is no longer
    # byte-identical to GT but still inside default OKS tolerance.
    shifted_dt = _kp_dt_list()
    shifted_kps = list(shifted_dt[0]["keypoints"])  # type: ignore[arg-type]
    for i in range(0, len(shifted_kps), 3):
        shifted_kps[i] += 5.0
    shifted_dt[0]["keypoints"] = shifted_kps
    res = gt.loadRes(shifted_dt)  # pyright: ignore[reportArgumentType]

    default = COCOeval(gt, res, iouType="keypoints")
    default.evaluate()
    default.accumulate()
    default.summarize()

    tight = COCOeval(gt, res, iouType="keypoints")
    tight.params.kpt_oks_sigmas = np.full(17, 1e-3, dtype=np.float64)
    tight.evaluate()
    tight.accumulate()
    tight.summarize()

    # AP collapses under tight sigmas; default sigmas tolerate the shift.
    assert tight.stats[0] < default.stats[0]


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
