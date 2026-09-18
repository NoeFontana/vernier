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

import contextlib
import copy
import io
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval as PycocotoolsReferenceCOCOeval

from vernier import COCOeval
from vernier import _compat as vernier_compat
from vernier._compat import PycocotoolsCOCOeval
from vernier.instance import Evaluator, Keypoints

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


def test_eval_imgs_is_built_only_when_read(
    perfect_match_coco: tuple[COCO, COCO],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Materializing pycocotools' per-image dicts costs as much as the
    # evaluation, and nothing in evaluate/accumulate/summarize reads them.
    calls: list[int] = []

    class _CountingGrid:
        def __init__(self, grid: Any) -> None:
            self._grid = grid

        def __getattr__(self, name: str) -> Any:
            return getattr(self._grid, name)

        def eval_imgs(self) -> list[dict[str, Any] | None]:
            calls.append(1)
            return self._grid.eval_imgs()

    build_grid = vernier_compat.evaluate_bbox_grid
    monkeypatch.setattr(
        vernier_compat,
        "evaluate_bbox_grid",
        lambda *args, **kwargs: _CountingGrid(build_grid(*args, **kwargs)),
    )
    gt, dt = perfect_match_coco
    e = COCOeval(gt, dt, iouType="bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        e.evaluate()
        e.accumulate()
        e.summarize()
    assert calls == []
    first = e.evalImgs
    assert e.evalImgs is first
    assert calls == [1]
    assert any(record is not None for record in first)


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


def _run_kp(cls: Any, gt: COCO, dt: COCO, use_cats: int, **kwargs: Any) -> NDArray[np.float64]:
    """Drive a pycocotools-shaped keypoints evaluator with a ``useCats`` override."""
    e = cls(gt, dt, iouType="keypoints", **kwargs)
    e.params.useCats = use_cats
    e.evaluate()
    e.accumulate()
    e.summarize()
    return np.asarray(e.stats, dtype=np.float64)


def test_keypoints_use_cats_zero_strict_matches_pycocotools(
    perfect_match_kp_coco: tuple[COCO, COCO],
) -> None:
    # Quirk F6 (corrected, ADR-0058). `COCOeval.evaluate` computes the
    # per-cell similarity over `catIds = p.catIds if p.useCats else [-1]`
    # (ce:142). `computeIoU` forks on `p.useCats` and gathers every
    # category under the `-1` sentinel (ce:165-170); `computeOks` does
    # not (ce:195-196) — it indexes `self._gts[imgId, -1]`, finds
    # nothing, and returns `[]`. `evaluateImg` still gathers the cell's
    # GTs and DTs (ce:241-246), so the perfect DT survives as an
    # unmatched FP and keypoint AP comes out 0.
    #
    # Strict is the drop-in's default parity mode, so the default path
    # must reproduce that 0 exactly — bugs included. Assert against the
    # real library, not a hand-written expectation.
    from pycocotools.cocoeval import COCOeval as PycocoEval

    gt, dt = perfect_match_kp_coco
    ref = _run_kp(PycocoEval, gt, dt, use_cats=0)
    # Pin the direction: the oracle really does report AP 0 here, even
    # though the DT is byte-identical to the GT.
    assert ref[0] == 0.0
    shim_default = _run_kp(COCOeval, gt, dt, use_cats=0)
    shim_strict = _run_kp(COCOeval, gt, dt, use_cats=0, parity_mode="strict")
    np.testing.assert_array_equal(shim_default, ref)
    np.testing.assert_array_equal(shim_strict, ref)


def test_keypoints_use_cats_zero_corrected_scores_the_collapsed_cell(
    perfect_match_kp_coco: tuple[COCO, COCO],
) -> None:
    # The other direction of quirk F6: corrected mode keeps vernier's
    # category-agnostic gather (quirk L4) on the OKS path too, so the
    # perfect DT matches and AP is 1.0 — the same answer `useCats=1`
    # gives, which is what "ignore category labels" is supposed to mean
    # on a single-category dataset.
    from pycocotools.cocoeval import COCOeval as PycocoEval

    gt, dt = perfect_match_kp_coco
    corrected = _run_kp(COCOeval, gt, dt, use_cats=0, parity_mode="corrected")
    ref_use_cats_one = _run_kp(PycocoEval, gt, dt, use_cats=1)
    np.testing.assert_array_equal(corrected, ref_use_cats_one)
    assert corrected[0] == pytest.approx(1.0)
    # And it is a real divergence from the oracle's `useCats=0` answer —
    # that divergence is the whole point of the `corrected` disposition.
    assert corrected[0] != _run_kp(PycocoEval, gt, dt, use_cats=0)[0]


def test_keypoints_use_cats_zero_corrected_matches_across_categories(
    tmp_path: Path,
) -> None:
    # Cross-category evidence for quirk F6's corrected side: a GT
    # labelled `person` and a keypoint-identical DT labelled `animal`.
    # `useCats=0` means "ignore category labels", so the semantically
    # right answer is a match (AP 1.0). pycocotools and strict mode both
    # report 0; corrected recovers the match.
    from pycocotools.cocoeval import COCOeval as PycocoEval

    gt_dict = _kp_gt_dict()
    gt_dict["categories"] = [{"id": 1, "name": "person"}, {"id": 2, "name": "animal"}]
    dt_list = _kp_dt_list()
    dt_list[0]["category_id"] = 2

    gt_path = tmp_path / "xcat_gt.json"
    dt_path = tmp_path / "xcat_dt.json"
    gt_path.write_text(json.dumps(gt_dict))
    dt_path.write_text(json.dumps(dt_list))
    gt = COCO(str(gt_path))
    dt = gt.loadRes(str(dt_path))

    assert _run_kp(PycocoEval, gt, dt, use_cats=0)[0] == 0.0
    assert _run_kp(COCOeval, gt, dt, use_cats=0, parity_mode="strict")[0] == 0.0
    assert _run_kp(COCOeval, gt, dt, use_cats=0, parity_mode="corrected")[0] == pytest.approx(1.0)


def test_keypoints_use_cats_one_is_unaffected_by_f6(
    perfect_match_kp_coco: tuple[COCO, COCO],
) -> None:
    # The F6 blackout is scoped to the `useCats=0` collapse: the
    # standard keypoints configuration must be untouched in both modes.
    from pycocotools.cocoeval import COCOeval as PycocoEval

    gt, dt = perfect_match_kp_coco
    ref = _run_kp(PycocoEval, gt, dt, use_cats=1)
    for mode in ("strict", "corrected"):
        np.testing.assert_array_equal(_run_kp(COCOeval, gt, dt, use_cats=1, parity_mode=mode), ref)


def test_bbox_use_cats_zero_is_not_blacked_out(
    perfect_match_coco: tuple[COCO, COCO],
) -> None:
    # F6 is keypoints-only: bbox routes through `computeIoU`, which
    # *does* have the `useCats` fork, so strict mode must keep matching
    # under `useCats=0` and stay bit-equal to the oracle.
    from pycocotools.cocoeval import COCOeval as PycocoEval

    gt, dt = perfect_match_coco
    ref = PycocoEval(gt, dt, iouType="bbox")
    ref.params.useCats = 0
    ref.evaluate()
    ref.accumulate()
    ref.summarize()
    shim = COCOeval(gt, dt, iouType="bbox", parity_mode="strict")
    shim.params.useCats = 0
    shim.evaluate()
    shim.accumulate()
    shim.summarize()
    assert ref.stats[0] == pytest.approx(1.0)
    np.testing.assert_array_equal(shim.stats, np.asarray(ref.stats, dtype=np.float64))


def test_keypoints_shim_matches_evaluator_api(
    perfect_match_kp_coco: tuple[COCO, COCO],
) -> None:
    # The shim's keypoints path dispatches to the same Rust kernel as
    # `vernier.instance.Evaluator(iou=Keypoints())`; on a shared GT/DT pair the
    # summary statistics agree element-wise.
    gt, dt = perfect_match_kp_coco
    shim = COCOeval(gt, dt, iouType="keypoints", parity_mode="strict")
    shim.evaluate()
    shim.accumulate()
    shim.summarize()

    gt_bytes = json.dumps(_kp_gt_dict()).encode()
    dt_bytes = json.dumps(_kp_dt_list()).encode()
    direct = Evaluator(iou=Keypoints(), parity_mode="strict").evaluate(gt_bytes, dt_bytes)
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
    # Quirk A2 (strict): pycocotools' cocoeval.py:137 opens
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


def _coco(dataset: dict[str, Any]) -> COCO:
    # Build a COCO the way TorchMetrics and other in-memory callers do:
    # assign `dataset` and index it, with no `loadRes` pass (so detection
    # `area` stays whatever the caller wrote — quirk J3).
    coco = COCO()
    # pycocotools' stubs type `dataset` as its file-loaded TypedDict.
    cast(Any, coco).dataset = copy.deepcopy(dataset)
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def _rle(rows: slice, cols: slice) -> dict[str, Any]:
    mask = np.zeros((128, 128), dtype=np.uint8, order="F")
    mask[rows, cols] = 1
    return dict(mask_utils.encode(mask))


def _mask_ann(ann_id: int, image_id: int, rows: slice, cols: slice, **extra: Any) -> dict[str, Any]:
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [cols.start, rows.start, cols.stop - cols.start, rows.stop - rows.start],
        "area": float((rows.stop - rows.start) * (cols.stop - cols.start)),
        "segmentation": _rle(rows, cols),
        **extra,
    }


def _run_eval(
    evaluator_type: type[Any],
    gt: dict[str, Any],
    dt: dict[str, Any],
    iou_type: str,
    **params: Any,
) -> Any:
    evaluator = evaluator_type(_coco(gt), _coco(dt), iouType=iou_type)
    for name, value in params.items():
        setattr(evaluator.params, name, value)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return evaluator


def _assert_matches_pycocotools(
    gt: dict[str, Any], dt: dict[str, Any], iou_type: str, **params: Any
) -> PycocotoolsCOCOeval:
    reference = _run_eval(PycocotoolsReferenceCOCOeval, gt, dt, iou_type, **copy.deepcopy(params))
    candidate = _run_eval(COCOeval, gt, dt, iou_type, **copy.deepcopy(params))
    np.testing.assert_array_equal(candidate.stats, reference.stats)
    np.testing.assert_array_equal(candidate.eval["precision"], reference.eval["precision"])
    np.testing.assert_array_equal(candidate.eval["recall"], reference.eval["recall"])
    return candidate


@pytest.fixture(scope="module")
def synthetic_bbox_datasets() -> tuple[dict[str, Any], dict[str, Any]]:
    # Jittered copies of random GT boxes: fractional AP/AR on every line,
    # so a mis-sliced threshold or maxDets entry cannot hide.
    rng = np.random.default_rng(0)
    images = [{"id": i, "width": 128, "height": 128} for i in range(4)]
    categories = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    gt_anns: list[dict[str, Any]] = []
    dt_anns: list[dict[str, Any]] = []
    for image in images:
        image_gts = []
        for _ in range(6):
            x, y = rng.uniform(0, 80, 2).tolist()
            w, h = rng.uniform(5, 60, 2).tolist()
            ann = {
                "id": len(gt_anns) + 1,
                "image_id": image["id"],
                "category_id": int(rng.integers(1, 3)),
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
            gt_anns.append(ann)
            image_gts.append(ann)
        for _ in range(20):
            target = image_gts[int(rng.integers(len(image_gts)))]
            x, y, w, h = (np.asarray(target["bbox"]) + rng.normal(0, 4, 4)).tolist()
            w, h = abs(w) + 1.0, abs(h) + 1.0
            dt_anns.append(
                {
                    "id": len(dt_anns) + 1,
                    "image_id": image["id"],
                    "category_id": target["category_id"],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "score": float(rng.uniform()),
                }
            )
    gt = {"images": images, "categories": categories, "annotations": gt_anns}
    dt = {"images": images, "categories": categories, "annotations": dt_anns}
    return gt, dt


def test_float32_rounded_thresholds_match_pycocotools(
    synthetic_bbox_datasets: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    # Callers that round-trip the canonical ladders through float32
    # (TorchMetrics' `torch.linspace(...).tolist()`) land ~2e-8 off the
    # defaults; the shim must evaluate on those exact values, as
    # pycocotools does, instead of rejecting them.
    gt, dt = synthetic_bbox_datasets
    _assert_matches_pycocotools(
        gt,
        dt,
        "bbox",
        iouThrs=np.linspace(0.5, 0.95, 10, dtype=np.float32).astype(np.float64),
        recThrs=np.linspace(0.0, 1.0, 101, dtype=np.float32).astype(np.float64),
    )


def test_custom_iou_thrs_match_pycocotools(
    synthetic_bbox_datasets: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    gt, dt = synthetic_bbox_datasets
    _assert_matches_pycocotools(gt, dt, "bbox", iouThrs=np.array([0.5, 0.75]))


@pytest.mark.parametrize("max_dets", [[1, 10, 500], [2, 5, 20]])
def test_max_dets_without_100_match_pycocotools(
    synthetic_bbox_datasets: tuple[dict[str, Any], dict[str, Any]],
    max_dets: list[int],
) -> None:
    # Quirk L9 (strict): stats[0] reads maxDets=100 (-1 when absent) and
    # AR_1/AR_10/AR_100 read maxDets[0|1|2] positionally.
    gt, dt = synthetic_bbox_datasets
    evaluator = _assert_matches_pycocotools(gt, dt, "bbox", maxDets=max_dets)
    assert evaluator.stats[0] == -1.0
    assert evaluator.stats[8] > 0.0


def test_corrected_mode_reads_aggregate_ap_at_the_largest_max_det(
    synthetic_bbox_datasets: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    # Quirk L9 (corrected): stats[0] is AP at maxDets[-1], computed from
    # the same precision array the strict AP lines read.
    gt, dt = synthetic_bbox_datasets
    evaluator = COCOeval(_coco(gt), _coco(dt), iouType="bbox", parity_mode="corrected")
    evaluator.params.maxDets = [1, 10, 500]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    assert evaluator.stats[0] == np.mean(precision[precision > -1])
    assert evaluator.stats[0] > 0.0


def test_supplied_detection_area_matches_pycocotools() -> None:
    # Quirk J3: COCOeval buckets a detection by the `area` its cocoDt
    # carries. The top-scored false positive has a 25x25 box (small) but a
    # 60x60 mask and area (medium); reading the bbox area would count it
    # against AP_small and halve it.
    images = [{"id": 0, "width": 128, "height": 128}]
    categories = [{"id": 1, "name": "a"}]
    gt = {
        "images": images,
        "categories": categories,
        "annotations": [
            {
                "id": 1,
                "image_id": 0,
                "category_id": 1,
                "iscrowd": 0,
                "bbox": [0, 0, 20, 20],
                "area": 400.0,
                "segmentation": _rle(slice(0, 20), slice(0, 20)),
            }
        ],
    }
    dt = {
        "images": images,
        "categories": categories,
        "annotations": [
            {
                "id": 1,
                "image_id": 0,
                "category_id": 1,
                "score": 0.5,
                "bbox": [0, 0, 20, 20],
                "area": 625.0,
                "segmentation": _rle(slice(0, 25), slice(0, 25)),
            },
            {
                "id": 2,
                "image_id": 0,
                "category_id": 1,
                "score": 0.95,
                "bbox": [40, 40, 25, 25],
                "area": 3600.0,
                "segmentation": _rle(slice(40, 100), slice(40, 100)),
            },
        ],
    }
    # `mask_utils.encode` emits bytes counts, which the shim must also
    # serialize (quirk K3).
    assert isinstance(gt["annotations"][0]["segmentation"]["counts"], bytes)
    evaluator = _assert_matches_pycocotools(gt, dt, "segm")
    assert evaluator.stats[3] == pytest.approx(0.3)


def test_bbox_images_without_sizes_match_pycocotools(
    synthetic_bbox_datasets: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    # pycocotools reads image sizes only to rasterize segmentations.
    gt, dt = copy.deepcopy(synthetic_bbox_datasets)
    for image in gt["images"]:
        del image["width"], image["height"]
    gt_coco = _coco(gt)
    _assert_matches_pycocotools(gt, dt, "bbox")
    evaluator = COCOeval(gt_coco, _coco(dt), iouType="bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
    assert all("width" not in image for image in gt_coco.dataset["images"])


def _segm_datasets_with_sizeless_second_image() -> tuple[dict[str, Any], dict[str, Any]]:
    # Image 0 is sized and matched; image 1 has no size on the GT side,
    # the shape TorchMetrics gives an image with no GT masks.
    categories = [{"id": 1, "name": "a"}]
    gt = {
        "images": [{"id": 0, "width": 128, "height": 128}, {"id": 1}],
        "categories": categories,
        "annotations": [_mask_ann(1, 0, slice(0, 20), slice(0, 20), iscrowd=0)],
    }
    dt = {
        "images": [{"id": 0, "width": 128, "height": 128}, {"id": 1}],
        "categories": categories,
        "annotations": [_mask_ann(1, 0, slice(0, 22), slice(0, 20), score=0.9)],
    }
    return gt, dt


def test_segm_sizeless_image_without_annotations_matches_pycocotools() -> None:
    # annToRLE reads an image's size only for annotations on it, so an
    # image nothing points at never needs one.
    gt, dt = _segm_datasets_with_sizeless_second_image()
    _assert_matches_pycocotools(gt, dt, "segm")
    assert gt["images"][1] == {"id": 1}


def test_segm_sizeless_gt_image_takes_the_detection_image_size() -> None:
    # A DT mask on an image with no GT masks is converted against
    # cocoDt.imgs, which carries the size the GT image lacks.
    gt, dt = _segm_datasets_with_sizeless_second_image()
    dt["images"][1] = {"id": 1, "width": 128, "height": 128}
    dt["annotations"].append(_mask_ann(2, 1, slice(40, 60), slice(40, 60), score=0.95))
    evaluator = _assert_matches_pycocotools(gt, dt, "segm")
    assert 0.0 < evaluator.stats[0] < 1.0


def test_segm_images_without_sizes_raise(
    perfect_match_segm_coco: tuple[COCO, COCO],
) -> None:
    # A polygon GT on a sizeless image: pycocotools raises KeyError in
    # annToRLE, and a 0x0 fill would rasterize the polygon to nothing and
    # score silently.
    gt, dt = perfect_match_segm_coco
    dataset = cast(dict[str, Any], copy.deepcopy(gt.dataset))
    for image in dataset["images"]:
        del image["width"], image["height"]
    reference = PycocotoolsReferenceCOCOeval(_coco(dataset), dt, iouType="segm")
    with pytest.raises(KeyError, match="height"), contextlib.redirect_stdout(io.StringIO()):
        reference.evaluate()
    evaluator = COCOeval(_coco(dataset), dt, iouType="segm")
    with pytest.raises(ValueError, match="width"):
        evaluator.evaluate()


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


def _assert_ious_match(reference: Any, candidate: Any, *, exact: bool = True) -> None:
    # `COCOeval.ious` is `{(imgId, catId): (D, G) array}` with a bare `[]`
    # wherever one side of the pair is empty (quirk F5). Both the key set
    # and the empty/array split are part of the surface: a consumer that
    # branches on `len(...)` sees the difference.
    #
    # `exact=False` is for datasets whose coordinates are arbitrary
    # decimals. The kernel arithmetic is bit-identical to `bbIou` — the
    # array-ingest path reproduces pycocotools exactly on the same boxes
    # — but serde_json's default number parser is not correctly rounded,
    # so a coordinate can land one ULP off what CPython's `strtod`
    # produced and carry that into the quotient. Same root cause as the
    # `dtScores` drift; ADR-0054 turns on `float_roundtrip` and closes
    # it. Matching is unaffected here: the `eval` tensors these same
    # datasets produce compare bit-equal.
    assert set(candidate.ious) == set(reference.ious)
    for key, expected in reference.ious.items():
        actual = candidate.ious[key]
        assert np.shape(actual) == np.shape(expected), f"{key}: shape"
        if len(expected) == 0:
            assert isinstance(actual, list), f"{key}: empty pair must stay a bare list"
            continue
        if exact:
            np.testing.assert_array_equal(actual, expected, err_msg=f"{key}")
        else:
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0, err_msg=f"{key}")


def test_ious_match_pycocotools(
    synthetic_bbox_datasets: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    # TorchMetrics reads `coco_eval.ious` whenever `extended_summary=True`
    # (torchmetrics/detection/helpers.py), so it is drop-in surface.
    gt, dt = synthetic_bbox_datasets
    reference = _run_eval(PycocotoolsReferenceCOCOeval, gt, dt, "bbox")
    candidate = _run_eval(COCOeval, gt, dt, "bbox")
    _assert_ious_match(reference, candidate, exact=False)
    assert any(np.size(matrix) for matrix in candidate.ious.values())


def test_ious_match_pycocotools_under_segm(
    perfect_match_segm_coco: tuple[COCO, COCO],
) -> None:
    gt_coco, dt_coco = perfect_match_segm_coco
    gt = cast(dict[str, Any], gt_coco.dataset)
    dt = cast(dict[str, Any], dt_coco.dataset)
    reference = _run_eval(PycocotoolsReferenceCOCOeval, gt, dt, "segm")
    candidate = _run_eval(COCOeval, gt, dt, "segm")
    _assert_ious_match(reference, candidate)


def test_ious_collapse_onto_the_minus_one_key_without_use_cats(
    synthetic_bbox_datasets: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    # Quirk L4 / cocoeval.py:522: `catIds = p.catIds if p.useCats else [-1]`,
    # so a collapsed evaluation keys every pair on the sentinel.
    #
    # The G axis is compared as a set per row, not positionally. With
    # `useCats=0` pycocotools concatenates the cell's ground truths
    # category by category (`[_ for cId in p.catIds for _ in
    # self._gts[imgId, cId]]`, cocoeval.py:255) while vernier keeps them
    # in annotation order, so the two agree on the matrix up to a
    # permutation of its columns. The D axis is score-sorted on both
    # sides and does line up. This is a pre-existing property of the
    # collapsed gather that `evalImgs[...]["gtIds"]` already carried; it
    # is visible here rather than introduced here, and the `eval` tensors
    # for this dataset still compare bit-equal.
    gt, dt = synthetic_bbox_datasets
    reference = _run_eval(PycocotoolsReferenceCOCOeval, gt, dt, "bbox", useCats=0)
    candidate = _run_eval(COCOeval, gt, dt, "bbox", useCats=0)
    assert {cat for _, cat in candidate.ious} == {-1}
    assert set(candidate.ious) == set(reference.ious)
    for key, expected in reference.ious.items():
        actual = candidate.ious[key]
        assert np.shape(actual) == np.shape(expected), f"{key}: shape"
        if len(expected) == 0:
            assert isinstance(actual, list), f"{key}: empty pair must stay a bare list"
            continue
        np.testing.assert_allclose(
            np.sort(np.asarray(actual), axis=1),
            np.sort(np.asarray(expected), axis=1),
            rtol=1e-13,
            atol=0,
            err_msg=f"{key}",
        )


def test_ious_is_empty_before_evaluate(perfect_match_coco: tuple[COCO, COCO]) -> None:
    gt, dt = perfect_match_coco
    assert COCOeval(gt, dt, iouType="bbox").ious == {}


def test_optional_retention_is_off_until_an_attribute_is_read(
    perfect_match_coco: tuple[COCO, COCO],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `retain_meta` roughly doubles the per-cell allocations and
    # `retain_iou` holds an O(G x D) matrix per cell; the
    # evaluate → accumulate → summarize cycle reads neither, and most
    # callers (TorchMetrics among them) never touch `evalImgs` or `ious`.
    retentions: list[tuple[bool, bool]] = []
    build_grid = vernier_compat.evaluate_bbox_grid

    def recording(*args: Any, **kwargs: Any) -> Any:
        retentions.append((kwargs["retain_meta"], kwargs["retain_iou"]))
        return build_grid(*args, **kwargs)

    monkeypatch.setattr(vernier_compat, "evaluate_bbox_grid", recording)

    gt, dt = perfect_match_coco
    e = COCOeval(gt, dt, iouType="bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        e.evaluate()
        e.accumulate()
        e.summarize()
    assert retentions == [(False, False)]

    assert e.evalImgs
    assert retentions == [(False, False), (True, False)]

    assert e.ious
    # Widening to `retain_iou` re-evaluates once, with both retentions,
    # and every later read is served from the cache.
    assert retentions == [(False, False), (True, False), (True, True)]
    first_ious, first_eval_imgs = e.ious, e.evalImgs
    assert e.ious is first_ious
    assert e.evalImgs is first_eval_imgs
    assert retentions == [(False, False), (True, False), (True, True)]


@pytest.mark.parametrize("category_id", [1, 2])
def test_cat_ids_subsetting_matches_pycocotools(
    synthetic_bbox_datasets: tuple[dict[str, Any], dict[str, Any]],
    category_id: int,
) -> None:
    # `MeanAveragePrecision(class_metrics=True)` reuses one evaluator and
    # assigns `params.catIds = [class_id]` per class
    # (torchmetrics/detection/helpers.py). pycocotools implements this in
    # `_prepare`, which loads only `getAnnIds(catIds=p.catIds)`.
    gt, dt = synthetic_bbox_datasets
    _assert_matches_pycocotools(gt, dt, "bbox", catIds=[category_id])


def test_cat_ids_subsetting_matches_pycocotools_under_segm(
    perfect_match_segm_coco: tuple[COCO, COCO],
) -> None:
    gt_coco, dt_coco = perfect_match_segm_coco
    gt = cast(dict[str, Any], gt_coco.dataset)
    dt = cast(dict[str, Any], dt_coco.dataset)
    _assert_matches_pycocotools(gt, dt, "segm", catIds=[1])


def test_cat_ids_subsetting_for_an_absent_category_matches_pycocotools(
    synthetic_bbox_datasets: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    # pycocotools evaluates a category the dataset never declares as a
    # K axis of one with nothing in it, i.e. a row of -1s. Dropping it
    # instead would shorten the axis and renumber the rest.
    gt, dt = synthetic_bbox_datasets
    candidate = _assert_matches_pycocotools(gt, dt, "bbox", catIds=[9999])
    np.testing.assert_array_equal(candidate.stats, np.full(12, -1.0))


def test_one_evaluator_walks_the_class_loop(
    synthetic_bbox_datasets: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    # The exact shape TorchMetrics uses: construct once, re-assign
    # `params.catIds` and re-run the whole cycle per class. Each pass must
    # land where a fresh evaluator for that class would.
    gt, dt = synthetic_bbox_datasets
    shared = COCOeval(_coco(gt), _coco(dt), iouType="bbox")
    for category_id in (1, 2, 1):
        shared.params.catIds = [category_id]
        with contextlib.redirect_stdout(io.StringIO()):
            shared.evaluate()
            shared.accumulate()
            shared.summarize()
        expected = _run_eval(PycocotoolsReferenceCOCOeval, gt, dt, "bbox", catIds=[category_id])
        np.testing.assert_array_equal(shared.stats, expected.stats)
        _assert_ious_match(expected, shared, exact=False)


def test_ious_are_bit_exact_on_exactly_representable_boxes() -> None:
    # The `exact=False` arms above absorb a JSON-parse ULP, not a kernel
    # difference. On coordinates that are exact binary fractions — so
    # every parser agrees on the double — the IoU matrix is bit-identical
    # to pycocotools', which is what strict parity claims.
    images = [{"id": 0, "width": 128, "height": 128}]
    categories = [{"id": 1, "name": "a"}]
    boxes = [
        ([8.0, 8.0, 16.0, 16.0], [9.5, 8.25, 15.5, 17.0]),
        ([40.0, 12.5, 20.0, 10.0], [41.25, 13.0, 18.5, 11.5]),
        ([70.0, 70.0, 12.0, 12.0], [96.0, 96.0, 8.0, 8.0]),
    ]
    gt: dict[str, Any] = {"images": images, "categories": categories, "annotations": []}
    dt: dict[str, Any] = {"images": images, "categories": categories, "annotations": []}
    for index, (gt_box, dt_box) in enumerate(boxes):
        gt["annotations"].append(
            {
                "id": index + 1,
                "image_id": 0,
                "category_id": 1,
                "bbox": gt_box,
                "area": gt_box[2] * gt_box[3],
                "iscrowd": 0,
            }
        )
        dt["annotations"].append(
            {
                "id": index + 1,
                "image_id": 0,
                "category_id": 1,
                "bbox": dt_box,
                "area": dt_box[2] * dt_box[3],
                "score": 0.9 - 0.125 * index,
            }
        )
    reference = _run_eval(PycocotoolsReferenceCOCOeval, gt, dt, "bbox")
    candidate = _run_eval(COCOeval, gt, dt, "bbox")
    _assert_ious_match(reference, candidate)
    assert np.size(candidate.ious[0, 1]) == 9


# ---------------------------------------------------------------------------
# ADR-0057: the drop-in's detections go in as the caller's own list of
# result dicts, not as `json.dumps` bytes. That is a route change, and a
# route change has to be proven invisible — not on summary stats, which
# would survive a permuted id assignment, but cell for cell on
# `evalImgs`, which carries `dtIds`, `dtScores`, `dtMatches` and
# `dtIgnore`. Same shape of assertion as
# `tests/python/test_ingest_route_equivalence.py`, one level up: there
# the routes are compared at the FFI, here through the whole shim.
#
# The fixture below is built for that job. `perfect_match` and
# `perfect_match_segm` carry *one* detection on *one* image, which pins
# nothing about ordering — a route that reordered or renumbered the
# payload would evaluate identically — so they are not used here.
# ---------------------------------------------------------------------------

_ROUTE_IMAGES: list[dict[str, Any]] = [
    {"id": 1, "width": 128, "height": 128},
    {"id": 2, "width": 128, "height": 128},
]
_ROUTE_CATEGORIES: list[dict[str, Any]] = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]

# (id, image_id, category_id, x, y, w, h)
_ROUTE_GT_BOXES: tuple[tuple[int, int, int, int, int, int, int], ...] = (
    (1, 1, 1, 10, 10, 30, 30),
    (2, 1, 1, 60, 60, 40, 40),
    (3, 1, 2, 70, 20, 40, 30),
    (4, 2, 1, 5, 5, 40, 40),
    (5, 2, 2, 50, 50, 40, 40),
)


def _rect_poly(x: float, y: float, w: float, h: float) -> list[list[float]]:
    """The COCO polygon spelling of a rectangle: one flat, nested ring."""
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def _kp_at(x: float, y: float) -> list[float]:
    """The 17-keypoint skeleton translated to ``(x, y)``, all visible."""
    flat: list[float] = []
    for kx, ky in _KP_COORDS:
        flat.extend((kx + x, ky + y, 2.0))
    return flat


def _route_equivalence_gt() -> dict[str, Any]:
    """Five ground truths over two images and two categories.

    Every annotation carries a polygon *and* keypoints, so one dataset
    serves all four iou types.
    """
    return {
        "images": [dict(image) for image in _ROUTE_IMAGES],
        "categories": [dict(category) for category in _ROUTE_CATEGORIES],
        "annotations": [
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
                "segmentation": _rect_poly(x, y, w, h),
                "keypoints": _kp_at(x, y),
                "num_keypoints": 17,
            }
            for ann_id, image_id, category_id, x, y, w, h in _ROUTE_GT_BOXES
        ],
    }


def _route_equivalence_dt() -> dict[str, Any]:
    """Eight detections, and every pycocotools-shaped spelling at once.

    Ordering first: the list is unsorted by score, interleaved across
    both images, and carries two score ties *inside* one
    ``(image, category)`` cell each — ids 3/4 at 0.80 on image 1
    category 1, ids 2/7 at 0.55 on image 2 category 1. The sort is
    stable, so a tie is broken by input position, which is also what
    quirk **J1** numbers ``dtIds`` by: a route that handed the sorter a
    different order would show up cell for cell.

    Payload shapes second, because the two routes reach the same
    ``Vec<DetectionInput>`` through different readers and the spellings
    are where they could diverge:

    - ``bytes`` ``counts`` straight out of ``pycocotools.mask.encode``
      (ids 3, 4, 6) — quirk **K3**. The JSON route ASCII-decodes them to
      a ``str`` on the way through; the list route hands the ``bytes``
      to the RLE decoder untouched.
    - polygons (ids 2, 5, 7).
    - *no* ``segmentation`` at all (ids 1, 8): under ``segm`` /
      ``boundary`` that is quirk **J2**, a rectangle synthesized from
      the bbox, and it has to be J2 identically on both routes.
    - NumPy scalar ``score`` / ``category_id`` / ``area`` (id 4), which
      ``loadRes`` and ``mask.area`` leave behind.
    - a ``tuple`` ``bbox`` (id 2).
    - a supplied ``area`` on every entry, deliberately *not* the bbox
      area — the shim reads it rather than deriving it (quirk **J3**),
      so it moves detections between area buckets and the routes have to
      agree about that too.

    Scores are short decimals on purpose. vernier's JSON number parser
    can land one ULP off the ``float`` Python already holds (PR #265,
    real-data only) and the list route never goes through it, so a
    fixture carrying real prediction scores would make the
    exact-equality assertions below flaky in a way that says nothing
    about the routes.
    """
    annotations: list[dict[str, Any]] = [
        {
            # Top score in its cell, and no `segmentation` at all (J2).
            "id": 1,
            "image_id": 1,
            "category_id": 2,
            "score": 0.95,
            "bbox": [69.0, 21.0, 40.0, 30.0],
            "area": 1100.0,
            "keypoints": _kp_at(69, 21),
        },
        {
            # Polygon, and a `tuple` bbox.
            "id": 2,
            "image_id": 2,
            "category_id": 1,
            "score": 0.55,
            "bbox": (6.0, 4.0, 39.0, 41.0),
            "area": 1500.0,
            "segmentation": _rect_poly(6, 4, 39, 41),
            "keypoints": _kp_at(6, 4),
        },
        {
            # `bytes` counts, straight out of `mask.encode`.
            "id": 3,
            "image_id": 1,
            "category_id": 1,
            "score": 0.80,
            "bbox": [11.0, 9.0, 30.0, 31.0],
            "area": 900.0,
            "segmentation": _rle(slice(9, 40), slice(11, 41)),
            "keypoints": _kp_at(11, 9),
        },
        {
            # NumPy scalars where `loadRes` / `mask.area` leave them, and
            # a score tied with id 3 inside the same cell.
            "id": 4,
            "image_id": 1,
            "category_id": np.int64(1),
            "score": np.float64(0.80),
            "bbox": [59.0, 61.0, 41.0, 39.0],
            "area": np.float64(1500.0),
            "segmentation": _rle(slice(61, 100), slice(59, 100)),
            "keypoints": _kp_at(59, 61),
        },
        {
            "id": 5,
            "image_id": 2,
            "category_id": 2,
            "score": 0.31,
            "bbox": [52.0, 48.0, 36.0, 44.0],
            "area": 1000.0,
            "segmentation": _rect_poly(52, 48, 36, 44),
            "keypoints": _kp_at(52, 48),
        },
        {
            "id": 6,
            "image_id": 2,
            "category_id": 2,
            "score": 0.67,
            "bbox": [49.0, 51.0, 42.0, 38.0],
            "area": 1600.0,
            "segmentation": _rle(slice(51, 89), slice(49, 91)),
            "keypoints": _kp_at(49, 51),
        },
        {
            # Ties id 2 at 0.55, same (image, category) cell.
            "id": 7,
            "image_id": 2,
            "category_id": 1,
            "score": 0.55,
            "bbox": [8.0, 8.0, 36.0, 36.0],
            "area": 1300.0,
            "segmentation": _rect_poly(8, 8, 36, 36),
            "keypoints": _kp_at(8, 8),
        },
        {
            # No `segmentation` again (J2), lowest score in its cell.
            "id": 8,
            "image_id": 1,
            "category_id": 1,
            "score": 0.42,
            "bbox": [15.0, 15.0, 22.0, 22.0],
            "area": 500.0,
            "keypoints": _kp_at(15, 15),
        },
    ]
    return {
        "images": [dict(image) for image in _ROUTE_IMAGES],
        "categories": [dict(category) for category in _ROUTE_CATEGORIES],
        "annotations": annotations,
    }


@pytest.fixture(scope="module")
def route_equivalence_coco() -> tuple[COCO, COCO]:
    # `_coco` (no `loadRes`) is the point: it keeps the ids, the supplied
    # `area`, the `bytes` counts and the NumPy scalars exactly as written
    # above — which is what an in-memory caller such as TorchMetrics
    # hands the shim.
    return _coco(_route_equivalence_gt()), _coco(_route_equivalence_dt())


def test_route_equivalence_fixture_carries_the_shapes_it_claims() -> None:
    # The fixture's teeth are load-bearing and invisible at the call
    # site: `perfect_match_segm` reads as a mask fixture while carrying
    # only polygons, and the test that used it claimed `bytes` coverage
    # it did not have. Assert the claims rather than commenting them.
    annotations = _route_equivalence_dt()["annotations"]
    scores = [float(ann["score"]) for ann in annotations]
    cells = [(ann["image_id"], int(ann["category_id"])) for ann in annotations]

    assert len({ann["image_id"] for ann in annotations}) == 2
    assert scores != sorted(scores, reverse=True), "must be unsorted by score"
    assert any(
        scores[a] == scores[b] and cells[a] == cells[b]
        for a in range(len(annotations))
        for b in range(a + 1, len(annotations))
    ), "must carry a score tie inside one (image, category) cell"
    assert max(cells.count(cell) for cell in set(cells)) >= 3, "must carry a multi-detection cell"

    segmentations = [ann.get("segmentation") for ann in annotations]
    assert any(isinstance(s, dict) and isinstance(s["counts"], bytes) for s in segmentations)
    assert any(isinstance(s, list) for s in segmentations)
    assert any(s is None for s in segmentations), "J2 needs a DT with no segmentation"
    assert any(isinstance(ann["bbox"], tuple) for ann in annotations)
    assert any(isinstance(ann["score"], np.floating) for ann in annotations)
    assert any(isinstance(ann["category_id"], np.integer) for ann in annotations)
    assert all("area" in ann for ann in annotations)
    assert any(float(ann["area"]) != ann["bbox"][2] * ann["bbox"][3] for ann in annotations), (
        "a supplied area that differs from the bbox area is what exercises J3"
    )


class _BytesRouteCOCOeval(PycocotoolsCOCOeval):
    """The drop-in as it was: DT serialized to JSON before the grid.

    Overrides the one method that decides the detection payload, so
    everything downstream — params handling, `catIds` filtering, area
    resolution, retention widening — is the shipping code path, and the
    ingest route is the only difference between this and `COCOeval`.
    """

    def _prepare_inputs(self) -> tuple[bytes, Any]:
        gt_bytes, dt_anns = super()._prepare_inputs()
        return gt_bytes, vernier_compat.to_coco_json(dt_anns)


def _render_cells(eval_imgs: list[Any]) -> list[dict[str, Any]]:
    """`evalImgs` as plain comparable Python, in a stable order."""
    rendered: list[dict[str, Any]] = []
    for cell in eval_imgs:
        if cell is None:
            continue
        rendered.append(
            {
                key: np.asarray(value).tolist() if isinstance(value, np.ndarray) else value
                for key, value in cell.items()
            }
        )
    rendered.sort(key=lambda c: (c["image_id"], c["category_id"], str(c["aRng"]), c["maxDet"]))
    return rendered


def _assert_routes_agree(
    gt: COCO,
    dt: COCO,
    iou_type: str,
    *,
    reference_type: type[PycocotoolsCOCOeval] = _BytesRouteCOCOeval,
    **params: Any,
) -> PycocotoolsCOCOeval:
    native = COCOeval(gt, dt, iouType=iou_type)
    bytes_route = reference_type(gt, dt, iouType=iou_type)
    for name, value in params.items():
        setattr(native.params, name, copy.deepcopy(value))
        setattr(bytes_route.params, name, copy.deepcopy(value))
    with contextlib.redirect_stdout(io.StringIO()):
        for evaluator in (native, bytes_route):
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()

    # Exact equality, not `allclose`: the two routes converge on the same
    # `Vec<DetectionInput>`, so any difference at all is a real one. That
    # is only safe because the fixture's scores are short decimals — see
    # `_route_equivalence_dt` on the PR #265 parser drift.
    assert _render_cells(native.evalImgs) == _render_cells(bytes_route.evalImgs)
    np.testing.assert_array_equal(native.stats, bytes_route.stats)
    np.testing.assert_array_equal(native.eval["precision"], bytes_route.eval["precision"])
    np.testing.assert_array_equal(native.eval["recall"], bytes_route.eval["recall"])
    np.testing.assert_array_equal(native.eval["scores"], bytes_route.eval["scores"])
    assert native.ious.keys() == bytes_route.ious.keys()
    for key, matrix in native.ious.items():
        np.testing.assert_array_equal(np.asarray(matrix), np.asarray(bytes_route.ious[key]))
    return native


@pytest.mark.parametrize("iou_type", ["bbox", "segm", "boundary", "keypoints"])
def test_native_dt_route_matches_the_bytes_route_cell_for_cell(
    route_equivalence_coco: tuple[COCO, COCO], iou_type: str
) -> None:
    gt, dt = route_equivalence_coco
    _assert_routes_agree(gt, dt, iou_type)


@pytest.mark.parametrize("category_id", [1, 2])
def test_native_dt_route_matches_the_bytes_route_under_cat_ids(
    route_equivalence_coco: tuple[COCO, COCO], category_id: int
) -> None:
    # `params.catIds` subsetting is unchanged by ADR-0057 — the same
    # comprehension over the same dicts it always was — but it is the one
    # place the shim hands the route a list it built rather than the
    # caller's own, so the survivors are worth pinning cell for cell.
    gt, dt = route_equivalence_coco
    _assert_routes_agree(gt, dt, "segm", catIds=[category_id])


@pytest.mark.parametrize("category_id", [1, 2])
def test_native_dt_route_matches_the_bytes_route_on_synthetic_cat_ids(
    synthetic_bbox_datasets: tuple[dict[str, Any], dict[str, Any]], category_id: int
) -> None:
    # Eighty jittered detections over four images: fractional AP on every
    # summary line, so a route divergence has nowhere flat to hide.
    gt, dt = synthetic_bbox_datasets
    _assert_routes_agree(_coco(gt), _coco(dt), "bbox", catIds=[category_id])


def test_native_dt_route_holds_the_callers_list_without_copying_it(
    perfect_match_coco: tuple[COCO, COCO],
) -> None:
    # Holding a reference is the point — a deep copy would give back the
    # memory the route exists to save. Evaluating every category needs
    # no filtering, so the shim keeps the caller's list itself.
    gt, dt = perfect_match_coco
    e = COCOeval(gt, dt, iouType="bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        e.evaluate()
    assert e._dt_anns is dt.dataset["annotations"]


def test_cat_ids_subsetting_filters_in_memory_without_copying_annotations(
    synthetic_bbox_datasets: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    # A `catIds` subset is a new list of the *same* dicts — the filter
    # `COCOeval._prepare` applies, done in memory rather than by
    # serializing the survivors.
    gt_dict, dt_dict = synthetic_bbox_datasets
    gt, dt = _coco(gt_dict), _coco(dt_dict)
    e = COCOeval(gt, dt, iouType="bbox")
    e.params.catIds = [1]
    with contextlib.redirect_stdout(io.StringIO()):
        e.evaluate()
    held = e._dt_anns
    annotations = dt.dataset["annotations"]
    assert held is not annotations
    assert [id(ann) for ann in held] == [id(ann) for ann in annotations if ann["category_id"] == 1]


def test_retention_widening_works_off_the_held_list(
    perfect_match_coco: tuple[COCO, COCO],
) -> None:
    # `evalImgs` / `ious` re-evaluate the grid at a wider retention
    # level, which means re-ingesting the detections. That has to work
    # off the held list, since nothing serializes them any more.
    gt, dt = perfect_match_coco
    e = COCOeval(gt, dt, iouType="bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        e.evaluate()
        e.accumulate()
        e.summarize()
    assert e.evalImgs
    assert e.ious
    assert not hasattr(e, "_dt_bytes")


def test_detections_mutated_after_evaluate_reach_the_lazy_attributes() -> None:
    # The aliasing window documented in `vernier._compat`'s module
    # docstring, pinned so it cannot change silently: `stats` is frozen
    # at `evaluate()` time, but `evalImgs` / `ious` re-ingest the held
    # list on first read, so a detection mutated in place afterwards
    # reaches them and not `stats`. It is the cost of holding the
    # caller's list instead of a snapshot of it — which is the whole
    # memory win, so the answer is to document it, not to copy.
    gt = _coco(_route_equivalence_gt())
    dt = _coco(_route_equivalence_dt())
    # pycocotools' stubs type an annotation as its file-loaded
    # TypedDict; a detection result dict carries `score` too.
    held: list[dict[str, Any]] = cast(Any, dt.dataset["annotations"])

    e = COCOeval(gt, dt, iouType="bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        e.evaluate()
        e.accumulate()
        e.summarize()
    frozen_stats = e.stats.copy()

    held[0]["score"] = 0.01
    assert np.array_equal(e.stats, frozen_stats)
    assert 0.01 in [score for cell in e.evalImgs if cell for score in cell["dtScores"]]

    e2 = COCOeval(gt, dt, iouType="bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        e2.evaluate()
    held.clear()
    assert all(not len(cell["dtIds"]) for cell in e2.evalImgs if cell)


class _ScoreReversedBytesRouteCOCOeval(_BytesRouteCOCOeval):
    """`_BytesRouteCOCOeval` with the score column reversed.

    The negative control for :func:`_assert_routes_agree`: an assertion
    that never fails proves nothing, so one deliberate divergence is
    driven through the same harness.
    """

    def _prepare_inputs(self) -> tuple[bytes, Any]:
        gt_bytes, dt_anns = PycocotoolsCOCOeval._prepare_inputs(self)
        scores = [float(ann["score"]) for ann in dt_anns]
        reversed_scores = [{**ann, "score": score} for ann, score in zip(dt_anns, reversed(scores))]
        return gt_bytes, vernier_compat.to_coco_json(reversed_scores)


@pytest.mark.parametrize("iou_type", ["bbox", "segm", "boundary", "keypoints"])
def test_route_equivalence_harness_fails_on_a_perturbed_payload(
    route_equivalence_coco: tuple[COCO, COCO], iou_type: str
) -> None:
    # Measured on this fixture, reversing the score column: 6 of the 7
    # assertions in `_assert_routes_agree` fail under bbox / segm /
    # boundary (`evalImgs`, `stats`, `precision`, `recall`, `scores`,
    # `ious`) and 5 under keypoints, where `recall` happens to survive.
    # `ious.keys()` is the seventh and cannot fail here — the key axis is
    # `imgIds x catIds` and no score touches it — so 6 is the ceiling for
    # a score perturbation, not a shortfall.
    #
    # Two weaker perturbations, for calibration: a *monotone* rescale
    # (`s -> s/2 + 0.01`), which changes values but not the ranking,
    # fails only 2 of 7 (`evalImgs`, `scores`); Gaussian jitter at
    # sigma=0.2 fails 5 of 7. And on a one-detection fixture such as
    # `perfect_match`, reversal fails **0** of 7 — there is nothing to
    # reorder. That is why this fixture exists.
    gt, dt = route_equivalence_coco
    with pytest.raises(AssertionError):
        _assert_routes_agree(gt, dt, iou_type, reference_type=_ScoreReversedBytesRouteCOCOeval)
