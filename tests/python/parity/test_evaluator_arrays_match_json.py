"""ADR-0030 array-ingest parity oracle for the foreground evaluator.

For each fixture, ``Evaluator.evaluate(...)`` with array-form ``Detections``
must produce a byte-identical ``Summary.stats`` to the JSON-bytes path —
both the no-tables and the ``tables=`` route, with and without a
parsed-once ``CocoDataset``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest

from vernier.instance import (
    Bbox,
    Boundary,
    CocoDataset,
    Evaluator,
    Keypoints,
    Segm,
)

from .conftest import loadres_to_detections
from .test_parity import BBOX_FIXTURES, KEYPOINTS_FIXTURES, SEGM_FIXTURES

IouType = Literal["bbox", "segm", "boundary", "keypoints"]

FIXTURES = Path(__file__).parent / "fixtures"

# `heterogeneous_dt_segm` is a corrected-mode rejection fixture; skip.
_SEGM_FIXTURES = [f for f in SEGM_FIXTURES if f != "heterogeneous_dt_segm"]


def _iou_kind_for(
    iou_type: IouType, sigmas: Mapping[int, tuple[float, ...]] | None
) -> Bbox | Segm | Boundary | Keypoints:
    match iou_type:
        case "bbox":
            return Bbox()
        case "segm":
            return Segm()
        case "boundary":
            return Boundary()
        case "keypoints":
            return Keypoints(sigmas=sigmas or {})


_PARITY_CASES: list[tuple[str, IouType, Mapping[int, tuple[float, ...]] | None]] = [
    *((f, "bbox", None) for f in BBOX_FIXTURES),
    *((f, "segm", None) for f in _SEGM_FIXTURES),
    *((f, "boundary", None) for f in _SEGM_FIXTURES),
    *((f, "keypoints", sigmas) for f, sigmas in KEYPOINTS_FIXTURES),
]


@pytest.mark.parity
@pytest.mark.parametrize(("fixture", "iou_type", "sigmas"), _PARITY_CASES)
def test_evaluator_array_dt_matches_json(
    fixture: str,
    iou_type: IouType,
    sigmas: Mapping[int, tuple[float, ...]] | None,
) -> None:
    gt_path = FIXTURES / fixture / "gt.json"
    dt_path = FIXTURES / fixture / "dt.json"
    gt_bytes = gt_path.read_bytes()
    dt_bytes = dt_path.read_bytes()

    gt_records = json.loads(gt_bytes)
    dt_records = json.loads(dt_bytes)

    iou = _iou_kind_for(iou_type, sigmas)
    ev = Evaluator(iou=iou, parity_mode="strict")

    s_json = ev.evaluate(gt_bytes, dt_bytes)

    detections = loadres_to_detections(gt_records, dt_records, iou_type)
    if not detections:
        pytest.skip("fixture has no detections; array path is a no-op input")
    s_arr = ev.evaluate(gt_bytes, detections)

    assert len(s_json.stats) == len(s_arr.stats)
    for i, (j, a) in enumerate(zip(s_json.stats, s_arr.stats, strict=True)):
        assert a == pytest.approx(j, rel=0, abs=1e-12), (
            f"stat[{i}] diverged: json={j!r} arr={a!r} (fixture={fixture}, iou_type={iou_type})"
        )


@pytest.mark.parity
@pytest.mark.parametrize(("fixture", "iou_type", "sigmas"), _PARITY_CASES)
def test_evaluator_array_dt_matches_json_with_dataset(
    fixture: str,
    iou_type: IouType,
    sigmas: Mapping[int, tuple[float, ...]] | None,
) -> None:
    gt_path = FIXTURES / fixture / "gt.json"
    dt_path = FIXTURES / fixture / "dt.json"
    gt_bytes = gt_path.read_bytes()
    dt_bytes = dt_path.read_bytes()

    gt_records = json.loads(gt_bytes)
    dt_records = json.loads(dt_bytes)

    iou = _iou_kind_for(iou_type, sigmas)
    ev = Evaluator(iou=iou, parity_mode="strict")
    dataset = CocoDataset.from_json(gt_bytes)

    s_json = ev.evaluate(dataset, dt_bytes)

    detections = loadres_to_detections(gt_records, dt_records, iou_type)
    if not detections:
        pytest.skip("fixture has no detections; array path is a no-op input")
    s_arr = ev.evaluate(dataset, detections)

    assert len(s_json.stats) == len(s_arr.stats)
    for i, (j, a) in enumerate(zip(s_json.stats, s_arr.stats, strict=True)):
        assert a == pytest.approx(j, rel=0, abs=1e-12), (
            f"stat[{i}] diverged: json={j!r} arr={a!r} (fixture={fixture}, iou_type={iou_type})"
        )


@pytest.mark.parity
def test_evaluator_array_dt_with_tables_per_detection() -> None:
    """``tables=('per_detection',)`` reads dt once: the grid retains
    the parsed `CocoDetections` for the per_detection builder, so the
    array path doesn't pay a second DLPack walk + RLE-counts copy."""
    fixture = BBOX_FIXTURES[0]
    gt_bytes = (FIXTURES / fixture / "gt.json").read_bytes()
    dt_bytes = (FIXTURES / fixture / "dt.json").read_bytes()
    gt_records = json.loads(gt_bytes)
    dt_records = json.loads(dt_bytes)

    detections = loadres_to_detections(gt_records, dt_records, "bbox")
    if not detections:
        pytest.skip("fixture has no detections")

    ev = Evaluator(iou=Bbox(), parity_mode="strict")

    json_result = ev.evaluate(gt_bytes, dt_bytes, tables=("per_detection",))
    arr_result = ev.evaluate(gt_bytes, detections, tables=("per_detection",))

    # Default-grid evaluator → summary is always populated (the
    # ADR-0040 custom-grid `summary=None` path doesn't apply here).
    assert json_result.summary is not None
    assert arr_result.summary is not None
    for i, (j, a) in enumerate(
        zip(json_result.summary.stats, arr_result.summary.stats, strict=True)
    ):
        assert a == pytest.approx(j, rel=0, abs=1e-12), f"stat[{i}] diverged"

    assert arr_result.per_detection is not None
    assert json_result.per_detection is not None


def _downcast(detections: list[Any]) -> list[Any]:
    """f64/i64 array ``Detections`` -> f32/i32, which the strict path
    rejects and ``cast_inputs=True`` promotes back.

    `Detections` is total=False at the type level so accessor narrowing
    requires `.get(...)`; every kernel carries the first four fields at
    runtime, and `keypoints` rides along when the fixture is a keypoints
    one.
    """
    out: list[Any] = []
    for d in detections:
        d_any: dict[str, Any] = dict(d)
        cast: dict[str, Any] = {
            "image_id": d_any["image_id"],
            "boxes": np.asarray(d_any["boxes"], dtype=np.float32),
            "scores": np.asarray(d_any["scores"], dtype=np.float32),
            "labels": np.asarray(d_any["labels"], dtype=np.int32),
        }
        if "keypoints" in d_any:
            cast["keypoints"] = np.asarray(d_any["keypoints"], dtype=np.float32)
        out.append(cast)
    return out


@pytest.mark.parity
def test_evaluator_cast_inputs_promotes_dtypes() -> None:
    """``cast_inputs=True`` accepts f32/i32 inputs by silently promoting
    via numpy.ascontiguousarray; the strict default rejects them."""
    fixture = BBOX_FIXTURES[0]
    gt_bytes = (FIXTURES / fixture / "gt.json").read_bytes()
    dt_bytes = (FIXTURES / fixture / "dt.json").read_bytes()
    gt_records = json.loads(gt_bytes)
    dt_records = json.loads(dt_bytes)
    detections = loadres_to_detections(gt_records, dt_records, "bbox")
    if not detections:
        pytest.skip("fixture has no detections")

    cast_dets = _downcast(detections)

    ev_strict = Evaluator(iou=Bbox(), parity_mode="strict", cast_inputs=False)
    with pytest.raises(TypeError):
        ev_strict.evaluate(gt_bytes, cast_dets)

    ev_cast = Evaluator(iou=Bbox(), parity_mode="strict", cast_inputs=True)
    with pytest.warns(UserWarning, match="vernier-0030"):
        s_cast = ev_cast.evaluate(gt_bytes, cast_dets)

    s_json = ev_strict.evaluate(gt_bytes, dt_bytes)
    for i, (j, a) in enumerate(zip(s_json.stats, s_cast.stats, strict=True)):
        assert a == pytest.approx(j, rel=0, abs=1e-6), f"stat[{i}] diverged: {j} vs {a}"


@pytest.mark.parity
def test_keypoints_calibration_path_honours_cast_inputs() -> None:
    """Regression: the keypoints branch of the tables/calibration path
    must forward ``cast_inputs``.

    It used to bind the flag to the wrong parameter --- ``retain_iou=
    self.cast_inputs``, with ``cast_inputs`` never passed at all. The
    three sibling kernels were correct, so nothing caught it: keypoints
    rejects ``tables=``, which leaves ``calibration=True`` as the only
    way into this branch, and no test took it with ``cast_inputs=True``.
    The symptom was the exact failure the flag exists to suppress, plus
    IoU matrices silently retained for nobody.
    """
    gt_bytes = (FIXTURES / "keypoints_perfect_match" / "gt.json").read_bytes()
    dt_bytes = (FIXTURES / "keypoints_perfect_match" / "dt.json").read_bytes()
    cast_dets = _downcast(
        loadres_to_detections(json.loads(gt_bytes), json.loads(dt_bytes), "keypoints")
    )

    strict = Evaluator(iou=Keypoints(), cast_inputs=False)
    with pytest.raises(TypeError):
        strict.evaluate(gt_bytes, cast_dets, calibration=True)

    casting = Evaluator(iou=Keypoints(), cast_inputs=True)
    with pytest.warns(UserWarning, match="vernier-0030"):
        result = casting.evaluate(gt_bytes, cast_dets, calibration=True)

    assert result.summary is not None
    # `retain_iou` is driven by the requested tables, and keypoints
    # rejects `tables=` -- so nothing here asked the spine to hold its
    # IoU matrices, and the accessor says so rather than handing back a
    # table nobody requested.
    with pytest.raises(RuntimeError, match="per_detection"):
        _ = result.per_detection


@pytest.mark.parity
def test_keypoints_calibration_summary_matches_the_batch_path() -> None:
    """Regression: the tables/calibration path must summarize keypoints
    with the keypoints plan.

    The batch path reaches its summary through
    ``evaluate_keypoints_summary``, which picks the plan from the
    kernel; the calibration path summarizes the grid directly and was
    defaulting to the detection plan. Keypoints has a 3-bucket area grid
    (ADR-0012) and the detection plan indexes AreaRng 3, so
    ``Evaluator(iou=Keypoints()).evaluate(gt, dt, calibration=True)``
    raised ``ValueError: AreaRng index 3 is out of range`` for every
    input -- the path was unreachable, not merely wrong.
    """
    gt_bytes = (FIXTURES / "keypoints_perfect_match" / "gt.json").read_bytes()
    dt_bytes = (FIXTURES / "keypoints_perfect_match" / "dt.json").read_bytes()

    ev = Evaluator(iou=Keypoints())
    batch = ev.evaluate(gt_bytes, dt_bytes)
    calibrated = ev.evaluate(gt_bytes, dt_bytes, calibration=True)

    assert calibrated.summary is not None
    # The 10-stat keypoints plan, not the 12-stat detection one.
    assert len(batch.stats) == 10
    assert list(calibrated.summary.stats) == list(batch.stats)
