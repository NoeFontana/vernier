"""ADR-0030 array-ingest parity oracle.

For each fixture, the array-form ``StreamingEvaluator.update(...)`` must
produce a byte-identical ``Summary.stats`` to the legacy JSON-bytes path.
The two paths share every line of code from
``CocoDetections::from_inputs`` onward; this test pins that the array
path's translation to ``DetectionInput`` records is faithful.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import pytest

from vernier.instance import StreamingEvaluator

from ..conftest import loadres_to_detections
from ..test_parity import BBOX_FIXTURES, KEYPOINTS_FIXTURES, SEGM_FIXTURES

IouType = Literal["bbox", "segm", "boundary", "keypoints"]

FIXTURES = Path(__file__).parent.parent / "fixtures"

# `heterogeneous_dt_segm` is intentionally omitted — it's a corrected-mode
# rejection fixture that errors before reaching either ingest path.
_SEGM_FIXTURES = [f for f in SEGM_FIXTURES if f != "heterogeneous_dt_segm"]


def _kp_sigmas(
    sigmas: Mapping[int, tuple[float, ...]] | None,
) -> dict[int, list[float]] | None:
    return None if sigmas is None else {k: list(v) for k, v in sigmas.items()}


_PARITY_CASES: list[tuple[str, IouType, dict[int, list[float]] | None]] = [
    *((f, "bbox", None) for f in BBOX_FIXTURES),
    *((f, "segm", None) for f in _SEGM_FIXTURES),
    *((f, "boundary", None) for f in _SEGM_FIXTURES),
    *((f, "keypoints", _kp_sigmas(sigmas)) for f, sigmas in KEYPOINTS_FIXTURES),
]


@pytest.mark.parity
@pytest.mark.parametrize(("fixture", "iou_type", "sigmas"), _PARITY_CASES)
def test_array_ingest_matches_json_bytes(
    fixture: str,
    iou_type: IouType,
    sigmas: dict[int, list[float]] | None,
) -> None:
    gt_path = FIXTURES / fixture / "gt.json"
    dt_path = FIXTURES / fixture / "dt.json"
    gt_bytes = gt_path.read_bytes()
    dt_bytes = dt_path.read_bytes()

    gt_records = json.loads(gt_bytes)
    dt_records = json.loads(dt_bytes)

    kwargs: dict[str, Any] = {"iou_type": iou_type, "parity_mode": "strict"}
    if sigmas is not None:
        kwargs["sigmas"] = sigmas

    ev_json = StreamingEvaluator(gt_bytes, **kwargs)
    ev_json.update(dt_bytes)
    s_json = ev_json.finalize()

    detections = loadres_to_detections(gt_records, dt_records, iou_type)
    ev_arr = StreamingEvaluator(gt_bytes, **kwargs)
    if detections:
        ev_arr.update(detections)
    s_arr = ev_arr.finalize()

    assert len(s_json.stats) == len(s_arr.stats)
    for i, (j, a) in enumerate(zip(s_json.stats, s_arr.stats, strict=True)):
        assert a == pytest.approx(j, rel=0, abs=1e-12), (
            f"stat[{i}] diverged: json={j!r} arr={a!r} (fixture={fixture}, iou_type={iou_type})"
        )
