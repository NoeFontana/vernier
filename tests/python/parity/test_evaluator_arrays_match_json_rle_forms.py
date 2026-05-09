"""Parity oracle for the broadened ``Detections.rles`` ingest.

The two new forms (compressed dict, 2-D bitmask in either C- or F-order)
must produce byte-identical ``Summary.stats`` to the JSON-bytes path.
The existing uncompressed-dict form is already covered by
``test_evaluator_arrays_match_json.py``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vernier.instance import Evaluator, Segm

from .conftest import (
    _segmentation_to_bitmask,
    _segmentation_to_compressed_rle,
)
from .test_parity import SEGM_FIXTURES

FIXTURES = Path(__file__).parent / "fixtures"

# ``heterogeneous_dt_segm`` is a corrected-mode rejection fixture; skip.
_SEGM_FIXTURES = [f for f in SEGM_FIXTURES if f != "heterogeneous_dt_segm"]
_FIXTURE = _SEGM_FIXTURES[0]

RleFactory = Callable[[Any, int, int], Any]


@pytest.fixture(scope="module")
def fixture_data() -> tuple[bytes, bytes, dict[str, Any], list[dict[str, Any]]]:
    gt_bytes = (FIXTURES / _FIXTURE / "gt.json").read_bytes()
    dt_bytes = (FIXTURES / _FIXTURE / "dt.json").read_bytes()
    return gt_bytes, dt_bytes, json.loads(gt_bytes), json.loads(dt_bytes)


def _build_detections(
    gt_records: dict[str, Any],
    dt_records: list[dict[str, Any]],
    rle_factory: Callable[[Any, int, int, int], Any],
) -> list[dict[str, Any]]:
    """Mirror the segm path of ``loadres_to_detections`` but apply
    ``rle_factory(index, segmentation, h, w)`` per detection. Returns
    plain dicts so the tests stay independent of TypedDict tightening."""
    image_dims = {
        int(im["id"]): (int(im["height"]), int(im["width"])) for im in gt_records["images"]
    }
    by_image: dict[int, list[dict[str, Any]]] = {}
    for r in dt_records:
        by_image.setdefault(int(r["image_id"]), []).append(r)
    out: list[dict[str, Any]] = []
    for image_id in sorted(by_image.keys()):
        dets = by_image[image_id]
        h, w = image_dims[image_id]
        out.append(
            {
                "image_id": image_id,
                "boxes": np.asarray(
                    [[float(x) for x in d["bbox"]] for d in dets], dtype=np.float64
                ),
                "scores": np.asarray([float(d["score"]) for d in dets], dtype=np.float64),
                "labels": np.asarray([int(d["category_id"]) for d in dets], dtype=np.int64),
                "rles": [rle_factory(j, d["segmentation"], h, w) for j, d in enumerate(dets)],
            }
        )
    return out


def _assert_stats_match(s_json: Any, s_arr: Any) -> None:
    for i, (j, a) in enumerate(zip(s_json.stats, s_arr.stats, strict=True)):
        assert a == pytest.approx(j, rel=0, abs=1e-12), f"stat[{i}] diverged: json={j!r} arr={a!r}"


def _ignore_index(factory: RleFactory) -> Callable[[int, Any, int, int], Any]:
    return lambda _i, seg, h, w: factory(seg, h, w)


_FORM_FACTORIES: list[tuple[str, RleFactory]] = [
    ("compressed_dict", _segmentation_to_compressed_rle),
    ("bitmask_c_order", lambda seg, h, w: _segmentation_to_bitmask(seg, h, w, order="C")),
    ("bitmask_f_order", lambda seg, h, w: _segmentation_to_bitmask(seg, h, w, order="F")),
]


@pytest.mark.parity
@pytest.mark.parametrize(("form", "factory"), _FORM_FACTORIES, ids=[f for f, _ in _FORM_FACTORIES])
def test_form_matches_json(
    form: str,
    factory: RleFactory,
    fixture_data: tuple[bytes, bytes, dict[str, Any], list[dict[str, Any]]],
) -> None:
    gt_bytes, dt_bytes, gt_records, dt_records = fixture_data
    ev = Evaluator(iou=Segm(), parity_mode="strict")
    s_json = ev.evaluate(gt_bytes, dt_bytes)
    detections = _build_detections(gt_records, dt_records, _ignore_index(factory))
    if not detections:
        pytest.skip("fixture has no detections")
    s_arr = ev.evaluate(gt_bytes, detections)  # type: ignore[arg-type]
    _assert_stats_match(s_json, s_arr)


@pytest.mark.parity
def test_mixed_rle_forms_in_one_sequence_match_json(
    fixture_data: tuple[bytes, bytes, dict[str, Any], list[dict[str, Any]]],
) -> None:
    """Round-robin the three forms across detections in the same image —
    the per-item dispatcher must classify each independently."""
    gt_bytes, dt_bytes, gt_records, dt_records = fixture_data
    ev = Evaluator(iou=Segm(), parity_mode="strict")
    s_json = ev.evaluate(gt_bytes, dt_bytes)

    def factory(j: int, seg: Any, h: int, w: int) -> Any:
        match j % 3:
            case 0:
                return _segmentation_to_compressed_rle(seg, h, w)
            case 1:
                return _segmentation_to_bitmask(seg, h, w, order="C")
            case _:
                return _segmentation_to_bitmask(seg, h, w, order="F")

    detections = _build_detections(gt_records, dt_records, factory)
    if not detections:
        pytest.skip("fixture has no detections")
    s_arr = ev.evaluate(gt_bytes, detections)  # type: ignore[arg-type]
    _assert_stats_match(s_json, s_arr)
