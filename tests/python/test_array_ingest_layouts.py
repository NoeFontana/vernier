"""Array-ingest layout contract (ADR-0030): which buffers count as contiguous.

Contiguity follows NumPy's flags and ``torch.Tensor.is_contiguous``: a
size-1 axis's stride is never read, and an array with no elements is
contiguous whatever its strides. Detector output routinely has both shapes
(one detection, or none, on an image), so rejecting them would reject
ordinary per-image batches.
"""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import pytest
from numpy.lib.stride_tricks import as_strided

from vernier.instance import Bbox, Detections, Evaluator

_GT = json.dumps(
    {
        "images": [
            {"id": 1, "width": 100, "height": 100},
            {"id": 2, "width": 100, "height": 100},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 50, 50],
                "area": 2500,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "widget"}],
    }
).encode()

_DT_JSON = json.dumps(
    [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "score": 0.9}]
).encode()


def _stats(dt: Any) -> list[float]:
    return Evaluator(iou=Bbox(), parity_mode="strict").evaluate(_GT, dt).stats


def _one_detection(boxes: Any) -> Detections:
    return cast(
        Detections,
        {
            "image_id": 1,
            "boxes": boxes,
            "scores": np.array([0.9]),
            "labels": np.array([1], dtype=np.int64),
        },
    )


def _no_detections(boxes: Any, scores: Any, labels: Any) -> Detections:
    return cast(Detections, {"image_id": 2, "boxes": boxes, "scores": scores, "labels": labels})


def test_single_row_with_an_arbitrary_row_stride_is_accepted() -> None:
    base = np.array([[10.0, 10.0, 50.0, 50.0]])
    boxes = as_strided(base, shape=(1, 4), strides=(24, 8))
    assert boxes.flags.c_contiguous
    assert _stats([_one_detection(boxes)]) == _stats(_DT_JSON)


def test_images_without_detections_are_accepted() -> None:
    empty = _no_detections(
        as_strided(np.empty((0, 4)), shape=(0, 4), strides=(0, 0)),
        as_strided(np.empty(0), shape=(0,), strides=(0,)),
        as_strided(np.empty(0, dtype=np.int64), shape=(0,), strides=(0,)),
    )
    one = _one_detection(np.array([[10.0, 10.0, 50.0, 50.0]]))
    assert _stats([one, empty]) == _stats(_DT_JSON)
    assert _stats([empty]) == _stats(b"[]")


def test_torch_tensors_from_detector_state_are_accepted() -> None:
    # `tensor.double().contiguous()` as TorchMetrics state hands it over;
    # an empty tensor exports a null data pointer.
    torch = pytest.importorskip("torch")
    one = _one_detection(torch.tensor([[10.0, 10.0, 50.0, 50.0]]).double().contiguous())
    empty = _no_detections(
        torch.zeros((0, 4), dtype=torch.float64),
        torch.zeros(0, dtype=torch.float64),
        torch.zeros(0, dtype=torch.int64),
    )
    assert _stats([one, empty]) == _stats(_DT_JSON)


def test_stepped_arrays_are_still_rejected() -> None:
    scores = np.array([0.9, 0.0, 0.8, 0.0])[::2]
    assert not scores.flags.c_contiguous
    payload = cast(
        Detections,
        {
            "image_id": 1,
            "boxes": np.array([[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 50.0, 50.0]]),
            "scores": scores,
            "labels": np.array([1, 1], dtype=np.int64),
        },
    )
    with pytest.raises(TypeError, match="not C-contiguous"):
        _stats([payload])
