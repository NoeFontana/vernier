"""A federated ``CocoDataset`` streams to the batch evaluator's numbers (ADR-0065).

The ADR-0026 AC2 per-image cap is applied per ``submit()``. That is the
whole-dataset trim because an image's detections must arrive in one batch.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from numpy.typing import NDArray

from vernier.instance import Bbox, CocoDataset, Evaluator

_IMAGES = (1, 2)


def _federated_handle() -> CocoDataset:
    images: list[dict[str, object]] = [
        {
            "id": i,
            "width": 100,
            "height": 100,
            "neg_category_ids": [],
            "not_exhaustive_category_ids": [],
        }
        for i in _IMAGES
    ]
    annotations: list[dict[str, object]] = [
        {"id": 10 * i + c, "image_id": i, "category_id": c, "bbox": box, "area": 100, "iscrowd": 0}
        for i in _IMAGES
        for c, box in ((1, [0, 0, 10, 10]), (2, [50, 50, 10, 10]))
    ]
    categories = [
        {"id": 1, "name": "a", "frequency": "f"},
        {"id": 2, "name": "b", "frequency": "f"},
    ]
    gt: dict[str, object] = {"images": images, "annotations": annotations, "categories": categories}
    return CocoDataset.from_lvis_json(json.dumps(gt).encode())


def _crowded(image_id: int) -> NDArray[np.float64]:
    """100 category-1 false positives outscoring the image's only
    category-2 true positive, which the cap of 100 therefore drops."""
    fps = [[image_id, 80, 80, 5, 5, 0.99 - k * 1e-3, 1] for k in range(100)]
    return np.asarray([*fps, [image_id, 50, 50, 10, 10, 0.01, 2]], dtype=np.float64)


@pytest.mark.parametrize("num_threads", [None, 2], ids=["serial", "parallel"])
def test_streamed_federated_handle_matches_the_batch_evaluator(num_threads: int | None) -> None:
    handle = _federated_handle()
    ev = Evaluator(iou=Bbox())
    batch = ev.evaluate(handle, np.concatenate([_crowded(i) for i in _IMAGES]))

    with ev.background(handle, num_threads=num_threads) as bg:
        for i in _IMAGES:
            bg.submit(_crowded(i))
        streamed = bg.finalize()

    assert streamed.stats == batch.stats
    assert batch.stats[0] == 0.0
