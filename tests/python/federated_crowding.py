"""An LVIS scene the ADR-0026 AC2 per-image cap changes the score of.

Each image has a category-1 GT at the origin and a category-2 GT at
(50, 50). ``crowded`` puts 100 category-1 false positives above the
image's only category-2 true positive, so the default cap of 100 drops
the true positive: capped, AP is 0; uncapped, category 2 scores AP 1.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from vernier.instance import CocoDataset


def federated_handle(image_ids: Sequence[int]) -> CocoDataset:
    images: list[dict[str, object]] = [
        {
            "id": i,
            "width": 100,
            "height": 100,
            "neg_category_ids": [],
            "not_exhaustive_category_ids": [],
        }
        for i in image_ids
    ]
    annotations = [
        {"id": 10 * i + c, "image_id": i, "category_id": c, "bbox": box, "area": 100, "iscrowd": 0}
        for i in image_ids
        for c, box in ((1, [0, 0, 10, 10]), (2, [50, 50, 10, 10]))
    ]
    categories = [
        {"id": 1, "name": "a", "frequency": "f"},
        {"id": 2, "name": "b", "frequency": "f"},
    ]
    gt: dict[str, object] = {"images": images, "annotations": annotations, "categories": categories}
    return CocoDataset.from_lvis_json(json.dumps(gt).encode())


def crowded(image_id: int) -> NDArray[np.float64]:
    """The image's ``(N, 7)`` detection matrix: 100 false positives, then the true positive."""
    fps = [[image_id, 80, 80, 5, 5, 0.99 - k * 1e-3, 1] for k in range(100)]
    return np.asarray([*fps, [image_id, 50, 50, 10, 10, 0.01, 2]], dtype=np.float64)
