"""``dt_area="mask"`` on the array path (quirk J3).

pycocotools buckets a detection by the ``area`` its ``cocoDt`` carries:
``loadRes`` writes the box area for bbox results and ``maskUtils.area`` for
segm results, and TorchMetrics carries both as ``area_bbox`` / ``area_segm``.
Joint evaluation from one set of array-form detections therefore needs the
bbox pass on box areas and the segm pass on mask areas. The fixture's
top-scored false positive has a 25x25 box (small) and a 60x60 mask
(medium), so either pass reading the other's area moves ``AP_small``.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
from typing import Any, Literal, cast

import numpy as np
import pytest
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from vernier import _core
from vernier.instance import Detections

_SIZE = 128
_MAX_DETS = [1, 10, 100]


def _mask(rows: slice, cols: slice) -> np.ndarray[Any, np.dtype[np.uint8]]:
    mask = np.zeros((_SIZE, _SIZE), dtype=np.uint8, order="F")
    mask[rows, cols] = 1
    return mask


# (bbox xywh, mask, score) per detection; GT is one 20x20 object.
_GT_MASK = _mask(slice(0, 20), slice(0, 20))
_DETECTIONS = [
    ([0.0, 0.0, 20.0, 20.0], _mask(slice(0, 25), slice(0, 25)), 0.5),
    ([40.0, 40.0, 25.0, 25.0], _mask(slice(40, 100), slice(40, 100)), 0.95),
]


def _uncompressed_counts(mask: np.ndarray[Any, np.dtype[np.uint8]]) -> np.ndarray[Any, Any]:
    flat = mask.ravel(order="F")
    changes = np.flatnonzero(np.diff(flat)) + 1
    edges = np.concatenate(([0], changes, [flat.size]))
    runs = np.diff(edges)
    if flat[0]:
        runs = np.concatenate(([0], runs))
    return runs.astype(np.uint32)


def _rle_forms(mask: np.ndarray[Any, np.dtype[np.uint8]]) -> dict[str, Any]:
    return {
        "compressed": dict(mask_utils.encode(mask)),
        "uncompressed": {"counts": _uncompressed_counts(mask), "size": (_SIZE, _SIZE)},
        "bitmask": mask.astype(bool),
    }


def _gt_dataset() -> dict[str, Any]:
    encoded = mask_utils.encode(_GT_MASK)
    # pycocotools emits bytes counts at runtime; its stubs say str.
    rle = {"size": encoded["size"], "counts": cast(bytes, encoded["counts"]).decode("ascii")}
    return {
        "images": [{"id": 0, "width": _SIZE, "height": _SIZE}],
        "categories": [{"id": 1, "name": "a"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 0,
                "category_id": 1,
                "iscrowd": 0,
                "bbox": [0, 0, 20, 20],
                "area": 400.0,
                "segmentation": rle,
            }
        ],
    }


def _array_detections(rle_form: str) -> list[Detections]:
    return [
        cast(
            Detections,
            {
                "image_id": 0,
                "boxes": np.array([bbox for bbox, _, _ in _DETECTIONS]),
                "scores": np.array([score for _, _, score in _DETECTIONS]),
                "labels": np.ones(len(_DETECTIONS), dtype=np.int64),
                "rles": [_rle_forms(mask)[rle_form] for _, mask, _ in _DETECTIONS],
            },
        )
    ]


def _pycocotools(iou_type: Literal["bbox", "segm"]) -> COCOeval:
    gt, dt = COCO(), COCO()
    dt_anns = []
    for ann_id, (bbox, mask, score) in enumerate(_DETECTIONS, start=1):
        rle = mask_utils.encode(mask)
        area = bbox[2] * bbox[3] if iou_type == "bbox" else float(mask_utils.area(rle))
        dt_anns.append(
            {
                "id": ann_id,
                "image_id": 0,
                "category_id": 1,
                "bbox": bbox,
                "score": score,
                "area": area,
                "segmentation": rle,
            }
        )
    gt_dataset = _gt_dataset()
    cast(Any, gt).dataset = copy.deepcopy(gt_dataset)
    cast(Any, dt).dataset = {**copy.deepcopy(gt_dataset), "annotations": dt_anns}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt.createIndex()
        evaluator = COCOeval(gt, dt, iouType=iou_type)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return evaluator


def _assert_grid_matches(grid: Any, reference: COCOeval) -> list[float]:
    accumulated = grid.accumulate(_MAX_DETS)
    stats = accumulated.summarize().stats
    np.testing.assert_array_equal(stats, reference.stats)
    np.testing.assert_array_equal(accumulated.precision, reference.eval["precision"])
    np.testing.assert_array_equal(accumulated.recall, reference.eval["recall"])
    return stats


@pytest.mark.parametrize("rle_form", ["compressed", "uncompressed", "bitmask"])
def test_joint_bbox_and_mask_areas_match_pycocotools(rle_form: str) -> None:
    gt_bytes = json.dumps(_gt_dataset()).encode()
    detections = _array_detections(rle_form)

    bbox_grid = _core.evaluate_bbox_grid(gt_bytes, detections, "strict", 100, True, dt_area="bbox")
    segm_grid = _core.evaluate_segm_grid(gt_bytes, detections, "strict", 100, True, dt_area="mask")

    bbox_stats = _assert_grid_matches(bbox_grid, _pycocotools("bbox"))
    segm_stats = _assert_grid_matches(segm_grid, _pycocotools("segm"))
    assert bbox_stats[3] == pytest.approx(0.5)
    assert segm_stats[3] == pytest.approx(0.3)


def test_mask_area_is_rejected_on_the_bbox_grid() -> None:
    gt_bytes = json.dumps(_gt_dataset()).encode()
    with pytest.raises(ValueError, match="evaluate_segm_grid and evaluate_boundary_grid"):
        _core.evaluate_bbox_grid(
            gt_bytes,
            _array_detections("compressed"),
            "strict",
            100,
            True,
            dt_area=cast(Any, "mask"),
        )


def test_mask_area_requires_a_segmentation() -> None:
    gt_bytes = json.dumps(_gt_dataset()).encode()
    dt_bytes = json.dumps(
        [{"image_id": 0, "category_id": 1, "bbox": [0, 0, 20, 20], "score": 0.9}]
    ).encode()
    with pytest.raises(ValueError, match="requires an RLE `segmentation`"):
        _core.evaluate_segm_grid(gt_bytes, dt_bytes, "strict", 100, True, dt_area="mask")
