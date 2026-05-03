"""Per-image top-K trim parity (ADR-0026 PR-5).

The trim runs internally inside the federated FFI grid path
(`evaluate_*_grid_with_dataset` calls `CocoDetections::lvis_trim` on
the DT side when the GT carries federated metadata). For unit-level
parity we don't have a direct Python handle on the trim — but we
can pin the cross-class crowding behavior end-to-end: the resulting
``eval_imgs`` cell counts must match what `LVISResults` would emit
after its own input-side trim.

Resolves ADR-0026 appendix questions:

- **Q1.** AC2 trim observability — 500 single-category preds on one
  image trim to 300.
- **Q2.** AC3 cross-class crowding — 250 cat-A + 350 cat-B preds on
  one image trim to **300 total**, not 250 + min(350, 300).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pytest

# A trivial GT with two categories on one image. The fixtures in
# this file are stress-tests for the trim, not full parity exercises;
# we just need a GT whose dataset shape allows lvis-api to load.
_TRIM_FIXTURE_GT: dict[str, object] = {
    "images": [
        {
            "id": 1,
            "width": 100,
            "height": 100,
            "neg_category_ids": [],
            "not_exhaustive_category_ids": [],
        }
    ],
    "annotations": [
        # One GT per category so `pos[1] = {1, 2}` and the trim's
        # output is what feeds the matching engine on both classes.
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [0, 0, 10, 10],
            "area": 100,
            "iscrowd": 0,
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 2,
            "bbox": [0, 0, 10, 10],
            "area": 100,
            "iscrowd": 0,
        },
    ],
    "categories": [
        {"id": 1, "name": "alpha", "frequency": "f"},
        {"id": 2, "name": "beta", "frequency": "f"},
    ],
}


def _gt_bytes() -> bytes:
    return json.dumps(_TRIM_FIXTURE_GT).encode("utf-8")


def _dt_bytes(detections: list[Mapping[str, object]]) -> bytes:
    return json.dumps(detections).encode("utf-8")


def _lvis_results_count(gt_bytes: bytes, dt_bytes: bytes, max_dets: int) -> int:
    """Run `LVISResults` (the oracle) and return how many detections
    survive its input-side trim. The oracle trims at construction;
    we read the post-trim length off `dataset['annotations']`.
    """
    from lvis import LVIS, LVISResults  # type: ignore[import-not-found]

    with NamedTemporaryFile("wb", suffix="_gt.json", delete=False) as fgt:
        fgt.write(gt_bytes)
        gt_path = fgt.name
    with NamedTemporaryFile("wb", suffix="_dt.json", delete=False) as fdt:
        fdt.write(dt_bytes)
        dt_path = fdt.name
    try:
        oracle = LVIS(gt_path)
        results = LVISResults(oracle, dt_path, max_dets=max_dets)
        return len(results.dataset["annotations"])
    finally:
        Path(gt_path).unlink(missing_ok=True)
        Path(dt_path).unlink(missing_ok=True)


def _vernier_total_dts_seen(gt_bytes: bytes, dt_bytes: bytes, max_dets: int) -> int:
    """Count of unique post-trim DTs vernier feeds into the matching
    engine. The eval_imgs grid replicates each DT across A area
    buckets and across (image, category) cells; deduping by DT id —
    vernier's FFI `eval_img_dict` exposes `dtIds` per cell — gives
    the post-trim DT count, mirroring
    `len(LVISResults.dataset['annotations'])`.
    """
    # We need the raw FFI dict (which carries `dtIds`); the parity
    # harness's snapshot path strips that key. Build a fresh grid
    # here just for the dedupe.
    from vernier import Dataset
    from vernier._core import evaluate_bbox_grid_with_dataset

    gt_ds = Dataset.from_lvis_json(gt_bytes)
    grid = evaluate_bbox_grid_with_dataset(gt_ds, dt_bytes, "strict", max_dets, True)
    seen_ids: set[int] = set()
    for cell in grid.eval_imgs():
        if cell is None:
            continue
        for ann_id in cell["dtIds"]:
            seen_ids.add(int(ann_id))
    return len(seen_ids)


@pytest.mark.parity_lvis
def test_q1_500_single_category_dts_trim_to_300() -> None:
    # Quirk AC2 / appendix Q1: 500 single-category detections on one
    # image must trim to exactly 300 — both at vernier's internal
    # trim and at the oracle's `LVISResults`.
    detections: list[Mapping[str, object]] = [
        {
            "image_id": 1,
            "category_id": 1,
            "score": 1.0 - i / 1000.0,
            "bbox": [0, 0, 1, 1],
        }
        for i in range(500)
    ]
    dt_b = _dt_bytes(detections)
    gt_b = _gt_bytes()
    assert _lvis_results_count(gt_b, dt_b, max_dets=300) == 300
    assert _vernier_total_dts_seen(gt_b, dt_b, max_dets=300) == 300


@pytest.mark.parity_lvis
def test_q2_cross_class_crowding_trims_to_300_total() -> None:
    # Quirk AC3 / appendix Q2: 250 cat-1 + 350 cat-2 detections on
    # one image must trim to **300 total**, not 250 + min(350, 300)
    # = 550. Score layouts are interleaved enough that a per-class
    # trim would diverge from the cross-class one (the oracle's
    # path is per-image, the corrected reading from PR-2).
    detections: list[Mapping[str, object]] = []
    for i in range(250):
        detections.append(
            {
                "image_id": 1,
                "category_id": 1,
                "score": 0.5 - i * 0.002,
                "bbox": [0, 0, 1, 1],
            }
        )
    for i in range(350):
        detections.append(
            {
                "image_id": 1,
                "category_id": 2,
                "score": 1.0 - i * 0.002,
                "bbox": [0, 0, 1, 1],
            }
        )
    dt_b = _dt_bytes(detections)
    gt_b = _gt_bytes()
    assert _lvis_results_count(gt_b, dt_b, max_dets=300) == 300
    assert _vernier_total_dts_seen(gt_b, dt_b, max_dets=300) == 300


@pytest.mark.parity_lvis
def test_lvis_trim_uses_well_separated_scores_for_strict_membership() -> None:
    # Bit-equal membership check using a fixture where scores never
    # collide across classes, so the float-arithmetic-at-boundary
    # ambiguity (different decimal serializations of arithmetically-
    # equal scores) doesn't muddle the diff.
    #
    # cat-1 scores are in [0.500, 0.749] (250 entries, step 0.001),
    # cat-2 scores are in [0.000, 0.349] (350 entries, step 0.001).
    # cat-1 strictly dominates cat-2; top 300 = all 250 cat-1 + top
    # 50 cat-2 (scores 0.300..0.349).
    from lvis import LVIS, LVISResults  # type: ignore[import-not-found]

    detections: list[Mapping[str, object]] = []
    for i in range(250):
        detections.append(
            {
                "image_id": 1,
                "category_id": 1,
                "score": 0.500 + i * 0.001,
                "bbox": [0, 0, 1, 1],
            }
        )
    for i in range(350):
        detections.append(
            {
                "image_id": 1,
                "category_id": 2,
                "score": 0.000 + i * 0.001,
                "bbox": [0, 0, 1, 1],
            }
        )
    gt_b = _gt_bytes()
    dt_b = _dt_bytes(detections)

    with NamedTemporaryFile("wb", suffix="_gt.json", delete=False) as fgt:
        fgt.write(gt_b)
        gt_path = fgt.name
    with NamedTemporaryFile("wb", suffix="_dt.json", delete=False) as fdt:
        fdt.write(dt_b)
        dt_path = fdt.name
    try:
        oracle = LVIS(gt_path)
        results = LVISResults(oracle, dt_path, max_dets=300)
        oracle_scores = sorted(float(d["score"]) for d in results.dataset["annotations"])
    finally:
        Path(gt_path).unlink(missing_ok=True)
        Path(dt_path).unlink(missing_ok=True)

    from vernier import Dataset
    from vernier._core import evaluate_bbox_grid_with_dataset

    gt_ds = Dataset.from_lvis_json(gt_b)
    grid = evaluate_bbox_grid_with_dataset(gt_ds, dt_b, "strict", 300, True)
    vernier_by_id: dict[int, float] = {}
    for cell in grid.eval_imgs():
        if cell is None:
            continue
        for ann_id, score in zip(cell["dtIds"], cell["dtScores"], strict=True):
            vernier_by_id[int(ann_id)] = float(score)
    vernier_scores_sorted = sorted(vernier_by_id.values())

    np.testing.assert_array_equal(np.asarray(oracle_scores), np.asarray(vernier_scores_sorted))
