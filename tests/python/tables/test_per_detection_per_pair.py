"""Week 2.4 (ADR-0019) tests for per_detection and per_pair tables.

Validates the retention-dependent tables end-to-end: schema pinning,
geometry gate, the ``PerPairOverflowError`` path (currently surfaces
as a ``ValueError`` in v0.5; promotion to a typed Python exception
lands alongside Week 2.5 streaming integration).
"""

from __future__ import annotations

import json

import pytest

from vernier.instance import Evaluator, TablesConfig

# Three images, two categories. Designed to produce TPs, FPs, and at
# least one matched (DT, GT) pair so per_pair has rows to emit.
_GT = json.dumps(
    {
        "images": [
            {"id": 1, "width": 100, "height": 100},
            {"id": 2, "width": 100, "height": 100},
            {"id": 3, "width": 100, "height": 100},
        ],
        "annotations": [
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
                "image_id": 2,
                "category_id": 2,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 3,
                "category_id": 1,
                "bbox": [20, 20, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}],
    }
).encode()

_DT = json.dumps(
    [
        # Perfect match for image 1.
        {"image_id": 1, "category_id": 1, "score": 0.9, "bbox": [0, 0, 10, 10]},
        # Unmatched FP for image 2 (DT far from GT).
        {"image_id": 2, "category_id": 2, "score": 0.8, "bbox": [80, 80, 10, 10]},
        # IoU=0.5 match for image 3 (DT shifted by 5 pixels — half overlap).
        {"image_id": 3, "category_id": 1, "score": 0.7, "bbox": [25, 20, 10, 10]},
    ]
).encode()


def test_per_detection_schema_pins_eight_columns_by_default() -> None:
    polars = pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    result = Evaluator().evaluate(_GT, _DT, tables=("per_detection",))
    df = result.per_detection
    assert isinstance(df, polars.DataFrame)
    assert df.columns == [
        "detection_id",
        "image_id",
        "category_id",
        "score",
        "area",
        "match_status_at_50",
        "matched_gt_id_at_50",
        "best_iou",
    ]
    assert df.height == 3


def test_per_detection_geometry_gate_adds_bbox_xywh_column() -> None:
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    cfg = TablesConfig(per_detection_with_geometry=True)
    result = Evaluator().evaluate(_GT, _DT, tables=("per_detection",), tables_config=cfg)
    df = result.per_detection
    assert "bbox_xywh" in df.columns
    # Each entry is a list of length 4 (x, y, w, h).
    for row in df["bbox_xywh"].to_list():
        assert len(row) == 4


def test_per_detection_match_status_dictionary_pin() -> None:
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    result = Evaluator().evaluate(_GT, _DT, tables=("per_detection",))
    statuses = result.per_detection["match_status_at_50"].to_list()
    # 3 rows: image 1 perfect TP, image 2 FP, image 3 TP at IoU=0.5
    # threshold but score 0.7 with 50%-overlap may or may not pass —
    # accept either "tp" or "fp" but require all values are in the
    # pinned dictionary.
    assert all(s in ("tp", "fp", "ignored") for s in statuses)


def test_per_detection_best_iou_populated_when_retention_active() -> None:
    """The Python wrapper turns retain_iou=True automatically when
    per_detection or per_pair is requested. Best_iou is populated for
    detections whose image has any same-category GT."""
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    result = Evaluator().evaluate(_GT, _DT, tables=("per_detection",))
    bi = result.per_detection["best_iou"].to_list()
    # Each fixture row has at least one same-category GT, so best_iou
    # is populated for every row.
    assert all(v is not None for v in bi)


def test_per_pair_emits_rows_above_iou_floor() -> None:
    polars = pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    cfg = TablesConfig(per_pair_iou_floor=0.0)
    result = Evaluator().evaluate(_GT, _DT, tables=("per_pair",), tables_config=cfg)
    df = result.per_pair
    assert isinstance(df, polars.DataFrame)
    assert df.columns == [
        "detection_id",
        "ground_truth_id",
        "image_id",
        "category_id",
        "iou",
    ]
    # Three images each with one DT and one same-category GT → 3 pairs.
    assert df.height == 3
    # All IoU values are non-negative.
    assert all(v >= 0.0 for v in df["iou"].to_list())


def test_per_pair_high_floor_drops_low_overlap_pairs() -> None:
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    cfg = TablesConfig(per_pair_iou_floor=0.99)
    result = Evaluator().evaluate(_GT, _DT, tables=("per_pair",), tables_config=cfg)
    # Only image 1's perfect-match pair (IoU=1.0) survives.
    assert result.per_pair.height == 1


def test_per_pair_overflow_raises_with_retain_iou_in_message() -> None:
    """Tiny cap forces the overflow path. Error must carry the typed
    `PerPairOverflow` message; the Python wrapper exposes it as
    ValueError in v0.5 (typed PerPairOverflowError lands in 2.5)."""
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    cfg = TablesConfig(per_pair_iou_floor=0.0, per_pair_max_rows=1)
    with pytest.raises(ValueError, match="per_pair"):
        Evaluator().evaluate(_GT, _DT, tables=("per_pair",), tables_config=cfg)


def test_tables_all_returns_all_four_dataframes() -> None:
    polars = pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    result = Evaluator().evaluate(_GT, _DT, tables="all")
    assert isinstance(result.per_image, polars.DataFrame)
    assert isinstance(result.per_class, polars.DataFrame)
    assert isinstance(result.per_detection, polars.DataFrame)
    assert isinstance(result.per_pair, polars.DataFrame)


def test_per_pair_without_request_raises_runtime_error() -> None:
    """Cached property accessor for an unrequested table should raise
    a structured RuntimeError naming the missing table."""
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    result = Evaluator().evaluate(_GT, _DT, tables=("per_image",))
    with pytest.raises(RuntimeError, match="per_pair"):
        _ = result.per_pair
    with pytest.raises(RuntimeError, match="per_detection"):
        _ = result.per_detection
