"""Tests for streaming + result tables.

The headline contract: a streaming finalize over the same detections
(submitted in any number of batches) must produce table outputs
bit-equal to a batch ``Evaluator.evaluate(...)`` over the union. That's
what makes streaming usable for hard-example mining mid-training
without divergence from the offline report.

per_detection / per_pair require ``retain_iou=True`` at construction;
asking for either on a non-retaining stream raises ``ValueError``.
"""

from __future__ import annotations

import json

import pytest

from vernier import Evaluator, StreamingEvaluator
from vernier._core import per_class_to_arrow_pycapsule, per_image_to_arrow_pycapsule  # noqa: F401

# Three images, two categories. Designed so per_image and per_class
# both have a non-trivial mix of TPs/FPs/FNs.
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

# Explicit `id` on every detection — streaming auto-assigns starting at
# 1 per batch (per ADR-0013 §"Detection identifiers"), so a user-provided
# id is the only way to match batch and streaming detection_ids exactly.
_DT_BATCH_1 = json.dumps(
    [
        {"id": 11, "image_id": 1, "category_id": 1, "score": 0.9, "bbox": [0, 0, 10, 10]},
    ]
).encode()
_DT_BATCH_2 = json.dumps(
    [
        {"id": 22, "image_id": 2, "category_id": 2, "score": 0.8, "bbox": [80, 80, 10, 10]},
        {"id": 33, "image_id": 3, "category_id": 1, "score": 0.7, "bbox": [25, 20, 10, 10]},
    ]
).encode()
_DT_BATCH_FULL = json.dumps(
    [
        {"id": 11, "image_id": 1, "category_id": 1, "score": 0.9, "bbox": [0, 0, 10, 10]},
        {"id": 22, "image_id": 2, "category_id": 2, "score": 0.8, "bbox": [80, 80, 10, 10]},
        {"id": 33, "image_id": 3, "category_id": 1, "score": 0.7, "bbox": [25, 20, 10, 10]},
    ]
).encode()


def test_streaming_finalize_with_tables_returns_per_image_and_per_class() -> None:
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    ev = StreamingEvaluator(_GT)
    ev.update(_DT_BATCH_1)
    ev.update(_DT_BATCH_2)
    summary, per_image, per_class, _, _ = ev.finalize_with_tables(per_image=True, per_class=True)
    assert summary is not None
    assert per_image is not None
    assert per_class is not None


def test_streaming_finalize_with_tables_per_image_matches_batch() -> None:
    """Streaming + tables must reproduce the batch per_image table
    exactly (modulo row ordering — both sort by image_id ascending)."""
    polars = pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    batch_result = Evaluator().evaluate(_GT, _DT_BATCH_FULL, tables=("per_image",))
    batch_df = batch_result.per_image.sort("image_id")

    ev = StreamingEvaluator(_GT)
    ev.update(_DT_BATCH_1)
    ev.update(_DT_BATCH_2)
    _summary, stream_arrow, _per_class, _, _ = ev.finalize_with_tables(per_image=True)
    assert stream_arrow is not None
    stream_df = polars.from_arrow(stream_arrow).sort("image_id")

    from polars.testing import assert_frame_equal

    assert_frame_equal(batch_df, stream_df)


def test_streaming_finalize_with_tables_per_class_matches_batch() -> None:
    polars = pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    batch_result = Evaluator().evaluate(_GT, _DT_BATCH_FULL, tables=("per_class",))
    batch_df = batch_result.per_class.sort("category_id")

    ev = StreamingEvaluator(_GT)
    ev.update(_DT_BATCH_FULL)
    _summary, _per_image, stream_arrow, _, _ = ev.finalize_with_tables(per_class=True)
    assert stream_arrow is not None
    stream_df = polars.from_arrow(stream_arrow).sort("category_id")

    from polars.testing import assert_frame_equal

    assert_frame_equal(batch_df, stream_df)


def test_streaming_snapshot_with_tables_does_not_consume_evaluator() -> None:
    """`snapshot_with_tables` is the mid-stream variant; subsequent
    `update()` calls must still work."""
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    ev = StreamingEvaluator(_GT)
    ev.update(_DT_BATCH_1)
    summary1, per_image1, _, _, _ = ev.snapshot_with_tables(per_image=True)
    assert summary1 is not None
    assert per_image1 is not None

    # Still usable.
    ev.update(_DT_BATCH_2)
    summary2, per_image2, _, _, _ = ev.snapshot_with_tables(per_image=True)
    assert summary2 is not None
    assert per_image2 is not None


def test_streaming_per_detection_per_pair_require_retain_iou() -> None:
    """Asking for per_detection or per_pair on a non-retaining stream
    must raise — the spine cannot reconstruct IoU matrices after the
    fact."""
    ev = StreamingEvaluator(_GT)
    ev.update(_DT_BATCH_FULL)
    with pytest.raises(ValueError, match="retain_iou"):
        ev.snapshot_with_tables(per_detection=True)
    with pytest.raises(ValueError, match="retain_iou"):
        ev.finalize_with_tables(per_pair=True)


def test_streaming_finalize_per_detection_matches_batch() -> None:
    """With `retain_iou=True`, streaming per_detection must match the
    batch path bit-for-bit (modulo row ordering — both sort by
    detection_id ascending)."""
    polars = pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    batch_result = Evaluator().evaluate(_GT, _DT_BATCH_FULL, tables=("per_detection",))
    batch_df = batch_result.per_detection.sort("detection_id")

    ev = StreamingEvaluator(_GT, retain_iou=True)
    ev.update(_DT_BATCH_1)
    ev.update(_DT_BATCH_2)
    _summary, _, _, stream_arrow, _ = ev.finalize_with_tables(per_detection=True)
    assert stream_arrow is not None
    stream_df = polars.from_arrow(stream_arrow).sort("detection_id")

    from polars.testing import assert_frame_equal

    assert_frame_equal(batch_df, stream_df)


def test_streaming_finalize_per_pair_matches_batch() -> None:
    """With `retain_iou=True`, streaming per_pair must match the batch
    path bit-for-bit (modulo row ordering)."""
    polars = pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    batch_result = Evaluator().evaluate(_GT, _DT_BATCH_FULL, tables=("per_pair",))
    batch_df = batch_result.per_pair.sort(["image_id", "detection_id", "ground_truth_id"])

    ev = StreamingEvaluator(_GT, retain_iou=True)
    ev.update(_DT_BATCH_1)
    ev.update(_DT_BATCH_2)
    _summary, _, _, _, stream_arrow = ev.finalize_with_tables(per_pair=True)
    assert stream_arrow is not None
    stream_df = polars.from_arrow(stream_arrow).sort(
        ["image_id", "detection_id", "ground_truth_id"]
    )

    from polars.testing import assert_frame_equal

    assert_frame_equal(batch_df, stream_df)
