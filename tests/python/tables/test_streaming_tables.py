"""Week 2.5 (ADR-0019) tests for streaming + result tables.

The headline contract for streaming integration: a streaming
finalize over the same detections (submitted in any number of
batches) must produce per_image / per_class tables bit-equal to a
batch ``Evaluator.evaluate(...)`` over the union. That's what makes
streaming usable for hard-example mining mid-training without divergence
from the offline report.

v0.5 supports per_image / per_class on the streaming path; per_detection
and per_pair surface as ``ValueError`` (the underlying integration
defers detection / EvalImageMeta tracking to a follow-up — see
``StreamingEvaluator::finalize_with_tables`` rustdoc).
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

_DT_BATCH_1 = json.dumps(
    [
        {"image_id": 1, "category_id": 1, "score": 0.9, "bbox": [0, 0, 10, 10]},
    ]
).encode()
_DT_BATCH_2 = json.dumps(
    [
        {"image_id": 2, "category_id": 2, "score": 0.8, "bbox": [80, 80, 10, 10]},
        {"image_id": 3, "category_id": 1, "score": 0.7, "bbox": [25, 20, 10, 10]},
    ]
).encode()
_DT_BATCH_FULL = json.dumps(
    [
        {"image_id": 1, "category_id": 1, "score": 0.9, "bbox": [0, 0, 10, 10]},
        {"image_id": 2, "category_id": 2, "score": 0.8, "bbox": [80, 80, 10, 10]},
        {"image_id": 3, "category_id": 1, "score": 0.7, "bbox": [25, 20, 10, 10]},
    ]
).encode()


def test_streaming_finalize_with_tables_returns_per_image_and_per_class() -> None:
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    ev = StreamingEvaluator(_GT)
    ev.update(_DT_BATCH_1)
    ev.update(_DT_BATCH_2)
    summary, per_image, per_class = ev.finalize_with_tables(per_image=True, per_class=True)
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
    _summary, stream_arrow, _per_class = ev.finalize_with_tables(per_image=True)
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
    _summary, _per_image, stream_arrow = ev.finalize_with_tables(per_class=True)
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
    summary1, per_image1, _ = ev.snapshot_with_tables(per_image=True)
    assert summary1 is not None
    assert per_image1 is not None

    # Still usable.
    ev.update(_DT_BATCH_2)
    summary2, per_image2, _ = ev.snapshot_with_tables(per_image=True)
    assert summary2 is not None
    assert per_image2 is not None
