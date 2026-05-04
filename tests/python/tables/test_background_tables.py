"""Background-evaluator + result-tables integration tests.

Mirrors the streaming integration: a ``BackgroundEvaluator.finalize_with_tables``
over the same detections (submitted in any number of batches) must produce
table outputs bit-equal to a batch ``Evaluator.evaluate(...)`` over the union.
"""

from __future__ import annotations

import json

import pytest

from vernier.instance import BackgroundEvaluator, Evaluator

# Three images, two categories. Same fixture shape as
# `test_streaming_tables.py` but with explicit `id` on every detection so
# auto-id sequencing across submits doesn't drift from the batch path.
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


def test_background_finalize_with_tables_per_image_matches_batch() -> None:
    polars = pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    batch_result = Evaluator().evaluate(_GT, _DT_BATCH_FULL, tables=("per_image",))
    batch_df = batch_result.per_image.sort("image_id")

    with BackgroundEvaluator(_GT) as ev:
        ev.submit(_DT_BATCH_1)
        ev.submit(_DT_BATCH_2)
        _, bg_arrow, _, _, _ = ev.finalize_with_tables(per_image=True)
    assert bg_arrow is not None
    bg_df = polars.from_arrow(bg_arrow).sort("image_id")

    from polars.testing import assert_frame_equal

    assert_frame_equal(batch_df, bg_df)


def test_background_finalize_with_tables_per_class_matches_batch() -> None:
    polars = pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    batch_result = Evaluator().evaluate(_GT, _DT_BATCH_FULL, tables=("per_class",))
    batch_df = batch_result.per_class.sort("category_id")

    with BackgroundEvaluator(_GT) as ev:
        ev.submit(_DT_BATCH_FULL)
        _, _, bg_arrow, _, _ = ev.finalize_with_tables(per_class=True)
    assert bg_arrow is not None
    bg_df = polars.from_arrow(bg_arrow).sort("category_id")

    from polars.testing import assert_frame_equal

    assert_frame_equal(batch_df, bg_df)


def test_background_per_detection_per_pair_require_retain_iou() -> None:
    """The retention contract is enforced by the underlying streaming
    evaluator; the background path inherits it."""
    with BackgroundEvaluator(_GT) as ev:
        ev.submit(_DT_BATCH_FULL)
        with pytest.raises(ValueError, match="retain_iou"):
            ev.snapshot_with_tables(per_detection=True)


def test_background_finalize_per_detection_matches_batch() -> None:
    polars = pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    batch_result = Evaluator().evaluate(_GT, _DT_BATCH_FULL, tables=("per_detection",))
    batch_df = batch_result.per_detection.sort("detection_id")

    with BackgroundEvaluator(_GT, retain_iou=True) as ev:
        ev.submit(_DT_BATCH_1)
        ev.submit(_DT_BATCH_2)
        _, _, _, bg_arrow, _ = ev.finalize_with_tables(per_detection=True)
    assert bg_arrow is not None
    bg_df = polars.from_arrow(bg_arrow).sort("detection_id")

    from polars.testing import assert_frame_equal

    assert_frame_equal(batch_df, bg_df)


def test_background_snapshot_with_tables_does_not_consume() -> None:
    """``snapshot_with_tables`` must leave the worker usable."""
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    with BackgroundEvaluator(_GT) as ev:
        ev.submit(_DT_BATCH_1)
        s1, p1, _, _, _ = ev.snapshot_with_tables(per_image=True)
        assert s1 is not None
        assert p1 is not None

        ev.submit(_DT_BATCH_2)
        s2, p2, _, _, _ = ev.snapshot_with_tables(per_image=True)
        assert s2 is not None
        assert p2 is not None
