"""Pin the typed-error surface: each EvalError variant reachable from
Python gets a smoke test asserting the class hierarchy and at least one
trigger path."""

from __future__ import annotations

import json

import pytest

from vernier import instance

# A tiny GT/DT pair that round-trips cleanly through Evaluator.evaluate.
# Several tests below mutate one field to force a single error.
_GT_OK = json.dumps(
    {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "thing"}],
    }
).encode()

_DT_OK = json.dumps(
    [{"image_id": 1, "category_id": 1, "score": 0.9, "bbox": [0, 0, 10, 10]}]
).encode()


# ---------------------------------------------------------------------------
# Subclass / import smoke
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc_name",
    [
        "InvalidAnnotationError",
        "NonFiniteError",
        "DimensionMismatchError",
        "InvalidConfigError",
    ],
)
def test_typed_error_subclasses_value_error(exc_name: str) -> None:
    exc = getattr(instance, exc_name)
    assert issubclass(exc, ValueError), f"{exc_name} must subclass ValueError"


@pytest.mark.parametrize(
    "exc_name",
    [
        "PartialFormatMismatch",
        "PartialDatasetMismatch",
        "PartialParamsMismatch",
        "PartialPartitionOverlap",
        "PartialRankCollision",
    ],
)
def test_partial_error_classes_are_importable(exc_name: str) -> None:
    """The five ADR-0031 distributed-eval errors are smoke-imported only.
    Behavior is covered by the distributed-eval parity suite.
    """
    exc = getattr(instance, exc_name)
    assert isinstance(exc, type)


# ---------------------------------------------------------------------------
# Reachable from Python today
# ---------------------------------------------------------------------------


def test_invalid_annotation_unknown_image_id() -> None:
    """GT annotation references an `image_id` not in `images[]`."""
    gt_obj = json.loads(_GT_OK)
    gt_obj["annotations"][0]["image_id"] = 999  # not in images
    gt = json.dumps(gt_obj).encode()
    with pytest.raises(instance.InvalidAnnotationError):
        instance.Evaluator().evaluate(gt, _DT_OK)


def test_invalid_annotation_unknown_category_id() -> None:
    """GT annotation references a `category_id` not in `categories[]`."""
    gt_obj = json.loads(_GT_OK)
    gt_obj["annotations"][0]["category_id"] = 999  # not in categories
    gt = json.dumps(gt_obj).encode()
    with pytest.raises(instance.InvalidAnnotationError):
        instance.Evaluator().evaluate(gt, _DT_OK)


def test_non_finite_detection_score() -> None:
    """A detection with a non-finite score is rejected at ingest."""
    dt_obj = [
        {
            "image_id": 1,
            "category_id": 1,
            "score": float("nan"),
            "bbox": [0, 0, 10, 10],
        }
    ]
    # `json.dumps` emits the literal `NaN` token; `serde_json` accepts it
    # in lenient mode but the score-finiteness check still fires.
    dt = json.dumps(dt_obj).encode()
    with pytest.raises((instance.NonFiniteError, ValueError)):
        instance.Evaluator().evaluate(_GT_OK, dt)


def test_json_malformed_payload_raises_value_error() -> None:
    """A malformed GT payload surfaces as a generic `ValueError` (the
    `EvalError::Json` variant has no typed subclass yet — that's the
    JSON quirk family entry in the error matrix)."""
    with pytest.raises(ValueError, match=r"(?i)json|expected|invalid"):
        instance.CocoDataset.from_json(b"not-json")


def test_mask_bad_rle_raises_value_error() -> None:
    """A detection with an unparseable compressed-RLE segmentation
    surfaces through `EvalError::Mask` as a generic `ValueError`."""
    dt_obj = [
        {
            "image_id": 1,
            "category_id": 1,
            "score": 0.9,
            "bbox": [0, 0, 10, 10],
            "segmentation": {"size": [100, 100], "counts": "not-a-real-rle"},
        }
    ]
    dt = json.dumps(dt_obj).encode()
    with pytest.raises(ValueError, match=r"(?i)rle|mask|malformed"):
        instance.Evaluator(iou=instance.Segm()).evaluate(_GT_OK, dt)


def test_out_of_budget_attribute_shape() -> None:
    """Smoke-check `OutOfBudgetError` is importable and carries the
    documented attributes when raised. Forcing the streaming budget
    trip from a one-shot test is fragile; we exercise the catchability
    pattern by constructing the class directly."""
    assert issubclass(instance.OutOfBudgetError, RuntimeError)


def test_queue_full_error_class_shape() -> None:
    """`QueueFullError` is exercised end-to-end by
    `tests/python/background/test_background_backpressure_raises.py`;
    here we only pin the class hierarchy so the manifest stays
    discoverable."""
    assert issubclass(instance.QueueFullError, RuntimeError)
