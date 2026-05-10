"""Tests for the panoptic evaluation-parameter surface (ADR-0042).

Covers the four new fields on ``vernier.panoptic.Evaluator`` plus the
``StuffThingPartition`` value type:

- ``pq_iou_threshold: float | None``
- ``category_filter: CategoryFilter | None``
- ``class_grouping: Breakdown | None``
- ``stuff_thing_partition: StuffThingPartition | None``

Construction-time validation, ``__post_init__`` cross-field checks,
and the runtime gate that fires until the kernel plumbing lands
alongside the ADR-0039 distributed-eval phase.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vernier.panoptic import (
    Breakdown,
    CategoryFilterByGrouping,
    CategoryFilterByIds,
    CategoryFilterFrequency,
    Dataset,
    Evaluator,
    InvalidPanopticParams,
    Predictions,
    StuffThingPartition,
)

# --- defaults: existing behavior unchanged ----------------------------------


def test_default_evaluator_has_none_for_class_params() -> None:
    e = Evaluator()
    assert e.pq_iou_threshold is None
    assert e.category_filter is None
    assert e.class_grouping is None
    assert e.stuff_thing_partition is None
    assert e._has_custom_class_params() is False


# --- pq_iou_threshold validation --------------------------------------------


def test_pq_iou_threshold_zero_rejected() -> None:
    """Strict-zero is rejected as a footgun per ADR-0042."""
    with pytest.raises(InvalidPanopticParams, match=r"\(0\.0, 1\.0\]"):
        Evaluator(pq_iou_threshold=0.0)


def test_pq_iou_threshold_above_one_rejected() -> None:
    with pytest.raises(InvalidPanopticParams, match=r"\(0\.0, 1\.0\]"):
        Evaluator(pq_iou_threshold=1.5)


def test_pq_iou_threshold_negative_rejected() -> None:
    with pytest.raises(InvalidPanopticParams, match=r"\(0\.0, 1\.0\]"):
        Evaluator(pq_iou_threshold=-0.1)


def test_pq_iou_threshold_nan_rejected() -> None:
    with pytest.raises(InvalidPanopticParams, match="finite"):
        Evaluator(pq_iou_threshold=float("nan"))


def test_pq_iou_threshold_inf_rejected() -> None:
    with pytest.raises(InvalidPanopticParams, match="finite"):
        Evaluator(pq_iou_threshold=float("inf"))


@pytest.mark.parametrize("t", [0.001, 0.3, 0.5, 0.7, 1.0])
def test_pq_iou_threshold_valid_accepted(t: float) -> None:
    Evaluator(pq_iou_threshold=t)


# --- category_filter validation ---------------------------------------------


def test_category_filter_frequency_rejected_with_lvis_pointer() -> None:
    with pytest.raises(InvalidPanopticParams, match="LVIS-only"):
        Evaluator(category_filter=CategoryFilterFrequency(tag="r"))


def test_category_filter_byids_empty_rejected() -> None:
    with pytest.raises(InvalidPanopticParams, match="at least one"):
        Evaluator(category_filter=CategoryFilterByIds(ids=frozenset()))


def test_category_filter_bygrouping_without_grouping_rejected() -> None:
    with pytest.raises(InvalidPanopticParams, match="requires class_grouping"):
        Evaluator(category_filter=CategoryFilterByGrouping(label="x"))


def test_category_filter_bygrouping_unknown_label_rejected() -> None:
    g = Breakdown.from_class_groups("g", [("a", [1])])
    with pytest.raises(InvalidPanopticParams, match="not a label of class_grouping"):
        Evaluator(
            category_filter=CategoryFilterByGrouping(label="missing"),
            class_grouping=g,
        )


def test_category_filter_bygrouping_known_label_accepted() -> None:
    g = Breakdown.from_class_groups("g", [("vehicles", [1, 2])])
    Evaluator(
        category_filter=CategoryFilterByGrouping(label="vehicles"),
        class_grouping=g,
    )


# --- class_grouping shape ---------------------------------------------------


def test_class_grouping_must_be_class_groups_breakdown() -> None:
    range_bd = Breakdown.from_ranges("area", [("all", 0.0, 1e10)])
    with pytest.raises(InvalidPanopticParams, match="class-groups Breakdown"):
        Evaluator(class_grouping=range_bd)


# --- StuffThingPartition validation -----------------------------------------


def test_stuff_thing_partition_empty_stuff_rejected() -> None:
    with pytest.raises(InvalidPanopticParams, match="at least one"):
        StuffThingPartition(stuff=frozenset(), things=frozenset([1]))


def test_stuff_thing_partition_empty_things_rejected() -> None:
    with pytest.raises(InvalidPanopticParams, match="at least one"):
        StuffThingPartition(stuff=frozenset([1]), things=frozenset())


def test_stuff_thing_partition_overlap_rejected() -> None:
    with pytest.raises(InvalidPanopticParams, match="disjoint"):
        StuffThingPartition(stuff=frozenset([1, 2]), things=frozenset([2, 3]))


def test_stuff_thing_partition_disjoint_accepted() -> None:
    p = StuffThingPartition(stuff=frozenset([1, 2]), things=frozenset([3, 4]))
    assert p.stuff == frozenset([1, 2])
    assert p.things == frozenset([3, 4])


def test_stuff_thing_partition_threads_through_evaluator() -> None:
    p = StuffThingPartition(stuff=frozenset([1]), things=frozenset([2]))
    e = Evaluator(stuff_thing_partition=p)
    assert e.stuff_thing_partition is p
    assert e._has_custom_class_params() is True


# --- _has_custom_class_params discrimination --------------------------------


def test_any_custom_field_marks_evaluator_as_custom() -> None:
    assert Evaluator(pq_iou_threshold=0.7)._has_custom_class_params()
    assert Evaluator(
        category_filter=CategoryFilterByIds(ids=frozenset([1]))
    )._has_custom_class_params()
    g = Breakdown.from_class_groups("g", [("a", [1])])
    assert Evaluator(class_grouping=g)._has_custom_class_params()
    p = StuffThingPartition(stuff=frozenset([1]), things=frozenset([2]))
    assert Evaluator(stuff_thing_partition=p)._has_custom_class_params()


# --- end-to-end runs with custom params -------------------------------------


def _make_two_class_pair() -> tuple[Dataset, Predictions]:
    """4x4 fixture with two perfectly-matched segments. Category 1 is
    a thing, category 2 is stuff (per the categories JSON below).
    Predictions exactly match GT — every metric should be 1.0 unless
    a custom param changes the matching."""
    gt_map = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.uint32)
    dt_map = gt_map.copy()
    gt_segs = json.dumps(
        {
            "1": [
                {"id": 1, "category_id": 1, "iscrowd": 0, "area": 8},
                {"id": 2, "category_id": 2, "iscrowd": 0, "area": 8},
            ]
        }
    ).encode()
    dt_segs = json.dumps(
        {
            "1": [
                {"id": 1, "category_id": 1, "area": 8},
                {"id": 2, "category_id": 2, "area": 8},
            ]
        }
    ).encode()
    cats = json.dumps([{"id": 1, "isthing": True}, {"id": 2, "isthing": False}]).encode()
    gt = Dataset.from_arrays({1: gt_map}, gt_segs, cats)
    dt = Predictions.from_arrays({1: dt_map}, dt_segs)
    return gt, dt


def test_evaluate_default_params_unchanged() -> None:
    """No custom fields → kernel runs; baseline output intact."""
    gt, dt = _make_two_class_pair()
    summary = Evaluator().evaluate(gt, dt)
    assert summary.pq == pytest.approx(1.0)
    # per_group is empty when no grouping configured.
    assert summary.per_group() == {}


def test_evaluate_with_class_grouping_returns_per_group() -> None:
    gt, dt = _make_two_class_pair()
    g = Breakdown.from_class_groups("g", [("things_only", [1]), ("stuff_only", [2])])
    summary = Evaluator(class_grouping=g).evaluate(gt, dt)
    per_group = summary.per_group()
    assert set(per_group.keys()) == {"things_only", "stuff_only"}
    # Both groups have a single perfectly-matched category → PQ=1.0.
    assert per_group["things_only"].pq == pytest.approx(1.0)
    assert per_group["stuff_only"].pq == pytest.approx(1.0)
    assert per_group["things_only"].member_category_ids == [1]


def test_evaluate_with_category_filter_restricts_global_pq() -> None:
    """Filter to category 1 only → global PQ averages only its row."""
    gt, dt = _make_two_class_pair()
    summary = Evaluator(
        category_filter=CategoryFilterByIds(ids=frozenset([1])),
        things_stuff_split=False,
    ).evaluate(gt, dt)
    assert summary.pq == pytest.approx(1.0)
    assert summary.n == 1


def test_evaluate_with_stuff_thing_partition_overrides_dataset() -> None:
    """User-supplied partition flips the things/stuff split. With
    category 1 marked stuff and category 2 marked things, the buckets
    are swapped relative to the dataset's own `isthing` flags."""
    gt, dt = _make_two_class_pair()
    p = StuffThingPartition(stuff=frozenset([1]), things=frozenset([2]))
    summary = Evaluator(stuff_thing_partition=p).evaluate(gt, dt)
    # Both categories perfectly match → both buckets are 1.0; the
    # override doesn't change the result on a perfect-match fixture
    # but does change which category lands in which bucket.
    # n_things should now reflect category 2; n_stuff should reflect
    # category 1.
    assert summary.n_things == 1
    assert summary.n_stuff == 1
    assert summary.pq_things == pytest.approx(1.0)
    assert summary.pq_stuff == pytest.approx(1.0)


def test_evaluate_with_pq_iou_threshold_threads_through() -> None:
    """A threshold of 1.0 + 1ulp is unmatchable; perfect-match fixture
    falls to all-FN. Pins that the parameter actually reaches the
    matcher."""
    gt, dt = _make_two_class_pair()
    # Threshold > 1.0 makes matching impossible (the gate is `iou >
    # threshold` and IoU is in [0, 1]).
    summary = Evaluator(pq_iou_threshold=1.0, things_stuff_split=False).evaluate(gt, dt)
    assert summary.pq == pytest.approx(0.0)


def test_evaluate_to_partial_with_custom_params_raises() -> None:
    """Streaming kernel plumbing for ADR-0042 custom params is a
    follow-up; gate at the Python boundary."""
    e = Evaluator(pq_iou_threshold=0.7)
    with pytest.raises(InvalidPanopticParams, match="evaluate_to_partial"):
        e.evaluate_to_partial(iter([]), categories=b"[]", rank_id=0)
