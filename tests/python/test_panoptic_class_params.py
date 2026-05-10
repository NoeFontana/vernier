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

import pytest

from vernier.panoptic import (
    Breakdown,
    CategoryFilterByGrouping,
    CategoryFilterByIds,
    CategoryFilterFrequency,
    Evaluator,
    InvalidPanopticParams,
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
