"""Tests for the Python ``Breakdown`` type (ADR-0039 Phase 2A + Phase 2B).

The Rust ``Breakdown`` lifts to Python as a single class with two
factories — ``from_ranges`` (f64-keyed area-style buckets) and
``from_class_groups`` (class-id partitions for semantic / panoptic
grouping) — and read-only accessors keyed by the active variant.
Construction validates inputs and raises ``ValueError`` on degenerate
shape; accessing the wrong-variant getter raises ``AttributeError``
with a pointer to the correct one.
"""

from __future__ import annotations

import pytest

from vernier import instance, panoptic, semantic


def test_breakdown_class_object_is_shared_across_paradigms() -> None:
    assert instance.Breakdown is semantic.Breakdown
    assert instance.Breakdown is panoptic.Breakdown


def test_from_ranges_round_trips_canonical_coco_area_layout() -> None:
    b = instance.Breakdown.from_ranges(
        "area",
        [
            ("all", 0.0, 1e10),
            ("small", 0.0, 32.0**2),
            ("medium", 32.0**2, 96.0**2),
            ("large", 96.0**2, 1e10),
        ],
    )
    assert b.axis == "area"
    assert len(b) == 4
    assert b.buckets[0] == ("all", 0.0, 1e10)
    assert b.buckets[1] == ("small", 0.0, 1024.0)
    assert b.buckets[2] == ("medium", 1024.0, 9216.0)
    assert b.buckets[3] == ("large", 9216.0, 1e10)


def test_from_ranges_supports_custom_axis_and_labels() -> None:
    b = instance.Breakdown.from_ranges(
        "distance_m",
        [("near", 0.0, 5.0), ("mid", 5.0, 15.0), ("far", 15.0, 100.0)],
    )
    assert b.axis == "distance_m"
    assert [label for label, _, _ in b.buckets] == ["near", "mid", "far"]


def test_repr_shape() -> None:
    b = instance.Breakdown.from_ranges("area", [("all", 0.0, 1.0)])
    assert repr(b) == 'Breakdown(kind="range", axis="area", len=1)'


def test_equality_compares_structurally() -> None:
    b1 = instance.Breakdown.from_ranges("area", [("a", 0.0, 1.0), ("b", 1.0, 2.0)])
    b2 = instance.Breakdown.from_ranges("area", [("a", 0.0, 1.0), ("b", 1.0, 2.0)])
    b3 = instance.Breakdown.from_ranges("area", [("a", 0.0, 1.0), ("c", 1.0, 2.0)])
    b4 = instance.Breakdown.from_ranges("other", [("a", 0.0, 1.0), ("b", 1.0, 2.0)])
    assert b1 == b2
    assert b1 != b3
    assert b1 != b4


def test_from_ranges_rejects_empty_buckets() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        instance.Breakdown.from_ranges("area", [])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_from_ranges_rejects_nonfinite_lo(bad: float) -> None:
    with pytest.raises(ValueError, match="lo must be finite"):
        instance.Breakdown.from_ranges("area", [("x", bad, 1.0)])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_from_ranges_rejects_nonfinite_hi(bad: float) -> None:
    with pytest.raises(ValueError, match="hi must be finite"):
        instance.Breakdown.from_ranges("area", [("x", 0.0, bad)])


def test_from_ranges_rejects_negative_lo() -> None:
    with pytest.raises(ValueError, match="lo must be >= 0"):
        instance.Breakdown.from_ranges("area", [("x", -1.0, 1.0)])


def test_from_ranges_rejects_lo_greater_than_hi() -> None:
    with pytest.raises(ValueError, match="lo <= hi"):
        instance.Breakdown.from_ranges("area", [("x", 100.0, 50.0)])


def test_from_ranges_rejects_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="duplicate bucket label"):
        instance.Breakdown.from_ranges("area", [("dup", 0.0, 1.0), ("dup", 1.0, 2.0)])


def test_from_ranges_accepts_lo_equal_hi() -> None:
    """Closed-on-both-ends inclusion (ADR-0016 quirk D6) means a degenerate
    bucket with ``lo == hi`` is valid — it matches keys exactly equal to
    that single value."""
    b = instance.Breakdown.from_ranges("area", [("point", 5.0, 5.0)])
    assert b.buckets[0] == ("point", 5.0, 5.0)


def test_from_ranges_kind_is_range() -> None:
    b = instance.Breakdown.from_ranges("area", [("all", 0.0, 1.0)])
    assert b.kind == "range"


def test_range_breakdown_class_groups_getter_raises() -> None:
    b = instance.Breakdown.from_ranges("area", [("all", 0.0, 1.0)])
    with pytest.raises(AttributeError, match="from_ranges"):
        _ = b.class_groups


def test_from_class_groups_round_trips_basic_partition() -> None:
    b = instance.Breakdown.from_class_groups(
        "vehicle_taxonomy",
        [("cars", [3, 6]), ("trucks", [8, 10])],
    )
    assert b.kind == "class_groups"
    assert b.axis == "vehicle_taxonomy"
    assert len(b) == 2
    assert b.class_groups == [("cars", [3, 6]), ("trucks", [8, 10])]


def test_from_class_groups_repr_shape() -> None:
    b = instance.Breakdown.from_class_groups("g", [("a", [1])])
    assert repr(b) == 'Breakdown(kind="class_groups", axis="g", len=1)'


def test_class_groups_breakdown_buckets_getter_raises() -> None:
    b = instance.Breakdown.from_class_groups("g", [("a", [1])])
    with pytest.raises(AttributeError, match="from_class_groups"):
        _ = b.buckets


def test_from_class_groups_rejects_empty_groups() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        instance.Breakdown.from_class_groups("g", [])


def test_from_class_groups_rejects_empty_class_ids() -> None:
    with pytest.raises(ValueError, match="empty class_ids"):
        instance.Breakdown.from_class_groups("g", [("a", []), ("b", [1])])


def test_from_class_groups_rejects_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="duplicate group label"):
        instance.Breakdown.from_class_groups("g", [("dup", [1]), ("dup", [2])])


def test_from_class_groups_rejects_partition_violation() -> None:
    with pytest.raises(ValueError, match="appears in multiple groups"):
        instance.Breakdown.from_class_groups("g", [("a", [1, 2, 3]), ("b", [3, 4])])


def test_class_groups_equality_is_structural() -> None:
    b1 = instance.Breakdown.from_class_groups("g", [("a", [1, 2])])
    b2 = instance.Breakdown.from_class_groups("g", [("a", [1, 2])])
    b3 = instance.Breakdown.from_class_groups("g", [("a", [1, 3])])
    b_range = instance.Breakdown.from_ranges("g", [("a", 0.0, 1.0)])
    assert b1 == b2
    assert b1 != b3
    assert b1 != b_range  # different variant => not equal even when shapes look similar
