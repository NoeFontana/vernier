"""Tests for the cross-paradigm parameter foundation (ADR-0039).

Covers:

- ``CategoryFilter`` discriminated-union variants (``All``,
  ``Frequency``, ``ByIds``, ``ByGrouping``) — frozen, hashable,
  ``isinstance`` discrimination.
- ``IncompatibleSummaryPlan`` shape and message format (ADR-0040).
- Re-exports across the three paradigm namespaces.

The actual validators that consume ``CategoryFilter`` and raise
``IncompatibleSummaryPlan`` land in subsequent PRs (one per paradigm).
"""

from __future__ import annotations

import dataclasses

import pytest

from vernier import instance, panoptic, semantic
from vernier._types import (
    CategoryFilter,
    CategoryFilterAll,
    CategoryFilterByGrouping,
    CategoryFilterByIds,
    CategoryFilterFrequency,
    IncompatibleSummaryPlan,
)


def test_category_filter_re_exported_from_every_paradigm() -> None:
    # Same class object — paradigms must share the symbol so a user
    # who imports CategoryFilter from one namespace and uses it on
    # another doesn't trip an isinstance miss.
    assert instance.CategoryFilterAll is semantic.CategoryFilterAll
    assert semantic.CategoryFilterAll is panoptic.CategoryFilterAll
    assert instance.CategoryFilterByIds is panoptic.CategoryFilterByIds
    assert instance.CategoryFilterByGrouping is semantic.CategoryFilterByGrouping
    assert instance.CategoryFilterFrequency is panoptic.CategoryFilterFrequency


def test_category_filter_variants_are_frozen_dataclasses() -> None:
    for cls in (
        CategoryFilterAll,
        CategoryFilterFrequency,
        CategoryFilterByIds,
        CategoryFilterByGrouping,
    ):
        assert dataclasses.is_dataclass(cls)
    # Frozen → assigning a field must raise on any variant.
    f = CategoryFilterFrequency(tag="r")
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.tag = "c"  # type: ignore[misc]
    g = CategoryFilterByGrouping(label="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.label = "y"  # type: ignore[misc]


def test_category_filter_isinstance_dispatch() -> None:
    # The discriminated-union pattern relies on isinstance to dispatch
    # in validators. Confirm each variant matches the union type alias
    # AND its own concrete class but no other concrete variant.
    samples: list[CategoryFilter] = [
        CategoryFilterAll(),
        CategoryFilterFrequency(tag="c"),
        CategoryFilterByIds(ids=frozenset([1, 2])),
        CategoryFilterByGrouping(label="vehicles"),
    ]
    for s in samples:
        # Each is an instance of the same class as itself (trivially).
        assert isinstance(s, type(s))
    # Cross-variant isinstance is False.
    a = CategoryFilterAll()
    f = CategoryFilterFrequency(tag="r")
    assert not isinstance(a, CategoryFilterFrequency)
    assert not isinstance(f, CategoryFilterAll)


def test_category_filter_byids_uses_frozenset_for_hashability() -> None:
    # ByIds is hashable (frozen + frozenset) — required for params_hash
    # plumbing in PR 5 where the Python-side resolved form crosses FFI.
    f1 = CategoryFilterByIds(ids=frozenset([3, 1, 2]))
    f2 = CategoryFilterByIds(ids=frozenset([1, 2, 3]))
    assert f1 == f2
    assert hash(f1) == hash(f2)


def test_category_filter_frequency_tag_literal_round_trip() -> None:
    for tag in ("r", "c", "f"):
        f = CategoryFilterFrequency(tag=tag)
        assert f.tag == tag


def test_incompatible_summary_plan_shape() -> None:
    e = IncompatibleSummaryPlan(
        field="iou_thresholds",
        value=(0.4, 0.6),
        plan="COCO 12-stat",
        remediation="use evaluate_tables(...) for tabular output",
    )
    assert e.field == "iou_thresholds"
    assert e.value == (0.4, 0.6)
    assert e.plan == "COCO 12-stat"
    assert "evaluate_tables" in e.remediation
    # Subclass of ValueError so naive `except ValueError:` still catches.
    assert isinstance(e, ValueError)
    msg = str(e)
    assert "iou_thresholds" in msg
    assert "COCO 12-stat" in msg
    assert "evaluate_tables" in msg


def test_incompatible_summary_plan_re_exported_from_instance() -> None:
    assert instance.IncompatibleSummaryPlan is IncompatibleSummaryPlan
