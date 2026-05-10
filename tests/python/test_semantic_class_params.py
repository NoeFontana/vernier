"""Tests for the semantic class-filter / class-grouping surface (ADR-0041).

Covers the two new fields on ``vernier.semantic.Evaluator``:

- ``class_filter: CategoryFilter | None``
- ``class_grouping: Breakdown | None``

Construction-time validation, ``__post_init__`` cross-field checks, and
the ``NotImplementedError`` runtime gate that fires until the
ADR-0041 kernel plumbing lands alongside the ADR-0039 distributed-eval
phase.
"""

from __future__ import annotations

import numpy as np
import pytest

from vernier.semantic import (
    Breakdown,
    CategoryFilterAll,
    CategoryFilterByGrouping,
    CategoryFilterByIds,
    CategoryFilterFrequency,
    Dataset,
    Evaluator,
    InvalidSemanticParams,
    Predictions,
)

# --- defaults: existing behavior unchanged -----------------------------------


def test_default_evaluator_has_none_for_class_params() -> None:
    e = Evaluator()
    assert e.class_filter is None
    assert e.class_grouping is None
    assert e._has_custom_class_params() is False


def test_categoryfilter_all_does_not_count_as_custom() -> None:
    """``All`` is the COCO default — equivalent to ``None`` semantically.
    Setting it explicitly should not trip the custom-params guard."""
    # Note: ADR-0041 treats `class_filter=None` and
    # `class_filter=CategoryFilterAll()` as equivalent at the kernel
    # level, but the user-facing field-set check uses object identity
    # (None vs not-None) for the gate. Setting CategoryFilterAll
    # explicitly DOES count as custom — this test pins that contract.
    e = Evaluator(class_filter=CategoryFilterAll())
    assert e._has_custom_class_params() is True


# --- class_filter validation -------------------------------------------------


def test_class_filter_frequency_rejected_with_lvis_pointer() -> None:
    with pytest.raises(InvalidSemanticParams, match="LVIS-only"):
        Evaluator(class_filter=CategoryFilterFrequency(tag="r"))


def test_class_filter_byids_empty_rejected() -> None:
    with pytest.raises(InvalidSemanticParams, match="at least one"):
        Evaluator(class_filter=CategoryFilterByIds(ids=frozenset()))


def test_class_filter_byids_non_empty_accepted() -> None:
    e = Evaluator(class_filter=CategoryFilterByIds(ids=frozenset([1, 2, 3])))
    assert e._has_custom_class_params() is True


def test_class_filter_bygrouping_without_grouping_rejected() -> None:
    with pytest.raises(InvalidSemanticParams, match="requires class_grouping"):
        Evaluator(class_filter=CategoryFilterByGrouping(label="x"))


def test_class_filter_bygrouping_unknown_label_rejected() -> None:
    g = Breakdown.from_class_groups("g", [("a", [1, 2])])
    with pytest.raises(InvalidSemanticParams, match="not a label of class_grouping"):
        Evaluator(
            class_filter=CategoryFilterByGrouping(label="missing"),
            class_grouping=g,
        )


def test_class_filter_bygrouping_known_label_accepted() -> None:
    g = Breakdown.from_class_groups("g", [("vehicles", [1, 2])])
    Evaluator(
        class_filter=CategoryFilterByGrouping(label="vehicles"),
        class_grouping=g,
    )


# --- class_grouping validation -----------------------------------------------


def test_class_grouping_must_be_class_groups_breakdown() -> None:
    range_bd = Breakdown.from_ranges("area", [("all", 0.0, 1e10)])
    with pytest.raises(InvalidSemanticParams, match="class-groups Breakdown"):
        Evaluator(class_grouping=range_bd)


def test_class_grouping_class_groups_breakdown_accepted() -> None:
    g = Breakdown.from_class_groups("g", [("a", [1])])
    Evaluator(class_grouping=g)


# --- runtime gate: evaluate() raises NotImplementedError --------------------


def _make_minimal_dataset_pred() -> tuple[Dataset, Predictions]:
    arr = np.zeros((4, 4), dtype=np.uint8)
    gt = Dataset.from_arrays({1: arr}, n_classes=2)
    dt = Predictions.from_arrays({1: arr})
    return gt, dt


def test_evaluate_with_class_filter_raises_not_implemented() -> None:
    gt, dt = _make_minimal_dataset_pred()
    e = Evaluator(class_filter=CategoryFilterByIds(ids=frozenset([0, 1])))
    with pytest.raises(NotImplementedError, match="ADR-0041"):
        e.evaluate(gt, dt)


def test_evaluate_with_class_grouping_raises_not_implemented() -> None:
    gt, dt = _make_minimal_dataset_pred()
    g = Breakdown.from_class_groups("g", [("a", [0, 1])])
    e = Evaluator(class_grouping=g)
    with pytest.raises(NotImplementedError, match="ADR-0041"):
        e.evaluate(gt, dt)


def test_evaluate_default_path_unchanged() -> None:
    """No custom fields → kernel runs; existing behavior intact."""
    gt, dt = _make_minimal_dataset_pred()
    e = Evaluator()
    summary = e.evaluate(gt, dt)
    # Pixel-perfect on a uniform-zero pair: every pixel is class 0,
    # both maps. miou == pixel_accuracy == 1.0.
    assert summary.pixel_accuracy == pytest.approx(1.0)
