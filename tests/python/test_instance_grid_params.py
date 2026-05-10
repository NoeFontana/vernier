"""Tests for the instance grid-parameter surface (ADR-0040).

Covers the three new fields on ``vernier.instance.Evaluator``:

- ``iou_thresholds: tuple[float, ...] | None``
- ``recall_thresholds: tuple[float, ...] | None``
- ``area_ranges: Breakdown | None``

Plus the construction-time validation, the
``IncompatibleSummaryPlan`` redirect from ``evaluate(...)``, the
``evaluate_tables(...)`` redirect target, and the COCOeval shim
``__setattr__`` guard.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vernier._compat import _Params
from vernier.instance import (
    Bbox,
    Breakdown,
    Evaluator,
    IncompatibleSummaryPlan,
    InvalidInstanceParams,
)

# --- defaults: existing behavior is unchanged --------------------------------


def test_default_evaluator_has_none_for_all_grid_fields() -> None:
    e = Evaluator()
    assert e.iou_thresholds is None
    assert e.recall_thresholds is None
    assert e.area_ranges is None


def test_default_evaluator_evaluate_path_unaffected() -> None:
    """A default Evaluator never trips IncompatibleSummaryPlan; the
    redirect only fires when a custom-grid field is set."""
    e = Evaluator()
    assert e._has_custom_grid() is False


# --- iou_thresholds validation ----------------------------------------------


def test_iou_thresholds_accepts_canonical_ladder() -> None:
    Evaluator(iou_thresholds=tuple(round(0.5 + 0.05 * i, 2) for i in range(10)))


def test_iou_thresholds_rejects_empty() -> None:
    with pytest.raises(InvalidInstanceParams, match="non-empty"):
        Evaluator(iou_thresholds=())


def test_iou_thresholds_rejects_out_of_range() -> None:
    with pytest.raises(InvalidInstanceParams, match=r"\[0.0, 1.0\]"):
        Evaluator(iou_thresholds=(1.5,))


def test_iou_thresholds_rejects_negative() -> None:
    with pytest.raises(InvalidInstanceParams, match=r"\[0.0, 1.0\]"):
        Evaluator(iou_thresholds=(-0.1,))


def test_iou_thresholds_rejects_unsorted() -> None:
    with pytest.raises(InvalidInstanceParams, match="sorted"):
        Evaluator(iou_thresholds=(0.5, 0.3, 0.7))


def test_iou_thresholds_rejects_duplicates() -> None:
    with pytest.raises(InvalidInstanceParams, match="duplicate"):
        Evaluator(iou_thresholds=(0.5, 0.5))


def test_iou_thresholds_rejects_nan() -> None:
    with pytest.raises(InvalidInstanceParams, match="finite"):
        Evaluator(iou_thresholds=(float("nan"),))


def test_iou_thresholds_rejects_inf() -> None:
    with pytest.raises(InvalidInstanceParams, match="finite"):
        Evaluator(iou_thresholds=(float("inf"),))


# --- recall_thresholds validation -------------------------------------------


def test_recall_thresholds_uses_same_validator() -> None:
    # Spot-check that recall_thresholds gets the same validation rule
    # (tested exhaustively under iou_thresholds above).
    with pytest.raises(InvalidInstanceParams, match="non-empty"):
        Evaluator(recall_thresholds=())
    with pytest.raises(InvalidInstanceParams, match="duplicate"):
        Evaluator(recall_thresholds=(0.5, 0.5))


# --- area_ranges validation -------------------------------------------------


def test_area_ranges_accepts_range_breakdown() -> None:
    bd = Breakdown.from_ranges("area", [("all", 0.0, 1e10), ("small", 0.0, 1024.0)])
    Evaluator(area_ranges=bd)


def test_area_ranges_rejects_class_groups_breakdown() -> None:
    bd = Breakdown.from_class_groups("g", [("a", [1])])
    with pytest.raises(InvalidInstanceParams, match="range Breakdown"):
        Evaluator(area_ranges=bd)


# --- IncompatibleSummaryPlan redirect ---------------------------------------


_FIXTURE_GT = json.dumps(
    {
        "images": [{"id": 1, "width": 64, "height": 64}],
        "categories": [{"id": 1, "name": "x"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 10.0, 20.0, 20.0],
                "area": 400.0,
                "iscrowd": 0,
            }
        ],
    }
).encode("utf-8")
_FIXTURE_DT = json.dumps(
    [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [10.0, 10.0, 20.0, 20.0],
            "score": 0.9,
        }
    ]
).encode("utf-8")


def test_evaluate_raises_on_iou_thresholds_set() -> None:
    e = Evaluator(iou=Bbox(), iou_thresholds=(0.5,))
    with pytest.raises(IncompatibleSummaryPlan) as ei:
        e.evaluate(_FIXTURE_GT, _FIXTURE_DT)
    assert ei.value.field == "iou_thresholds"
    assert "evaluate_tables" in ei.value.remediation


def test_evaluate_raises_on_recall_thresholds_set() -> None:
    e = Evaluator(iou=Bbox(), recall_thresholds=(0.0, 0.5, 1.0))
    with pytest.raises(IncompatibleSummaryPlan) as ei:
        e.evaluate(_FIXTURE_GT, _FIXTURE_DT)
    assert ei.value.field == "recall_thresholds"


def test_evaluate_raises_on_area_ranges_set() -> None:
    bd = Breakdown.from_ranges("area", [("all", 0.0, 1e10)])
    e = Evaluator(iou=Bbox(), area_ranges=bd)
    with pytest.raises(IncompatibleSummaryPlan) as ei:
        e.evaluate(_FIXTURE_GT, _FIXTURE_DT)
    assert ei.value.field == "area_ranges"


def test_evaluate_with_tables_keyword_also_raises_on_custom_grid() -> None:
    """`evaluate(tables="all")` raises the same exception — the user
    must use `evaluate_tables(...)` explicitly to bypass."""
    e = Evaluator(iou=Bbox(), iou_thresholds=(0.5,))
    with pytest.raises(IncompatibleSummaryPlan):
        e.evaluate(_FIXTURE_GT, _FIXTURE_DT, tables="all")


def test_evaluate_default_grid_returns_summary_unchanged() -> None:
    e = Evaluator(iou=Bbox())
    summary = e.evaluate(_FIXTURE_GT, _FIXTURE_DT)
    assert len(summary.stats) == 12  # canonical 12-stat plan


# --- evaluate_tables redirect target ----------------------------------------


def test_evaluate_tables_default_grid_returns_eval_result() -> None:
    e = Evaluator(iou=Bbox())
    result = e.evaluate_tables(_FIXTURE_GT, _FIXTURE_DT)
    assert result.summary is not None
    assert len(result.summary.stats) == 12


def test_evaluate_tables_custom_grid_raises_not_implemented_for_now() -> None:
    """PR 2 ships the surface; kernel plumbing for custom grids is a
    follow-up (params_hash split lands first). The redirect target
    raises a clear message until that lands."""
    e = Evaluator(iou=Bbox(), iou_thresholds=(0.5,))
    with pytest.raises(NotImplementedError, match="kernel-side custom-grid plumbing"):
        e.evaluate_tables(_FIXTURE_GT, _FIXTURE_DT)


# --- with_options threading -------------------------------------------------


def test_with_options_unset_default_leaves_grid_fields_unchanged() -> None:
    e = Evaluator(iou_thresholds=(0.5,))
    e2 = e.with_options(parity_mode="strict")
    assert e2.iou_thresholds == (0.5,)


def test_with_options_explicit_none_resets_grid_field() -> None:
    e = Evaluator(iou_thresholds=(0.5,))
    e2 = e.with_options(iou_thresholds=None)
    assert e2.iou_thresholds is None


def test_with_options_explicit_value_sets_grid_field() -> None:
    e = Evaluator()
    e2 = e.with_options(recall_thresholds=(0.0, 1.0))
    assert e2.recall_thresholds == (0.0, 1.0)


# --- COCOeval shim guard ----------------------------------------------------


def _make_shim_params() -> _Params:
    return _Params()


@pytest.mark.parametrize(
    "name",
    ["iou_thresholds", "recall_thresholds", "area_ranges"],
)
def test_compat_params_setattr_rejects_native_field_names(name: str) -> None:
    p = _make_shim_params()
    with pytest.raises(AttributeError, match=r"vernier\.instance\.Evaluator"):
        setattr(p, name, "anything")


def test_compat_params_pycocotools_camelcase_still_mutates() -> None:
    """The migration-path camelCase fields (iouThrs, recThrs, areaRng)
    keep mutating normally — the guard is targeted at the native
    snake_case names only."""
    p = _make_shim_params()
    new_thrs = np.array([0.5, 0.75], dtype=np.float64)
    p.iouThrs = new_thrs
    assert (p.iouThrs == new_thrs).all()
