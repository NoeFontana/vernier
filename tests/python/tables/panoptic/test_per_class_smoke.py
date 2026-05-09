"""ADR-0038 smoke test for ``vernier.panoptic`` per_class result table.

Validates the end-to-end FFI plumbing: vernier-panoptic builds the
``PerClassTable`` from a panoptic summary, vernier-ffi packs it into
an Arrow ``RecordBatch``, exposes it via the Arrow PyCapsule
Interface, and polars consumes it zero-copy. Spot-checks PQ values
against the canonical ``Summary.per_class`` map to catch any drift
between the table builder and the summary itself.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

import vernier.panoptic as vp

# 1x10 image, two GT segments perfectly covered by two DT segments.
# cat 100 is "thing", cat 200 is "stuff". Both pairs produce iou=1.0.
_GT_LABEL = np.array([[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]], dtype=np.uint32)
_DT_LABEL = np.array([[10, 10, 10, 10, 10, 11, 11, 11, 11, 11]], dtype=np.uint32)
_GT_SEGS = json.dumps(
    {
        "1": [
            {"id": 1, "category_id": 100, "iscrowd": False, "area": 5},
            {"id": 2, "category_id": 200, "iscrowd": False, "area": 5},
        ]
    }
).encode()
_DT_SEGS = json.dumps(
    {
        "1": [
            {"id": 10, "category_id": 100, "iscrowd": False, "area": 5},
            {"id": 11, "category_id": 200, "iscrowd": False, "area": 5},
        ]
    }
).encode()
_CATS = json.dumps([{"id": 100, "isthing": True}, {"id": 200, "isthing": False}]).encode()


def _build_inputs() -> tuple[vp.Dataset, vp.Predictions]:
    gt = vp.Dataset.from_arrays({1: _GT_LABEL}, _GT_SEGS, _CATS)
    dt = vp.Predictions.from_arrays({1: _DT_LABEL}, _DT_SEGS)
    return gt, dt


def test_per_class_arrow_pycapsule_round_trips_through_polars() -> None:
    polars = pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    gt, dt = _build_inputs()

    result = vp.Evaluator().evaluate(gt, dt, tables=("per_class",))
    assert isinstance(result, vp.EvalResult)
    assert isinstance(result.summary, type(vp.Evaluator().evaluate(gt, dt)))

    df = result.per_class
    assert isinstance(df, polars.DataFrame)

    # Schema pin: 8 columns in pinned order. Any rename or addition is
    # gated on an explicit edit to this list (which is the schema
    # golden for ADR-0038's panoptic per_class table).
    expected_columns = [
        "category_id",
        "pq",
        "sq",
        "rq",
        "n_tp",
        "n_fp",
        "n_fn",
        "iou_sum",
    ]
    assert list(df.columns) == expected_columns

    # 2 categories, BTreeMap order on category_id ascending.
    assert df.height == 2
    assert df["category_id"].to_list() == [100, 200]
    assert df["pq"].to_list() == [1.0, 1.0]
    assert df["sq"].to_list() == [1.0, 1.0]
    assert df["rq"].to_list() == [1.0, 1.0]
    assert df["n_tp"].to_list() == [1, 1]
    assert df["n_fp"].to_list() == [0, 0]
    assert df["n_fn"].to_list() == [0, 0]
    assert df["iou_sum"].to_list() == [1.0, 1.0]


def test_per_class_table_aligns_with_summary_per_class() -> None:
    """Each row's PQ must equal ``summary.per_class[cat].pq`` to within
    f64 ULP — the table is a presentation, not a parallel computation."""
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    gt, dt = _build_inputs()
    result = vp.Evaluator().evaluate(gt, dt, tables=("per_class",))
    summary_per_class = result.summary.per_class()  # method, not property
    df_rows = result.per_class.iter_rows(named=True)
    for row in df_rows:
        cat = row["category_id"]
        assert row["pq"] == summary_per_class[cat].pq
        assert row["sq"] == summary_per_class[cat].sq
        assert row["rq"] == summary_per_class[cat].rq
        assert row["iou_sum"] == summary_per_class[cat].iou_sum


def test_polars_imported_lazily_only_after_first_access() -> None:
    """Importing vernier.panoptic and calling evaluate without tables=
    must not pull in polars. First attribute access on EvalResult is
    what triggers the import."""
    if "polars" in sys.modules:
        pytest.skip(
            "polars already imported by another test in this session; "
            "the lazy-import contract is asserted at the FFI level too"
        )
    gt, dt = _build_inputs()

    vp.Evaluator().evaluate(gt, dt)
    assert "polars" not in sys.modules

    result = vp.Evaluator().evaluate(gt, dt, tables=("per_class",))
    assert "polars" not in sys.modules

    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    _ = result.per_class
    assert "polars" in sys.modules


def test_tables_all_alias_expands_to_supported_set() -> None:
    """``tables="all"`` must expand to the panoptic-supported subset
    (per_class only in this iteration). Additive when per_image lands."""
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    gt, dt = _build_inputs()
    result = vp.Evaluator().evaluate(gt, dt, tables="all")
    assert result.per_class is not None  # touches cached_property


def test_bare_string_tables_arg_rejected() -> None:
    """Mirrors the instance behavior: a bare string (not a tuple, not
    'all') is rejected with a helpful message pointing at the tuple
    fix."""
    gt, dt = _build_inputs()
    with pytest.raises(ValueError, match="bare string"):
        vp.Evaluator().evaluate(gt, dt, tables="per_class")  # type: ignore[arg-type]
