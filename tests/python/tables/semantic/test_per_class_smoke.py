"""ADR-0038 smoke test for ``vernier.semantic`` per_class result table.

Validates the end-to-end FFI plumbing: vernier-semantic builds the
``PerClassTable`` from a semantic summary plus its retained
``ConfusionMatrix``, vernier-ffi packs it into an Arrow ``RecordBatch``
through the PyCapsule Interface, and polars consumes it zero-copy.
The 9-column shape matches the canonical mmseg / ADE20K research
report (per ADR-0036's vendored oracle).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import vernier.semantic as vs

# 2-class, 4-pixel image. GT = [0, 0, 1, 1], DT = [0, 1, 1, 1].
# Confusion: counts[0,0]=1, counts[0,1]=1, counts[1,1]=2.
# Per-class:
#   class 0: TP=1, FP=0, FN=1, n_gt=2, n_dt=1, IoU=0.5,  recall=0.5,  precision=1.0
#   class 1: TP=2, FP=1, FN=0, n_gt=2, n_dt=3, IoU=2/3,  recall=1.0,  precision=2/3
_GT = np.array([[0, 0, 1, 1]], dtype=np.uint32)
_DT = np.array([[0, 1, 1, 1]], dtype=np.uint32)


def _build_inputs() -> tuple[vs.Dataset, vs.Predictions]:
    gt = vs.Dataset.from_arrays({1: _GT}, n_classes=2)
    dt = vs.Predictions.from_arrays({1: _DT})
    return gt, dt


def test_per_class_arrow_pycapsule_round_trips_through_polars() -> None:
    polars = pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    gt, dt = _build_inputs()

    result = vs.Evaluator().evaluate(gt, dt, tables=("per_class",))
    assert isinstance(result, vs.EvalResult)

    df = result.per_class
    assert isinstance(df, polars.DataFrame)

    # Schema pin: 9 columns in mmseg / ADE20K order.
    expected_columns = [
        "category_id",
        "iou",
        "accuracy",
        "precision",
        "n_gt_pixels",
        "n_dt_pixels",
        "tp_pixels",
        "fp_pixels",
        "fn_pixels",
    ]
    assert list(df.columns) == expected_columns

    assert df.height == 2
    assert df["category_id"].to_list() == [0, 1]
    assert df["iou"].to_list() == [pytest.approx(0.5), pytest.approx(2.0 / 3.0)]
    assert df["accuracy"].to_list() == [pytest.approx(0.5), pytest.approx(1.0)]
    assert df["precision"].to_list() == [pytest.approx(1.0), pytest.approx(2.0 / 3.0)]
    assert df["n_gt_pixels"].to_list() == [2, 2]
    assert df["n_dt_pixels"].to_list() == [1, 3]
    assert df["tp_pixels"].to_list() == [1, 2]
    assert df["fp_pixels"].to_list() == [0, 1]
    assert df["fn_pixels"].to_list() == [1, 0]


def test_per_class_table_aligns_with_summary_per_class() -> None:
    """Per-row IoU/recall/precision must equal
    ``summary.per_class[cls].iou`` etc. The table is a presentation
    over the retained ``ClassSemanticStats`` map plus a single
    diagonal-read off the confusion matrix."""
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    gt, dt = _build_inputs()
    result = vs.Evaluator().evaluate(gt, dt, tables=("per_class",))
    summary_per_class = result.summary.per_class()
    for row in result.per_class.iter_rows(named=True):
        cls = row["category_id"]
        assert row["iou"] == summary_per_class[cls].iou
        assert row["accuracy"] == summary_per_class[cls].accuracy
        assert row["precision"] == summary_per_class[cls].precision


def test_polars_imported_lazily_only_after_first_access() -> None:
    if "polars" in sys.modules:
        pytest.skip(
            "polars already imported by another test in this session; "
            "the lazy-import contract is asserted at the FFI level too"
        )
    gt, dt = _build_inputs()

    vs.Evaluator().evaluate(gt, dt)
    assert "polars" not in sys.modules

    result = vs.Evaluator().evaluate(gt, dt, tables=("per_class",))
    assert "polars" not in sys.modules

    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    _ = result.per_class
    assert "polars" in sys.modules


def test_tables_all_alias_expands_to_supported_set() -> None:
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    gt, dt = _build_inputs()
    result = vs.Evaluator().evaluate(gt, dt, tables="all")
    assert result.per_class is not None


def test_bare_string_tables_arg_rejected() -> None:
    gt, dt = _build_inputs()
    with pytest.raises(ValueError, match="bare string"):
        vs.Evaluator().evaluate(gt, dt, tables="per_class")  # type: ignore[arg-type]
