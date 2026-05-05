"""Week 2.2 (ADR-0019) tests for the user-facing EvalResult surface.

Exercises the new ``Evaluator.evaluate(tables=...)`` keyword, the
per_image table, the schema pinning, and the lazy-polars import. The
zero-overhead microbenchmark is in
``tests/python/bench/test_zero_overhead_default_path.py``.
"""

from __future__ import annotations

import json
import sys

import pytest

from vernier.instance import EvalResult, Evaluator, Summary

# Two images, two categories. Image 1: perfect-match DT for cat 1.
# Image 2: unmatched DT (FP) for cat 2 (DT is far from the GT).
_GT = json.dumps(
    {
        "images": [
            {"id": 1, "width": 100, "height": 100},
            {"id": 2, "width": 100, "height": 100},
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
        ],
        "categories": [
            {"id": 1, "name": "alpha"},
            {"id": 2, "name": "beta"},
        ],
    }
).encode()

_DT = json.dumps(
    [
        {"image_id": 1, "category_id": 1, "score": 0.9, "bbox": [0, 0, 10, 10]},
        {"image_id": 2, "category_id": 2, "score": 0.8, "bbox": [50, 50, 10, 10]},
    ]
).encode()


def test_evaluate_default_path_returns_summary_unchanged() -> None:
    """`tables=None` must return the existing Summary type — bit-identical
    to the 0.0.1 release. This is the headline 'zero overhead' contract."""
    summary = Evaluator().evaluate(_GT, _DT)
    assert isinstance(summary, Summary)
    # Sanity: 12 detection stats present.
    assert len(summary.stats) == 12


def test_evaluate_with_tables_all_returns_eval_result_with_two_dataframes() -> None:
    polars = pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    result = Evaluator().evaluate(_GT, _DT, tables="all")
    assert isinstance(result, EvalResult)
    # Default-shape Summary is preserved on the wrapper.
    assert isinstance(result.summary, Summary)
    # `result.stats` is a pass-through.
    assert result.stats == result.summary.stats

    per_image = result.per_image
    per_class = result.per_class
    assert isinstance(per_image, polars.DataFrame)
    assert isinstance(per_class, polars.DataFrame)
    assert per_image.height == 2
    assert per_class.height == 2


def test_evaluate_with_tables_tuple_subset_only_builds_requested() -> None:
    polars = pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    result = Evaluator().evaluate(_GT, _DT, tables=("per_class",))
    assert isinstance(result, EvalResult)
    assert isinstance(result.per_class, polars.DataFrame)
    # per_image was not requested — accessing must raise a clear
    # RuntimeError (no silent build).
    with pytest.raises(RuntimeError, match="per_image"):
        _ = result.per_image


def test_per_image_schema_omits_ap_and_ap50_columns() -> None:
    """ADR-0019 §`per_image` deliberately omits per-image AP / AP50;
    the schema is encoded in the test, not just the docs (per-image AP
    explanation in `docs/explanation/why-no-per-image-ap.md`)."""
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    result = Evaluator().evaluate(_GT, _DT, tables=("per_image",))
    cols = result.per_image.columns
    # Pinned schema (10 columns).
    assert cols == [
        "image_id",
        "n_gt",
        "n_dt",
        "tp_at_50",
        "fp_at_50",
        "fn_at_50",
        "tp_at_75",
        "fp_at_75",
        "fn_at_75",
        "tp_mean_iou",
    ]
    assert "ap" not in cols
    assert "ap_50" not in cols


def test_per_image_tp_fp_fn_match_hand_counted_values() -> None:
    """Image 1 has a perfect-match DT (TP at every threshold). Image 2
    has an unmatched DT (FP at every threshold) and an unmatched GT (FN
    at every threshold). Hand-counted expectations."""
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    result = Evaluator().evaluate(_GT, _DT, tables=("per_image",))
    df = result.per_image
    # Sort by image_id so the row order is deterministic.
    df = df.sort("image_id")
    assert df["image_id"].to_list() == [1, 2]
    assert df["n_gt"].to_list() == [1, 1]
    assert df["n_dt"].to_list() == [1, 1]
    assert df["tp_at_50"].to_list() == [1, 0]
    assert df["fp_at_50"].to_list() == [0, 1]
    assert df["fn_at_50"].to_list() == [0, 1]
    # IoU=0.75 is below the perfect-match IoU=1.0; same TP shape.
    assert df["tp_at_75"].to_list() == [1, 0]
    assert df["fp_at_75"].to_list() == [0, 1]
    assert df["fn_at_75"].to_list() == [0, 1]


def test_polars_imported_lazily_only_after_first_access() -> None:
    """Importing vernier and running an evaluate(tables=None) cycle
    must not pull in polars. The first attribute access on EvalResult
    is what triggers the import. Mitigates wheel cold-start cost for
    callers who never touch the table API."""
    if "polars" in sys.modules:
        pytest.skip(
            "polars already imported by another test in this session; "
            "the lazy-import contract is asserted at the FFI level too"
        )
    # Default-path eval — must not import polars.
    Evaluator().evaluate(_GT, _DT)
    assert "polars" not in sys.modules

    # Tables-path eval — still must not import polars until the
    # cached_property is read.
    result = Evaluator().evaluate(_GT, _DT, tables=("per_class",))
    assert "polars" not in sys.modules

    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    _ = result.per_class
    assert "polars" in sys.modules


def test_evaluate_tables_with_dataset_handle_is_not_yet_supported() -> None:
    """Passing a Dataset handle on the tables path is reserved for
    Week 2.5 (when streaming/background integration also lands).
    Document the limitation as a typed error so callers don't get a
    misleading low-level message."""
    from vernier.instance import CocoDataset

    ds = CocoDataset.from_json(_GT)
    with pytest.raises(NotImplementedError, match="bytes"):
        Evaluator().evaluate(ds, _DT, tables=("per_class",))


def test_per_class_table_alignes_with_summary_ap() -> None:
    """Cross-check: mean of per_class.ap (over non-null cats) must
    agree with Summary.stats[0]. Both come from the same Accumulated;
    the table is presentation, not parallel computation."""
    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    result = Evaluator().evaluate(_GT, _DT, tables="all")
    table_ap_values = [v for v in result.per_class["ap"].to_list() if v is not None]
    assert table_ap_values
    table_mean = sum(table_ap_values) / len(table_ap_values)
    assert table_mean == pytest.approx(result.summary.stats[0])
