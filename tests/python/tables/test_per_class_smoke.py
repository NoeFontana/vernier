"""Week 2.1 (ADR-0019) smoke test for the per_class result table.

Validates the end-to-end FFI plumbing: vernier-core builds the
``PerClassTable``, vernier-ffi packs it into an Arrow ``RecordBatch``,
exposes it via the Arrow PyCapsule Interface, and polars consumes it
zero-copy. Spot-checks one AP value against the synchronous
``Summary.stats`` to catch any drift between the table builder and the
canonical 12-stat summary.

This is the *first end-to-end through the new pipeline*; later phases
add cross-language round-trip (polars/pandas/duckdb) and lazy-import
assertions in their own files.
"""

from __future__ import annotations

import json

import pytest

import vernier
from vernier import Dataset, Evaluator
from vernier._core import per_class_to_arrow_pycapsule

# Two perfectly-overlapping detections on a single image, two
# categories. Mirrors the shape of tests/python/test_evaluator.py's
# perfect_match fixture but adds a second category so per_class has
# more than one row to exercise.
_GT = json.dumps(
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
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [50, 50, 10, 10],
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
        {"image_id": 1, "category_id": 2, "score": 0.8, "bbox": [50, 50, 10, 10]},
    ]
).encode()


def _build_grid_and_accum() -> tuple[object, object, Dataset]:
    """Build the FFI primitives the table entry point consumes.

    Return type is `object` for the grid and accumulator because they
    are FFI-internal types deliberately not re-exported on
    ``vernier`` — callers don't construct them directly today, and the
    public API will hide them behind ``EvalResult`` in Week 2.2.
    """
    grid = vernier._core.evaluate_bbox_grid(_GT, _DT, "corrected", 100, True)
    accum = grid.accumulate([1, 10, 100])
    dataset = Dataset.from_json(_GT)
    return grid, accum, dataset


def test_per_class_arrow_pycapsule_round_trips_through_polars() -> None:
    polars = pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    grid, accum, dataset = _build_grid_and_accum()

    batch = per_class_to_arrow_pycapsule(grid, accum, dataset)

    # Sanity check the producer side: __arrow_c_array__ must return two
    # PyCapsules, one for the schema and one for the array, named per
    # the Arrow C-Data-Interface spec.
    schema_capsule, array_capsule = batch.__arrow_c_array__()
    assert type(schema_capsule).__name__ == "PyCapsule"
    assert type(array_capsule).__name__ == "PyCapsule"

    df = polars.from_arrow(batch)
    # polars sometimes wraps a single batch as a DataFrame, sometimes as
    # a Series (chunked array path). Accept both — the producer
    # guarantee is the protocol, not the consumer-side wrapping.
    if isinstance(df, polars.Series):
        df = df.to_frame()
    assert isinstance(df, polars.DataFrame)

    # Schema pin: 13 columns in pinned order. Any rename/add is gated on
    # an explicit edit to this test (which doubles as the schema golden
    # for Week 2.1; Week 2.2 splits this out into a JSON golden).
    expected_columns = [
        "category_id",
        "category_name",
        "ap",
        "ap50",
        "ap75",
        "ap_s",
        "ap_m",
        "ap_l",
        "ar_max_1",
        "ar_max_10",
        "ar_max_100",
        "n_gt",
        "n_dt",
    ]
    assert list(df.columns) == expected_columns

    # 2 categories → 2 rows, sorted by category_id ascending (matches
    # the grid's K-axis ordering).
    assert df.height == 2
    assert df["category_id"].to_list() == [1, 2]
    assert df["category_name"].to_list() == ["alpha", "beta"]
    assert df["n_gt"].to_list() == [1, 1]
    assert df["n_dt"].to_list() == [1, 1]

    # Spot-check: AP@.50:.95 area=all per-category should be 1.0 on a
    # perfect-match fixture. The values should be drawn from the same
    # accumulator the synchronous Summary uses, so they match within
    # f64 epsilon.
    ap_values = df["ap"].to_list()
    assert ap_values[0] == pytest.approx(1.0)
    assert ap_values[1] == pytest.approx(1.0)


def test_per_class_table_alignes_with_synchronous_summary_stats() -> None:
    """Mean of per_class.ap (over non-null categories) must match the
    canonical Summary.stats[0] (AP across all categories).

    Both are computed from the same Accumulated tensor; the table
    surface is a presentation, not a parallel computation.
    """
    polars = pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    grid, accum, dataset = _build_grid_and_accum()
    batch = per_class_to_arrow_pycapsule(grid, accum, dataset)
    df = polars.from_arrow(batch)
    if isinstance(df, polars.Series):
        df = df.to_frame()

    summary = Evaluator().evaluate(_GT, _DT)
    summary_ap = summary.stats[0]  # AP @ all areas, all IoUs, maxDets=largest

    # All categories on this fixture populate the AP cell; the
    # arithmetic mean of two ones is one. (We don't assert the
    # filter-then-mean equivalence here — that's Week 2.2's
    # cross-language round-trip's job.)
    table_ap_values = [v for v in df["ap"].to_list() if v is not None]
    assert table_ap_values
    assert sum(table_ap_values) / len(table_ap_values) == pytest.approx(summary_ap)
