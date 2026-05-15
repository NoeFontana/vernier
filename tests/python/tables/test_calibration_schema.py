"""Calibration RecordBatch schema goldens (ADR-0018, Unit 5).

The two tables produced by `EvalCells.calibrate(...)` —
``calibration_reliability`` and ``calibration_per_class`` — each have a
golden JSON schema under ``tests/python/tables/schemas/``. The harness
mints a live ``EvalCells`` handle through
``vernier._core.cells_from_grid`` against a tiny bbox fixture, runs the
summarizer with both ``per_class=False`` and ``per_class=True``, then
compares the resulting Arrow `RecordBatch` schemas (column names,
types, nullability, and the two-key ``vernier.schema_version`` /
``vernier.table`` metadata) against the goldens.

Mirrors ``test_slices_schema.py``; the calibration tables use the
``table_metadata`` (2-key) flavor rather than ``slices_metadata``
(4-key) — they are not slices.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vernier._core import cells_from_grid, evaluate_bbox_grid

from ._schema_assertions import assert_matches_golden

_FIXTURE = Path(__file__).parent.parent / "parity" / "fixtures" / "perfect_match"


def _make_cells() -> object:
    """Build a fresh `EvalCells` handle off the perfect_match bbox fixture.

    Drives the FFI directly (Unit 3's Python wrapper has not landed
    yet); this matches the schema-pinning harness's contract of
    exercising the live Rust builders.
    """
    gt = (_FIXTURE / "gt.json").read_bytes()
    dt_bytes = (_FIXTURE / "dt.json").read_bytes()
    grid = evaluate_bbox_grid(
        gt,
        dt_bytes,
        "strict",
        100,
        True,
        False,
    )
    return cells_from_grid(grid)


def test_calibration_reliability_schema_matches_golden() -> None:
    pa = pytest.importorskip("pyarrow")
    cells = _make_cells()
    # iou=0.5 is index 0 in the default linspace(0.5, 0.95, 10) ladder.
    # min_score=0.0 so the single fixture detection (score=0.9) survives
    # the cutoff; binning="equal_width" sidesteps quantile-edge merging
    # on a one-detection corpus.
    _ece, _mce, _n, _eff, reliability, per_class = cells.calibrate(  # type: ignore[attr-defined]
        0,
        15,
        "equal_width",
        0.0,
        "wilson",
        False,
        "macro",
    )
    assert per_class is None
    batch = pa.record_batch(reliability)
    assert_matches_golden(batch.schema, "calibration_reliability")


def test_calibration_per_class_schema_matches_golden() -> None:
    pa = pytest.importorskip("pyarrow")
    cells = _make_cells()
    _ece, _mce, _n, _eff, _reliability, per_class = cells.calibrate(  # type: ignore[attr-defined]
        0,
        15,
        "equal_width",
        0.0,
        "wilson",
        True,
        "macro",
    )
    assert per_class is not None
    batch = pa.record_batch(per_class)
    assert_matches_golden(batch.schema, "calibration_per_class")
