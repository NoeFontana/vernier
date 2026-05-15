"""Parity tests for the ADR-0018 calibration kernel (Unit 4c).

Wires the numpy oracle (:mod:`tests.python.parity_calibration.numpy_oracle`)
against vernier's live calibration kernel
(:mod:`vernier._core.EvalCells.calibrate`) and asserts bit-equality
(strict mode) or 4-ULP equivalence (aligned mode) across the five
fixtures produced by :mod:`tests.python.parity_calibration.fixtures.seed`.

The harness feeds **the same per-image cells** to both implementations
via :meth:`vernier._core.EvalCells.from_python_cells` — a test-only
constructor that lets the parity test exercise the production
``EvalCells.calibrate`` codepath without driving the full
GT/DT → Evaluator → grid → cells pipeline.

A separate end-to-end smoketest
(:func:`test_calibration_end_to_end_smoke`) does drive the full
pipeline against a tiny COCO-shaped fixture to prove the
``Evaluator.evaluate(..., calibration=True)`` surface stays well-formed;
that test asserts structural sanity, not bit-equality to the oracle
(the oracle's contract is over cells, not over GT/DT).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import vernier._core as _vernier_core
from vernier.calibration import StreamingSnapshot
from vernier.instance import BackgroundEvaluator, Bbox, Evaluator

from .harness import (
    assert_snapshots_match,
    load_fixture_cells,
    snapshot_oracle,
    snapshot_vernier,
)
from .numpy_oracle import CalibrationParams

_FIXTURE_ROOT: Path = Path(__file__).parent / "fixtures"
_ALL_FIXTURES: tuple[str, ...] = (
    "cal_perfect",
    "cal_overconfident",
    "cal_ignore_regions",
    "cal_segm_smoketest",
    "cal_keypoints_smoketest",
)

#: Default params for the parity tests. Mirrors ADR-0018's
#: DETR-aware defaults — the same params the user-facing
#: ``result.calibration()`` exposes.
_DEFAULT_PARAMS = CalibrationParams(
    iou_index=0,
    n_bins=15,
    binning="quantile",
    min_score=0.05,
    confidence="wilson",
    per_class=False,
    per_class_aggregation="macro",
)


# ---------------------------------------------------------------------------
# Strict + aligned parametrized tests over the five fixtures.
# ---------------------------------------------------------------------------


@pytest.mark.parity_calibration
@pytest.mark.parametrize("fixture", _ALL_FIXTURES)
def test_calibration_parity_strict(fixture: str) -> None:
    """Bit-equal oracle ↔ vernier across the canonical fixtures."""
    cells_by_k, n_t = load_fixture_cells(_FIXTURE_ROOT / fixture)
    oracle = snapshot_oracle(cells_by_k, _DEFAULT_PARAMS)
    vernier = snapshot_vernier(cells_by_k, _DEFAULT_PARAMS, n_t)
    assert_snapshots_match(oracle, vernier, mode="strict")


@pytest.mark.parity_calibration
@pytest.mark.parametrize("fixture", _ALL_FIXTURES)
def test_calibration_parity_aligned(fixture: str) -> None:
    """Same fixtures under the 4-ULP aligned-mode tolerance.

    Strict subsumes aligned for these synthetic fixtures, but the
    aligned path is exercised separately to prove the 4-ULP ceiling
    documented in ADR-0018 holds — and to keep a tolerance regression
    visible if a future kernel change replaces a bit-stable reduction
    with a parallel one.
    """
    cells_by_k, n_t = load_fixture_cells(_FIXTURE_ROOT / fixture)
    oracle = snapshot_oracle(cells_by_k, _DEFAULT_PARAMS)
    vernier = snapshot_vernier(cells_by_k, _DEFAULT_PARAMS, n_t)
    assert_snapshots_match(oracle, vernier, mode="aligned")


# ---------------------------------------------------------------------------
# iou_type-genericity at the cell-store level.
# ---------------------------------------------------------------------------


_IOU_TYPE_FIXTURE: dict[str, str] = {
    "bbox": "cal_perfect",
    "segm": "cal_segm_smoketest",
    "keypoints": "cal_keypoints_smoketest",
}


@pytest.mark.parity_calibration
@pytest.mark.parametrize("iou_type", ["bbox", "segm", "keypoints"])
def test_calibration_iou_type_genericity(iou_type: str) -> None:
    """The kernel folds over cells without consulting any iou_type tag.

    Loads the per-paradigm fixture and confirms vernier and the oracle
    produce identical summaries — i.e. the same cell content yields
    the same calibration summary regardless of which iou_type tag is
    attached to the fixture. This is the ADR-0018 "tested
    iou_type-genericity" assertion at the *cell-store* level (the
    full-pipeline assertion lives in
    :func:`test_calibration_end_to_end_smoke`).
    """
    fixture = _IOU_TYPE_FIXTURE[iou_type]
    cells_by_k, n_t = load_fixture_cells(_FIXTURE_ROOT / fixture)
    oracle = snapshot_oracle(cells_by_k, _DEFAULT_PARAMS)
    vernier = snapshot_vernier(cells_by_k, _DEFAULT_PARAMS, n_t)
    assert_snapshots_match(oracle, vernier, mode="strict")


# ---------------------------------------------------------------------------
# Per-class breakdown.
# ---------------------------------------------------------------------------


@pytest.mark.parity_calibration
def test_calibration_per_class_breakdown() -> None:
    """``per_class=True`` produces a K-row table whose ``n`` column sums
    to the marginal ``n_detections``.

    Exercises the per-class kernel path on ``cal_overconfident`` (two
    classes, asymmetric matched/ignore patterns). Parity is bit-equal
    between oracle and vernier; the assertion also pins the per-class
    counts invariant from ADR-0018 §"Decision outcome" → "Per-class
    layout".
    """
    fixture = "cal_overconfident"
    cells_by_k, n_t = load_fixture_cells(_FIXTURE_ROOT / fixture)
    params = CalibrationParams(
        iou_index=0,
        n_bins=15,
        binning="quantile",
        min_score=0.05,
        confidence="wilson",
        per_class=True,
        per_class_aggregation="macro",
    )
    oracle = snapshot_oracle(cells_by_k, params)
    vernier = snapshot_vernier(cells_by_k, params, n_t)
    assert_snapshots_match(oracle, vernier, mode="strict")

    assert oracle.per_class is not None
    n_classes = len(cells_by_k)
    assert oracle.per_class["class_id"].shape[0] == n_classes
    assert int(oracle.per_class["n"].sum()) == oracle.n_detections
    expected_columns = {"class_id", "ece", "mce", "n"}
    assert set(oracle.per_class.keys()) == expected_columns


# ---------------------------------------------------------------------------
# Phase-2 surface: clopper_pearson raises on both sides.
# ---------------------------------------------------------------------------


@pytest.mark.parity_calibration
def test_calibration_clopper_pearson_phase2() -> None:
    """``confidence="clopper_pearson"`` is Phase-2; both implementations
    raise. Keeps the documented error path covered."""
    fixture = "cal_perfect"
    cells_by_k, n_t = load_fixture_cells(_FIXTURE_ROOT / fixture)
    params = CalibrationParams(
        iou_index=0,
        n_bins=15,
        binning="quantile",
        min_score=0.05,
        confidence="clopper_pearson",
        per_class=False,
        per_class_aggregation="macro",
    )
    with pytest.raises(NotImplementedError):
        snapshot_oracle(cells_by_k, params)

    # The vernier-side error is ``ValueError`` (the FFI re-wraps the
    # kernel's ``EvalError::InvalidConfig`` into ``PyValueError``); it
    # is *not* identical to the oracle's ``NotImplementedError``. This
    # asymmetry is documented in ``docs/engineering/calibration-quirks.md``;
    # the parity contract is over the **happy path** numerical output,
    # not the error type. Both surfaces refuse — that's the assertion.
    with pytest.raises((NotImplementedError, ValueError)):
        snapshot_vernier(cells_by_k, params, n_t)


# ---------------------------------------------------------------------------
# End-to-end smoketest — full pipeline from GT/DT to calibration().
# ---------------------------------------------------------------------------


@pytest.mark.parity_calibration
def test_calibration_end_to_end_smoke() -> None:
    """Drive the full ``Evaluator(...).evaluate(..., calibration=True)``
    pipeline on a real COCO-shaped fixture and assert structural sanity.

    This is the end-to-end iou_type-genericity proof — vernier's
    user-facing surface produces a well-formed
    :class:`vernier.calibration.CalibrationResult` against a tiny
    bbox-only GT/DT pair. Bit-equality to the oracle is *not* asserted
    here — the unit-level oracle parity tests above already cover that
    contract over cells; what this test pins is that the pipeline
    plumbs through correctly (cells retained on the result, scalars
    finite, reliability table shaped per the Arrow schema).
    """
    # ``partition_tiny`` carries six detections at distinct scores —
    # enough for the quantile dedup to leave a non-degenerate ladder
    # (``perfect_match`` has one detection, which collapses every
    # quantile edge into a single value and yields ``effective_n_bins
    # = 0``; that exercises a different code path the unit-level
    # tests already cover).
    gt_path = Path(__file__).parent.parent / "parity" / "fixtures" / "partition_tiny" / "gt.json"
    dt_path = Path(__file__).parent.parent / "parity" / "fixtures" / "partition_tiny" / "dt.json"
    evaluator = Evaluator(iou=Bbox(), parity_mode="strict")
    result = evaluator.evaluate(
        gt_path.read_bytes(),
        dt_path.read_bytes(),
        calibration=True,
    )
    # ``_eval_cells`` is the opaque PyO3 handle.
    assert isinstance(result._eval_cells, _vernier_core.EvalCells)
    cal = result.calibration(iou=0.5, n_bins=15)
    assert isinstance(cal.ece, float)
    assert isinstance(cal.mce, float)
    assert cal.n_detections > 0
    # Reliability table: 9 columns, ``effective_n_bins`` rows.
    reliability = cal.reliability
    assert reliability.shape == (cal.effective_n_bins, 9)
    expected_cols = {
        "bin_id",
        "score_lo",
        "score_hi",
        "mean_score",
        "accuracy",
        "count",
        "gap",
        "ci_lo",
        "ci_hi",
    }
    assert set(reliability.columns) == expected_cols


# ---------------------------------------------------------------------------
# ADR-0018 Unit 6 — streaming integration end-to-end.
# ---------------------------------------------------------------------------


@pytest.mark.parity_calibration
def test_streaming_calibration_end_to_end_smoke() -> None:
    """Drive ``BackgroundEvaluator.finalize_with_cells`` through the
    ``StreamingSnapshot`` Python wrapper and assert the streaming
    calibration surface matches the batch surface bit-for-bit on the
    same GT/DT pair (ADR-0018 Unit 6).

    Both paths consume the same per-image cells; what differs is
    plumbing — batch fills the cell store inline via ``evaluate``,
    streaming via a single ``submit`` round-trip through the worker
    thread. The summary axis must be bit-equal and the calibration
    fold (post-cells) must produce the same ``ece`` / ``mce`` /
    ``n_detections`` because the kernel is deterministic over
    identical inputs.
    """
    gt_path = Path(__file__).parent.parent / "parity" / "fixtures" / "partition_tiny" / "gt.json"
    dt_path = Path(__file__).parent.parent / "parity" / "fixtures" / "partition_tiny" / "dt.json"
    gt_bytes = gt_path.read_bytes()
    dt_bytes = dt_path.read_bytes()

    # Batch reference.
    evaluator = Evaluator(iou=Bbox(), parity_mode="strict")
    batch_result = evaluator.evaluate(gt_bytes, dt_bytes, calibration=True)
    batch_cal = batch_result.calibration(iou=0.5, n_bins=15)

    # Streaming candidate.
    bg = BackgroundEvaluator(gt_bytes, parity_mode="strict")
    bg.submit(dt_bytes)
    snap = StreamingSnapshot.from_background(bg)

    assert isinstance(snap._eval_cells, _vernier_core.EvalCells)
    # Summary axis: bit-equal stat-by-stat. Streaming `finalize` on the
    # same GT/DT input must reproduce the batch summary exactly under
    # `strict` mode.
    assert batch_result.summary is not None  # default-grid path always populates this
    assert list(snap.summary.stats) == list(batch_result.summary.stats)

    snap_cal = snap.calibration(iou=0.5, n_bins=15)
    assert snap_cal.ece == batch_cal.ece
    assert snap_cal.mce == batch_cal.mce
    assert snap_cal.n_detections == batch_cal.n_detections
    assert snap_cal.effective_n_bins == batch_cal.effective_n_bins
    # Reliability table shape mirrors the batch surface.
    assert snap_cal.reliability.shape == (snap_cal.effective_n_bins, 9)
