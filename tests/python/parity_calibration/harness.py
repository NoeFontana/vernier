"""Parity harness for the ADR-0018 calibration kernel.

Mirrors the ADR-0010 isolation pattern used by
:mod:`tests.python.parity_boundary.harness`. The harness:

1. Loads a ``cells.json`` fixture produced by
   :mod:`tests.python.parity_calibration.fixtures.seed`.
2. Feeds the **same cells** to the numpy oracle
   (:func:`numpy_oracle.numpy_calibration`) and the vernier kernel
   (via :meth:`vernier._core.EvalCells.from_python_cells` followed by
   :meth:`vernier._core.EvalCells.calibrate`).
3. Compares the two outputs under either bit-equality (``"strict"``) or
   4-ULP tolerance (``"aligned"``) — the ADR-0018 parity model.

The vernier path goes through the *production* ``EvalCells.calibrate``
codepath; only the cell-store *constructor* is test-only. This ensures
parity is tested on the kernel users actually call, not a parallel
implementation.

The harness intentionally remaps the fixture's user-facing
``class_id`` field to dense 0-based ``k`` indices before feeding either
side. The Rust kernel iterates ``0..n_categories`` and emits ``k`` in
the per-class table; the oracle uses whatever keys appear in
``cells_by_class``. Aligning both on the same dense encoding makes the
per-class breakdown row-for-row comparable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

import vernier._core as _vernier_core

from .numpy_oracle import CalibrationParams, PerImageCell, numpy_calibration

Mode = Literal["strict", "aligned"]

#: Aligned-mode tolerance (4 ULP at f64 = 4 * f64.eps). Strict-mode
#: passes ``rtol=0, atol=0`` to ``assert_allclose``.
_ALIGNED_RTOL: float = 4.0 * float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class CalibrationSnapshot:
    """Side-by-side snapshot of a calibration summary.

    Columns of ``reliability`` / ``per_class`` mirror the ADR-0019
    Arrow schema (``calibration_reliability`` /
    ``calibration_per_class``). NaNs in the float columns are
    significant (zero-count bins emit NaN per the R2 convention) — the
    matcher treats two NaNs at the same index as equal.
    """

    ece: float
    mce: float
    n_detections: int
    effective_n_bins: int
    reliability: dict[str, np.ndarray]
    per_class: dict[str, np.ndarray] | None


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


def load_fixture_cells(
    fixture_dir: Path,
) -> tuple[dict[int, list[PerImageCell]], int]:
    """Load a fixture's ``cells.json`` and remap to dense ``k`` indices.

    Returns ``(cells_by_k, n_iou_thresholds)``. ``cells_by_k`` is keyed
    by dense 0-based ``k`` (the kernel's slot index), not the fixture's
    JSON ``class_id``. The mapping ``k -> class_id`` is monotonic in
    ``class_id`` (sorted ascending), so per-class table ordering is
    preserved.
    """
    path = fixture_dir / "cells.json"
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    n_t = int(payload["n_iou_thresholds"])
    raw_by_class: dict[int, list[PerImageCell]] = {}
    for raw in payload["cells"]:
        class_id = int(raw["class_id"])
        scores = np.asarray(raw["dt_scores"], dtype=np.float64)
        matched = np.asarray(raw["dt_matched"], dtype=bool)
        ignore = np.asarray(raw["dt_ignore"], dtype=bool)
        if matched.ndim == 1:
            matched = matched.reshape(1, -1)
        if ignore.ndim == 1:
            ignore = ignore.reshape(1, -1)
        cell = PerImageCell(dt_scores=scores, dt_matched=matched, dt_ignore=ignore)
        raw_by_class.setdefault(class_id, []).append(cell)
    cells_by_k: dict[int, list[PerImageCell]] = {}
    for k, class_id in enumerate(sorted(raw_by_class.keys())):
        cells_by_k[k] = raw_by_class[class_id]
    return cells_by_k, n_t


# ---------------------------------------------------------------------------
# Snapshot builders
# ---------------------------------------------------------------------------


def snapshot_oracle(
    cells_by_k: dict[int, list[PerImageCell]],
    params: CalibrationParams,
) -> CalibrationSnapshot:
    """Run the numpy oracle on the (dense-keyed) cells."""
    out = numpy_calibration(cells_by_k, params)
    reliability = _ensure_dict_of_arrays(out["reliability"])
    per_class_raw = out["per_class"]
    per_class = _ensure_dict_of_arrays(per_class_raw) if per_class_raw is not None else None
    return CalibrationSnapshot(
        ece=float(out["ece"]),  # type: ignore[arg-type]
        mce=float(out["mce"]),  # type: ignore[arg-type]
        n_detections=int(out["n_detections"]),  # type: ignore[arg-type]
        effective_n_bins=int(out["effective_n_bins"]),  # type: ignore[arg-type]
        reliability=reliability,
        per_class=per_class,
    )


def snapshot_vernier(
    cells_by_k: dict[int, list[PerImageCell]],
    params: CalibrationParams,
    n_iou_thresholds: int,
) -> CalibrationSnapshot:
    """Build an :class:`EvalCells` via the FFI test-only constructor and
    fold through :meth:`EvalCells.calibrate`."""
    cells_dict = _build_cells_dict(cells_by_k, n_iou_thresholds)
    handle = _vernier_core.EvalCells.from_python_cells(cells_dict)
    ece, mce, n_det, eff_bins, reliability_batch, per_class_batch = handle.calibrate(
        params.iou_index,
        params.n_bins,
        params.binning,
        params.min_score,
        params.confidence,
        params.per_class,
        params.per_class_aggregation,
    )
    reliability = _record_batch_to_arrays(reliability_batch)
    per_class = _record_batch_to_arrays(per_class_batch) if per_class_batch else None
    return CalibrationSnapshot(
        ece=float(ece),
        mce=float(mce),
        n_detections=int(n_det),
        effective_n_bins=int(eff_bins),
        reliability=reliability,
        per_class=per_class,
    )


def _build_cells_dict(
    cells_by_k: dict[int, list[PerImageCell]],
    n_iou_thresholds: int,
) -> dict[str, Any]:
    """Flatten the (k -> list[cell]) mapping into the dense
    ``(k * A * I + a * I + i)`` cells list the FFI consumes. ``A = 1``
    (calibration only consults the ``all`` area bucket); ``I = max(len
    over k)`` so every class has the same I-axis length, with
    short-list classes padded with ``None``.
    """
    n_categories = len(cells_by_k)
    n_area_ranges = 1
    n_images = max((len(v) for v in cells_by_k.values()), default=0)
    # Pin the IoU thresholds to the canonical 10-point COCO ladder.
    # The oracle reads only ``iou_index`` (default 0), and the kernel's
    # ``iou_to_index`` isn't exercised by the harness — we pass the
    # integer index directly. The threshold values must still have
    # ``len == T`` so the kernel's per-cell shape check passes.
    iou_thresholds = np.linspace(0.5, 0.95, n_iou_thresholds, dtype=np.float64).tolist()
    cells: list[dict[str, Any] | None] = []
    for k in range(n_categories):
        per_image = cells_by_k.get(k, [])
        for i in range(n_images):
            if i < len(per_image):
                cell = per_image[i]
                cells.append(
                    {
                        "dt_scores": cell.dt_scores.tolist(),
                        "dt_matched": cell.dt_matched.tolist(),
                        "dt_ignore": cell.dt_ignore.tolist(),
                    }
                )
            else:
                cells.append(None)
    return {
        "n_categories": n_categories,
        "n_area_ranges": n_area_ranges,
        "n_images": n_images,
        "iou_thresholds": iou_thresholds,
        "parity_mode": "strict",
        "cells": cells,
    }


def _record_batch_to_arrays(batch: Any) -> dict[str, np.ndarray]:
    """Pull a :class:`pyarrow.RecordBatch` into a column-wise dict of
    numpy arrays. We import :mod:`pyarrow` lazily so the harness module
    doesn't pay the import cost when only the oracle is exercised."""
    import pyarrow as pa

    rb = pa.record_batch(batch)
    out: dict[str, np.ndarray] = {}
    for name in rb.schema.names:
        out[name] = rb.column(name).to_numpy(zero_copy_only=False)
    return out


def _ensure_dict_of_arrays(d: object) -> dict[str, np.ndarray]:
    """Defensive cast to make pyright happy. ``numpy_calibration``
    returns ``dict[str, object]`` because the per-class slot is
    ``dict | None``; the columns themselves are always
    ``np.ndarray``."""
    assert isinstance(d, dict)
    out: dict[str, np.ndarray] = {}
    for key, val in d.items():
        assert isinstance(val, np.ndarray), f"column {key!r} not an ndarray"
        out[str(key)] = val
    return out


# ---------------------------------------------------------------------------
# Assertion
# ---------------------------------------------------------------------------


def assert_snapshots_match(
    oracle: CalibrationSnapshot,
    vernier: CalibrationSnapshot,
    mode: Mode,
) -> None:
    """Diff two snapshots elementwise.

    ``"strict"`` requires bit-equality (NaN positions must match).
    ``"aligned"`` allows up to 4 ULP (``4 * f64.eps``) relative error
    on float columns; integer columns and NaN positions must still
    match bit-exact.
    """
    rtol = 0.0 if mode == "strict" else _ALIGNED_RTOL

    # Scalars first (cheap, points at the most likely failure mode).
    assert oracle.n_detections == vernier.n_detections, (
        f"n_detections: oracle={oracle.n_detections} vs vernier={vernier.n_detections}"
    )
    assert oracle.effective_n_bins == vernier.effective_n_bins, (
        f"effective_n_bins: oracle={oracle.effective_n_bins} vs vernier={vernier.effective_n_bins}"
    )
    _assert_scalar_close(oracle.ece, vernier.ece, rtol=rtol, name="ece")
    _assert_scalar_close(oracle.mce, vernier.mce, rtol=rtol, name="mce")

    # Reliability table.
    _assert_table_match(oracle.reliability, vernier.reliability, rtol=rtol, label="reliability")

    # Per-class table (when requested).
    if oracle.per_class is None and vernier.per_class is None:
        return
    assert oracle.per_class is not None, "vernier emitted per_class but oracle did not"
    assert vernier.per_class is not None, "oracle emitted per_class but vernier did not"
    _assert_table_match(oracle.per_class, vernier.per_class, rtol=rtol, label="per_class")


def _assert_scalar_close(a: float, b: float, *, rtol: float, name: str) -> None:
    if np.isnan(a) and np.isnan(b):
        return
    if rtol == 0.0:
        # Strict: bit-equality (handles -0.0 vs 0.0 correctly and NaN
        # symmetry via the early return above).
        if not (a == b):
            raise AssertionError(f"{name}: strict mismatch {a!r} vs {b!r}")
        return
    np.testing.assert_allclose(a, b, rtol=rtol, atol=0.0, err_msg=name)


def _assert_table_match(
    a: dict[str, np.ndarray],
    b: dict[str, np.ndarray],
    *,
    rtol: float,
    label: str,
) -> None:
    assert set(a.keys()) == set(b.keys()), (
        f"{label}: column sets differ a={sorted(a)} b={sorted(b)}"
    )
    for col in a:
        ac = a[col]
        bc = b[col]
        assert ac.shape == bc.shape, f"{label}.{col}: shape {ac.shape} vs {bc.shape}"
        if np.issubdtype(ac.dtype, np.floating) or np.issubdtype(bc.dtype, np.floating):
            # NaN positions must match before the value diff.
            nan_a = np.isnan(ac)
            nan_b = np.isnan(bc)
            assert np.array_equal(nan_a, nan_b), (
                f"{label}.{col}: NaN positions differ\noracle={nan_a}\nvernier={nan_b}"
            )
            if rtol == 0.0:
                # Strict-mode bit-equality on the non-NaN entries.
                if not np.array_equal(ac[~nan_a], bc[~nan_b]):
                    raise AssertionError(
                        f"{label}.{col}: strict mismatch\noracle ={ac}\nvernier={bc}"
                    )
            else:
                np.testing.assert_allclose(
                    ac[~nan_a],
                    bc[~nan_b],
                    rtol=rtol,
                    atol=0.0,
                    err_msg=f"{label}.{col}",
                )
        else:
            # Integer columns must match bit-exact regardless of mode.
            if not np.array_equal(ac, bc):
                raise AssertionError(f"{label}.{col}: integer mismatch\noracle ={ac}\nvernier={bc}")
