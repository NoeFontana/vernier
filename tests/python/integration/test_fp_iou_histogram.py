"""End-to-end tests for ``vernier.fp_iou_histogram``.

The Rust ↔ surface contract is exercised in
``crates/vernier-core/tests/tide_fp_iou_histogram.rs`` (5 tests on
canonical fixtures). This file proves the Python wrapper carries the
FFI numbers through verbatim and dispatches to the right kernel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vernier
from vernier import Bbox, Boundary, Keypoints, Segm, fp_iou_histogram

_FIXTURES_ROOT = Path(__file__).resolve().parents[1] / "oracle" / "tide" / "fixtures"


def _load(name: str) -> tuple[bytes, bytes]:
    fix = _FIXTURES_ROOT / name
    return (fix / "gt.json").read_bytes(), (fix / "dt.json").read_bytes()


def test_all_perfect_yields_zero_fps() -> None:
    gt, dt = _load("all_perfect")
    h = fp_iou_histogram(gt, dt, iou=Bbox())
    assert h.n_fps == 0
    assert h.iou_same.shape == (0,)
    assert h.iou_cross.shape == (0,)
    assert h.kernel == "bbox"
    assert h.t_f == 0.5


def test_all_bkg_emits_low_iou_same() -> None:
    gt, dt = _load("all_bkg")
    h = fp_iou_histogram(gt, dt, iou=Bbox())
    assert h.n_fps > 0
    # Bkg fixture: every FP's same-class IoU is below the standard t_b.
    assert (h.iou_same < 0.1).all()


def test_kernel_dispatch_segm() -> None:
    gt, dt = _load("segm_all_perfect")
    h = fp_iou_histogram(gt, dt, iou=Segm())
    assert h.kernel == "segm"
    assert h.n_fps == 0


def test_kernel_dispatch_boundary() -> None:
    gt, dt = _load("boundary_all_perfect")
    h = fp_iou_histogram(gt, dt, iou=Boundary(dilation_ratio=0.02))
    assert h.kernel == "boundary"


def test_keypoints_rejected() -> None:
    gt, dt = _load("all_perfect")
    with pytest.raises(NotImplementedError, match="ADR-0024"):
        fp_iou_histogram(gt, dt, iou=Keypoints({1: tuple([0.05] * 17)}))


def test_explicit_t_f_overrides_default() -> None:
    gt, dt = _load("all_loc")
    # At t_f=0.99 every match becomes an FP, so n_fps should grow vs the
    # default. The exact number depends on the fixture; test the
    # direction.
    h_default = fp_iou_histogram(gt, dt, iou=Bbox())
    h_strict = fp_iou_histogram(gt, dt, iou=Bbox(), t_f=0.99)
    assert h_strict.n_fps >= h_default.n_fps
    assert h_strict.t_f == 0.99


def test_iou_arrays_are_numpy_float64() -> None:
    gt, dt = _load("all_loc")
    h = fp_iou_histogram(gt, dt, iou=Bbox())
    assert isinstance(h.iou_same, np.ndarray)
    assert isinstance(h.iou_cross, np.ndarray)
    assert h.iou_same.dtype == np.float64
    assert h.iou_cross.dtype == np.float64
    assert h.iou_same.shape == (h.n_fps,)
    assert h.iou_cross.shape == (h.n_fps,)


def test_dataset_handle_rejected() -> None:
    gt, dt = _load("all_perfect")
    ds = vernier.Dataset.from_json(gt)
    with pytest.raises(NotImplementedError, match="Dataset handle"):
        fp_iou_histogram(ds, dt, iou=Bbox())


def test_public_symbols_exported() -> None:
    """``vernier.fp_iou_histogram`` and ``vernier.FpIouHistogram`` are
    on the public surface alongside ``error_decomposition``."""
    assert vernier.fp_iou_histogram is fp_iou_histogram
    assert hasattr(vernier, "FpIouHistogram")
