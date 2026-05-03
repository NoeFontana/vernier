"""End-to-end tests for the public ``vernier.error_decomposition`` surface.

The Rust ↔ numpy-oracle parity contract is exercised in
:mod:`tests.python.oracle.tide.test_rust_matches_oracle` (per ADR-0021).
This file's job is narrower: prove the Python wrapper in
:mod:`vernier._tide` carries the FFI's numbers through verbatim, dispatches
to the right kernel, resolves per-kernel ADR-0022 default thresholds, and
rejects the deferred surfaces (Keypoints per ADR-0024 and ``Dataset``
handles per the 0.5.x follow-up note in the docstring).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import vernier
from vernier import Bbox, Boundary, Dataset, Keypoints, Segm, TideReport, error_decomposition

_core = pytest.importorskip("vernier._core")
for _required in (
    "error_decomposition_bbox",
    "error_decomposition_segm",
    "error_decomposition_boundary",
):
    if not hasattr(_core, _required):
        pytest.skip(
            f"vernier._core.{_required} not yet built into the wheel. "
            "Run `just develop` after pulling main.",
            allow_module_level=True,
        )


# Mirror the parity test's tolerance — same tightness, same reasoning
# (ADR-0021): the wrapper does not perform any numerical work, so any
# drift relative to the FFI dict is a bug, not a tolerance budget.
TOL = 1e-12


# Reuse the TIDE oracle fixtures already in the repo (ADR-0021).
_FIXTURES_ROOT = Path(__file__).resolve().parents[1] / "oracle" / "tide" / "fixtures"


_BBOX_FIXTURES = [
    "all_perfect",
    "all_bkg",
    "all_cls",
    "all_loc",
    "all_dupe",
    "with_ignore",
    "loc_vs_both_priority",
]

_SEGM_FIXTURES = [
    "segm_all_perfect",
    "segm_all_loc",
    "segm_all_cls",
]

_BOUNDARY_FIXTURES = [
    "boundary_all_perfect",
    "boundary_all_loc",
    "boundary_all_cls",
]

_BOUNDARY_DILATION_RATIO = 0.02
# ADR-0022 boundary default; mirrored here so the test catches a
# silent change to the default in `_tide.py`.
_BOUNDARY_T_B = 0.05


def _load(name: str) -> tuple[bytes, bytes]:
    fix_dir = _FIXTURES_ROOT / name
    gt_bytes = (fix_dir / "gt.json").read_bytes()
    dt_bytes = (fix_dir / "dt.json").read_bytes()
    return gt_bytes, dt_bytes


def _ffi_for(kernel: str, gt: bytes, dt: bytes) -> dict[str, Any]:
    """Call the underlying FFI directly so we can assert the wrapper
    is bit-equivalent without re-running the eight-pass machinery
    against an oracle (that's :mod:`test_rust_matches_oracle`'s job)."""
    if kernel == "bbox":
        return _core.error_decomposition_bbox(gt, dt, "corrected", 0.5, 0.1, 100, True)
    if kernel == "segm":
        return _core.error_decomposition_segm(gt, dt, "corrected", 0.5, 0.1, 100, True)
    if kernel == "boundary":
        return _core.error_decomposition_boundary(
            gt, dt, "corrected", 0.5, _BOUNDARY_T_B, 100, True, _BOUNDARY_DILATION_RATIO
        )
    raise AssertionError(f"unknown kernel {kernel!r}")


_PARAMS = (
    [pytest.param(name, "bbox", id=f"bbox-{name}") for name in _BBOX_FIXTURES]
    + [pytest.param(name, "segm", id=f"segm-{name}") for name in _SEGM_FIXTURES]
    + [pytest.param(name, "boundary", id=f"boundary-{name}") for name in _BOUNDARY_FIXTURES]
)


@pytest.mark.parametrize(("fixture", "kernel"), _PARAMS)
def test_wrapper_round_trips_ffi_output(fixture: str, kernel: str) -> None:
    """``error_decomposition`` returns the FFI's numbers verbatim.

    Asserts every field of :class:`TideReport` (baseline_map, all six
    delta bins, delta_all_fp_removed, and the resolved config) matches
    the underlying ``_core.error_decomposition_<kernel>`` dict. Proves
    the wrapper is data-conversion-only — any drift means we're losing
    or transforming numbers we shouldn't be.
    """
    gt_bytes, dt_bytes = _load(fixture)
    if kernel == "bbox":
        report = error_decomposition(gt_bytes, dt_bytes, iou=Bbox())
    elif kernel == "segm":
        report = error_decomposition(gt_bytes, dt_bytes, iou=Segm())
    elif kernel == "boundary":
        report = error_decomposition(
            gt_bytes,
            dt_bytes,
            iou=Boundary(dilation_ratio=_BOUNDARY_DILATION_RATIO),
        )
    else:
        raise AssertionError(f"unhandled kernel {kernel!r}")

    raw = _ffi_for(kernel, gt_bytes, dt_bytes)

    assert isinstance(report, TideReport)
    assert abs(report.baseline_map - raw["baseline_map"]) < TOL
    assert abs(report.delta_all_fp_removed - raw["delta_all_fp_removed"]) < TOL
    for bin_name in ("cls", "loc", "both", "dupe", "bkg", "missed"):
        assert bin_name in report.delta, f"{bin_name} missing from wrapper delta"
        assert abs(report.delta[bin_name] - raw["delta"][bin_name]) < TOL, (
            f"{kernel}/{fixture}: delta[{bin_name}] drift "
            f"wrapper={report.delta[bin_name]!r} ffi={raw['delta'][bin_name]!r}"
        )

    assert report.config.kernel == kernel
    assert report.config.t_f == pytest.approx(raw["config"]["t_f"])
    assert report.config.t_b == pytest.approx(raw["config"]["t_b"])


def test_default_thresholds_resolve_per_kernel_per_adr_0022() -> None:
    """``t_f=None, t_b=None`` resolves to the ADR-0022 per-kernel defaults.

    The defaults table lives in :data:`vernier._tide._DEFAULT_THRESHOLDS`;
    this test pins it against the values ADR-0022 commits to. A silent
    change to either side fails here. ``t_f`` is ``0.5`` everywhere;
    ``t_b`` is per-kernel.
    """
    gt_bbox, dt_bbox = _load("all_perfect")
    bbox_report = error_decomposition(gt_bbox, dt_bbox, iou=Bbox())
    assert bbox_report.config.t_f == pytest.approx(0.5)
    assert bbox_report.config.t_b == pytest.approx(0.1)
    assert bbox_report.config.kernel == "bbox"

    gt_segm, dt_segm = _load("segm_all_perfect")
    segm_report = error_decomposition(gt_segm, dt_segm, iou=Segm())
    assert segm_report.config.t_f == pytest.approx(0.5)
    assert segm_report.config.t_b == pytest.approx(0.1)
    assert segm_report.config.kernel == "segm"

    gt_b, dt_b = _load("boundary_all_perfect")
    b_report = error_decomposition(
        gt_b, dt_b, iou=Boundary(dilation_ratio=_BOUNDARY_DILATION_RATIO)
    )
    assert b_report.config.t_f == pytest.approx(0.5)
    assert b_report.config.t_b == pytest.approx(_BOUNDARY_T_B)
    assert b_report.config.kernel == "boundary"


def test_default_iou_is_bbox() -> None:
    """``iou`` defaults to ``Bbox()`` when omitted."""
    gt_bytes, dt_bytes = _load("all_perfect")
    report = error_decomposition(gt_bytes, dt_bytes)
    assert report.config.kernel == "bbox"


def test_explicit_thresholds_override_defaults() -> None:
    """Explicit ``t_f`` / ``t_b`` win over the ADR-0022 defaults."""
    gt_bytes, dt_bytes = _load("all_perfect")
    report = error_decomposition(gt_bytes, dt_bytes, iou=Bbox(), t_f=0.75, t_b=0.2)
    assert report.config.t_f == pytest.approx(0.75)
    assert report.config.t_b == pytest.approx(0.2)


def test_keypoints_iou_raises_not_implemented_per_adr_0024() -> None:
    """``Keypoints(...)`` is rejected with a pointer to ADR-0024.

    TIDE on OKS is deferred (single-class workload makes Cls/Both
    structurally empty; OKS is not IoU; no published convention).
    """
    gt_bytes, dt_bytes = _load("all_perfect")
    with pytest.raises(NotImplementedError, match="ADR-0024"):
        error_decomposition(gt_bytes, dt_bytes, iou=Keypoints())


def test_dataset_handle_raises_not_implemented_forward_compat() -> None:
    """Passing a :class:`Dataset` handle raises with a clear follow-up note.

    The type signature accepts ``Dataset`` for forward-compat (mirrors
    :meth:`vernier.Evaluator.evaluate`'s overload), but the TIDE FFI is
    not yet wired through the parsed-once cache (ADR-0020). 0.5.x
    follow-up.
    """
    gt_bytes, dt_bytes = _load("all_perfect")
    handle = Dataset.from_json(gt_bytes)
    with pytest.raises(NotImplementedError, match="Dataset"):
        error_decomposition(handle, dt_bytes, iou=Bbox())


def test_unknown_iou_kind_raises_type_error() -> None:
    """A garbage ``iou`` argument raises :class:`TypeError`, not a panic."""
    gt_bytes, dt_bytes = _load("all_perfect")
    with pytest.raises(TypeError, match="unsupported iou kernel"):
        error_decomposition(gt_bytes, dt_bytes, iou="bbox")


def test_public_symbols_exported() -> None:
    """``vernier.error_decomposition`` / ``TideReport`` / ``TideConfig``
    are reachable from the top-level package and listed in ``__all__``.
    """
    assert "error_decomposition" in vernier.__all__
    assert "TideReport" in vernier.__all__
    assert "TideConfig" in vernier.__all__
    assert callable(vernier.error_decomposition)


def test_report_dict_round_trip() -> None:
    """``TideReport._from_dict`` round-trips a synthetic FFI-shaped dict.

    The real FFI shape is exercised by every other test; this one pins
    the parsing logic so a future FFI extension (e.g., new bin) lands
    a clean error here rather than at a load-bearing call site.
    """
    sample: dict[str, Any] = {
        "baseline_map": 0.42,
        "delta": {
            "cls": 0.01,
            "loc": 0.02,
            "both": 0.005,
            "dupe": 0.003,
            "bkg": 0.04,
            "missed": 0.07,
        },
        "delta_all_fp_removed": 0.15,
        "config": {"t_f": 0.5, "t_b": 0.1, "kernel": "bbox"},
    }
    report = TideReport._from_dict(sample)
    assert report.baseline_map == pytest.approx(0.42)
    assert report.delta["loc"] == pytest.approx(0.02)
    assert report.delta_all_fp_removed == pytest.approx(0.15)
    assert report.config == vernier.TideConfig(t_f=0.5, t_b=0.1, kernel="bbox")


def test_report_from_dict_rejects_unexpected_kernel() -> None:
    """An unexpected ``config.kernel`` value surfaces a structured error.

    The Rust :class:`KernelMarker::as_str` enumeration guarantees one of
    three literals; if the FFI starts emitting something else, that's a
    serious breakage we want to fail loud rather than silently coerce.
    """
    sample: dict[str, Any] = {
        "baseline_map": 0.0,
        "delta": {
            "cls": 0.0,
            "loc": 0.0,
            "both": 0.0,
            "dupe": 0.0,
            "bkg": 0.0,
            "missed": 0.0,
        },
        "delta_all_fp_removed": 0.0,
        "config": {"t_f": 0.5, "t_b": 0.1, "kernel": "keypoints"},
    }
    with pytest.raises(RuntimeError, match="unexpected kernel name"):
        TideReport._from_dict(sample)


def _smoke_load_json(name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Sanity-check helper used by the JSON-shape test below."""
    fix = _FIXTURES_ROOT / name
    gt = json.loads((fix / "gt.json").read_text())
    dt = json.loads((fix / "dt.json").read_text())
    return gt, dt


def test_fixtures_are_well_formed_json() -> None:
    """Guard against a fixture file getting corrupted on a rebase."""
    for name in _BBOX_FIXTURES + _SEGM_FIXTURES + _BOUNDARY_FIXTURES:
        gt, dt = _smoke_load_json(name)
        assert "annotations" in gt
        assert "images" in gt
        assert "categories" in gt
        assert isinstance(dt, list)
