"""Rust ↔ numpy-oracle parity for TIDE error decomposition.

Per ADR-0021 the numpy oracle in `oracle.py` is the executable spec; the
Rust implementation in `vernier_core::tide::error_decomposition_*` is
correct iff `|delta_rust - delta_oracle| < 1e-9` per bin per fixture.

This file covers all three kernels:

- **bbox** (Week 2) — `error_decomposition_bbox` against the seven
  bbox fixtures.
- **segm** (Week 3) — `error_decomposition_segm` against the three
  segm fixtures.
- **boundary** (Week 3) — `error_decomposition_boundary` against the
  three boundary fixtures with `dilation_ratio = 0.02` (ADR-0010 COCO
  default) and `t_b = 0.05` (ADR-0022 boundary default, tentative —
  see the ADR's "Decision gate" section).

The `1e-9` tolerance is the ADR-0021 contract; loosening it silently
masks a Rust bug — STOP and report instead.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import pytest

from .oracle import boundary_iou, error_decomposition, segm_iou

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


FIXTURES = Path(__file__).parent / "fixtures"

# Per ADR-0021. The oracle is pure-Python f64; the Rust implementation
# uses f64 throughout. Anything wider than this is a real disagreement.
TOL = 1e-9

_ALL_FIXTURES = [
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

# Boundary fixtures — see `test_oracle.py` for the matching docstring
# assertions. Each fixture's image is 200x200 so the boundary band
# radius (d = round(0.02 * sqrt(80000)) = 6) is large enough that the
# band is a non-trivial 6-pixel frame.
_BOUNDARY_FIXTURES = [
    "boundary_all_perfect",
    "boundary_all_loc",
    "boundary_all_cls",
]
_BOUNDARY_DILATION_RATIO = 0.02
_BOUNDARY_T_B = 0.05  # ADR-0022 boundary default.
_BOUNDARY_IMAGE_HW = (200, 200)


def _load(name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fix_dir = FIXTURES / name
    gt = json.loads((fix_dir / "gt.json").read_text())
    dt = json.loads((fix_dir / "dt.json").read_text())
    return gt, dt


def _assert_close(actual: float, expected: float, label: str) -> None:
    assert abs(actual - expected) < TOL, (
        f"{label}: rust={actual!r}, oracle={expected!r}, "
        f"diff={actual - expected!r} exceeds tolerance {TOL!r}"
    )


@pytest.mark.parametrize("name", _ALL_FIXTURES)
def test_rust_matches_oracle(name: str) -> None:
    """Rust FFI bin-deltas agree with the numpy oracle within 1e-9 (ADR-0021)."""
    gt_dict, dt_list = _load(name)
    gt_bytes = json.dumps(gt_dict).encode()
    dt_bytes = json.dumps(dt_list).encode()

    oracle_out = error_decomposition(gt_dict, dt_list)
    rust_out = _core.error_decomposition_bbox(
        gt_bytes,
        dt_bytes,
        "strict",
        0.5,
        0.1,
        100,
        True,
    )

    _assert_close(rust_out["baseline_map"], oracle_out["baseline_map"], f"{name}: baseline_map")
    for bin_name in ("cls", "loc", "both", "dupe", "bkg", "missed"):
        _assert_close(
            rust_out["delta"][bin_name],
            oracle_out["delta"][bin_name],
            f"{name}: delta[{bin_name}]",
        )
    _assert_close(
        rust_out["delta_all_fp_removed"],
        oracle_out["delta_all_fp_removed"],
        f"{name}: delta_all_fp_removed",
    )


@pytest.mark.parametrize("name", _ALL_FIXTURES)
def test_rust_report_carries_resolved_config(name: str) -> None:
    """ADR-0022: the report records the (t_f, t_b, kernel) it was called with."""
    gt_dict, dt_list = _load(name)
    gt_bytes = json.dumps(gt_dict).encode()
    dt_bytes = json.dumps(dt_list).encode()

    rust_out = _core.error_decomposition_bbox(
        gt_bytes,
        dt_bytes,
        "strict",
        0.5,
        0.1,
        100,
        True,
    )

    config = rust_out["config"]
    assert config["kernel"] == "bbox", f"{name}: kernel"
    assert config["t_f"] == pytest.approx(0.5), f"{name}: t_f"
    assert config["t_b"] == pytest.approx(0.1), f"{name}: t_b"


@pytest.mark.parametrize("name", _SEGM_FIXTURES)
def test_rust_segm_matches_oracle(name: str) -> None:
    """Segm Rust FFI bin-deltas agree with the numpy oracle within 1e-9.

    Mirrors :func:`test_rust_matches_oracle` for the segm kernel: parses
    the segm fixture, runs `vernier._core.error_decomposition_segm`,
    runs the oracle's `error_decomposition` with `similarity_fn=segm_iou`,
    and asserts every delta agrees within `1e-9`. Same ADR-0021 tolerance
    contract as the bbox tests; loosening it masks a Rust bug.
    """
    gt_dict, dt_list = _load(name)
    gt_bytes = json.dumps(gt_dict).encode()
    dt_bytes = json.dumps(dt_list).encode()

    oracle_out = error_decomposition(gt_dict, dt_list, segm_iou)
    rust_out = _core.error_decomposition_segm(
        gt_bytes,
        dt_bytes,
        "strict",
        0.5,
        0.1,
        100,
        True,
    )

    _assert_close(rust_out["baseline_map"], oracle_out["baseline_map"], f"{name}: baseline_map")
    for bin_name in ("cls", "loc", "both", "dupe", "bkg", "missed"):
        _assert_close(
            rust_out["delta"][bin_name],
            oracle_out["delta"][bin_name],
            f"{name}: delta[{bin_name}]",
        )
    _assert_close(
        rust_out["delta_all_fp_removed"],
        oracle_out["delta_all_fp_removed"],
        f"{name}: delta_all_fp_removed",
    )


@pytest.mark.parametrize("name", _SEGM_FIXTURES)
def test_rust_segm_report_carries_resolved_config(name: str) -> None:
    """ADR-0022: the segm report records `(t_f, t_b, kernel="segm")`."""
    gt_dict, dt_list = _load(name)
    gt_bytes = json.dumps(gt_dict).encode()
    dt_bytes = json.dumps(dt_list).encode()

    rust_out = _core.error_decomposition_segm(
        gt_bytes,
        dt_bytes,
        "strict",
        0.5,
        0.1,
        100,
        True,
    )

    config = rust_out["config"]
    assert config["kernel"] == "segm", f"{name}: kernel"
    assert config["t_f"] == pytest.approx(0.5), f"{name}: t_f"
    assert config["t_b"] == pytest.approx(0.1), f"{name}: t_b"


@pytest.mark.skipif(
    not hasattr(_core, "error_decomposition_boundary"),
    reason="vernier._core.error_decomposition_boundary not yet built into the wheel "
    "(Week-3 Rust PR). Rebuild with `just develop` after that lands.",
)
@pytest.mark.parametrize("name", _BOUNDARY_FIXTURES)
def test_rust_matches_oracle_boundary(name: str) -> None:
    """Boundary FFI bin-deltas agree with the numpy oracle within 1e-9.

    Both implementations rasterize axis-aligned-rectangle polygons onto
    the fixture's declared image grid (200x200) and run the ADR-0010
    boundary kernel at `dilation_ratio = 0.02`. The Rust path goes
    through `vernier_mask`'s polygon rasterizer + van Herk separable
    erosion; the oracle uses numpy slicing + iterative 3x3 erosion.
    Both produce the same Chebyshev-ball erosion on integer-aligned
    rectangles, so the IoU values should match bit-for-bit (and
    therefore every per-bin ΔmAP within 1e-9).
    """
    gt_dict, dt_list = _load(name)
    gt_bytes = json.dumps(gt_dict).encode()
    dt_bytes = json.dumps(dt_list).encode()

    sim = functools.partial(
        boundary_iou,
        dilation_ratio=_BOUNDARY_DILATION_RATIO,
        image_hw=_BOUNDARY_IMAGE_HW,
    )
    oracle_out = error_decomposition(
        gt_dict,
        dt_list,
        sim,
        t_f=0.5,
        t_b=_BOUNDARY_T_B,
        kernel_name="boundary",
    )
    rust_out = _core.error_decomposition_boundary(
        gt_bytes,
        dt_bytes,
        "strict",
        0.5,
        _BOUNDARY_T_B,
        100,
        True,
        _BOUNDARY_DILATION_RATIO,
    )

    _assert_close(rust_out["baseline_map"], oracle_out["baseline_map"], f"{name}: baseline_map")
    for bin_name in ("cls", "loc", "both", "dupe", "bkg", "missed"):
        _assert_close(
            rust_out["delta"][bin_name],
            oracle_out["delta"][bin_name],
            f"{name}: delta[{bin_name}]",
        )
    _assert_close(
        rust_out["delta_all_fp_removed"],
        oracle_out["delta_all_fp_removed"],
        f"{name}: delta_all_fp_removed",
    )


@pytest.mark.skipif(
    not hasattr(_core, "error_decomposition_boundary"),
    reason="vernier._core.error_decomposition_boundary not yet built into the wheel.",
)
@pytest.mark.parametrize("name", _BOUNDARY_FIXTURES)
def test_rust_boundary_report_carries_resolved_config(name: str) -> None:
    """ADR-0022: the boundary report records ``kernel = "boundary"`` and the
    `(t_f, t_b)` it was called with.
    """
    gt_dict, dt_list = _load(name)
    gt_bytes = json.dumps(gt_dict).encode()
    dt_bytes = json.dumps(dt_list).encode()

    rust_out = _core.error_decomposition_boundary(
        gt_bytes,
        dt_bytes,
        "strict",
        0.5,
        _BOUNDARY_T_B,
        100,
        True,
        _BOUNDARY_DILATION_RATIO,
    )

    config = rust_out["config"]
    assert config["kernel"] == "boundary", f"{name}: kernel"
    assert config["t_f"] == pytest.approx(0.5), f"{name}: t_f"
    assert config["t_b"] == pytest.approx(_BOUNDARY_T_B), f"{name}: t_b"
