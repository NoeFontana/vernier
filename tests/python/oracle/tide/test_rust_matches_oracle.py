"""Rust ↔ numpy-oracle parity for TIDE error decomposition.

Per ADR-0021 the numpy oracle in `oracle.py` is the executable spec; the
Rust implementation in `vernier_core::tide::error_decomposition_bbox` is
correct iff `|delta_rust - delta_oracle| < 1e-9` per bin per fixture.

This test runs the Rust FFI (`vernier._core.error_decomposition_bbox`)
and the oracle on the same seven Week-1 fixtures and asserts the parity
contract on `baseline_map`, every per-bin delta, and the all-FPs-removed
sanity total. The `1e-9` tolerance is the ADR-0021 contract; loosening it
silently masks a Rust bug — STOP and report instead.

Cross-agent dependency: this file imports
`vernier._core.error_decomposition_bbox` which is added by a sibling PR
(Agent A's Week-2 Rust implementation). Until that PR merges, the symbol
is absent from the FFI module and these tests skip cleanly via the
attribute check below.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from .oracle import error_decomposition

_core = pytest.importorskip("vernier._core")
if not hasattr(_core, "error_decomposition_bbox"):
    pytest.skip(
        "vernier._core.error_decomposition_bbox not yet built into the wheel "
        "(Agent A's Week-2 Rust PR). Rebase + `just develop` after that lands.",
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
