"""ADR-0017 test plan §3 — inject a known divergence and assert the
report names the right index, values, and tolerance."""

from __future__ import annotations

import json

import numpy as np

from bench.harness.orchestrate import CellRun
from bench.harness.parity import (
    BOUNDARY_PARITY_EPS,
    PARITY_EPS,
    compare_cell,
    write_report,
)


def test_strict_tier_catches_one_ulp_delta_at_known_index(zero_tensor: np.ndarray) -> None:
    target = (3, 50, 0, 1, 2)
    t2 = zero_tensor.copy()
    t2[target] = PARITY_EPS  # one ULP above zero — strict rejects.

    report = compare_cell(
        workload_id="smoke",
        iou_type="bbox",
        impl_tensors={"vernier": zero_tensor, "pycocotools": t2},
        impl_sha256={"vernier": "a" * 64, "pycocotools": "b" * 64},
    )
    assert not report.passed
    strict = next(t for t in report.tiers if t.tier == "strict")
    assert strict.divergent_count == 1
    assert strict.first_divergence is not None
    assert strict.first_divergence.index == target
    assert strict.first_divergence.value_a == 0.0
    assert strict.first_divergence.value_b == PARITY_EPS
    assert strict.first_divergence.abs_diff == PARITY_EPS
    assert strict.tensor_sha256_a == "a" * 12
    assert strict.tensor_sha256_b == "b" * 12


def test_aligned_tier_catches_above_tolerance_delta(zero_tensor: np.ndarray) -> None:
    target = (0, 0, 0, 0, 0)
    t2 = zero_tensor.copy()
    t2[target] = 1e-12  # well above 4*PARITY_EPS

    report = compare_cell(
        workload_id="smoke",
        iou_type="bbox",
        impl_tensors={"vernier": zero_tensor, "faster-coco-eval": t2},
        impl_sha256={"vernier": "a" * 64, "faster-coco-eval": "b" * 64},
    )
    aligned = next(t for t in report.tiers if t.tier == "aligned")
    assert not aligned.passed
    assert aligned.first_divergence is not None
    assert aligned.first_divergence.index == target


def test_boundary_tier_catches_above_tolerance_delta(zero_tensor: np.ndarray) -> None:
    target = (5, 0, 0, 2, 1)
    t2 = zero_tensor.copy()
    t2[target] = BOUNDARY_PARITY_EPS * 10  # an order of magnitude over.

    report = compare_cell(
        workload_id="smoke",
        iou_type="boundary",
        impl_tensors={"vernier": zero_tensor, "boundary-iou-api": t2},
        impl_sha256={"vernier": "a" * 64, "boundary-iou-api": "b" * 64},
    )
    boundary = next(t for t in report.tiers if t.tier == "boundary")
    assert not boundary.passed
    assert boundary.first_divergence is not None
    assert boundary.first_divergence.index == target


def test_write_report_persists_round_trippable_json(tmp_path, zero_tensor: np.ndarray) -> None:
    target = (2, 10, 0, 1, 0)
    t2 = zero_tensor.copy()
    t2[target] = 1.0

    report = compare_cell(
        workload_id="smoke",
        iou_type="bbox",
        impl_tensors={"vernier": zero_tensor, "pycocotools": t2},
        impl_sha256={"vernier": "a" * 64, "pycocotools": "b" * 64},
    )
    out = write_report(report, tmp_path)
    assert out == tmp_path / "divergence_report.json"

    payload = json.loads(out.read_text())
    assert payload["schema_version"] == 1
    assert payload["iou_type"] == "bbox"
    strict = next(t for t in payload["tiers"] if t["tier"] == "strict")
    assert strict["passed"] is False
    assert strict["first_divergence"]["index"] == list(target)


def test_cell_run_dataclass_carries_parity() -> None:
    """``CellRun`` is the orchestrator's view of a finished cell. The
    ``parity`` field mirrors the report; ``divergence_report_path`` is
    set iff a report file was written."""
    run = CellRun(impl_jsons={}, parity=None, divergence_report_path=None, iqr_outcomes={})
    assert run.parity is None
    assert run.divergence_report_path is None
    assert run.iqr_outcomes == {}
