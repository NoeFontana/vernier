"""Identical tensors pass every tier. Skipped pairs (the requested
impl pair didn't both run) drop out silently."""

from __future__ import annotations

import numpy as np

from bench.harness.parity import (
    BOUNDARY_PARITY_EPS,
    FLOAT_TOLERANCE_ATOL,
    CellParityReport,
    compare_cell,
)


def test_bit_equal_and_float_tolerance_pass_on_identical_tensors(zero_tensor: np.ndarray) -> None:
    report = compare_cell(
        workload_id="smoke",
        iou_type="bbox",
        impl_tensors={
            "vernier": zero_tensor,
            "pycocotools": zero_tensor,
            "faster-coco-eval": zero_tensor,
        },
        impl_sha256={
            "vernier": "a" * 64,
            "pycocotools": "b" * 64,
            "faster-coco-eval": "c" * 64,
        },
    )
    assert {tier.tier for tier in report.tiers} == {"bit-equal", "float-tolerance"}
    assert report.passed
    assert all(t.divergent_count == 0 and t.first_divergence is None for t in report.tiers)


def test_boundary_tier_passes_on_identical_tensors(zero_tensor: np.ndarray) -> None:
    report = compare_cell(
        workload_id="smoke",
        iou_type="boundary",
        impl_tensors={"vernier": zero_tensor, "boundary-iou-api": zero_tensor},
        impl_sha256={"vernier": "a" * 64, "boundary-iou-api": "b" * 64},
    )
    assert [tier.tier for tier in report.tiers] == ["boundary-tolerance"]
    assert report.tiers[0].atol == BOUNDARY_PARITY_EPS
    assert report.passed


def test_float_tolerance_tier_skipped_when_pair_incomplete(zero_tensor: np.ndarray) -> None:
    report = compare_cell(
        workload_id="smoke",
        iou_type="bbox",
        impl_tensors={"vernier": zero_tensor, "pycocotools": zero_tensor},
        impl_sha256={"vernier": "a" * 64, "pycocotools": "b" * 64},
    )
    assert [tier.tier for tier in report.tiers] == ["bit-equal"]
    assert report.tiers[0].atol == 0.0
    assert report.passed


def test_report_round_trips_through_pydantic(zero_tensor: np.ndarray) -> None:
    report = compare_cell(
        workload_id="smoke",
        iou_type="bbox",
        impl_tensors={"vernier": zero_tensor, "pycocotools": zero_tensor},
        impl_sha256={"vernier": "a" * 64, "pycocotools": "b" * 64},
    )
    json_str = report.model_dump_json()
    restored = CellParityReport.model_validate_json(json_str)
    assert restored == report


def test_float_tolerance_tier_passes_within_tolerance(zero_tensor: np.ndarray) -> None:
    t2 = zero_tensor.copy()
    # Inside the 4-ULP atol — bit-equal would reject this,
    # float-tolerance must accept it.
    t2[0, 0, 0, 0, 0] = FLOAT_TOLERANCE_ATOL / 2
    report = compare_cell(
        workload_id="smoke",
        iou_type="bbox",
        impl_tensors={"vernier": zero_tensor, "faster-coco-eval": t2},
        impl_sha256={"vernier": "a" * 64, "faster-coco-eval": "b" * 64},
    )
    banded = next(t for t in report.tiers if t.tier == "float-tolerance")
    assert banded.passed
    assert banded.divergent_count == 0
