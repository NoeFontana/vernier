"""Identical tensors pass every tier. Skipped pairs (the requested
impl pair didn't both run) drop out silently."""

from __future__ import annotations

import numpy as np

from bench.harness.parity import (
    ALIGNED_ATOL,
    BOUNDARY_PARITY_EPS,
    CellParityReport,
    compare_cell,
)


def test_strict_and_aligned_pass_on_identical_tensors(zero_tensor: np.ndarray) -> None:
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
    assert {tier.tier for tier in report.tiers} == {"strict", "aligned"}
    assert report.passed
    assert all(t.divergent_count == 0 and t.first_divergence is None for t in report.tiers)


def test_boundary_tier_passes_on_identical_tensors(zero_tensor: np.ndarray) -> None:
    report = compare_cell(
        workload_id="smoke",
        iou_type="boundary",
        impl_tensors={"vernier": zero_tensor, "boundary-iou-api": zero_tensor},
        impl_sha256={"vernier": "a" * 64, "boundary-iou-api": "b" * 64},
    )
    assert [tier.tier for tier in report.tiers] == ["boundary"]
    assert report.tiers[0].atol == BOUNDARY_PARITY_EPS
    assert report.passed


def test_aligned_tier_skipped_when_pair_incomplete(zero_tensor: np.ndarray) -> None:
    report = compare_cell(
        workload_id="smoke",
        iou_type="bbox",
        impl_tensors={"vernier": zero_tensor, "pycocotools": zero_tensor},
        impl_sha256={"vernier": "a" * 64, "pycocotools": "b" * 64},
    )
    assert [tier.tier for tier in report.tiers] == ["strict"]
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


def test_aligned_tier_passes_within_tolerance(zero_tensor: np.ndarray) -> None:
    t2 = zero_tensor.copy()
    # Inside the 4-ULP atol — strict would reject this, aligned must
    # accept it.
    t2[0, 0, 0, 0, 0] = ALIGNED_ATOL / 2
    report = compare_cell(
        workload_id="smoke",
        iou_type="bbox",
        impl_tensors={"vernier": zero_tensor, "faster-coco-eval": t2},
        impl_sha256={"vernier": "a" * 64, "faster-coco-eval": "b" * 64},
    )
    aligned = next(t for t in report.tiers if t.tier == "aligned")
    assert aligned.passed
    assert aligned.divergent_count == 0
