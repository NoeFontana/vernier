"""End-to-end tests for the Phase 1 ``vernier.Evaluator`` extended API.

The Rust core's algorithmic behavior is exercised by the Rust test
suite. These tests verify that the FFI bridge wires the dataset →
evaluate → accumulate → summarize pipeline together correctly and that
the immutable Python ``Evaluator`` shape behaves as documented.
"""

from __future__ import annotations

import json

import pytest

import vernier
from vernier import Bbox, Boundary, Evaluator, Segm, Summary

# Two perfectly-overlapping detections on a single image; expected stats
# collapse to 1.0 across every populated bucket and -1.0 elsewhere
# (quirk C5).
GT_PERFECT = json.dumps(
    {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [50, 50, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "thing"}],
    }
).encode()

DT_PERFECT = json.dumps(
    [
        {"image_id": 1, "category_id": 1, "score": 0.9, "bbox": [0, 0, 10, 10]},
        {"image_id": 1, "category_id": 1, "score": 0.8, "bbox": [50, 50, 10, 10]},
    ]
).encode()


# Indices into the canonical 12-stat pycocotools detection vector; see
# docs/reference/coco-summary-stats.md (Phase 1 deliverable) for the full table.
AP, AP_S, AP_M, AP_L, AR_100 = 0, 3, 4, 5, 8


def test_evaluator_perfect_match_yields_perfect_ap() -> None:
    summary = Evaluator().evaluate(GT_PERFECT, DT_PERFECT)
    stats = summary.stats
    assert isinstance(summary, Summary)
    # AP, AP_S populated; AP_M / AP_L absent (quirk C5 → -1).
    assert stats[AP] == pytest.approx(1.0)
    assert stats[AP_S] == pytest.approx(1.0)
    assert stats[AP_M] == -1.0
    assert stats[AP_L] == -1.0
    assert stats[AR_100] == pytest.approx(1.0)


def test_evaluator_pretty_lines_match_pycocotools_shape() -> None:
    summary = Evaluator().evaluate(GT_PERFECT, DT_PERFECT)
    lines = summary.pretty_lines()
    assert len(lines) == 12
    # Format mirrors pycocotools' fixed-width column (leading space, two
    # spaces around the metric tag); end of line is the value.
    assert "Average Precision" in lines[0]
    assert "(AP) @[" in lines[0]
    assert "Average Recall" in lines[8]
    assert "(AR) @[" in lines[8]
    assert lines[0].rstrip().endswith("1.000")


def test_evaluator_max_dets_order_does_not_affect_stats() -> None:
    # Quirk A2 (aligned): the FFI surface sorts `max_dets` at the
    # boundary, so a permuted ladder must produce the same `stats`
    # vector as the canonical order. Without the sort, the M-axis slots
    # bind to the wrong threshold and AR_1 / AR_10 / AR_100 silently
    # swap.
    canonical = Evaluator(max_dets=(1, 10, 100)).evaluate(GT_PERFECT, DT_PERFECT)
    permuted = Evaluator(max_dets=(100, 1, 10)).evaluate(GT_PERFECT, DT_PERFECT)
    assert canonical.stats == permuted.stats


def test_evaluator_is_immutable() -> None:
    e = Evaluator()
    # frozen dataclass raises FrozenInstanceError, a subclass of AttributeError.
    with pytest.raises(AttributeError):
        e.parity_mode = "strict"  # pyright: ignore[reportAttributeAccessIssue]


def test_with_options_returns_new_instance() -> None:
    base = Evaluator()
    strict = base.with_options(parity_mode="strict")
    assert base.parity_mode == "corrected"
    assert strict.parity_mode == "strict"
    assert base is not strict


def test_evaluator_rejects_non_kernel_iou() -> None:
    # The union is closed at the type level; the runtime fallthrough is
    # the only safety net for callers who bypass static checking.
    e = Evaluator(iou="bbox")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="unsupported iou kernel"):
        e.evaluate(GT_PERFECT, DT_PERFECT)


def test_evaluator_iou_kernels_construct_and_dispatch() -> None:
    assert isinstance(Evaluator().iou, Bbox)
    assert isinstance(Evaluator(iou=Segm()).iou, Segm)
    assert Evaluator(iou=Boundary()).iou == Boundary(dilation_ratio=0.02)
    boundary = Evaluator(iou=Boundary(dilation_ratio=0.008)).iou
    assert isinstance(boundary, Boundary)
    assert boundary.dilation_ratio == 0.008


def test_evaluator_rejects_invalid_parity_mode() -> None:
    e = Evaluator(parity_mode="aligned")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="parity_mode"):
        e.evaluate(GT_PERFECT, DT_PERFECT)


def test_evaluator_propagates_dataset_validation_errors() -> None:
    bad_gt = json.dumps(
        {
            "images": [{"id": 1, "width": 10, "height": 10}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 99,
                    "category_id": 1,
                    "bbox": [0, 0, 1, 1],
                    "area": 1,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "thing"}],
        }
    ).encode()
    with pytest.raises(ValueError, match="image_id=99"):
        Evaluator().evaluate(bad_gt, b"[]")


def test_module_exports_public_api() -> None:
    for name in (
        "Bbox",
        "Boundary",
        "Evaluator",
        "IouKind",
        "ParityMode",
        "Segm",
        "Summary",
        "version",
    ):
        assert hasattr(vernier, name), f"missing public export: {name}"
