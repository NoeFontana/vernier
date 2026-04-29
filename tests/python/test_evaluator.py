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
from vernier import Bbox, Boundary, Evaluator, Keypoints, Segm, Summary

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


def test_evaluator_default_max_dets_is_sentinel() -> None:
    # ADR-0012: the field stores the sentinel; resolution happens at
    # dispatch via ``_resolve_max_dets``.
    assert Evaluator().max_dets is None


def test_resolve_max_dets_uses_kernel_default_for_bbox() -> None:
    assert Evaluator()._resolve_max_dets() == [1, 10, 100]


def test_resolve_max_dets_uses_kernel_default_for_segm() -> None:
    assert Evaluator(iou=Segm())._resolve_max_dets() == [1, 10, 100]


def test_resolve_max_dets_uses_kernel_default_for_boundary() -> None:
    assert Evaluator(iou=Boundary())._resolve_max_dets() == [1, 10, 100]


def test_resolve_max_dets_explicit_override_wins() -> None:
    assert Evaluator(max_dets=(1, 5))._resolve_max_dets() == [1, 5]


def test_resolve_max_dets_empty_tuple_is_legal_explicit_value() -> None:
    # An empty tuple is a deliberate caller choice ("evaluate at no
    # ladder rungs") and must not collapse back to the kernel default.
    assert Evaluator(max_dets=())._resolve_max_dets() == []


def test_with_options_leaves_max_dets_unchanged_by_default() -> None:
    base = Evaluator(max_dets=(2, 5))
    derived = base.with_options(parity_mode="strict")
    assert derived.max_dets == (2, 5)


def test_with_options_max_dets_none_resets_to_kernel_canonical() -> None:
    base = Evaluator(max_dets=(2, 5))
    reset = base.with_options(max_dets=None)
    assert reset.max_dets is None
    assert reset._resolve_max_dets() == [1, 10, 100]


def test_with_options_max_dets_tuple_overrides() -> None:
    base = Evaluator()
    derived = base.with_options(max_dets=(2, 5))
    assert derived.max_dets == (2, 5)
    assert derived._resolve_max_dets() == [2, 5]


def test_module_exports_public_api() -> None:
    for name in (
        "Bbox",
        "Boundary",
        "Evaluator",
        "IouKind",
        "Keypoints",
        "ParityMode",
        "Segm",
        "Summary",
        "version",
    ):
        assert hasattr(vernier, name), f"missing public export: {name}"


# --- Keypoints (OKS / ADR-0012) -----------------------------------------------

# Synthetic kp fixture: one image, one person GT with 17 visible keypoints,
# one DT with the same shape and a high score. AP@all should land at 1.0
# because the predicted keypoints are byte-identical to the GT's.
_KP_COORDS: tuple[tuple[float, float], ...] = (
    (10.0, 10.0),
    (12.0, 8.0),
    (8.0, 8.0),
    (14.0, 9.0),
    (6.0, 9.0),
    (16.0, 20.0),
    (4.0, 20.0),
    (18.0, 30.0),
    (2.0, 30.0),
    (20.0, 40.0),
    (0.0, 40.0),
    (14.0, 50.0),
    (6.0, 50.0),
    (16.0, 65.0),
    (4.0, 65.0),
    (18.0, 80.0),
    (2.0, 80.0),
)


def _flatten_kp(coords: tuple[tuple[float, float], ...], visibility: int = 2) -> list[float]:
    flat: list[float] = []
    for x, y in coords:
        flat.extend((x, y, float(visibility)))
    return flat


GT_KP = json.dumps(
    {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                # bbox area lands the annotation in the "large" kp area
                # bucket (>32**2 == 1024).
                "bbox": [0, 0, 40, 80],
                "area": 3200,
                "iscrowd": 0,
                "num_keypoints": 17,
                "keypoints": _flatten_kp(_KP_COORDS),
            },
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
).encode()

DT_KP = json.dumps(
    [
        {
            "image_id": 1,
            "category_id": 1,
            "score": 0.99,
            # DT carries an explicit bbox (quirk J3 derives area from it);
            # the kp kernel still consumes the bbox for area-bucket binning.
            "bbox": [0, 0, 40, 80],
            "keypoints": _flatten_kp(_KP_COORDS),
        },
    ]
).encode()


def test_keypoints_kernel_max_dets_default() -> None:
    # Per ADR-0012, the OKS kernel ladder is the single rung (20,) — distinct
    # from the (1, 10, 100) detection ladder shared by Bbox/Segm/Boundary.
    assert Evaluator(iou=Keypoints())._resolve_max_dets() == [20]


def test_keypoints_max_dets_explicit_override() -> None:
    assert Evaluator(iou=Keypoints(), max_dets=(50,))._resolve_max_dets() == [50]


def test_keypoints_with_options_resets_to_kernel_default() -> None:
    base = Evaluator(iou=Keypoints(), max_dets=(50,))
    reset = base.with_options(max_dets=None)
    assert reset.max_dets is None
    assert reset._resolve_max_dets() == [20]


def test_keypoints_default_sigmas_is_empty_mapping() -> None:
    # The default sigmas mapping is empty; the FFI maps an empty dict to
    # pycocotools' COCO-person 17-sigma table for every category (quirk F1).
    assert Keypoints().sigmas == {}


def test_keypoints_evaluator_dispatches_to_oks() -> None:
    summary = Evaluator(iou=Keypoints(), parity_mode="strict").evaluate(GT_KP, DT_KP)
    assert isinstance(summary, Summary)
    # Quirk D5: kp summary is 10 stats (re-indexed A-axis, no `_S` row).
    assert len(summary.stats) == 10
    # AP@all and AR@all collapse to 1.0 on a perfect prediction.
    ap_all, ar_all = summary.stats[0], summary.stats[5]
    assert ap_all == pytest.approx(1.0)
    assert ar_all == pytest.approx(1.0)


def test_keypoints_per_category_sigmas() -> None:
    # Override the COCO-person table with a uniform per-keypoint sigma for
    # category 1; assert evaluation succeeds and shape is unchanged.
    custom = Keypoints(sigmas={1: (0.05,) * 17})
    summary = Evaluator(iou=custom, parity_mode="strict").evaluate(GT_KP, DT_KP)
    assert isinstance(summary, Summary)
    assert len(summary.stats) == 10
