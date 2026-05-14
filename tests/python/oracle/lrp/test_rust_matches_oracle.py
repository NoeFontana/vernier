"""Rust <-> numpy-oracle parity for LRP / oLRP.

The numpy oracle in ``oracle.py`` is the executable spec; once the Rust
``vernier.instance.optimal_lrp`` (plus the ``segm`` / ``boundary`` /
``keypoints`` kernel variants and the panoptic counterpart) lands, this
file enforces ``|delta_rust - delta_oracle| < 1e-9`` per fixture.

This file ships pre-skipped: ``optimal_lrp`` has not yet been wired into
the Python wrapper, so the import below sets ``_HAS_LRP = False`` and
``pytestmark`` skips every test in the module. The moment the Rust
shipping PR adds ``vernier.instance.optimal_lrp`` (and the matching
``vernier.panoptic.optimal_lrp``), these tests start running with no
edit needed.

Tolerance: ``1e-9`` — the oracle is pure-Python f64 and the Rust
implementation will be f64 throughout. Anything wider is a real
disagreement. Loosening it silently masks a Rust bug — STOP and report
instead of widening here.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from .oracle import bbox_iou, optimal_lrp

rust_optimal_lrp: Any
Bbox: Any
Segm: Any
Boundary: Any
Keypoints: Any
try:
    from vernier.instance import (
        Bbox,
        Boundary,
        Keypoints,
        Segm,
    )
    from vernier.instance import optimal_lrp as rust_optimal_lrp

    _HAS_LRP = True
except (ImportError, AttributeError):
    _HAS_LRP = False
    rust_optimal_lrp = None
    Bbox = Segm = Boundary = Keypoints = None

panoptic_optimal_lrp: Any
try:
    from vernier.panoptic import optimal_lrp as panoptic_optimal_lrp

    _HAS_PANOPTIC_LRP = True
except (ImportError, AttributeError):
    _HAS_PANOPTIC_LRP = False
    panoptic_optimal_lrp = None

pytestmark = pytest.mark.skipif(
    not _HAS_LRP,
    reason="vernier.instance.optimal_lrp not yet shipped; parity tests skipped.",
)


def _to_coco_gt(gt: list[dict]) -> bytes:
    """Wrap an annotation list into a COCO GT dict and serialize."""
    image_ids = sorted({int(g["image_id"]) for g in gt})
    category_ids = sorted({int(g["category_id"]) for g in gt})
    coco = {
        "images": [{"id": i, "width": 640, "height": 480} for i in image_ids],
        "annotations": [{**g, "iscrowd": int(g.get("iscrowd", 0))} for g in gt],
        "categories": [{"id": c, "name": f"cat{c}"} for c in category_ids],
    }
    return json.dumps(coco).encode()


def _to_dt_bytes(dt: list[dict]) -> bytes:
    return json.dumps(dt).encode()


def _normalize_rust(report: Any) -> dict[str, Any]:
    """Project the shipped LrpReport dataclass into the oracle's dict shape."""
    olrp_per_class = {int(row.category_id): float(row.olrp) for row in report.per_class}
    tau_per_class = {int(row.category_id): float(row.tau) for row in report.per_class}
    return {
        "olrp": float(report.olrp),
        "loc": float(report.loc),
        "fp": float(report.fp),
        "fn": float(report.fn),
        "olrp_per_class": olrp_per_class,
        "tau_per_class": tau_per_class,
    }


# Per the module docstring contract.
TOL = 1e-9


def _gt(
    *,
    ann_id: int,
    image_id: int,
    category_id: int,
    bbox: list[float],
    iscrowd: int = 0,
) -> dict:
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": iscrowd,
    }


def _dt(*, image_id: int, category_id: int, bbox: list[float], score: float) -> dict:
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "score": score,
    }


def _assert_close(actual: float, expected: float, label: str) -> None:
    assert abs(actual - expected) < TOL, (
        f"{label}: rust={actual!r}, oracle={expected!r}, "
        f"diff={actual - expected!r} exceeds tolerance {TOL!r}"
    )


def _assert_reports_match(rust_out: dict[str, Any], oracle_out: dict[str, Any]) -> None:
    """Every headline + per-class value agrees within ``TOL``.

    NaNs propagate uniformly: per-class NaN in the oracle must coincide
    with per-class NaN in Rust (and vice versa). Headline NaN is treated
    the same way.
    """
    for key in ("olrp", "loc", "fp", "fn"):
        r, o = float(rust_out[key]), float(oracle_out[key])
        if np.isnan(o):
            assert np.isnan(r), f"{key}: rust={r!r}, oracle is NaN"
            continue
        _assert_close(r, o, key)
    rust_olrp_cls = rust_out["olrp_per_class"]
    oracle_olrp_cls = oracle_out["olrp_per_class"]
    assert set(rust_olrp_cls.keys()) == set(oracle_olrp_cls.keys())
    for k, o_v in oracle_olrp_cls.items():
        r_v = float(rust_olrp_cls[k])
        o_v = float(o_v)
        if np.isnan(o_v):
            assert np.isnan(r_v), f"olrp_per_class[{k}]: rust={r_v!r}, oracle is NaN"
            continue
        _assert_close(r_v, o_v, f"olrp_per_class[{k}]")
    rust_tau_cls = rust_out["tau_per_class"]
    oracle_tau_cls = oracle_out["tau_per_class"]
    assert set(rust_tau_cls.keys()) == set(oracle_tau_cls.keys())
    for k, o_v in oracle_tau_cls.items():
        r_v = float(rust_tau_cls[k])
        o_v = float(o_v)
        if np.isnan(o_v):
            assert np.isnan(r_v), f"tau_per_class[{k}]: rust={r_v!r}, oracle is NaN"
            continue
        _assert_close(r_v, o_v, f"tau_per_class[{k}]")


# ---------------------------------------------------------------------------
# Inline fixtures: the same shape as test_oracle.py uses, deliberately small
# so the parity tests run in milliseconds.
# ---------------------------------------------------------------------------


def _all_perfect() -> tuple[list[dict], list[dict]]:
    gt = [
        _gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10]),
        _gt(ann_id=2, image_id=1, category_id=2, bbox=[100, 100, 20, 20]),
    ]
    dt = [
        _dt(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=1.0),
        _dt(image_id=1, category_id=2, bbox=[100, 100, 20, 20], score=1.0),
    ]
    return gt, dt


def _single_tp_per_class_mixed_scores() -> tuple[list[dict], list[dict]]:
    gt = [
        _gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10]),
        _gt(ann_id=2, image_id=1, category_id=2, bbox=[100, 100, 20, 20]),
    ]
    dt = [
        _dt(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=0.8),
        _dt(image_id=1, category_id=1, bbox=[50, 50, 10, 10], score=0.3),
        _dt(image_id=1, category_id=2, bbox=[100, 100, 20, 20], score=0.6),
    ]
    return gt, dt


def _with_crowd() -> tuple[list[dict], list[dict]]:
    gt = [
        _gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10], iscrowd=1),
        _gt(ann_id=2, image_id=1, category_id=1, bbox=[100, 100, 10, 10], iscrowd=0),
    ]
    dt = [
        _dt(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=0.9),
        _dt(image_id=1, category_id=1, bbox=[100, 100, 10, 10], score=0.6),
        _dt(image_id=1, category_id=1, bbox=[200, 200, 10, 10], score=0.3),
    ]
    return gt, dt


_FIXTURES: dict[str, Callable[[], tuple[list[dict], list[dict]]]] = {
    "all_perfect": _all_perfect,
    "single_tp_mixed_scores": _single_tp_per_class_mixed_scores,
    "with_crowd": _with_crowd,
}


@pytest.mark.parametrize("name", list(_FIXTURES.keys()))
def test_rust_matches_oracle_bbox(name: str) -> None:
    """Rust bbox LRP matches the oracle within 1e-9 across the headline keys."""
    gt, dt = _FIXTURES[name]()
    assert rust_optimal_lrp is not None  # narrowed by pytestmark.
    rust_report = rust_optimal_lrp(_to_coco_gt(gt), _to_dt_bytes(dt), iou=Bbox(), tp_threshold=0.5)
    rust_out = _normalize_rust(rust_report)
    oracle_out = optimal_lrp(gt, dt, similarity_fn=bbox_iou, tp_threshold=0.5)
    _assert_reports_match(rust_out, oracle_out)


@pytest.mark.parametrize("name", list(_FIXTURES.keys()))
def test_rust_matches_oracle_segm(name: str) -> None:
    """Rust segm LRP matches the segm-oracle within 1e-9.

    The oracle's segm-kernel callable is built per fixture (this file
    does not vendor a segm rasterizer — the test imports the oracle's
    own bbox callable as the analytic kernel, since the fixtures here
    use axis-aligned bboxes whose segm mask is the same rectangle).
    Once the Rust segm path lands, the parity is bit-equal on
    integer-aligned rectangles.
    """
    gt, dt = _FIXTURES[name]()
    # Decorate each annotation with a polygon segmentation derived from
    # its bbox so the segm kernel has something to rasterize.
    for ann in (*gt, *dt):
        x, y, w, h = ann["bbox"]
        ann["segmentation"] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
    assert rust_optimal_lrp is not None
    rust_report = rust_optimal_lrp(_to_coco_gt(gt), _to_dt_bytes(dt), iou=Segm(), tp_threshold=0.5)
    rust_out = _normalize_rust(rust_report)
    # The oracle's bbox_iou agrees with mask-IoU on axis-aligned
    # integer-pixel rectangles (the only shapes in the fixtures), so we
    # reuse it as the analytic reference.
    oracle_out = optimal_lrp(gt, dt, similarity_fn=bbox_iou, tp_threshold=0.5)
    _assert_reports_match(rust_out, oracle_out)


@pytest.mark.parametrize("name", list(_FIXTURES.keys()))
def test_rust_matches_oracle_boundary(name: str) -> None:
    """Rust boundary LRP matches the oracle within 1e-9.

    The boundary kernel's hand-computability is the same story as the
    segm test: axis-aligned rectangles rasterize to identical mask /
    band sets in both implementations on integer-aligned rectangles, so
    the IoU values feeding LRP are bit-equal.
    """
    gt, dt = _FIXTURES[name]()
    for ann in (*gt, *dt):
        x, y, w, h = ann["bbox"]
        ann["segmentation"] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
    assert rust_optimal_lrp is not None
    rust_report = rust_optimal_lrp(
        _to_coco_gt(gt),
        _to_dt_bytes(dt),
        iou=Boundary(dilation_ratio=0.02),
        tp_threshold=0.5,
    )
    rust_out = _normalize_rust(rust_report)
    oracle_out = optimal_lrp(gt, dt, similarity_fn=bbox_iou, tp_threshold=0.5)
    _assert_reports_match(rust_out, oracle_out)


def _keypoints_perfect() -> tuple[list[dict], list[dict]]:
    """OKS-kernel fixture: one keypoint annotation per image, perfect OKS."""
    gt = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [0, 0, 100, 100],
            "area": 10000,
            "iscrowd": 0,
            "num_keypoints": 17,
            "keypoints": [50, 50, 2] + [0, 0, 0] * 16,
        }
    ]
    dt = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [0, 0, 100, 100],
            "score": 0.9,
            "keypoints": [50, 50, 2] + [0, 0, 0] * 16,
        }
    ]
    return gt, dt


def test_rust_matches_oracle_keypoints_perfect() -> None:
    """Rust keypoint (OKS) LRP matches the oracle for a perfect prediction.

    The OKS kernel is not implemented in this oracle (no vendored
    sigma table). The test asserts the Rust output against a hand-pinned
    expected report — once Rust ships, the constant values here pin
    the parity contract.
    """
    gt, dt = _keypoints_perfect()
    assert rust_optimal_lrp is not None
    rust_report = rust_optimal_lrp(
        _to_coco_gt(gt), _to_dt_bytes(dt), iou=Keypoints(), tp_threshold=0.5
    )
    rust_out = _normalize_rust(rust_report)
    # Hand-pinned expected: one perfect OKS match scoring 0.9 → NTP=1,
    # NFP=0, NFN=0 ⇒ LRP = 0 at every tau ≤ 0.9. At any tau > 0.9 the
    # TP is filtered out, the GT becomes an FN, and LRP = 1. The argmin
    # range is therefore [0.0, 0.9]; with "larger tau wins on tie"
    # (ADR-0043), the deployable tau is 0.9 — NOT 1.0.
    expected = {
        "olrp": 0.0,
        "loc": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "olrp_per_class": {1: 0.0},
        "tau_per_class": {1: 0.9},
    }
    _assert_reports_match(rust_out, expected)


@pytest.mark.skipif(
    not _HAS_PANOPTIC_LRP,
    reason="vernier.panoptic.optimal_lrp not yet shipped.",
)
def test_panoptic_lrp_raises_not_implemented() -> None:
    """Panoptic LRP entry point is reachable but not yet wired.

    The implementation agent found that panoptic predictions carry no
    per-segment score (``SegmentInfo`` has only ``id`` / ``category_id``
    / ``iscrowd`` / ``area``), so the tau sweep ADR-0043 specified has
    nothing to scan. The entry point ships as a typed stub that raises
    :class:`NotImplementedError` with a pointer to ADR-0043. This test
    pins that behaviour until the panoptic prediction format is
    extended.
    """
    assert panoptic_optimal_lrp is not None
    with pytest.raises(NotImplementedError):
        panoptic_optimal_lrp(b"{}", b"[]")
