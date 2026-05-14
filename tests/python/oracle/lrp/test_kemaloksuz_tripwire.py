"""Cross-check the LRP oracle against the vendored kemaloksuz/LRP-Error.

This is a SANITY GATE, not a parity contract. When this test fires it
means the oracle and the first-party reference implementation disagree
on a constructed fixture — investigate and resolve manually. Do NOT
auto-resolve by widening the tolerance or by silently re-pinning
expected values.

The numpy oracle in ``oracle.py`` is authoritative for vernier; this
test reports drift FROM the first-party reference, which is one of:

  - Our oracle has a real bug we missed in the hand-computed fixtures.
  - The first-party reference has a bug.
  - The two are computing subtly different things (e.g., distinct
    tau-grid resolutions, distinct argmin tie-break rules).

Any of those three is worth a human looking at the result — that is the
whole point of a tripwire.

The vendor tree at ``tests/python/oracle/lrp/vendor/lrp_error/`` is not
checked into the repo (see ``vendor/README.md`` for the
git-submodule / curl bootstrap recipe). When absent the test skips. The
``@pytest.mark.tripwire`` marker also lets CI exclude this test from
the default suite — tripwires run on demand, not on every PR.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pytest

from .oracle import bbox_iou, optimal_lrp

VENDOR_ROOT = Path(__file__).parent / "vendor" / "lrp_error"
_VENDOR_PRESENT = VENDOR_ROOT.is_dir() and (VENDOR_ROOT / "__init__.py").is_file()

pytestmark = [
    pytest.mark.tripwire,
    pytest.mark.skipif(
        not _VENDOR_PRESENT,
        reason=f"kemaloksuz/LRP-Error vendor tree not present at {VENDOR_ROOT}. "
        "See vendor/README.md for the bootstrap recipe.",
    ),
]

# Tripwire tolerance: 1e-6. Much looser than the Rust parity test
# (1e-9), tighter than typical f32-vs-f64 noise. The two implementations
# walk the same algorithm at f64 precision but may differ in tau-grid
# resolution, tie-breaks, or normalisation details — 1e-6 catches drift
# without firing on negligible numerics.
TRIPWIRE_TOL = 1e-6


def _gt(*, ann_id: int, image_id: int, category_id: int, bbox: list[float]) -> dict[str, Any]:
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": 0,
    }


def _dt(*, image_id: int, category_id: int, bbox: list[float], score: float) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "score": score,
    }


def _run_kemaloksuz(gt: list[dict[str, Any]], dt: list[dict[str, Any]]) -> dict[str, float]:
    """Invoke the vendored kemaloksuz ``LRPError`` on the same fixture.

    Schema-translation goes here. The vendor's ``LRPError`` class
    (canonical entry point in `LRP-Error` repo) typically takes
    cocoEval-style detections and ground truth; this shim builds the
    minimal in-memory analogues. The exact API surface depends on the
    pinned commit (see vendor/README.md); the import path here matches
    a vendoring of the upstream into ``tests/python/oracle/lrp/vendor/lrp_error/``.

    Returns a dict with at least the four keys ``olrp``, ``loc``, ``fp``,
    ``fn`` so the assertion site below is small.
    """
    sys.path.insert(0, str(VENDOR_ROOT.parent))
    try:
        # The exact import varies by upstream commit; this is the path
        # the README pins.
        from lrp_error import LRPError  # type: ignore[import-not-found]
    finally:
        if str(VENDOR_ROOT.parent) in sys.path:
            sys.path.remove(str(VENDOR_ROOT.parent))

    # Schema translation — keep the bridge minimal. The vendor expects
    # COCO-style structures; we mirror the test's inline dicts.
    coco_gt = {
        "images": sorted({g["image_id"] for g in gt} | {d["image_id"] for d in dt}),
        "annotations": gt,
        "categories": sorted({g["category_id"] for g in gt} | {d["category_id"] for d in dt}),
    }
    lrp = LRPError(coco_gt, dt, tau=0.5)
    result = lrp.evaluate()
    # Coerce to a flat dict keyed the same way the oracle returns.
    return {
        "olrp": float(result.get("olrp", float("nan"))),
        "loc": float(result.get("loc", float("nan"))),
        "fp": float(result.get("fp", float("nan"))),
        "fn": float(result.get("fn", float("nan"))),
    }


def _assert_finite_close(actual: float, expected: float, label: str) -> None:
    if math.isnan(actual) and math.isnan(expected):
        return
    assert abs(actual - expected) < TRIPWIRE_TOL, (
        f"TRIPWIRE: {label}: oracle={actual!r}, kemaloksuz={expected!r}, "
        f"|diff|={abs(actual - expected)!r} exceeds {TRIPWIRE_TOL!r}. "
        "Investigate manually — see this file's docstring."
    )


def test_kemaloksuz_agrees_on_single_tp_per_class_fixture() -> None:
    """One TP at score=0.8 and one FP at score=0.3; oracle and kemaloksuz agree.

    This is a constructed fixture small enough that hand-derivation
    (see test_oracle.py::test_optimal_tau_lands_at_detection_score_breakpoint)
    pins oLRP=0, oLRP_Loc=0, oLRP_FP=0, oLRP_FN=0 in the oracle. The
    first-party reference implementation should agree within 1e-6.

    A failure means investigate — do NOT auto-resolve. See the module
    docstring for the disposition discipline.
    """
    gt = [_gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10])]
    dt = [
        _dt(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=0.8),
        _dt(image_id=1, category_id=1, bbox=[50, 50, 10, 10], score=0.3),
    ]

    oracle_out = optimal_lrp(gt, dt, similarity_fn=bbox_iou, tp_threshold=0.5)
    kemaloksuz_out = _run_kemaloksuz(gt, dt)

    _assert_finite_close(float(oracle_out["olrp"]), kemaloksuz_out["olrp"], "olrp")
    _assert_finite_close(float(oracle_out["loc"]), kemaloksuz_out["loc"], "loc")
    _assert_finite_close(float(oracle_out["fp"]), kemaloksuz_out["fp"], "fp")
    _assert_finite_close(float(oracle_out["fn"]), kemaloksuz_out["fn"], "fn")


def test_kemaloksuz_agrees_on_mixed_fixture() -> None:
    """A two-class fixture with one TP and one FN per class.

    Math (per class, identical):
        - 2 GTs cat k. 1 DT cat k matching one GT (IoU=1) score=0.7.
        - Active tau range gives NTP=1, NFP=0, NFN=1 always (one GT
          unmatched at any tau where the TP is active).
        - LRP(s) = (0 + 0 + 1) / 2 = 0.5 for s in [0, 0.7], and
          LRP(s) = (0 + 0 + 2) / 2 = 1.0 for s > 0.7. Min = 0.5 at
          tau_star = 0.7.
        - oLRP = 0.5. Components: Loc=0, FP=0, FN=1/(1+1)=0.5.
        - Two classes, identical -> headline = 0.5 / 0 / 0 / 0.5.

    The first-party reference should produce the same numbers within
    1e-6 — any drift fires the tripwire.
    """
    gt = [
        _gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10]),
        _gt(ann_id=2, image_id=1, category_id=1, bbox=[100, 100, 10, 10]),
        _gt(ann_id=3, image_id=2, category_id=2, bbox=[0, 0, 10, 10]),
        _gt(ann_id=4, image_id=2, category_id=2, bbox=[100, 100, 10, 10]),
    ]
    dt = [
        _dt(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=0.7),
        _dt(image_id=2, category_id=2, bbox=[0, 0, 10, 10], score=0.7),
    ]
    oracle_out = optimal_lrp(gt, dt, similarity_fn=bbox_iou, tp_threshold=0.5)
    kemaloksuz_out = _run_kemaloksuz(gt, dt)

    _assert_finite_close(float(oracle_out["olrp"]), kemaloksuz_out["olrp"], "olrp")
    _assert_finite_close(float(oracle_out["loc"]), kemaloksuz_out["loc"], "loc")
    _assert_finite_close(float(oracle_out["fp"]), kemaloksuz_out["fp"], "fp")
    _assert_finite_close(float(oracle_out["fn"]), kemaloksuz_out["fn"], "fn")

    # Sanity: pin the oracle's own value too so the test still has
    # something to assert if the vendor skip path ever changes.
    assert oracle_out["olrp"] == pytest.approx(0.5, abs=1e-12)
    assert oracle_out["loc"] == pytest.approx(0.0, abs=1e-12)
    assert oracle_out["fp"] == pytest.approx(0.0, abs=1e-12)
    assert oracle_out["fn"] == pytest.approx(0.5, abs=1e-12)
