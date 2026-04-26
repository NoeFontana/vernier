"""Parity tests against pycocotools 2.0.8.

Today both the reference and the candidate run pycocotools — `_run_vernier`
in harness.py is a delegating shim. The suite is therefore a tautology now,
but the plumbing (fixture corpus, snapshot capture, comparator, CI
integration) is real. As Rust evaluator components ship and the shim peels
back, the same test names start meaning something.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .harness import IouType, assert_snapshots_equal, snapshot

FIXTURES = Path(__file__).parent / "fixtures"

ALL_FIXTURES = [
    "perfect_match",
    "zero_overlap",
    "crowd_region",
    "missing_dt_image",
    "iou_at_threshold",
    "score_ties",
]


@pytest.mark.parity
@pytest.mark.parametrize("fixture", ALL_FIXTURES)
@pytest.mark.parametrize("iou_type", ["bbox"])
def test_parity_against_reference(fixture: str, iou_type: IouType) -> None:
    gt = FIXTURES / fixture / "gt.json"
    dt = FIXTURES / fixture / "dt.json"
    ref = snapshot("pycocotools", gt, dt, iou_type)
    cand = snapshot("vernier", gt, dt, iou_type)
    assert_snapshots_equal(ref, cand)


@pytest.mark.parity
def test_harness_catches_real_differences() -> None:
    # Sanity gate: if the comparator can't tell two genuinely different
    # fixtures apart, every real parity bug slips through silently.
    a = snapshot(
        "pycocotools",
        FIXTURES / "perfect_match" / "gt.json",
        FIXTURES / "perfect_match" / "dt.json",
        "bbox",
    )
    b = snapshot(
        "pycocotools",
        FIXTURES / "zero_overlap" / "gt.json",
        FIXTURES / "zero_overlap" / "dt.json",
        "bbox",
    )
    with pytest.raises(AssertionError):
        assert_snapshots_equal(a, b)


@pytest.mark.parity
def test_perfect_match_baseline_ap() -> None:
    # Without this absolute check, both `snapshot()` paths could be returning
    # all zeros and the parametrized parity tests would still pass.
    snap = snapshot(
        "pycocotools",
        FIXTURES / "perfect_match" / "gt.json",
        FIXTURES / "perfect_match" / "dt.json",
        "bbox",
    )
    assert snap.stats[0] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parity
def test_zero_overlap_baseline_ap() -> None:
    snap = snapshot(
        "pycocotools",
        FIXTURES / "zero_overlap" / "gt.json",
        FIXTURES / "zero_overlap" / "dt.json",
        "bbox",
    )
    assert snap.stats[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parity
def test_iou_at_threshold_only_matches_lowest_threshold() -> None:
    # Pins B1 (cocoeval.py:276): `iou = min(t, 1 - 1e-10)` boundary fudge.
    # IoU is exactly 0.5 → matches at iouThr=0.5, fails at iouThr>=0.55.
    # Mean AP across 10 thresholds should be ~0.1.
    snap = snapshot(
        "pycocotools",
        FIXTURES / "iou_at_threshold" / "gt.json",
        FIXTURES / "iou_at_threshold" / "dt.json",
        "bbox",
    )
    assert snap.stats[0] == pytest.approx(0.1, abs=1e-9)
