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

from .harness import IouType, SigmasMap, assert_snapshots_equal, snapshot

FIXTURES = Path(__file__).parent / "fixtures"

BBOX_FIXTURES = [
    "perfect_match",
    "zero_overlap",
    "crowd_region",
    "missing_dt_image",
    "iou_at_threshold",
    "score_ties",
    "crowd_overlap_tiebreak",
]

SEGM_FIXTURES = [
    "perfect_match_segm",
    "zero_overlap_segm",
    "crowd_region_segm",
    "score_ties_segm",
    "missing_dt_image_segm",
    "multi_polygon_gt_segm",
    "polygon_at_image_edge_segm",
    "self_intersecting_polygon_segm",
    "crowd_rle_gt_segm",
    "boundary_area_segm",
    "heterogeneous_dt_segm",
]

# Per-category sigma override, expressed in pycocotools' post-divide form
# (i.e. matching the values stored on `Params.kpt_oks_sigmas` after
# `setKpParams` runs). Used by the F1 fixture below; the harness routes
# the same vector to both pycocotools (`params.kpt_oks_sigmas = ...`) and
# vernier (`Keypoints(sigmas={cat: ...})`).
_KEYPOINTS_F1_SIGMAS: tuple[float, ...] = tuple(0.05 for _ in range(17))

# Keypoints fixtures and the quirk(s) each pins. Authored alongside
# Phase 3 (ADR-0012); each entry exercises a specific quirk disposition
# in `docs/engineering/pycocotools-quirks.md`.
KEYPOINTS_FIXTURES: list[tuple[str, SigmasMap | None]] = [
    # Sanity baseline (AP=1.0). Nothing quirk-specific; pins that the
    # keypoints harness path is not all-zero.
    ("keypoints_perfect_match", None),
    # D2: GT with all visibilities=0 is implicit-ignore. The DT must not
    # be charged as a false positive.
    ("keypoints_zero_visibility", None),
    # F3 + F4: GT has zero visible keypoints but a real bbox. OKS falls
    # back to the 2x bbox surrogate; DT keypoints sit in the asymmetric
    # `[bb_x - bb_w, bb_x + 2*bb_w]` band.
    ("keypoints_no_keypoints_surrogate", None),
    # D5: keypoints kp grid drops the "small" bucket; verifies the
    # 3-bucket areaRng (all/medium/large) — GTs are sized to land in
    # medium and large buckets.
    ("keypoints_areaRng_buckets", None),
    # F1: per-category sigma override. Both sides see the same sigma
    # vector (single-category fixture).
    ("keypoints_custom_sigmas", {1: _KEYPOINTS_F1_SIGMAS}),
]

PARITY_CASES: list[tuple[str, IouType]] = [
    *((f, "bbox") for f in BBOX_FIXTURES),
    *((f, "segm") for f in SEGM_FIXTURES),
]


@pytest.mark.parity
@pytest.mark.parametrize(("fixture", "iou_type"), PARITY_CASES)
def test_parity_against_reference(fixture: str, iou_type: IouType) -> None:
    gt = FIXTURES / fixture / "gt.json"
    dt = FIXTURES / fixture / "dt.json"
    ref = snapshot("pycocotools", gt, dt, iou_type)
    cand = snapshot("vernier", gt, dt, iou_type)
    assert_snapshots_equal(ref, cand)


@pytest.mark.parity
@pytest.mark.parametrize(("fixture", "sigmas"), KEYPOINTS_FIXTURES)
def test_keypoints_parity_against_reference(fixture: str, sigmas: SigmasMap | None) -> None:
    gt = FIXTURES / fixture / "gt.json"
    dt = FIXTURES / fixture / "dt.json"
    ref = snapshot("pycocotools", gt, dt, "keypoints", sigmas=sigmas)
    cand = snapshot("vernier", gt, dt, "keypoints", sigmas=sigmas)
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


@pytest.mark.parity
def test_heterogeneous_dt_segm_rejects_in_corrected_mode() -> None:
    # Pins J6 (`corrected`): per-entry dispatch under `iouType="segm"`.
    # The fixture mixes a DT with `segmentation` and a DT without —
    # corrected mode raises rather than silently letting the first-entry
    # type decide for the whole list (the pycocotools quirk J6
    # documents).
    import vernier._core as _vernier_core

    gt_bytes = (FIXTURES / "heterogeneous_dt_segm" / "gt.json").read_bytes()
    dt_bytes = (FIXTURES / "heterogeneous_dt_segm" / "dt.json").read_bytes()
    with pytest.raises(ValueError, match="J2/J6"):
        _vernier_core.evaluate_segm_grid(gt_bytes, dt_bytes, "corrected", 100, use_cats=True)
