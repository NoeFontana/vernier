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

# ADR-0047: cross-thread strict-mode bit-equality axis. The vernier
# candidate runs at each thread count and the result is asserted bit-
# equal both to pycocotools (the existing oracle) and to its own
# sequential output (the new property). `None` is the library default
# and exercises the `VERNIER_NUM_THREADS`-fallback path; `1` pins the
# explicit-sequential shape; `2/4/8` exercise the parallel path under
# different rayon scheduling regimes.
ADR_0047_THREAD_COUNTS: tuple[int | None, ...] = (None, 1, 2, 4, 8)


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


@pytest.mark.parity_threads
@pytest.mark.parametrize(("fixture", "iou_type"), PARITY_CASES)
@pytest.mark.parametrize("num_threads", ADR_0047_THREAD_COUNTS)
def test_parity_across_thread_counts_strict_bit_equal(
    fixture: str, iou_type: IouType, num_threads: int | None
) -> None:
    """ADR-0047 load-bearing parity assertion: every existing fixture,
    every paradigm, every thread count produces stats bit-equal to the
    sequential (`num_threads=None`) baseline.

    This is a stronger property than vernier-vs-pycocotools — it
    catches future regressions that introduce a parallel f64 reduction
    in the wrong place, even when the parallel result still
    incidentally agrees with pycocotools to 4 ULP.
    """
    gt = FIXTURES / fixture / "gt.json"
    dt = FIXTURES / fixture / "dt.json"
    baseline = snapshot("vernier", gt, dt, iou_type, num_threads=None)
    cand = snapshot("vernier", gt, dt, iou_type, num_threads=num_threads)
    assert_snapshots_equal(baseline, cand)


@pytest.mark.parity_threads
@pytest.mark.parametrize(("fixture", "sigmas"), KEYPOINTS_FIXTURES)
@pytest.mark.parametrize("num_threads", ADR_0047_THREAD_COUNTS)
def test_keypoints_parity_across_thread_counts(
    fixture: str, sigmas: SigmasMap | None, num_threads: int | None
) -> None:
    """ADR-0047 cross-thread bit-equality for the keypoints (OKS) kernel."""
    gt = FIXTURES / fixture / "gt.json"
    dt = FIXTURES / fixture / "dt.json"
    baseline = snapshot("vernier", gt, dt, "keypoints", sigmas=sigmas, num_threads=None)
    cand = snapshot("vernier", gt, dt, "keypoints", sigmas=sigmas, num_threads=num_threads)
    assert_snapshots_equal(baseline, cand)


# ADR-0047 Stage B: BackgroundEvaluator's per-worker scoped pool must
# produce the same stats as the batch `Evaluator.evaluate(num_threads=N)`
# path on the same inputs. The substrate is shared, the property
# carries through — this test pins it across the full thread-count
# sweep so a regression in either surface is caught here, not at
# integration time. Bbox `perfect_match` is the canonical fixture; one
# fixture is sufficient because the cross-thread bit-equality property
# is already exercised on every existing fixture by the two tests
# above — this test asserts the batch-vs-streaming axis specifically.
_BACKGROUND_THREAD_COUNTS: tuple[int | None, ...] = (None, 1, 2, 4)


@pytest.mark.parity_threads
@pytest.mark.parametrize("num_threads", _BACKGROUND_THREAD_COUNTS)
def test_background_matches_batch_across_thread_counts(num_threads: int | None) -> None:
    """ADR-0047 Stage B: ``BackgroundEvaluator(num_threads=N)`` and
    ``Evaluator.evaluate(..., num_threads=N)`` produce bit-equal stats
    for the same fixture, across every thread count in the sweep.

    Pins:
    - ``None`` (today's BackgroundEvaluator default) is byte-equal to
      the batch sequential path — ADR-0014's one-core bound is the
      default and the trainer persona observes zero change.
    - ``1`` collapses to the same code path as ``None``.
    - ``2``/``4`` exercise the per-worker scoped rayon pool (built at
      worker spawn, owned by the worker, dropped at worker exit).
    """
    import vernier._core as _vernier_core
    from vernier.instance import Bbox, Evaluator

    gt_path = FIXTURES / "perfect_match" / "gt.json"
    dt_path = FIXTURES / "perfect_match" / "dt.json"
    gt_bytes = gt_path.read_bytes()
    dt_bytes = dt_path.read_bytes()

    # Batch path: `Evaluator(...).evaluate(gt, dt, num_threads=N)`.
    # `parity_mode="strict"` matches the parity harness's contract;
    # `use_cats` defaults to True; default max_dets=(1, 10, 100).
    evaluator = Evaluator(iou=Bbox(), parity_mode="strict")
    batch_summary = evaluator.evaluate(gt_bytes, dt_bytes, num_threads=num_threads)
    batch_stats = list(batch_summary.stats)

    # Streaming path: spawn a BackgroundEvaluator, submit the whole
    # batch in one shot, finalize. The single-submit shape is the
    # bit-equal-to-batch slice of ADR-0013's determinism contract.
    bg = _vernier_core.BackgroundEvaluator(
        gt_bytes,
        iou_type="bbox",
        parity_mode="strict",
        max_dets=[1, 10, 100],
        use_cats=True,
        num_threads=num_threads,
    )
    try:
        bg.submit(dt_bytes)
        stream_summary = bg.finalize()
    finally:
        # finalize() consumes the worker; the `try/finally` is here for
        # the early-error path. No-op once finalize() succeeded.
        pass
    stream_stats = list(stream_summary.stats)

    assert batch_stats == stream_stats, (
        f"BackgroundEvaluator(num_threads={num_threads}) vs "
        f"Evaluator.evaluate(num_threads={num_threads}) stats differ:\n"
        f"  batch:  {batch_stats}\n"
        f"  stream: {stream_stats}"
    )


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
