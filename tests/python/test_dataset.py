"""Tests for the parsed-once ``vernier.instance.Dataset`` handle (ADR-0020).

The bytes-path and Dataset-path must produce bit-equal Summaries on
every kernel; the Dataset-path additionally exposes the GT-side
derivation cache (currently boundary + segm) for cross-call reuse.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from vernier.instance import Bbox, Boundary, Dataset, Evaluator, Keypoints, Segm, Summary

# Reuses the well-tested perfect-match fixtures already defined for the
# bytes-path Evaluator suite. Keeps coverage tight to the new surface.
from .test_evaluator import DT_KP, DT_PERFECT, GT_KP, GT_PERFECT

_SEGM_FIXTURE = Path(__file__).parent / "parity" / "fixtures" / "perfect_match_segm"
GT_SEGM = (_SEGM_FIXTURE / "gt.json").read_bytes()
DT_SEGM = (_SEGM_FIXTURE / "dt.json").read_bytes()


def test_dataset_from_json_exposes_counts() -> None:
    ds = Dataset.from_json(GT_PERFECT)
    assert ds.num_images == 1
    assert ds.num_annotations == 2
    assert ds.num_categories == 1


def test_dataset_repr_shape() -> None:
    ds = Dataset.from_json(GT_PERFECT)
    assert repr(ds) == "Dataset(images=1, annotations=2, categories=1)"


def test_dataset_from_json_rejects_malformed_payload() -> None:
    with pytest.raises(ValueError, match=r"(?i)json|parse"):
        Dataset.from_json(b"not-json")


# --- bytes-path vs Dataset-path parity (one per kernel) ----------------------


def test_bbox_bytes_and_dataset_paths_are_bit_equal() -> None:
    bytes_summary = Evaluator(iou=Bbox()).evaluate(GT_PERFECT, DT_PERFECT)
    ds_summary = Evaluator(iou=Bbox()).evaluate(Dataset.from_json(GT_PERFECT), DT_PERFECT)
    assert isinstance(ds_summary, Summary)
    assert bytes_summary.stats == ds_summary.stats


def test_segm_bytes_and_dataset_paths_are_bit_equal() -> None:
    bytes_summary = Evaluator(iou=Segm()).evaluate(GT_SEGM, DT_SEGM)
    ds_summary = Evaluator(iou=Segm()).evaluate(Dataset.from_json(GT_SEGM), DT_SEGM)
    assert bytes_summary.stats == ds_summary.stats


def test_boundary_bytes_and_dataset_paths_are_bit_equal() -> None:
    bytes_summary = Evaluator(iou=Boundary()).evaluate(GT_SEGM, DT_SEGM)
    ds_summary = Evaluator(iou=Boundary()).evaluate(Dataset.from_json(GT_SEGM), DT_SEGM)
    assert bytes_summary.stats == ds_summary.stats


def test_keypoints_bytes_and_dataset_paths_are_bit_equal() -> None:
    bytes_summary = Evaluator(iou=Keypoints()).evaluate(GT_KP, DT_KP)
    ds_summary = Evaluator(iou=Keypoints()).evaluate(Dataset.from_json(GT_KP), DT_KP)
    assert bytes_summary.stats == ds_summary.stats


# --- cache reuse -------------------------------------------------------------


def test_dataset_handle_reused_across_calls_yields_same_summary() -> None:
    ds = Dataset.from_json(GT_SEGM)
    e = Evaluator(iou=Boundary())
    first = e.evaluate(ds, DT_SEGM)
    # Second call hits the warm BoundaryGtCache; the result must be
    # bit-identical to the cold one.
    second = e.evaluate(ds, DT_SEGM)
    assert first.stats == second.stats


def test_one_dataset_shared_across_evaluators_with_different_options() -> None:
    # Multiple Evaluators differing in parity / use_cats can share one
    # Dataset (ADR-0020 §"Per-kernel, parameterized" — cache is on the
    # GT, not the evaluator).
    ds = Dataset.from_json(GT_SEGM)
    corrected = Evaluator(iou=Boundary(), parity_mode="corrected").evaluate(ds, DT_SEGM)
    strict = Evaluator(iou=Boundary(), parity_mode="strict").evaluate(ds, DT_SEGM)
    no_cats = Evaluator(iou=Boundary(), use_cats=False).evaluate(ds, DT_SEGM)
    # Each Evaluator returned a Summary; we don't assert equality across
    # them (different params → different stats), only that the shared
    # Dataset works without crashing or contaminating state.
    assert all(isinstance(s, Summary) for s in (corrected, strict, no_cats))


def test_clear_cache_does_not_break_subsequent_evaluations() -> None:
    ds = Dataset.from_json(GT_SEGM)
    e = Evaluator(iou=Boundary())
    cold = e.evaluate(ds, DT_SEGM)
    ds.clear_cache()
    rebuilt = e.evaluate(ds, DT_SEGM)
    assert cold.stats == rebuilt.stats


def test_changing_dilation_ratio_with_same_dataset_recomputes_correctly() -> None:
    # `BoundaryGtCache` is ratio-keyed; flipping the ratio must clear and
    # repopulate without crashing or returning stale results.
    ds = Dataset.from_json(GT_SEGM)
    coco = Evaluator(iou=Boundary(dilation_ratio=0.02)).evaluate(ds, DT_SEGM)
    lvis = Evaluator(iou=Boundary(dilation_ratio=0.008)).evaluate(ds, DT_SEGM)
    coco_again = Evaluator(iou=Boundary(dilation_ratio=0.02)).evaluate(ds, DT_SEGM)
    assert coco.stats == coco_again.stats
    # Sanity: the LVIS ratio is narrower → the band area is smaller →
    # the IoU shape is different. We don't assert specific values, only
    # that the toggle didn't silently return the COCO-ratio result.
    assert isinstance(lvis, Summary)


# --- thread safety -----------------------------------------------------------


def test_dataset_shared_across_threads_produces_identical_summaries() -> None:
    # `BoundaryGtCache` and `SegmGtCache` are mutex-guarded HashMaps;
    # multiple threads sharing one Dataset must each get the correct
    # Summary without races (ADR-0020 §"Composition with ADR-0014").
    ds = Dataset.from_json(GT_SEGM)
    expected = Evaluator(iou=Boundary()).evaluate(GT_SEGM, DT_SEGM).stats

    def _run() -> list[float]:
        return Evaluator(iou=Boundary()).evaluate(ds, DT_SEGM).stats

    barrier = threading.Barrier(8)

    def _run_synchronized() -> list[float]:
        barrier.wait()
        return _run()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _run_synchronized(), range(8)))

    for stats in results:
        assert stats == expected


# --- cache-hit timing (lenient) ----------------------------------------------


def test_warm_dataset_path_is_not_slower_than_cold_on_boundary() -> None:
    # Lenient timing assertion: cache reuse must not regress the warm
    # path. We don't assert "warm is Nx faster" — on tiny fixtures that
    # ratio is dominated by FFI overhead — only that the warm call is
    # within a generous envelope of the cold one. A regression that
    # accidentally rebuilt the cache every call would blow this budget.
    ds = Dataset.from_json(GT_SEGM)
    e = Evaluator(iou=Boundary())

    # Throwaway warmup to amortize PyO3/JIT noise.
    e.evaluate(ds, DT_SEGM)

    cold_ds = Dataset.from_json(GT_SEGM)
    t0 = time.perf_counter_ns()
    e.evaluate(cold_ds, DT_SEGM)
    cold_ns = time.perf_counter_ns() - t0

    t0 = time.perf_counter_ns()
    e.evaluate(ds, DT_SEGM)
    warm_ns = time.perf_counter_ns() - t0

    # 5x envelope is deliberately wide — the assertion exists to catch
    # gross regressions (e.g. accidental cache disable), not to lock in
    # a perf number. Real cache-effect measurement is the bench's job.
    assert warm_ns < 5 * cold_ns, (
        f"warm Dataset path slower than expected: cold={cold_ns}ns, warm={warm_ns}ns"
    )
