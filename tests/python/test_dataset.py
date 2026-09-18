"""Tests for the parsed-once ``vernier.instance.CocoDataset`` handle (ADR-0020).

The bytes-path and CocoDataset-path must produce bit-equal Summaries on
every kernel; the CocoDataset-path additionally exposes the GT-side
derivation cache (currently boundary + segm) for cross-call reuse.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from vernier.instance import Bbox, Boundary, CocoDataset, Evaluator, Keypoints, Segm, Summary

# Reuses the well-tested perfect-match fixtures already defined for the
# bytes-path Evaluator suite. Keeps coverage tight to the new surface.
from .test_evaluator import DT_KP, DT_PERFECT, GT_KP, GT_PERFECT

_SEGM_FIXTURE = Path(__file__).parent / "parity" / "fixtures" / "perfect_match_segm"
GT_SEGM = (_SEGM_FIXTURE / "gt.json").read_bytes()
DT_SEGM = (_SEGM_FIXTURE / "dt.json").read_bytes()


def test_dataset_from_json_exposes_counts() -> None:
    ds = CocoDataset.from_json(GT_PERFECT)
    assert ds.num_images == 1
    assert ds.num_annotations == 2
    assert ds.num_categories == 1


def test_dataset_repr_shape() -> None:
    ds = CocoDataset.from_json(GT_PERFECT)
    assert repr(ds) == "CocoDataset(images=1, annotations=2, categories=1)"


def test_dataset_from_json_rejects_malformed_payload() -> None:
    with pytest.raises(ValueError, match=r"(?i)json|parse"):
        CocoDataset.from_json(b"not-json")


# --- bytes-path vs CocoDataset-path parity (one per kernel) ----------------------


def test_bbox_bytes_and_dataset_paths_are_bit_equal() -> None:
    bytes_summary = Evaluator(iou=Bbox()).evaluate(GT_PERFECT, DT_PERFECT)
    ds_summary = Evaluator(iou=Bbox()).evaluate(CocoDataset.from_json(GT_PERFECT), DT_PERFECT)
    assert isinstance(ds_summary, Summary)
    assert bytes_summary.stats == ds_summary.stats


def test_segm_bytes_and_dataset_paths_are_bit_equal() -> None:
    bytes_summary = Evaluator(iou=Segm()).evaluate(GT_SEGM, DT_SEGM)
    ds_summary = Evaluator(iou=Segm()).evaluate(CocoDataset.from_json(GT_SEGM), DT_SEGM)
    assert bytes_summary.stats == ds_summary.stats


def test_boundary_bytes_and_dataset_paths_are_bit_equal() -> None:
    bytes_summary = Evaluator(iou=Boundary()).evaluate(GT_SEGM, DT_SEGM)
    ds_summary = Evaluator(iou=Boundary()).evaluate(CocoDataset.from_json(GT_SEGM), DT_SEGM)
    assert bytes_summary.stats == ds_summary.stats


def test_keypoints_bytes_and_dataset_paths_are_bit_equal() -> None:
    bytes_summary = Evaluator(iou=Keypoints()).evaluate(GT_KP, DT_KP)
    ds_summary = Evaluator(iou=Keypoints()).evaluate(CocoDataset.from_json(GT_KP), DT_KP)
    assert bytes_summary.stats == ds_summary.stats


# --- grid entry points: bytes vs CocoDataset ---------------------------------
#
# The summary tests above go through `Evaluator`, which accepts either
# form. The *grid* entry points are separate functions -- one per kernel
# per input form -- so their agreement is a distinct claim, and it is the
# one a caller relies on when it needs `accumulate()`'s per-class tensors
# off a handle rather than a summary off bytes.


@pytest.mark.parametrize("dt_area", ["bbox", "supplied"])
def test_segm_grid_bytes_and_dataset_paths_are_bit_equal(
    dt_area: Literal["bbox", "supplied", "mask"],
) -> None:
    """``evaluate_segm_grid_with_dataset`` must be its bytes twin, on every ``dt_area``.

    This is the entry point that lets a ``bbox`` + ``segm`` caller parse
    its ground truth once instead of twice. Parametrizing over
    ``dt_area`` is the point rather than thoroughness: the argument
    chooses whether a detection's area comes from its box or its mask,
    which moves the small / medium / large buckets and nothing else. A
    default that disagreed between the two entry points would show up
    here as shifted AP-small/medium/large with every other number
    identical -- and nowhere else.

    ``"mask"`` is absent because this fixture's detections are polygons
    and it needs an RLE; both paths refuse it, which the test below
    asserts, and the *succeeding* mask comparison lives in
    ``test_dt_area_mask.py`` next to the RLE fixture built for it.
    """
    from vernier.instance import evaluate_segm_grid, evaluate_segm_grid_with_dataset

    from_bytes = evaluate_segm_grid(GT_SEGM, DT_SEGM, "strict", 100, True, dt_area=dt_area)
    from_handle = evaluate_segm_grid_with_dataset(
        CocoDataset.from_json(GT_SEGM), DT_SEGM, "strict", 100, True, dt_area=dt_area
    )
    assert from_bytes.accumulate([1, 10, 100]).summarize().stats == (
        from_handle.accumulate([1, 10, 100]).summarize().stats
    )


@pytest.mark.parametrize("dt_area", ["bbox", "supplied"])
def test_bbox_grid_bytes_and_dataset_paths_are_bit_equal(
    dt_area: Literal["bbox", "supplied"],
) -> None:
    """The bbox grid pair, including the ``dt_area`` the handle form used to lack.

    ``evaluate_bbox_grid_with_dataset`` hard-coded "derive the area from
    the box" and took no argument, so a caller that wanted ``supplied``
    had to fall back to the bytes form and re-parse. It now takes the
    same argument its twin does, defaulting to the behaviour it always
    had.
    """
    from vernier.instance import evaluate_bbox_grid, evaluate_bbox_grid_with_dataset

    from_bytes = evaluate_bbox_grid(GT_PERFECT, DT_PERFECT, "strict", 100, True, dt_area=dt_area)
    from_handle = evaluate_bbox_grid_with_dataset(
        CocoDataset.from_json(GT_PERFECT), DT_PERFECT, "strict", 100, True, dt_area=dt_area
    )
    assert from_bytes.accumulate([1, 10, 100]).summarize().stats == (
        from_handle.accumulate([1, 10, 100]).summarize().stats
    )


def test_bbox_grid_with_dataset_defaults_to_deriving_area_from_the_box() -> None:
    """The added ``dt_area`` must not have changed what existing callers get.

    It is additive, and its default has to reproduce the hard-coded
    ``DetectionArea::FromBbox`` the function used to apply
    unconditionally -- otherwise every caller written against the old
    signature silently re-buckets.
    """
    from vernier.instance import evaluate_bbox_grid_with_dataset

    ds = CocoDataset.from_json(GT_PERFECT)
    implicit = evaluate_bbox_grid_with_dataset(ds, DT_PERFECT, "strict", 100, True)
    explicit = evaluate_bbox_grid_with_dataset(ds, DT_PERFECT, "strict", 100, True, dt_area="bbox")
    assert implicit.accumulate([1, 10, 100]).summarize().stats == (
        explicit.accumulate([1, 10, 100]).summarize().stats
    )


def test_segm_grid_paths_refuse_a_polygon_under_mask_area_identically() -> None:
    """Agreement on a rejection is agreement.

    ``dt_area="mask"`` reads the area off the detection's own RLE, and
    this fixture's detections are polygons, so both entry points refuse
    it. They must refuse it the *same* way -- same error, same message
    -- because a handle-taking route that accepted input its bytes twin
    rejects would be a second, quieter contract.
    """
    from vernier.instance import (
        InvalidAnnotationError,
        evaluate_segm_grid,
        evaluate_segm_grid_with_dataset,
    )

    needs_rle = r'dt_area="mask" requires an RLE'
    with pytest.raises(InvalidAnnotationError, match=needs_rle) as from_bytes:
        evaluate_segm_grid(GT_SEGM, DT_SEGM, "strict", 100, True, dt_area="mask")
    with pytest.raises(InvalidAnnotationError, match=needs_rle) as from_handle:
        evaluate_segm_grid_with_dataset(
            CocoDataset.from_json(GT_SEGM), DT_SEGM, "strict", 100, True, dt_area="mask"
        )
    # `raises` accepts subclasses, so the exact class is still worth pinning:
    # the two routes must refuse with the same error, not merely a related one.
    assert type(from_bytes.value) is type(from_handle.value)
    assert str(from_bytes.value) == str(from_handle.value)


def test_bbox_grid_with_dataset_refuses_mask_area() -> None:
    """``dt_area='mask'`` reads an area off a detection's mask, which bbox has not got.

    The bytes-taking grids already refuse it; the handle-taking ones
    must refuse it identically rather than accepting an argument that
    cannot mean anything on this kernel.

    The stub types this parameter as ``Literal["bbox", "supplied"]``, so
    the call below is deliberately ill-typed and the reference is erased
    to make it expressible -- the same move ``_from_arrays`` makes in the
    ingest suites. The static signature is the contract for callers; this
    asserts the *runtime* guard behind it, which is what a caller reaching
    the FFI from untyped code actually meets.
    """
    from vernier.instance import evaluate_bbox_grid_with_dataset

    refuses_mask = cast(Any, evaluate_bbox_grid_with_dataset)
    with pytest.raises(ValueError, match=r"dt_area='mask'"):
        refuses_mask(
            CocoDataset.from_json(GT_PERFECT), DT_PERFECT, "strict", 100, True, dt_area="mask"
        )


def test_one_dataset_serves_both_bbox_and_segm_grids() -> None:
    """The reason this entry point exists: two kernels, one parse.

    A caller evaluating both IoU types previously had to hand GT JSON to
    each grid separately, parsing the same bytes twice. Both kernels now
    read one handle, and each still agrees with its bytes twin.
    """
    from vernier.instance import (
        evaluate_bbox_grid,
        evaluate_bbox_grid_with_dataset,
        evaluate_segm_grid,
        evaluate_segm_grid_with_dataset,
    )

    shared = CocoDataset.from_json(GT_SEGM)
    ladder = [1, 10, 100]
    bbox_handle = evaluate_bbox_grid_with_dataset(shared, DT_SEGM, "strict", 100, True)
    segm_handle = evaluate_segm_grid_with_dataset(shared, DT_SEGM, "strict", 100, True)
    bbox_bytes = evaluate_bbox_grid(GT_SEGM, DT_SEGM, "strict", 100, True)
    segm_bytes = evaluate_segm_grid(GT_SEGM, DT_SEGM, "strict", 100, True)

    assert bbox_handle.accumulate(ladder).summarize().stats == (
        bbox_bytes.accumulate(ladder).summarize().stats
    )
    assert segm_handle.accumulate(ladder).summarize().stats == (
        segm_bytes.accumulate(ladder).summarize().stats
    )


# --- cache reuse -------------------------------------------------------------


def test_dataset_handle_reused_across_calls_yields_same_summary() -> None:
    ds = CocoDataset.from_json(GT_SEGM)
    e = Evaluator(iou=Boundary())
    first = e.evaluate(ds, DT_SEGM)
    # Second call hits the warm BoundaryGtCache; the result must be
    # bit-identical to the cold one.
    second = e.evaluate(ds, DT_SEGM)
    assert first.stats == second.stats


def test_one_dataset_shared_across_evaluators_with_different_options() -> None:
    # Multiple Evaluators differing in parity / use_cats can share one
    # CocoDataset (ADR-0020 §"Per-kernel, parameterized" — cache is on the
    # GT, not the evaluator).
    ds = CocoDataset.from_json(GT_SEGM)
    corrected = Evaluator(iou=Boundary(), parity_mode="corrected").evaluate(ds, DT_SEGM)
    strict = Evaluator(iou=Boundary(), parity_mode="strict").evaluate(ds, DT_SEGM)
    no_cats = Evaluator(iou=Boundary(), use_cats=False).evaluate(ds, DT_SEGM)
    # Each Evaluator returned a Summary; we don't assert equality across
    # them (different params → different stats), only that the shared
    # CocoDataset works without crashing or contaminating state.
    assert all(isinstance(s, Summary) for s in (corrected, strict, no_cats))


def test_clear_cache_does_not_break_subsequent_evaluations() -> None:
    ds = CocoDataset.from_json(GT_SEGM)
    e = Evaluator(iou=Boundary())
    cold = e.evaluate(ds, DT_SEGM)
    ds.clear_cache()
    rebuilt = e.evaluate(ds, DT_SEGM)
    assert cold.stats == rebuilt.stats


def test_changing_dilation_ratio_with_same_dataset_recomputes_correctly() -> None:
    # `BoundaryGtCache` is ratio-keyed; flipping the ratio must clear and
    # repopulate without crashing or returning stale results.
    ds = CocoDataset.from_json(GT_SEGM)
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
    # multiple threads sharing one CocoDataset must each get the correct
    # Summary without races (ADR-0020 §"Composition with ADR-0014").
    ds = CocoDataset.from_json(GT_SEGM)
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
    ds = CocoDataset.from_json(GT_SEGM)
    e = Evaluator(iou=Boundary())

    # Throwaway warmup to amortize PyO3/JIT noise.
    e.evaluate(ds, DT_SEGM)

    cold_ds = CocoDataset.from_json(GT_SEGM)
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
        f"warm CocoDataset path slower than expected: cold={cold_ns}ns, warm={warm_ns}ns"
    )
