"""``BackgroundSemanticEvaluator`` end-to-end equals
``StreamingSemanticEvaluator``.

The background semantic evaluator is a worker-thread wrapper around the
streaming evaluator (ADR-0014 generalized to semantic via ADR-0032);
its output for the same image sequence must bit-equal the streaming
baseline. We pin that here, plus the partial-inheritance property
(``bg.finalize_to_partial()`` round-trips through
``StreamingEvaluator.from_partials`` exactly like the streaming case).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

import vernier.semantic as sem
from vernier._impl import StreamingSemanticEvaluator


def _label_maps(
    seed: int = 0, n_images: int = 4
) -> tuple[Mapping[int, np.ndarray], Mapping[int, np.ndarray]]:
    """Synthetic GT/DT label-map pairs. Same shape as the parity
    distributed-merge harness, kept independent so this test stands
    alone.
    """
    rng = np.random.default_rng(seed)
    gt_maps: dict[int, np.ndarray] = {}
    dt_maps: dict[int, np.ndarray] = {}
    for i in range(n_images):
        h, w = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        gt = rng.integers(0, 3, size=(h, w), dtype=np.uint32)
        flip = rng.random(size=(h, w)) < 0.30
        dt = gt.copy()
        dt[flip] = (gt[flip] + 1) % 3
        gt_maps[i] = gt
        dt_maps[i] = dt
    return gt_maps, dt_maps


def test_background_finalize_equals_streaming() -> None:
    """For the same image sequence, background finalize bit-equals the
    streaming finalize. Confusion-matrix sums are u64-additive — no
    FP wobble to worry about.
    """
    gt_maps, dt_maps = _label_maps(seed=42, n_images=6)
    n_classes = 3

    streaming = StreamingSemanticEvaluator(n_classes, "strict")
    for image_id in sorted(gt_maps):
        streaming.update(image_id, gt_maps[image_id], dt_maps[image_id])
    streaming_summary = streaming.finalize()

    bg = sem.BackgroundEvaluator(n_classes, "strict")
    for image_id in sorted(gt_maps):
        bg.submit(image_id, gt_maps[image_id], dt_maps[image_id])
    bg_summary = bg.finalize()

    np.testing.assert_array_equal(
        bg_summary.confusion_matrix.counts(),
        streaming_summary.confusion_matrix.counts(),
    )
    assert bg_summary.miou == streaming_summary.miou
    assert bg_summary.fwiou == streaming_summary.fwiou
    assert bg_summary.pixel_accuracy == streaming_summary.pixel_accuracy


def test_background_to_partial_round_trips_through_from_partials() -> None:
    """``bg.finalize_to_partial()`` ships a wire-format byte string
    that ``StreamingEvaluator.from_partials`` reconstructs into an
    evaluator equivalent to the original. Pins the ADR-0032 partial-
    inheritance property at the background layer.
    """
    gt_maps, dt_maps = _label_maps(seed=7, n_images=4)
    n_classes = 3

    bg = sem.BackgroundEvaluator(n_classes, "strict", rank_id=0)
    for image_id in sorted(gt_maps):
        bg.submit(image_id, gt_maps[image_id], dt_maps[image_id])
    blob = bg.finalize_to_partial()

    restored = StreamingSemanticEvaluator.from_partials(n_classes, [blob], "strict")
    restored_summary = restored.finalize()

    streaming = StreamingSemanticEvaluator(n_classes, "strict")
    for image_id in sorted(gt_maps):
        streaming.update(image_id, gt_maps[image_id], dt_maps[image_id])
    direct_summary = streaming.finalize()

    np.testing.assert_array_equal(
        restored_summary.confusion_matrix.counts(),
        direct_summary.confusion_matrix.counts(),
    )


def test_background_evaluator_builder_routes_through_evaluator() -> None:
    """The ``Evaluator(parity_mode=...).background(...)`` builder
    propagates ``parity_mode`` through to the background evaluator.
    """
    bg = sem.Evaluator(parity_mode="strict").background(n_classes=3)
    assert bg.n_classes == 3
    assert bg.n_images == 0
    assert bg.queue_depth == 0


def test_context_manager_finalize() -> None:
    """The ``with`` statement provides cooperative shutdown; explicit
    ``finalize`` inside the block is the canonical pattern.
    """
    gt_maps, dt_maps = _label_maps(seed=1, n_images=2)
    n_classes = 3
    with sem.BackgroundEvaluator(n_classes, "corrected") as bg:
        for image_id in sorted(gt_maps):
            bg.submit(image_id, gt_maps[image_id], dt_maps[image_id])
        summary = bg.finalize()
        assert summary.miou >= 0.0


def test_finalize_then_use_raises() -> None:
    """After ``finalize()`` the evaluator is in a finalized state;
    further calls raise.
    """
    bg = sem.BackgroundEvaluator(3, "corrected")
    bg.finalize()
    with pytest.raises(Exception, match="already been finalized"):
        bg.finalize()
