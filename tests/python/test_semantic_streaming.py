"""Streaming-evaluator tests for the ``vernier.semantic`` substrate.

Per ADR-0035, the streaming class lives at ``vernier._impl`` (not on the
public ``vernier.semantic`` namespace). This file pins the FFI-level
contract — that the streaming substrate computes per-image confusion
matrices correctly and that streaming-``finalize()`` bit-equals the
batch-``Evaluator.evaluate(...)`` over the same images. That equality is
what lets ``vernier.semantic.Evaluator.evaluate_to_partial`` /
``from_partials`` ride on the same kernel without divergence.
"""

from __future__ import annotations

import numpy as np
import pytest

from vernier._impl import StreamingSemanticEvaluator as StreamingEvaluator
from vernier.semantic import (
    Dataset,
    Evaluator,
    Predictions,
    Summary,
)


def _toy_pair(
    n_classes: int = 3,
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], Dataset, Predictions]:
    """Produce both a streaming sequence and a batch (Dataset, Predictions)
    pair carrying the same per-image content. Lets the batch-vs-stream
    parity test exercise both paths against identical inputs."""
    images = [
        (1, np.array([[0, 1, 2]], dtype=np.uint32), np.array([[0, 1, 2]], dtype=np.uint32)),
        (2, np.array([[0, 0, 1, 2]], dtype=np.uint32), np.array([[0, 1, 1, 2]], dtype=np.uint32)),
        (3, np.array([[2, 2]], dtype=np.uint32), np.array([[2, 0]], dtype=np.uint32)),
    ]
    gt_maps = {iid: gt for iid, gt, _ in images}
    dt_maps = {iid: dt for iid, _, dt in images}
    gt = Dataset.from_arrays(gt_maps, n_classes=n_classes)
    dt = Predictions.from_arrays(dt_maps)
    return images, gt, dt


def test_stream_constructs() -> None:
    ev = StreamingEvaluator(3, "corrected")
    assert isinstance(ev, StreamingEvaluator)
    assert ev.n_classes == 3
    assert ev.n_images == 0


def test_streaming_finalize_bit_equals_batch_evaluate() -> None:
    """Load-bearing invariant: streaming finalize and batch evaluate
    produce the same numbers on the same images."""
    images, gt, dt = _toy_pair()

    batch_summary = Evaluator(parity_mode="strict").evaluate(gt, dt)

    stream_ev = StreamingEvaluator(3, "strict")
    for image_id, gt_arr, dt_arr in images:
        stream_ev.update(image_id, gt_arr, dt_arr)
    assert stream_ev.n_images == len(images)
    stream_summary = stream_ev.finalize()

    assert isinstance(stream_summary, Summary)
    # f64 bit-equality on the four headline scalars.
    assert stream_summary.miou == batch_summary.miou
    assert stream_summary.fwiou == batch_summary.fwiou
    assert stream_summary.pixel_accuracy == batch_summary.pixel_accuracy
    assert stream_summary.mean_accuracy == batch_summary.mean_accuracy


def test_streaming_snapshot_does_not_consume() -> None:
    ev = StreamingEvaluator(2, "corrected")
    ev.update(1, np.array([[0, 1]], dtype=np.uint32), np.array([[0, 1]], dtype=np.uint32))
    snap = ev.snapshot()
    assert snap.miou == pytest.approx(1.0)
    # Evaluator is still usable after snapshot.
    ev.update(2, np.array([[0]], dtype=np.uint32), np.array([[0]], dtype=np.uint32))
    assert ev.n_images == 2


def test_streaming_with_ignore_label() -> None:
    ev = StreamingEvaluator(2, "corrected", ignore_label=255)
    ev.update(
        1,
        np.array([[0, 255, 1, 1]], dtype=np.uint32),
        np.array([[0, 0, 1, 1]], dtype=np.uint32),
    )
    summary = ev.finalize()
    # 3 non-ignore pixels, all diagonal → mIoU = 1.0.
    assert summary.miou == pytest.approx(1.0)
    assert summary.confusion_matrix.total == 3


def test_streaming_shape_mismatch_raises() -> None:
    ev = StreamingEvaluator(2, "corrected")
    gt = np.array([[0, 1]], dtype=np.uint32)
    dt = np.array([[0, 1, 0]], dtype=np.uint32)
    with pytest.raises(ValueError, match="shape mismatch"):
        ev.update(7, gt, dt)
    # Failed update does not advance n_images.
    assert ev.n_images == 0


def test_streaming_n_classes_zero_rejected() -> None:
    with pytest.raises(ValueError, match="n_classes"):
        StreamingEvaluator(0, "corrected")


def test_streaming_unknown_parity_mode_rejected() -> None:
    with pytest.raises(ValueError, match="parity_mode"):
        StreamingEvaluator(3, "aligned")  # type: ignore[arg-type]


def test_streaming_finalize_resets_state() -> None:
    # finalize() consumes the inner state but leaves the pyobject
    # usable in a reset shape. snapshot() on the post-finalize state
    # returns zeros (defensive — empty confusion matrix → 0.0 means).
    ev = StreamingEvaluator(3, "corrected")
    ev.update(1, np.array([[0, 1]], dtype=np.uint32), np.array([[0, 1]], dtype=np.uint32))
    _ = ev.finalize()
    # Post-finalize: n_images is reset to 0, confusion is empty.
    assert ev.n_images == 0
    snap_after = ev.snapshot()
    assert snap_after.miou == 0.0


def test_streaming_repr_carries_progress() -> None:
    ev = StreamingEvaluator(5, "corrected")
    assert "StreamingSemanticEvaluator" in repr(ev)
    assert "n_classes=5" in repr(ev)
    assert "n_images=0" in repr(ev)
    ev.update(1, np.zeros((2, 2), dtype=np.uint32), np.zeros((2, 2), dtype=np.uint32))
    assert "n_images=1" in repr(ev)
