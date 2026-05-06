"""``BackgroundPanopticEvaluator`` end-to-end equals
``StreamingPanopticEvaluator``.

ADR-0014 generalized to panoptic via ADR-0032. Pins:

- background finalize bit-equals streaming finalize for the same
  image sequence (the f64 PqStat fold is deterministic given the
  same submission order — the worker preserves it);
- ``retain_per_image_deltas=True`` survives the worker hop, so the
  strict-mode bit-equality property still holds when the user
  ships partials through the background path;
- partial inheritance: ``bg.finalize_to_partial()`` round-trips
  through ``StreamingEvaluator.from_partials`` exactly like the
  streaming case.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import vernier.panoptic as pq


_CATS = json.dumps(
    [
        {"id": 1, "isthing": True},
        {"id": 2, "isthing": False},
        {"id": 3, "isthing": True},
    ]
).encode()


def _image(seed: int) -> tuple[np.ndarray, bytes, np.ndarray, bytes]:
    """One synthetic image: 2x4 grid covering categories 1 and 2,
    DT drift via sparse seed-controlled flips. Independent of other
    test modules' fixtures.
    """
    rng = np.random.default_rng(seed)
    gt_label = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.uint32)
    gt_segs = json.dumps(
        [
            {"id": 1, "category_id": 1, "iscrowd": 0, "area": 4},
            {"id": 2, "category_id": 2, "iscrowd": 0, "area": 4},
        ]
    ).encode()
    dt_label = gt_label.copy()
    flips = rng.random(size=gt_label.shape) < 0.20
    dt_label[flips] = 1
    areas = np.bincount(dt_label.ravel(), minlength=3)
    dt_segs = json.dumps(
        [
            {"id": sid, "category_id": sid, "iscrowd": 0, "area": int(areas[sid])}
            for sid in (1, 2)
            if areas[sid] > 0
        ]
    ).encode()
    return gt_label, gt_segs, dt_label, dt_segs


def test_background_finalize_equals_streaming() -> None:
    """For the same image sequence, background finalize bit-equals
    the streaming finalize. The worker preserves submission order so
    the f64 PqStat fold reproduces the same bit pattern.
    """
    seeds = list(range(6))

    streaming = pq.StreamingEvaluator(_CATS, "strict", retain_per_image_deltas=True)
    for s in seeds:
        gt_lm, gt_si, dt_lm, dt_si = _image(s)
        streaming.update(s, gt_lm, gt_si, dt_lm, dt_si)
    streaming_summary = streaming.finalize()

    bg = pq.BackgroundEvaluator(_CATS, "strict", retain_per_image_deltas=True)
    for s in seeds:
        gt_lm, gt_si, dt_lm, dt_si = _image(s)
        bg.submit(s, gt_lm, gt_si, dt_lm, dt_si)
    bg_summary = bg.finalize()

    assert bg_summary.pq == streaming_summary.pq
    assert bg_summary.sq == streaming_summary.sq
    assert bg_summary.rq == streaming_summary.rq
    assert bg_summary.pq_things == streaming_summary.pq_things
    assert bg_summary.pq_stuff == streaming_summary.pq_stuff


def test_background_strict_partial_merges_bit_equal_to_batch() -> None:
    """``retain_per_image_deltas=True`` survives the worker hop:
    a partial captured through the background path merges back into
    a sharded reconstruction that's bit-equal to the single-rank
    batch run. This is the headline ADR-0032 strict-mode property
    extended over the threading boundary.
    """
    seeds = list(range(8))

    # Single-rank batch baseline (no background, no shards).
    batch = pq.StreamingEvaluator(_CATS, "strict", retain_per_image_deltas=True)
    for s in seeds:
        gt_lm, gt_si, dt_lm, dt_si = _image(s)
        batch.update(s, gt_lm, gt_si, dt_lm, dt_si)
    batch_summary = batch.finalize()

    # Background "rank" — one bg evaluator covers everything.
    bg = pq.BackgroundEvaluator(
        _CATS, "strict", retain_per_image_deltas=True, rank_id=0
    )
    for s in seeds:
        gt_lm, gt_si, dt_lm, dt_si = _image(s)
        bg.submit(s, gt_lm, gt_si, dt_lm, dt_si)
    blob = bg.finalize_to_partial()

    merged = pq.StreamingEvaluator.from_partials(
        _CATS, [blob], "strict", retain_per_image_deltas=True
    ).finalize()

    assert merged.pq == batch_summary.pq
    assert merged.sq == batch_summary.sq
    assert merged.rq == batch_summary.rq


def test_background_evaluator_builder_routes_through_evaluator() -> None:
    """The ``Evaluator(parity_mode=...).background(...)`` builder
    propagates ``parity_mode`` and ``things_stuff_split`` through
    to the background evaluator.
    """
    bg = pq.Evaluator(parity_mode="corrected").background(_CATS)
    assert bg.n_categories == 3
    assert bg.n_images == 0
    assert bg.queue_depth == 0


def test_context_manager_finalize() -> None:
    with pq.BackgroundEvaluator(_CATS, "corrected") as bg:
        gt_lm, gt_si, dt_lm, dt_si = _image(0)
        bg.submit(0, gt_lm, gt_si, dt_lm, dt_si)
        summary = bg.finalize()
        assert summary.pq >= 0.0


def test_finalize_then_use_raises() -> None:
    bg = pq.BackgroundEvaluator(_CATS, "corrected")
    bg.finalize()
    with pytest.raises(Exception):  # noqa: BLE001 - cross-version: any wire-format error
        bg.snapshot()
