"""ADR-0032 PR-F: pin the strict-mode bit-equality contract.

The headline determinism property of ADR-0032 is paradigm-specific:

- **Semantic** — strict-mode merge is **unconditionally** bit-equal
  to a batch run over the union. Confusion-matrix sums are u64
  integer-additive. No ``(score, rank_id, local_position)`` tiebreak
  needed.
- **Panoptic** — strict-mode merge is **opt-in** bit-equal via
  ``retain_per_image_deltas=True`` on every rank. The merge
  accumulator re-sorts per-image PqStat deltas by ``image_id`` and
  re-sums in batch order, recovering bit-equality despite f64
  non-associativity.
- **Instance** — still pending the ADR-0013 ``(score, rank_id,
  local_position)`` tiebreak; covered by ``pytest.skip`` in the
  instance distributed-merge test and out of scope here.

This module is the no-skip pin. If either property regresses, this
test fails — which is exactly what the determinism contract table
in ADR-0032 §"Decision drivers" promises a CI gate user.
"""

from __future__ import annotations

import functools
import json

import numpy as np
import pytest

import vernier.semantic as sem
from vernier._impl import StreamingPanopticEvaluator, StreamingSemanticEvaluator

# ---------------------------------------------------------------------------
# Semantic: strict-mode bit-equality is unconditional.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_ranks", [2, 4, 8])
def test_semantic_strict_merge_bit_equals_batch(n_ranks: int) -> None:
    """Confusion-matrix counts and all derived metrics (mIoU / FWIoU /
    pixel accuracy) are bit-equal under sharded merge with
    ``parity_mode="strict"``. No skip, no caveat.
    """
    rng = np.random.default_rng(2026)
    n_classes = 4
    n_images = 16

    gt_maps: dict[int, np.ndarray] = {}
    dt_maps: dict[int, np.ndarray] = {}
    for i in range(n_images):
        h, w = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        gt = rng.integers(0, n_classes, size=(h, w), dtype=np.uint32)
        flip = rng.random(size=(h, w)) < 0.25
        dt = gt.copy()
        dt[flip] = (gt[flip] + 1) % n_classes
        gt_maps[i] = gt
        dt_maps[i] = dt

    batch = sem.Evaluator(parity_mode="strict").evaluate(
        sem.Dataset.from_arrays(gt_maps, n_classes=n_classes),
        sem.Predictions.from_arrays(dt_maps),
    )

    partials: list[bytes] = []
    for rank in range(n_ranks):
        ev = StreamingSemanticEvaluator(n_classes, "strict", rank_id=rank)
        for image_id in sorted(gt_maps):
            if image_id % n_ranks == rank:
                ev.update(image_id, gt_maps[image_id], dt_maps[image_id])
        partials.append(ev.finalize_to_partial())

    merged = StreamingSemanticEvaluator.from_partials(n_classes, partials, "strict").finalize()

    # u64 confusion matrix → bit-equal element-wise.
    np.testing.assert_array_equal(
        merged.confusion_matrix.counts(),
        batch.confusion_matrix.counts(),
    )
    # Derived f64 metrics inherit bit-equality from integer counts.
    assert merged.miou == batch.miou
    assert merged.fwiou == batch.fwiou
    assert merged.pixel_accuracy == batch.pixel_accuracy
    assert merged.mean_accuracy == batch.mean_accuracy


# ---------------------------------------------------------------------------
# Panoptic: strict-mode bit-equality is opt-in via per-image deltas.
# ---------------------------------------------------------------------------


_PANOPTIC_CATS = json.dumps(
    [
        {"id": 1, "isthing": True},
        {"id": 2, "isthing": False},
        {"id": 3, "isthing": True},
    ]
).encode()


@functools.cache
def _panoptic_image(seed: int) -> tuple[np.ndarray, bytes, np.ndarray, bytes]:
    """Build one (gt_label_map, gt_segs, dt_label_map, dt_segs) pair
    keyed by ``seed`` — same shape as the panoptic distributed-merge
    fixture, kept independent so this test stands alone.
    """
    rng = np.random.default_rng(seed)
    gt_label = np.array([[1, 1, 2, 2, 3], [1, 1, 2, 2, 3]], dtype=np.uint32)
    gt_segs = json.dumps(
        [
            {"id": 1, "category_id": 1, "iscrowd": 0, "area": 4},
            {"id": 2, "category_id": 2, "iscrowd": 0, "area": 4},
            {"id": 3, "category_id": 3, "iscrowd": 0, "area": 2},
        ]
    ).encode()
    dt_label = gt_label.copy()
    flips = rng.random(size=gt_label.shape) < 0.20
    dt_label[flips] = 1
    areas = np.bincount(dt_label.ravel(), minlength=4)
    cats = {1: 1, 2: 2, 3: 3}
    dt_segs = json.dumps(
        [
            {"id": sid, "category_id": cats[sid], "iscrowd": 0, "area": int(areas[sid])}
            for sid in (1, 2, 3)
            if areas[sid] > 0
        ]
    ).encode()
    return gt_label, gt_segs, dt_label, dt_segs


@pytest.mark.parametrize("n_ranks", [2, 4, 8])
def test_panoptic_strict_merge_bit_equals_batch_with_deltas(
    n_ranks: int,
) -> None:
    """With ``retain_per_image_deltas=True`` on every rank, panoptic
    strict-mode merge is bit-equal to a batch run over the union.
    Re-sorting deltas by ``image_id`` re-establishes the batch summation
    order, recovering bit-equality despite f64 non-associativity.
    """
    seeds = list(range(16))

    # Single-rank baseline (the "batch" run).
    baseline = StreamingPanopticEvaluator(_PANOPTIC_CATS, "strict", retain_per_image_deltas=True)
    for s in seeds:
        gt_lm, gt_si, dt_lm, dt_si = _panoptic_image(s)
        baseline.update(s, gt_lm, gt_si, dt_lm, dt_si)
    batch = baseline.finalize()

    # Sharded merge with deltas retained on every rank.
    partials: list[bytes] = []
    for rank in range(n_ranks):
        ev = StreamingPanopticEvaluator(
            _PANOPTIC_CATS,
            "strict",
            retain_per_image_deltas=True,
            rank_id=rank,
        )
        for s in seeds:
            if s % n_ranks == rank:
                gt_lm, gt_si, dt_lm, dt_si = _panoptic_image(s)
                ev.update(s, gt_lm, gt_si, dt_lm, dt_si)
        partials.append(ev.finalize_to_partial())

    merged = StreamingPanopticEvaluator.from_partials(
        _PANOPTIC_CATS,
        partials,
        "strict",
        retain_per_image_deltas=True,
    ).finalize()

    # The headline trio: PQ / SQ / RQ all bit-equal under reorder.
    assert merged.pq == batch.pq
    assert merged.sq == batch.sq
    assert merged.rq == batch.rq
    # Things / stuff buckets are independent unweighted means; pin
    # them too since they ride on the same per-image fold.
    assert merged.pq_things == batch.pq_things
    assert merged.pq_stuff == batch.pq_stuff
