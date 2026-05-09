"""Public-surface DDP tests for ``Evaluator.evaluate_to_partial`` /
``Evaluator.from_partials`` (ADR-0035).

The streaming-evaluator-hosted versions of these methods are exercised
extensively in the existing parity suites under
``tests/python/parity/streaming/{,semantic,panoptic}/test_distributed_merge.py``.
This module pins the *public* entry points: the methods now living on
``vernier.{instance,panoptic,semantic}.Evaluator``. The contract is that
sharding inputs across N ranks via ``evaluate_to_partial`` and merging
the gathered partials with ``from_partials`` produces the same Summary
as a single-rank batch ``evaluate`` over the union (under the
paradigm's strict-mode determinism rules from ADR-0032).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import vernier.instance as inst
import vernier.panoptic as pq
import vernier.semantic as sem

# ---------------------------------------------------------------------------
# Instance: corrected-mode shard-and-merge stays within 4-ULP of batch
# (strict-mode rank-order tiebreak is pending per ADR-0013, same as the
# streaming-class harness).
# ---------------------------------------------------------------------------


_INSTANCE_GT = json.dumps(
    {
        "images": [
            {"id": 1, "width": 100, "height": 100},
            {"id": 2, "width": 100, "height": 100},
            {"id": 3, "width": 100, "height": 100},
            {"id": 4, "width": 100, "height": 100},
        ],
        "annotations": [
            {
                "id": i,
                "image_id": i,
                "category_id": 1,
                "bbox": [10.0, 10.0, 20.0, 20.0],
                "area": 400.0,
                "iscrowd": 0,
            }
            for i in (1, 2, 3, 4)
        ],
        "categories": [{"id": 1, "name": "thing"}],
    }
).encode()


def _instance_dt_for(image_ids: list[int]) -> bytes:
    return json.dumps(
        [
            {"image_id": i, "category_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.9}
            for i in image_ids
        ]
    ).encode()


def test_instance_evaluate_to_partial_round_trips_to_batch() -> None:
    ev = inst.Evaluator(iou=inst.Bbox(), parity_mode="corrected")
    batch = ev.evaluate(_INSTANCE_GT, _instance_dt_for([1, 2, 3, 4]))

    p_a = ev.evaluate_to_partial(_INSTANCE_GT, _instance_dt_for([1, 2]), rank_id=0)
    p_b = ev.evaluate_to_partial(_INSTANCE_GT, _instance_dt_for([3, 4]), rank_id=1)
    merged = inst.Evaluator.from_partials(
        _INSTANCE_GT, [p_a, p_b], iou=inst.Bbox(), parity_mode="corrected"
    )

    assert merged.stats == batch.stats


# ---------------------------------------------------------------------------
# Semantic: strict-mode merge is unconditionally bit-equal (u64-additive
# confusion-matrix sums). ADR-0032 PR-D.
# ---------------------------------------------------------------------------


def _semantic_pair() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    rng = np.random.default_rng(42)
    gt_maps: dict[int, np.ndarray] = {}
    dt_maps: dict[int, np.ndarray] = {}
    for image_id in range(8):
        h, w = 4, 4
        gt = rng.integers(0, 3, size=(h, w), dtype=np.uint32)
        flip_mask = rng.random(size=(h, w)) < 0.30
        dt = gt.copy()
        dt[flip_mask] = (gt[flip_mask] + 1) % 3
        gt_maps[image_id] = gt
        dt_maps[image_id] = dt
    return gt_maps, dt_maps


def _split_dataset(
    gt_maps: dict[int, np.ndarray], dt_maps: dict[int, np.ndarray], n_ranks: int
) -> list[tuple[sem.Dataset, sem.Predictions]]:
    image_ids = sorted(gt_maps.keys())
    shards: list[tuple[sem.Dataset, sem.Predictions]] = []
    for rank in range(n_ranks):
        rank_ids = image_ids[rank::n_ranks]
        rank_gt = sem.Dataset.from_arrays({iid: gt_maps[iid] for iid in rank_ids}, n_classes=3)
        rank_dt = sem.Predictions.from_arrays({iid: dt_maps[iid] for iid in rank_ids})
        shards.append((rank_gt, rank_dt))
    return shards


@pytest.mark.parametrize("n_ranks", [2, 4])
def test_semantic_evaluate_to_partial_strict_bit_equals_batch(n_ranks: int) -> None:
    gt_maps, dt_maps = _semantic_pair()
    ev = sem.Evaluator(parity_mode="strict")
    batch = ev.evaluate(
        sem.Dataset.from_arrays(gt_maps, n_classes=3),
        sem.Predictions.from_arrays(dt_maps),
    )

    shards = _split_dataset(gt_maps, dt_maps, n_ranks)
    partials = [
        ev.evaluate_to_partial(rank_gt, rank_dt, rank_id=rank)
        for rank, (rank_gt, rank_dt) in enumerate(shards)
    ]
    merged = sem.Evaluator.from_partials(3, partials, parity_mode="strict")

    # Strict-mode is unconditionally bit-equal for semantic.
    assert merged.miou == batch.miou
    assert merged.fwiou == batch.fwiou
    assert merged.pixel_accuracy == batch.pixel_accuracy
    assert merged.mean_accuracy == batch.mean_accuracy


# ---------------------------------------------------------------------------
# Panoptic: strict-mode merge is bit-equal opt-in via
# ``retain_per_image_deltas=True`` on every rank (ADR-0032 PR-E).
# ---------------------------------------------------------------------------


_PANOPTIC_CATS = json.dumps(
    [
        {"id": 1, "isthing": True},
        {"id": 2, "isthing": False},
    ]
).encode()


def _segs(triples: list[tuple[int, int, int]]) -> bytes:
    return json.dumps(
        [{"id": sid, "category_id": cid, "iscrowd": 0, "area": a} for sid, cid, a in triples]
    ).encode()


def _panoptic_image_pair(seed: int) -> tuple[int, np.ndarray, bytes, np.ndarray, bytes]:
    rng = np.random.default_rng(seed)
    gt_lm = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.uint32)
    gt_si = _segs([(1, 1, 4), (2, 2, 4)])
    dt_lm = gt_lm.copy()
    flips = rng.random(size=gt_lm.shape) < 0.15
    dt_lm[flips] = 1
    dt_areas = np.bincount(dt_lm.ravel(), minlength=3)
    dt_si = _segs([(1, 1, int(dt_areas[1])), (2, 2, int(dt_areas[2]))])
    return seed, gt_lm, gt_si, dt_lm, dt_si


def test_panoptic_evaluate_to_partial_strict_bit_equals_streaming() -> None:
    """With ``retain_per_image_deltas=True``, the merged Summary is
    bit-equal to a single-rank streaming finalize over the union.
    Pinned against the streaming substrate (not batch) because batch
    panoptic uses a different code path and its f64 ordering doesn't
    match streaming-merge ordering bit-equally — that's expected and
    documented in ADR-0032.
    """
    from vernier._impl import StreamingPanopticEvaluator

    seeds = list(range(8))
    images = [_panoptic_image_pair(s) for s in seeds]

    baseline = StreamingPanopticEvaluator(_PANOPTIC_CATS, "strict", retain_per_image_deltas=True)
    for image in images:
        baseline.update(*image)
    baseline_summary = baseline.finalize()

    ev = pq.Evaluator(parity_mode="strict")
    shard_a = images[::2]
    shard_b = images[1::2]
    p_a = ev.evaluate_to_partial(
        shard_a, categories=_PANOPTIC_CATS, rank_id=0, retain_per_image_deltas=True
    )
    p_b = ev.evaluate_to_partial(
        shard_b, categories=_PANOPTIC_CATS, rank_id=1, retain_per_image_deltas=True
    )
    merged = pq.Evaluator.from_partials(
        _PANOPTIC_CATS,
        [p_a, p_b],
        parity_mode="strict",
        retain_per_image_deltas=True,
    )

    assert merged.pq == baseline_summary.pq
    assert merged.sq == baseline_summary.sq
    assert merged.rq == baseline_summary.rq
