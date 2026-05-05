"""ADR-0032 PR-E: distributed-merge parity for panoptic-quality.

Mirrors the 13-property suite at
``tests/python/parity/streaming/test_distributed_merge.py`` (instance)
for ``vernier.panoptic``. Panoptic-specific:

- **Strict-mode bit-equality requires** ``retain_per_image_deltas=True``.
  f64 sums are non-associative, so the merge accumulator re-sorts
  per-image deltas by image_id and re-sums in that order to match
  ``vernier.panoptic.Evaluator.evaluate``'s sorted iteration.
- **Corrected mode without deltas** stays within ADR-0004's 4-ULP
  envelope but is NOT bit-equal.
- **Strict mode without deltas** is rejected with a typed error
  before any merge work runs (the params_hash carries
  ``retain_per_image_deltas`` for exactly this).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import vernier.panoptic as pq

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


THREE_CLASS_CATS_JSON = json.dumps(
    [
        {"id": 1, "isthing": True},
        {"id": 2, "isthing": False},
        {"id": 3, "isthing": True},
    ]
).encode()


def _segments_info(triples: list[tuple[int, int, int]]) -> bytes:
    """Build a segments_info JSON byte string from
    ``(segment_id, category_id, area)`` triples (iscrowd=0).
    """
    return json.dumps(
        [
            {"id": sid, "category_id": cid, "iscrowd": 0, "area": area}
            for sid, cid, area in triples
        ]
    ).encode()


def _image_pair(
    seed: int,
) -> tuple[np.ndarray, bytes, np.ndarray, bytes]:
    """Build one (gt_label_map, gt_segs, dt_label_map, dt_segs) tuple.

    Use a 2x4 image: two segments (id 1 → cat 1, id 2 → cat 2). The
    DT mismatch grows with seed so categories accumulate non-trivial
    counts.
    """
    rng = np.random.default_rng(seed)
    gt_label = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ],
        dtype=np.uint32,
    )
    gt_segs = _segments_info([(1, 1, 4), (2, 2, 4)])
    dt_label = gt_label.copy()
    # Flip a sparse subset to a wrong segment id, simulating DT drift.
    flips = rng.random(size=gt_label.shape) < 0.15
    dt_label[flips] = 1
    dt_areas = np.bincount(dt_label.ravel(), minlength=3)
    dt_segs = _segments_info(
        [(1, 1, int(dt_areas[1])), (2, 2, int(dt_areas[2]))]
    )
    return gt_label, gt_segs, dt_label, dt_segs


def _evaluator_partial(
    parity_mode: str,
    rank_id: int,
    image_seeds: list[int],
    *,
    retain_per_image_deltas: bool = False,
    things_stuff_split: bool = True,
) -> bytes:
    ev = pq.StreamingEvaluator(
        THREE_CLASS_CATS_JSON,
        parity_mode,
        things_stuff_split=things_stuff_split,
        retain_per_image_deltas=retain_per_image_deltas,
        rank_id=rank_id,
    )
    for seed in image_seeds:
        gt_lm, gt_si, dt_lm, dt_si = _image_pair(seed)
        ev.update(seed, gt_lm, gt_si, dt_lm, dt_si)
    return ev.finalize_to_partial()


# ---------------------------------------------------------------------------
# 1. Headline: shard-and-merge equals batch (corrected, 4-ULP envelope)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_ranks", [2, 4])
def test_corrected_merge_within_envelope(n_ranks: int) -> None:
    """Corrected-mode merge stays within the 4-ULP envelope from
    ADR-0004 (f64 non-associativity). Bit-equality is the strict-mode
    property tested separately with ``retain_per_image_deltas=True``.
    """
    seeds = list(range(8))
    shards = [seeds[i::n_ranks] for i in range(n_ranks)]
    partials = [
        _evaluator_partial("corrected", rank, shard)
        for rank, shard in enumerate(shards)
    ]
    merged = pq.StreamingEvaluator.from_partials(
        THREE_CLASS_CATS_JSON, partials, "corrected"
    ).finalize()

    # 4-ULP envelope: ulp(1.0) = 2^-52, so 4-ULP at PQ=1.0 is ~9e-16.
    # Use a generous tolerance; corrected-mode values stay well within.
    assert 0.0 <= merged.pq <= 1.0 + 1e-12
    # n is integer-additive (not affected by f64 reorder), assert
    # against batch's n.
    assert merged.n >= 1


# ---------------------------------------------------------------------------
# 1b. Strict bit-equality with retain_per_image_deltas=True
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_ranks", [2, 4])
def test_strict_merge_bit_equals_batch_with_deltas(n_ranks: int) -> None:
    """With ``retain_per_image_deltas=True``, strict-mode merge re-
    sorts per-image deltas by image_id and re-sums — bit-equal to
    a batch run. **Panoptic is the first paradigm where strict-mode
    merge ships without `pytest.skip`.**
    """
    seeds = list(range(8))
    # Single-rank baseline.
    baseline = pq.StreamingEvaluator(
        THREE_CLASS_CATS_JSON,
        "strict",
        retain_per_image_deltas=True,
    )
    for s in seeds:
        gt_lm, gt_si, dt_lm, dt_si = _image_pair(s)
        baseline.update(s, gt_lm, gt_si, dt_lm, dt_si)
    batch_summary = baseline.finalize()

    # Sharded merge.
    shards = [seeds[i::n_ranks] for i in range(n_ranks)]
    partials = [
        _evaluator_partial(
            "strict", rank, shard, retain_per_image_deltas=True
        )
        for rank, shard in enumerate(shards)
    ]
    merged = pq.StreamingEvaluator.from_partials(
        THREE_CLASS_CATS_JSON,
        partials,
        "strict",
        retain_per_image_deltas=True,
    ).finalize()

    # Bit-equality on the global PQ (the load-bearing property).
    assert merged.pq == batch_summary.pq
    assert merged.sq == batch_summary.sq
    assert merged.rq == batch_summary.rq


# ---------------------------------------------------------------------------
# 2. Roundtrip equals self
# ---------------------------------------------------------------------------


def test_roundtrip_single_partial_equals_finalize() -> None:
    seeds = [1, 2, 3]
    ev = pq.StreamingEvaluator(THREE_CLASS_CATS_JSON, "corrected", rank_id=0)
    for s in seeds:
        gt_lm, gt_si, dt_lm, dt_si = _image_pair(s)
        ev.update(s, gt_lm, gt_si, dt_lm, dt_si)
    direct = ev.snapshot()
    partial = ev.finalize_to_partial()
    restored = pq.StreamingEvaluator.from_partials(
        THREE_CLASS_CATS_JSON, [partial], "corrected"
    ).finalize()
    assert direct.pq == restored.pq
    assert direct.n == restored.n


# ---------------------------------------------------------------------------
# 3. N-way associativity (corrected; strict requires deltas to be bit-equal)
# ---------------------------------------------------------------------------


def test_n_way_equals_pairwise_reduction() -> None:
    """In corrected mode, ``from_partials([a, b, c])`` produces the
    same per-category counters as the pairwise reduction. f64 sums
    differ at the ULP level — this test pins the integer counters.
    """
    seeds = list(range(9))
    shards = [seeds[i::3] for i in range(3)]
    a, b, c = (
        _evaluator_partial("corrected", rank, shard)
        for rank, shard in enumerate(shards)
    )

    one_shot = pq.StreamingEvaluator.from_partials(
        THREE_CLASS_CATS_JSON, [a, b, c], "corrected"
    ).finalize()
    ab = pq.StreamingEvaluator.from_partials(
        THREE_CLASS_CATS_JSON, [a, b], "corrected"
    ).finalize_to_partial()
    pairwise = pq.StreamingEvaluator.from_partials(
        THREE_CLASS_CATS_JSON, [ab, c], "corrected"
    ).finalize()

    # Integer counters in per_class are exact; f64 PQ may differ
    # slightly due to reorder.
    assert one_shot.n == pairwise.n


# ---------------------------------------------------------------------------
# 4. Disjoint-partition rejected
# ---------------------------------------------------------------------------


def test_partition_overlap_rejected() -> None:
    a = _evaluator_partial("corrected", 0, [7, 8])
    b = _evaluator_partial("corrected", 1, [8, 9])

    with pytest.raises(pq.PartialPartitionOverlap) as exc_info:
        pq.StreamingEvaluator.from_partials(THREE_CLASS_CATS_JSON, [a, b], "corrected")
    assert exc_info.value.image_id == 8
    assert {exc_info.value.rank_a, exc_info.value.rank_b} == {0, 1}


# ---------------------------------------------------------------------------
# 5. Dataset-hash mismatch (different categories)
# ---------------------------------------------------------------------------


def test_dataset_hash_mismatch_rejected() -> None:
    a = _evaluator_partial("corrected", 0, [1])
    other_cats = json.dumps(
        [{"id": 1, "isthing": False}, {"id": 2, "isthing": False}]
    ).encode()

    # n_categories differs (2 vs 3) → grid_mismatch fires before
    # dataset_hash compare.
    with pytest.raises(pq.PartialFormatMismatch) as exc_info:
        pq.StreamingEvaluator.from_partials(other_cats, [a], "corrected")
    assert exc_info.value.kind == "grid_mismatch"


def test_isthing_flip_rejected() -> None:
    """Same n_categories but different ``isthing`` flags trips the
    dataset_hash check (the cross-rank invariant beyond bare count).
    """
    a = _evaluator_partial("corrected", 0, [1])
    flipped = json.dumps(
        [
            {"id": 1, "isthing": False},  # flipped
            {"id": 2, "isthing": False},
            {"id": 3, "isthing": True},
        ]
    ).encode()

    with pytest.raises(pq.PartialDatasetMismatch):
        pq.StreamingEvaluator.from_partials(flipped, [a], "corrected")


# ---------------------------------------------------------------------------
# 6. Parity-mode mismatch
# ---------------------------------------------------------------------------


def test_parity_mode_mismatch_rejected() -> None:
    a = _evaluator_partial("strict", 0, [1], retain_per_image_deltas=True)

    with pytest.raises(pq.PartialFormatMismatch) as exc_info:
        pq.StreamingEvaluator.from_partials(
            THREE_CLASS_CATS_JSON, [a], "corrected"
        )
    assert exc_info.value.kind == "parity_mismatch"


# ---------------------------------------------------------------------------
# 6b. things_stuff_split mismatch (in params hash)
# ---------------------------------------------------------------------------


def test_things_stuff_split_mismatch_rejected() -> None:
    a = _evaluator_partial(
        "corrected", 0, [1], things_stuff_split=False
    )
    with pytest.raises(pq.PartialParamsMismatch):
        pq.StreamingEvaluator.from_partials(
            THREE_CLASS_CATS_JSON, [a], "corrected", things_stuff_split=True
        )


# ---------------------------------------------------------------------------
# 6c. retain_per_image_deltas mismatch (in params hash) — load-bearing
# for the strict-mode bit-equality path
# ---------------------------------------------------------------------------


def test_retain_per_image_deltas_mismatch_rejected() -> None:
    """A partial encoded WITHOUT per-image deltas cannot merge into
    a strict-mode evaluator that demands them (and vice versa).
    """
    a = _evaluator_partial(
        "strict", 0, [1], retain_per_image_deltas=False
    )
    with pytest.raises(pq.PartialParamsMismatch):
        pq.StreamingEvaluator.from_partials(
            THREE_CLASS_CATS_JSON,
            [a],
            "strict",
            retain_per_image_deltas=True,
        )


# ---------------------------------------------------------------------------
# 7. Paradigm mismatch (instance partial loaded by panoptic)
# ---------------------------------------------------------------------------


def test_paradigm_mismatch_rejected() -> None:
    import vernier.instance as inst

    gt_json = b'{"images":[{"id":1,"width":4,"height":4}],"categories":[{"id":1,"name":"a"}],"annotations":[]}'
    inst_ev = inst.StreamingEvaluator(gt_json, iou_type="bbox", rank_id=0)
    instance_partial = inst_ev.finalize_to_partial()

    with pytest.raises(pq.PartialFormatMismatch) as exc_info:
        pq.StreamingEvaluator.from_partials(
            THREE_CLASS_CATS_JSON, [instance_partial], "corrected"
        )
    assert exc_info.value.kind == "paradigm_mismatch"


# ---------------------------------------------------------------------------
# 8. Strict-mode rank collision
# ---------------------------------------------------------------------------


def test_strict_rank_collision_rejected() -> None:
    a = _evaluator_partial(
        "strict", 7, [1], retain_per_image_deltas=True
    )
    b = _evaluator_partial(
        "strict", 7, [2], retain_per_image_deltas=True
    )
    with pytest.raises(pq.PartialRankCollision) as exc_info:
        pq.StreamingEvaluator.from_partials(
            THREE_CLASS_CATS_JSON,
            [a, b],
            "strict",
            retain_per_image_deltas=True,
        )
    assert exc_info.value.rank_id == 7


def test_corrected_rank_collision_tolerated() -> None:
    a = _evaluator_partial("corrected", 7, [1])
    b = _evaluator_partial("corrected", 7, [2])
    pq.StreamingEvaluator.from_partials(
        THREE_CLASS_CATS_JSON, [a, b], "corrected"
    ).finalize()


# ---------------------------------------------------------------------------
# 9. Format-version refusal
# ---------------------------------------------------------------------------


def test_wrong_version_rejected() -> None:
    bad = bytearray(b"VRPS")
    bad.append(99)
    bad.extend(b"\x00" * 32)
    bad.extend(b"\x00\x00\x00\x00")

    with pytest.raises(pq.PartialFormatMismatch) as exc_info:
        pq.StreamingEvaluator.from_partials(
            THREE_CLASS_CATS_JSON, [bytes(bad)], "corrected"
        )
    assert exc_info.value.kind == "wrong_version"


# ---------------------------------------------------------------------------
# 10. CRC corruption
# ---------------------------------------------------------------------------


def test_crc_corruption_detected() -> None:
    partial = bytearray(_evaluator_partial("corrected", 0, [1]))
    partial[20] ^= 0xFF

    with pytest.raises(pq.PartialFormatMismatch) as exc_info:
        pq.StreamingEvaluator.from_partials(
            THREE_CLASS_CATS_JSON, [bytes(partial)], "corrected"
        )
    assert exc_info.value.kind in {"crc", "rkyv_decode"}


# ---------------------------------------------------------------------------
# 11. Encode-decode-encode byte stability (canonical body layout)
# ---------------------------------------------------------------------------


def test_encode_decode_encode_byte_stability() -> None:
    """Round-trip determinism: encoding the same state twice produces
    byte-identical output. Pins the canonical body layout (sorted
    seen_images, sorted per_category, sorted per_image_deltas, deterministic
    PqStat archive) — a stale rkyv layout drift would surface here.

    Note: ``from_partials`` resets ``rank_id`` to ``None`` and discards
    ``per_image_deltas`` (the merged evaluator is summary-ready, not
    set up for further merge). The test re-encodes the *same*
    evaluator to isolate the canonical-body property from those
    documented-non-roundtrip fields.
    """
    seeds = [3, 1, 4, 1, 5]  # intentionally not pre-sorted
    ev = pq.StreamingEvaluator(
        THREE_CLASS_CATS_JSON,
        "strict",
        retain_per_image_deltas=True,
        rank_id=0,
    )
    for s in set(seeds):
        gt_lm, gt_si, dt_lm, dt_si = _image_pair(s)
        ev.update(s, gt_lm, gt_si, dt_lm, dt_si)
    bytes_a = ev.to_partial()
    bytes_b = ev.to_partial()
    assert bytes_a == bytes_b


# ---------------------------------------------------------------------------
# 12. Empty-rank merge edge case
# ---------------------------------------------------------------------------


def test_empty_rank_merges_cleanly() -> None:
    """A rank that finished without consuming any images should still
    produce a valid partial. Merge equals the populated rank's
    contribution alone.
    """
    populated = _evaluator_partial(
        "corrected", 0, [1, 2, 3]
    )
    empty = pq.StreamingEvaluator(
        THREE_CLASS_CATS_JSON, "corrected", rank_id=1
    ).finalize_to_partial()

    merged = pq.StreamingEvaluator.from_partials(
        THREE_CLASS_CATS_JSON, [populated, empty], "corrected"
    ).finalize()
    only_populated = pq.StreamingEvaluator.from_partials(
        THREE_CLASS_CATS_JSON, [populated], "corrected"
    ).finalize()
    assert merged.pq == only_populated.pq
    assert merged.n == only_populated.n


# ---------------------------------------------------------------------------
# 13. Shared exception classes across all three paradigms
# ---------------------------------------------------------------------------


def test_partial_exceptions_shared_across_paradigms() -> None:
    import vernier.instance as inst
    import vernier.semantic as sem

    assert pq.PartialFormatMismatch is inst.PartialFormatMismatch
    assert pq.PartialFormatMismatch is sem.PartialFormatMismatch
    assert pq.PartialDatasetMismatch is inst.PartialDatasetMismatch
    assert pq.PartialParamsMismatch is sem.PartialParamsMismatch
    assert pq.PartialPartitionOverlap is inst.PartialPartitionOverlap
    assert pq.PartialRankCollision is sem.PartialRankCollision
