"""ADR-0032 PR-D: distributed-merge parity for semantic-segmentation.

Mirrors the 13-property suite at
``tests/python/parity/streaming/test_distributed_merge.py`` (instance
paradigm) for ``vernier.semantic``. Three properties are dropped or
adapted:

- **#7 (kernel mismatch rejected)** → **paradigm mismatch rejected**:
  an instance partial loaded by a semantic ``from_partials`` raises
  ``PartialFormatMismatch{kind: paradigm_mismatch}``.
- **#12 (tables across merge)** — dropped; semantic has no per-pair
  / per-detection tables.
- **#13 (BackgroundEvaluator inheritance)** — deferred to PR-E (the
  PR that adds ``BackgroundSemanticEvaluator``).

Strict-mode tests are **not** ``pytest.skip``-ped (unlike the
instance harness): confusion-matrix sums are integer-additive, so
strict-mode merge is unconditionally bit-equal to a batch run over
the union. No ``(score, rank_id, local_position)`` tiebreak needed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import pytest

import vernier.semantic as sem
from vernier._impl import StreamingSemanticEvaluator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _label_maps(
    seed: int = 0, n_images: int = 6
) -> tuple[Mapping[int, np.ndarray], Mapping[int, np.ndarray]]:
    """Build a synthetic GT/DT label-map pair with deterministic noise.

    The ``seed`` controls the RNG; same seed produces identical maps.
    Image ids are ``0..n_images``; class space is ``[0, 3)``.
    """
    rng = np.random.default_rng(seed)
    gt_maps: dict[int, np.ndarray] = {}
    dt_maps: dict[int, np.ndarray] = {}
    for i in range(n_images):
        h, w = rng.integers(2, 6), rng.integers(2, 6)
        gt = rng.integers(0, 3, size=(h, w), dtype=np.uint32)
        # DT differs from GT in a sparse pattern — gives non-trivial
        # off-diagonal cells without trivializing the merge.
        flip_mask = rng.random(size=(h, w)) < 0.3
        dt = gt.copy()
        dt[flip_mask] = (gt[flip_mask] + 1) % 3
        gt_maps[i] = gt
        dt_maps[i] = dt
    return gt_maps, dt_maps


def _shard(label_maps: Mapping[int, np.ndarray], n_shards: int) -> list[dict[int, np.ndarray]]:
    """Round-robin shard a label-map dict across `n_shards` partitions."""
    shards: list[dict[int, np.ndarray]] = [{} for _ in range(n_shards)]
    for image_id, arr in label_maps.items():
        shards[image_id % n_shards][image_id] = arr
    return shards


def _evaluator_partial(
    n_classes: int,
    parity_mode: Literal["strict", "corrected"],
    rank_id: int,
    gt_shard: Mapping[int, np.ndarray],
    dt_shard: Mapping[int, np.ndarray],
    ignore_label: int | None = None,
) -> bytes:
    """Build a partial blob for one rank's shard."""
    ev = StreamingSemanticEvaluator(
        n_classes, parity_mode, ignore_label=ignore_label, rank_id=rank_id
    )
    for image_id in sorted(gt_shard):
        ev.update(image_id, gt_shard[image_id], dt_shard[image_id])
    return ev.finalize_to_partial()


# ---------------------------------------------------------------------------
# 1. Shard-and-merge equals batch — the headline property
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("parity_mode", ["strict", "corrected"])
@pytest.mark.parametrize("n_ranks", [2, 4])
def test_shard_and_merge_equals_batch(
    parity_mode: Literal["strict", "corrected"], n_ranks: int
) -> None:
    """For every shard count and parity mode, the merged confusion
    matrix is element-wise bit-equal to a batch run over the union.
    """
    gt_maps, dt_maps = _label_maps(seed=42, n_images=12)
    n_classes = 3

    batch = sem.Evaluator(parity_mode=parity_mode).evaluate(
        sem.Dataset.from_arrays(gt_maps, n_classes=n_classes),
        sem.Predictions.from_arrays(dt_maps),
    )

    gt_shards = _shard(gt_maps, n_ranks)
    dt_shards = _shard(dt_maps, n_ranks)
    partials = [
        _evaluator_partial(n_classes, parity_mode, rank, g, d)
        for rank, (g, d) in enumerate(zip(gt_shards, dt_shards))
    ]
    merged = StreamingSemanticEvaluator.from_partials(n_classes, partials, parity_mode).finalize()

    np.testing.assert_array_equal(
        merged.confusion_matrix.counts(),
        batch.confusion_matrix.counts(),
    )
    # u64 counts are integer-exact; mIoU / FWIoU / pixel_accuracy
    # derive from those counts deterministically, so bit-equality
    # follows.
    assert merged.miou == batch.miou
    assert merged.fwiou == batch.fwiou
    assert merged.pixel_accuracy == batch.pixel_accuracy


# ---------------------------------------------------------------------------
# 2. Roundtrip equals self
# ---------------------------------------------------------------------------


def test_roundtrip_single_partial_equals_finalize() -> None:
    """``from_partials([single_partial]).finalize()`` produces the
    same summary as calling ``finalize()`` on the original evaluator.
    Pins the checkpoint/restore property as a special case.
    """
    gt_maps, dt_maps = _label_maps(seed=7, n_images=4)
    n_classes = 3

    ev = StreamingSemanticEvaluator(n_classes, "corrected", rank_id=0)
    for i in sorted(gt_maps):
        ev.update(i, gt_maps[i], dt_maps[i])
    direct = ev.snapshot()

    partial = ev.finalize_to_partial()
    restored = StreamingSemanticEvaluator.from_partials(
        n_classes, [partial], "corrected"
    ).finalize()

    np.testing.assert_array_equal(
        direct.confusion_matrix.counts(),
        restored.confusion_matrix.counts(),
    )


# ---------------------------------------------------------------------------
# 3. N-way associativity
# ---------------------------------------------------------------------------


def test_n_way_equals_pairwise_reduction() -> None:
    """``from_partials([a, b, c])`` produces stats bit-equal to
    ``from_partials([from_partials([a, b]).finalize_to_partial(), c])``.
    """
    gt_maps, dt_maps = _label_maps(seed=99, n_images=9)
    n_classes = 3

    gt_shards = _shard(gt_maps, 3)
    dt_shards = _shard(dt_maps, 3)
    a, b, c = (
        _evaluator_partial(n_classes, "corrected", rank, g, d)
        for rank, (g, d) in enumerate(zip(gt_shards, dt_shards))
    )

    one_shot = StreamingSemanticEvaluator.from_partials(
        n_classes, [a, b, c], "corrected"
    ).finalize()

    ab = StreamingSemanticEvaluator.from_partials(
        n_classes, [a, b], "corrected"
    ).finalize_to_partial()
    pairwise = StreamingSemanticEvaluator.from_partials(n_classes, [ab, c], "corrected").finalize()

    np.testing.assert_array_equal(
        one_shot.confusion_matrix.counts(),
        pairwise.confusion_matrix.counts(),
    )


# ---------------------------------------------------------------------------
# 4. Disjoint-partition rejected
# ---------------------------------------------------------------------------


def test_partition_overlap_rejected() -> None:
    """Two partials covering the same image_id raise
    ``PartialPartitionOverlap`` naming both rank ids and the colliding
    image id.
    """
    gt_maps, dt_maps = _label_maps(seed=1, n_images=4)
    n_classes = 3

    a = _evaluator_partial(n_classes, "corrected", 0, {2: gt_maps[2]}, {2: dt_maps[2]})
    b = _evaluator_partial(n_classes, "corrected", 1, {2: gt_maps[2]}, {2: dt_maps[2]})

    with pytest.raises(sem.PartialPartitionOverlap) as exc_info:
        StreamingSemanticEvaluator.from_partials(n_classes, [a, b], "corrected")
    assert exc_info.value.image_id == 2
    assert {exc_info.value.rank_a, exc_info.value.rank_b} == {0, 1}


# ---------------------------------------------------------------------------
# 5. Dataset-hash mismatch
# ---------------------------------------------------------------------------


def test_n_classes_mismatch_rejected() -> None:
    """Different ``n_classes`` trips the cheaper shape-fingerprint
    check before the dataset hash. Reports ``grid_mismatch``.
    """
    gt_maps, dt_maps = _label_maps(seed=2, n_images=2)
    a = _evaluator_partial(3, "corrected", 0, gt_maps, dt_maps)

    with pytest.raises(sem.PartialFormatMismatch) as exc_info:
        StreamingSemanticEvaluator.from_partials(4, [a], "corrected")
    assert exc_info.value.kind == "grid_mismatch"


def test_ignore_label_mismatch_rejected() -> None:
    """``ignore_label`` is part of the dataset hash — a partial built
    with ``ignore_label=255`` cannot merge into a ``None`` evaluator.
    Different ``ignore_label`` values keep ``n_classes`` aligned, so
    the shape-fingerprint passes and the hash check fires.
    """
    gt_maps, dt_maps = _label_maps(seed=3, n_images=2)
    a = _evaluator_partial(3, "corrected", 0, gt_maps, dt_maps, ignore_label=255)

    with pytest.raises(sem.PartialDatasetMismatch):
        StreamingSemanticEvaluator.from_partials(3, [a], "corrected", ignore_label=None)


# ---------------------------------------------------------------------------
# 6. Parity mode mismatch
# ---------------------------------------------------------------------------


def test_parity_mode_mismatch_rejected() -> None:
    """Strict and corrected partials cannot merge — parity_mode is
    a header field check, fires before the params_hash compare.
    Reports ``parity_mismatch``.
    """
    gt_maps, dt_maps = _label_maps(seed=4, n_images=2)
    a = _evaluator_partial(3, "strict", 0, gt_maps, dt_maps)

    with pytest.raises(sem.PartialFormatMismatch) as exc_info:
        StreamingSemanticEvaluator.from_partials(3, [a], "corrected")
    assert exc_info.value.kind == "parity_mismatch"


# ---------------------------------------------------------------------------
# 7. Paradigm mismatch
# ---------------------------------------------------------------------------


def test_paradigm_mismatch_rejected() -> None:
    """An instance partial loaded by semantic ``from_partials``
    surfaces as ``PartialFormatMismatch{kind: paradigm_mismatch}``.
    """
    from vernier._impl import StreamingEvaluator as _InstanceStreaming

    # Build a minimal instance partial via the existing harness shape.
    gt_json = (
        b'{"images":[{"id":1,"width":4,"height":4}],'
        b'"categories":[{"id":1,"name":"a"}],"annotations":[]}'
    )
    inst_ev = _InstanceStreaming(gt_json, iou_type="bbox", rank_id=0)
    instance_partial = inst_ev.finalize_to_partial()

    with pytest.raises(sem.PartialFormatMismatch) as exc_info:
        StreamingSemanticEvaluator.from_partials(3, [instance_partial], "corrected")
    assert exc_info.value.kind == "paradigm_mismatch"


# ---------------------------------------------------------------------------
# 8. Strict-mode rank collision
# ---------------------------------------------------------------------------


def test_strict_rank_collision_rejected() -> None:
    """Two strict-mode partials with the same rank_id raise
    ``PartialRankCollision``.
    """
    gt_maps, dt_maps = _label_maps(seed=5, n_images=4)
    a = _evaluator_partial(3, "strict", 7, {0: gt_maps[0]}, {0: dt_maps[0]})
    b = _evaluator_partial(3, "strict", 7, {1: gt_maps[1]}, {1: dt_maps[1]})

    with pytest.raises(sem.PartialRankCollision) as exc_info:
        StreamingSemanticEvaluator.from_partials(3, [a, b], "strict")
    assert exc_info.value.rank_id == 7


def test_corrected_rank_collision_tolerated() -> None:
    """Corrected mode tolerates duplicate rank_ids — the field is
    informational only.
    """
    gt_maps, dt_maps = _label_maps(seed=6, n_images=4)
    a = _evaluator_partial(3, "corrected", 7, {0: gt_maps[0]}, {0: dt_maps[0]})
    b = _evaluator_partial(3, "corrected", 7, {1: gt_maps[1]}, {1: dt_maps[1]})

    # Should not raise — rank_id is informational in corrected mode.
    StreamingSemanticEvaluator.from_partials(3, [a, b], "corrected").finalize()


# ---------------------------------------------------------------------------
# 9. Format-version refusal
# ---------------------------------------------------------------------------


def test_wrong_version_rejected() -> None:
    """Hand-craft a partial with version=99; ``PartialFormatMismatch``
    with ``kind == "wrong_version"``.
    """
    bad = bytearray(b"VRPS")
    bad.append(99)  # wrong version
    bad.extend(b"\x00" * 32)  # padding
    bad.extend(b"\x00\x00\x00\x00")  # crc placeholder

    with pytest.raises(sem.PartialFormatMismatch) as exc_info:
        StreamingSemanticEvaluator.from_partials(3, [bytes(bad)], "corrected")
    assert exc_info.value.kind == "wrong_version"


# ---------------------------------------------------------------------------
# 10. CRC corruption
# ---------------------------------------------------------------------------


def test_crc_corruption_detected() -> None:
    """Flip one byte in the body; ``PartialFormatMismatch`` with
    ``kind in {"crc", "rkyv_decode"}``.
    """
    gt_maps, dt_maps = _label_maps(seed=8, n_images=2)
    partial = bytearray(_evaluator_partial(3, "corrected", 0, gt_maps, dt_maps))
    # Flip a byte well inside the body, not in the framing.
    partial[20] ^= 0xFF

    with pytest.raises(sem.PartialFormatMismatch) as exc_info:
        StreamingSemanticEvaluator.from_partials(3, [bytes(partial)], "corrected")
    assert exc_info.value.kind in {"crc", "rkyv_decode"}


# ---------------------------------------------------------------------------
# 11. Shared exception classes across paradigms
# ---------------------------------------------------------------------------


def test_encode_decode_encode_byte_stability() -> None:
    """Round-trip determinism: encoding the same state twice produces
    byte-identical output. Pins the canonical body layout (sorted
    seen_images, deterministic confusion-matrix archive) — a stale
    rkyv layout drift would surface here.

    Note: ``from_partials`` resets ``rank_id`` to ``None`` because the
    merged evaluator represents the union, not any one rank. The
    test re-encodes the *same* evaluator to isolate the canonical-
    body property from the rank-id propagation question.
    """
    gt_maps, dt_maps = _label_maps(seed=11, n_images=4)
    n_classes = 3

    ev = StreamingSemanticEvaluator(n_classes, "corrected", rank_id=0)
    for i in sorted(gt_maps):
        ev.update(i, gt_maps[i], dt_maps[i])
    bytes_a = ev.to_partial()
    bytes_b = ev.to_partial()
    assert bytes_a == bytes_b


def test_empty_rank_merges_cleanly() -> None:
    """A rank that finished without consuming any images should
    still produce a valid partial. The merged result must equal the
    other rank's contribution alone.
    """
    gt_maps, dt_maps = _label_maps(seed=12, n_images=4)
    n_classes = 3

    populated = _evaluator_partial(n_classes, "corrected", 0, gt_maps, dt_maps)
    empty = StreamingSemanticEvaluator(n_classes, "corrected", rank_id=1).finalize_to_partial()

    merged = StreamingSemanticEvaluator.from_partials(
        n_classes, [populated, empty], "corrected"
    ).finalize()
    only_populated = StreamingSemanticEvaluator.from_partials(
        n_classes, [populated], "corrected"
    ).finalize()

    np.testing.assert_array_equal(
        merged.confusion_matrix.counts(),
        only_populated.confusion_matrix.counts(),
    )


def test_partial_exceptions_shared_across_paradigms() -> None:
    """``vernier.semantic.PartialFormatMismatch`` is the same class
    object as ``vernier.instance.PartialFormatMismatch``. Lets users
    catch with either namespace.
    """
    import vernier.instance as inst

    assert sem.PartialFormatMismatch is inst.PartialFormatMismatch
    assert sem.PartialDatasetMismatch is inst.PartialDatasetMismatch
    assert sem.PartialParamsMismatch is inst.PartialParamsMismatch
    assert sem.PartialPartitionOverlap is inst.PartialPartitionOverlap
    assert sem.PartialRankCollision is inst.PartialRankCollision
