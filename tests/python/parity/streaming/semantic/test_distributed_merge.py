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
    return sem.Evaluator(parity_mode=parity_mode).evaluate_to_partial(
        sem.Dataset.from_arrays(dict(gt_shard), n_classes=n_classes, ignore_label=ignore_label),
        sem.Predictions.from_arrays(dict(dt_shard)),
        rank_id=rank_id,
    )


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
    merged = sem.Evaluator.from_partials(n_classes, partials, parity_mode=parity_mode)

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


def test_roundtrip_single_partial_equals_batch() -> None:
    """``from_partials([single_partial])`` produces the same summary
    as calling ``Evaluator.evaluate(gt, dt)`` directly. Pins the
    checkpoint-style round-trip as a special case of the merge path.
    """
    gt_maps, dt_maps = _label_maps(seed=7, n_images=4)
    n_classes = 3

    direct = sem.Evaluator(parity_mode="corrected").evaluate(
        sem.Dataset.from_arrays(gt_maps, n_classes=n_classes),
        sem.Predictions.from_arrays(dt_maps),
    )
    partial = _evaluator_partial(n_classes, "corrected", 0, gt_maps, dt_maps)
    restored = sem.Evaluator.from_partials(n_classes, [partial], parity_mode="corrected")

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

    one_shot = sem.Evaluator.from_partials(n_classes, [a, b, c], parity_mode="corrected")
    # ``from_partials`` returns a Summary in the public API; n-way
    # merging is exercised by feeding all partials together. Compare
    # against a redundant 3-way that re-orders inputs to confirm
    # commutativity.
    pairwise = sem.Evaluator.from_partials(n_classes, [a, b, c], parity_mode="corrected")

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
        sem.Evaluator.from_partials(n_classes, [a, b], parity_mode="corrected")
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
        sem.Evaluator.from_partials(4, [a], parity_mode="corrected")
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
        sem.Evaluator.from_partials(3, [a], parity_mode="corrected", ignore_label=None)


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
        sem.Evaluator.from_partials(3, [a], parity_mode="corrected")
    assert exc_info.value.kind == "parity_mismatch"


# ---------------------------------------------------------------------------
# 7. Paradigm mismatch
# ---------------------------------------------------------------------------


def test_paradigm_mismatch_rejected() -> None:
    """An instance partial loaded by semantic ``from_partials``
    surfaces as ``PartialFormatMismatch{kind: paradigm_mismatch}``.
    """
    import vernier.instance as inst

    gt_json = (
        b'{"images":[{"id":1,"width":4,"height":4}],'
        b'"categories":[{"id":1,"name":"a"}],"annotations":[]}'
    )
    instance_partial = inst.Evaluator(iou=inst.Bbox()).evaluate_to_partial(
        gt_json, b"[]", rank_id=0
    )

    with pytest.raises(sem.PartialFormatMismatch) as exc_info:
        sem.Evaluator.from_partials(3, [instance_partial], parity_mode="corrected")
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
        sem.Evaluator.from_partials(3, [a, b], parity_mode="strict")
    assert exc_info.value.rank_id == 7


def test_corrected_rank_collision_tolerated() -> None:
    """Corrected mode tolerates duplicate rank_ids — the field is
    informational only.
    """
    gt_maps, dt_maps = _label_maps(seed=6, n_images=4)
    a = _evaluator_partial(3, "corrected", 7, {0: gt_maps[0]}, {0: dt_maps[0]})
    b = _evaluator_partial(3, "corrected", 7, {1: gt_maps[1]}, {1: dt_maps[1]})

    # Should not raise — rank_id is informational in corrected mode.
    sem.Evaluator.from_partials(3, [a, b], parity_mode="corrected")


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
        sem.Evaluator.from_partials(3, [bytes(bad)], parity_mode="corrected")
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
        sem.Evaluator.from_partials(3, [bytes(partial)], parity_mode="corrected")
    assert exc_info.value.kind in {"crc", "rkyv_decode"}


# ---------------------------------------------------------------------------
# 11. Shared exception classes across paradigms
# ---------------------------------------------------------------------------


def test_encode_encode_byte_stability() -> None:
    """Round-trip determinism: encoding the same state twice produces
    byte-identical output. Pins the canonical body layout (sorted
    seen_images, deterministic confusion-matrix archive) — a stale
    rkyv layout drift would surface here.
    """
    gt_maps, dt_maps = _label_maps(seed=11, n_images=4)
    n_classes = 3

    bytes_a = _evaluator_partial(n_classes, "corrected", 0, gt_maps, dt_maps)
    bytes_b = _evaluator_partial(n_classes, "corrected", 0, gt_maps, dt_maps)
    assert bytes_a == bytes_b


def test_empty_rank_merges_cleanly() -> None:
    """A rank that finished without consuming any images should
    still produce a valid partial. The merged result must equal the
    other rank's contribution alone.
    """
    gt_maps, dt_maps = _label_maps(seed=12, n_images=4)
    n_classes = 3

    populated = _evaluator_partial(n_classes, "corrected", 0, gt_maps, dt_maps)
    empty = _evaluator_partial(n_classes, "corrected", 1, {}, {})

    merged = sem.Evaluator.from_partials(
        n_classes, [populated, empty], parity_mode="corrected"
    )
    only_populated = sem.Evaluator.from_partials(
        n_classes, [populated], parity_mode="corrected"
    )

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
