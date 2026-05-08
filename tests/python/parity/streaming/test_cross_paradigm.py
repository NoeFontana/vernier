"""ADR-0032 PR-F: cross-paradigm rejection and shared-exception identity.

Two contracts pinned here:

1. **Paradigm-mismatch is a structural rejection.** Loading a partial
   from one paradigm into another paradigm's ``from_partials`` raises
   ``PartialFormatMismatch`` with ``kind == "paradigm_mismatch"``
   *before* any body archive validation runs. The header carries
   ``paradigm_kind: u8`` exactly so the receiver can integer-compare
   and reject.

2. **The five ``Partial*`` exception classes are paradigm-shared.**
   Each paradigm's ``__init__.py`` re-exports the same Python class
   object from ``vernier._core``, so ``vernier.instance.PartialX
   is vernier.semantic.PartialX is vernier.panoptic.PartialX`` holds
   at runtime — a user's top-level handler catches one class and
   gets all three paradigms.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import vernier.instance as inst
import vernier.panoptic as pq
import vernier.semantic as sem

# ---------------------------------------------------------------------------
# Paradigm-shared exception identity (Contract 2).
# ---------------------------------------------------------------------------


_SHARED_NAMES = (
    "PartialFormatMismatch",
    "PartialDatasetMismatch",
    "PartialParamsMismatch",
    "PartialPartitionOverlap",
    "PartialRankCollision",
)


@pytest.mark.parametrize("name", _SHARED_NAMES)
def test_partial_exception_is_shared_across_paradigms(name: str) -> None:
    """Each of the five ``Partial*`` classes is a single Python class
    object re-exported from every paradigm namespace.
    """
    cls_inst = getattr(inst, name)
    cls_sem = getattr(sem, name)
    cls_pq = getattr(pq, name)

    assert cls_inst is cls_sem
    assert cls_sem is cls_pq


# ---------------------------------------------------------------------------
# Paradigm-mismatch rejection (Contract 1).
#
# Build one partial in each paradigm; load it in each *other* paradigm's
# ``from_partials``; assert the typed ``paradigm_mismatch`` rejection.
# Six pairings cover the full cross product.
# ---------------------------------------------------------------------------


_FIXTURES = Path(__file__).parent.parent / "fixtures"
_PERFECT_MATCH_GT = (_FIXTURES / "perfect_match" / "gt.json").read_bytes()
_PERFECT_MATCH_DT = (_FIXTURES / "perfect_match" / "dt.json").read_bytes()


def _instance_partial() -> bytes:
    """Build a one-rank instance partial from the perfect_match fixture."""
    ev = inst.StreamingEvaluator(
        _PERFECT_MATCH_GT,
        iou_type="bbox",
        parity_mode="corrected",
        rank_id=0,
    )
    ev.update(_PERFECT_MATCH_DT)
    return ev.finalize_to_partial()


def _semantic_partial() -> bytes:
    """Build a one-rank semantic partial."""
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 3, size=(4, 4), dtype=np.uint32)
    dt = gt.copy()
    ev = sem.StreamingEvaluator(3, "corrected", rank_id=0)
    ev.update(0, gt, dt)
    return ev.finalize_to_partial()


_PANOPTIC_CATS = json.dumps([{"id": 1, "isthing": True}, {"id": 2, "isthing": False}]).encode()


def _panoptic_partial() -> bytes:
    """Build a one-rank panoptic partial."""
    label_map = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.uint32)
    segs = json.dumps(
        [
            {"id": 1, "category_id": 1, "iscrowd": 0, "area": 4},
            {"id": 2, "category_id": 2, "iscrowd": 0, "area": 4},
        ]
    ).encode()
    ev = pq.StreamingEvaluator(_PANOPTIC_CATS, "corrected", rank_id=0)
    ev.update(0, label_map, segs, label_map, segs)
    return ev.finalize_to_partial()


# ---------------------------------------------------------------------------
# Six receiver / sender pairings.
# ---------------------------------------------------------------------------


def _load_into_instance(partial: bytes) -> None:
    inst.StreamingEvaluator.from_partials(
        _PERFECT_MATCH_GT, [partial], iou_type="bbox", parity_mode="corrected"
    )


def _load_into_semantic(partial: bytes) -> None:
    sem.StreamingEvaluator.from_partials(3, [partial], "corrected")


def _load_into_panoptic(partial: bytes) -> None:
    pq.StreamingEvaluator.from_partials(_PANOPTIC_CATS, [partial], "corrected")


@pytest.mark.parametrize(
    ("sender_name", "build_partial", "receiver_name", "load"),
    [
        # Instance sender.
        ("instance", _instance_partial, "semantic", _load_into_semantic),
        ("instance", _instance_partial, "panoptic", _load_into_panoptic),
        # Semantic sender.
        ("semantic", _semantic_partial, "instance", _load_into_instance),
        ("semantic", _semantic_partial, "panoptic", _load_into_panoptic),
        # Panoptic sender.
        ("panoptic", _panoptic_partial, "instance", _load_into_instance),
        ("panoptic", _panoptic_partial, "semantic", _load_into_semantic),
    ],
)
def test_paradigm_mismatch_rejected(
    sender_name: str,
    build_partial,
    receiver_name: str,
    load,
) -> None:
    """A partial built in paradigm A and loaded by paradigm B's
    ``from_partials`` must raise
    ``PartialFormatMismatch{kind: paradigm_mismatch}``. The shared
    exception class — caught from any paradigm namespace — is the
    contract.
    """
    partial = build_partial()

    # Catch through the sender's namespace; identity test asserts it
    # is the same class object the receiver would raise from.
    with pytest.raises(inst.PartialFormatMismatch) as exc_info:
        load(partial)

    assert exc_info.value.kind == "paradigm_mismatch", (
        f"expected paradigm_mismatch routing {sender_name} → "
        f"{receiver_name}, got kind={exc_info.value.kind!r}"
    )
