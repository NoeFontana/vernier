"""Comparator-registry shape (ADR-0033 §"Comparator registry").

Asserts that:
- the instance comparator is registered and dispatches against the
  legacy ``compare_cell`` shape (bit-identical to the v1 behaviour);
- panoptic / semantic / streaming entries exist as stubs that raise
  ``NotImplementedError`` from ``compare()`` — which proves the
  registry shape works without forcing B1/B2/B3 completion;
- ``register_comparator`` rejects re-registering the instance entry.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.harness.parity import (
    ComparableArtifact,
    Comparator,
    ConfusionMatrix,
    PanopticSnapshot,
    StreamingPair,
    TensorArtifact,
    compare_cell,
    get_comparator,
    register_comparator,
)
from bench.harness.schema import IouType, Paradigm


def test_instance_comparator_is_registered() -> None:
    cmp = get_comparator("instance")
    assert isinstance(cmp, Comparator)
    assert cmp.paradigm == "instance"


@pytest.mark.parametrize("paradigm", ["semantic", "streaming"])
def test_stub_comparators_raise_not_implemented(paradigm: Paradigm) -> None:
    """Stub registry entries for B2/B3 paradigms that haven't landed
    their concrete comparator yet. B1 (panoptic) is excluded — its
    real comparator is registered (see :class:`_PanopticComparator`)."""
    cmp = get_comparator(paradigm)
    assert isinstance(cmp, Comparator)
    assert cmp.paradigm == paradigm
    with pytest.raises(NotImplementedError, match=paradigm):
        cmp.compare(workload_id="anything", iou_type="bbox", impl_outputs={})


def test_panoptic_comparator_is_registered() -> None:
    """B1 has registered :class:`_PanopticComparator`. The registry
    returns it (no stub) and ``compare`` accepts the ``"pq"`` metric."""
    cmp = get_comparator("panoptic")
    assert isinstance(cmp, Comparator)
    assert cmp.paradigm == "panoptic"
    # Empty impl_outputs is a degenerate-but-valid input — just yields
    # a report with zero tiers.
    report = cmp.compare(workload_id="x", iou_type="pq", impl_outputs={})
    assert report.tiers == []


def test_instance_comparator_matches_legacy_compare_cell_on_perfect_pair() -> None:
    """The legacy ``compare_cell`` entry-point goes through the
    registered instance comparator. Two identical tensors produce a
    passing strict-tier report."""
    tensor = np.zeros((10, 101, 1, 4, 3), dtype=np.float64)
    report = compare_cell(
        workload_id="smoke",
        iou_type="bbox",
        impl_tensors={"vernier": tensor, "pycocotools": tensor.copy()},
        impl_sha256={"vernier": "a" * 64, "pycocotools": "b" * 64},
    )
    assert report.passed
    strict = next(t for t in report.tiers if t.tier == "strict")
    assert strict.passed
    assert strict.divergent_count == 0


def test_instance_comparator_catches_strict_divergence() -> None:
    """Bit-equivalent to the existing ``test_strict_tier_catches`` —
    proves the registry indirection didn't change behaviour."""
    a = np.zeros((10, 101, 1, 4, 3), dtype=np.float64)
    b = a.copy()
    b[3, 50, 0, 1, 2] = 1e-12
    report = compare_cell(
        workload_id="smoke",
        iou_type="bbox",
        impl_tensors={"vernier": a, "pycocotools": b},
        impl_sha256={"vernier": "a" * 64, "pycocotools": "b" * 64},
    )
    strict = next(t for t in report.tiers if t.tier == "strict")
    assert not strict.passed


def test_register_comparator_rejects_instance_paradigm() -> None:
    """The instance comparator is the construction-per-call shape
    that ``compare_cell`` exercises. ``register_comparator('instance',
    ...)`` would silently break the legacy entry-point — refuse it."""
    cmp = get_comparator("panoptic")
    with pytest.raises(ValueError, match="instance"):
        register_comparator("instance", cmp)


def test_register_comparator_replaces_stub_for_other_paradigms() -> None:
    """B1/B2/B3 swap their concrete comparators in via this entry-
    point. The registry must accept and dispatch the replacement."""
    from typing import ClassVar

    class _FakePanopticComparator:
        paradigm: ClassVar[Paradigm] = "panoptic"

        def compare(self, *, workload_id, iou_type, impl_outputs):  # type: ignore[no-untyped-def]
            from bench.harness.parity import CellParityReport

            return CellParityReport(workload_id=workload_id, iou_type=iou_type, tiers=[])

    original = get_comparator("panoptic")
    try:
        register_comparator("panoptic", _FakePanopticComparator())
        assert isinstance(get_comparator("panoptic"), _FakePanopticComparator)
        report = get_comparator("panoptic").compare(
            workload_id="x", iou_type="bbox", impl_outputs={}
        )
        assert report.tiers == []
    finally:
        register_comparator("panoptic", original)


def test_comparable_artifact_variants_round_trip_canonical_form() -> None:
    """Every variant of ``ComparableArtifact`` knows how to render its
    canonical form for divergence reporting."""
    tensor = TensorArtifact(sha256="a" * 64, shape=(10, 101, 1, 4, 3))
    snapshot = PanopticSnapshot(pq=0.5, sq=0.6, rq=0.7)
    confusion = ConfusionMatrix(n_classes=19, counts_sha256="c" * 64)
    streaming = StreamingPair(batch_stats={"AP": 0.3}, stream_stats={"AP": 0.3})

    for art in (tensor, snapshot, confusion, streaming):
        canon = art.to_canonical_form()
        assert isinstance(canon, dict)
        assert canon["kind"] == art.kind


_ALL_ARTIFACT_TYPES: tuple[type[ComparableArtifact], ...] = (
    TensorArtifact,
    PanopticSnapshot,
    ConfusionMatrix,
    StreamingPair,
)


@pytest.mark.parametrize(
    "iou_type",
    ["bbox", "segm", "keypoints", "boundary"],
)
def test_instance_iou_types_drive_legacy_tier_pairs(iou_type: IouType) -> None:
    """Every instance iou-type still produces tier pairs through the
    registry; defensive against a future refactor that drops bbox /
    segm / keypoints / boundary from the registered entry."""
    a = np.zeros((10, 101, 1, 4, 3), dtype=np.float64)
    impl_filter: dict[str, np.ndarray]
    if iou_type == "boundary":
        impl_filter = {"vernier": a, "boundary-iou-api": a.copy()}
        sha = {"vernier": "a" * 64, "boundary-iou-api": "b" * 64}
    else:
        impl_filter = {"vernier": a, "pycocotools": a.copy()}
        sha = {"vernier": "a" * 64, "pycocotools": "b" * 64}
    report = compare_cell(
        workload_id="x",
        iou_type=iou_type,
        impl_tensors=impl_filter,
        impl_sha256=sha,
    )
    assert report.tiers, f"no tiers produced for {iou_type}"
    assert all(t.passed for t in report.tiers)
