"""Per-paradigm parity coupling for the bench harness (ADR-0017 §D2,
ADR-0033 §"Comparator registry").

After every cell finishes, every runner has written its result
artifacts to disk. The comparator for that cell's paradigm loads them
and asserts the cross-impl invariants documented in the parity ADRs:

- **instance** — three-tier strict / aligned / boundary contract from
  ADR-0002:
    - *strict* — vernier reproduces pycocotools bit-exactly
      (``np.array_equal``).
    - *aligned* — vernier matches faster-coco-eval within a small
      absolute tolerance; faster-coco-eval has documented float-order
      quirks.
    - *boundary* — vernier matches the boundary-iou-api oracle within
      ``BOUNDARY_PARITY_EPS``.
- **panoptic** — strict vs ``pq_compute_single_core(proc_id=0, ...)``
  per ADR-0025. Registered by B1.
- **semantic** — strict on integer confusion-matrix counts for
  Cityscapes; aligned for ADE20K-vs-mmseg pending PR-B6/7/8 vendoring.
  Registered by B2.
- **streaming** — bit-equal ``Summary.stats`` between batch and stream
  per ADR-0032; no external oracle. Registered by B3.

A cell that fails its tier(s) writes ``divergence_report.json`` next
to its impl results. Dev mode keeps going (the orchestrator only
logs); release mode (M5) will turn this into a hard fail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from bench.harness.schema import IouType, Paradigm

# Mirror of ``crates/vernier-core/src/parity.rs::PARITY_EPS`` —
# ``np.spacing(1)`` for f64. A unit test in that crate checks the
# numeric value; mirroring here avoids a vernier import in the harness.
PARITY_EPS: float = 2.220446049250313e-16

# Mirror of ``crates/vernier-core/src/boundary_parity.rs::BOUNDARY_PARITY_EPS``.
BOUNDARY_PARITY_EPS: float = 1e-9

# Aligned tier: 4 ULP. Faster-coco-eval reorders some accumulations so
# the tensor differs by a handful of ULP from pycocotools, even on
# fixtures where the strict tier is bit-equal.
ALIGNED_ATOL: float = 4.0 * PARITY_EPS

Tier = Literal["strict", "aligned", "boundary"]


class Divergence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: tuple[int, ...]
    value_a: float
    value_b: float
    abs_diff: float


class TierResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier: Tier
    impl_a: str
    impl_b: str
    atol: float
    passed: bool
    divergent_count: int
    first_divergence: Divergence | None
    tensor_sha256_a: str
    tensor_sha256_b: str


class CellParityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    workload_id: str
    iou_type: IouType
    tiers: list[TierResult]

    @property
    def passed(self) -> bool:
        return all(t.passed for t in self.tiers)


# ---------------------------------------------------------------------------
# Comparable artifacts (Pydantic discriminated union, JSON-portable).
#
# Each variant carries a ``to_canonical_form()`` so the divergence
# report can render a stable snapshot of either side regardless of
# paradigm. Instance keeps the legacy "first 16 bytes of tensor hash +
# divergent index" via the existing ``CellParityReport`` shape;
# B1/B2/B3 plug in their canonical-form snapshots when they register
# their comparators.
# ---------------------------------------------------------------------------


class _ArtifactBase(BaseModel):
    """Common base for ``ComparableArtifact`` variants. Subclasses set
    ``kind`` to a unique discriminator literal."""

    model_config = ConfigDict(extra="forbid")

    def to_canonical_form(self) -> dict[str, Any]:
        raise NotImplementedError("subclasses must implement to_canonical_form")


class TensorArtifact(_ArtifactBase):
    """Detection precision tensor — the v1 shape carried into v2."""

    kind: Literal["tensor"] = "tensor"
    sha256: str
    shape: tuple[int, ...]
    # ``data`` is intentionally non-serialized (NumPy arrays are
    # supplied at-runtime by the loader). Keeping it ``None``-as-
    # default-for-deserialization preserves the round-trip when the
    # divergence report is read back.
    dtype: str = "float64"

    def to_canonical_form(self) -> dict[str, Any]:
        return {"kind": self.kind, "sha256": self.sha256, "shape": list(self.shape)}


class PanopticSnapshot(_ArtifactBase):
    """Panoptic per-class table + scalar PQ/SQ/RQ. Stub fields; B1
    populates the full shape (per_class table, IoU sums, TP/FP/FN
    counters) when it lands ``compare()``."""

    kind: Literal["panoptic_snapshot"] = "panoptic_snapshot"
    pq: float = 0.0
    sq: float = 0.0
    rq: float = 0.0
    # Per-class entries are open per-paradigm; B1 populates with the
    # shape from ``tests/python/parity_panoptic/harness.py``.
    per_class: dict[str, dict[str, float]] = Field(default_factory=dict)

    def to_canonical_form(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "pq": self.pq,
            "sq": self.sq,
            "rq": self.rq,
            "per_class": dict(self.per_class),
        }


class ConfusionMatrix(_ArtifactBase):
    """Semantic NxN integer confusion matrix. Stub fields; B2
    populates the full shape (counts uint64 NxN, derived per-class
    IoU, mIoU)."""

    kind: Literal["confusion_matrix"] = "confusion_matrix"
    n_classes: int = 0
    # SHA of the .npy with the integer counts; the float mIoU is
    # derived bit-deterministically from the counts so the SHA alone
    # is the parity carrier.
    counts_sha256: str = ""

    def to_canonical_form(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_classes": self.n_classes,
            "counts_sha256": self.counts_sha256,
        }


class StreamingPair(_ArtifactBase):
    """Streaming batch-vs-stream Summary.stats pair. Stub fields; B3
    populates the full shape (batch stats vector, stream stats vector,
    optional RSS curve reference)."""

    kind: Literal["streaming_pair"] = "streaming_pair"
    batch_stats: dict[str, float] = Field(default_factory=dict)
    stream_stats: dict[str, float] = Field(default_factory=dict)

    def to_canonical_form(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "batch_stats": dict(self.batch_stats),
            "stream_stats": dict(self.stream_stats),
        }


# Discriminated union; each variant carries its ``kind`` literal.
ComparableArtifact = TensorArtifact | PanopticSnapshot | ConfusionMatrix | StreamingPair


# ---------------------------------------------------------------------------
# Comparator protocol + registry.
# ---------------------------------------------------------------------------


@runtime_checkable
class Comparator(Protocol):
    """A paradigm-keyed comparator. ``paradigm`` is the registry key;
    ``compare`` returns the cell's parity report.

    The ``impl_outputs`` mapping pairs each impl name with the artifact
    that impl produced for the cell. The instance comparator unwraps
    the ``TensorArtifact``s into the legacy NumPy-tensor pipeline; the
    panoptic / semantic / streaming comparators consume their own
    variant directly.
    """

    paradigm: ClassVar[Paradigm]

    def compare(
        self,
        *,
        workload_id: str,
        iou_type: IouType,
        impl_outputs: dict[str, ComparableArtifact],
    ) -> CellParityReport: ...


# Each iou cell defines the tier comparisons that apply when the named
# impls have produced tensors. Read as: "if both impls ran this cell,
# compare them at this tier with this tolerance."
_TIER_PAIRS: dict[IouType, tuple[tuple[Tier, str, str, float], ...]] = {
    "bbox": (
        ("strict", "vernier", "pycocotools", 0.0),
        ("aligned", "vernier", "faster-coco-eval", ALIGNED_ATOL),
    ),
    "segm": (
        ("strict", "vernier", "pycocotools", 0.0),
        ("aligned", "vernier", "faster-coco-eval", ALIGNED_ATOL),
    ),
    "keypoints": (
        ("strict", "vernier", "pycocotools", 0.0),
        ("aligned", "vernier", "faster-coco-eval", ALIGNED_ATOL),
    ),
    "boundary": (("boundary", "vernier", "boundary-iou-api", BOUNDARY_PARITY_EPS),),
}


def _compare_pair(
    *,
    tier: Tier,
    impl_a: str,
    impl_b: str,
    tensor_a: np.ndarray,
    tensor_b: np.ndarray,
    sha_a: str,
    sha_b: str,
    atol: float,
) -> TierResult:
    if tensor_a.shape != tensor_b.shape:
        raise ValueError(
            f"shape mismatch comparing {impl_a} vs {impl_b}: {tensor_a.shape} vs {tensor_b.shape}"
        )

    diff = np.abs(tensor_a - tensor_b)
    # Strict tier: bit-equality, not "diff <= 0" (NaNs in either tensor
    # would make a finite-diff check pass spuriously).
    divergent_mask = ~(tensor_a == tensor_b) if tier == "strict" else diff > atol

    divergent_count = int(divergent_mask.sum())
    first_divergence: Divergence | None = None
    if divergent_count > 0:
        first_idx = tuple(int(i) for i in np.argwhere(divergent_mask)[0])
        first_divergence = Divergence(
            index=first_idx,
            value_a=float(tensor_a[first_idx]),
            value_b=float(tensor_b[first_idx]),
            abs_diff=float(diff[first_idx]),
        )

    return TierResult(
        tier=tier,
        impl_a=impl_a,
        impl_b=impl_b,
        atol=atol,
        passed=divergent_count == 0,
        divergent_count=divergent_count,
        first_divergence=first_divergence,
        tensor_sha256_a=sha_a[:12],
        tensor_sha256_b=sha_b[:12],
    )


class _InstanceComparator:
    """Detection (bbox/segm/keypoints/boundary) three-tier comparator.

    Bit-identical to the v1 ``compare_cell`` pipeline; the registry
    indirection only changes how it's selected, not what it does.
    """

    paradigm: ClassVar[Paradigm] = "instance"

    def __init__(
        self,
        *,
        impl_tensors: dict[str, np.ndarray],
        impl_sha256: dict[str, str],
    ) -> None:
        self._tensors = impl_tensors
        self._sha256 = impl_sha256

    def compare(
        self,
        *,
        workload_id: str,
        iou_type: IouType,
        impl_outputs: dict[str, ComparableArtifact],
    ) -> CellParityReport:
        # ``impl_outputs`` is required by the Comparator protocol but
        # the legacy entry-point passes tensors directly via the
        # constructor; both shapes converge here.
        tiers: list[TierResult] = []
        for tier, impl_a, impl_b, atol in _TIER_PAIRS[iou_type]:
            if impl_a not in self._tensors or impl_b not in self._tensors:
                continue
            tiers.append(
                _compare_pair(
                    tier=tier,
                    impl_a=impl_a,
                    impl_b=impl_b,
                    tensor_a=self._tensors[impl_a],
                    tensor_b=self._tensors[impl_b],
                    sha_a=self._sha256[impl_a],
                    sha_b=self._sha256[impl_b],
                    atol=atol,
                )
            )
        return CellParityReport(workload_id=workload_id, iou_type=iou_type, tiers=tiers)


class _StubComparator:
    """Placeholder comparator for non-instance paradigms.

    B1/B2/B3 register their concrete comparators when their cells
    land. Until then, the registry knows the paradigm exists but
    refuses to dispatch — proves the registry shape works without
    forcing B-stream completion.
    """

    def __init__(self, paradigm: Paradigm) -> None:
        self.paradigm = paradigm

    def compare(
        self,
        *,
        workload_id: str,
        iou_type: IouType,
        impl_outputs: dict[str, ComparableArtifact],
    ) -> CellParityReport:
        raise NotImplementedError(
            f"comparator for paradigm {self.paradigm!r} is registered by the "
            f"corresponding B-stream (B1 panoptic / B2 semantic / B3 streaming)"
        )


_REGISTRY: dict[Paradigm, Comparator] = {
    "instance": _InstanceComparator(impl_tensors={}, impl_sha256={}),  # type: ignore[arg-type]
    "panoptic": _StubComparator("panoptic"),
    "semantic": _StubComparator("semantic"),
    "streaming": _StubComparator("streaming"),
}


def get_comparator(paradigm: Paradigm) -> Comparator:
    """Look up the comparator for ``paradigm``. Raises ``KeyError`` for
    an unknown paradigm — no silent fallback so registration mistakes
    surface at the call site."""
    return _REGISTRY[paradigm]


def register_comparator(paradigm: Paradigm, comparator: Comparator) -> None:
    """Replace the comparator for ``paradigm``. B-streams call this at
    import time to swap the stub for their concrete implementation.

    The instance comparator is special-cased — it stays construction-
    based (``compare_cell`` constructs a fresh one per call) because
    its tensor inputs aren't a great fit for the per-paradigm
    artifact-dict signature. ``register_comparator("instance", ...)``
    raises rather than silently break the legacy entry point.
    """
    if paradigm == "instance":
        raise ValueError(
            "the instance comparator is the per-call constructor pattern; "
            "register_comparator('instance', ...) is rejected"
        )
    _REGISTRY[paradigm] = comparator


def compare_cell(
    *,
    workload_id: str,
    iou_type: IouType,
    impl_tensors: dict[str, np.ndarray],
    impl_sha256: dict[str, str],
) -> CellParityReport:
    """Run every applicable tier for ``iou_type`` over the impls present
    in ``impl_tensors``. Pairs whose impls aren't both present are
    silently skipped (e.g., bbox cell with only vernier + pycocotools
    skips the aligned tier).

    This is the legacy detection entry-point; the orchestrator calls
    it directly for instance cells. B-stream cells route through
    ``get_comparator(paradigm).compare(...)`` instead.
    """
    comparator = _InstanceComparator(impl_tensors=impl_tensors, impl_sha256=impl_sha256)
    return comparator.compare(
        workload_id=workload_id,
        iou_type=iou_type,
        impl_outputs={},
    )


def write_report(report: CellParityReport, out_dir: Path) -> Path:
    """Persist ``divergence_report.json`` next to the cell's impl results."""
    path = out_dir / "divergence_report.json"
    path.write_text(report.model_dump_json(indent=2))
    return path
