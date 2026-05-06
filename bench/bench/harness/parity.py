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

from bench.harness.schema import IouType, Metric, Paradigm

# Mirror of ``crates/vernier-core/src/parity.rs::PARITY_EPS`` —
# ``np.spacing(1)`` for f64. A unit test in that crate checks the
# numeric value; mirroring here avoids a vernier import in the harness.
PARITY_EPS: float = 2.220446049250313e-16

# Mirror of ``crates/vernier-core/src/boundary_parity.rs::BOUNDARY_PARITY_EPS``.
BOUNDARY_PARITY_EPS: float = 1e-9

# Mirror of ``crates/vernier-panoptic/src/parity.rs::PANOPTIC_PARITY_EPS``.
# Aligned-mode tolerance for panoptic comparisons (B1).
PANOPTIC_PARITY_EPS: float = 1e-9

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
    # ``Metric`` is a superset of ``IouType`` (bbox/segm/keypoints/
    # boundary plus pq/miou/throughput/p99/rss). Detection cells stay
    # IouType-shaped; B1/B2/B3 paradigms use their paradigm metric
    # (panoptic = "pq", semantic = "miou", streaming family).
    iou_type: Metric
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
    """Panoptic per-class table + scalar PQ/SQ/RQ across the All /
    Things / Stuff buckets (ADR-0025).

    Mirrors the ``PanopticSnapshot`` dataclass at
    ``tests/python/parity_panoptic/harness.py``: every bucket carries
    its own ``pq``/``sq``/``rq`` floats and ``n`` count, plus a
    ``per_class`` map keyed by the COCO category id (kept as ``str``
    for JSON-portability — the comparator decodes back to int via the
    union of keys).

    Empty-bucket coercion mirrors :func:`summary_to_snapshot` in the
    harness: vernier returns ``Option<f64>`` / ``Option<usize>`` for
    empty Things/Stuff buckets; this model normalizes to ``0.0`` /
    ``0`` so both impls produce identical JSON regardless of fixture
    shape.
    """

    kind: Literal["panoptic_snapshot"] = "panoptic_snapshot"

    # All bucket (every present category, things + stuff merged).
    pq: float = 0.0
    sq: float = 0.0
    rq: float = 0.0
    n: int = 0

    # Things bucket — categories with ``isthing == 1``.
    pq_things: float = 0.0
    sq_things: float = 0.0
    rq_things: float = 0.0
    n_things: int = 0

    # Stuff bucket — categories with ``isthing == 0``.
    pq_stuff: float = 0.0
    sq_stuff: float = 0.0
    rq_stuff: float = 0.0
    n_stuff: int = 0

    # Per-class rows under the All bucket. Keys are stringified category
    # ids (Pydantic JSON keys are strings; the comparator decodes back
    # to int when union-merging keys across both impls). Each row
    # carries the strict W8 shape ``{pq, sq, rq}``.
    per_class: dict[str, dict[str, float]] = Field(default_factory=dict)

    def to_canonical_form(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "pq": self.pq,
            "sq": self.sq,
            "rq": self.rq,
            "n": self.n,
            "pq_things": self.pq_things,
            "sq_things": self.sq_things,
            "rq_things": self.rq_things,
            "n_things": self.n_things,
            "pq_stuff": self.pq_stuff,
            "sq_stuff": self.sq_stuff,
            "rq_stuff": self.rq_stuff,
            "n_stuff": self.n_stuff,
            "per_class": {k: dict(v) for k, v in self.per_class.items()},
        }


class ConfusionMatrix(_ArtifactBase):
    """Semantic NxN integer confusion matrix (ADR-0033 §B2).

    Two carriers travel together on the strict path:

    - ``counts`` — the `(n_classes, n_classes)` ``uint64`` array
      itself, populated by the comparator after loading the impl's
      ``.npy`` artifact. Optional in the schema (``None`` when the
      object is constructed for canonical-form inspection without
      the underlying tensor — e.g., re-reading a divergence report).
    - ``counts_sha256`` — SHA of the on-disk ``.npy`` file. Always
      populated; serves as the stable identity carrier in the
      divergence-report JSON (the array itself isn't serialized —
      large + non-portable).

    Strict-tier comparison is ``np.array_equal(a.counts, b.counts)``
    on integer arrays; the four headline float metrics (mIoU / FWIoU
    / pixel_accuracy / mean_accuracy) are derived bit-deterministically
    from the counts so equal counts ⇒ equal floats.
    """

    # The ``counts`` field is a NumPy array, which Pydantic doesn't
    # serialize natively — populated post-validation by the comparator.
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    kind: Literal["confusion_matrix"] = "confusion_matrix"
    n_classes: int = 0
    ignore_label: int | None = None
    # In-memory NxN counts; ``None`` when the object is reconstructed
    # from a divergence-report JSON (only ``counts_sha256`` round-trips).
    counts: np.ndarray | None = Field(default=None, exclude=True)
    # SHA of the ``.npy`` with the integer counts; the float mIoU is
    # derived bit-deterministically from the counts so the SHA alone
    # is the parity carrier.
    counts_sha256: str = ""
    # Optional human-readable class names. Populated for Cityscapes
    # (`["road", "sidewalk", ...]`); empty for unlabeled callers.
    label_set: list[str] = Field(default_factory=list)

    def to_canonical_form(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_classes": self.n_classes,
            "ignore_label": self.ignore_label,
            "counts_sha256": self.counts_sha256,
            "label_set": list(self.label_set),
        }


class StreamingPair(_ArtifactBase):
    """Streaming batch-vs-stream ``Summary.stats`` pair (B3 / ADR-0033).

    Carries the two summary-stats vectors that the streaming comparator
    bit-equality-checks against each other:

    - ``batch_summary`` — stats produced by the reference batch path
      (vernier ``Evaluator.evaluate(...)`` on the same GT/DT pair, or
      pycocotools ``cocoeval.summarize()`` for the vs-naive cell).
    - ``stream_summary`` — stats produced by the streaming path
      (``StreamingEvaluator.update()...finalize()`` per ADR-0013).

    Each summary is a ``stat_<i>`` → float dict; the comparator walks
    the two dicts in lockstep. The runner emits ``stat_<i>`` keys
    positionally (no kernel-name dispatch) so the comparator works
    uniformly across bbox / segm / boundary / keypoints — the order
    is whatever ``Summary.stats`` produces for the configured kernel.
    """

    kind: Literal["streaming_pair"] = "streaming_pair"
    batch_summary: dict[str, float] = Field(default_factory=dict)
    stream_summary: dict[str, float] = Field(default_factory=dict)
    # Optional pointer to the side-by-side RSS curves (informational).
    # The comparator does not gate on this field; reports do.
    rss_curve_paths: dict[str, str] = Field(default_factory=dict)

    def to_canonical_form(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "batch_summary": dict(self.batch_summary),
            "stream_summary": dict(self.stream_summary),
            "rss_curve_paths": dict(self.rss_curve_paths),
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

    The ``iou_type`` parameter is typed ``Metric`` (a superset of
    ``IouType``) so non-instance paradigms can pass their paradigm-
    specific metric name (``"pq"`` for panoptic, ``"miou"`` for
    semantic, the streaming family for B3). Detection cells continue
    to pass an ``IouType`` value unchanged.
    """

    # Non-ClassVar: ``_InstanceComparator`` declares ``paradigm`` as a
    # ClassVar (still accessible on instances, so still satisfies the
    # protocol), while ``_StubComparator`` needs ``paradigm`` set per
    # construction. Both shapes converge on instance-level access.
    paradigm: Paradigm

    def compare(
        self,
        *,
        workload_id: str,
        iou_type: Metric,
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
        iou_type: Metric,
        impl_outputs: dict[str, ComparableArtifact],
    ) -> CellParityReport:
        # ``impl_outputs`` is required by the Comparator protocol but
        # the legacy entry-point passes tensors directly via the
        # constructor; both shapes converge here.
        if iou_type not in _TIER_PAIRS:
            raise ValueError(
                f"instance comparator received non-instance metric {iou_type!r}; "
                f"valid: {sorted(_TIER_PAIRS.keys())}"
            )
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


# Per-bucket scalar fields on ``PanopticSnapshot`` that participate in
# the float comparison. The matching count fields (``n``, ``n_things``,
# ``n_stuff``) are always compared exactly — never gated by tolerance.
_PANOPTIC_FLOAT_FIELDS: tuple[str, ...] = (
    "pq",
    "sq",
    "rq",
    "pq_things",
    "sq_things",
    "rq_things",
    "pq_stuff",
    "sq_stuff",
    "rq_stuff",
)
_PANOPTIC_COUNT_FIELDS: tuple[str, ...] = ("n", "n_things", "n_stuff")
_PANOPTIC_PER_CLASS_METRICS: tuple[str, ...] = ("pq", "sq", "rq")


def _compare_panoptic_pair(
    *,
    tier: Tier,
    impl_a: str,
    impl_b: str,
    snap_a: PanopticSnapshot,
    snap_b: PanopticSnapshot,
    atol: float,
) -> TierResult:
    """Field-by-field comparison of two ``PanopticSnapshot`` instances.

    Adapted from :func:`assert_snapshots_equal` in
    ``tests/python/parity_panoptic/harness.py``: per-bucket
    pq/sq/rq elementwise + per-class dict union; count fields always
    exact regardless of tier; floats compared with ``abs(a-b) <= atol``
    on the strict tier (atol=0 → bit-equality) or aligned tier
    (atol=PANOPTIC_PARITY_EPS).

    The ``TierResult.first_divergence`` slot points at the first
    failing field via a sentinel ``index`` tuple; the field name is
    encoded in the index magnitude so ``divergence_report.json``
    readers can decode which bucket / per-class entry diverged. This
    matches the existing detection comparator's "first index" idiom
    rather than introducing a new shape just for panoptic.
    """
    divergent_count = 0
    first: Divergence | None = None

    canon_a = snap_a.to_canonical_form()
    canon_b = snap_b.to_canonical_form()

    def _record(value_a: float, value_b: float, index_marker: tuple[int, ...]) -> None:
        nonlocal divergent_count, first
        # Strict tier: bit-equality (NaNs in either side would make a
        # finite-diff check pass spuriously). Aligned tier: tolerance.
        diverges = (value_a != value_b) if tier == "strict" else abs(value_a - value_b) > atol
        if not diverges:
            return
        divergent_count += 1
        if first is None:
            first = Divergence(
                index=index_marker,
                value_a=float(value_a),
                value_b=float(value_b),
                abs_diff=float(abs(value_a - value_b)),
            )

    # Bucket scalars (offset 0..n_buckets in the flat index).
    for offset, field in enumerate(_PANOPTIC_FLOAT_FIELDS):
        _record(float(canon_a[field]), float(canon_b[field]), (offset,))
    # Counts: also recorded but always strict regardless of tier.
    for offset, field in enumerate(_PANOPTIC_COUNT_FIELDS, start=len(_PANOPTIC_FLOAT_FIELDS)):
        a_count = int(canon_a[field])
        b_count = int(canon_b[field])
        if a_count != b_count:
            divergent_count += 1
            if first is None:
                first = Divergence(
                    index=(offset,),
                    value_a=float(a_count),
                    value_b=float(b_count),
                    abs_diff=float(abs(a_count - b_count)),
                )

    # Per-class union: keys are stringified ints in the JSON; sort
    # lexicographically — comparing ``"10"`` before ``"2"`` is fine
    # because the divergence index is a positional marker, not a
    # category id (the JSON itself carries the labelled values).
    per_a: dict[str, dict[str, float]] = canon_a["per_class"]
    per_b: dict[str, dict[str, float]] = canon_b["per_class"]
    all_keys = sorted(set(per_a) | set(per_b))
    base = len(_PANOPTIC_FLOAT_FIELDS) + len(_PANOPTIC_COUNT_FIELDS)
    for k_idx, k in enumerate(all_keys):
        row_a = per_a.get(k, {"pq": 0.0, "sq": 0.0, "rq": 0.0})
        row_b = per_b.get(k, {"pq": 0.0, "sq": 0.0, "rq": 0.0})
        for m_idx, metric in enumerate(_PANOPTIC_PER_CLASS_METRICS):
            _record(
                float(row_a.get(metric, 0.0)),
                float(row_b.get(metric, 0.0)),
                (base + k_idx, m_idx),
            )

    return TierResult(
        tier=tier,
        impl_a=impl_a,
        impl_b=impl_b,
        atol=atol,
        passed=divergent_count == 0,
        divergent_count=divergent_count,
        first_divergence=first,
        # Use the snapshot's canonical-form-hash as a stable identifier.
        # 12-char prefix matches the detection comparator's convention.
        tensor_sha256_a=_panoptic_canonical_hash(snap_a)[:12],
        tensor_sha256_b=_panoptic_canonical_hash(snap_b)[:12],
    )


def _panoptic_canonical_hash(snap: PanopticSnapshot) -> str:
    """Stable hash of a snapshot's canonical form. Used as the
    ``tensor_sha256_*`` slot on ``TierResult`` for panoptic so the
    divergence report carries a comparable identifier across reps.

    The hash is over the JSON-serialized canonical form with sorted
    keys; floats are written via the default JSON encoder (matches
    Python's ``repr`` for most values; the eps tier accepts noise
    below ``PANOPTIC_PARITY_EPS`` so cross-rep determinism for the
    canonical form is the relevant invariant, not bit-equal hashes).
    """
    import hashlib
    import json as _json

    canon = snap.to_canonical_form()
    blob = _json.dumps(canon, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


# Panoptic tier table (B1). Only ``vernier_panoptic`` vs ``panopticapi``
# is wired today; Stage 3 may add a strict-vs-aligned split if the
# multi-process oracle path is ever pinned, but per ADR-0025 the
# strict comparison is against ``pq_compute_single_core(proc_id=0,
# ...)`` only.
_PANOPTIC_TIER_PAIRS: tuple[tuple[Tier, str, str, float], ...] = (
    ("strict", "vernier_panoptic", "panopticapi", 0.0),
    ("aligned", "vernier_panoptic", "panopticapi", PANOPTIC_PARITY_EPS),
)


class _PanopticComparator:
    """Panoptic-quality comparator (ADR-0025 + ADR-0033 §B1).

    Consumes ``PanopticSnapshot`` artifacts (the panopticapi-shaped
    All / Things / Stuff buckets + per-class table). The strict tier
    matches ``pq_compute_single_core(proc_id=0, ...)`` bit-exactly per
    the parity contract; the aligned tier permits
    ``PANOPTIC_PARITY_EPS`` per the parity.rs constant.

    Adapts :func:`assert_snapshots_equal` from
    ``tests/python/parity_panoptic/harness.py``: every bucket scalar
    and every per-class row is compared with the per-tier tolerance;
    count fields are always exact.
    """

    paradigm: ClassVar[Paradigm] = "panoptic"

    def compare(
        self,
        *,
        workload_id: str,
        iou_type: Metric,
        impl_outputs: dict[str, ComparableArtifact],
    ) -> CellParityReport:
        if iou_type != "pq":
            raise ValueError(
                f"panoptic comparator received metric {iou_type!r}; "
                f"only 'pq' is supported (ADR-0025)"
            )
        # Narrow each artifact to PanopticSnapshot — the registry promise.
        snapshots: dict[str, PanopticSnapshot] = {}
        for impl, art in impl_outputs.items():
            if not isinstance(art, PanopticSnapshot):
                raise TypeError(
                    f"panoptic comparator expected PanopticSnapshot for impl "
                    f"{impl!r}, got {type(art).__name__}"
                )
            snapshots[impl] = art

        tiers: list[TierResult] = []
        for tier, impl_a, impl_b, atol in _PANOPTIC_TIER_PAIRS:
            if impl_a not in snapshots or impl_b not in snapshots:
                continue
            tiers.append(
                _compare_panoptic_pair(
                    tier=tier,
                    impl_a=impl_a,
                    impl_b=impl_b,
                    snap_a=snapshots[impl_a],
                    snap_b=snapshots[impl_b],
                    atol=atol,
                )
            )
        return CellParityReport(workload_id=workload_id, iou_type=iou_type, tiers=tiers)


# Per-paradigm parity-tier metadata (ADR-0033 §"Comparator registry").
# Mirrors the per-cell `parity_tier` flag the report layer renders so
# reviewers can see at a glance which strict claims are real vs aligned-
# pending-vendoring. ADE20K vs mmseg lands in S3-B and registers
# `("vernier_semantic", "mmseg")` as ``aligned``; the Cityscapes-vs-
# cityscapesScripts pair below is ``strict`` today.
_SEMANTIC_TIER_PAIRS: tuple[tuple[Tier, str, str], ...] = (
    ("strict", "vernier_semantic", "cityscapesscripts"),
)


class _SemanticComparator:
    """Semantic-segmentation comparator (ADR-0033 §B2).

    Strict-tier path: ``np.array_equal(a.counts, b.counts)`` on the
    integer NxN confusion matrices. mIoU / FWIoU / pixel_accuracy /
    mean_accuracy are derived bit-deterministically from the counts,
    so equal counts ⇒ equal floats — checking the integer array is
    necessary and sufficient.

    The comparator's docstring is the canonical place to register the
    future ADE20K-vs-mmseg path: that pair lands as an ``aligned`` tier
    with ``rtol=1e-9`` (mirroring ``SEMANTIC_PARITY_EPS`` in
    ``crates/vernier-semantic/src/parity.rs``) until PR-B6/B7/B8
    vendor mmseg at a pinned SHA, at which point the tier re-grades
    to ``strict``. The flag travels per-cell as ``parity_tier`` on
    the ``TierResult`` so the report layer can flag the gap.
    """

    paradigm: ClassVar[Paradigm] = "semantic"

    def compare(
        self,
        *,
        workload_id: str,
        iou_type: Metric,
        impl_outputs: dict[str, ComparableArtifact],
    ) -> CellParityReport:
        # Filter to ConfusionMatrix entries — defensive; the orchestrator
        # only routes semantic outputs here, but the protocol signature
        # is paradigm-agnostic.
        matrices: dict[str, ConfusionMatrix] = {
            impl: art for impl, art in impl_outputs.items() if isinstance(art, ConfusionMatrix)
        }
        tiers: list[TierResult] = []
        for tier, impl_a, impl_b in _SEMANTIC_TIER_PAIRS:
            if impl_a not in matrices or impl_b not in matrices:
                continue
            tiers.append(_compare_confusion(tier=tier, impl_a=impl_a, impl_b=impl_b, matrices=matrices))
        return CellParityReport(workload_id=workload_id, iou_type=iou_type, tiers=tiers)


def _compare_confusion(
    *,
    tier: Tier,
    impl_a: str,
    impl_b: str,
    matrices: dict[str, ConfusionMatrix],
) -> TierResult:
    """Strict integer-equality between two ``ConfusionMatrix`` artifacts.

    On divergence, the first divergent ``(gt_class, pred_class)`` cell
    coordinate is captured along with the count delta on each side —
    matches the divergent-index pattern the instance comparator uses
    for tensors.
    """
    a = matrices[impl_a]
    b = matrices[impl_b]
    if a.n_classes != b.n_classes:
        raise ValueError(
            f"n_classes mismatch comparing {impl_a} vs {impl_b}: "
            f"{a.n_classes} vs {b.n_classes}"
        )
    counts_a = a.counts
    counts_b = b.counts
    if counts_a is None or counts_b is None:
        # The orchestrator populates ``counts`` after loading the
        # ``.npy`` artifact; receiving ``None`` here is a wiring bug,
        # not a parity divergence. Surface it loudly.
        raise ValueError(
            f"_SemanticComparator requires populated `counts` arrays; "
            f"got counts_a={'set' if counts_a is not None else 'None'}, "
            f"counts_b={'set' if counts_b is not None else 'None'}"
        )
    if counts_a.shape != counts_b.shape:
        raise ValueError(
            f"shape mismatch comparing {impl_a} vs {impl_b}: "
            f"{counts_a.shape} vs {counts_b.shape}"
        )

    divergent_mask = counts_a != counts_b
    divergent_count = int(divergent_mask.sum())
    first_divergence: Divergence | None = None
    if divergent_count > 0:
        first_idx = tuple(int(i) for i in np.argwhere(divergent_mask)[0])
        first_divergence = Divergence(
            index=first_idx,
            value_a=float(counts_a[first_idx]),
            value_b=float(counts_b[first_idx]),
            abs_diff=float(abs(int(counts_a[first_idx]) - int(counts_b[first_idx]))),
        )

    return TierResult(
        tier=tier,
        impl_a=impl_a,
        impl_b=impl_b,
        atol=0.0,
        passed=divergent_count == 0,
        divergent_count=divergent_count,
        first_divergence=first_divergence,
        tensor_sha256_a=a.counts_sha256[:12],
        tensor_sha256_b=b.counts_sha256[:12],
    )


# Streaming comparator parity tolerance — mirrors
# ``tests/python/parity/streaming/test_streaming_finalize_equals_batch.py``'s
# ``pytest.approx(rel=0, abs=1e-12)``. The streaming surface guarantees
# bit-equality up to a sub-ULP wobble that comes from accumulate seeing
# cells in a different (k, a, i) iteration order; ``1e-12`` absorbs
# that without admitting any algorithmic divergence.
STREAMING_PARITY_ATOL: float = 1e-12


class _StreamingComparator:
    """Streaming paradigm comparator (B3 / ADR-0033).

    Three cell shapes route through this one comparator, distinguished
    by which ``StreamingPair`` keys are populated:

    1. **Batch-vs-stream** (the throughput cell's parity gate) — both
       impls produce a ``StreamingPair`` whose ``batch_summary`` and
       ``stream_summary`` come from the same impl (vernier batch via
       ``Evaluator`` + vernier stream via ``StreamingEvaluator``); we
       assert bit-equality between the two within the same artifact.
    2. **DLPack-vs-JSON** — the comparator asserts
       ``batch_summary == stream_summary`` where ``batch_summary`` is
       the JSON ingest path's stats and ``stream_summary`` is the
       array (DLPack) ingest path's stats (per ADR-0030).
    3. **Streaming-vs-naive** — two impls each produce a
       ``StreamingPair``. The parity gate is summary-stats bit-equality
       between the two impls; throughput delta + RSS curves are
       informational and don't gate.

    Throughput and RSS measurements are not gated here; the cell
    metadata (``parity_tier="informational"``) records that. The
    divergence-report shape uses the existing ``Divergence`` /
    ``TierResult`` plumbing for consistency with the instance
    comparator's report.
    """

    paradigm: ClassVar[Paradigm] = "streaming"

    def compare(
        self,
        *,
        workload_id: str,
        iou_type: Metric,
        impl_outputs: dict[str, ComparableArtifact],
    ) -> CellParityReport:
        tiers: list[TierResult] = []
        # 1. Each impl's own batch-vs-stream (or json-vs-array) bit-
        # equality check. Skip silently for impls that ship with only
        # one summary populated (the ``naive_python`` runner, which
        # doesn't have a meaningful ``stream_summary``).
        for impl_name, artifact in impl_outputs.items():
            if not isinstance(artifact, StreamingPair):
                raise ValueError(
                    f"streaming comparator expected StreamingPair for impl "
                    f"{impl_name!r}, got {type(artifact).__name__}"
                )
            if artifact.batch_summary and artifact.stream_summary:
                tiers.append(
                    _compare_streaming_pair_internal(
                        impl=impl_name,
                        artifact=artifact,
                        atol=STREAMING_PARITY_ATOL,
                    )
                )
        # 2. Cross-impl: when two impls participate (vs-naive cell),
        # bit-equal each impl's ``batch_summary`` against the other's
        # (same parity contract as the long-standing detection cell —
        # vernier batch == pycocotools).
        impls = sorted(impl_outputs.keys())
        if len(impls) == 2:
            a_name, b_name = impls
            a_art = impl_outputs[a_name]
            b_art = impl_outputs[b_name]
            assert isinstance(a_art, StreamingPair)
            assert isinstance(b_art, StreamingPair)
            tiers.append(
                _compare_streaming_cross_impl(
                    impl_a=a_name,
                    impl_b=b_name,
                    a_summary=a_art.batch_summary,
                    b_summary=b_art.batch_summary,
                    atol=STREAMING_PARITY_ATOL,
                    tier="aligned",
                )
            )
        return CellParityReport(workload_id=workload_id, iou_type=iou_type, tiers=tiers)


def _stats_dict_to_array(stats: dict[str, float]) -> np.ndarray:
    """Materialize a ``stat_<i>``-keyed dict to a positional ndarray.

    Uses the integer suffix as the axis index; raises if a key isn't of
    the expected ``stat_<i>`` shape — sloppy keys would mask a runner
    bug, so fail loud.
    """
    if not stats:
        return np.empty((0,), dtype=np.float64)
    indexed: list[tuple[int, float]] = []
    for k, v in stats.items():
        if not k.startswith("stat_"):
            raise ValueError(f"streaming summary key {k!r} not in 'stat_<i>' form")
        try:
            i = int(k.removeprefix("stat_"))
        except ValueError as e:
            raise ValueError(f"streaming summary key {k!r} not in 'stat_<i>' form") from e
        indexed.append((i, float(v)))
    indexed.sort()
    return np.asarray([v for _i, v in indexed], dtype=np.float64)


def _compare_streaming_pair_internal(
    *,
    impl: str,
    artifact: StreamingPair,
    atol: float,
) -> TierResult:
    """Bit-equality check between the two halves of a single impl's
    ``StreamingPair`` (batch vs stream, or json vs array).

    Uses ``tier="aligned"`` so ``_compare_pair`` exercises the
    ``diff > atol`` branch (``STREAMING_PARITY_ATOL = 1e-12`` per the
    streaming-vs-batch parity test); ``tier="strict"`` would force
    bit-equality and reject the documented sub-ULP wobble.
    """
    batch = _stats_dict_to_array(artifact.batch_summary)
    stream = _stats_dict_to_array(artifact.stream_summary)
    return _compare_pair(
        tier="aligned",
        impl_a=f"{impl}/batch",
        impl_b=f"{impl}/stream",
        tensor_a=batch,
        tensor_b=stream,
        sha_a="",
        sha_b="",
        atol=atol,
    )


def _compare_streaming_cross_impl(
    *,
    impl_a: str,
    impl_b: str,
    a_summary: dict[str, float],
    b_summary: dict[str, float],
    atol: float,
    tier: Tier,
) -> TierResult:
    """Cross-impl bit-equality on the ``batch_summary`` half."""
    a = _stats_dict_to_array(a_summary)
    b = _stats_dict_to_array(b_summary)
    return _compare_pair(
        tier=tier,
        impl_a=impl_a,
        impl_b=impl_b,
        tensor_a=a,
        tensor_b=b,
        sha_a="",
        sha_b="",
        atol=atol,
    )


_REGISTRY: dict[Paradigm, Comparator] = {
    "instance": _InstanceComparator(impl_tensors={}, impl_sha256={}),  # type: ignore[arg-type]
    "panoptic": _PanopticComparator(),
    "semantic": _SemanticComparator(),
    "streaming": _StreamingComparator(),
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
