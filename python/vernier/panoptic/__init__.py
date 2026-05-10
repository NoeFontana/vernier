"""Panoptic-quality (PQ) evaluation surface (ADR-0025).

Per ADR-0029, the panoptic-segmentation evaluation paradigm lives under
``vernier.panoptic``. Sibling to :mod:`vernier.instance` and
:mod:`vernier.semantic`. The Rust kernel ships in the
``vernier-panoptic`` crate; this module is a thin Python wrapper.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from numpy.typing import NDArray

from vernier._core import (
    BackgroundPanopticEvaluator as BackgroundEvaluator,
)

# Re-export the five distributed-eval exception types under the
# panoptic namespace so callers catching `vernier.panoptic.PartialFormatMismatch`
# match the same class object as `vernier.instance.PartialFormatMismatch`
# (ADR-0032: shared paradigm-agnostic error classes).
from vernier._core import (
    Breakdown,
    ClassPanopticStats,
    PartialDatasetMismatch,
    PartialFormatMismatch,
    PartialParamsMismatch,
    PartialPartitionOverlap,
    PartialRankCollision,
    evaluate_panoptic,
    panoptic_per_class_to_arrow_pycapsule,
)
from vernier._core import (
    PanopticDataset as Dataset,
)
from vernier._core import (
    PanopticPredictions as Predictions,
)
from vernier._core import (
    PanopticSummary as Summary,
)
from vernier._core import (
    evaluate_panoptic_to_partial as _evaluate_panoptic_to_partial,
)
from vernier._core import (
    merge_panoptic_partials as _merge_panoptic_partials,
)
from vernier._tables import arrow_to_dataframe
from vernier._types import (
    CategoryFilter,
    CategoryFilterAll,
    CategoryFilterByGrouping,
    CategoryFilterByIds,
    CategoryFilterFrequency,
    InvalidEvalParams,
    InvalidPanopticParams,
    ParityMode,
    normalize_tables_arg,
)

if TYPE_CHECKING:  # pragma: no cover — type-checker only
    import polars as pl

#: Tables ``Evaluator.evaluate(tables=...)`` accepts on the panoptic
#: paradigm. Per-detection and per-pair tables are instance-only (no
#: IoU-curve matching for panoptic); per-image PQ is deferred.
TableName = Literal["per_class"]
SUPPORTED_TABLES: frozenset[TableName] = frozenset({"per_class"})

__all__ = [
    "BackgroundEvaluator",
    "Breakdown",
    "CategoryFilter",
    "CategoryFilterAll",
    "CategoryFilterByGrouping",
    "CategoryFilterByIds",
    "CategoryFilterFrequency",
    "ClassPanopticStats",
    "Dataset",
    "EvalResult",
    "Evaluator",
    "InvalidEvalParams",
    "InvalidPanopticParams",
    "ParityMode",
    "PartialDatasetMismatch",
    "PartialFormatMismatch",
    "PartialParamsMismatch",
    "PartialPartitionOverlap",
    "PartialRankCollision",
    "Predictions",
    "StuffThingPartition",
    "Summary",
    "TableName",
    "decode_label_map_png",
]


@dataclass(frozen=True)
class EvalResult:
    """Opt-in result of :meth:`Evaluator.evaluate` when ``tables=`` is
    passed. Carries :class:`Summary` plus a polars DataFrame view of
    the per-class panoptic-quality breakdown."""

    summary: Summary
    _per_class_batch: object | None = field(default=None, repr=False)

    @cached_property
    def per_class(self) -> pl.DataFrame:
        """One row per category. Columns: ``category_id``, ``pq``,
        ``sq``, ``rq``, ``n_tp``, ``n_fp``, ``n_fn``, ``iou_sum``."""
        return arrow_to_dataframe(self._per_class_batch, "per_class")


def decode_label_map_png(path: str | Path) -> NDArray[np.uint32]:
    """Decode a panoptic RGB PNG into a `(H, W)` ``uint32`` segment-id
    label map via the ``rgb2id`` convention (panopticapi/evaluation.py
    rgb2id: ``r + 256*g + 256²*b``).

    Lazy-imports Pillow; raises a structured :class:`ImportError` if it
    isn't installed. Three channels are required — a non-RGB PNG is
    rejected with :class:`ValueError`. Single-channel class-id label
    maps (semantic-segmentation) belong in
    :func:`vernier.semantic.Dataset.from_files`, which has its own
    decoder.
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Pillow is required for `vernier.panoptic.decode_label_map_png`; "
            "install via `pip install Pillow` (or include it in your dev "
            "environment)."
        ) from e
    rgb = np.array(Image.open(path), dtype=np.uint32)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(
            f"panoptic PNG {path!s} must be RGB (3 channels); got shape {rgb.shape!r}. "
            f"Single-channel semantic label maps belong in `vernier.semantic`."
        )
    return rgb[:, :, 0] + 256 * rgb[:, :, 1] + 256 * 256 * rgb[:, :, 2]


@dataclass(frozen=True, slots=True)
class StuffThingPartition:
    """User-supplied override of the GT-derived stuff/thing split
    (ADR-0042).

    Setting this on :class:`Evaluator.stuff_thing_partition` overrides
    the dataset's `isthing` flag per category for the purpose of the
    `PQ_St` / `PQ_Th` rollup. ``stuff`` and ``things`` must be disjoint
    and both non-empty (validated at construction). Membership against
    the dataset's category set is checked at ``evaluate()`` time once
    the Dataset is in scope.
    """

    stuff: frozenset[int]
    things: frozenset[int]

    def __post_init__(self) -> None:
        if not self.stuff:
            raise InvalidPanopticParams(
                field="stuff_thing_partition.stuff",
                value=self.stuff,
                remediation="must contain at least one category id (ADR-0042)",
            )
        if not self.things:
            raise InvalidPanopticParams(
                field="stuff_thing_partition.things",
                value=self.things,
                remediation="must contain at least one category id (ADR-0042)",
            )
        overlap = self.stuff & self.things
        if overlap:
            raise InvalidPanopticParams(
                field="stuff_thing_partition",
                value=self,
                remediation=(
                    f"stuff and things must be disjoint; overlap={sorted(overlap)!r} (ADR-0042)"
                ),
            )


def _validate_panoptic_class_filter(cf: CategoryFilter, class_grouping: Breakdown | None) -> None:
    """Validate a ``category_filter`` for the panoptic paradigm
    (ADR-0042).

    ``Frequency`` is rejected (panoptic has no per-class frequency tag
    analogous to LVIS r/c/f). ``ByIds`` requires non-empty unique ids;
    bounds against the dataset's category set are checked at evaluate
    time. ``ByGrouping`` requires that ``class_grouping`` is configured
    and that the named label exists in it.
    """
    if isinstance(cf, CategoryFilterFrequency):
        raise InvalidPanopticParams(
            field="category_filter",
            value=cf,
            remediation=(
                "Frequency is LVIS-only - panoptic has no per-class frequency "
                "tag analogous to r/c/f. Use ByIds or ByGrouping instead "
                "(ADR-0026 lines 178-182, ADR-0042)"
            ),
        )
    if isinstance(cf, CategoryFilterByIds) and len(cf.ids) == 0:
        raise InvalidPanopticParams(
            field="category_filter",
            value=cf,
            remediation="ByIds must contain at least one category id (ADR-0042)",
        )
    if isinstance(cf, CategoryFilterByGrouping):
        if class_grouping is None:
            raise InvalidPanopticParams(
                field="category_filter",
                value=cf,
                remediation=(
                    "ByGrouping requires class_grouping to also be set; "
                    "the label is resolved against the grouping's labels (ADR-0042)"
                ),
            )
        labels = {label for label, _ in class_grouping.class_groups}
        if cf.label not in labels:
            raise InvalidPanopticParams(
                field="category_filter",
                value=cf,
                remediation=(
                    f"ByGrouping label {cf.label!r} is not a label of class_grouping "
                    f"(known labels: {sorted(labels)!r}); ADR-0042"
                ),
            )


def _resolve_category_filter(
    cf: CategoryFilter | None,
    class_grouping: Breakdown | None,
) -> list[int] | None:
    """Resolve an ADR-0042 ``category_filter`` to the kernel's id-list
    form. Mirrors the semantic resolver shape; ``ByGrouping`` looks up
    the label against the active partition."""
    if cf is None or isinstance(cf, CategoryFilterAll):
        return None
    if isinstance(cf, CategoryFilterByIds):
        return sorted(cf.ids)
    if isinstance(cf, CategoryFilterByGrouping):
        assert class_grouping is not None
        for label, ids in class_grouping.class_groups:
            if label == cf.label:
                return list(ids)
        raise InvalidPanopticParams(
            field="category_filter",
            value=cf,
            remediation=f"label {cf.label!r} disappeared from class_grouping",
        )
    raise InvalidPanopticParams(
        field="category_filter",
        value=cf,
        remediation="unsupported CategoryFilter variant",
    )


def _resolve_class_grouping(
    bd: Breakdown | None,
) -> list[tuple[str, list[int]]] | None:
    """Resolve a ``class_grouping`` :class:`Breakdown` to the kernel's
    list-of-pairs form."""
    if bd is None:
        return None
    return [(label, list(ids)) for label, ids in bd.class_groups]


def _resolve_stuff_thing_partition(
    p: StuffThingPartition | None,
) -> tuple[list[int], list[int]] | None:
    """Resolve a :class:`StuffThingPartition` to the FFI's
    ``(stuff_ids, things_ids)`` tuple; ``None`` passes through and the
    kernel uses the dataset's ``isthing`` flags."""
    if p is None:
        return None
    return (sorted(p.stuff), sorted(p.things))


@dataclass(frozen=True, slots=True)
class Evaluator:
    """Panoptic-quality (PQ) evaluator (ADR-0025, ADR-0042).

    Sibling to :class:`vernier.instance.Evaluator`. Per category,
    ``PQ_c`` is computed directly as
    ``iou_c / (TP_c + 0.5*FP_c + 0.5*FN_c)`` (panopticapi form, quirk
    **W1**) — algebraically equal to ``SQ_c * RQ_c`` but f64
    non-associative, so the direct form is what holds bit-equality.
    Global ``PQ`` / ``SQ`` / ``RQ`` are unweighted means over the
    present-categories subset (W2, W3, W7); things / stuff buckets are
    independent unweighted means over their subsets (W4).

    Defaults match panopticapi's ``pq_compute`` shape, except for
    ``parity_mode``, which defaults to ``"corrected"`` (the ADR-0002
    recommendation for net-new users); migrating users wanting
    bit-exact panopticapi behavior should set ``parity_mode="strict"``.

    ``boundary=True`` raises :class:`NotImplementedError`. Boundary PQ
    is deferred to a follow-up ADR (ADR-0025 §"explicitly does not
    decide" Q3 / Z1). The composition rule in the bowenc0221 fork is
    not the instance-case ``min(mask_iou, boundary_iou)`` and resolving
    it requires its own pass.

    The four ADR-0042 fields parameterize the evaluation scope and
    rollup: ``pq_iou_threshold`` overrides the canonical 0.5 PQ match
    threshold (single float, not a ladder); ``category_filter`` and
    ``class_grouping`` mirror the semantic surface (ADR-0041);
    ``stuff_thing_partition`` overrides the dataset-derived
    stuff/thing split for the ``PQ_St`` / ``PQ_Th`` rollup.

    **PR scope cut:** kernel-side plumbing for honoring the four
    custom fields (per_group rollups, threshold-aware matching,
    partition override) lands alongside the ADR-0039 distributed-eval
    phase. Until then, ``evaluate()`` raises
    :class:`NotImplementedError` when any custom field is set; the
    surface — fields, validation, ``StuffThingPartition`` value type —
    is in place.
    """

    parity_mode: ParityMode = "corrected"
    things_stuff_split: bool = True
    boundary: bool = False
    pq_iou_threshold: float | None = None
    category_filter: CategoryFilter | None = None
    class_grouping: Breakdown | None = None
    stuff_thing_partition: StuffThingPartition | None = None

    def __post_init__(self) -> None:
        if self.boundary:
            raise NotImplementedError(
                "boundary panoptic-quality is deferred to a follow-up ADR "
                "(ADR-0025 §'explicitly does not decide' Q3 / Z1). "
                "The composition rule in the bowenc0221 fork is not the "
                "instance-case min(mask_iou, boundary_iou) and resolving it "
                "requires its own pass."
            )
        if self.pq_iou_threshold is not None:
            t = self.pq_iou_threshold
            if not math.isfinite(t):
                raise InvalidPanopticParams(
                    field="pq_iou_threshold",
                    value=t,
                    remediation="must be finite (ADR-0042)",
                )
            if not 0.0 < t <= 1.0:
                raise InvalidPanopticParams(
                    field="pq_iou_threshold",
                    value=t,
                    remediation=(
                        "must lie in (0.0, 1.0]; strict-zero is rejected as a footgun (ADR-0042)"
                    ),
                )
        if self.class_grouping is not None and self.class_grouping.kind != "class_groups":
            raise InvalidPanopticParams(
                field="class_grouping",
                value=self.class_grouping,
                remediation=(
                    "must be a class-groups Breakdown "
                    "(Breakdown.from_class_groups(...)); range Breakdowns "
                    "belong on instance.Evaluator.area_ranges (ADR-0042)"
                ),
            )
        if self.category_filter is not None:
            _validate_panoptic_class_filter(self.category_filter, self.class_grouping)

    def _has_custom_class_params(self) -> bool:
        """``True`` when any ADR-0042 custom field is set."""
        return (
            self.pq_iou_threshold is not None
            or self.category_filter is not None
            or self.class_grouping is not None
            or self.stuff_thing_partition is not None
        )

    @overload
    def evaluate(
        self,
        gt: Dataset,
        dt: Predictions,
        *,
        tables: None = None,
    ) -> Summary: ...

    @overload
    def evaluate(
        self,
        gt: Dataset,
        dt: Predictions,
        *,
        tables: Literal["all"] | tuple[TableName, ...],
    ) -> EvalResult: ...

    def evaluate(
        self,
        gt: Dataset,
        dt: Predictions,
        *,
        tables: Literal["all"] | tuple[TableName, ...] | None = None,
    ) -> Summary | EvalResult:
        """Run the panoptic-quality evaluation.

        ``gt`` and ``dt`` must have been built via
        :meth:`Dataset.from_arrays` / :meth:`Predictions.from_arrays`
        (file-loading helpers ship in a follow-up).

        ``tables=`` is the opt-in keyword for result tables (ADR-0038).
        Defaults to ``None``, returning :class:`Summary` (existing
        behavior, bit-identical to the pre-tables release). Pass
        ``"all"`` or a tuple of :data:`TableName` values to opt into
        the wider :class:`EvalResult` return type.
        """
        # ADR-0042 custom axes resolve Python-side. ByGrouping → ByIds;
        # the kernel's category-filter primitive is id-keyed only.
        resolved_filter = _resolve_category_filter(self.category_filter, self.class_grouping)
        resolved_groups = _resolve_class_grouping(self.class_grouping)
        resolved_partition = _resolve_stuff_thing_partition(self.stuff_thing_partition)
        summary = evaluate_panoptic(
            gt,
            dt,
            self.parity_mode,
            self.things_stuff_split,
            pq_iou_threshold=self.pq_iou_threshold,
            category_filter=resolved_filter,
            class_grouping=resolved_groups,
            stuff_thing_partition=resolved_partition,
        )
        if tables is None:
            return summary
        requested = normalize_tables_arg(tables, SUPPORTED_TABLES)
        per_class_batch = (
            panoptic_per_class_to_arrow_pycapsule(summary) if "per_class" in requested else None
        )
        return EvalResult(summary=summary, _per_class_batch=per_class_batch)

    def evaluate_to_partial(
        self,
        images: Iterable[tuple[int, NDArray[np.uint32], bytes, NDArray[np.uint32], bytes]],
        *,
        categories: bytes,
        rank_id: int,
        retain_per_image_deltas: bool = False,
    ) -> bytes:
        """Run the panoptic evaluation as a per-rank streaming submit
        and return the serialized partial bytes (ADR-0032, ADR-0035).

        ``images`` is an iterable of per-image tuples of the form
        ``(image_id, gt_label_map, gt_segments_info, dt_label_map,
        dt_segments_info)`` — the same shape the streaming substrate's
        ``update`` consumes. The asymmetry with :meth:`evaluate`
        (which takes pre-built :class:`Dataset` / :class:`Predictions`)
        is intentional: ``PanopticDataset`` does not yet expose
        per-image accessors, so the streaming path consumes per-image
        records directly. A future ADR may close the gap by adding
        :class:`Dataset` accessors.

        ``rank_id`` identifies this evaluator's rank in a multi-process
        eval. ``retain_per_image_deltas=True`` is required on every
        rank for strict-mode bit-equality across the merge (ADR-0032
        §"Determinism") at ~2x streaming memory cost.

        The partial bytes can be gathered across ranks and merged on
        the head rank with :meth:`from_partials` to produce a global
        :class:`Summary`.

        Per ADR-0042, raises :class:`InvalidPanopticParams` when any
        of ``pq_iou_threshold`` / ``category_filter`` / ``class_grouping``
        / ``stuff_thing_partition`` is set: extending the ADR-0032
        wire format to carry the resolved custom axes is a follow-up.
        Single-rank custom-params eval works today via :meth:`evaluate`.
        """
        if self._has_custom_class_params():
            raise InvalidPanopticParams(
                field="custom_panoptic_params",
                value=self,
                remediation=(
                    "evaluate_to_partial() with a custom pq_iou_threshold / "
                    "category_filter / class_grouping / stuff_thing_partition "
                    "is deferred to a follow-up to ADR-0042 (extends the "
                    "ADR-0032 wire format to carry the resolved axes). Use "
                    "evaluate(...) for single-rank custom panoptic eval "
                    "today; multi-rank custom params land in the next phase."
                ),
            )
        return _evaluate_panoptic_to_partial(
            list(images),
            categories,
            self.parity_mode,
            rank_id,
            things_stuff_split=self.things_stuff_split,
            retain_per_image_deltas=retain_per_image_deltas,
        )

    @classmethod
    def from_partials(
        cls,
        categories: bytes,
        partials: Sequence[bytes],
        /,
        *,
        parity_mode: ParityMode = "corrected",
        things_stuff_split: bool = True,
        retain_per_image_deltas: bool = False,
    ) -> Summary:
        """Merge ``partials`` (one per rank) into a global :class:`Summary`
        (ADR-0032, ADR-0035).

        ``categories``, ``parity_mode``, ``things_stuff_split``, and
        ``retain_per_image_deltas`` must match what each rank used to
        produce its partial. Mismatches raise the structured
        ``Partial*`` errors re-exported on this module.
        """
        return _merge_panoptic_partials(
            categories,
            list(partials),
            parity_mode,
            things_stuff_split=things_stuff_split,
            retain_per_image_deltas=retain_per_image_deltas,
        )

    def background(
        self,
        categories: bytes,
        *,
        retain_per_image_deltas: bool = False,
        rank_id: int | None = None,
        queue_capacity: int = 8,
        worker_affinity: int | None = None,
        worker_nice: int = 5,
        shutdown_timeout_seconds: float = 5.0,
    ) -> BackgroundEvaluator:
        """Build a :class:`BackgroundEvaluator` (ADR-0014 + ADR-0032)
        that shares this evaluator's ``parity_mode`` and
        ``things_stuff_split``.

        The returned wrapper owns a single dedicated worker thread
        running a :class:`StreamingEvaluator` of the same shape;
        :meth:`BackgroundEvaluator.submit` enqueues per-image
        ``(gt_label_map, gt_segments_info, dt_label_map,
        dt_segments_info)`` tuples and returns immediately. Use this
        when the panoptic-quality kernel measurably stalls the
        training loop — the per-image attribute pass is the dominant
        cost at COCO-panoptic scale.

        ``retain_per_image_deltas=True`` enables strict-mode bit-
        equality across distributed-eval ranks (ADR-0032
        §"Determinism") at ~2x streaming memory cost. The five
        queueing / scheduling knobs mirror
        :class:`vernier.instance.Evaluator.background`.
        """
        return BackgroundEvaluator(
            categories,
            self.parity_mode,
            things_stuff_split=self.things_stuff_split,
            retain_per_image_deltas=retain_per_image_deltas,
            rank_id=rank_id,
            queue_capacity=queue_capacity,
            worker_affinity=worker_affinity,
            worker_nice=worker_nice,
            shutdown_timeout_seconds=shutdown_timeout_seconds,
        )
