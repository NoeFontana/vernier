"""Instance-segmentation / detection / keypoints evaluation surface.

Per ADR-0029, the AP-fold evaluation paradigm (bbox, segm, boundary,
keypoints) lives under ``vernier.instance``. Sibling to
:mod:`vernier.panoptic` and :mod:`vernier.semantic`.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal, NoReturn, TypeAlias, overload

from vernier._array_types import (
    CompressedRLE,
    Detections,
    DetectionsInput,
    RLEInput,
    UncompressedRLE,
)
from vernier._confusion import confusion_matrix
from vernier._core import (
    BackgroundEvaluator,
    Breakdown,
    CocoDataset,
    DimensionMismatchError,
    InvalidAnnotationError,
    InvalidConfigError,
    MemoryBudgetWarning,
    NonFiniteError,
    OutOfBudgetError,
    PartialDatasetMismatch,
    PartialFormatMismatch,
    PartialParamsMismatch,
    PartialPartitionOverlap,
    PartialRankCollision,
    QueueFullError,
    Summary,
    evaluate_bbox_grid,
    evaluate_bbox_partitioned,
    evaluate_bbox_summary,
    evaluate_bbox_summary_with_dataset,
    evaluate_boundary_grid,
    evaluate_boundary_partitioned,
    evaluate_boundary_summary,
    evaluate_boundary_summary_with_dataset,
    evaluate_keypoints_grid,
    evaluate_keypoints_partitioned,
    evaluate_keypoints_summary,
    evaluate_keypoints_summary_with_dataset,
    evaluate_segm_grid,
    evaluate_segm_partitioned,
    evaluate_segm_summary,
    evaluate_segm_summary_with_dataset,
    per_class_to_arrow_pycapsule,
    per_detection_to_arrow_pycapsule,
    per_image_to_arrow_pycapsule,
    per_pair_to_arrow_pycapsule,
)
from vernier._core import (
    cells_from_grid as _cells_from_grid,
)
from vernier._core import (
    evaluate_instance_to_partial as _evaluate_instance_to_partial,
)
from vernier._core import (
    merge_instance_partials as _merge_instance_partials,
)
from vernier._lrp import (
    LrpConfig,
    LrpPerClass,
    LrpReport,
    PartitionedLrpReport,
    optimal_lrp,
)
from vernier._tide import (
    FpIouHistogram,
    TideConfig,
    TideReport,
    error_decomposition,
    fp_iou_histogram,
)
from vernier._types import (
    DEFAULT_DILATION_RATIO,
    SUPPORTED_TABLES,
    CategoryFilter,
    CategoryFilterAll,
    CategoryFilterByGrouping,
    CategoryFilterByIds,
    CategoryFilterFrequency,
    EvalResult,
    IncompatibleSummaryPlan,
    InvalidEvalParams,
    InvalidInstanceParams,
    ParityMode,
    TableName,
    TablesConfig,
    normalize_tables_arg,
)

__all__ = [
    "BackgroundEvaluator",
    "Bbox",
    "Boundary",
    "Breakdown",
    "CategoryFilter",
    "CategoryFilterAll",
    "CategoryFilterByGrouping",
    "CategoryFilterByIds",
    "CategoryFilterFrequency",
    "CocoDataset",
    "CompressedRLE",
    "Detections",
    "DetectionsInput",
    "DimensionMismatchError",
    "EvalResult",
    "Evaluator",
    "FpIouHistogram",
    "IncompatibleSummaryPlan",
    "InvalidAnnotationError",
    "InvalidConfigError",
    "InvalidEvalParams",
    "InvalidInstanceParams",
    "IouKind",
    "Keypoints",
    "LrpConfig",
    "LrpPerClass",
    "LrpReport",
    "Manifest",
    "MemoryBudgetWarning",
    "NonFiniteError",
    "OutOfBudgetError",
    "PartialDatasetMismatch",
    "PartialFormatMismatch",
    "PartialParamsMismatch",
    "PartialPartitionOverlap",
    "PartialRankCollision",
    "PartitionedLrpReport",
    "QueueFullError",
    "RLEInput",
    "Segm",
    "Summary",
    "TableName",
    "TablesConfig",
    "TideConfig",
    "TideReport",
    "UncompressedRLE",
    "confusion_matrix",
    "error_decomposition",
    "fp_iou_histogram",
    "optimal_lrp",
]


@dataclass(frozen=True, slots=True)
class Bbox:
    """Bounding-box IoU kernel selector. No parameters."""


@dataclass(frozen=True, slots=True)
class Segm:
    """Segmentation-mask IoU kernel selector. No parameters."""


@dataclass(frozen=True, slots=True)
class Boundary:
    """Boundary IoU kernel selector (ADR-0010).

    ``dilation_ratio`` is the boundary band width as a fraction of the
    image diagonal. ``0.02`` is the COCO default; ``0.008`` is the LVIS
    variant.
    """

    dilation_ratio: float = DEFAULT_DILATION_RATIO


@dataclass(frozen=True, slots=True)
class Keypoints:
    """OKS (Object Keypoint Similarity) kernel selector (ADR-0012).

    ``sigmas`` maps ``category_id`` -> per-keypoint sigma tuple. An empty
    mapping (the default) uses pycocotools' COCO-person 17-sigma table
    for every category. Per-category overrides honor quirk **F1**
    ("corrected"): pycocotools hard-codes the COCO-person sigmas; vernier
    accepts a per-category mapping while keeping the default byte-identical
    on single-category-person datasets.
    """

    sigmas: Mapping[int, tuple[float, ...]] = field(
        default_factory=lambda: dict[int, tuple[float, ...]](),
    )


#: Discriminated union of the kernels :class:`Evaluator` accepts (ADR-0011).
#: Per-kernel parameters live on each variant; pattern-match on
#: :attr:`Evaluator.iou` to dispatch.
IouKind = Bbox | Segm | Boundary | Keypoints


#: Acceptable shapes for the ``manifest=`` keyword on
#: :meth:`Evaluator.evaluate` (ADR-0046). One of:
#:
#: - a ``dict`` matching the canonical JSON-records shape;
#: - a file path (``str`` or :class:`os.PathLike`) to a ``.json`` manifest;
#: - any object exposing the Arrow PyCapsule Interface
#:   (``__arrow_c_array__`` / ``__arrow_c_stream__``) — a polars,
#:   pandas, pyarrow, or duckdb DataFrame of per-image metadata
#:   passes straight in.
Manifest: TypeAlias = Mapping[str, Any] | str | os.PathLike[str] | Any


def _normalize_cross_axes(
    cross_axes: Sequence[Sequence[str]] | None,
) -> list[list[str]] | None:
    """Coerce ``cross_axes`` into the ``list[list[str]]`` shape the FFI
    consumes. ``None`` passes through unchanged so the kwarg's default
    on the FFI side (no cross product) is what fires."""
    if cross_axes is None:
        return None
    return [list(axes) for axes in cross_axes]


#: Per-kernel canonical ``max_dets`` ladders used when
#: :attr:`Evaluator.max_dets` is left at its sentinel default. Mirrors
#: pycocotools' coupling of summary defaults to the chosen IoU kernel
#: (ADR-0012). The ``Keypoints`` ladder is ``(20,)`` per pycocotools'
#: ``setKpParams``; the other three kernels share the detection ladder.
_KERNEL_MAX_DETS: Final[dict[type[IouKind], tuple[int, ...]]] = {
    Bbox: (1, 10, 100),
    Segm: (1, 10, 100),
    Boundary: (1, 10, 100),
    Keypoints: (20,),
}


class _UnsetType:
    """Singleton sentinel type for ``with_options`` keyword defaults.

    A dedicated class — rather than ``object()`` — lets pyright narrow
    on ``isinstance(arg, _UnsetType)`` cleanly without typing the
    parameter as ``Any``. Mirrors the pattern used by
    :data:`dataclasses.MISSING` and :data:`typing.NoDefault`.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<UNSET>"


_UNSET: Final[_UnsetType] = _UnsetType()


def _first_custom_grid_field_and_value(ev: Evaluator) -> tuple[str, object] | None:
    """Return ``(name, value)`` for the first custom-grid field set in
    field-declaration order, or ``None`` if none are set. Used by the
    :class:`IncompatibleSummaryPlan` redirect to point at *one* offending
    field even when multiple are set."""
    if ev.iou_thresholds is not None:
        return ("iou_thresholds", ev.iou_thresholds)
    if ev.recall_thresholds is not None:
        return ("recall_thresholds", ev.recall_thresholds)
    if ev.area_ranges is not None:
        return ("area_ranges", ev.area_ranges)
    return None


_INCOMPATIBLE_SUMMARY_REMEDIATION: Final[str] = (
    "the canonical summary plans are keyed on hardcoded slot indices in the "
    "(T, R, K, A, M) tensor — AP_S is 'the second area-bucket entry of the "
    "all-IoU slice at maxDet=100', not 'the small-area slot'. Your custom "
    "grid breaks this index assumption. Use evaluate_tables(...) for tabular "
    "output that carries explicit labels per row, or remove the custom grid "
    "(set the field back to None) to use evaluate(). See ADR-0040."
)


def _validate_threshold_ladder(ladder: tuple[float, ...], *, field_name: str) -> None:
    """Validate an IoU or recall threshold ladder per ADR-0040.

    Non-empty, every element finite and in ``[0.0, 1.0]``, sorted
    ascending, no duplicates. Raises :class:`InvalidInstanceParams`
    with a remediation pointer on any violation.
    """
    if len(ladder) == 0:
        raise InvalidInstanceParams(
            field=field_name,
            value=ladder,
            remediation="must be non-empty (ADR-0040). Default canonical ladder applies when None.",
        )
    prev: float | None = None
    seen: set[float] = set()
    for v in ladder:
        if not math.isfinite(v):
            raise InvalidInstanceParams(
                field=field_name,
                value=ladder,
                remediation=f"every element must be finite (got {v!r}; ADR-0040)",
            )
        if not 0.0 <= v <= 1.0:
            raise InvalidInstanceParams(
                field=field_name,
                value=ladder,
                remediation=f"every element must lie in [0.0, 1.0] (got {v!r}; ADR-0040)",
            )
        if v in seen:
            raise InvalidInstanceParams(
                field=field_name,
                value=ladder,
                remediation=f"duplicate value {v!r}; ADR-0040 requires distinct entries",
            )
        if prev is not None and v < prev:
            raise InvalidInstanceParams(
                field=field_name,
                value=ladder,
                remediation="must be sorted ascending; ADR-0040",
            )
        seen.add(v)
        prev = v


def _validate_area_ranges_breakdown(bd: Breakdown) -> None:
    """Validate the ``area_ranges`` Breakdown per ADR-0040.

    Must be a range-keyed Breakdown (built via
    :meth:`Breakdown.from_ranges`); class-group breakdowns are rejected.
    The Breakdown's own constructor already enforced
    non-empty / finite / lo <= hi / unique-label invariants — this
    function only adds the variant-shape check.
    """
    if bd.kind != "range":
        raise InvalidInstanceParams(
            field="area_ranges",
            value=bd,
            remediation=(
                "must be a range Breakdown (Breakdown.from_ranges(...)); "
                "class-group Breakdowns belong on semantic / panoptic class_grouping (ADR-0040)"
            ),
        )


@dataclass(frozen=True, slots=True)
class Evaluator:
    """Extended-API COCO-style evaluator.

    The instance is immutable per ADR-0006: construct once, call
    :meth:`evaluate` per dataset/detections pair. To change a parameter,
    use :meth:`with_options` (which returns a new evaluator).

    Defaults match pycocotools' detection eval grid, except for
    ``parity_mode``, which defaults to ``"corrected"`` (the ADR-0002
    recommendation for net-new users); migrating users wanting bit-exact
    pycocotools behavior should set ``parity_mode="strict"``.

    The ``iou`` field is a discriminated dataclass union (:data:`IouKind`);
    each variant carries its own kernel-specific parameters (per ADR-0011).
    Use ``Bbox()`` / ``Segm()`` / ``Boundary(dilation_ratio=...)``.

    ``max_dets`` defaults to ``None``, meaning "use the canonical ladder
    for the selected ``iou`` kernel" (ADR-0012). Resolution happens at
    dispatch via :data:`_KERNEL_MAX_DETS`; explicit values always win.
    The current three kernels all resolve to ``(1, 10, 100)``.

    ``cast_inputs`` (ADR-0030) gates one-shot ``f32→f64`` / ``i32→i64``
    promotion when array-form ``Detections`` are passed to
    :meth:`evaluate`; off by default to preserve the strict ADR-0004
    boundary. JSON-bytes detections ignore this flag.
    """

    iou: IouKind = field(default_factory=Bbox)
    parity_mode: ParityMode = "corrected"
    max_dets: tuple[int, ...] | None = None
    iou_thresholds: tuple[float, ...] | None = None
    recall_thresholds: tuple[float, ...] | None = None
    area_ranges: Breakdown | None = None
    use_cats: bool = True
    cast_inputs: bool = False

    def __post_init__(self) -> None:
        if self.iou_thresholds is not None:
            _validate_threshold_ladder(self.iou_thresholds, field_name="iou_thresholds")
        if self.recall_thresholds is not None:
            _validate_threshold_ladder(self.recall_thresholds, field_name="recall_thresholds")
        if self.area_ranges is not None:
            _validate_area_ranges_breakdown(self.area_ranges)

    def _has_custom_grid(self) -> bool:
        """``True`` when any of the ADR-0040 custom-grid fields is set."""
        return _first_custom_grid_field_and_value(self) is not None

    def _resolve_max_dets(self) -> list[int]:
        """Materialize the effective ``max_dets`` ladder for this evaluator.

        Falls back to an empty ladder when ``iou`` is an unrecognized
        type so the dispatch ``case _:`` arm in :meth:`evaluate` can
        surface the friendly :class:`TypeError` instead of a ``KeyError``.
        """
        explicit = self.max_dets
        if explicit is not None:
            return list(explicit)
        return list(_KERNEL_MAX_DETS.get(type(self.iou), ()))

    def with_options(
        self,
        *,
        iou: IouKind | None = None,
        parity_mode: ParityMode | None = None,
        max_dets: tuple[int, ...] | None | _UnsetType = _UNSET,
        iou_thresholds: tuple[float, ...] | None | _UnsetType = _UNSET,
        recall_thresholds: tuple[float, ...] | None | _UnsetType = _UNSET,
        area_ranges: Breakdown | None | _UnsetType = _UNSET,
        use_cats: bool | None = None,
        cast_inputs: bool | None = None,
    ) -> Evaluator:
        """Return a copy of this evaluator with the given fields overridden.

        Sentinel-keyed fields (``max_dets``, ``iou_thresholds``,
        ``recall_thresholds``, ``area_ranges``) are three-valued:
        the default ``_UNSET`` leaves the field unchanged, ``None``
        resets to the kernel-canonical default, and a value sets an
        explicit override.
        """
        kwargs: dict[str, object] = {}
        if iou is not None:
            kwargs["iou"] = iou
        if parity_mode is not None:
            kwargs["parity_mode"] = parity_mode
        if not isinstance(max_dets, _UnsetType):
            kwargs["max_dets"] = max_dets
        if not isinstance(iou_thresholds, _UnsetType):
            kwargs["iou_thresholds"] = iou_thresholds
        if not isinstance(recall_thresholds, _UnsetType):
            kwargs["recall_thresholds"] = recall_thresholds
        if not isinstance(area_ranges, _UnsetType):
            kwargs["area_ranges"] = area_ranges
        if use_cats is not None:
            kwargs["use_cats"] = use_cats
        if cast_inputs is not None:
            kwargs["cast_inputs"] = cast_inputs
        return replace(self, **kwargs)

    @overload
    def evaluate(
        self,
        gt: bytes | CocoDataset,
        dt: DetectionsInput,
        *,
        tables: None = None,
        tables_config: TablesConfig | None = None,
        manifest: None = None,
        cross_axes: None = None,
        calibration: Literal[False] = False,
    ) -> Summary: ...

    @overload
    def evaluate(
        self,
        gt: bytes | CocoDataset,
        dt: DetectionsInput,
        *,
        tables: Literal["all"] | tuple[TableName, ...],
        tables_config: TablesConfig | None = None,
        manifest: None = None,
        cross_axes: None = None,
        calibration: bool = False,
    ) -> EvalResult: ...

    @overload
    def evaluate(
        self,
        gt: bytes | CocoDataset,
        dt: DetectionsInput,
        *,
        tables: None = None,
        tables_config: TablesConfig | None = None,
        manifest: Manifest,
        cross_axes: Sequence[Sequence[str]] | None = None,
        calibration: Literal[False] = False,
    ) -> EvalResult: ...

    @overload
    def evaluate(
        self,
        gt: bytes | CocoDataset,
        dt: DetectionsInput,
        *,
        tables: None = None,
        tables_config: TablesConfig | None = None,
        manifest: None = None,
        cross_axes: None = None,
        calibration: Literal[True],
    ) -> EvalResult: ...

    def evaluate(
        self,
        gt: bytes | CocoDataset,
        dt: DetectionsInput,
        *,
        tables: Literal["all"] | tuple[TableName, ...] | None = None,
        tables_config: TablesConfig | None = None,
        manifest: Manifest | None = None,
        cross_axes: Sequence[Sequence[str]] | None = None,
        calibration: bool = False,
    ) -> Summary | EvalResult:
        """Run the evaluation pipeline against a GT/DT pair.

        ``dt`` accepts the COCO ``loadRes``-shaped JSON payload as
        ``bytes``, **or** the array-form ``Detections`` shapes
        introduced by ADR-0030 (a single per-image dict or a sequence
        of them). The array path skips JSON serialization end-to-end
        and reads NumPy / DLPack buffers directly into the kernel.

        ``gt`` is either the GT JSON bytes (parse-and-discard, identical
        to prior behavior) or a :class:`CocoDataset` handle (parsed-once,
        with the cache reused across calls — see ADR-0020).

        ``tables=`` is the opt-in keyword for result tables. Defaults
        to ``None``, returning :class:`Summary` (existing behavior,
        bit-identical to 0.0.1). Pass ``"all"`` or a tuple of
        :data:`TableName`\\ s to opt into the wider :class:`EvalResult`
        return type.

        ``manifest=`` opts into ADR-0046 partitioned eval. Accepts a
        dict (the canonical JSON-records shape), a file path
        (``.json``), or any object exposing the Arrow PyCapsule
        Interface (a polars / pandas / pyarrow DataFrame of per-image
        metadata). Returns an :class:`EvalResult` whose ``.summary``
        is bit-identical to the un-partitioned call and whose
        ``.slices`` property is a polars DataFrame with one row per
        ``(axis, value)`` cell. ``cross_axes=`` opts joint cells in
        (per ADR-0046 §E2; marginals are the default).

        ``calibration=`` opts into ADR-0018 detection-family calibration.
        When ``True``, the per-image cell store is retained on the
        returned :class:`EvalResult` (as ``_eval_cells``) and
        :meth:`EvalResult.calibration` becomes available; the canonical
        ``tables=None`` fast path is upgraded from :class:`Summary` to
        :class:`EvalResult` to carry the handle. Not currently supported
        on the ``manifest=`` partitioned path.

        Per ADR-0040, raises :class:`IncompatibleSummaryPlan` when
        ``iou_thresholds`` / ``recall_thresholds`` / ``area_ranges``
        is set explicitly: the canonical 12-stat summary plan is keyed
        on hardcoded slot indices that don't generalize. Use
        :meth:`evaluate_tables` for tabular output that carries
        explicit labels per row.
        """
        offender = _first_custom_grid_field_and_value(self)
        if offender is not None:
            field_name, value = offender
            raise IncompatibleSummaryPlan(
                field=field_name,
                value=value,
                plan="COCO 12-stat / keypoints 10-stat / LVIS 13-stat",
                remediation=_INCOMPATIBLE_SUMMARY_REMEDIATION,
            )
        max_dets_list = self._resolve_max_dets()
        if manifest is not None:
            if tables is not None:
                raise ValueError(
                    "tables= and manifest= cannot be combined. The partitioned eval "
                    "returns headline metrics per slice on `result.slices`; for the "
                    "per-class by per-slice cross product, run per_class evaluate once "
                    "per slice with a filtered detection set — see "
                    "docs/how-to/per-class-by-slice.md."
                )
            if calibration:
                raise ValueError(
                    "calibration=True and manifest= cannot be combined; "
                    "calibration over the per-image cell store is not yet wired on "
                    "the ADR-0046 partitioned path. Run un-partitioned with "
                    "calibration=True to fold the full-dataset cells."
                )
            if isinstance(gt, CocoDataset):
                raise NotImplementedError(
                    "manifest= currently requires GT JSON bytes; CocoDataset handles "
                    "on the partitioned path are a follow-up."
                )
            return self._evaluate_partitioned(gt, dt, max_dets_list, manifest, cross_axes)
        if tables is not None or calibration:
            return self._evaluate_with_tables(
                gt,
                dt,
                max_dets_list,
                tables,
                tables_config or TablesConfig(),
                calibration=calibration,
            )
        if isinstance(gt, CocoDataset):
            return self._evaluate_with_dataset(gt, dt, max_dets_list)
        match self.iou:
            case Bbox():
                return evaluate_bbox_summary(
                    gt, dt, self.parity_mode, max_dets_list, self.use_cats, self.cast_inputs
                )
            case Segm():
                return evaluate_segm_summary(
                    gt, dt, self.parity_mode, max_dets_list, self.use_cats, self.cast_inputs
                )
            case Boundary(dilation_ratio=r):
                return evaluate_boundary_summary(
                    gt, dt, self.parity_mode, max_dets_list, self.use_cats, r, self.cast_inputs
                )
            case Keypoints(sigmas=s):
                return evaluate_keypoints_summary(
                    gt,
                    dt,
                    self.parity_mode,
                    max_dets_list,
                    self.use_cats,
                    _normalize_sigmas(s),
                    self.cast_inputs,
                )
            case _:
                _reject_unknown_iou(self.iou)

    def _evaluate_partitioned(
        self,
        gt: bytes,
        dt: DetectionsInput,
        max_dets_list: list[int],
        manifest: Manifest,
        cross_axes: Sequence[Sequence[str]] | None,
    ) -> EvalResult:
        """ADR-0046 partitioned eval dispatch. Routes the user's
        manifest input through the kernel-specific FFI entry, then
        wraps the resulting `PartitionedSummary` in an `EvalResult`
        carrying the slices RecordBatch."""
        cross = _normalize_cross_axes(cross_axes)
        match self.iou:
            case Bbox():
                psum = evaluate_bbox_partitioned(
                    gt,
                    dt,
                    self.parity_mode,
                    max_dets_list[-1],
                    self.use_cats,
                    manifest,
                    self.cast_inputs,
                    cross_axes=cross,
                )
            case Segm():
                psum = evaluate_segm_partitioned(
                    gt,
                    dt,
                    self.parity_mode,
                    max_dets_list[-1],
                    self.use_cats,
                    manifest,
                    self.cast_inputs,
                    cross_axes=cross,
                )
            case Boundary(dilation_ratio=r):
                psum = evaluate_boundary_partitioned(
                    gt,
                    dt,
                    self.parity_mode,
                    max_dets_list[-1],
                    self.use_cats,
                    r,
                    manifest,
                    self.cast_inputs,
                    cross_axes=cross,
                )
            case Keypoints(sigmas=s):
                psum = evaluate_keypoints_partitioned(
                    gt,
                    dt,
                    self.parity_mode,
                    max_dets_list[-1],
                    self.use_cats,
                    _normalize_sigmas(s),
                    manifest,
                    self.cast_inputs,
                    cross_axes=cross,
                )
            case _:
                _reject_unknown_iou(self.iou)
        slices_batch = psum.slices_capsule()
        return EvalResult(
            summary=psum.overall,
            _slices_batch=slices_batch,
            overall_n_images=int(psum.overall_n_images),
            overall_n_detections=int(psum.overall_n_detections),
        )

    def evaluate_tables(
        self,
        gt: bytes | CocoDataset,
        dt: DetectionsInput,
        *,
        tables: Literal["all"] | tuple[TableName, ...] = "all",
        tables_config: TablesConfig | None = None,
    ) -> EvalResult:
        """Tables-only evaluate path (ADR-0040 redirect target).

        Equivalent to :meth:`evaluate` with ``tables=`` set, but
        bypasses the :class:`IncompatibleSummaryPlan` redirect so
        custom-grid users can reach the result tables. Honors
        ``iou_thresholds`` / ``recall_thresholds`` / ``area_ranges``
        when set, falling through to the canonical COCO grid otherwise.
        """
        max_dets_list = self._resolve_max_dets()
        return self._evaluate_with_tables(
            gt, dt, max_dets_list, tables, tables_config or TablesConfig()
        )

    def _evaluate_with_tables(
        self,
        gt: bytes | CocoDataset,
        dt: DetectionsInput,
        max_dets_list: list[int],
        tables: Literal["all"] | tuple[TableName, ...] | None,
        tables_config: TablesConfig,
        *,
        calibration: bool = False,
    ) -> EvalResult:
        """Tables-enabled evaluate path. Builds the EvalGrid, runs
        accumulate + summarize, then dispatches per-table FFI builders
        for the requested set.

        ``calibration=True`` additionally materializes an
        :class:`vernier._core.EvalCells` handle off the grid and stashes
        it on the returned :class:`EvalResult` for the ADR-0018
        calibration fold. With both ``tables=None`` and
        ``calibration=True`` we still go through this path so the grid
        is available; only the per-table FFI calls are short-circuited.
        """
        requested: set[TableName] = (
            normalize_tables_arg(tables, SUPPORTED_TABLES) if tables is not None else set()
        )

        # The tables= path needs JSON bytes today; pre-parsed CocoDataset
        # handles aren't threaded through yet.
        if isinstance(gt, CocoDataset):
            raise NotImplementedError(
                "tables= path requires GT JSON bytes; CocoDataset handles are not "
                "yet supported on this path"
            )

        # per_detection (best_iou) and per_pair require the spine to
        # retain its IoU matrices.
        need_retention = bool(requested & {"per_detection", "per_pair"})

        # ADR-0040 custom-grid axes (None → kernel canonical, resolved
        # in `vernier_ffi::resolve_grid_axes`).
        custom_iou = None if self.iou_thresholds is None else list(self.iou_thresholds)
        custom_recall = None if self.recall_thresholds is None else list(self.recall_thresholds)
        custom_areas = self.area_ranges

        match self.iou:
            case Bbox():
                grid = evaluate_bbox_grid(
                    gt,
                    dt,
                    self.parity_mode,
                    max_dets_list[-1],
                    self.use_cats,
                    need_retention,
                    self.cast_inputs,
                    iou_thresholds=custom_iou,
                    recall_thresholds=custom_recall,
                    area_ranges=custom_areas,
                )
            case Segm():
                grid = evaluate_segm_grid(
                    gt,
                    dt,
                    self.parity_mode,
                    max_dets_list[-1],
                    self.use_cats,
                    need_retention,
                    self.cast_inputs,
                    iou_thresholds=custom_iou,
                    recall_thresholds=custom_recall,
                    area_ranges=custom_areas,
                )
            case Boundary(dilation_ratio=r):
                grid = evaluate_boundary_grid(
                    gt,
                    dt,
                    self.parity_mode,
                    max_dets_list[-1],
                    self.use_cats,
                    r,
                    need_retention,
                    self.cast_inputs,
                    iou_thresholds=custom_iou,
                    recall_thresholds=custom_recall,
                    area_ranges=custom_areas,
                )
            case Keypoints(sigmas=s):
                if requested:
                    raise NotImplementedError(
                        "tables= is detection-only in v0.5; keypoints uses a 3-bucket "
                        "area grid that per_image/per_class do not target"
                    )
                grid = evaluate_keypoints_grid(
                    gt,
                    dt,
                    self.parity_mode,
                    max_dets_list[-1],
                    self.use_cats,
                    _normalize_sigmas(s),
                    self.cast_inputs,
                    iou_thresholds=custom_iou,
                    recall_thresholds=custom_recall,
                    area_ranges=custom_areas,
                )
            case _:
                _reject_unknown_iou(self.iou)

        accum = grid.accumulate(max_dets_list)
        # ADR-0040: the canonical 12-stat detection summary is keyed on
        # hardcoded slot indices (AP, AP_50, AP_75, AP_S, ...) that
        # assume the canonical grid; pairing it with a user-defined
        # grid would silently misindex. evaluate_tables() with a
        # custom grid emits per-axis tables and `summary=None`.
        summary = None if self._has_custom_grid() else accum.summarize(max_dets_list)
        # `evaluate_*_grid` parsed `gt` once and retained the dataset on
        # the grid; reuse instead of paying a second JSON parse.
        dataset = grid.dataset()

        per_image_batch = (
            per_image_to_arrow_pycapsule(grid, dataset) if "per_image" in requested else None
        )
        per_class_batch = (
            per_class_to_arrow_pycapsule(grid, accum, dataset) if "per_class" in requested else None
        )
        per_detection_batch = (
            per_detection_to_arrow_pycapsule(grid, tables_config.per_detection_with_geometry)
            if "per_detection" in requested
            else None
        )
        per_pair_batch = (
            per_pair_to_arrow_pycapsule(
                grid,
                tables_config.per_pair_iou_floor,
                tables_config.per_pair_max_rows,
            )
            if "per_pair" in requested
            else None
        )
        eval_cells = _cells_from_grid(grid) if calibration else None
        return EvalResult(
            summary=summary,
            _per_image_batch=per_image_batch,
            _per_class_batch=per_class_batch,
            _per_detection_batch=per_detection_batch,
            _per_pair_batch=per_pair_batch,
            _eval_cells=eval_cells,
        )

    def _evaluate_with_dataset(
        self, gt: CocoDataset, dt: DetectionsInput, max_dets_list: list[int]
    ) -> Summary:
        match self.iou:
            case Bbox():
                return evaluate_bbox_summary_with_dataset(
                    gt, dt, self.parity_mode, max_dets_list, self.use_cats, self.cast_inputs
                )
            case Segm():
                return evaluate_segm_summary_with_dataset(
                    gt, dt, self.parity_mode, max_dets_list, self.use_cats, self.cast_inputs
                )
            case Boundary(dilation_ratio=r):
                return evaluate_boundary_summary_with_dataset(
                    gt, dt, self.parity_mode, max_dets_list, self.use_cats, r, self.cast_inputs
                )
            case Keypoints(sigmas=s):
                return evaluate_keypoints_summary_with_dataset(
                    gt,
                    dt,
                    self.parity_mode,
                    max_dets_list,
                    self.use_cats,
                    _normalize_sigmas(s),
                    self.cast_inputs,
                )
            case _:
                _reject_unknown_iou(self.iou)

    def evaluate_to_partial(
        self,
        gt: bytes,
        dt: DetectionsInput,
        *,
        rank_id: int,
    ) -> bytes:
        """Run the evaluation as a per-rank streaming submit and return
        the serialized partial bytes (ADR-0031, ADR-0035).

        ``rank_id`` identifies this evaluator's rank in a multi-process
        eval. The partial bytes can be gathered across ranks (e.g. via
        ``torch.distributed.all_gather_object``) and merged on the head
        rank with :meth:`from_partials` to produce a global Summary
        bit-equal to a batch :meth:`evaluate` over the union (in
        ``parity_mode="strict"`` once the ``(score, rank_id,
        local_position)`` tiebreak lands; under ADR-0004's 4-ULP
        envelope today).

        Per ADR-0040, raises :class:`InvalidInstanceParams` when any
        of ``iou_thresholds`` / ``recall_thresholds`` / ``area_ranges``
        is set: extending the ADR-0031 wire format to carry the
        resolved custom grid + bumping ``params_hash`` to cover the
        new fields is a follow-up. Batch :meth:`evaluate_tables`
        already honors the custom grid; pair it with a single-rank
        run until the streaming follow-up ships.
        """
        offender = _first_custom_grid_field_and_value(self)
        if offender is not None:
            field_name, value = offender
            raise InvalidInstanceParams(
                field=field_name,
                value=value,
                remediation=(
                    "evaluate_to_partial() with a custom iou_thresholds / "
                    "recall_thresholds / area_ranges grid is deferred to a "
                    "follow-up to ADR-0040 (extends the ADR-0031 wire format "
                    "and the params_hash hash to cover the resolved custom "
                    "grid). Use evaluate_tables(...) for single-rank custom "
                    "grids today; multi-rank custom grids land in the next "
                    "phase."
                ),
            )
        kwargs = self._streaming_kwargs()
        iou_type = kwargs.pop("iou_type")
        return _evaluate_instance_to_partial(gt, dt, iou_type, rank_id, **kwargs)

    @classmethod
    def from_partials(
        cls,
        gt: bytes,
        partials: Sequence[bytes],
        /,
        *,
        iou: IouKind | None = None,
        parity_mode: ParityMode = "corrected",
        max_dets: tuple[int, ...] | None = None,
        use_cats: bool = True,
        cast_inputs: bool = False,
    ) -> Summary:
        """Merge ``partials`` (one per rank) into a global :class:`Summary`
        (ADR-0031, ADR-0035).

        The kwargs mirror :class:`Evaluator`'s config fields and must
        match what each rank used to produce its partial. Mismatches
        raise the structured ``Partial*`` errors (re-exported on this
        module).
        """
        config = cls(
            iou=iou if iou is not None else Bbox(),
            parity_mode=parity_mode,
            max_dets=max_dets,
            use_cats=use_cats,
            cast_inputs=cast_inputs,
        )
        kwargs = config._streaming_kwargs()
        iou_type = kwargs.pop("iou_type")
        return _merge_instance_partials(gt, list(partials), iou_type, **kwargs)

    def _streaming_kwargs(self) -> dict[str, Any]:
        # iou/dilation_ratio/sigmas live in this dict because the FFI
        # constructor takes a stringly-typed `iou_type=` plus discriminated
        # extras, not the typed IouKind union. Bridging is owned here so
        # callers stay typed.
        max_dets_list = self._resolve_max_dets()
        kwargs: dict[str, Any] = {
            "parity_mode": self.parity_mode,
            "max_dets": max_dets_list,
            "use_cats": self.use_cats,
            "cast_inputs": self.cast_inputs,
        }
        match self.iou:
            case Bbox():
                kwargs["iou_type"] = "bbox"
            case Segm():
                kwargs["iou_type"] = "segm"
            case Boundary(dilation_ratio=r):
                kwargs["iou_type"] = "boundary"
                kwargs["dilation_ratio"] = r
            case Keypoints(sigmas=s):
                kwargs["iou_type"] = "keypoints"
                kwargs["sigmas"] = _normalize_sigmas(s)
            case _:
                _reject_unknown_iou(self.iou)
        return kwargs

    def background(
        self,
        gt: bytes | CocoDataset,
        *,
        memory_budget_bytes: int | None = None,
        queue_capacity: int = 8,
        worker_affinity: int | None = None,
        worker_nice: int = 5,
        shutdown_timeout_seconds: float = 5.0,
        retain_iou: bool = False,
        rank_id: int | None = None,
        record_latency_samples: bool = False,
    ) -> BackgroundEvaluator:
        """Build a :class:`BackgroundEvaluator` (ADR-0014, ADR-0020) that
        shares this evaluator's ``iou``, ``parity_mode``, ``max_dets``,
        ``use_cats``, and ``cast_inputs``.

        Passing a :class:`CocoDataset` for ``gt`` reuses the parsed-once
        handle's per-kernel GT-side derivation caches across every
        ``submit()`` round (ADR-0020). For boundary IoU this collapses
        the dominant per-epoch cost — building the GT band per
        annotation — from O(epochs) to O(1). Bbox and keypoints have
        no GT-side cache today, so the win there is just the JSON
        parse; segm sits between.

        The five queueing / scheduling knobs mirror the keyword-only
        parameters on :class:`BackgroundEvaluator`'s constructor.
        """
        kwargs = self._streaming_kwargs()
        return BackgroundEvaluator(
            gt,
            memory_budget_bytes=memory_budget_bytes,
            queue_capacity=queue_capacity,
            worker_affinity=worker_affinity,
            worker_nice=worker_nice,
            shutdown_timeout_seconds=shutdown_timeout_seconds,
            retain_iou=retain_iou,
            rank_id=rank_id,
            record_latency_samples=record_latency_samples,
            **kwargs,
        )


def _normalize_sigmas(
    sigmas: Mapping[int, tuple[float, ...]],
) -> dict[int, list[float]]:
    # PyO3's `extract::<Vec<f64>>` accepts iterables, but tuple iteration
    # has varied across minor versions; convert to list at the boundary.
    return {cat: list(sigs) for cat, sigs in sigmas.items()}


def _reject_unknown_iou(iou: object) -> NoReturn:
    raise TypeError(
        f"unsupported iou kernel {iou!r}; expected Bbox(), Segm(), "
        f"Boundary(...), or Keypoints(...) — see vernier.instance.IouKind"
    )
