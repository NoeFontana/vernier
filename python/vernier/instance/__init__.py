"""Instance-segmentation / detection / keypoints evaluation surface.

Per ADR-0029, the AP-fold evaluation paradigm (bbox, segm, boundary,
keypoints) lives under ``vernier.instance``. Sibling to
:mod:`vernier.panoptic` and :mod:`vernier.semantic`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Final, Literal, NoReturn, overload

from vernier._array_types import RLE, Detections, DetectionsInput
from vernier._confusion import confusion_matrix
from vernier._core import (
    BackgroundEvaluator,
    CocoDataset,
    MemoryBudgetWarning,
    OutOfBudgetError,
    PartialDatasetMismatch,
    PartialFormatMismatch,
    PartialParamsMismatch,
    PartialPartitionOverlap,
    PartialRankCollision,
    QueueFullError,
    Summary,
    evaluate_bbox_grid,
    evaluate_bbox_summary,
    evaluate_bbox_summary_with_dataset,
    evaluate_boundary_grid,
    evaluate_boundary_summary,
    evaluate_boundary_summary_with_dataset,
    evaluate_keypoints_summary,
    evaluate_keypoints_summary_with_dataset,
    evaluate_segm_grid,
    evaluate_segm_summary,
    evaluate_segm_summary_with_dataset,
    per_class_to_arrow_pycapsule,
    per_detection_to_arrow_pycapsule,
    per_image_to_arrow_pycapsule,
    per_pair_to_arrow_pycapsule,
)
from vernier._impl import StreamingEvaluator as _StreamingEvaluator
from vernier._tide import (
    FpIouHistogram,
    TideConfig,
    TideReport,
    error_decomposition,
    fp_iou_histogram,
)
from vernier._types import (
    DEFAULT_DILATION_RATIO,
    EvalResult,
    ParityMode,
    TableName,
    TablesConfig,
    normalize_tables_arg,
)

__all__ = [
    "RLE",
    "BackgroundEvaluator",
    "Bbox",
    "Boundary",
    "CocoDataset",
    "Detections",
    "DetectionsInput",
    "EvalResult",
    "Evaluator",
    "FpIouHistogram",
    "IouKind",
    "Keypoints",
    "MemoryBudgetWarning",
    "OutOfBudgetError",
    "PartialDatasetMismatch",
    "PartialFormatMismatch",
    "PartialParamsMismatch",
    "PartialPartitionOverlap",
    "PartialRankCollision",
    "QueueFullError",
    "Segm",
    "Summary",
    "TableName",
    "TablesConfig",
    "TideConfig",
    "TideReport",
    "confusion_matrix",
    "error_decomposition",
    "fp_iou_histogram",
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
    use_cats: bool = True
    cast_inputs: bool = False

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
        use_cats: bool | None = None,
        cast_inputs: bool | None = None,
    ) -> Evaluator:
        """Return a copy of this evaluator with the given fields overridden.

        ``max_dets`` is three-valued: the default sentinel leaves the
        field unchanged, ``None`` resets to the kernel-canonical ladder,
        and a tuple sets an explicit override.
        """
        kwargs: dict[str, object] = {}
        if iou is not None:
            kwargs["iou"] = iou
        if parity_mode is not None:
            kwargs["parity_mode"] = parity_mode
        if not isinstance(max_dets, _UnsetType):
            kwargs["max_dets"] = max_dets
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
    ) -> Summary: ...

    @overload
    def evaluate(
        self,
        gt: bytes | CocoDataset,
        dt: DetectionsInput,
        *,
        tables: Literal["all"] | tuple[TableName, ...],
        tables_config: TablesConfig | None = None,
    ) -> EvalResult: ...

    def evaluate(
        self,
        gt: bytes | CocoDataset,
        dt: DetectionsInput,
        *,
        tables: Literal["all"] | tuple[TableName, ...] | None = None,
        tables_config: TablesConfig | None = None,
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
        """
        max_dets_list = self._resolve_max_dets()
        if tables is not None:
            return self._evaluate_with_tables(
                gt, dt, max_dets_list, tables, tables_config or TablesConfig()
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

    def _evaluate_with_tables(
        self,
        gt: bytes | CocoDataset,
        dt: DetectionsInput,
        max_dets_list: list[int],
        tables: Literal["all"] | tuple[TableName, ...],
        tables_config: TablesConfig,
    ) -> EvalResult:
        """Tables-enabled evaluate path. Builds the EvalGrid, runs
        accumulate + summarize, then dispatches per-table FFI builders
        for the requested set."""
        requested = normalize_tables_arg(tables)

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
                )
            case Keypoints():
                raise NotImplementedError(
                    "tables= is detection-only in v0.5; keypoints uses a 3-bucket "
                    "area grid that per_image/per_class do not target"
                )
            case _:
                _reject_unknown_iou(self.iou)

        accum = grid.accumulate(max_dets_list)
        summary = accum.summarize(max_dets_list)
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
        return EvalResult(
            summary=summary,
            _per_image_batch=per_image_batch,
            _per_class_batch=per_class_batch,
            _per_detection_batch=per_detection_batch,
            _per_pair_batch=per_pair_batch,
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
        """
        ev = self._build_streaming(gt, rank_id=rank_id)
        ev.update(dt)
        return ev.finalize_to_partial()

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
        merged = _StreamingEvaluator.from_partials(
            gt, partials, **config._streaming_kwargs()
        )
        return merged.finalize()

    def _streaming_kwargs(self) -> dict[str, object]:
        """Translate this evaluator's config into the keyword arguments
        accepted by :class:`vernier._impl.StreamingEvaluator`."""
        max_dets_list = self._resolve_max_dets()
        kwargs: dict[str, object] = {
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

    def _build_streaming(self, gt: bytes, *, rank_id: int | None) -> _StreamingEvaluator:
        kwargs = self._streaming_kwargs()
        if rank_id is not None:
            kwargs["rank_id"] = rank_id
        return _StreamingEvaluator(gt, **kwargs)


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
