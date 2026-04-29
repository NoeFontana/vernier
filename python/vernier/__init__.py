"""vernier: high-performance, parity-preserving COCO-style evaluation.

The Python package is a thin wrapper around the Rust core. Public symbols
documented here are the supported API; anything imported from
:mod:`vernier._core` directly is considered implementation detail and may
change without a deprecation cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Final, NoReturn

from vernier._compat import ParityMode
from vernier._compat import PycocotoolsCOCOeval as COCOeval
from vernier._core import (
    BackgroundEvaluator,
    MemoryBudgetWarning,
    OutOfBudgetError,
    QueueFullError,
    StreamingEvaluator,
    Summary,
    evaluate_bbox_summary,
    evaluate_boundary_summary,
    evaluate_keypoints_summary,
    evaluate_segm_summary,
    version,
)

__all__ = [
    "BackgroundEvaluator",
    "Bbox",
    "Boundary",
    "COCOeval",
    "Evaluator",
    "IouKind",
    "Keypoints",
    "MemoryBudgetWarning",
    "OutOfBudgetError",
    "ParityMode",
    "QueueFullError",
    "Segm",
    "StreamingEvaluator",
    "Summary",
    "__version__",
    "version",
]

__version__: str = version()


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

    dilation_ratio: float = 0.02


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
    """

    iou: IouKind = field(default_factory=Bbox)
    parity_mode: ParityMode = "corrected"
    max_dets: tuple[int, ...] | None = None
    use_cats: bool = True

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
        return replace(self, **kwargs)

    def evaluate(self, gt: bytes, dt: bytes) -> Summary:
        """Run the evaluation pipeline against a GT/DT JSON pair.

        ``gt`` and ``dt`` are the raw COCO JSON payloads as bytes (the
        same shapes pycocotools' ``COCO(...)`` and ``COCO.loadRes(...)``
        consume).
        """
        max_dets_list = self._resolve_max_dets()
        match self.iou:
            case Bbox():
                return evaluate_bbox_summary(gt, dt, self.parity_mode, max_dets_list, self.use_cats)
            case Segm():
                return evaluate_segm_summary(gt, dt, self.parity_mode, max_dets_list, self.use_cats)
            case Boundary(dilation_ratio=r):
                return evaluate_boundary_summary(
                    gt, dt, self.parity_mode, max_dets_list, self.use_cats, r
                )
            case Keypoints(sigmas=s):
                # PyO3's `extract::<Vec<f64>>` accepts iterables; converting
                # tuple -> list at the boundary is the conservative shape
                # (some PyO3 minor versions vary on tuple iteration).
                return evaluate_keypoints_summary(
                    gt,
                    dt,
                    self.parity_mode,
                    max_dets_list,
                    self.use_cats,
                    {cat: list(sigs) for cat, sigs in s.items()},
                )
            case _:
                _reject_unknown_iou(self.iou)


def _reject_unknown_iou(iou: object) -> NoReturn:
    raise TypeError(
        f"unsupported iou kernel {iou!r}; expected Bbox(), Segm(), "
        f"Boundary(...), or Keypoints(...) — see vernier.IouKind"
    )
