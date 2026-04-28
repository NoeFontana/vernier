"""vernier: high-performance, parity-preserving COCO-style evaluation.

The Python package is a thin wrapper around the Rust core. Public symbols
documented here are the supported API; anything imported from
:mod:`vernier._core` directly is considered implementation detail and may
change without a deprecation cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import NoReturn

from vernier._compat import ParityMode
from vernier._compat import PycocotoolsCOCOeval as COCOeval
from vernier._core import (
    Summary,
    evaluate_bbox_summary,
    evaluate_boundary_summary,
    evaluate_segm_summary,
    version,
)

__all__ = [
    "Bbox",
    "Boundary",
    "COCOeval",
    "Evaluator",
    "IouKind",
    "ParityMode",
    "Segm",
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


#: Discriminated union of the kernels :class:`Evaluator` accepts (ADR-0011).
#: Per-kernel parameters live on each variant; pattern-match on
#: :attr:`Evaluator.iou` to dispatch.
IouKind = Bbox | Segm | Boundary


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
    """

    iou: IouKind = field(default_factory=Bbox)
    parity_mode: ParityMode = "corrected"
    max_dets: tuple[int, ...] = (1, 10, 100)
    use_cats: bool = True

    def with_options(
        self,
        *,
        iou: IouKind | None = None,
        parity_mode: ParityMode | None = None,
        max_dets: tuple[int, ...] | None = None,
        use_cats: bool | None = None,
    ) -> Evaluator:
        """Return a copy of this evaluator with the given fields overridden."""
        kwargs: dict[str, object] = {}
        if iou is not None:
            kwargs["iou"] = iou
        if parity_mode is not None:
            kwargs["parity_mode"] = parity_mode
        if max_dets is not None:
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
        max_dets_list = list(self.max_dets)
        match self.iou:
            case Bbox():
                return evaluate_bbox_summary(gt, dt, self.parity_mode, max_dets_list, self.use_cats)
            case Segm():
                return evaluate_segm_summary(gt, dt, self.parity_mode, max_dets_list, self.use_cats)
            case Boundary(dilation_ratio=r):
                return evaluate_boundary_summary(
                    gt, dt, self.parity_mode, max_dets_list, self.use_cats, r
                )
            case _:
                _reject_unknown_iou(self.iou)


def _reject_unknown_iou(iou: object) -> NoReturn:
    raise TypeError(
        f"unsupported iou kernel {iou!r}; expected Bbox(), Segm(), or "
        f"Boundary(...) — see vernier.IouKind"
    )
