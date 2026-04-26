"""vernier: high-performance, parity-preserving COCO-style evaluation.

The Python package is a thin wrapper around the Rust core. Public symbols
documented here are the supported API; anything imported from
:mod:`vernier._core` directly is considered implementation detail and may
change without a deprecation cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from vernier._compat import ParityMode
from vernier._compat import PycocotoolsCOCOeval as COCOeval
from vernier._core import Summary, evaluate_bbox_summary, version

__all__ = [
    "COCOeval",
    "Evaluator",
    "IouType",
    "ParityMode",
    "Summary",
    "__version__",
    "version",
]

__version__: str = version()

#: Similarity / IoU type. Phase 1 ships ``"bbox"``; ``"segm"`` and
#: ``"keypoints"`` arrive in Phase 2/3.
IouType = Literal["bbox"]


@dataclass(frozen=True, slots=True)
class Evaluator:
    """Extended-API COCO-style evaluator (Phase 1: bbox only).

    The instance is immutable per ADR-0006: construct once, call
    :meth:`evaluate` per dataset/detections pair. To change a parameter,
    use :meth:`with_options` (which returns a new evaluator).

    Defaults match pycocotools' detection eval grid, except for
    ``parity_mode``, which defaults to ``"corrected"`` (the ADR-0002
    recommendation for net-new users); migrating users wanting bit-exact
    pycocotools behavior should set ``parity_mode="strict"``.
    """

    iou_type: IouType = "bbox"
    parity_mode: ParityMode = "corrected"
    max_dets: tuple[int, ...] = (1, 10, 100)
    use_cats: bool = True

    def with_options(
        self,
        *,
        iou_type: IouType | None = None,
        parity_mode: ParityMode | None = None,
        max_dets: tuple[int, ...] | None = None,
        use_cats: bool | None = None,
    ) -> Evaluator:
        """Return a copy of this evaluator with the given fields overridden."""
        kwargs: dict[str, object] = {}
        if iou_type is not None:
            kwargs["iou_type"] = iou_type
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
        if self.iou_type != "bbox":
            raise ValueError(f"unsupported iou_type {self.iou_type!r}; Phase 1 ships 'bbox' only")
        return evaluate_bbox_summary(gt, dt, self.parity_mode, list(self.max_dets), self.use_cats)
