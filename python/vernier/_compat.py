"""Pycocotools compatibility surface (Phase 1: bbox).

Implements :class:`PycocotoolsCOCOeval`, a drop-in replacement for
:class:`pycocotools.cocoeval.COCOeval`. Per ADR-0007 this is the
migration tool: downstream eval code that imports
``pycocotools.cocoeval.COCOeval`` runs unchanged once the symbol is
swapped (manually, or via :func:`vernier.adapters.patch_pycocotools`).

The class is named ``PycocotoolsCOCOeval`` so the swap is visible in
tracebacks and ``repr()`` even though it lives behind the ``COCOeval``
alias.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from vernier._core import EvalGrid, Summary, evaluate_bbox_grid

ParityMode = Literal["strict", "corrected"]

# Pycocotools' Params(iouType="bbox") defaults — mirrored verbatim so
# parity mode "strict" reproduces the upstream constants bit-exactly.
_DEFAULT_IOU_THRS: NDArray[np.float64] = np.linspace(
    0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
)
_DEFAULT_REC_THRS: NDArray[np.float64] = np.linspace(
    0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01)) + 1, endpoint=True
)
_DEFAULT_AREA_RNG: list[list[float]] = [
    [0, 1e10],
    [0, 32**2],
    [32**2, 96**2],
    [96**2, 1e10],
]
_DEFAULT_AREA_RNG_LBL: list[str] = ["all", "small", "medium", "large"]
_DEFAULT_MAX_DETS: list[int] = [1, 10, 100]


class _Params:
    """Mutable params namespace mirroring ``pycocotools.cocoeval.Params(iouType='bbox')``.

    Phase 1 only honors ``maxDets`` and ``useCats`` mutations; mutating
    the threshold or area-range fields raises ``NotImplementedError`` at
    :meth:`PycocotoolsCOCOeval.evaluate` time so the divergence is loud
    (vs. silently ignored).
    """

    def __init__(self) -> None:
        self.imgIds: list[int] = []
        self.catIds: list[int] = []
        self.iouThrs: NDArray[np.float64] = np.array(_DEFAULT_IOU_THRS, dtype=np.float64)
        self.recThrs: NDArray[np.float64] = np.array(_DEFAULT_REC_THRS, dtype=np.float64)
        self.maxDets: list[int] = list(_DEFAULT_MAX_DETS)
        self.areaRng: list[list[float]] = [list(r) for r in _DEFAULT_AREA_RNG]
        self.areaRngLbl: list[str] = list(_DEFAULT_AREA_RNG_LBL)
        self.useCats: int = 1
        self.useSegm: int | None = None
        self.iouType: str = "bbox"


class PycocotoolsCOCOeval:
    """Drop-in for ``pycocotools.cocoeval.COCOeval`` (bbox only in Phase 1).

    Constructed identically to the upstream class. The state machine
    mirrors pycocotools: :meth:`evaluate` populates :attr:`evalImgs`,
    :meth:`accumulate` populates :attr:`eval`, :meth:`summarize`
    populates :attr:`stats`.

    The keyword-only ``parity_mode`` argument is the one extension over
    the upstream signature; it defaults to
    :attr:`DEFAULT_PARITY_MODE` (``"strict"``), which
    :func:`vernier.adapters.patch_pycocotools` rebinds for the patch
    lifetime.
    """

    DEFAULT_PARITY_MODE: ClassVar[ParityMode] = "strict"

    def __init__(
        self,
        cocoGt: Any = None,  # noqa: N803  pycocotools API
        cocoDt: Any = None,  # noqa: N803  pycocotools API
        iouType: str = "segm",  # noqa: N803  pycocotools API
        *,
        parity_mode: ParityMode | None = None,
    ) -> None:
        if iouType != "bbox":
            raise NotImplementedError(
                f"vernier.COCOeval supports iouType='bbox' only (got {iouType!r}); "
                "segm/keypoints land in Phase 2/3"
            )
        self.cocoGt = cocoGt
        self.cocoDt = cocoDt
        self.params = _Params()
        self.params.iouType = iouType
        self._parity_mode: ParityMode = parity_mode or type(self).DEFAULT_PARITY_MODE
        self.evalImgs: list[dict[str, Any] | None] = []
        self.eval: dict[str, Any] = {}
        self.stats: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._grid: EvalGrid | None = None
        self._summary: Summary | None = None
        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def evaluate(self) -> None:
        if self.cocoGt is None or self.cocoDt is None:
            raise RuntimeError("evaluate requires both cocoGt and cocoDt")
        self._validate_supported_params()
        gt_bytes = json.dumps(self.cocoGt.dataset, default=_json_default).encode()
        dt_anns = self.cocoDt.dataset.get("annotations", [])
        dt_bytes = json.dumps(dt_anns, default=_json_default).encode()
        max_det_top = max(self.params.maxDets)
        use_cats = bool(self.params.useCats)
        self._grid = evaluate_bbox_grid(
            gt_bytes, dt_bytes, self._parity_mode, max_det_top, use_cats
        )
        self.evalImgs = self._grid.eval_imgs()

    def accumulate(self, p: Any = None) -> None:
        if self._grid is None:
            raise RuntimeError("Please run evaluate() first")
        max_dets = list(self.params.maxDets)
        acc = self._grid.accumulate(max_dets)
        self._summary = acc.summarize(max_dets)
        self.eval = {
            "params": self.params if p is None else p,
            "counts": list(acc.counts),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": np.asarray(acc.precision),
            "recall": np.asarray(acc.recall),
            "scores": np.asarray(acc.scores),
        }

    def summarize(self) -> None:
        if not self.eval or self._summary is None:
            raise RuntimeError("Please run accumulate() first")
        self.stats = np.asarray(self._summary.stats, dtype=np.float64)
        # Quirk L5 disposition: strict mirrors pycocotools' stdout side
        # effect; corrected stays silent (the structured Summary on
        # ``self._summary`` is the canonical surface).
        if self._parity_mode == "strict":
            for line in self._summary.pretty_lines():
                print(line)

    def _validate_supported_params(self) -> None:
        if not np.array_equal(self.params.iouThrs, _DEFAULT_IOU_THRS):
            raise NotImplementedError(
                "vernier.COCOeval does not yet support custom params.iouThrs; "
                "reset to the pycocotools default (linspace(0.5, 0.95, 10))"
            )
        if not np.array_equal(self.params.recThrs, _DEFAULT_REC_THRS):
            raise NotImplementedError(
                "vernier.COCOeval does not yet support custom params.recThrs; "
                "reset to the pycocotools default (linspace(0.0, 1.0, 101))"
            )
        if [list(r) for r in self.params.areaRng] != [list(r) for r in _DEFAULT_AREA_RNG]:
            raise NotImplementedError("vernier.COCOeval does not yet support custom params.areaRng")
        gt_img_ids = sorted(self.cocoGt.getImgIds())
        if sorted(self.params.imgIds) != gt_img_ids:
            raise NotImplementedError(
                "vernier.COCOeval does not yet support params.imgIds subsetting; "
                "evaluate the full dataset"
            )
        gt_cat_ids = sorted(self.cocoGt.getCatIds())
        if sorted(self.params.catIds) != gt_cat_ids:
            raise NotImplementedError(
                "vernier.COCOeval does not yet support params.catIds subsetting"
            )


def _json_default(obj: Any) -> Any:
    # numpy scalars from pycocotools' loadRes processing.
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")
