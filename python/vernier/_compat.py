"""Pycocotools compatibility surface (bbox / segm / boundary / keypoints).

Implements :class:`PycocotoolsCOCOeval`, a drop-in replacement for
:class:`pycocotools.cocoeval.COCOeval`. Per ADR-0007 this is the
migration tool: downstream eval code that imports
``pycocotools.cocoeval.COCOeval`` runs unchanged once the symbol is
swapped (manually, or via :func:`vernier.adapters.patch_pycocotools`).
The constructor also accepts ``iouType="boundary"`` and a
``dilation_ratio`` kwarg (ADR-0010), mirroring the
``bowenc0221/boundary-iou-api`` oracle's signature, and
``iouType="keypoints"`` (ADR-0012) with ``params.kpt_oks_sigmas``
honored on the params object per pycocotools convention.

The class is named ``PycocotoolsCOCOeval`` so the swap is visible in
tracebacks and ``repr()`` even though it lives behind the ``COCOeval``
alias.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any, ClassVar, Final, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from vernier._core import (
    EvalGrid,
    Summary,
    evaluate_bbox_grid,
    evaluate_boundary_grid,
    evaluate_keypoints_grid,
    evaluate_segm_grid,
)
from vernier._types import (
    DEFAULT_DILATION_RATIO,
    PARITY_STRICT,
    ParityMode,
)

IouType = Literal["bbox", "segm", "boundary", "keypoints"]

IOU_BBOX: Final[IouType] = "bbox"
IOU_SEGM: Final[IouType] = "segm"
IOU_BOUNDARY: Final[IouType] = "boundary"
IOU_KEYPOINTS: Final[IouType] = "keypoints"

# Pycocotools' Params(iouType="bbox") defaults — mirrored verbatim so
# parity mode "strict" reproduces the upstream constants bit-exactly.
_DEFAULT_IOU_THRS: Final[NDArray[np.float64]] = np.linspace(
    0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True, dtype=np.float64
)
_DEFAULT_REC_THRS: Final[NDArray[np.float64]] = np.linspace(
    0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01)) + 1, endpoint=True, dtype=np.float64
)
_DEFAULT_AREA_RNG: Final[list[list[float]]] = [
    [0, 1e10],
    [0, 32**2],
    [32**2, 96**2],
    [96**2, 1e10],
]
_DEFAULT_AREA_RNG_LBL: Final[list[str]] = ["all", "small", "medium", "large"]
_DEFAULT_MAX_DETS: Final[list[int]] = [1, 10, 100]

# Pycocotools' Params(iouType="keypoints").setKpParams() defaults — quirk
# D5 drops the "small" bucket (kp summary never asks for it) and the
# ladder is the single rung [20]. Mirrored verbatim so strict parity
# reproduces the upstream constants. The literal `1e5 ** 2` matches the
# upstream source even though it equals `1e10` numerically.
_KP_DEFAULT_AREA_RNG: Final[list[list[float]]] = [
    [0, 1e5**2],
    [32**2, 96**2],
    [96**2, 1e5**2],
]
_KP_DEFAULT_AREA_RNG_LBL: Final[list[str]] = ["all", "medium", "large"]
_KP_DEFAULT_MAX_DETS: Final[list[int]] = [20]
# COCO-person 17-keypoint sigma table from pycocotools' ``setKpParams``.
# Bound by reference for identity-checked shortcut in
# :meth:`PycocotoolsCOCOeval._validate_supported_params` — pycocotools
# itself rebinds the array (rather than mutating in place) when callers
# customize sigmas, so the shared reference is safe.
_KP_DEFAULT_OKS_SIGMAS: Final[NDArray[np.float64]] = (
    np.array(
        [
            0.26,
            0.25,
            0.25,
            0.35,
            0.35,
            0.79,
            0.79,
            0.72,
            0.72,
            0.62,
            0.62,
            1.07,
            1.07,
            0.87,
            0.87,
            0.89,
            0.89,
        ],
        dtype=np.float64,
    )
    / 10.0
)


class CocoLike(Protocol):
    """Structural type for the pycocotools.coco.COCO surface we touch.

    Defined as a Protocol because pycocotools ships no ``py.typed``
    marker; a hard import of the upstream class would force pyright
    into an ``Unknown`` cliff. The drop-in only depends on the four
    members below — anything else passes through ``self.cocoGt`` /
    ``self.cocoDt`` as ``Any``.
    """

    # Read-only property keeps the protocol covariant: a plain
    # ``dataset: Mapping`` attribute is invariant on pyright, which
    # rejects pycocotools' more specific ``_Dataset`` TypedDict. We
    # only ever read this attribute, so the property surface is the
    # honest annotation.
    @property
    def dataset(self) -> Mapping[str, Any]: ...

    def getImgIds(self) -> list[int]: ...  # noqa: N802  pycocotools API
    def getCatIds(self) -> list[int]: ...  # noqa: N802  pycocotools API


class _Params:
    """Mutable params namespace mirroring ``pycocotools.cocoeval.Params``.

    Phase 1 only honors ``maxDets`` and ``useCats`` mutations; mutating
    the threshold or area-range fields raises ``NotImplementedError`` at
    :meth:`PycocotoolsCOCOeval.evaluate` time so the divergence is loud
    (vs. silently ignored).

    Constructed with the iouType so the area-range / maxDets / sigmas
    defaults branch the same way pycocotools' ``__init__`` dispatches
    between ``setDetParams`` and ``setKpParams``. The keypoints branch
    (ADR-0012) drops the "small" bucket (quirk D5), pins the ladder to
    ``[20]``, and exposes ``kpt_oks_sigmas`` (quirk F1).
    """

    def __init__(self, iouType: IouType = IOU_BBOX) -> None:  # noqa: N803  pycocotools API
        self.imgIds: list[int] = []
        self.catIds: list[int] = []
        # Bind defaults by reference so _validate_supported_params can
        # short-circuit via identity (`is _DEFAULT_IOU_THRS`) when the
        # user has not mutated. Pycocotools rebinds rather than mutates
        # these arrays in place, so the shared reference is safe.
        self.iouThrs: NDArray[np.float64] = _DEFAULT_IOU_THRS
        self.recThrs: NDArray[np.float64] = _DEFAULT_REC_THRS
        if iouType == IOU_KEYPOINTS:
            self.maxDets: list[int] = list(_KP_DEFAULT_MAX_DETS)
            self.areaRng: list[list[float]] = [list(r) for r in _KP_DEFAULT_AREA_RNG]
            self.areaRngLbl: list[str] = list(_KP_DEFAULT_AREA_RNG_LBL)
            # pycocotools stores a single 17-tuple on the params object
            # that applies to every category; vernier's Rust core takes
            # `dict[int, list[float]]`, so the FFI translation in
            # `evaluate()` fans this out across `cocoGt.cats` keys.
            self.kpt_oks_sigmas: NDArray[np.float64] = _KP_DEFAULT_OKS_SIGMAS
        else:
            self.maxDets = list(_DEFAULT_MAX_DETS)
            self.areaRng = [list(r) for r in _DEFAULT_AREA_RNG]
            self.areaRngLbl = list(_DEFAULT_AREA_RNG_LBL)
        self.useCats: int = 1
        self.useSegm: int | None = None
        self.iouType: IouType = iouType

    def __setattr__(self, name: str, value: object) -> None:
        # Per ADR-0039 §"Drop-in shim non-exposure" + ADR-0040: the
        # snake_case grid fields belong on `vernier.instance.Evaluator`,
        # not on this pycocotools-shaped shim. Surfacing AttributeError
        # here keeps users from silently configuring a custom grid that
        # the shim cannot honor. The pycocotools-shaped camelCase names
        # (iouThrs, recThrs, areaRng) still mutate normally.
        if name in _ADR_0040_NATIVE_FIELDS:
            raise AttributeError(
                f"{name!r} belongs on vernier.instance.Evaluator (ADR-0040), "
                f"not the pycocotools-shaped COCOeval shim. The shim mirrors "
                f"pycocotools' surface for migration; for custom IoU / recall "
                f"thresholds or area ranges, use the native Evaluator: "
                f"`vernier.instance.Evaluator({name}=...)`. "
                f"(Pycocotools-shaped {_ADR_0040_NATIVE_FIELDS[name]!r} on this "
                f"shim still mutates normally.)"
            )
        object.__setattr__(self, name, value)


# Maps snake_case Evaluator field names to the pycocotools-shaped
# camelCase shim attribute that mutates normally. Used by the
# `_Params.__setattr__` guard above to point migrating users at the
# native `vernier.instance.Evaluator` surface.
_ADR_0040_NATIVE_FIELDS: Final[dict[str, str]] = {
    "iou_thresholds": "iouThrs",
    "recall_thresholds": "recThrs",
    "area_ranges": "areaRng",
}


class PycocotoolsCOCOeval:
    """Drop-in for ``pycocotools.cocoeval.COCOeval`` (bbox / segm / boundary / keypoints).

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

    DEFAULT_PARITY_MODE: ClassVar[ParityMode] = PARITY_STRICT

    def __init__(
        self,
        cocoGt: CocoLike | None = None,  # noqa: N803  pycocotools API
        cocoDt: CocoLike | None = None,  # noqa: N803  pycocotools API
        iouType: str = IOU_SEGM,  # noqa: N803  pycocotools API
        dilation_ratio: float = DEFAULT_DILATION_RATIO,
        *,
        parity_mode: ParityMode | None = None,
    ) -> None:
        if iouType not in (IOU_BBOX, IOU_SEGM, IOU_BOUNDARY, IOU_KEYPOINTS):
            raise NotImplementedError(
                f"vernier.COCOeval supports iouType in "
                f"('bbox', 'segm', 'boundary', 'keypoints') (got {iouType!r})"
            )
        # Pyright narrows `iouType: str` to the four-element `IouType`
        # Literal via the membership check above; the explicit annotation
        # makes that contract self-documenting at the dispatch boundary.
        kind: IouType = iouType
        self.cocoGt = cocoGt
        self.cocoDt = cocoDt
        self.params = _Params(kind)
        self._dilation_ratio = dilation_ratio
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
        if self.params.iouType == IOU_BBOX:
            self._grid = evaluate_bbox_grid(
                gt_bytes, dt_bytes, self._parity_mode, max_det_top, use_cats
            )
        elif self.params.iouType == IOU_SEGM:
            self._grid = evaluate_segm_grid(
                gt_bytes, dt_bytes, self._parity_mode, max_det_top, use_cats
            )
        elif self.params.iouType == IOU_BOUNDARY:
            self._grid = evaluate_boundary_grid(
                gt_bytes,
                dt_bytes,
                self._parity_mode,
                max_det_top,
                use_cats,
                self._dilation_ratio,
            )
        elif self.params.iouType == IOU_KEYPOINTS:
            self._grid = evaluate_keypoints_grid(
                gt_bytes,
                dt_bytes,
                self._parity_mode,
                max_det_top,
                use_cats,
                self._resolve_kp_sigmas(),
            )
        else:
            raise NotImplementedError(f"unsupported iouType {self.params.iouType!r}")
        self.evalImgs = self._grid.eval_imgs()

    def accumulate(self, p: Any = None) -> None:
        if self._grid is None:
            raise RuntimeError("Please run evaluate() first")
        # Quirk A2 (strict): pycocotools' cocoeval.py:137 opens
        # accumulate() with `p.maxDets = sorted(p.maxDets)` — silently
        # normalize the user-facing list and the local copy that flows
        # into the Rust side. Without this, AR_1 / AR_10 / AR_100 slots
        # bind to whatever order the user happened to pass.
        self.params.maxDets = sorted(self.params.maxDets)
        max_dets = list(self.params.maxDets)
        acc = self._grid.accumulate(max_dets)
        self.eval = {
            "params": self.params if p is None else p,
            "counts": list(acc.counts),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": np.asarray(acc.precision),
            "recall": np.asarray(acc.recall),
            "scores": np.asarray(acc.scores),
        }
        plan: Literal["detection", "keypoints"] = (
            "keypoints" if self.params.iouType == IOU_KEYPOINTS else "detection"
        )
        self._summary = acc.summarize(max_dets, plan=plan)

    def summarize(self) -> None:
        if not self.eval or self._summary is None:
            raise RuntimeError("Please run accumulate() first")
        self.stats = np.asarray(self._summary.stats, dtype=np.float64)
        # Quirk L5 disposition: strict mirrors pycocotools' stdout side
        # effect; corrected stays silent (the structured Summary on
        # ``self._summary`` is the canonical surface).
        if self._parity_mode == PARITY_STRICT:
            for line in self._summary.pretty_lines():
                print(line)

    def _validate_supported_params(self) -> None:
        assert self.cocoGt is not None  # evaluate() guards this
        _reject_use_segm(self.params.useSegm)
        _reject_if_mutated_array(self.params.iouThrs, _DEFAULT_IOU_THRS, "iouThrs")
        _reject_if_mutated_array(self.params.recThrs, _DEFAULT_REC_THRS, "recThrs")
        # Quirk D5 / ADR-0012: kp uses a 3-bucket area grid. The valid
        # default is per-iouType, so dispatch on iouType rather than
        # comparing against a single canonical list.
        canonical_area = (
            _KP_DEFAULT_AREA_RNG if self.params.iouType == IOU_KEYPOINTS else _DEFAULT_AREA_RNG
        )
        if [list(r) for r in self.params.areaRng] != [list(r) for r in canonical_area]:
            _raise_unsupported("areaRng")
        if sorted(self.params.imgIds) != sorted(self.cocoGt.getImgIds()):
            _raise_unsupported("imgIds", "subsetting; evaluate the full dataset")
        if sorted(self.params.catIds) != sorted(self.cocoGt.getCatIds()):
            _raise_unsupported("catIds", "subsetting")

    def _resolve_kp_sigmas(self) -> dict[int, list[float]]:
        """Translate ``params.kpt_oks_sigmas`` into the FFI's per-cat shape.

        Pycocotools stores a single 17-tuple on the params object that
        applies to every category; vernier's Rust core takes
        ``dict[int, list[float]]``. If the user has not mutated the
        sigma array the identity check short-circuits to an empty dict
        — the Rust core's ``OksSimilarity`` falls back to the COCO-person
        defaults, keeping output byte-identical to pycocotools on the
        common path. If the array has been replaced, fan it out across
        every ``cocoGt`` category id (quirk F1).
        """
        sigmas = self.params.kpt_oks_sigmas
        if sigmas is _KP_DEFAULT_OKS_SIGMAS:
            return {}
        if np.array_equal(sigmas, _KP_DEFAULT_OKS_SIGMAS):
            return {}
        assert self.cocoGt is not None  # evaluate() guards this
        custom = [float(v) for v in np.asarray(sigmas, dtype=np.float64).ravel()]
        return {int(cat): list(custom) for cat in self.cocoGt.getCatIds()}


def _reject_if_mutated_array(
    actual: NDArray[np.float64], default: NDArray[np.float64], name: str
) -> None:
    # Identity check first: the default _Params binds the canonical
    # arrays by reference, so an unmutated grid is a single pointer
    # compare instead of a 10/101-element scan on every evaluate().
    if actual is default:
        return
    if not np.array_equal(actual, default):
        _raise_unsupported(name)


def _raise_unsupported(name: str, detail: str = "") -> None:
    suffix = f"; {detail}" if detail else ""
    raise NotImplementedError(f"vernier.COCOeval does not yet support custom params.{name}{suffix}")


def _reject_use_segm(value: int | None) -> None:
    # Quirk L3 (corrected): pycocotools' Params.useSegm has been
    # deprecated for years but is still honored — if set, it silently
    # overrides iouType and prints a warning. Vernier drops the
    # honor-with-warning path entirely; users must pick iouType up
    # front so the dispatch is unambiguous.
    if value is not None:
        raise NotImplementedError(
            "params.useSegm was deprecated by pycocotools years ago and is not "
            "honored by vernier (quirk L3). Pass iouType='bbox' or iouType='segm' "
            "to COCOeval(...) instead."
        )


def _json_default(obj: Any) -> Any:
    # numpy scalars from pycocotools' loadRes processing.
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")
