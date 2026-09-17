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

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, ClassVar, Final, Literal, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from vernier._coco_json import (
    to_coco_json,
    with_mask_image_sizes,
    with_placeholder_image_sizes,
)
from vernier._core import (
    Accumulated,
    EvalGrid,
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

    Honors ``maxDets``, ``useCats``, ``iouThrs``, ``recThrs`` and
    ``catIds`` mutations; mutating ``areaRng`` or subsetting ``imgIds``
    raises ``NotImplementedError`` at :meth:`PycocotoolsCOCOeval.evaluate`
    time so the divergence is loud (vs. silently ignored).

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
        self.params: _Params = _Params(kind)
        self._dilation_ratio = dilation_ratio
        self._parity_mode: ParityMode = parity_mode or type(self).DEFAULT_PARITY_MODE
        # `[]` / `{}` before the first evaluate(), mirroring
        # pycocotools' `__init__`; `None` after, meaning "not built yet"
        # for the lazy accessors below.
        self._eval_imgs: list[dict[str, Any] | None] | None = []
        self._ious: dict[tuple[int, int], _IouMatrix] | None = {}
        self.eval: dict[str, Any] = {}
        self.stats: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._grid: EvalGrid | None = None
        # Which optional per-cell retention `self._grid` was built with,
        # as `(meta, iou)`. Both start off: `accumulate` / `summarize`
        # read neither, so the common evaluate → accumulate → summarize
        # cycle never pays for them (see :meth:`_grid_for`).
        self._retained: tuple[bool, bool] = (False, False)
        self._gt_bytes: bytes = b""
        self._dt_bytes: bytes = b""
        self._accumulated: Accumulated | None = None
        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def evaluate(self) -> None:
        if self.cocoGt is None or self.cocoDt is None:
            raise RuntimeError("evaluate requires both cocoGt and cocoDt")
        self._validate_supported_params()
        self._gt_bytes, self._dt_bytes = self._serialize_inputs()
        # Neither optional retention is on: `accumulate` reads the
        # matched cells, not the pycocotools-shaped bookkeeping, and
        # `evalImgs` / `ious` re-evaluate on first read (see
        # :meth:`_grid_for`). Retaining metadata for every cell roughly
        # doubles the evaluation's allocations, and the callers that
        # drive this shim mostly never look at either attribute.
        self._retained = (False, False)
        self._grid = self._build_grid(retain_meta=False, retain_iou=False)
        self._eval_imgs = None
        self._ious = None

    @property
    def evalImgs(self) -> list[dict[str, Any] | None]:  # noqa: N802  pycocotools API
        """Per-image evaluation records, built from the grid on first access.

        Reading this re-runs the per-image pass with per-cell metadata
        retained, once, and caches the result. Nothing else on this
        class needs the metadata, so the cost lands only on callers
        that ask.
        """
        if self._eval_imgs is None:
            self._eval_imgs = self._grid_for(meta=True).eval_imgs()
        return self._eval_imgs

    @evalImgs.setter
    def evalImgs(self, value: list[dict[str, Any] | None]) -> None:  # noqa: N802  pycocotools API
        self._eval_imgs = value

    @property
    def ious(self) -> dict[tuple[int, int], _IouMatrix]:
        """``{(imgId, catId): similarity matrix}``, built on first access.

        Mirrors ``pycocotools.cocoeval.COCOeval.ious``: one entry per
        ``(image, category)`` pair in ``params.imgIds x params.catIds``
        (``catId`` is ``-1`` when ``params.useCats`` is falsy), holding a
        ``(D, G)`` array of detections-by-ground-truths with detections
        score-descending and truncated to ``max(params.maxDets)``.

        Per quirk **F5**, a pair with no detections *or* no ground truths
        is a bare ``[]`` rather than an empty array, which is what
        ``maskUtils.iou`` and ``computeOks`` both return.

        Reading this re-runs the per-image pass with the similarity
        matrices retained, once, and caches the result — they are
        O(G x D) per cell and neither ``accumulate`` nor ``summarize``
        reads them.
        """
        if self._ious is None:
            retained = self._grid_for(iou=True).ious()
            empty: _IouMatrix = []
            self._ious = {
                (int(image_id), int(category_id)): retained.get((image_id, category_id), empty)
                for image_id in self.params.imgIds
                for category_id in self._axis_category_ids()
            }
        return self._ious

    @ious.setter
    def ious(self, value: dict[tuple[int, int], _IouMatrix]) -> None:
        self._ious = value

    def _grid_for(self, *, meta: bool = False, iou: bool = False) -> EvalGrid:
        """The evaluated grid, re-evaluated if it lacks a retention.

        The retention flags only ever widen: asking for ``ious`` after
        ``evalImgs`` rebuilds once with both, so a caller that reads
        both attributes pays for at most two extra passes over the
        lifetime of the object, not one per read.
        """
        if self._grid is None:
            raise RuntimeError("Please run evaluate() first")
        has_meta, has_iou = self._retained
        if (has_meta or not meta) and (has_iou or not iou):
            return self._grid
        want_iou = has_iou or iou
        # `retain_iou` builds the metadata too (the table builders read
        # it), so record that rather than rebuilding for it later.
        self._retained = (has_meta or meta or want_iou, want_iou)
        self._grid = self._build_grid(retain_meta=self._retained[0], retain_iou=self._retained[1])
        return self._grid

    def _build_grid(self, *, retain_meta: bool, retain_iou: bool) -> EvalGrid:
        max_det_top = max(self.params.maxDets)
        use_cats = bool(self.params.useCats)
        gt_bytes = self._gt_bytes
        dt_bytes = self._dt_bytes
        grid_options: _GridOptions = {
            "iou_thresholds": _custom_ladder(self.params.iouThrs, _DEFAULT_IOU_THRS),
            "recall_thresholds": _custom_ladder(self.params.recThrs, _DEFAULT_REC_THRS),
            # Quirk J3: COCOeval reads `d['area']` off the cocoDt it is
            # handed; only `loadRes` derives it.
            "dt_area": "supplied",
            "retain_meta": retain_meta,
            "retain_iou": retain_iou,
        }
        if self.params.iouType == IOU_BBOX:
            return evaluate_bbox_grid(
                gt_bytes, dt_bytes, self._parity_mode, max_det_top, use_cats, **grid_options
            )
        if self.params.iouType == IOU_SEGM:
            return evaluate_segm_grid(
                gt_bytes, dt_bytes, self._parity_mode, max_det_top, use_cats, **grid_options
            )
        if self.params.iouType == IOU_BOUNDARY:
            return evaluate_boundary_grid(
                gt_bytes,
                dt_bytes,
                self._parity_mode,
                max_det_top,
                use_cats,
                self._dilation_ratio,
                **grid_options,
            )
        if self.params.iouType == IOU_KEYPOINTS:
            return evaluate_keypoints_grid(
                gt_bytes,
                dt_bytes,
                self._parity_mode,
                max_det_top,
                use_cats,
                self._resolve_kp_sigmas(),
                **grid_options,
            )
        raise NotImplementedError(f"unsupported iouType {self.params.iouType!r}")

    def _serialize_inputs(self) -> tuple[bytes, bytes]:
        """GT dataset + DT annotations as the JSON bytes the grid parses.

        Applies, in pycocotools' own order, the two normalizations
        ``COCOeval._prepare`` performs before any kernel runs: the
        ``params.catIds`` filter, then the image-size resolution that
        ``annToRLE`` implies.
        """
        assert self.cocoGt is not None  # evaluate() guards this
        assert self.cocoDt is not None  # evaluate() guards this
        gt_dataset: Mapping[str, Any] = self.cocoGt.dataset
        dt_dataset: Mapping[str, Any] = self.cocoDt.dataset
        dt_anns: Sequence[Any] = dt_dataset.get("annotations", [])
        selected = self._selected_category_ids()
        if selected is not None:
            gt_dataset = _with_categories(gt_dataset, selected)
            dt_anns = [ann for ann in dt_anns if int(ann["category_id"]) in selected]
        if self.params.iouType in (IOU_BBOX, IOU_KEYPOINTS):
            gt_dataset = with_placeholder_image_sizes(gt_dataset)
        else:
            gt_dataset = with_mask_image_sizes(gt_dataset, _detection_sizes(dt_dataset))
        return to_coco_json(gt_dataset), to_coco_json(dt_anns)

    def _axis_category_ids(self) -> list[int]:
        """The ``catId`` axis of :attr:`ious`, as pycocotools keys it.

        ``cocoeval.py:522`` reads ``catIds = p.catIds if p.useCats else
        [-1]``, so a collapsed evaluation keys every pair on the single
        ``-1`` sentinel (quirk **L4**).
        """
        if not self.params.useCats:
            return [-1]
        return sorted({int(cat) for cat in self.params.catIds})

    def _selected_category_ids(self) -> dict[int, None] | None:
        """``params.catIds`` as an ordered set, or ``None`` if it is everything.

        ``None`` means "no filtering needed" and keeps the whole-dataset
        path allocation-free; it is the case every caller but a
        per-class loop hits.
        """
        assert self.cocoGt is not None  # evaluate() guards this
        requested = sorted({int(cat) for cat in self.params.catIds})
        if requested == sorted(self.cocoGt.getCatIds()):
            return None
        return dict.fromkeys(requested)

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
        self._accumulated = acc
        self.eval = {
            "params": self.params if p is None else p,
            "counts": list(acc.counts),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": np.asarray(acc.precision),
            "recall": np.asarray(acc.recall),
            "scores": np.asarray(acc.scores),
        }

    def summarize(self) -> None:
        if not self.eval or self._accumulated is None:
            raise RuntimeError("Please run accumulate() first")
        plan: Literal["detection", "keypoints"] = (
            "keypoints" if self.params.iouType == IOU_KEYPOINTS else "detection"
        )
        # The accumulator carries the grid's parity mode, which picks the
        # aggregate-AP cap on a ladder without 100 (quirk L9).
        summary = self._accumulated.summarize(plan=plan)
        self.stats = np.asarray(summary.stats, dtype=np.float64)
        # Quirk L5 disposition: strict mirrors pycocotools' stdout side
        # effect; corrected stays silent.
        if self._parity_mode == PARITY_STRICT:
            for line in summary.pretty_lines():
                print(line)

    def _validate_supported_params(self) -> None:
        assert self.cocoGt is not None  # evaluate() guards this
        _reject_use_segm(self.params.useSegm)
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
        # `params.catIds` subsetting *is* supported — see
        # `_selected_category_ids` / `_with_categories`. It is how
        # TorchMetrics' `class_metrics=True` drives one evaluator around
        # a per-class loop.

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


# Pycocotools' `ious` values: a `(D, G)` array, or a bare `[]` for a
# pair with nothing on one side (quirk **F5**).
_IouMatrix = NDArray[np.float64] | list[Any]


class _GridOptions(TypedDict):
    iou_thresholds: list[float] | None
    recall_thresholds: list[float] | None
    dt_area: Literal["bbox", "supplied"]
    retain_meta: bool
    retain_iou: bool


def _custom_ladder(actual: NDArray[np.float64], default: NDArray[np.float64]) -> list[float] | None:
    # Identity check first: the default _Params binds the canonical
    # arrays by reference, so an unmutated ladder is a single pointer
    # compare that keeps the grid on its canonical-axis fast path.
    if actual is default:
        return None
    return [float(v) for v in np.asarray(actual, dtype=np.float64).ravel()]


def _with_categories(dataset: Mapping[str, Any], selected: Mapping[int, None]) -> Mapping[str, Any]:
    # `COCOeval._prepare` loads only the annotations `getAnnIds(imgIds=
    # p.imgIds, catIds=p.catIds)` returns, so a subset `params.catIds`
    # evaluates a dataset that holds nothing else — which is exactly
    # what this builds, on a shallow copy. Restricting `categories` in
    # step keeps vernier's K axis the length pycocotools' is (its own
    # K axis is `len(p.catIds)`), and vernier rejects an annotation
    # that names a category the dataset does not declare.
    #
    # A requested category the dataset never declared is kept as an
    # empty one rather than dropped: pycocotools evaluates it to a row
    # of -1s, and a shorter K axis here would silently renumber the
    # rest.
    declared = {int(category["id"]): category for category in dataset.get("categories", [])}
    return {
        **dataset,
        "categories": [declared.get(cat, {"id": cat, "name": str(cat)}) for cat in selected],
        "annotations": [
            ann for ann in dataset.get("annotations", []) if int(ann["category_id"]) in selected
        ],
    }


def _detection_sizes(dataset: Mapping[str, Any]) -> dict[Any, tuple[int, int] | None]:
    # `{image_id: (height, width) | None}` over the images the cocoDt's
    # annotations point at — the shape
    # `vernier.adapters.with_mask_image_sizes` reads. `None` where that
    # side does not carry a size either, which is where pycocotools'
    # `annToRLE` raises `KeyError` and vernier's schema error stands in.
    known = {
        image["id"]: (image["height"], image["width"])
        for image in dataset.get("images", [])
        if "width" in image and "height" in image
    }
    return {ann["image_id"]: known.get(ann["image_id"]) for ann in dataset.get("annotations", [])}


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
