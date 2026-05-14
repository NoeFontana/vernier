"""LRP / oLRP public surface.

Wraps the Rust ``vernier._core.optimal_lrp_*`` FFI entry points into a
single dispatching :func:`optimal_lrp` function plus the
:class:`LrpReport` / :class:`LrpPerClass` / :class:`LrpConfig` Python
dataclasses callers hold onto.

The Rust crate is the source of semantic truth (validated against the
numpy oracle at ``tests/python/oracle/lrp/oracle.py`` per ADR-0043);
this module is data conversion + dispatch only.

Per-kernel default thresholds come from ADR-0044 — they are mirrored
here as Python constants so callers reading
``optimal_lrp(...)`` need not chase the Rust source to know what
``tp_threshold=None`` resolves to. The defaults table in ADR-0044 is
the canonical home; if it changes there, change it here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from vernier._core import (
    CocoDataset,
    lrp_default_tau_grid,
    optimal_lrp_bbox,
    optimal_lrp_boundary,
    optimal_lrp_keypoints,
    optimal_lrp_segm,
)
from vernier._types import ParityMode

if TYPE_CHECKING:
    from vernier._core import (
        _LrpReportDict as _FFILrpReportDict,  # pyright: ignore[reportPrivateUsage]
    )

#: Canonical lowercase kernel identifier carried on
#: :class:`LrpConfig`. Mirrors :class:`vernier_core::lrp::LrpKernelMarker`
#: (see ``crates/vernier-core/src/lrp/mod.rs``); pinned so a screenshot
#: of ``report.config.kernel`` survives any downstream tooling
#: round-trip.
KernelName = Literal["bbox", "segm", "boundary", "keypoints"]

#: Per-kernel default ``tp_threshold`` per ADR-0044. The single value
#: across kernels is intentional and defended on the ADR's grounds:
#: ``0.5`` is the operating point the Oksuz TPAMI 2021 paper anchors
#: LRP on, and the same reading transfers to every kernel's
#: similarity scale at the operating-point level.
_DEFAULT_TP_THRESHOLD: dict[KernelName, float] = {
    "bbox": 0.5,
    "segm": 0.5,
    "boundary": 0.5,
    "keypoints": 0.5,
}


@dataclass(frozen=True, slots=True)
class LrpConfig:
    """Resolved LRP configuration recorded on every :class:`LrpReport`.

    Mirrors :class:`vernier_core::lrp::LrpConfig` (see
    ``crates/vernier-core/src/lrp/mod.rs``). The fields carry the
    *resolved* values the call ran under — never ``None`` — so a
    report screenshot is self-describing per ADR-0044.

    ``tau_grid_len`` is the number of points in the tau grid (not the
    grid itself — that would inflate every report by 101 floats on the
    default path). Callers that need the grid back call
    :func:`default_tau_grid` or held onto their own override.
    """

    tp_threshold: float
    tau_grid_len: int
    kernel: KernelName


@dataclass(frozen=True, slots=True)
class LrpPerClass:
    """Per-class entry on an :class:`LrpReport`.

    Mirrors :class:`vernier_core::lrp::LrpPerClass`. Per ADR-0043,
    fields are ``float`` for classes with at least one TP at the
    optimal tau, ``float('nan')`` for the per-class-NaN states the
    oracle defines:

    - Classes with no positive (non-crowd / non-ignore) GTs report
      every field as ``NaN``. These are excluded from the headline
      means.
    - "All-FN" classes (positive GTs exist but no TP at any tau)
      report ``olrp = 1.0, tau = NaN, olrp_loc = NaN, olrp_fp = NaN
      or 0.0`` depending on whether any FPs surfaced, ``olrp_fn =
      1.0``. These ARE included in the headline mean — the worst-case
      is a real result, not missing data.

    The Rust side returns ``None`` for the missing states; this
    wrapper translates to ``NaN`` so the surface mirrors the oracle's
    output shape directly. Use :meth:`is_empty_class` to test for the
    "no positive GTs" state explicitly.
    """

    category_id: int
    olrp: float
    olrp_loc: float
    olrp_fp: float
    olrp_fn: float
    tau: float

    @property
    def is_empty_class(self) -> bool:
        """``True`` when this class has no positive GTs (all fields
        are ``NaN``)."""
        return math.isnan(self.olrp)


@dataclass(frozen=True, slots=True)
class LrpReport:
    """Output of an LRP pass — the headline numbers, per-class
    breakdown, and the resolved configuration.

    Mirrors :class:`vernier_core::lrp::LrpReport`. Per ADR-0043:

    - ``olrp`` is the **mean of per-class oLRP across classes with at
      least one positive GT.** All-FN classes (oLRP = 1.0) contribute
      to the mean; classes with no positive GTs do not.
    - ``loc`` / ``fp`` means are over classes with at least one TP at
      the optimal tau.
    - ``fn`` mean uses the same denominator as ``olrp`` — all-FN
      classes contribute their ``fn_rate = 1.0``.

    The per-class table is the actionable surface: each
    :class:`LrpPerClass` row carries the deployable :attr:`~LrpPerClass.tau`
    a practitioner would set on the model. The aggregated numbers are
    quick comparators between runs.
    """

    olrp: float
    loc: float
    fp: float
    fn: float
    per_class: list[LrpPerClass]
    n_empty_classes: int
    config: LrpConfig

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> LrpReport:
        """Build an :class:`LrpReport` from the FFI dict.

        Internal helper — the supported construction path is
        :func:`optimal_lrp`.
        """
        config_d = d["config"]
        per_class_raw = d["per_class"]
        per_class: list[LrpPerClass] = []
        for entry in per_class_raw:
            per_class.append(
                LrpPerClass(
                    category_id=int(entry["category_id"]),
                    olrp=_opt_to_nan(entry["olrp"]),
                    olrp_loc=_opt_to_nan(entry["olrp_loc"]),
                    olrp_fp=_opt_to_nan(entry["olrp_fp"]),
                    olrp_fn=_opt_to_nan(entry["olrp_fn"]),
                    tau=_opt_to_nan(entry["tau"]),
                )
            )
        return cls(
            olrp=float(d["olrp"]),
            loc=float(d["loc"]),
            fp=float(d["fp"]),
            fn=float(d["fn"]),
            per_class=per_class,
            n_empty_classes=int(d["n_empty_classes"]),
            config=LrpConfig(
                tp_threshold=float(config_d["tp_threshold"]),
                tau_grid_len=int(config_d["tau_grid_len"]),
                kernel=_kernel_name(config_d["kernel"]),
            ),
        )


def optimal_lrp(
    gt: bytes | CocoDataset,
    dt: bytes,
    *,
    iou: object = None,
    tp_threshold: float | None = None,
    tau_grid: Sequence[float] | None = None,
    max_dets_per_image: int = 100,
    use_cats: bool = True,
    parity_mode: ParityMode = "corrected",
) -> LrpReport:
    """LRP / oLRP error decomposition (Oksuz et al., ECCV 2018; TPAMI 2021).

    Splits a detection model's performance into a single number (oLRP)
    plus three components — ``oLRP_Loc`` / ``oLRP_FP`` / ``oLRP_FN`` —
    minimised over a per-class confidence threshold ``tau``. The
    metric's headline deliverable is the *(number, threshold)* pair:
    ``tau`` is the deployable cutoff a practitioner would set on the
    model to get the reported behaviour.

    ``gt`` is the GT JSON bytes (the same shape pycocotools' ``COCO``
    constructor consumes). ``dt`` is the detections JSON bytes (the
    shape ``COCO.loadRes`` consumes). The :class:`vernier.CocoDataset`
    parsed-once handle is accepted in the type signature for
    forward-compat but raises :class:`NotImplementedError` today.

    ``iou`` selects the kernel: ``Bbox()`` (default), ``Segm()``,
    ``Boundary(dilation_ratio=...)``, or ``Keypoints(sigmas=...)``.
    Unlike TIDE (ADR-0024 deferral), LRP supports the keypoints kernel
    — per ADR-0045 the structural objections that deferred TIDE-on-OKS
    do not transfer.

    ``tp_threshold`` is the IoU/OKS floor above which a matched pair
    is a TP. ``None`` resolves to the per-kernel default from
    ADR-0044: ``0.5`` for every kernel.

    ``tau_grid`` is the confidence-threshold grid scanned for the
    argmin. ``None`` resolves to the canonical 101-point grid
    ``0.00, 0.01, ..., 1.00`` (per ADR-0044 — matches deployment
    granularity practitioners tune confidence cutoffs at).

    ``max_dets_per_image`` defaults to ``100`` (the largest rung of
    the standard COCO detection ladder). ``use_cats`` defaults to
    ``True`` (per-class evaluation); set ``False`` for class-agnostic
    decomposition.

    ``parity_mode`` follows :class:`vernier.Evaluator`. Note that LRP
    has no pycocotools analogue — per ADR-0043 the three-tier
    disposition model does NOT extend to this metric — but the flag
    is accepted because the underlying matching engine reads it for
    crowd / ignore semantics that both strict and corrected paths
    honour identically for the LRP-specific quirk set.

    Returns an :class:`LrpReport` carrying the four aggregated
    numbers, the per-class breakdown (one row per class — including
    the deployable ``tau``), and the resolved :class:`LrpConfig`.
    """
    iou_kind = _resolve_iou(iou)
    kernel = _kernel_for(iou_kind)
    resolved_tp = tp_threshold if tp_threshold is not None else _DEFAULT_TP_THRESHOLD[kernel]
    resolved_grid: list[float] = list(tau_grid) if tau_grid is not None else default_tau_grid()

    if isinstance(gt, CocoDataset):
        # Mirror error_decomposition's NotImplementedError shape —
        # the LRP FFI is bytes-only today and the CocoDataset
        # pass-through is a 0.5.x follow-up.
        raise NotImplementedError(
            "vernier.instance.optimal_lrp does not yet accept a CocoDataset handle; "
            "pass GT JSON bytes for now. CocoDataset support is a 0.5.x follow-up."
        )

    raw = _dispatch(
        iou_kind,
        gt,
        dt,
        parity_mode,
        resolved_tp,
        resolved_grid,
        max_dets_per_image,
        use_cats,
    )
    return LrpReport._from_dict(raw)  # pyright: ignore[reportPrivateUsage]


def default_tau_grid() -> list[float]:
    """Return the canonical 101-point tau grid from ADR-0044.

    The grid lives in :class:`vernier_core::lrp::LrpDefaults` and is
    surfaced through the FFI; this helper exists so callers wanting
    to inspect or override the default (e.g. ``tau_grid =
    optimal_lrp.default_tau_grid()[::5]``) can do so without reaching
    into private surfaces.
    """
    return list(lrp_default_tau_grid())


def _opt_to_nan(value: float | None) -> float:
    """Translate ``None`` (the FFI's "undefined" sentinel) to ``NaN``
    (the oracle's "undefined" sentinel). The Rust side uses
    ``Option<f64>`` because the dict layer is the language boundary;
    the user-facing surface uses ``NaN`` so per-class tables compare
    cleanly to oracle output.
    """
    if value is None:
        return float("nan")
    return float(value)


def _resolve_iou(iou: object) -> object:
    """Late-bind ``iou=None`` to ``Bbox()``.

    Materialised inside the function rather than at the keyword
    default site to avoid a top-level import of :class:`vernier.Bbox`
    (circular import: ``vernier`` imports from ``vernier._lrp`` and
    vice versa).
    """
    if iou is None:
        from vernier.instance import Bbox  # local: see docstring.

        return Bbox()
    return iou


def _kernel_for(iou_kind: object) -> KernelName:
    """Map a :data:`vernier.IouKind` instance to its lowercase
    :data:`KernelName`. Unlike TIDE this accepts ``Keypoints`` per
    ADR-0045.
    """
    from vernier.instance import Bbox, Boundary, Keypoints, Segm

    match iou_kind:
        case Bbox():
            return "bbox"
        case Segm():
            return "segm"
        case Boundary():
            return "boundary"
        case Keypoints():
            return "keypoints"
        case _:
            _reject_unknown_iou(iou_kind)


def _dispatch(
    iou_kind: object,
    gt: bytes,
    dt: bytes,
    parity_mode: ParityMode,
    tp_threshold: float,
    tau_grid: list[float],
    max_dets_per_image: int,
    use_cats: bool,
) -> _FFILrpReportDict:
    """Call the right ``vernier._core.optimal_lrp_*`` entry."""
    from vernier.instance import Bbox, Boundary, Keypoints, Segm

    match iou_kind:
        case Bbox():
            return optimal_lrp_bbox(
                gt, dt, parity_mode, tp_threshold, tau_grid, max_dets_per_image, use_cats
            )
        case Segm():
            return optimal_lrp_segm(
                gt, dt, parity_mode, tp_threshold, tau_grid, max_dets_per_image, use_cats
            )
        case Boundary(dilation_ratio=r):
            return optimal_lrp_boundary(
                gt,
                dt,
                parity_mode,
                tp_threshold,
                tau_grid,
                max_dets_per_image,
                use_cats,
                r,
            )
        case Keypoints(sigmas=sigmas):
            # Keypoints carries `sigmas: Mapping[int, tuple[float, ...]]`
            # at the public surface. The FFI wants a plain dict of
            # lists; translate.
            sigma_map: dict[int, list[float]] = {int(k): list(v) for k, v in sigmas.items()}
            return optimal_lrp_keypoints(
                gt,
                dt,
                parity_mode,
                tp_threshold,
                tau_grid,
                max_dets_per_image,
                use_cats,
                sigma_map,
            )
        case _:
            _reject_unknown_iou(iou_kind)


def _kernel_name(value: object) -> KernelName:
    """Narrow an arbitrary object to :data:`KernelName`."""
    match value:
        case "bbox":
            return "bbox"
        case "segm":
            return "segm"
        case "boundary":
            return "boundary"
        case "keypoints":
            return "keypoints"
        case _:
            raise RuntimeError(
                f"LRP FFI returned unexpected kernel name {value!r}; expected "
                f"one of 'bbox' / 'segm' / 'boundary' / 'keypoints'. This is a "
                f"vernier bug — please file an issue."
            )


def _reject_unknown_iou(iou: object) -> NoReturn:
    raise TypeError(
        f"unsupported iou kernel {iou!r} for optimal_lrp; expected "
        f"Bbox(), Segm(), Boundary(...), or Keypoints(...) — see "
        f"vernier.IouKind."
    )
