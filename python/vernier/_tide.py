"""TIDE error-decomposition public surface.

Wraps the Rust ``vernier._core.error_decomposition_*`` FFI entry points
into a single dispatching :func:`error_decomposition` function plus the
:class:`TideReport` / :class:`TideConfig` Python dataclasses callers
hold onto.

The Rust crate is the source of semantic truth (validated against the
numpy oracle per ADR-0021); this module is data conversion + dispatch
only. Per-kernel default thresholds come from ADR-0022 — they are
mirrored here as Python constants so callers reading
``error_decomposition(...)`` need not chase the Rust source to know
what ``t_f=None`` resolves to. The defaults table in ADR-0022 is the
canonical home; if it changes there, change it here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from vernier._compat import ParityMode
from vernier._core import (
    CocoDataset,
    error_decomposition_bbox,
    error_decomposition_boundary,
    error_decomposition_segm,
    fp_iou_histogram_bbox,
    fp_iou_histogram_boundary,
    fp_iou_histogram_segm,
)

if TYPE_CHECKING:
    import numpy as np

    # `_TideReportDict` is a TypedDict declared in the `.pyi` stub for
    # the FFI's return shape; the runtime extension does not actually
    # export the symbol, so it's available to the type-checker only.
    from vernier._core import (
        _FpIouHistogramDict as _FFIFpIouHistogramDict,  # pyright: ignore[reportPrivateUsage]
    )
    from vernier._core import (
        _TideReportDict as _FFITideReportDict,  # pyright: ignore[reportPrivateUsage]
    )

# Kept in sync with the four-element discriminated union in
# ``vernier.__init__``. Re-imported lazily inside :func:`error_decomposition`
# to avoid a circular import.

#: Canonical lowercase kernel identifier carried on
#: :class:`TideConfig`. Mirrors :class:`vernier_core::tide::report::KernelMarker`
#: (see ``crates/vernier-core/src/tide/report.rs``); pinned so a screenshot
#: of ``report.config.kernel`` survives any downstream tooling round-trip.
KernelName = Literal["bbox", "segm", "boundary"]

#: Per-kernel default ``(t_f, t_b)`` thresholds per ADR-0022.
#: ``t_f`` is ``0.5`` everywhere — matches the TIDE paper, matches
#: AP@0.5 intuition, and makes the bin-assignment threshold the same
#: as the matching threshold (no double-cutoff to reason about).
#: ``t_b`` is per-kernel:
#:
#: - ``bbox``: ``0.1`` — TIDE paper, COCO bbox.
#: - ``segm``: ``0.1`` — extrapolated from the bbox row (segm IoU is
#:   bounded above by bbox IoU on the same instance and tracks within
#:   ~10% on standard models). **Tentative**, see ADR-0022.
#: - ``boundary``: ``0.05`` — geometric argument from the band-area
#:   compression of boundary IoU at ``dilation_ratio=0.02``.
#:   **Tentative**, see ADR-0022's "Decision gate" section.
#:
#: ADR-0022 is the canonical home; mirrored here as Python constants
#: so call-site readers can see what ``t_f=None``/``t_b=None`` resolves
#: to without chasing Rust source.
_DEFAULT_THRESHOLDS: dict[KernelName, tuple[float, float]] = {
    "bbox": (0.5, 0.1),
    "segm": (0.5, 0.1),
    "boundary": (0.5, 0.05),
}


@dataclass(frozen=True, slots=True)
class TideConfig:
    """Resolved TIDE configuration recorded on every :class:`TideReport`.

    Mirrors :class:`vernier_core::tide::report::TideConfig` (see
    ``crates/vernier-core/src/tide/report.rs``). The fields carry the
    *resolved* thresholds the call ran under — never ``None`` — so a
    report screenshot is self-describing per ADR-0022.

    ``cross_class_topk`` (per ADR-0023) is intentionally absent from
    this Python surface in 0.5.0. The Rust default (``None`` =
    materialize the full per-detection cross-class IoU vector) is the
    only behavior reachable from Python today; the knob will be exposed
    once a workload demands it.
    """

    t_f: float
    t_b: float
    kernel: KernelName


@dataclass(frozen=True, slots=True)
class TideReport:
    """Six-bin TIDE decomposition of a detection model's mAP gap.

    Returned by :func:`error_decomposition`. Each :attr:`delta` entry is
    the mAP increase the model would achieve if every detection assigned
    to that bin were corrected; :attr:`baseline_map` is the headline
    number before any correction; :attr:`delta_all_fp_removed` is the
    paper's "perfect rejection" upper bound (what mAP would be if every
    FP were dropped). The per-bin deltas should sum to at most
    :attr:`delta_all_fp_removed` — useful as a sanity check that the
    rewrite layer is internally consistent.

    Fields mirror :class:`vernier_core::tide::report::TideReport` (see
    ``crates/vernier-core/src/tide/report.rs``). The :attr:`config`
    field is the resolved :class:`TideConfig` so a single report tells
    the reader which thresholds it was produced under (ADR-0022).

    Bin names follow the TIDE paper:

    - ``cls`` — Classification: matched a GT of the *wrong* class.
    - ``loc`` — Localization: right class, IoU in ``[t_b, t_f)``.
    - ``both`` — Both classification *and* localization wrong.
    - ``dupe`` — Duplicate: a higher-scoring detection already matched.
    - ``bkg`` — Background: IoU ``< t_b`` against every GT.
    - ``missed`` — Missed GT: no detection survived to match.

    See the debugging tutorial (``docs/tutorials/debugging-with-tide.md``)
    and ADR-0021 for the algorithmic spec.
    """

    baseline_map: float
    #: Per-bin ΔmAP. Keys are the six bin names listed in the class
    #: docstring (``"cls"`` / ``"loc"`` / ``"both"`` / ``"dupe"`` /
    #: ``"bkg"`` / ``"missed"``). The FFI populates all six on every
    #: call (structurally-zero bins surface as ``0.0`` rather than
    #: absent), so direct subscripting is safe. Typed as
    #: ``dict[str, float]`` rather than ``Mapping[Literal[...], float]``
    #: to keep pyright strict happy without a typed-dict cast layer.
    delta: dict[str, float]
    delta_all_fp_removed: float
    config: TideConfig

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> TideReport:
        """Build a :class:`TideReport` from the FFI's dict-shaped output.

        The FFI guarantees all six bin keys are present and that
        ``config`` carries ``(t_f, t_b, kernel)``. Internal helper —
        the supported construction path is :func:`error_decomposition`.
        """
        config_d = d["config"]
        delta_d = d["delta"]
        return cls(
            baseline_map=float(d["baseline_map"]),
            delta={str(k): float(v) for k, v in delta_d.items()},
            delta_all_fp_removed=float(d["delta_all_fp_removed"]),
            config=TideConfig(
                t_f=float(config_d["t_f"]),
                t_b=float(config_d["t_b"]),
                kernel=_kernel_name(config_d["kernel"]),
            ),
        )


def error_decomposition(
    gt: bytes | CocoDataset,
    dt: bytes,
    *,
    iou: object = None,
    t_f: float | None = None,
    t_b: float | None = None,
    max_dets_per_image: int = 100,
    use_cats: bool = True,
    parity_mode: ParityMode = "corrected",
) -> TideReport:
    """TIDE error decomposition (Bolya et al. 2020).

    Splits the gap between a model's measured mAP and the perfect-mAP
    upper bound into six interpretable bins (Cls / Loc / Both / Dupe /
    Bkg / Missed), telling the user *which kind of error* is costing
    them the most points. Eight evaluation passes per call (one
    baseline plus one per bin plus the all-FPs-removed sanity total);
    expect roughly 6x the cost of a single :class:`Evaluator.evaluate`
    call.

    ``gt`` is the GT JSON bytes (the same shape pycocotools'
    ``COCO`` constructor consumes). ``dt`` is the detections JSON
    bytes (the shape ``COCO.loadRes`` consumes). The
    :class:`vernier.CocoDataset` parsed-once handle (ADR-0020) is accepted
    in the type signature for forward-compat but raises
    :class:`NotImplementedError` today — the TIDE FFI is not yet wired
    through the CocoDataset cache. Tracked as a 0.5.x follow-up.

    ``iou`` selects the kernel: ``Bbox()`` (default), ``Segm()``, or
    ``Boundary(dilation_ratio=...)``. ``Keypoints(...)`` raises
    :class:`NotImplementedError` per ADR-0024 — TIDE on OKS has no
    published convention and the Cls/Both bins are structurally empty
    on COCO keypoints (single-class).

    ``t_f`` (foreground / match threshold) and ``t_b`` (background
    threshold) carve the bin assignment. ``None`` resolves to the
    per-kernel defaults from ADR-0022:

    +-----------+--------+--------+
    | Kernel    | ``t_f``| ``t_b``|
    +===========+========+========+
    | bbox      | 0.5    | 0.1    |
    +-----------+--------+--------+
    | segm      | 0.5    | 0.1    |
    +-----------+--------+--------+
    | boundary  | 0.5    | 0.05   |
    +-----------+--------+--------+

    The bbox row matches the TIDE paper; segm and boundary rows are
    defensible-by-extrapolation defaults (segm) and geometry-anchored
    (boundary), both tentative pending the empirical work tracked in
    ADR-0022's "Decision gate" section. Override per call by passing
    explicit ``t_f`` / ``t_b`` floats; the report's :attr:`config`
    records the resolved values either way.

    ``max_dets_per_image`` defaults to ``100`` (the largest rung of the
    standard COCO detection ladder). ``use_cats`` defaults to ``True``
    (per-class evaluation, the COCO standard); set ``False`` for
    class-agnostic decomposition.

    ``parity_mode`` follows :class:`vernier.Evaluator`: ``"corrected"``
    (default) applies vernier's opinionated fixes for known
    pycocotools quirks; ``"strict"`` reproduces pycocotools bit-exactly
    (per ADR-0002).

    Returns a :class:`TideReport` carrying the six per-bin ΔmAP values,
    the baseline mAP, the all-FPs-removed sanity total, and the
    resolved :class:`TideConfig`.

    See the debugging tutorial (``docs/tutorials/debugging-with-tide.md``)
    for a worked example, ADR-0021 for the algorithmic spec, ADR-0022
    for the threshold defaults, and ADR-0024 for the keypoints
    deferral.

    .. note::
       The opt-in ``mode="per_threshold"`` variant of TIDE (10x
       passes, one per IoU threshold in the AP grid) is not exposed in
       0.5.0; planned as a 0.5.x follow-up. The single-``t_f`` form is
       the paper-faithful default. Per-class drill-down on
       :class:`TideReport` is similarly deferred to a 0.5.x follow-up;
       composing this call with :class:`vernier.Evaluator`'s
       ``tables="per_class"`` path is the recommended workaround until
       it lands.
    """
    iou_kind = _resolve_iou(iou)
    kernel = _kernel_for(iou_kind)
    resolved_t_f, resolved_t_b = _defaults_for(kernel)
    if t_f is not None:
        resolved_t_f = t_f
    if t_b is not None:
        resolved_t_b = t_b

    if isinstance(gt, CocoDataset):
        # ADR-0020 wired CocoDataset through Evaluator.evaluate but not yet
        # through TIDE; the FFI surface is bytes-only today. Mirror the
        # NotImplementedError shape Evaluator._evaluate_with_tables
        # uses (__init__.py:296-300) so the boundary is consistent.
        raise NotImplementedError(
            "vernier.error_decomposition does not yet accept a CocoDataset handle; "
            "pass GT JSON bytes for now. CocoDataset support is a 0.5.x follow-up "
            "(the TIDE FFI is not yet wired through the parsed-once cache)."
        )

    raw = _dispatch(
        iou_kind,
        gt,
        dt,
        parity_mode,
        resolved_t_f,
        resolved_t_b,
        max_dets_per_image,
        use_cats,
    )
    # `_from_dict` is documented as an internal classmethod (the supported
    # construction path is this function); the leading underscore on the
    # API spec is intentional so ignore the private-usage warning here.
    return TideReport._from_dict(raw)  # pyright: ignore[reportPrivateUsage]


def _defaults_for(kernel: KernelName) -> tuple[float, float]:
    """Return the per-kernel ``(t_f, t_b)`` defaults from ADR-0022."""
    return _DEFAULT_THRESHOLDS[kernel]


def _resolve_iou(iou: object) -> object:
    """Late-bind ``iou=None`` to ``Bbox()``.

    The default is materialized inside the function rather than at the
    keyword-default site to avoid a top-level import of
    :class:`vernier.Bbox` (which would create a circular import:
    ``vernier`` imports from ``vernier._tide`` and vice versa).
    """
    if iou is None:
        from vernier.instance import Bbox  # local import: see docstring.

        return Bbox()
    return iou


def _kernel_for(iou_kind: object) -> KernelName:
    """Map an :data:`vernier.IouKind` instance to its lowercase
    :data:`KernelName`. Rejects :class:`vernier.Keypoints` per ADR-0024
    and unknown types with a :class:`TypeError`.
    """
    # Local import: avoids a top-level circular import with
    # ``vernier.__init__`` which re-exports :func:`error_decomposition`.
    from vernier.instance import Bbox, Boundary, Keypoints, Segm

    match iou_kind:
        case Bbox():
            return "bbox"
        case Segm():
            return "segm"
        case Boundary():
            return "boundary"
        case Keypoints():
            raise NotImplementedError(
                "TIDE on keypoints (OKS) is deferred per ADR-0024: the Cls/Both "
                "bins are structurally empty on the only multi-class workload "
                "available (COCO keypoints is single-class), and there is no "
                "published TIDE-on-OKS convention. Use the per-keypoint OKS "
                "drill-down planned for a future minor release instead."
            )
        case _:
            _reject_unknown_iou(iou_kind)


def _dispatch(
    iou_kind: object,
    gt: bytes,
    dt: bytes,
    parity_mode: ParityMode,
    t_f: float,
    t_b: float,
    max_dets_per_image: int,
    use_cats: bool,
) -> _FFITideReportDict:
    """Call the right ``vernier._core.error_decomposition_*`` entry."""
    from vernier.instance import Bbox, Boundary, Segm

    match iou_kind:
        case Bbox():
            return error_decomposition_bbox(
                gt,
                dt,
                parity_mode,
                t_f,
                t_b,
                max_dets_per_image,
                use_cats,
            )
        case Segm():
            return error_decomposition_segm(
                gt,
                dt,
                parity_mode,
                t_f,
                t_b,
                max_dets_per_image,
                use_cats,
            )
        case Boundary(dilation_ratio=r):
            return error_decomposition_boundary(
                gt,
                dt,
                parity_mode,
                t_f,
                t_b,
                max_dets_per_image,
                use_cats,
                r,
            )
        case _:
            # _kernel_for rejected this already; keeping the arm for
            # exhaustiveness so adding a kernel later is a clean delta.
            _reject_unknown_iou(iou_kind)


def _kernel_name(value: object) -> KernelName:
    """Narrow an arbitrary object to :data:`KernelName`.

    Used only to type-launder the FFI dict's ``"kernel"`` field — the
    Rust side guarantees one of the three literals (see
    :class:`vernier_core::tide::report::KernelMarker::as_str`), so an
    unexpected value is a serious FFI breakage and we surface it loudly.
    """
    match value:
        case "bbox":
            return "bbox"
        case "segm":
            return "segm"
        case "boundary":
            return "boundary"
        case _:
            raise RuntimeError(
                f"TIDE FFI returned unexpected kernel name {value!r}; expected "
                f"one of 'bbox' / 'segm' / 'boundary'. This is a vernier bug — "
                f"please file an issue."
            )


def _reject_unknown_iou(iou: object) -> NoReturn:
    raise TypeError(
        f"unsupported iou kernel {iou!r} for error_decomposition; expected "
        f"Bbox(), Segm(), or Boundary(...) — see vernier.IouKind. "
        f"Keypoints is deferred per ADR-0024."
    )


@dataclass(frozen=True, slots=True)
class FpIouHistogram:
    """FP-IoU histogram for ADR-0022 `t_b` ratification.

    For every detection that bin assignment classifies as a false
    positive (Cls / Loc / Both / Dupe / Bkg — anything that's not TP
    and not Ignore), :attr:`iou_same` and :attr:`iou_cross` carry the
    best same-class and cross-class IoUs at the time of the bin pick.

    Bin-as-Bkg fraction at a candidate ``t_b`` is::

        max_iou = np.maximum(h.iou_same, h.iou_cross)
        bkg_fraction = (max_iou < t_b).mean()

    Sweeping `t_b` over a range and plotting ``bkg_fraction(t_b)``
    surfaces the "valley" between genuine backgrounds (IoU≈0) and
    near-misses; the right `t_b` sits in that valley.

    See the analysis CLI at
    ``tests/python/integration/real_models/tide/extract_fp_histogram.py``
    for the recipe.
    """

    iou_same: np.ndarray
    iou_cross: np.ndarray
    kernel: KernelName
    t_f: float
    n_total_dts: int
    n_fps: int

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any]) -> FpIouHistogram:
        return cls(
            iou_same=d["iou_same"],
            iou_cross=d["iou_cross"],
            kernel=_kernel_name(d["kernel"]),
            t_f=float(d["t_f"]),
            n_total_dts=int(d["n_total_dts"]),
            n_fps=int(d["n_fps"]),
        )


def fp_iou_histogram(
    gt: bytes | CocoDataset,
    dt: bytes,
    *,
    iou: object = None,
    t_f: float | None = None,
    max_dets_per_image: int = 100,
    use_cats: bool = True,
    parity_mode: ParityMode = "corrected",
) -> FpIouHistogram:
    """Extract per-FP `(iou_same, iou_cross)` for ADR-0022 ratification.

    Sister entry point to :func:`error_decomposition`. Same dispatch
    logic (kernel selection, parity mode, max-dets) but emits the raw
    IoU pairs instead of the six-bin ΔmAP. Caller bins the values
    Python-side to compute the bin-as-Bkg fraction at candidate `t_b`.

    Args:
        gt: GT JSON bytes (CocoDataset handle deferred, same as
            :func:`error_decomposition`).
        dt: Detection JSON bytes.
        iou: Kernel selector — :class:`vernier.Bbox` (default),
            :class:`vernier.Segm`, or :class:`vernier.Boundary`.
            :class:`vernier.Keypoints` raises per ADR-0024.
        t_f: Foreground threshold for identifying TP / Ignore.
            Defaults to `0.5` (ADR-0022 standard); the `t_b` parameter
            on :func:`error_decomposition` is *not* consumed here.
        max_dets_per_image: Per-image detection cap; same default as
            :func:`error_decomposition`.
        use_cats: Per-class evaluation; same default.
        parity_mode: Same as :func:`error_decomposition`.

    Returns:
        :class:`FpIouHistogram` carrying parallel `iou_same` /
        `iou_cross` numpy arrays plus the metadata the report
        consumer needs.
    """
    iou_kind = _resolve_iou(iou)
    kernel = _kernel_for(iou_kind)
    resolved_t_f, _ = _defaults_for(kernel)
    if t_f is not None:
        resolved_t_f = t_f

    if isinstance(gt, CocoDataset):
        raise NotImplementedError(
            "vernier.fp_iou_histogram does not yet accept a CocoDataset handle; "
            "pass GT JSON bytes for now. Mirrors error_decomposition's "
            "0.5.x follow-up."
        )

    raw = _dispatch_histogram(
        iou_kind,
        gt,
        dt,
        parity_mode,
        resolved_t_f,
        max_dets_per_image,
        use_cats,
    )
    return FpIouHistogram._from_dict(raw)  # pyright: ignore[reportPrivateUsage]


def _dispatch_histogram(
    iou_kind: object,
    gt: bytes,
    dt: bytes,
    parity_mode: ParityMode,
    t_f: float,
    max_dets_per_image: int,
    use_cats: bool,
) -> _FFIFpIouHistogramDict:
    """Call the right ``vernier._core.fp_iou_histogram_*`` entry."""
    from vernier.instance import Bbox, Boundary, Segm

    match iou_kind:
        case Bbox():
            return fp_iou_histogram_bbox(gt, dt, parity_mode, t_f, max_dets_per_image, use_cats)
        case Segm():
            return fp_iou_histogram_segm(gt, dt, parity_mode, t_f, max_dets_per_image, use_cats)
        case Boundary(dilation_ratio=r):
            return fp_iou_histogram_boundary(
                gt, dt, parity_mode, t_f, max_dets_per_image, use_cats, r
            )
        case _:
            _reject_unknown_iou(iou_kind)
