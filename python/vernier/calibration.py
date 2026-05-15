"""ADR-0018 detection-family calibration summarizer.

User-facing wrapper sitting on top of the Rust kernel
(:mod:`vernier_core::calibration`) and the FFI layer
(:class:`vernier._core.EvalCells`). ``result.calibration(...)`` is the
single public entry point on the batch surface;
:meth:`StreamingSnapshot.calibration` is the streaming sibling
(ADR-0018 Unit 6). :class:`CalibrationResult` is not re-exported from
:mod:`vernier` per the "one canonical path per item" rule.

The reliability and per-class tables are exposed lazily as
:class:`polars.DataFrame` via ``@cached_property``, mirroring
:class:`vernier.instance.EvalResult`. The underlying Arrow
``RecordBatch`` is retained under ``_reliability_batch`` /
``_per_class_batch`` for zero-copy consumers; the documented bridge
is the :meth:`ArrowRecordBatch.__arrow_c_array__` PyCapsule method.

See ``docs/adr/0018-calibration.md`` and the parity oracle at
``tests/python/parity_calibration/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Literal

from vernier._tables import arrow_to_dataframe

if TYPE_CHECKING:  # pragma: no cover — type-checker only
    import polars as pl

    from vernier._core import BackgroundEvaluator, EvalCells, Summary

#: Bin-edge strategy for the reliability table. ``"quantile"`` mirrors
#: the kernel's default (parity-pinned to ``numpy.quantile(method='linear')``,
#: quirk **P1**); ``"equal_width"`` uses ``[0, 1]`` divided into
#: ``n_bins`` evenly spaced buckets.
Binning = Literal["quantile", "equal_width"]

#: Per-bin confidence-interval flavor. ``"wilson"`` is the parity-pinned
#: default (quirk **P4**); ``"clopper_pearson"`` is the conservative
#: exact-binomial alternative.
Confidence = Literal["wilson", "clopper_pearson"]

#: Per-class aggregation kind for the marginal ECE/MCE rollup.
#: ``"macro"`` is the parity-pinned default (quirk **P6**) — unweighted
#: mean across classes; ``"micro"`` pools detections globally.
Aggregation = Literal["macro", "micro"]


@dataclass(frozen=True)
class CalibrationResult:
    """Result of :meth:`vernier.instance.EvalResult.calibration`.

    Scalars are plain Python numbers; tabular outputs are exposed as
    :class:`polars.DataFrame` via lazy ``@cached_property`` (mirrors
    :class:`vernier.instance.EvalResult`). The underlying Arrow
    ``RecordBatch`` is retained under ``_reliability_batch`` /
    ``_per_class_batch`` for callers that want the zero-copy Arrow
    surface; :meth:`__arrow_c_array__` is the documented bridge.

    ``effective_n_bins`` is the bin count *after* duplicate-edge
    merging on small samples (quirk **P5**); it may be less than the
    requested ``n_bins`` and is what ``reliability.height`` reports.
    """

    #: Expected Calibration Error — score-weighted mean per-bin gap.
    ece: float
    #: Maximum Calibration Error — worst per-bin gap.
    mce: float
    #: Total detection count behind the histogram (after the
    #: ``min_score`` cutoff and ignore-region exclusion).
    n_detections: int
    #: Bin count after duplicate-edge merging; ``<= n_bins``.
    effective_n_bins: int
    #: Arrow ``RecordBatch`` capsule producer for the reliability table.
    #: Implementation detail; use :attr:`reliability` for a polars view.
    _reliability_batch: object = field(repr=False)
    #: Arrow ``RecordBatch`` capsule producer for the per-class table.
    #: ``None`` when ``per_class=False`` was passed.
    _per_class_batch: object | None = field(default=None, repr=False)

    @cached_property
    def reliability(self) -> pl.DataFrame:
        """One row per effective bin. Columns: ``bin_id``, ``score_lo``,
        ``score_hi``, ``mean_score``, ``accuracy``, ``count``, ``gap``,
        ``ci_lo``, ``ci_hi``. Zero-count bins emit ``NaN`` for the float
        columns per the kernel's R2 convention."""
        return arrow_to_dataframe(self._reliability_batch, "calibration_reliability")

    @cached_property
    def per_class(self) -> pl.DataFrame:
        """One row per class. Columns: ``class_id``, ``ece``, ``mce``,
        ``n``. Raises :class:`RuntimeError` when ``per_class=True`` was
        not requested."""
        if self._per_class_batch is None:
            raise RuntimeError(
                "per_class table is unavailable: call result.calibration(per_class=True)"
            )
        return arrow_to_dataframe(self._per_class_batch, "calibration_per_class")

    def worst(self, k: int = 5) -> pl.DataFrame:
        """Top-``k`` worst-calibrated classes by ECE. Raises
        :class:`RuntimeError` when the per-class table wasn't requested.
        """
        return self.per_class.sort("ece", descending=True).head(k)


def _calibrate_from_cells(
    cells: EvalCells,
    *,
    iou: float,
    n_bins: int,
    binning: Binning,
    min_score: float,
    confidence: Confidence,
    per_class: bool,
    per_class_aggregation: Aggregation,
) -> CalibrationResult:
    """Resolve ``iou`` against the cell-store's pinned T-axis and fold
    into a :class:`CalibrationResult`.

    Shared helper called by both :meth:`vernier.instance.EvalResult.calibration`
    (batch path) and :meth:`StreamingSnapshot.calibration` (ADR-0018
    Unit 6 streaming path) — the post-cells fold is identical across
    both surfaces, so there's exactly one place that knows the kernel's
    return-tuple layout.
    """
    iou_index = cells.iou_to_index(iou)
    ece, mce, n_det, eff_bins, reliability_b, per_class_b = cells.calibrate(
        iou_index,
        n_bins,
        binning,
        min_score,
        confidence,
        per_class,
        per_class_aggregation,
    )
    return CalibrationResult(
        ece=ece,
        mce=mce,
        n_detections=n_det,
        effective_n_bins=eff_bins,
        _reliability_batch=reliability_b,
        _per_class_batch=per_class_b,
    )


@dataclass(frozen=True)
class StreamingSnapshot:
    """Bundle returned by
    :meth:`StreamingSnapshot.from_background` — pairs the canonical
    :class:`vernier._core.Summary` produced by the streaming evaluator
    with the opaque :class:`vernier._core.EvalCells` handle the
    calibration summarizer consumes (ADR-0018 Unit 6).

    The :attr:`summary` axis is bit-identical to what
    :meth:`vernier._core.BackgroundEvaluator.finalize` would have
    produced for the same evaluator state; calibration adds the cells
    alongside, never mutates the canonical kernel maths.

    Mirrors the :class:`vernier.instance.EvalResult` convention: the
    raw cell handle is kept under :attr:`_eval_cells` (impl detail —
    do not depend on its shape) and the :meth:`calibration` method
    folds it into a :class:`CalibrationResult` with the same params
    the batch surface accepts.
    """

    #: Canonical :class:`Summary` from the streaming evaluator —
    #: bit-identical to :meth:`BackgroundEvaluator.finalize` on the
    #: same state.
    summary: Summary
    #: Opaque cell-store handle (the :class:`EvalCells` pyclass). Not
    #: re-exported from :mod:`vernier`; consume via :meth:`calibration`.
    _eval_cells: EvalCells = field(repr=False)

    @classmethod
    def from_background(cls, bg: BackgroundEvaluator) -> StreamingSnapshot:
        """Consume a :class:`vernier._core.BackgroundEvaluator` via
        :meth:`finalize_with_cells` and wrap the returned tuple into a
        :class:`StreamingSnapshot`.

        The background evaluator follows finalize-consumes-the-worker
        semantics (ADR-0014, ADR-0035); a second call to any
        ``finalize_*`` method raises "already finalized". Use
        :meth:`calibration` on the returned snapshot to fold the
        retained cell store into a :class:`CalibrationResult`.
        """
        summary, cells = bg.finalize_with_cells()
        return cls(summary=summary, _eval_cells=cells)

    def calibration(
        self,
        *,
        iou: float = 0.5,
        n_bins: int = 15,
        binning: Binning = "quantile",
        min_score: float = 0.05,
        confidence: Confidence = "wilson",
        per_class: bool = False,
        per_class_aggregation: Aggregation = "macro",
    ) -> CalibrationResult:
        """Fold the retained streaming cell store into a calibration
        summary (ADR-0018 Unit 6).

        Mirrors :meth:`vernier.instance.EvalResult.calibration` keyword
        for keyword — same parity-pinned defaults, same return type;
        re-folding with different params is cheap (no re-matching).
        ``iou`` resolves to the kernel's T-axis index under
        :data:`vernier_core::parity::PARITY_EPS`; values that don't
        land on a pinned threshold raise :class:`ValueError`.
        """
        return _calibrate_from_cells(
            self._eval_cells,
            iou=iou,
            n_bins=n_bins,
            binning=binning,
            min_score=min_score,
            confidence=confidence,
            per_class=per_class,
            per_class_aggregation=per_class_aggregation,
        )
