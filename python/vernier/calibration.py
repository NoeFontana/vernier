"""ADR-0018 detection-family calibration summarizer.

User-facing wrapper sitting on top of the Rust kernel
(:mod:`vernier_core::calibration`) and the FFI layer
(:class:`vernier._core.EvalCells`). ``result.calibration(...)`` is the
single public entry point — :class:`CalibrationResult` is not
re-exported from :mod:`vernier` per the "one canonical path per item"
rule.

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
