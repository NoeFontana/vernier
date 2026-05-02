"""Per-stage aggregation helpers (median / IQR / min / max).

The release-mode IQR-relative-to-median gate fires when the spread on
the ``total`` stage exceeds a configurable fraction of the median —
the signal that the machine is too noisy for the result to mean
anything (ADR-0017 §"Run modes").
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from bench.harness.schema import IqrGateResult, RepResult, StageAggregation

# 5% of the median is the ADR's documented gate. Tunable per machine in
# a future config file; for now it's a single shared default.
DEFAULT_IQR_RELATIVE_THRESHOLD: float = 0.05


def aggregate_stage(values_ns: Sequence[int]) -> StageAggregation:
    """``values_ns`` is the per-rep wall_ns for a single stage."""
    if not values_ns:
        raise ValueError("aggregate_stage requires at least one rep")
    arr = np.asarray(values_ns, dtype=np.int64)
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    return StageAggregation(
        median_ns=int(median),
        iqr_ns=int(q3 - q1),
        min_ns=int(arr.min()),
        max_ns=int(arr.max()),
    )


def aggregate_reps(reps: Iterable[RepResult]) -> dict[str, StageAggregation]:
    """Per-stage aggregation across measurement reps. Warmup reps are filtered."""
    measurements = [r for r in reps if not r.warmup]
    if not measurements:
        raise ValueError("aggregate_reps requires at least one non-warmup rep")
    stage_names: set[str] = set()
    for r in measurements:
        stage_names.update(r.stages)
    return {
        name: aggregate_stage([r.stages[name].wall_ns for r in measurements if name in r.stages])
        for name in sorted(stage_names)
    }


def iqr_gate(
    aggregation: dict[str, StageAggregation],
    *,
    stage: str = "total",
    threshold: float = DEFAULT_IQR_RELATIVE_THRESHOLD,
) -> IqrGateResult:
    """Apply the IQR-relative-to-median gate on ``stage`` (default: total)."""
    if stage not in aggregation:
        raise KeyError(f"stage {stage!r} not in aggregation: {sorted(aggregation)}")
    s = aggregation[stage]
    relative = s.iqr_ns / s.median_ns if s.median_ns > 0 else float("inf")
    return IqrGateResult(
        stage=stage,
        relative=relative,
        threshold=threshold,
        passed=relative <= threshold,
    )
