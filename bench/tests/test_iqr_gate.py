"""IQR-relative-to-median gate (ADR-0017 §"Run modes").

The gate is what turns a noisy machine into an aborted run rather than
a result the operator believes too easily. Tests pin: the math, the
threshold semantics, and the aggregation contract that wraps it.
"""

from __future__ import annotations

import pytest

from bench.harness.schema import RepResult, StageTimings
from bench.harness.stats import (
    DEFAULT_IQR_RELATIVE_THRESHOLD,
    aggregate_reps,
    aggregate_stage,
    iqr_gate,
)


def _rep(rep: int, total_ns: int, *, warmup: bool = False) -> RepResult:
    return RepResult(
        rep=rep,
        warmup=warmup,
        stages={"total": StageTimings(wall_ns=total_ns)},
        summary_stats={"AP": 1.0},
        ru_maxrss_bytes=1,
        parent_wall_ns=total_ns,
    )


def test_aggregate_stage_matches_numpy_quartiles() -> None:
    s = aggregate_stage([100, 110, 120, 130, 140, 150, 160, 170, 180, 200])
    assert s.median_ns == 145
    # Q1=122.5, Q3=167.5 → IQR=45
    assert s.iqr_ns == 45
    assert s.min_ns == 100
    assert s.max_ns == 200


def test_iqr_gate_passes_under_threshold() -> None:
    # Tight cluster around 1000 ns: IQR / median well under 5%.
    reps = [_rep(i, 1000 + i) for i in range(10)]
    aggr = aggregate_reps(reps)
    outcome = iqr_gate(aggr)
    assert outcome.passed
    assert outcome.relative < DEFAULT_IQR_RELATIVE_THRESHOLD


def test_iqr_gate_fails_over_threshold() -> None:
    # Wide spread: median ~1000, IQR ~400 → 40% relative.
    spread = [500, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    reps = [_rep(i, ns) for i, ns in enumerate(spread)]
    aggr = aggregate_reps(reps)
    outcome = iqr_gate(aggr)
    assert not outcome.passed
    assert outcome.relative > DEFAULT_IQR_RELATIVE_THRESHOLD


def test_iqr_gate_threshold_is_tunable() -> None:
    reps = [_rep(i, ns) for i, ns in enumerate([900, 950, 1000, 1050, 1100])]
    aggr = aggregate_reps(reps)
    # ~10% spread; passes a 20% threshold, fails a 1% threshold.
    assert iqr_gate(aggr, threshold=0.20).passed
    assert not iqr_gate(aggr, threshold=0.01).passed


def test_aggregate_reps_excludes_warmup() -> None:
    """A warmup rep with wildly different timings must not feed the IQR."""
    reps = [
        _rep(0, 999_999, warmup=True),
        *(_rep(i + 1, 1000) for i in range(5)),
    ]
    aggr = aggregate_reps(reps)
    assert aggr["total"].median_ns == 1000
    assert aggr["total"].iqr_ns == 0


def test_aggregate_reps_requires_a_measurement_rep() -> None:
    with pytest.raises(ValueError, match="at least one non-warmup rep"):
        aggregate_reps([_rep(0, 1000, warmup=True)])
