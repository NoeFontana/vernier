"""Cross-paradigm compare requests are rejected with a clear message
(ADR-0032 §"Cross-paradigm category error", ADR-0033 §"Comparator
registry").

The bench's ``compare`` operates per-paradigm. Asking for, e.g.,
``--paradigm instance --metric pq`` mixes the AP-fold metric units
with the PQ metric units; the result-units don't compose. The rejection
happens at the request-validation layer before any DataFrame slicing.
"""

from __future__ import annotations

import pytest

from bench.reports.compare import (
    CrossParadigmCompareError,
    reject_cross_paradigm_request,
)


@pytest.mark.parametrize(
    ("paradigm", "metric"),
    [
        ("instance", "pq"),
        ("instance", "miou"),
        ("instance", "throughput"),
        ("panoptic", "bbox"),
        ("panoptic", "miou"),
        ("semantic", "pq"),
        ("semantic", "bbox"),
        ("streaming", "pq"),
        ("streaming", "miou"),
    ],
)
def test_cross_paradigm_combinations_rejected(paradigm, metric) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(CrossParadigmCompareError) as exc_info:
        reject_cross_paradigm_request(paradigm=paradigm, metric=metric)
    msg = str(exc_info.value)
    assert metric in msg
    assert paradigm in msg
    # The error message must point at ADR-0032 so users can find the
    # rationale rather than guessing at which side was wrong.
    assert "ADR-0032" in msg


@pytest.mark.parametrize(
    ("paradigm", "metric"),
    [
        ("instance", "bbox"),
        ("instance", "segm"),
        ("instance", "keypoints"),
        ("instance", "boundary"),
        ("panoptic", "pq"),
        ("semantic", "miou"),
        ("streaming", "throughput"),
        ("streaming", "p99"),
        ("streaming", "rss"),
    ],
)
def test_native_paradigm_metric_pairs_accepted(paradigm, metric) -> None:  # type: ignore[no-untyped-def]
    """Each metric belongs to exactly one paradigm; the matching pair
    must pass through without raising."""
    reject_cross_paradigm_request(paradigm=paradigm, metric=metric)


def test_unknown_metric_is_passthrough() -> None:
    """Unknown metrics fall through to other validators (e.g.
    ``IMPL_PARADIGM_SUPPORT`` at runtime). Rejection here is
    *category* error, not exhaustiveness."""
    reject_cross_paradigm_request(paradigm="instance", metric="not_a_metric_yet")
