"""``_StreamingComparator`` parity-gate tests.

The comparator is paradigm-keyed; B3's contract is bit-equal
``Summary.stats`` between batch and stream paths (per ADR-0032 +
``tests/python/parity/streaming/test_streaming_finalize_equals_batch.py``).
Throughput numbers and RSS curves are informational and don't gate.
"""

from __future__ import annotations

import pytest

from bench.harness.parity import (
    STREAMING_PARITY_ATOL,
    StreamingPair,
    get_comparator,
)


def _stats_dict(values: list[float]) -> dict[str, float]:
    """Match the runner's ``stat_<i>`` convention so comparator parsing
    sees the same keys it would see in production."""
    return {f"stat_{i}": v for i, v in enumerate(values)}


def test_streaming_comparator_passes_on_bit_equal_summaries() -> None:
    pair = StreamingPair(
        batch_summary=_stats_dict([0.5, 0.6, 0.7]),
        stream_summary=_stats_dict([0.5, 0.6, 0.7]),
    )
    cmp = get_comparator("streaming")
    report = cmp.compare(
        workload_id="coco_val2017_streaming_throughput",
        iou_type="bbox",
        impl_outputs={"vernier_streaming": pair},
    )
    assert report.passed
    assert len(report.tiers) == 1
    tier = report.tiers[0]
    assert tier.divergent_count == 0
    assert tier.impl_a == "vernier_streaming/batch"
    assert tier.impl_b == "vernier_streaming/stream"


def test_streaming_comparator_fails_on_divergent_summaries() -> None:
    pair = StreamingPair(
        batch_summary=_stats_dict([0.5, 0.6, 0.7]),
        stream_summary=_stats_dict([0.5, 0.6, 0.700001]),  # >> 1e-12
    )
    cmp = get_comparator("streaming")
    report = cmp.compare(
        workload_id="coco_val2017_streaming_throughput",
        iou_type="bbox",
        impl_outputs={"vernier_streaming": pair},
    )
    assert not report.passed
    tier = report.tiers[0]
    assert tier.divergent_count == 1
    assert tier.first_divergence is not None
    assert tier.first_divergence.index == (2,)
    assert tier.first_divergence.abs_diff > STREAMING_PARITY_ATOL


def test_streaming_comparator_tolerates_sub_ulp_wobble() -> None:
    """ULP-tier divergence within the documented atol must pass — the
    streaming-vs-batch parity test pins this same disposition."""
    pair = StreamingPair(
        batch_summary=_stats_dict([0.5]),
        # 1e-13 is below STREAMING_PARITY_ATOL (1e-12).
        stream_summary=_stats_dict([0.5 + 1e-13]),
    )
    cmp = get_comparator("streaming")
    report = cmp.compare(
        workload_id="x", iou_type="bbox", impl_outputs={"vernier_streaming": pair}
    )
    assert report.passed


def test_streaming_comparator_cross_impl_pairing() -> None:
    """vs-naive cell: two impls each carry a ``StreamingPair``; the
    comparator pairs ``batch_summary`` across the two."""
    a = StreamingPair(
        batch_summary=_stats_dict([0.5, 0.6]),
        stream_summary=_stats_dict([0.5, 0.6]),
    )
    b = StreamingPair(
        batch_summary=_stats_dict([0.5, 0.6]),
        stream_summary={},  # naive_python doesn't have a stream half
    )
    cmp = get_comparator("streaming")
    report = cmp.compare(
        workload_id="coco_val2017_streaming_vs_naive",
        iou_type="bbox",
        impl_outputs={"vernier_streaming": a, "naive_python": b},
    )
    assert report.passed
    # One internal (vernier_streaming batch-vs-stream) + one cross
    # (vernier_streaming/naive_python) tier.
    assert len(report.tiers) == 2


def test_streaming_comparator_rejects_wrong_artifact_type() -> None:
    """A panoptic snapshot fed to the streaming comparator must surface
    a clear ``ValueError`` — proves the dispatcher doesn't silently
    miscompare."""
    from bench.harness.parity import PanopticSnapshot

    snap = PanopticSnapshot(pq=0.5)
    cmp = get_comparator("streaming")
    with pytest.raises(ValueError, match="StreamingPair"):
        cmp.compare(
            workload_id="x", iou_type="bbox", impl_outputs={"x": snap}
        )


def test_streaming_pair_canonical_form_includes_both_summaries() -> None:
    pair = StreamingPair(
        batch_summary={"stat_0": 0.5},
        stream_summary={"stat_0": 0.5},
        rss_curve_paths={"vernier": "rss.json"},
    )
    canon = pair.to_canonical_form()
    assert canon["kind"] == "streaming_pair"
    assert canon["batch_summary"] == {"stat_0": 0.5}
    assert canon["stream_summary"] == {"stat_0": 0.5}
    assert canon["rss_curve_paths"] == {"vernier": "rss.json"}
