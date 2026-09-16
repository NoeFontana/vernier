"""StageTable evidence fields: CPU time, observed affinity, and exact
per-stage RSS peaks."""

from __future__ import annotations

import os

from bench.harness.timing import StageTable


def test_record_total_sums_cpu_and_notes_affinity() -> None:
    stages = StageTable()
    with stages.stage("load"):
        sum(range(10_000))
    with stages.stage("evaluate"):
        sum(range(10_000))
    stages.record_total()
    table = stages.to_dict()
    total = table["total"]
    assert total.wall_ns == table["load"].wall_ns + table["evaluate"].wall_ns
    assert total.cpu_ns == (table["load"].cpu_ns or 0) + (table["evaluate"].cpu_ns or 0)
    expected = ",".join(str(c) for c in sorted(os.sched_getaffinity(0)))
    assert total.notes == [f"cpu_affinity={expected}"]


def test_stage_peak_rss_is_per_stage_not_lifetime() -> None:
    """A large allocation freed before a later stage must not inflate that
    stage's peak: ``VmHWM`` is reset at every stage start."""
    stages = StageTable()
    big = 256 << 20
    with stages.stage("load"):
        buf = bytearray(big)
        buf[:: 1 << 12] = b"x" * len(buf[:: 1 << 12])  # fault every page in
        del buf
    with stages.stage("evaluate"):
        sum(range(10_000))
    stages.record_total()
    table = stages.to_dict()
    load, evaluate, total = table["load"], table["evaluate"], table["total"]
    assert load.peak_rss_bytes is not None
    assert load.rss_start_bytes is not None
    assert evaluate.peak_rss_bytes is not None
    assert evaluate.rss_start_bytes is not None
    assert load.peak_rss_bytes - load.rss_start_bytes >= big * 0.9
    assert evaluate.peak_rss_bytes - evaluate.rss_start_bytes < big // 4
    assert total.rss_start_bytes == load.rss_start_bytes
    assert total.peak_rss_bytes == max(load.peak_rss_bytes, evaluate.peak_rss_bytes)
