"""``report --since`` — perf over time on a single machine.

Filters the result tree by mtime cutoff and groups by ``(workload, iou,
impl)`` so each series is a stable cell across commits. The mtime
proxy for "when was this run captured" is good enough at v1 — the
schema doesn't carry a wall-clock timestamp and adding one is a v2
change. ``mtime`` is monotonic per file on most filesystems; if the
result tree is rsync'd between machines, callers should re-stat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import polars as pl

_DURATION_RE = re.compile(r"^(\d+)([dhmw])$")
_DURATION_UNITS_SECONDS: dict[str, int] = {
    "m": 60,
    "h": 60 * 60,
    "d": 60 * 60 * 24,
    "w": 60 * 60 * 24 * 7,
}


def parse_since(spec: str) -> timedelta:
    """``"30d"`` → ``timedelta(days=30)``; ``"6h"`` → 6 hours; ``"2w"`` → 14 days."""
    match = _DURATION_RE.match(spec)
    if not match:
        raise ValueError(f"--since must be <int><unit> with unit in d/h/m/w; got {spec!r}")
    n = int(match.group(1))
    unit = match.group(2)
    return timedelta(seconds=n * _DURATION_UNITS_SECONDS[unit])


@dataclass(frozen=True)
class SeriesKey:
    machine_fingerprint: str
    workload_id: str
    iou_type: str
    impl: str


@dataclass(frozen=True)
class SeriesPoint:
    timestamp: datetime
    git_sha: str
    median_ns: int
    iqr_ns: int


def filter_since(
    df: pl.DataFrame, *, since: timedelta, now: datetime | None = None
) -> pl.DataFrame:
    """Drop rows older than ``now - since``. ``now`` is parameterised for tests."""
    cutoff = (now or datetime.now(tz=timezone.utc)).timestamp() - since.total_seconds()
    return df.filter(pl.col("mtime") >= cutoff)


def build_series(df: pl.DataFrame) -> dict[SeriesKey, list[SeriesPoint]]:
    """Group rows into time-ordered series keyed by ``(machine, workload, iou, impl)``."""
    series: dict[SeriesKey, list[SeriesPoint]] = {}
    for r in df.iter_rows(named=True):
        if r["total_median_ns"] is None:
            continue
        key = SeriesKey(
            machine_fingerprint=str(r["machine_fingerprint"]),
            workload_id=str(r["workload_id"]),
            iou_type=str(r["iou_type"]),
            impl=str(r["impl"]),
        )
        point = SeriesPoint(
            timestamp=datetime.fromtimestamp(float(r["mtime"]), tz=timezone.utc),
            git_sha=str(r["git_sha"]),
            median_ns=int(r["total_median_ns"]),
            iqr_ns=int(r["total_iqr_ns"]) if r["total_iqr_ns"] is not None else 0,
        )
        series.setdefault(key, []).append(point)

    for points in series.values():
        points.sort(key=lambda p: p.timestamp)
    return series
