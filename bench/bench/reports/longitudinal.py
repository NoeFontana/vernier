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

from bench.harness.schema import Paradigm

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
    ru_maxrss_bytes: int | None = None


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
        rss = r.get("ru_maxrss_median_bytes")
        point = SeriesPoint(
            timestamp=datetime.fromtimestamp(float(r["mtime"]), tz=timezone.utc),
            git_sha=str(r["git_sha"]),
            median_ns=int(r["total_median_ns"]),
            iqr_ns=int(r["total_iqr_ns"]) if r["total_iqr_ns"] is not None else 0,
            ru_maxrss_bytes=int(rss) if rss is not None else None,
        )
        series.setdefault(key, []).append(point)

    for points in series.values():
        points.sort(key=lambda p: p.timestamp)
    return series


@dataclass(frozen=True)
class ParadigmSeriesSummary:
    """Per-paradigm aggregate over its time-series.

    Carries a few headline stats so the longitudinal report can show
    "panoptic median moved from X to Y over the window" without the
    consumer having to walk every series itself. ``n_series`` is the
    distinct ``(workload, iou, impl)`` count; ``earliest`` / ``latest``
    are the bracket of the window for this paradigm; ``median_ns`` is
    the median of every point's median (a coarse, robust summary —
    finer breakdowns are per-series).
    """

    paradigm: Paradigm
    n_series: int
    n_points: int
    earliest: datetime | None
    latest: datetime | None
    # Median across every series' median; ``None`` for an empty
    # paradigm. Coarse on purpose — the per-series tables drill in.
    median_ns: int | None


def build_series_per_paradigm(
    df: pl.DataFrame,
) -> dict[Paradigm, dict[SeriesKey, list[SeriesPoint]]]:
    """One ``build_series`` call per paradigm present in ``df``.

    A v1-only tree (no ``paradigm`` column) routes everything under
    ``"instance"`` — matches the read-side shim and keeps detection
    callers working unchanged.

    Paradigms with zero matching rows are omitted; the report renderer
    iterates the result keys.
    """
    if df.is_empty():
        return {}

    if "paradigm" not in df.columns:
        return {"instance": build_series(df)}

    out: dict[Paradigm, dict[SeriesKey, list[SeriesPoint]]] = {}
    for p in sorted(df["paradigm"].unique().to_list()):
        if p is None:
            continue
        df_p = df.filter(df["paradigm"] == p)
        series = build_series(df_p)
        if series:
            out[p] = series
    return out


def summarize_paradigm(
    paradigm: Paradigm, series: dict[SeriesKey, list[SeriesPoint]]
) -> ParadigmSeriesSummary:
    """Roll one paradigm's series dict into a one-line summary.

    Uses ``statistics.median`` rather than NumPy to avoid pulling
    NumPy into the report-only path; the input is small (one int per
    series×point) so the pure-Python version is fast enough.
    """
    points = [p for plist in series.values() for p in plist]
    if not points:
        return ParadigmSeriesSummary(
            paradigm=paradigm,
            n_series=0,
            n_points=0,
            earliest=None,
            latest=None,
            median_ns=None,
        )
    medians = sorted(p.median_ns for p in points)
    mid = medians[len(medians) // 2]
    earliest = min(p.timestamp for p in points)
    latest = max(p.timestamp for p in points)
    return ParadigmSeriesSummary(
        paradigm=paradigm,
        n_series=len(series),
        n_points=len(points),
        earliest=earliest,
        latest=latest,
        median_ns=mid,
    )
