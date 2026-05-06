"""``report --scaling`` — perf vs a single varying workload parameter.

Companion to :mod:`bench.reports.longitudinal`: where longitudinal
plots time-series across commits at a fixed cell, the scaling view
plots performance across cells whose workload IDs vary in one
parameter (e.g. ``synthetic_n*_c80_g10_d30_s0``) at a fixed commit.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from bench.workloads import synthetic


@dataclass(frozen=True)
class ScalingPoint:
    """One (x, median, iqr, rss) sample on a scaling curve."""

    x_value: float
    median_ns: int
    iqr_ns: int
    ru_maxrss_bytes: int | None


def parse_synthetic_param(workload_id: str, param: str) -> int | None:
    """Pull a single synthetic-workload-id parameter as an int.

    ``None`` for non-synthetic workload IDs, unknown params, or float
    params (only int-typed params are scaling axes).
    """
    if param not in synthetic.SCALING_AXES:
        raise ValueError(
            f"unknown synthetic param {param!r}; choices: {list(synthetic.SCALING_AXES)}"
        )
    parsed = synthetic.parse_workload_id(workload_id)
    if parsed is None:
        return None
    val = parsed.get(param)
    return val if isinstance(val, int) else None


def group_by_synthetic_param(
    df: pl.DataFrame,
    *,
    vary: str,
    fix: dict[str, int],
    iou_type: str,
) -> dict[str, list[ScalingPoint]]:
    """Slice ``df`` to a single (vary-param) curve per impl.

    Rows without ``total_median_ns`` are dropped silently — incomplete
    cells shouldn't crash a report render.
    """
    if df.is_empty():
        return {}

    out: dict[str, list[ScalingPoint]] = {}
    for r in df.iter_rows(named=True):
        if r["iou_type"] != iou_type or r["total_median_ns"] is None:
            continue
        wid = str(r["workload_id"])
        parsed = synthetic.parse_workload_id(wid)
        if parsed is None or vary not in parsed:
            continue
        if not all(parsed.get(fk) == fv for fk, fv in fix.items()):
            continue
        rss = r["ru_maxrss_median_bytes"]
        point = ScalingPoint(
            x_value=float(parsed[vary]),
            median_ns=int(r["total_median_ns"]),
            iqr_ns=int(r["total_iqr_ns"]) if r["total_iqr_ns"] is not None else 0,
            ru_maxrss_bytes=int(rss) if rss is not None else None,
        )
        out.setdefault(str(r["impl"]), []).append(point)

    for points in out.values():
        points.sort(key=lambda p: p.x_value)
    return out
