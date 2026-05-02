"""Base-vs-head cross-walk over the result tree.

Joins the loaded DataFrame against itself on ``(machine_fp, workload,
iou, impl)`` so each row is "this impl on this cell, base vs head". A
positive ``delta_ns`` means head is slower than base (regression).

Cells that exist in only one side surface as ``"base_only"`` /
``"head_only"`` rows so the consumer can flag missing coverage —
silently dropping them would hide intentional matrix changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

CompareStatus = Literal["ok", "base_only", "head_only"]


@dataclass(frozen=True)
class CompareKey:
    machine_fingerprint: str
    workload_id: str
    iou_type: str
    impl: str


@dataclass(frozen=True)
class CompareRow:
    key: CompareKey
    base_median_ns: int | None
    head_median_ns: int | None
    delta_ns: int | None
    # head/base - 1; None when base is missing/zero.
    delta_relative: float | None
    status: CompareStatus


def _filter_sha(df: pl.DataFrame, git_sha: str) -> pl.DataFrame:
    return df.filter(pl.col("git_sha") == git_sha)


def compare_shas(df: pl.DataFrame, *, base_sha: str, head_sha: str) -> list[CompareRow]:
    """Build a row per ``(machine, workload, iou, impl)`` cell.

    ``df`` is the output of :func:`bench.reports.load.load_tree`. The
    join is full-outer so ``base_only`` and ``head_only`` cells aren't
    silently dropped.
    """
    join_keys = ["machine_fingerprint", "workload_id", "iou_type", "impl"]
    base = (
        _filter_sha(df, base_sha)
        .select([*join_keys, "total_median_ns"])
        .rename({"total_median_ns": "base_median_ns"})
    )
    head = (
        _filter_sha(df, head_sha)
        .select([*join_keys, "total_median_ns"])
        .rename({"total_median_ns": "head_median_ns"})
    )
    joined = base.join(head, on=join_keys, how="full", coalesce=True).sort(join_keys)

    rows: list[CompareRow] = []
    for r in joined.iter_rows(named=True):
        base_ns = r.get("base_median_ns")
        head_ns = r.get("head_median_ns")
        status: CompareStatus
        if base_ns is None and head_ns is not None:
            status = "head_only"
            delta_ns = None
            delta_relative = None
        elif head_ns is None and base_ns is not None:
            status = "base_only"
            delta_ns = None
            delta_relative = None
        else:
            status = "ok"
            base_int = int(base_ns)
            delta_ns = int(head_ns) - base_int
            delta_relative = (delta_ns / base_int) if base_int != 0 else None
        rows.append(
            CompareRow(
                key=CompareKey(
                    machine_fingerprint=str(r["machine_fingerprint"]),
                    workload_id=str(r["workload_id"]),
                    iou_type=str(r["iou_type"]),
                    impl=str(r["impl"]),
                ),
                base_median_ns=int(base_ns) if base_ns is not None else None,
                head_median_ns=int(head_ns) if head_ns is not None else None,
                delta_ns=delta_ns,
                delta_relative=delta_relative,
                status=status,
            )
        )
    return rows
