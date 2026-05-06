"""Base-vs-head cross-walk over the result tree.

Joins the loaded DataFrame against itself on ``(machine_fp, workload,
iou, impl)`` so each row is "this impl on this cell, base vs head". A
positive ``delta_ns`` means head is slower than base (regression).

Cells that exist in only one side surface as ``"base_only"`` /
``"head_only"`` rows so the consumer can flag missing coverage —
silently dropping them would hide intentional matrix changes.

Per ADR-0033 + ADR-0032, the compare always scopes per-paradigm. The
:func:`compare_shas_per_paradigm` driver groups rows by paradigm and
returns one section per paradigm; cross-paradigm comparison
(``--paradigm instance --metric pq``) is rejected by
:func:`reject_cross_paradigm_request` — the metric units don't compose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from bench.harness.schema import Paradigm

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
    base_ru_maxrss_bytes: int | None = None
    head_ru_maxrss_bytes: int | None = None


def _filter_sha(df: pl.DataFrame, git_sha: str) -> pl.DataFrame:
    return df.filter(pl.col("git_sha") == git_sha)


def compare_shas(df: pl.DataFrame, *, base_sha: str, head_sha: str) -> list[CompareRow]:
    """Build a row per ``(machine, workload, iou, impl)`` cell.

    ``df`` is the output of :func:`bench.reports.load.load_tree`. The
    join is full-outer so ``base_only`` and ``head_only`` cells aren't
    silently dropped.
    """
    join_keys = ["machine_fingerprint", "workload_id", "iou_type", "impl"]
    select_cols = [*join_keys, "total_median_ns", "ru_maxrss_median_bytes"]
    base = (
        _filter_sha(df, base_sha)
        .select(select_cols)
        .rename(
            {
                "total_median_ns": "base_median_ns",
                "ru_maxrss_median_bytes": "base_ru_maxrss_bytes",
            }
        )
    )
    head = (
        _filter_sha(df, head_sha)
        .select(select_cols)
        .rename(
            {
                "total_median_ns": "head_median_ns",
                "ru_maxrss_median_bytes": "head_ru_maxrss_bytes",
            }
        )
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
        base_rss = r.get("base_ru_maxrss_bytes")
        head_rss = r.get("head_ru_maxrss_bytes")
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
                base_ru_maxrss_bytes=int(base_rss) if base_rss is not None else None,
                head_ru_maxrss_bytes=int(head_rss) if head_rss is not None else None,
            )
        )
    return rows


# Metric ↔ paradigm map used by ``reject_cross_paradigm_request``. The
# dispatch is structurally rejected (ADR-0032 §"Cross-paradigm category
# error"); listing the canonical metric per paradigm makes the error
# message diagnostic — "you asked for ``pq`` under ``instance``" rather
# than a generic "wrong combination".
_METRIC_HOME: dict[str, Paradigm] = {
    "bbox": "instance",
    "segm": "instance",
    "keypoints": "instance",
    "boundary": "instance",
    "pq": "panoptic",
    "miou": "semantic",
    "throughput": "streaming",
    "p99": "streaming",
    "rss": "streaming",
}


class CrossParadigmCompareError(ValueError):
    """Raised when a compare request mixes paradigms (ADR-0032).

    Examples that trigger this:

    - ``--paradigm instance --metric pq`` (PQ belongs to panoptic)
    - ``--paradigm panoptic --metric bbox``

    The error message names the offending pair so the user can fix
    whichever side was wrong.
    """


def reject_cross_paradigm_request(*, paradigm: Paradigm, metric: str) -> None:
    """Raise ``CrossParadigmCompareError`` if ``metric`` doesn't live
    in ``paradigm``'s metric set.

    A no-op for unknown metrics — those fall through other validators
    (``IMPL_PARADIGM_SUPPORT`` for runtime dispatch). The point here
    is the *category* error, not the exhaustiveness check.
    """
    home = _METRIC_HOME.get(metric)
    if home is None:
        return
    if home != paradigm:
        raise CrossParadigmCompareError(
            f"metric {metric!r} belongs to paradigm {home!r}, not {paradigm!r}; "
            f"per ADR-0032 the result-units don't compose across paradigms — "
            f"either drop --metric or pass --paradigm {home!r}."
        )


@dataclass(frozen=True)
class ParadigmCompareSection:
    """One paradigm's slice of a compare run.

    ``rows`` is the list of (machine, workload, iou, impl) cells the
    cross-walk produced for this paradigm; ``cells`` carries the
    matched workload-ids (used by report-fragment renderers that need
    paradigm-shaped context, e.g., a panoptic per-class table).
    """

    paradigm: Paradigm
    rows: list[CompareRow]


def compare_shas_per_paradigm(
    df: pl.DataFrame, *, base_sha: str, head_sha: str
) -> list[ParadigmCompareSection]:
    """Build one :class:`ParadigmCompareSection` per paradigm present in ``df``.

    Cross-paradigm rows never compose: each section is computed
    independently from the rows tagged with that paradigm. A v1-only
    tree (no ``paradigm`` column) is treated as all-instance — this
    matches the read-side migration shim's ``paradigm="instance"``
    default and keeps detection-only callers working unchanged.

    Empty paradigms (no rows for that paradigm in either ``base_sha``
    or ``head_sha``) are omitted from the output rather than carrying
    an empty list — the consumer renders one section per non-empty
    paradigm.
    """
    if df.is_empty():
        return []

    if "paradigm" not in df.columns:
        # v1-only tree; everything is instance.
        rows = compare_shas(df, base_sha=base_sha, head_sha=head_sha)
        return [ParadigmCompareSection(paradigm="instance", rows=rows)] if rows else []

    sections: list[ParadigmCompareSection] = []
    paradigms_seen = sorted(df["paradigm"].unique().to_list())
    for p in paradigms_seen:
        if p is None:
            continue
        df_p = df.filter(df["paradigm"] == p)
        rows = compare_shas(df_p, base_sha=base_sha, head_sha=head_sha)
        if rows:
            sections.append(ParadigmCompareSection(paradigm=p, rows=rows))
    return sections
