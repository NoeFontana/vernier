"""Result-tree walker — JSON files into a single polars DataFrame.

The v2 result tree (ADR-0033) is
``results/<git_sha>/<machine_fp>/<paradigm>/<workload>/<metric>/<impl>.json``.
v1 trees (no paradigm segment) are still readable through the
``migrations.upgrade`` shim. :func:`load_tree` walks both shapes and
returns one row per ``BenchResult`` with the fields ``compare`` and
``longitudinal`` need: identity tuple, median/iqr on the ``total``
stage, run mode, and the result file's mtime (used by
``report --since`` as the "when did this run happen" timestamp — the
schema doesn't carry a wall-clock and we don't want to add one just to
enable a sort).

The walker accepts pre-filters (``shas`` / ``mtime_after``) so callers
that only need a slice of the tree don't pay for parsing every file:
``compare`` knows two SHAs, ``report --since`` knows a cutoff. Both
filter at the FS layer before ``BenchResult.model_validate_json``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from bench.harness.migrations import upgrade as upgrade_schema
from bench.harness.migrations.v1_to_v2 import TENSOR_KEY
from bench.harness.schema import BenchResult
from bench.harness.stats import aggregate_memory


def _row_from_result(result: BenchResult, mtime: float) -> dict[str, object]:
    """One DataFrame row per :class:`BenchResult`, flattened for polars.

    ``aggregation`` is dev-mode-optional, so we surface the rep-0 timings
    when it's missing — at one rep, median == that rep's wall_ns and IQR
    is zero, which is the right answer for downstream comparisons.
    """
    measurement_reps = [r for r in result.reps if not r.warmup]

    if result.aggregation is not None and "total" in result.aggregation.stages:
        s = result.aggregation.stages["total"]
        total_median: int | None = s.median_ns
        total_iqr: int | None = s.iqr_ns
    elif measurement_reps and "total" in measurement_reps[0].stages:
        total_median = measurement_reps[0].stages["total"].wall_ns
        total_iqr = 0
    else:
        total_median = None
        total_iqr = None

    # Older result files (no memory aggregation) fall back to recomputing
    # from per-rep RSS values via the same helper orchestrate uses.
    if result.aggregation is not None and result.aggregation.memory is not None:
        ru_maxrss_median: int | None = result.aggregation.memory.median_bytes
        ru_maxrss_max: int | None = result.aggregation.memory.max_bytes
    elif measurement_reps:
        m = aggregate_memory(measurement_reps)
        ru_maxrss_median = m.median_bytes
        ru_maxrss_max = m.max_bytes
    else:
        ru_maxrss_median = None
        ru_maxrss_max = None

    # v2 keeps the tensor under the canonical ``"tensor"`` slot for
    # detection cells; downstream filters and tests still expect a
    # single ``tensor_sha256`` column. Non-instance paradigms simply
    # leave this column empty (B-streams render their own divergence
    # snapshot).
    tensor_sha256 = result.artifact_sha256.get(TENSOR_KEY, "")

    return {
        "git_sha": result.git_sha,
        "machine_fingerprint": result.machine_fingerprint,
        "paradigm": result.paradigm,
        "workload_id": result.workload_id,
        "iou_type": result.iou_type,
        "impl": result.impl,
        "impl_version": result.impl_version,
        "mode": result.mode,
        "reps_count": result.reps_count,
        "total_median_ns": total_median,
        "total_iqr_ns": total_iqr,
        "ru_maxrss_median_bytes": ru_maxrss_median,
        "ru_maxrss_max_bytes": ru_maxrss_max,
        "tensor_sha256": tensor_sha256,
        "mtime": mtime,
    }


def iter_result_files(results_root: Path, *, shas: set[str] | None = None) -> Iterable[Path]:
    """``*.json`` files under the result tree, excluding ``divergence_report.json``.

    Both v1 (``<sha>/<fp>/<workload>/<iou>/<impl>.json``) and v2
    (``<sha>/<fp>/<paradigm>/<workload>/<metric>/<impl>.json``) layouts
    are walked. ``shas``, when set, narrows the glob per entry so the
    walker doesn't parse JSON files for SHAs the caller doesn't care
    about (the typical ``compare`` shape).
    """
    if not results_root.exists():
        return []
    # v2 paths are five segments below the sha; v1 is four. The walker
    # globs both and the per-file ``upgrade()`` call normalizes the
    # parsed dict.
    patterns = ("*/*/*/*/*.json", "*/*/*/*/*/*.json")
    if shas is None:
        candidates = (p for pat in patterns for p in results_root.glob(pat))
    else:
        candidates = (
            p
            for sha in shas
            for pat in patterns
            for p in results_root.glob(f"{sha}/{pat[len('*/'):]}")
        )
    return (p for p in candidates if p.name != "divergence_report.json")


_EMPTY_SCHEMA: dict[str, pl.DataType] = {
    "git_sha": pl.Utf8,
    "machine_fingerprint": pl.Utf8,
    "paradigm": pl.Utf8,
    "workload_id": pl.Utf8,
    "iou_type": pl.Utf8,
    "impl": pl.Utf8,
    "impl_version": pl.Utf8,
    "mode": pl.Utf8,
    "reps_count": pl.Int64,
    "total_median_ns": pl.Int64,
    "total_iqr_ns": pl.Int64,
    "ru_maxrss_median_bytes": pl.Int64,
    "ru_maxrss_max_bytes": pl.Int64,
    "tensor_sha256": pl.Utf8,
    "mtime": pl.Float64,
}


def load_tree(
    results_root: Path,
    *,
    shas: set[str] | None = None,
    mtime_after: float | None = None,
) -> pl.DataFrame:
    """Eagerly walk the result tree into one DataFrame.

    Empty (no JSON files) → empty DataFrame with the expected columns
    so downstream filters don't crash on missing keys. ``mtime_after``
    short-circuits at ``stat()`` so files outside the report window
    never reach ``BenchResult.model_validate_json``.
    """
    rows: list[dict[str, object]] = []
    for json_path in iter_result_files(results_root, shas=shas):
        mtime = json_path.stat().st_mtime
        if mtime_after is not None and mtime < mtime_after:
            continue
        # v1-shaped JSON parses through the upgrade shim; v2 is
        # idempotent. Going through ``upgrade`` (rather than calling
        # ``model_validate_json`` directly) is what keeps detection
        # cells from before the v2 migration tool ran still readable.
        payload = upgrade_schema(json.loads(json_path.read_bytes()))
        result = BenchResult.model_validate(payload)
        rows.append(_row_from_result(result, mtime))

    if not rows:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)
    return pl.DataFrame(rows)
