#!/usr/bin/env python3
"""Render docs/benchmarks.md from the harness result tree.

Walks ``bench/results/<git_sha>/<machine_fp>/<paradigm>/<workload>/<iou>/<impl>.json``
(per ADR-0017 + ADR-0033), computes the median total-stage wall_ns over
non-warmup reps for each cell, and emits a Markdown comparison table.

Usage:

    python tools/render_benchmarks.py [--sha SHA] [--mfp MFP] \\
        [--output docs/benchmarks.md]

The default ``--sha`` picks the most comprehensive SHA available — the
one with the most ``(paradigm, workload, iou, impl)`` cells against
non-vernier baselines (so a vernier-only round of runs doesn't replace
a mixed-impl headline). ``--mfp`` defaults to the only machine
fingerprint under that SHA when there's exactly one.

Run from the repo root. No third-party dependencies — stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "bench" / "results"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "benchmarks.md"

# Display names for impl ids in the rendered tables. Anything not in
# this map renders as the raw impl id from the JSON.
IMPL_LABELS: dict[str, str] = {
    "vernier": "vernier",
    "vernier_panoptic": "vernier",
    "vernier_semantic": "vernier",
    "vernier_lvis": "vernier",
    "pycocotools": "pycocotools",
    "faster-coco-eval": "faster-coco-eval",
    "hotcoco": "hotcoco",
    "hotcoco_lvis": "hotcoco",
    "boundary-iou-api": "boundary-iou-api",
    "panopticapi": "panopticapi",
    "mmsegmentation": "mmsegmentation",
    "lvis-api": "lvis-api",
}

# Stable column order so the headline impl (vernier-family) renders first.
IMPL_ORDER: list[str] = [
    "vernier",
    "vernier_panoptic",
    "vernier_semantic",
    "vernier_lvis",
    "hotcoco",
    "hotcoco_lvis",
    "faster-coco-eval",
    "pycocotools",
    "boundary-iou-api",
    "panopticapi",
    "mmsegmentation",
    "lvis-api",
]

# Per-paradigm display order for IoU/metric subsections.
IOU_ORDER: dict[str, list[str]] = {
    "instance": ["bbox", "segm", "boundary", "keypoints"],
    "panoptic": ["pq"],
    "semantic": ["miou"],
    # LVIS shares IouType with instance but is a separate paradigm —
    # bbox-only at the vernier side until ``evaluate_segm_grid_with_dataset``
    # lands; lvis-api supports both natively.
    "lvis": ["bbox", "segm"],
}

PARADIGM_TITLE = {
    "instance": "Instance — bbox / segm / boundary / keypoints (AP)",
    "panoptic": "Panoptic — PQ",
    "semantic": "Semantic — mIoU",
    "lvis": "Instance — LVIS federated AP",
}

# Iteration order for the rendered document.
PARADIGM_RENDER_ORDER: tuple[str, ...] = ("instance", "panoptic", "semantic", "lvis")


@dataclass(frozen=True)
class CellKey:
    paradigm: str
    workload: str
    iou: str
    impl: str


@dataclass(frozen=True)
class CellStats:
    median_ns: int
    iqr_ns: int | None
    iqr_relative: float | None
    iqr_gate_passed: bool | None
    max_rss_bytes: int
    # Median over reps of total-stage process CPU time / wall time: the
    # impl's effective parallelism. ``None`` for results recorded before
    # stages carried ``cpu_ns``.
    cpu_util: float | None = None
    # Median over reps of total-stage peak RSS minus RSS at the start of
    # the first stage: memory the evaluation added on top of the
    # interpreter + imports. ``None`` for results recorded before stages
    # carried RSS fields.
    eval_rss_bytes: int | None = None
    # Harness mode the cell was recorded in. Scale workloads run in
    # ``dev`` (one rep) next to a ``release`` headline.
    mode: str = ""


def format_ns(ns: int | None) -> str:
    if ns is None:
        return "—"
    if ns >= 1_000_000_000:
        return f"{ns / 1_000_000_000:.3f} s"
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.1f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.1f} μs"
    return f"{ns} ns"


def format_bytes(b: int | None) -> str:
    if b is None:
        return "—"
    if b >= 1_073_741_824:
        return f"{b / 1_073_741_824:.2f} GiB"
    if b >= 1_048_576:
        return f"{b / 1_048_576:.0f} MiB"
    if b >= 1024:
        return f"{b / 1024:.0f} KiB"
    return f"{b} B"


def format_speedup(ratio: float | None) -> str:
    if ratio is None:
        return "—"
    return f"{ratio:.2f}×"  # noqa: RUF001


def format_iqr(
    iqr_ns: int | None,
    iqr_relative: float | None,
    iqr_gate_passed: bool | None,
) -> str:
    if iqr_ns is None:
        return "—"
    rel_str = f" ({iqr_relative * 100:.2f}%)" if iqr_relative is not None else ""
    fail_marker = "" if iqr_gate_passed in (None, True) else " *"
    return f"{format_ns(iqr_ns)}{rel_str}{fail_marker}"


def _fmt_iqr_col(stats: CellStats) -> str:
    return format_iqr(stats.iqr_ns, stats.iqr_relative, stats.iqr_gate_passed)


def _fmt_cpu_col(stats: CellStats) -> str:
    return "—" if stats.cpu_util is None else f"{stats.cpu_util:.2f}"


def is_vernier(impl: str) -> bool:
    return impl == "vernier" or impl.startswith("vernier_")


def discover_shas() -> list[str]:
    if not RESULTS_ROOT.is_dir():
        return []
    return sorted(p.name for p in RESULTS_ROOT.iterdir() if p.is_dir())


def auto_select_sha() -> str:
    """Pick the SHA with the most non-vernier impl coverage.

    Ties broken by total cell count, then lexically. A SHA with no
    third-party baseline runs (vernier-only round) loses to any SHA
    that has at least one comparison cell. When more than one SHA is
    on disk, prints the choice + runner-up to stderr so the docs
    author can spot a wrong-pick (e.g. an annex-heavy older round
    outranking a newer headline run) and override with `--sha`.
    """
    candidates: list[tuple[int, int, str]] = []
    for sha in discover_shas():
        sha_root = RESULTS_ROOT / sha
        comparison_cells = 0
        total_cells = 0
        for path in sha_root.rglob("*.json"):
            if ".intermediate" in path.parts or path.name.endswith(".snapshot.json"):
                continue
            total_cells += 1
            if not is_vernier(path.stem):
                comparison_cells += 1
        if total_cells > 0:
            candidates.append((comparison_cells, total_cells, sha))
    if not candidates:
        sys.exit("error: no result JSONs found under bench/results/")
    candidates.sort(reverse=True)
    best = candidates[0]
    if len(candidates) > 1:
        runner_up = candidates[1]
        print(
            f"auto-selected sha={best[2]} ({best[0]} comparison / "
            f"{best[1]} total cells), runner-up sha={runner_up[2]} "
            f"({runner_up[0]} / {runner_up[1]}); pass --sha to override",
            file=sys.stderr,
        )
    return best[2]


def auto_select_mfp(sha: str) -> str:
    sha_root = RESULTS_ROOT / sha
    mfps = sorted(p.name for p in sha_root.iterdir() if p.is_dir())
    if not mfps:
        sys.exit(f"error: no machine-fingerprints under bench/results/{sha}/")
    if len(mfps) > 1:
        # Prefer the mfp with the most cells.
        counts = {mfp: sum(1 for _ in (sha_root / mfp).rglob("*.json")) for mfp in mfps}
        return max(mfps, key=lambda m: counts[m])
    return mfps[0]


def load_cell(path: Path) -> tuple[CellStats, str, str, str | None, str | None] | None:
    """Load a single result JSON; return (stats, mode, impl_version, cpu_model, cpu_arch) or None.

    Returns None if the file has no non-warmup reps (e.g., an aborted
    run or a snapshot artifact). ``cpu_model`` / ``cpu_arch`` are
    ``None`` when read from result files written before those fields
    landed in the schema.
    """
    with path.open() as f:
        data = json.load(f)
    reps = [r for r in data.get("reps", []) if not r.get("warmup", False)]
    if not reps:
        return None
    walls = [r["stages"]["total"]["wall_ns"] for r in reps]
    cpu_ratios = [
        r["stages"]["total"]["cpu_ns"] / r["stages"]["total"]["wall_ns"]
        for r in reps
        if r["stages"]["total"].get("cpu_ns") is not None and r["stages"]["total"]["wall_ns"] > 0
    ]
    eval_rsses = [
        r["stages"]["total"]["peak_rss_bytes"] - r["stages"]["total"]["rss_start_bytes"]
        for r in reps
        if r["stages"]["total"].get("peak_rss_bytes") is not None
        and r["stages"]["total"].get("rss_start_bytes") is not None
    ]
    # Prefer the in-runner peak over the timed stages. Runners that
    # record per-stage peaks reset the kernel's RSS high-water mark,
    # which also resets what ``getrusage`` later reports as
    # ``ru_maxrss``; for those results ``ru_maxrss_bytes`` only covers
    # the final stage onward and must not be read as a process peak.
    rsses = [
        r["stages"]["total"].get("peak_rss_bytes") or r.get("ru_maxrss_bytes", 0) for r in reps
    ]
    aggregation = data.get("aggregation") or {}
    total_agg = aggregation.get("stages", {}).get("total", {})
    iqr_ns_raw = total_agg.get("iqr_ns")
    iqr_ns = int(iqr_ns_raw) if isinstance(iqr_ns_raw, int) else None
    iqr_gate = aggregation.get("iqr_gate") or {}
    iqr_relative_raw = iqr_gate.get("relative")
    iqr_relative = float(iqr_relative_raw) if isinstance(iqr_relative_raw, (int, float)) else None
    iqr_gate_passed_raw = iqr_gate.get("passed")
    iqr_gate_passed = bool(iqr_gate_passed_raw) if isinstance(iqr_gate_passed_raw, bool) else None
    stats = CellStats(
        median_ns=int(statistics.median(walls)),
        iqr_ns=iqr_ns,
        iqr_relative=iqr_relative,
        iqr_gate_passed=iqr_gate_passed,
        max_rss_bytes=max(rsses) if rsses else 0,
        cpu_util=statistics.median(cpu_ratios) if cpu_ratios else None,
        eval_rss_bytes=int(statistics.median(eval_rsses)) if eval_rsses else None,
        mode=str(data.get("mode", "")),
    )
    cpu_model = data.get("cpu_model")
    cpu_arch = data.get("cpu_arch")
    return (
        stats,
        str(data.get("mode", "")),
        str(data.get("impl_version", "")),
        cpu_model if isinstance(cpu_model, str) else None,
        cpu_arch if isinstance(cpu_arch, str) else None,
    )


def gather_cells(
    sha: str, mfp: str
) -> tuple[dict[CellKey, CellStats], str, dict[str, str], str | None, str | None]:
    """Return the cells dict, the harness mode, impl→version pins, and CPU info.

    Warns on stderr if a result JSON has no `impl_version` field, or if
    the same impl is pinned to multiple versions across cells (the
    rendered baseline line would silently pick whichever was visited
    first). All cells under one ``<machine-fp>`` share a machine by
    construction, so CPU info from the first loaded cell is the canonical
    value; ``None`` for older result files written before those fields
    landed.
    """
    base = RESULTS_ROOT / sha / mfp
    if not base.is_dir():
        sys.exit(f"error: {base} not found")
    out: dict[CellKey, CellStats] = {}
    mode = ""
    versions_seen: dict[str, set[str]] = {}
    cpu_model: str | None = None
    cpu_arch: str | None = None
    for path in sorted(base.rglob("*.json")):
        if ".intermediate" in path.parts or path.name.endswith(".snapshot.json"):
            continue
        # Path under base: <paradigm>/<workload>/<iou>/<impl>.json
        rel = path.relative_to(base)
        parts = rel.parts
        if len(parts) != 4:
            continue
        paradigm, workload, iou, fname = parts
        impl = fname.removesuffix(".json")
        loaded = load_cell(path)
        if loaded is None:
            continue
        stats, cell_mode, impl_version, cell_cpu_model, cell_cpu_arch = loaded
        out[CellKey(paradigm, workload, iou, impl)] = stats
        # The headline mode is ``release`` whenever any cell used it;
        # per-workload deviations are annotated in the section itself.
        if not mode or cell_mode == "release":
            mode = cell_mode
        if cpu_model is None and cell_cpu_model is not None:
            cpu_model = cell_cpu_model
        if cpu_arch is None and cell_cpu_arch is not None:
            cpu_arch = cell_cpu_arch
        if impl_version:
            versions_seen.setdefault(impl, set()).add(impl_version)
        else:
            print(
                f"warning: {path.relative_to(REPO_ROOT)} has no impl_version",
                file=sys.stderr,
            )

    impl_versions: dict[str, str] = {}
    for impl, versions in versions_seen.items():
        if len(versions) > 1:
            chosen = sorted(versions)[0]
            print(
                f"warning: {impl} pinned to multiple versions across cells "
                f"({sorted(versions)}); rendering with {chosen}",
                file=sys.stderr,
            )
            impl_versions[impl] = chosen
        else:
            impl_versions[impl] = next(iter(versions))
    return out, mode, impl_versions, cpu_model, cpu_arch


def vernier_baseline_for(
    cells: dict[CellKey, CellStats],
    paradigm: str,
    workload: str,
    iou: str,
) -> CellStats | None:
    """Find the vernier(_*) cell to anchor the speedup column."""
    for impl_label in ("vernier", "vernier_panoptic", "vernier_semantic", "vernier_lvis"):
        key = CellKey(paradigm, workload, iou, impl_label)
        if key in cells:
            return cells[key]
    return None


def render_iou_table(
    cells: dict[CellKey, CellStats],
    paradigm: str,
    workload: str,
    iou: str,
) -> str:
    matching = {
        k.impl: v
        for k, v in cells.items()
        if k.paradigm == paradigm and k.workload == workload and k.iou == iou
    }
    if not matching:
        return ""
    baseline = vernier_baseline_for(cells, paradigm, workload, iou)
    if baseline is None:
        return ""
    impls = [i for i in IMPL_ORDER if i in matching]
    impls.extend(sorted(i for i in matching if i not in IMPL_ORDER))
    # One spec drives both the header and every row, so a column can't be
    # added to one and forgotten in the other. ``shown`` drops columns no
    # impl in this cell recorded (older result files carry neither CPU nor
    # RSS fields).
    optional: list[tuple[str, bool, Callable[[CellStats], str]]] = [
        ("IQR", any(matching[i].iqr_ns is not None for i in impls), _fmt_iqr_col),
        ("CPU/wall", any(matching[i].cpu_util is not None for i in impls), _fmt_cpu_col),
        ("peak RSS", True, lambda s: format_bytes(s.max_rss_bytes)),
        (
            "eval Δ RSS",
            any(matching[i].eval_rss_bytes is not None for i in impls),
            lambda s: format_bytes(s.eval_rss_bytes),
        ),
    ]
    shown = [(name, fmt) for name, on, fmt in optional if on]
    header = ["impl", "median", *(name for name, _ in shown), "vs vernier"]
    rows = [
        "| " + " | ".join(header) + " |",
        "| --- |" + " ---: |" * (len(header) - 1),
    ]
    for impl in impls:
        stats = matching[impl]
        speedup = format_speedup(stats.median_ns / baseline.median_ns)
        cell_label = IMPL_LABELS.get(impl, impl)
        if is_vernier(impl):
            cell_label = f"**{cell_label}**"
            speedup = f"**{speedup}**"
        cols = [cell_label, format_ns(stats.median_ns), *(fmt(stats) for _, fmt in shown), speedup]
        rows.append("| " + " | ".join(cols) + " |")
    return "\n".join(rows)


# Workload-id prefix → note rendered under the workload heading. Carries
# dataset attribution required by the source license.
_WORKLOAD_NOTES: dict[str, str] = {
    "objects365_val": (
        "Scale workload: Objects365 v2 val, 80,000 images · 1,240,587 GT boxes · "
        "365 categories, with ~1.06 M jittered detections (bbox only). Annotations "
        "© [Objects365 Consortium](https://www.objects365.org/), licensed under "
        "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); images are "
        "never downloaded."
    ),
}


def render_paradigm_section(
    cells: dict[CellKey, CellStats],
    paradigm: str,
    harness_mode: str = "",
) -> str:
    workloads = sorted(
        {k.workload for k in cells if k.paradigm == paradigm and not _THREADED_RE.match(k.workload)}
    )
    if not workloads:
        return ""
    out = [f"## {PARADIGM_TITLE.get(paradigm, paradigm.capitalize())}", ""]
    for workload in workloads:
        out.append(f"### Workload: `{workload}`")
        out.append("")
        for prefix, note in _WORKLOAD_NOTES.items():
            if workload.startswith(prefix):
                out += [f"*{note}*", ""]
        modes = {
            v.mode for k, v in cells.items() if k.paradigm == paradigm and k.workload == workload
        }
        if harness_mode and modes and modes != {harness_mode}:
            out += [
                f"*Recorded in harness mode `{'/'.join(sorted(modes))}` (not "
                f"`{harness_mode}`): one measurement rep per impl, no IQR gate.*",
                "",
            ]
        ious_present = {k.iou for k in cells if k.paradigm == paradigm and k.workload == workload}
        order = IOU_ORDER.get(paradigm, sorted(ious_present))
        for iou in order:
            if iou not in ious_present:
                continue
            table = render_iou_table(cells, paradigm, workload, iou)
            if not table:
                continue
            out.append(f"**`{iou}`**")
            out.append("")
            out.append(table)
            out.append("")
    return "\n".join(out)


# ADR-0047 thread-axis cells carry a ``_t<N>`` workload suffix; they
# render in the scaling section, not as standalone workloads.
_THREADED_RE = re.compile(r"^(?P<base>.+)_t(?P<nt>\d+)$")


def _scaling_row(impl: str, entries: list[str]) -> str:
    label = IMPL_LABELS.get(impl, impl)
    label = f"**{label}**" if is_vernier(impl) else label
    return f"| {label} | " + " | ".join(entries) + " |"


def render_scaling_section(cells: dict[CellKey, CellStats]) -> str:
    """One table per ``(paradigm, base workload, iou)`` with a thread
    axis: rows are impls, columns thread counts, each entry the median
    total plus its ratio to vernier at the same thread count."""
    axes: dict[tuple[str, str, str], set[int]] = {}
    for k in cells:
        m = _THREADED_RE.match(k.workload)
        if m:
            axes.setdefault((k.paradigm, m["base"], k.iou), set()).add(int(m["nt"]))
    if not axes:
        return ""
    out = ["## Thread scaling", ""]
    for (paradigm, base, iou), nts in sorted(axes.items()):
        thread_counts = sorted(nts)
        impls_present = {
            k.impl
            for k in cells
            if k.paradigm == paradigm
            and k.iou == iou
            and k.workload in {f"{base}_t{nt}" for nt in thread_counts}
        }
        impls = [i for i in IMPL_ORDER if i in impls_present]
        impls.extend(sorted(impls_present - set(IMPL_ORDER)))
        workloads = [f"{base}_t{nt}" for nt in thread_counts]
        header = "| impl | " + " | ".join(f"`nt={nt}`" for nt in thread_counts) + " |"
        align = "| --- |" + " ---: |" * len(thread_counts)
        # Both tables read the same cells; look each up once. The baseline
        # depends on the workload, not the impl, so it is hoisted too.
        baselines = {w: vernier_baseline_for(cells, paradigm, w, iou) for w in workloads}
        per_impl = {
            impl: [cells.get(CellKey(paradigm, w, iou, impl)) for w in workloads] for impl in impls
        }

        out += [
            f"**`{base}` · `{iou}`** — median total; ratio vs vernier at the same `nt`",
            "",
            header,
            align,
        ]
        for impl, per_nt in per_impl.items():
            entries = []
            for workload, stats in zip(workloads, per_nt, strict=True):
                baseline = baselines[workload]
                if stats is None:
                    entries.append("—")
                elif is_vernier(impl) or baseline is None:
                    entries.append(format_ns(stats.median_ns))
                else:
                    ratio = format_speedup(stats.median_ns / baseline.median_ns)
                    entries.append(f"{format_ns(stats.median_ns)} ({ratio})")
            out.append(_scaling_row(impl, entries))
        out.append("")

        if any(
            st is not None and st.eval_rss_bytes is not None for v in per_impl.values() for st in v
        ):
            out += [f"**`{base}` · `{iou}`** — eval Δ RSS", "", header, align]
            for impl, per_nt in per_impl.items():
                entries = [format_bytes(st.eval_rss_bytes if st else None) for st in per_nt]
                out.append(_scaling_row(impl, entries))
            out.append("")
    return "\n".join(out)


_PYPI_BASELINES: frozenset[str] = frozenset({"pycocotools", "faster-coco-eval", "hotcoco"})
_GH_BASELINES: dict[str, str] = {
    "panopticapi": "cocodataset/panopticapi",
    "boundary-iou-api": "bowenc0221/boundary-iou-api",
    # mmsegmentation is vendored at a pinned upstream SHA per ADR-0036
    # rather than pip-installed; the runner emits the SHA as
    # impl_version so the renderer links to the upstream commit, not
    # the PyPI release.
    "mmsegmentation": "open-mmlab/mmsegmentation",
    # lvis-api is pinned to commit ORACLE_LVIS_COMMIT_SHA in
    # ``crates/vernier-core/src/lvis_parity.rs`` (PyPI lvis==0.5.3,
    # uploaded 2020-06-18). Same SHA-link rendering as panopticapi /
    # boundary-iou-api.
    "lvis-api": "lvis-dataset/lvis-api",
}
assert set(IMPL_ORDER) >= _PYPI_BASELINES
assert set(IMPL_ORDER) >= _GH_BASELINES.keys()

_HEX_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


def _baseline_link(impl: str, version: str) -> str:
    if impl in _PYPI_BASELINES:
        return f"[`{impl}=={version}`](https://pypi.org/project/{impl}/{version}/)"
    if impl in _GH_BASELINES and _HEX_SHA_RE.match(version):
        return (
            f"[`{impl}` @ `{version[:7]}`]"
            f"(https://github.com/{_GH_BASELINES[impl]}/commit/{version})"
        )
    return f"`{impl}=={version}`"


def render_baselines_block(impl_versions: dict[str, str]) -> str:
    """Render the pinned-baselines line; skips vernier-family impls."""
    # Keyed by display label so one library serving two paradigms
    # (``hotcoco`` + ``hotcoco_lvis``) pins once.
    by_label: dict[str, str] = {}
    for impl in IMPL_ORDER:
        if not is_vernier(impl) and impl in impl_versions:
            by_label.setdefault(IMPL_LABELS.get(impl, impl), impl_versions[impl])
    pieces = [_baseline_link(label, version) for label, version in by_label.items()]
    if not pieces:
        return ""
    return (
        "**Baselines pinned for these numbers** — "
        + " · ".join(pieces)
        + ". Each baseline is locked in its own uv-managed venv per ADR-0017."
    )


def _cpu_provenance(cpu_model: str | None, cpu_arch: str | None) -> str:
    """Render the optional CPU clause of the provenance line.

    Empty string when both fields are ``None`` (older v2 result files
    written before the schema picked up CPU info), so the renderer
    falls back to the original fingerprint-only string.
    """
    if cpu_model is None and cpu_arch is None:
        return ""
    if cpu_model is not None and cpu_arch is not None:
        return f" · CPU {cpu_model} ({cpu_arch})"
    return f" · CPU {cpu_model or cpu_arch}"


def render_document(
    sha: str,
    mfp: str,
    cells: dict[CellKey, CellStats],
    harness_mode: str,
    impl_versions: dict[str, str],
    cpu_model: str | None,
    cpu_arch: str | None,
) -> str:
    if not cells:
        sys.exit("error: no usable cells in the selected SHA/mfp")

    baselines_block = render_baselines_block(impl_versions)
    baselines_section = ("\n\n" + baselines_block) if baselines_block else ""
    cpu_clause = _cpu_provenance(cpu_model, cpu_arch)
    has_iqr_failures = any(stats.iqr_gate_passed is False for stats in cells.values())

    header = f"""# Benchmarks

Comparison of vernier against the third-party libraries it targets parity
against, on a single machine and a single git revision. The numbers below
are the median total-stage wall time over the non-warmup reps recorded by
the local bench harness ([ADR-0017](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0017-local-bench-harness.md),
extended cross-paradigm in
[ADR-0033](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0033-multi-paradigm-bench.md)).
The IQR column reports the spread (Q3 - Q1) across the 10 measurement
reps and the same value as a percentage of the median; release mode
gates each cell at 5% relative IQR.

**Provenance** — git SHA `{sha}` · machine fingerprint `{mfp}`{cpu_clause} · harness
mode `{harness_mode}` · build profile = cargo release defaults
(`opt-level=3`, `lto=thin`, `codegen-units=1`, no `target-cpu`). The
release wheel on PyPI is built with the same profile — no
benchmarking-only flags.{baselines_section}

For the full per-cell deep-dive (per-stage breakdown, RSS evolution,
parity gating, narrative on what moved each round), see
[`docs/engineering/benchmarking/`](https://github.com/NoeFontana/vernier/tree/main/docs/engineering/benchmarking).

This page is regenerated from the harness result tree by
`tools/render_benchmarks.py`. To refresh after a new bench run, see the
[release runbook](https://github.com/NoeFontana/vernier/blob/main/docs/engineering/release-runbook.md)
§0.

"""
    sections = []
    for paradigm in PARADIGM_RENDER_ORDER:
        section = render_paradigm_section(cells, paradigm, harness_mode)
        if section:
            sections.append(section)
    scaling = render_scaling_section(cells)
    if scaling:
        sections.append(scaling)

    methodology = """## Methodology in one paragraph

Every cell runs in its own subprocess with its own uv-managed venv (one
per impl), so a single Python process never has competing
pycocotools-flavored packages on its `sys.path`. The harness records
`(load, evaluate, accumulate, summarize, total)` wall_ns per stage,
discards the warmup reps, and reports the median total plus the
inter-quartile range (IQR = Q3 - Q1, with the relative spread shown as
a percentage of the median). The timed span is the same for every impl:
annotation files on disk → summary stats, including JSON parsing and
index building, excluding interpreter start-up and imports. Per-stage
splits are *not* comparable across impls (vernier parses JSON inside
`evaluate`; the pycocotools-shaped libraries parse in `load`), so only
the total is reported. Instance and LVIS cells run under an enforced
CPU budget: every runner process is pinned (CPU affinity, one logical
CPU per physical core before SMT siblings) to 1 CPU for the headline
tables and `N` CPUs for `nt=N` cells, and the budget is also passed to
each library's own thread knob (vernier `num_threads`, hotcoco
`RAYON_NUM_THREADS`, faster-coco-eval `rle_iou_max_workers` /
`boundary_cpu_count`). The CPU/wall column is process CPU time over
wall time — ~1.00 means the impl used one core; anything well below
the budget means it spent wall time waiting rather than computing.
Memory is reported two ways. Peak RSS is the exact resident-memory
high-water mark over the timed stages (the kernel's `VmHWM`, reset
through `/proc/self/clear_refs` at every stage start, max across
stages and reps); it includes the interpreter and the library's
imports. eval Δ RSS is the median across reps of that peak minus RSS
just before the first stage: the memory the evaluation itself needed,
input parsing included.
Release mode (N=10 + 2 warmup) gates each impl on relative IQR ≤ 5%;
cells where the gate failed are marked with
` *` next to their IQR value — the median is still the best estimator,
just with a wider confidence band than the gate accepts. Parity is a
side effect of every timing run — strict-tier (vs pycocotools) and
aligned-tier (vs faster-coco-eval and hotcoco) where applicable;
a failed tier writes a divergence report next to the cell.
"""

    iqr_footnote = ""
    if has_iqr_failures:
        iqr_footnote = (
            "\n\n*Cells marked ` *` next to their IQR exceeded the release-mode "
            "5% relative-IQR gate. Median still reported; treat the gap to the "
            "next impl as the load-bearing signal rather than the precise ratio.*"
        )

    return header + "\n\n".join(sections) + iqr_footnote + "\n\n" + methodology


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render docs/benchmarks.md from the harness result tree."
    )
    parser.add_argument(
        "--sha",
        help="Git SHA prefix under bench/results/. Default: auto-select the "
        "SHA with the most non-vernier comparison cells.",
    )
    parser.add_argument(
        "--mfp",
        help="Machine fingerprint under bench/results/<sha>/. Default: the "
        "single mfp under the chosen SHA, or the one with the most cells.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output Markdown path (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}).",
    )
    args = parser.parse_args()

    sha = args.sha or auto_select_sha()
    mfp = args.mfp or auto_select_mfp(sha)
    cells, harness_mode, impl_versions, cpu_model, cpu_arch = gather_cells(sha, mfp)
    document = render_document(sha, mfp, cells, harness_mode, impl_versions, cpu_model, cpu_arch)
    args.output.write_text(document)
    try:
        rel = args.output.relative_to(REPO_ROOT)
    except ValueError:
        rel = args.output
    print(f"wrote {rel} from sha={sha} mfp={mfp} cells={len(cells)}")


if __name__ == "__main__":
    main()
