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
    "pycocotools": "pycocotools",
    "faster-coco-eval": "faster-coco-eval",
    "boundary-iou-api": "boundary-iou-api",
    "panopticapi": "panopticapi",
    "mmsegmentation": "mmsegmentation",
}

# Stable column order so the headline impl (vernier-family) renders first.
IMPL_ORDER: list[str] = [
    "vernier",
    "vernier_panoptic",
    "vernier_semantic",
    "faster-coco-eval",
    "pycocotools",
    "boundary-iou-api",
    "panopticapi",
    "mmsegmentation",
]

# Per-paradigm display order for IoU/metric subsections.
IOU_ORDER: dict[str, list[str]] = {
    "instance": ["bbox", "segm", "boundary", "keypoints"],
    "panoptic": ["pq"],
    "semantic": ["miou"],
}

PARADIGM_TITLE = {
    "instance": "Instance — bbox / segm / boundary / keypoints (AP)",
    "panoptic": "Panoptic — PQ",
    "semantic": "Semantic — mIoU",
}

# Iteration order for the rendered document.
PARADIGM_RENDER_ORDER: tuple[str, ...] = ("instance", "panoptic", "semantic")


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
    rsses = [r.get("ru_maxrss_bytes", 0) for r in reps]
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
        if not mode:
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
    for impl_label in ("vernier", "vernier_panoptic", "vernier_semantic"):
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
    has_iqr = any(matching[i].iqr_ns is not None for i in impls)
    rows = []
    if has_iqr:
        rows.append("| impl | median | IQR | RSS (max) | vs vernier |")
        rows.append("| --- | ---: | ---: | ---: | ---: |")
    else:
        rows.append("| impl | median | RSS (max) | vs vernier |")
        rows.append("| --- | ---: | ---: | ---: |")
    for impl in impls:
        stats = matching[impl]
        speedup = stats.median_ns / baseline.median_ns
        cell_label = IMPL_LABELS.get(impl, impl)
        if is_vernier(impl):
            cell_label = f"**{cell_label}**"
        ratio_cell = (
            f"**{format_speedup(speedup)}**" if is_vernier(impl) else format_speedup(speedup)
        )
        if has_iqr:
            iqr_cell = format_iqr(stats.iqr_ns, stats.iqr_relative, stats.iqr_gate_passed)
            rows.append(
                f"| {cell_label} | {format_ns(stats.median_ns)} | {iqr_cell} "
                f"| {format_bytes(stats.max_rss_bytes)} | {ratio_cell} |"
            )
        else:
            rows.append(
                f"| {cell_label} | {format_ns(stats.median_ns)} "
                f"| {format_bytes(stats.max_rss_bytes)} | {ratio_cell} |"
            )
    return "\n".join(rows)


def render_paradigm_section(
    cells: dict[CellKey, CellStats],
    paradigm: str,
) -> str:
    workloads = sorted({k.workload for k in cells if k.paradigm == paradigm})
    if not workloads:
        return ""
    out = [f"## {PARADIGM_TITLE.get(paradigm, paradigm.capitalize())}", ""]
    for workload in workloads:
        out.append(f"### Workload: `{workload}`")
        out.append("")
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


_PYPI_BASELINES: frozenset[str] = frozenset({"pycocotools", "faster-coco-eval", "mmsegmentation"})
_GH_BASELINES: dict[str, str] = {
    "panopticapi": "cocodataset/panopticapi",
    "boundary-iou-api": "bowenc0221/boundary-iou-api",
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
    pieces = [
        _baseline_link(impl, impl_versions[impl])
        for impl in IMPL_ORDER
        if not is_vernier(impl) and impl in impl_versions
    ]
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
        section = render_paradigm_section(cells, paradigm)
        if section:
            sections.append(section)

    methodology = """## Methodology in one paragraph

Every cell runs in its own subprocess with its own uv-managed venv (one
per impl), so a single Python process never has competing
pycocotools-flavored packages on its `sys.path`. The harness records
`(load, evaluate, accumulate, summarize, total)` wall_ns per stage,
discards the warmup reps, and reports the median total plus the
inter-quartile range (IQR = Q3 - Q1, with the relative spread shown as
a percentage of the median). Release mode (N=10 + 2 warmup) gates each
impl on relative IQR ≤ 5%; cells where the gate failed are marked with
` *` next to their IQR value — the median is still the best estimator,
just with a wider confidence band than the gate accepts. Parity is a
side effect of every timing run — strict-tier (vs pycocotools) and
aligned-tier (vs faster-coco-eval) where applicable; failed parity
fails the cell. Memory is `getrusage(RUSAGE_CHILDREN).ru_maxrss`,
high-water-marked across the rep set.
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
