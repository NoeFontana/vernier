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
    return f"{ratio:.2f}×"


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
    that has at least one comparison cell.
    """
    best: tuple[int, int, str] | None = None
    for sha in discover_shas():
        sha_root = RESULTS_ROOT / sha
        comparison_cells = 0
        total_cells = 0
        for path in sha_root.rglob("*.json"):
            if ".intermediate" in path.parts or path.name.endswith(".snapshot.json"):
                continue
            total_cells += 1
            impl = path.stem
            if not is_vernier(impl):
                comparison_cells += 1
        if total_cells == 0:
            continue
        candidate = (comparison_cells, total_cells, sha)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        sys.exit("error: no result JSONs found under bench/results/")
    return best[2]


def auto_select_mfp(sha: str) -> str:
    sha_root = RESULTS_ROOT / sha
    mfps = sorted(p.name for p in sha_root.iterdir() if p.is_dir())
    if not mfps:
        sys.exit(f"error: no machine-fingerprints under bench/results/{sha}/")
    if len(mfps) > 1:
        # Prefer the mfp with the most cells.
        counts = {
            mfp: sum(1 for _ in (sha_root / mfp).rglob("*.json"))
            for mfp in mfps
        }
        return max(mfps, key=lambda m: counts[m])
    return mfps[0]


def load_cell(path: Path) -> tuple[CellStats, str] | None:
    """Load a single result JSON; return (stats, harness_mode) or None.

    Returns None if the file has no non-warmup reps (e.g., an aborted
    run or a snapshot artifact).
    """
    with path.open() as f:
        data = json.load(f)
    reps = [r for r in data.get("reps", []) if not r.get("warmup", False)]
    if not reps:
        return None
    walls = [r["stages"]["total"]["wall_ns"] for r in reps]
    rsses = [r.get("ru_maxrss_bytes", 0) for r in reps]
    stats = CellStats(
        median_ns=int(statistics.median(walls)),
        max_rss_bytes=max(rsses) if rsses else 0,
    )
    return stats, str(data.get("mode", ""))


def gather_cells(sha: str, mfp: str) -> tuple[dict[CellKey, CellStats], str]:
    """Return the cells dict plus the harness mode shared across them."""
    base = RESULTS_ROOT / sha / mfp
    if not base.is_dir():
        sys.exit(f"error: {base} not found")
    out: dict[CellKey, CellStats] = {}
    mode = ""
    for path in base.rglob("*.json"):
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
        stats, cell_mode = loaded
        out[CellKey(paradigm, workload, iou, impl)] = stats
        if not mode:
            mode = cell_mode
    return out, mode


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
    matching = {k.impl: v for k, v in cells.items() if k.paradigm == paradigm and k.workload == workload and k.iou == iou}
    if not matching:
        return ""
    baseline = vernier_baseline_for(cells, paradigm, workload, iou)
    if baseline is None:
        return ""
    impls = [i for i in IMPL_ORDER if i in matching]
    impls.extend(sorted(i for i in matching if i not in IMPL_ORDER))
    rows = []
    rows.append("| impl | median | RSS (max) | vs vernier |")
    rows.append("| --- | ---: | ---: | ---: |")
    for impl in impls:
        stats = matching[impl]
        speedup = stats.median_ns / baseline.median_ns
        cell_label = IMPL_LABELS.get(impl, impl)
        if is_vernier(impl):
            cell_label = f"**{cell_label}**"
        ratio_cell = f"**{format_speedup(speedup)}**" if is_vernier(impl) else format_speedup(speedup)
        rows.append(
            f"| {cell_label} | {format_ns(stats.median_ns)} | {format_bytes(stats.max_rss_bytes)} | {ratio_cell} |"
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


def render_document(
    sha: str, mfp: str, cells: dict[CellKey, CellStats], harness_mode: str
) -> str:
    if not cells:
        sys.exit("error: no usable cells in the selected SHA/mfp")

    header = f"""# Benchmarks

Comparison of vernier against the third-party libraries it targets parity
against, on a single machine and a single git revision. The numbers below
are the median total-stage wall time over the non-warmup reps recorded by
the local bench harness ([ADR-0017](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0017-local-bench-harness.md),
extended cross-paradigm in
[ADR-0033](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0033-multi-paradigm-bench.md)).

**Provenance** — git SHA `{sha}` · machine fingerprint `{mfp}` · harness
mode `{harness_mode}` · build profile = cargo release defaults
(`opt-level=3`, `lto=thin`, `codegen-units=1`, no `target-cpu`). The
release wheel on PyPI is built with the same profile — no
benchmarking-only flags.

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
discards the warmup reps, and reports the median total. Parity is a
side effect of every timing run — strict-tier (vs pycocotools) and
aligned-tier (vs faster-coco-eval) where applicable; failed parity
fails the cell. Memory is `getrusage(RUSAGE_CHILDREN).ru_maxrss`,
high-water-marked across the rep set.
"""

    return header + "\n\n".join(sections) + "\n\n" + methodology


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
    cells, harness_mode = gather_cells(sha, mfp)
    document = render_document(sha, mfp, cells, harness_mode)
    args.output.write_text(document)
    try:
        rel = args.output.relative_to(REPO_ROOT)
    except ValueError:
        rel = args.output
    print(f"wrote {rel} from sha={sha} mfp={mfp} cells={len(cells)}")


if __name__ == "__main__":
    main()
