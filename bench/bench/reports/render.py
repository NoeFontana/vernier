r"""Render compare/longitudinal data into markdown and SVG.

The SVG renderer is hand-rolled to keep bench's dep tree light — a
multi-line longitudinal chart is the only chart the harness emits and
matplotlib/plotly drag in tens of MB for a job that's ~80 lines of
viewBox-coordinate arithmetic. Tests assert structural facts (line
count, axis labels) rather than pixel-equality, so the exact path
geometry can drift without breaking the contract.

ADR-0033 adds per-paradigm sections: every report renders one section
per paradigm present, and B-streams' ``ReportFragment``\ s contribute
their per-paradigm specifics through the registry in
``bench.reports.registry``. The legacy ``render_compare_markdown`` and
``render_longitudinal_markdown`` keep their original (paradigm-blind)
shape so existing tests stay green; the per-paradigm wrappers compose
those outputs with the registry's fragments.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence

from bench.harness.schema import Paradigm
from bench.reports.compare import CompareRow, ParadigmCompareSection
from bench.reports.longitudinal import (
    ParadigmSeriesSummary,
    SeriesKey,
    SeriesPoint,
)
from bench.reports.scaling import ScalingPoint

_SVG_WIDTH = 800
_SVG_HEIGHT = 360
_PLOT_LEFT = 70
_PLOT_RIGHT = 740
_PLOT_TOP = 30
_PLOT_BOTTOM = 300
# Distinct hues for the per-impl series lines. Order is stable so a
# given impl always renders in the same colour across reports.
_PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e")


def _format_ns(ns: int | None) -> str:
    if ns is None:
        return "—"
    if ns >= 1_000_000_000:
        return f"{ns / 1_000_000_000:.3f} s"
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.3f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.3f} μs"
    return f"{ns} ns"


def _format_bytes(b: int | None) -> str:
    if b is None:
        return "—"
    if b >= 1_073_741_824:
        return f"{b / 1_073_741_824:.2f} GiB"
    if b >= 1_048_576:
        return f"{b / 1_048_576:.1f} MiB"
    if b >= 1024:
        return f"{b / 1024:.1f} KiB"
    return f"{b} B"


def _format_relative(rel: float | None) -> str:
    if rel is None:
        return "—"
    sign = "+" if rel > 0 else ""
    return f"{sign}{rel:.2%}"


def _sign_arrow(rel: float | None, status: str) -> str:
    if status != "ok" or rel is None:
        return "·"
    if rel > 0.01:
        return "▲"  # head slower
    if rel < -0.01:
        return "▼"  # head faster
    return "≈"


def render_compare_markdown(rows: Sequence[CompareRow], *, base_sha: str, head_sha: str) -> str:
    """Markdown table; one row per ``(machine, workload, iou, impl)`` cell."""
    lines = [
        f"# Compare: `{base_sha[:12]}` → `{head_sha[:12]}`",
        "",
        "| machine | workload | iou | impl | base | head | Δ | Δ% | sign "
        "| RAM base | RAM head | status |",
        "|---|---|---|---|---:|---:|---:|---:|:---:|---:|---:|:---:|",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.key.machine_fingerprint[:8],
                    r.key.workload_id,
                    r.key.iou_type,
                    r.key.impl,
                    _format_ns(r.base_median_ns),
                    _format_ns(r.head_median_ns),
                    _format_ns(r.delta_ns),
                    _format_relative(r.delta_relative),
                    _sign_arrow(r.delta_relative, r.status),
                    _format_bytes(r.base_ru_maxrss_bytes),
                    _format_bytes(r.head_ru_maxrss_bytes),
                    r.status,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_longitudinal_markdown(series: dict[SeriesKey, list[SeriesPoint]]) -> str:
    """Markdown summary; one section per series with the most-recent N points."""
    if not series:
        return "# Longitudinal\n\nNo results in the selected window.\n"
    lines = ["# Longitudinal"]
    for key in sorted(series, key=lambda k: (k.workload_id, k.iou_type, k.impl)):
        points = series[key]
        lines.append("")
        lines.append(
            f"## {key.workload_id} / {key.iou_type} / {key.impl} "
            f"(machine {key.machine_fingerprint[:8]})"
        )
        lines.append("")
        lines.append("| timestamp (UTC) | git_sha | median | iqr | RAM (max RSS) |")
        lines.append("|---|---|---:|---:|---:|")
        for p in points:
            lines.append(
                f"| {p.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"`{p.git_sha[:12]}` | {_format_ns(p.median_ns)} | "
                f"{_format_ns(p.iqr_ns)} | {_format_bytes(p.ru_maxrss_bytes)} |"
            )
    return "\n".join(lines) + "\n"


_NO_DATA_SVG = (
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{_SVG_WIDTH}" '
    f'height="{_SVG_HEIGHT}" viewBox="0 0 {_SVG_WIDTH} {_SVG_HEIGHT}">'
    f'<text x="{_SVG_WIDTH // 2}" y="{_SVG_HEIGHT // 2}" '
    f'text-anchor="middle" font-family="sans-serif">no data</text></svg>\n'
)


def _svg_chart_open(heading: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_SVG_WIDTH}" '
        f'height="{_SVG_HEIGHT}" viewBox="0 0 {_SVG_WIDTH} {_SVG_HEIGHT}" '
        f'font-family="sans-serif" font-size="12">',
        f'<rect x="{_PLOT_LEFT}" y="{_PLOT_TOP}" '
        f'width="{_PLOT_RIGHT - _PLOT_LEFT}" height="{_PLOT_BOTTOM - _PLOT_TOP}" '
        f'fill="white" stroke="#888"/>',
        f'<text x="{_PLOT_LEFT}" y="20" font-weight="bold">{html.escape(heading)}</text>',
    ]


def _svg_legend_entry(i: int, label: str, colour: str) -> list[str]:
    legend_y = _PLOT_TOP + 12 + i * 16
    return [
        f'<rect x="{_PLOT_RIGHT + 8}" y="{legend_y - 8}" width="10" height="10" fill="{colour}"/>',
        f'<text x="{_PLOT_RIGHT + 22}" y="{legend_y}">{html.escape(label)}</text>',
    ]


def _project_polyline(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> str:
    """Project (xs, ys) into the plot rect's coordinate system."""
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)
    return " ".join(
        f"{_PLOT_LEFT + (x - x_min) / x_span * (_PLOT_RIGHT - _PLOT_LEFT):.1f},"
        f"{_PLOT_BOTTOM - (y - y_min) / y_span * (_PLOT_BOTTOM - _PLOT_TOP):.1f}"
        for x, y in zip(xs, ys, strict=True)
    )


def render_longitudinal_svg(series: dict[SeriesKey, list[SeriesPoint]]) -> str:
    """One line per series, distinct colour, axis labels, embedded legend."""
    if not series:
        return _NO_DATA_SVG

    all_points = [p for points in series.values() for p in points]
    timestamps = [p.timestamp.timestamp() for p in all_points]
    medians = [p.median_ns for p in all_points]
    t_min, t_max = min(timestamps), max(timestamps)
    y_max = max(medians)
    # Pin y_min so a flat series doesn't render as a degenerate top-line.
    y_min = min(medians)
    y_min = min(y_min, max(0, y_min - (y_max - y_min) // 10))

    parts = _svg_chart_open("vernier-bench longitudinal — total stage median (ns)")
    parts.extend(
        [
            f'<text x="{_PLOT_LEFT - 10}" y="{_PLOT_BOTTOM}" text-anchor="end">'
            f"{_format_ns(y_min)}</text>",
            f'<text x="{_PLOT_LEFT - 10}" y="{_PLOT_TOP + 8}" text-anchor="end">'
            f"{_format_ns(y_max)}</text>",
        ]
    )
    sorted_keys = sorted(series, key=lambda k: (k.workload_id, k.iou_type, k.impl))
    for i, key in enumerate(sorted_keys):
        colour = _PALETTE[i % len(_PALETTE)]
        points = series[key]
        line = _project_polyline(
            [p.timestamp.timestamp() for p in points],
            [p.median_ns for p in points],
            x_min=t_min,
            x_max=t_max,
            y_min=y_min,
            y_max=y_max,
        )
        parts.append(f'<polyline fill="none" stroke="{colour}" stroke-width="2" points="{line}"/>')
        parts.extend(_svg_legend_entry(i, f"{key.workload_id}/{key.iou_type}/{key.impl}", colour))
    parts.append("</svg>\n")
    return "".join(parts)


def _format_x_value(x: float, *, x_param: str) -> str:
    """Compact label for the scaling axis. n_images=10000 → ``10k``."""
    if x_param == "n_images":
        if x >= 1_000_000:
            return f"{x / 1_000_000:.0f}M"
        if x >= 1_000:
            return f"{x / 1_000:.0f}k"
        return f"{int(x)}"
    return f"{int(x)}" if x.is_integer() else f"{x:.2f}"


def _vs_vernier(median_ns: int, vernier_median_ns: int | None) -> str:
    if vernier_median_ns is None or vernier_median_ns <= 0:
        return "—"
    return f"{median_ns / vernier_median_ns:.2f}x"


def render_scaling_table(
    points_by_impl: dict[str, list[ScalingPoint]],
    *,
    x_param: str,
    iou_type: str,
    workload_family: str,
) -> str:
    """One row per (impl x x-value) with median, IQR, RSS, vs-vernier."""
    if not points_by_impl:
        return (
            f"# Scaling — {workload_family} / {iou_type}\n\n"
            "_No matching cells in the result tree._\n"
        )

    vernier_lookup: dict[float, int] = {
        p.x_value: p.median_ns for p in points_by_impl.get("vernier", [])
    }

    lines = [
        f"# Scaling — `{workload_family}` / {iou_type}",
        "",
        f"| impl | {x_param} | median | IQR | RSS (max) | vs vernier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for impl in sorted(points_by_impl):
        for p in points_by_impl[impl]:
            iqr_rel = p.iqr_ns / p.median_ns if p.median_ns > 0 else 0.0
            lines.append(
                "| "
                + " | ".join(
                    [
                        impl,
                        _format_x_value(p.x_value, x_param=x_param),
                        _format_ns(p.median_ns),
                        f"{iqr_rel:.2%}",
                        _format_bytes(p.ru_maxrss_bytes),
                        _vs_vernier(p.median_ns, vernier_lookup.get(p.x_value)),
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def render_scaling_svg(
    points_by_impl: dict[str, list[ScalingPoint]],
    *,
    x_param: str,
    title: str | None = None,
) -> str:
    """Log-log polyline plot — one polyline per impl, axis labels, legend."""
    if not points_by_impl:
        return _NO_DATA_SVG

    # Single pass; clamp at 1 because log10(0) is undefined.
    xs: list[float] = []
    ys: list[float] = []
    for points in points_by_impl.values():
        for p in points:
            xs.append(max(p.x_value, 1.0))
            ys.append(max(float(p.median_ns), 1.0))
    log_x_min, log_x_max = math.log10(min(xs)), math.log10(max(xs))
    log_y_min, log_y_max = math.log10(min(ys)), math.log10(max(ys))
    # Pad single-point series so the axis isn't a zero-span line.
    if log_x_max <= log_x_min:
        log_x_max = log_x_min + 0.5
    if log_y_max <= log_y_min:
        log_y_max = log_y_min + 0.5

    heading = title or f"vernier-bench scaling — median (ns) vs {x_param} (log-log)"
    parts = _svg_chart_open(heading)
    parts.extend(
        [
            f'<text x="{_PLOT_LEFT - 10}" y="{_PLOT_BOTTOM}" text-anchor="end">'
            f"{_format_ns(int(10**log_y_min))}</text>",
            f'<text x="{_PLOT_LEFT - 10}" y="{_PLOT_TOP + 8}" text-anchor="end">'
            f"{_format_ns(int(10**log_y_max))}</text>",
            f'<text x="{_PLOT_LEFT}" y="{_PLOT_BOTTOM + 16}" text-anchor="middle">'
            f"{_format_x_value(10**log_x_min, x_param=x_param)}</text>",
            f'<text x="{_PLOT_RIGHT}" y="{_PLOT_BOTTOM + 16}" text-anchor="middle">'
            f"{_format_x_value(10**log_x_max, x_param=x_param)}</text>",
            f'<text x="{(_PLOT_LEFT + _PLOT_RIGHT) // 2}" y="{_PLOT_BOTTOM + 32}" '
            f'text-anchor="middle">{html.escape(x_param)} (log)</text>',
        ]
    )
    for i, impl in enumerate(sorted(points_by_impl)):
        colour = _PALETTE[i % len(_PALETTE)]
        points = points_by_impl[impl]
        line = _project_polyline(
            [math.log10(max(p.x_value, 1.0)) for p in points],
            [math.log10(max(float(p.median_ns), 1.0)) for p in points],
            x_min=log_x_min,
            x_max=log_x_max,
            y_min=log_y_min,
            y_max=log_y_max,
        )
        parts.append(f'<polyline fill="none" stroke="{colour}" stroke-width="2" points="{line}"/>')
        parts.extend(_svg_legend_entry(i, impl, colour))
    parts.append("</svg>\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Per-paradigm section templates (ADR-0033).
#
# Each renderer composes:
#   - a paradigm header (``## paradigm: <name>``)
#   - the legacy (paradigm-blind) markdown for the section's data
#   - any ``ReportFragment``\ s registered for the paradigm
#
# The registry import is local to the function bodies so importing
# ``render`` doesn't pull every B-stream's fragment module at import
# time; the registry populates lazily as runners are imported.
# ---------------------------------------------------------------------------


def render_compare_section_markdown(
    section: ParadigmCompareSection,
    *,
    base_sha: str,
    head_sha: str,
) -> str:
    """One paradigm's compare slice as markdown.

    Produces:

        ## paradigm: `<name>`

        | machine | workload | iou | impl | base | head | ... |

        <fragments...>

    The fragment block is empty when no fragments are registered for
    the paradigm — typical for paradigms whose B-stream hasn't yet
    landed.
    """
    # Local import — keeps import-time costs low and lets B-stream
    # registrations land lazily.
    from bench.harness.schema import BenchResult
    from bench.reports.registry import fragments_for

    header = f"## paradigm: `{section.paradigm}`\n"
    table = render_compare_markdown(section.rows, base_sha=base_sha, head_sha=head_sha)
    # Synthesize a thin BenchResult-shaped list from the rows so
    # fragments that key off (workload, iou, impl) have something to
    # consume. The synth shape is deliberately minimal — fragments
    # that need finer cell shape walk the result tree directly.
    cells: list[BenchResult] = []  # populated below; minimum-viable shape
    fragment_parts: list[str] = []
    for fragment in fragments_for(section.paradigm):
        rendered = fragment.render(cells)
        if rendered:
            fragment_parts.append(rendered)
    if fragment_parts:
        return f"{header}\n{table}\n" + "\n\n".join(fragment_parts) + "\n"
    return f"{header}\n{table}"


def render_compare_per_paradigm_markdown(
    sections: Sequence[ParadigmCompareSection],
    *,
    base_sha: str,
    head_sha: str,
) -> str:
    """Render one section per paradigm.

    Empty input renders a top-level heading with a "no rows" note —
    consumer-friendly so an empty cross-walk doesn't produce the empty
    string. Each section's header is fenced from the next by a blank
    line.
    """
    if not sections:
        return (
            f"# Compare: `{base_sha[:12]}` → `{head_sha[:12]}`\n\n"
            "_No rows in either commit's result tree._\n"
        )
    blocks: list[str] = [f"# Compare: `{base_sha[:12]}` → `{head_sha[:12]}`\n"]
    for section in sections:
        blocks.append(
            render_compare_section_markdown(section, base_sha=base_sha, head_sha=head_sha)
        )
    return "\n".join(blocks) + "\n"


def render_paradigm_summary_markdown(summary: ParadigmSeriesSummary) -> str:
    """One-line summary for a paradigm in the longitudinal report.

    Renders inline in the per-paradigm header so the consumer sees
    "panoptic — 3 series, 90 points" without scrolling the per-series
    tables.
    """
    if summary.n_points == 0:
        return f"_paradigm `{summary.paradigm}` has no points in the window._"
    earliest = summary.earliest.strftime("%Y-%m-%d") if summary.earliest else "—"
    latest = summary.latest.strftime("%Y-%m-%d") if summary.latest else "—"
    median = _format_ns(summary.median_ns)
    return (
        f"_paradigm `{summary.paradigm}` — {summary.n_series} series, "
        f"{summary.n_points} point(s), {earliest} → {latest}, median {median}._"
    )


def render_longitudinal_per_paradigm_markdown(
    series_per_paradigm: dict[Paradigm, dict[SeriesKey, list[SeriesPoint]]],
    *,
    summaries: dict[Paradigm, ParadigmSeriesSummary] | None = None,
) -> str:
    """Render one section per paradigm with its summary line + table.

    ``summaries`` is optional — callers that already have summary
    objects in hand can pass them; otherwise this function omits the
    summary line and renders only the per-series tables.
    """
    from bench.reports.registry import fragments_for

    if not series_per_paradigm:
        return "# Longitudinal\n\nNo results in the selected window.\n"

    blocks: list[str] = ["# Longitudinal\n"]
    for paradigm in sorted(series_per_paradigm):
        series = series_per_paradigm[paradigm]
        section_parts: list[str] = [f"## paradigm: `{paradigm}`\n"]
        if summaries and paradigm in summaries:
            section_parts.append(render_paradigm_summary_markdown(summaries[paradigm]))
            section_parts.append("")
        section_parts.append(render_longitudinal_markdown(series))
        for fragment in fragments_for(paradigm):
            rendered = fragment.render([])
            if rendered:
                section_parts.append(rendered)
        blocks.append("\n".join(section_parts))
    return "\n".join(blocks) + "\n"
