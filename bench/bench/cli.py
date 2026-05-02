"""``vernier-bench`` command-line entry point.

``run`` fans out across the impls supported for the requested
(workload, iou) cell, then runs the cross-impl parity check unless
``--no-parity`` is passed. Skipped cells are silent for ``--impl all``
and loud for an explicit ``--impl <name>``. Release mode adds a CPU
governor pre-flight and an IQR-relative-to-median gate; profile mode
defaults to skipping parity (instrumentation perturbs measurement).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import cast, get_args

import click

from bench.harness.matrix import ALL_IMPLS, IMPL_IOU_SUPPORT, impls_for_iou
from bench.harness.orchestrate import CellSpec, run_cell
from bench.harness.paths import BENCH_ROOT, REPO_ROOT
from bench.harness.platform_check import UnsupportedPlatformError, ensure_linux
from bench.harness.schema import IouType, Mode
from bench.reports.compare import compare_shas
from bench.reports.load import load_tree
from bench.reports.longitudinal import build_series, parse_since
from bench.reports.render import (
    render_compare_markdown,
    render_longitudinal_markdown,
    render_longitudinal_svg,
)
from bench.workloads import resolve

_IOU_CHOICES: tuple[str, ...] = get_args(IouType)
_MODE_CHOICES: tuple[str, ...] = get_args(Mode)


@click.group()
def main() -> None:
    """vernier-bench — local benchmarking harness (ADR-0017)."""
    try:
        ensure_linux()
    except UnsupportedPlatformError as e:
        raise click.ClickException(str(e)) from e


@main.command("run")
@click.option(
    "--impl",
    type=click.Choice(["all", *ALL_IMPLS]),
    default="vernier",
    show_default=True,
    help="Implementation to run, or 'all' to fan out across the (impl, iou) matrix.",
)
@click.option(
    "--workload",
    type=str,
    default="smoke",
    show_default=True,
    help="Workload identifier (smoke | coco_val2017_jittered_seed<N> | synthetic:k=v,...).",
)
@click.option(
    "--iou",
    "iou_type",
    type=click.Choice(_IOU_CHOICES),
    default="bbox",
    show_default=True,
)
@click.option(
    "--mode",
    type=click.Choice(_MODE_CHOICES),
    default="dev",
    show_default=True,
    help=(
        "Run mode profile: dev (1 rep), release (10 reps + warmup + governor + IQR gate), "
        "profile (1 rep, parity skipped)."
    ),
)
@click.option("--seed", "run_seed", type=int, default=0, show_default=True)
@click.option(
    "--no-parity",
    is_flag=True,
    default=False,
    help="Skip the cross-impl parity check after the fan-out.",
)
def run_cmd(
    impl: str, workload: str, iou_type: str, mode: str, run_seed: int, no_parity: bool
) -> None:
    try:
        workload_obj = resolve(workload, REPO_ROOT)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    iou = cast(IouType, iou_type)
    mode_typed = cast(Mode, mode)

    if iou not in workload_obj.supported_iou_types:
        supported = ", ".join(sorted(workload_obj.supported_iou_types))
        raise click.ClickException(
            f"workload {workload!r} does not support --iou {iou_type}; supported: {supported}"
        )

    if impl == "all":
        impls = impls_for_iou(iou)
        if not impls:
            raise click.ClickException(f"no impl in the matrix supports --iou {iou_type}")
    else:
        if iou not in IMPL_IOU_SUPPORT[impl]:
            supported = ", ".join(sorted(IMPL_IOU_SUPPORT[impl]))
            raise click.ClickException(
                f"impl {impl!r} does not support --iou {iou_type}; supported: {supported}"
            )
        impls = [impl]

    cell = CellSpec(
        bench_root=BENCH_ROOT,
        repo_root=REPO_ROOT,
        impls=impls,
        workload_id=workload_obj.workload_id,
        iou_type=iou,
        gt_path=workload_obj.gt_path,
        dt_path=workload_obj.dt_path,
        mode=mode_typed,
        run_seed=run_seed,
    )
    # Profile mode skips parity by default per ADR-0017 §"Run modes":
    # instrumentation perturbs which code paths warm up.
    do_parity = not no_parity and mode_typed != "profile"
    result = run_cell(cell, parity=do_parity)

    for impl_name, json_path in result.impl_jsons.items():
        click.echo(f"{impl_name}: {json_path}")

    for impl_name, outcome in result.iqr_outcomes.items():
        status = "OK" if outcome.passed else "FAILED"
        click.echo(
            f"iqr_gate[{impl_name}]: {status} "
            f"(rel={outcome.relative:.3%}, threshold={outcome.threshold:.1%})",
            err=not outcome.passed,
        )

    if result.parity is not None:
        if result.parity.passed:
            click.echo(f"parity: OK ({len(result.parity.tiers)} tier(s) checked)")
        else:
            click.echo(
                f"parity: FAILED — see {result.divergence_report_path}",
                err=True,
            )


@main.command("compare")
@click.option("--base", "base_sha", required=True, help="Base git sha (full or 12-char prefix).")
@click.option("--head", "head_sha", required=True, help="Head git sha (full or 12-char prefix).")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write the markdown table to this path; default is stdout.",
)
def compare_cmd(base_sha: str, head_sha: str, output: str | None) -> None:
    """Render a per-cell delta table for ``base_sha`` vs ``head_sha``."""
    df = load_tree(BENCH_ROOT / "results", shas={base_sha, head_sha})
    rows = compare_shas(df, base_sha=base_sha, head_sha=head_sha)
    if not rows:
        raise click.ClickException(
            f"no rows for either {base_sha[:12]} or {head_sha[:12]} under bench/results/"
        )
    md = render_compare_markdown(rows, base_sha=base_sha, head_sha=head_sha)
    if output:
        Path(output).write_text(md)
        click.echo(f"compare: {output}")
    else:
        click.echo(md, nl=False)


@main.command("report")
@click.option(
    "--since",
    "since_spec",
    default="30d",
    show_default=True,
    help="Window like 30d / 6h / 2w / 90m.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, writable=True),
    default=None,
    help="Write report.md and report.svg here; default is stdout (markdown only).",
)
def report_cmd(since_spec: str, output_dir: str | None) -> None:
    """Render a longitudinal view for the last ``--since`` window."""
    try:
        since = parse_since(since_spec)
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    cutoff = (datetime.now(tz=timezone.utc) - since).timestamp()
    df = load_tree(BENCH_ROOT / "results", mtime_after=cutoff)
    series = build_series(df)
    md = render_longitudinal_markdown(series)
    svg = render_longitudinal_svg(series)
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(md)
        (out / "report.svg").write_text(svg)
        click.echo(f"report: {out / 'report.md'}")
        click.echo(f"report: {out / 'report.svg'}")
    else:
        click.echo(md, nl=False)


if __name__ == "__main__":
    sys.exit(main())
