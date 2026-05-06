"""``vernier-bench`` command-line entry point.

``run`` fans out across the impls supported for the requested
(paradigm, workload, metric) cell, then runs the cross-impl parity
check unless ``--no-parity`` is passed. Skipped cells are silent for
``--impl all`` and loud for an explicit ``--impl <name>``. Release mode
adds a CPU governor pre-flight and an IQR-relative-to-median gate;
profile mode defaults to skipping parity (instrumentation perturbs
measurement).

ADR-0033 adds the ``--paradigm`` flag. It auto-derives from the
workload variant when unambiguous (no workload name is reused across
paradigms today); the explicit override is the future-proofing escape
hatch and the only way to invoke a paradigm whose B-stream has
populated ``IMPL_PARADIGM_SUPPORT``.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import cast, get_args

import click

from bench.harness.matrix import (
    ALL_IMPLS,
    IMPL_IOU_SUPPORT,
    impls_for_iou,
)
from bench.harness.migrations.v1_to_v2_tree import migrate_tree
from bench.harness.orchestrate import CellSpec, run_cell
from bench.harness.paths import BENCH_ROOT, REPO_ROOT
from bench.harness.platform_check import UnsupportedPlatformError, ensure_linux
from bench.harness.schema import IouType, Mode, Paradigm
from bench.reports.compare import compare_shas
from bench.reports.load import load_tree
from bench.reports.longitudinal import build_series, parse_since
from bench.reports.render import (
    render_compare_markdown,
    render_longitudinal_markdown,
    render_longitudinal_svg,
)
from bench.workloads import InstanceWorkload, resolve

_IOU_CHOICES: tuple[str, ...] = get_args(IouType)
_MODE_CHOICES: tuple[str, ...] = get_args(Mode)
_PARADIGM_CHOICES: tuple[str, ...] = get_args(Paradigm)
# ``all`` fans across paradigms (Phase C dry-run target). The auto
# value resolves at run time from the workload's discriminator.
_PARADIGM_FLAG_CHOICES: tuple[str, ...] = ("auto", "all", *_PARADIGM_CHOICES)


@click.group()
def main() -> None:
    """vernier-bench — local benchmarking harness (ADR-0017, ADR-0033)."""
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
    "--paradigm",
    type=click.Choice(_PARADIGM_FLAG_CHOICES),
    default="auto",
    show_default=True,
    help=(
        "Evaluation paradigm. 'auto' (default) reads it from the workload's "
        "tagged-union discriminator; 'all' fans across paradigms. Explicit "
        "values override; an explicit value that disagrees with the workload "
        "is rejected at parse time."
    ),
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
    impl: str,
    workload: str,
    paradigm: str,
    iou_type: str,
    mode: str,
    run_seed: int,
    no_parity: bool,
) -> None:
    try:
        workload_obj = resolve(workload, REPO_ROOT)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    except NotImplementedError as e:
        raise click.ClickException(str(e)) from e
    iou = cast(IouType, iou_type)
    mode_typed = cast(Mode, mode)

    # Resolve --paradigm against the workload's discriminator. ``all``
    # is reserved for the Phase-C cross-paradigm dry-run; today only
    # the auto-derived path has any workloads to run.
    derived: Paradigm = workload_obj.paradigm
    if paradigm == "auto":
        paradigm_typed: Paradigm = derived
    elif paradigm == "all":
        # No fan-out target until B1/B2/B3 land their workloads.
        raise click.ClickException(
            "--paradigm all is reserved for the Phase-C cross-paradigm dry "
            "run; B1/B2/B3 register their workloads first."
        )
    else:
        explicit = cast(Paradigm, paradigm)
        if explicit != derived:
            raise click.ClickException(
                f"workload {workload!r} resolves to paradigm {derived!r}, "
                f"but --paradigm {explicit!r} was passed. Drop the flag "
                f"(default 'auto' derives it) or align the workload."
            )
        paradigm_typed = explicit

    # Per ADR-0033, paradigm/metric mismatches are rejected here. The
    # instance paradigm uses iou-types (bbox/segm/keypoints/boundary);
    # other paradigms use their own metric names (pq, miou,
    # throughput, ...) and the user shouldn't pass --iou for them at
    # all. Until B1/B2/B3 plumb a ``--metric`` flag, the safest
    # behaviour is to reject every non-instance run from this CLI.
    if paradigm_typed != "instance":
        raise click.ClickException(
            f"--paradigm {paradigm_typed!r} requires the corresponding B-stream's "
            f"runner registration to land before vernier-bench run can dispatch "
            f"to it. See ADR-0033 §'IMPL_PARADIGM_SUPPORT'."
        )

    if not isinstance(workload_obj, InstanceWorkload):  # narrowing for type checker
        raise click.ClickException(
            f"workload {workload!r} is not an instance workload; "
            f"the bench CLI currently only fans out instance cells."
        )

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
        paradigm=paradigm_typed,
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
    "--paradigm",
    type=click.Choice(_PARADIGM_FLAG_CHOICES),
    default="all",
    show_default=True,
    help=(
        "Scope the compare to a single paradigm, or 'all' to render one "
        "delta table per paradigm. Cross-paradigm comparison is rejected "
        "(category error per ADR-0032)."
    ),
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write the markdown table to this path; default is stdout.",
)
def compare_cmd(base_sha: str, head_sha: str, paradigm: str, output: str | None) -> None:
    """Render a per-cell delta table for ``base_sha`` vs ``head_sha``.

    Compare scopes per-paradigm: one delta table per paradigm. Passing
    ``--paradigm all`` (the default) renders every paradigm's table
    in turn; passing an explicit paradigm narrows to one. Cross-
    paradigm comparison ("PQ vs AP") is structurally rejected — the
    metric units don't compose.
    """
    if paradigm == "auto":
        # ``auto`` requires a workload to disambiguate; compare doesn't
        # know one.
        raise click.ClickException(
            "--paradigm auto isn't valid for compare; pass an explicit paradigm "
            "or 'all' (default)."
        )

    df = load_tree(BENCH_ROOT / "results", shas={base_sha, head_sha})
    if df.is_empty():
        raise click.ClickException(
            f"no rows for either {base_sha[:12]} or {head_sha[:12]} under bench/results/"
        )

    paradigms_to_render: tuple[Paradigm, ...] = (
        _PARADIGM_CHOICES  # type: ignore[assignment]
        if paradigm == "all"
        else (cast(Paradigm, paradigm),)
    )

    sections: list[str] = []
    any_rows = False
    for p in paradigms_to_render:
        if "paradigm" in df.columns:
            df_p = df.filter(df["paradigm"] == p)
        else:
            # v1-only tree (no paradigm column); only the instance
            # paradigm has any rows.
            df_p = df if p == "instance" else df.head(0)
        if df_p.is_empty():
            continue
        rows = compare_shas(df_p, base_sha=base_sha, head_sha=head_sha)
        if not rows:
            continue
        any_rows = True
        md = render_compare_markdown(rows, base_sha=base_sha, head_sha=head_sha)
        # Prepend a paradigm header to disambiguate the output —
        # consistent with ADR-0033's per-paradigm scoping.
        sections.append(f"## paradigm: `{p}`\n\n{md}")

    if not any_rows:
        raise click.ClickException(
            f"no rows for either {base_sha[:12]} or {head_sha[:12]} "
            f"under the requested paradigm scope ({paradigm!r})."
        )

    md_out = "\n".join(sections)
    if output:
        Path(output).write_text(md_out)
        click.echo(f"compare: {output}")
    else:
        click.echo(md_out, nl=False)


@main.command("report")
@click.option(
    "--since",
    "since_spec",
    default="30d",
    show_default=True,
    help="Window like 30d / 6h / 2w / 90m.",
)
@click.option(
    "--paradigm",
    type=click.Choice(_PARADIGM_FLAG_CHOICES),
    default="all",
    show_default=True,
    help="Scope the longitudinal report to a single paradigm, or 'all'.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, writable=True),
    default=None,
    help="Write report.md and report.svg here; default is stdout (markdown only).",
)
def report_cmd(since_spec: str, paradigm: str, output_dir: str | None) -> None:
    """Render a longitudinal view for the last ``--since`` window."""
    if paradigm == "auto":
        raise click.ClickException(
            "--paradigm auto isn't valid for report; pass an explicit paradigm "
            "or 'all' (default)."
        )
    try:
        since = parse_since(since_spec)
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    cutoff = (datetime.now(tz=timezone.utc) - since).timestamp()
    df = load_tree(BENCH_ROOT / "results", mtime_after=cutoff)

    paradigms_to_render: tuple[Paradigm, ...] = (
        _PARADIGM_CHOICES  # type: ignore[assignment]
        if paradigm == "all"
        else (cast(Paradigm, paradigm),)
    )

    md_sections: list[str] = []
    svg_sections: list[str] = []
    for p in paradigms_to_render:
        if "paradigm" in df.columns:
            df_p = df.filter(df["paradigm"] == p)
        else:
            df_p = df if p == "instance" else df.head(0)
        if df_p.is_empty():
            continue
        series = build_series(df_p)
        md_sections.append(f"## paradigm: `{p}`\n\n{render_longitudinal_markdown(series)}")
        svg_sections.append(render_longitudinal_svg(series))

    md = (
        "\n".join(md_sections)
        if md_sections
        else "# Longitudinal\n\nNo results in the selected window.\n"
    )
    # The SVG renderer is single-paradigm — keep the first non-empty
    # one for now; A-thick will extend the SVG renderer for
    # multi-paradigm output if anyone needs it.
    svg = svg_sections[0] if svg_sections else render_longitudinal_svg({})

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(md)
        (out / "report.svg").write_text(svg)
        click.echo(f"report: {out / 'report.md'}")
        click.echo(f"report: {out / 'report.svg'}")
    else:
        click.echo(md, nl=False)


@main.group("migrate")
def migrate_grp() -> None:
    """On-disk migrations for the result-store tree.

    Subcommands re-emit existing cells at the v2 paradigm-segmented
    paths (ADR-0033 §"Result-store path scheme v2"). The read-side
    ``v1_to_v2`` shim handles already-loaded v1 results; this group
    rewrites the on-disk shape so new writes don't sit alongside v1
    leftovers.
    """


@migrate_grp.command("v1-to-v2")
@click.option(
    "--keep-v1-paths/--no-keep-v1-paths",
    default=True,
    show_default=True,
    help=(
        "Keep the v1 cell directories alongside the new v2 mirrors "
        "(default; reversible). --no-keep-v1-paths deletes each v1 "
        "cell once its v2 mirror is in place."
    ),
)
@click.option(
    "--results-root",
    type=click.Path(file_okay=False, exists=False),
    default=None,
    help=(
        "Override the default ``bench/results/`` location (used by "
        "tests that operate on a synthetic tree)."
    ),
)
def migrate_v1_to_v2_cmd(keep_v1_paths: bool, results_root: str | None) -> None:
    """Forward-migrate a v1 result tree into the v2 paradigm-segmented layout.

    Idempotent — re-running on an already-migrated tree only walks the
    walker; per-file rewrites are content-compared and skipped when the
    target already matches.
    """
    root = Path(results_root) if results_root else (BENCH_ROOT / "results")
    stats = migrate_tree(root, keep_v1_paths=keep_v1_paths)
    click.echo(
        f"migrated {stats.cells_migrated} cell(s); "
        f"{stats.files_copied} file(s) written, "
        f"{stats.files_skipped_already_v2} already at v2"
    )
    if not keep_v1_paths:
        click.echo(f"removed {stats.v1_paths_removed} v1 cell director(y/ies)")


if __name__ == "__main__":
    sys.exit(main())
