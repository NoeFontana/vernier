"""``vernier-bench`` command-line entry point.

``run`` fans out across the impls supported for the requested
(workload, iou) cell, then runs the cross-impl parity check unless
``--no-parity`` is passed. Skipped cells are silent for ``--impl all``
and loud for an explicit ``--impl <name>``.
"""

from __future__ import annotations

import sys
from typing import cast, get_args

import click

from bench.harness.matrix import ALL_IMPLS, IMPL_IOU_SUPPORT, impls_for_iou
from bench.harness.orchestrate import CellSpec, run_cell
from bench.harness.paths import BENCH_ROOT, REPO_ROOT
from bench.harness.schema import IouType
from bench.workloads import resolve

_IOU_CHOICES: tuple[str, ...] = get_args(IouType)


@click.group()
def main() -> None:
    """vernier-bench — local benchmarking harness (ADR-0017)."""


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
    help="Workload identifier. M1/M2 only knows 'smoke'.",
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
    type=click.Choice(["dev"]),
    default="dev",
    show_default=True,
    help="Run mode. M1/M2 supports dev only; release/profile land in M5.",
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
    workload_obj = resolve(workload, REPO_ROOT)
    iou = cast(IouType, iou_type)

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
        mode=mode,
        run_seed=run_seed,
    )
    result = run_cell(cell, parity=not no_parity)

    for impl_name, json_path in result.impl_jsons.items():
        click.echo(f"{impl_name}: {json_path}")

    if result.parity is not None:
        if result.parity.passed:
            click.echo(f"parity: OK ({len(result.parity.tiers)} tier(s) checked)")
        else:
            click.echo(
                f"parity: FAILED — see {result.divergence_report_path}",
                err=True,
            )


if __name__ == "__main__":
    sys.exit(main())
