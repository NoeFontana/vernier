"""``vernier-bench`` command-line entry point.

``run`` fans out across the impls supported for the requested
(workload, iou) cell. Skipped cells are silent for ``--impl all`` and
loud for an explicit ``--impl <name>``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast, get_args

import click

from bench.harness.matrix import ALL_IMPLS, IMPL_IOU_SUPPORT, impls_for_iou
from bench.harness.orchestrate import RunSpec
from bench.harness.orchestrate import run as run_spec
from bench.harness.schema import IouType
from bench.workloads import resolve

_IOU_CHOICES: tuple[str, ...] = get_args(IouType)


def _bench_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _repo_root() -> Path:
    return _bench_root().parent


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
def run_cmd(impl: str, workload: str, iou_type: str, mode: str, run_seed: int) -> None:
    bench_root = _bench_root()
    repo_root = _repo_root()
    workload_obj = resolve(workload, repo_root)
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

    for impl_name in impls:
        spec = RunSpec(
            bench_root=bench_root,
            repo_root=repo_root,
            impl=impl_name,
            workload_id=workload_obj.workload_id,
            iou_type=iou,
            gt_path=workload_obj.gt_path,
            dt_path=workload_obj.dt_path,
            mode=mode,
            run_seed=run_seed,
        )
        out = run_spec(spec)
        click.echo(f"{impl_name}: {out}")


if __name__ == "__main__":
    sys.exit(main())
