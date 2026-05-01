"""``vernier-bench`` command-line entry point.

M1: a single ``run`` subcommand that exercises one (impl, workload, iou)
cell. M5 adds ``--mode`` defaults; M6 adds ``compare`` and ``report``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from bench.harness.orchestrate import RunSpec
from bench.harness.orchestrate import run as run_spec
from bench.workloads import resolve

# These are the only impls a runner exists for. ``--impl all`` expands to
# this list filtered by the (workload, iou) matrix in M2.
SUPPORTED_IMPLS = ("vernier",)


def _bench_root() -> Path:
    # ``bench/bench/cli.py`` -> ``bench/``
    return Path(__file__).resolve().parent.parent


def _repo_root() -> Path:
    return _bench_root().parent


@click.group()
def main() -> None:
    """vernier-bench — local benchmarking harness (ADR-0017)."""


@main.command("run")
@click.option(
    "--impl",
    type=click.Choice(["vernier", "all"]),
    default="vernier",
    show_default=True,
    help="Which implementation(s) to run. M1 supports vernier only.",
)
@click.option(
    "--workload",
    type=str,
    default="smoke",
    show_default=True,
    help="Workload identifier. M1 only knows 'smoke'.",
)
@click.option(
    "--iou",
    "iou_type",
    type=click.Choice(["bbox", "segm", "keypoints", "boundary"]),
    default="bbox",
    show_default=True,
)
@click.option(
    "--mode",
    type=click.Choice(["dev"]),
    default="dev",
    show_default=True,
    help="Run mode. M1 supports dev only; release/profile land in M5.",
)
@click.option("--seed", "run_seed", type=int, default=0, show_default=True)
def run_cmd(
    impl: str, workload: str, iou_type: str, mode: str, run_seed: int
) -> None:
    impls = list(SUPPORTED_IMPLS) if impl == "all" else [impl]

    bench_root = _bench_root()
    repo_root = _repo_root()
    workload_obj = resolve(workload, repo_root)

    for impl_name in impls:
        spec = RunSpec(
            bench_root=bench_root,
            repo_root=repo_root,
            impl=impl_name,
            workload_id=workload_obj.workload_id,
            iou_type=iou_type,
            gt_path=workload_obj.gt_path,
            dt_path=workload_obj.dt_path,
            mode=mode,
            run_seed=run_seed,
        )
        out = run_spec(spec)
        click.echo(f"{impl_name}: {out}")


if __name__ == "__main__":
    sys.exit(main())
