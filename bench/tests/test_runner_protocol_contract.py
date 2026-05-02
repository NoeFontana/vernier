"""Every runner under ``bench/runners/*_runner.py`` accepts the same CLI.

The orchestrator builds one argv shape and assumes it works for every
impl; this test introspects each runner's ``--help`` and asserts every
required flag from the shared argspec is present.
"""

from __future__ import annotations

import subprocess

import pytest

from bench.harness.matrix import ALL_IMPLS, runner_module, uv_run_argv
from tests.conftest import BENCH_ROOT, skip_if_no_env

_REQUIRED_FLAGS: tuple[str, ...] = (
    "--gt",
    "--dt",
    "--iou-type",
    "--workload-id",
    "--output",
    "--tensor-output",
)


@pytest.mark.parametrize("impl", ALL_IMPLS)
def test_runner_help_advertises_protocol(impl: str) -> None:
    skip_if_no_env(impl)

    proc = subprocess.run(
        uv_run_argv(BENCH_ROOT, impl, "-m", runner_module(impl), "--help"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"--help failed for {impl}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    for flag in _REQUIRED_FLAGS:
        assert flag in proc.stdout, (
            f"{impl} runner --help missing required flag {flag}; got:\n{proc.stdout}"
        )
