"""Every runner under ``bench/runners/*_runner.py`` accepts the same CLI.

The orchestrator builds one argv shape and assumes it works for every
impl; this test introspects each runner's ``--help`` and asserts every
required flag from the shared argspec is present.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from bench.harness.matrix import (
    ALL_IMPLS,
    IMPL_PARADIGM_SUPPORT,
    runner_module,
    uv_run_argv,
    uv_run_env,
)
from bench.harness.paths import BENCH_ROOT
from tests.conftest import skip_if_no_env

# The orchestrator builds one argv shape *per paradigm*, not one for the
# whole matrix: `_spawn_one_rep` (instance / lvis), `_spawn_one_rep_panoptic`
# and `_spawn_one_rep_semantic` in `bench/harness/orchestrate.py`. Each
# entry below is the flag set its spawn function actually passes, so this
# test fails if a runner drifts from the argv it will be invoked with.
_INSTANCE_FLAGS: tuple[str, ...] = (
    "--gt",
    "--dt",
    "--iou-type",
    "--workload-id",
    "--output",
    "--tensor-output",
)
_PANOPTIC_FLAGS: tuple[str, ...] = (
    "--gt-png-dir",
    "--gt-json",
    "--dt-png-dir",
    "--dt-json",
    "--categories-json",
    "--workload-id",
    "--paradigm",
    "--output",
    "--snapshot-output",
    "--per-class-output",
)
_SEMANTIC_FLAGS: tuple[str, ...] = (
    "--gt-label-map-dir",
    "--dt-label-map-dir",
    "--n-classes",
    "--ignore-label",
    "--workload-id",
    "--paradigm",
    "--output",
    "--snapshot-output",
    "--per-class-output",
    "--confusion-output",
)
_REQUIRED_FLAGS: dict[str, tuple[str, ...]] = {
    "instance": _INSTANCE_FLAGS,
    "lvis": _INSTANCE_FLAGS,
    "streaming": _INSTANCE_FLAGS,
    "panoptic": _PANOPTIC_FLAGS,
    "semantic": _SEMANTIC_FLAGS,
}

_IMPL_PARADIGM: dict[str, str] = {
    impl: paradigm for paradigm, table in IMPL_PARADIGM_SUPPORT.items() for impl in table
}


def _advertises(help_text: str, flag: str) -> bool:
    """Whole-flag match.

    A plain ``flag in help_text`` reports ``--gt`` as present in a
    runner whose only such option is ``--gt-png-dir``, which is how the
    panoptic and semantic runners passed four of this test's six
    assertions while advertising none of them.
    """
    return re.search(rf"(?<![-\w]){re.escape(flag)}(?![-\w])", help_text) is not None


@pytest.mark.parametrize("impl", ALL_IMPLS)
def test_runner_help_advertises_protocol(impl: str) -> None:
    skip_if_no_env(impl)

    proc = subprocess.run(
        uv_run_argv(BENCH_ROOT, impl, "-m", runner_module(impl), "--help"),
        env=uv_run_env(BENCH_ROOT, impl),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"--help failed for {impl}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    paradigm = _IMPL_PARADIGM[impl]
    for flag in _REQUIRED_FLAGS[paradigm]:
        assert _advertises(proc.stdout, flag), (
            f"{impl} runner ({paradigm}) --help missing required flag {flag}; got:\n{proc.stdout}"
        )
