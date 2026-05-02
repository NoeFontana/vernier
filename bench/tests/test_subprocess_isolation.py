"""ADR-0017 test plan §1 — subprocess-per-impl isolation.

Each baseline runner needs its own pycocotools-namespace flavor:
upstream, faster-coco-eval's drop-in shim, and the boundary oracle's
``boundary_iou.coco_instance_api.coco``. Two of them sharing a process
would let whichever loaded last win the namespace and silently
benchmark the wrong implementation.

Each runner imports the COCO class it expects via its own bootstrap;
this test runs that bootstrap and asserts the bound class lands where
the runner thinks it does.
"""

from __future__ import annotations

import subprocess

import pytest

from bench.harness.matrix import runner_module, uv_run_argv
from tests.conftest import BENCH_ROOT, skip_if_no_env

_PROBES: list[tuple[str, str]] = [
    ("pycocotools", "pycocotools.coco"),
    ("faster-coco-eval", "faster_coco_eval"),
    ("boundary-iou-api", "boundary_iou.coco_instance_api.coco"),
]


@pytest.mark.parametrize(("impl", "expected"), _PROBES, ids=[p[0] for p in _PROBES])
def test_runner_module_binds_expected_namespace(impl: str, expected: str) -> None:
    skip_if_no_env(impl)

    script = f"import {runner_module(impl)} as r; print(r.COCO.__module__)"
    proc = subprocess.run(
        uv_run_argv(BENCH_ROOT, impl, "-c", script),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"probe failed in env {impl}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert expected in proc.stdout, (
        f"env {impl}: expected COCO module to contain {expected!r}, got {proc.stdout.strip()!r}"
    )


def test_vernier_runner_does_not_import_pycocotools() -> None:
    """Guards against a refactor accidentally re-introducing a transitive
    pycocotools dep through the vernier runner."""
    skip_if_no_env("vernier")

    proc = subprocess.run(
        uv_run_argv(
            BENCH_ROOT,
            "vernier",
            "-c",
            "import sys; import bench.runners.vernier_runner; "
            "assert 'pycocotools' not in sys.modules, "
            "'vernier runner pulled in pycocotools'; "
            "assert 'faster_coco_eval' not in sys.modules; "
            "assert 'boundary_iou' not in sys.modules; print('ok')",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"probe failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "ok" in proc.stdout
