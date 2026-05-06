"""Vernier keypoints runner vs pycocotools keypoints runner — strict
parity over the ``(T, R, K, A, M)`` precision tensor on the parity-suite
keypoints fixture.

This guards two things at once:

- The vernier runner's keypoints dispatch (``evaluate_keypoints_grid``)
  is reachable and produces a complete output; if a future refactor
  drops the iou-type branch, the JSON write fails and surfaces the
  regression here, not in production.
- pycocotools' keypoints surface (``COCOeval(iouType='keypoints')``)
  matches vernier bit-for-bit on a perfect-match fixture. The fixture
  is the same one ``tests/python/parity/`` uses, so a divergence
  here also flags a parity-suite regression.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from bench.harness.matrix import runner_module, uv_run_argv, uv_run_env
from bench.harness.paths import BENCH_ROOT, REPO_ROOT
from tests.conftest import skip_if_no_env

_KP_FIXTURE = (
    REPO_ROOT / "tests" / "python" / "parity" / "fixtures" / "keypoints_perfect_match"
)


def _spawn_runner(
    impl: str,
    *,
    gt: Path,
    dt: Path,
    workload_id: str,
    output: Path,
    tensor_output: Path,
) -> None:
    cmd = uv_run_argv(
        BENCH_ROOT,
        impl,
        "-m",
        runner_module(impl),
        "--gt",
        str(gt),
        "--dt",
        str(dt),
        "--iou-type",
        "keypoints",
        "--workload-id",
        workload_id,
        "--output",
        str(output),
        "--tensor-output",
        str(tensor_output),
    )
    proc = subprocess.run(
        cmd, env=uv_run_env(BENCH_ROOT, impl), check=False, capture_output=True
    )
    assert proc.returncode == 0, (
        f"runner {impl!r} exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout.decode(errors='replace')}\n"
        f"stderr:\n{proc.stderr.decode(errors='replace')}\n"
    )


def test_keypoints_runner_strict_parity(tmp_path: Path) -> None:
    """vernier vs pycocotools on a hand-built perfect-match keypoints
    fixture. Strict tier — bit-equal precision tensors."""
    skip_if_no_env("vernier")
    skip_if_no_env("pycocotools")

    gt = _KP_FIXTURE / "gt.json"
    dt = _KP_FIXTURE / "dt.json"
    assert gt.exists(), gt
    assert dt.exists(), dt

    vernier_json = tmp_path / "vernier.json"
    vernier_npy = tmp_path / "vernier.npy"
    _spawn_runner(
        "vernier",
        gt=gt,
        dt=dt,
        workload_id="keypoints_perfect_match",
        output=vernier_json,
        tensor_output=vernier_npy,
    )

    pyc_json = tmp_path / "pycocotools.json"
    pyc_npy = tmp_path / "pycocotools.npy"
    _spawn_runner(
        "pycocotools",
        gt=gt,
        dt=dt,
        workload_id="keypoints_perfect_match",
        output=pyc_json,
        tensor_output=pyc_npy,
    )

    vernier_payload = json.loads(vernier_json.read_text())
    pyc_payload = json.loads(pyc_json.read_text())
    assert vernier_payload["iou_type"] == "keypoints"
    assert pyc_payload["iou_type"] == "keypoints"
    # Both runners must report the keypoints stat-name set (10 entries,
    # not the 12-entry bbox set).
    assert sorted(vernier_payload["summary_stats"].keys()) == sorted(
        pyc_payload["summary_stats"].keys()
    )

    vernier_tensor = np.load(vernier_npy)
    pyc_tensor = np.load(pyc_npy)
    # Strict tier: bit-equality. Use ``np.array_equal`` (not ``allclose``)
    # so a quiet float-order regression doesn't slip through.
    assert vernier_tensor.shape == pyc_tensor.shape, (vernier_tensor.shape, pyc_tensor.shape)
    assert np.array_equal(vernier_tensor, pyc_tensor), (
        f"keypoints precision tensor disagreement; max abs diff "
        f"{float(np.abs(vernier_tensor - pyc_tensor).max())}"
    )
