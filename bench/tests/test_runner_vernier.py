"""End-to-end test for the vernier runner — invoke it as a subprocess
against the smoke fixture, validate the JSON shape and tensor."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from bench.harness.matrix import runner_module, uv_run_argv
from tests.conftest import BENCH_ROOT, skip_if_no_env

REPO_ROOT = BENCH_ROOT.parent


def test_vernier_runner_smoke(tmp_path: Path) -> None:
    skip_if_no_env("vernier")
    from bench.workloads import resolve

    workload = resolve("smoke", REPO_ROOT)
    output = tmp_path / "vernier.json"
    tensor_output = tmp_path / "vernier.npy"

    cmd = uv_run_argv(
        BENCH_ROOT,
        "vernier",
        "-m",
        runner_module("vernier"),
        "--gt",
        str(workload.gt_path),
        "--dt",
        str(workload.dt_path),
        "--iou-type",
        "bbox",
        "--workload-id",
        workload.workload_id,
        "--output",
        str(output),
        "--tensor-output",
        str(tensor_output),
    )
    proc = subprocess.run(cmd, check=False, capture_output=True)
    assert proc.returncode == 0, (
        f"runner exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout.decode(errors='replace')}\n"
        f"stderr:\n{proc.stderr.decode(errors='replace')}\n"
    )

    assert output.exists()
    assert tensor_output.exists()

    payload = json.loads(output.read_text())
    assert payload["schema_version"] == 1
    assert payload["impl"] == "vernier"
    assert payload["iou_type"] == "bbox"
    assert payload["workload_id"] == workload.workload_id
    for stage in ("load", "evaluate", "accumulate", "summarize", "total"):
        assert stage in payload["stages"], f"missing stage: {stage}"
        assert payload["stages"][stage]["wall_ns"] >= 0

    # Smoke fixture is a single perfect match → AP should be 1.0.
    assert payload["summary_stats"]["AP"] == pytest.approx(1.0)
    assert payload["summary_stats"]["AP50"] == pytest.approx(1.0)

    expected_sha = hashlib.sha256(tensor_output.read_bytes()).hexdigest()
    assert payload["tensor_sha256"] == expected_sha

    # (T, R, K, A, M) for default detection: (10, 101, 1, 4, 3).
    tensor = np.load(tensor_output)
    assert tensor.shape == (10, 101, 1, 4, 3), tensor.shape


def test_orchestrator_run_writes_tree(tmp_path: Path) -> None:
    """Drive the orchestrator end-to-end against an isolated bench root.

    Uses the real envs/ tree (so the runner subprocess has its venv) but
    redirects results/ to tmp_path so we don't pollute the real tree.
    """
    skip_if_no_env("vernier")
    from bench.harness.orchestrate import RunSpec
    from bench.harness.orchestrate import run as run_spec
    from bench.workloads import resolve

    workload = resolve("smoke", REPO_ROOT)

    fake_bench_root = tmp_path / "bench"
    fake_bench_root.mkdir()
    (fake_bench_root / "envs").symlink_to(BENCH_ROOT / "envs")

    spec = RunSpec(
        bench_root=fake_bench_root,
        repo_root=REPO_ROOT,
        impl="vernier",
        workload_id=workload.workload_id,
        iou_type="bbox",
        gt_path=workload.gt_path,
        dt_path=workload.dt_path,
        mode="dev",
        run_seed=0,
    )
    out_json = run_spec(spec)
    assert out_json.exists()
    assert out_json.is_relative_to(fake_bench_root / "results")

    payload = json.loads(out_json.read_text())
    assert payload["schema_version"] == 1
    assert payload["impl"] == "vernier"
    assert payload["mode"] == "dev"
    assert payload["reps_count"] == 1
    assert len(payload["reps"]) == 1
    assert payload["reps"][0]["rep"] == 0
    assert payload["reps"][0]["ru_maxrss_bytes"] > 0
