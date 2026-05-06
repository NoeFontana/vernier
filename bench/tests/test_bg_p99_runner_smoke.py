"""End-to-end smoke for ``vernier_bg_p99_runner``.

Runs the saturation runner at a short ``--duration-s`` (1.5s per
queue depth) against the perfect-match smoke fixture; asserts the
``latency_cdf`` artifact is emitted with all five percentiles
populated for each of the three queue depths.

Skipped when ``bench/envs/vernier/.venv`` is missing.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from bench.harness.matrix import env_dir, runner_module, uv_run_argv, uv_run_env
from bench.harness.paths import BENCH_ROOT, REPO_ROOT
from bench.workloads import resolve
from tests.conftest import skip_if_no_env


def test_vernier_bg_p99_runner_smoke(tmp_path: Path) -> None:
    skip_if_no_env("vernier_bg")
    workload = resolve("smoke", REPO_ROOT)
    output = tmp_path / "vernier_bg.json"

    cmd = uv_run_argv(
        BENCH_ROOT,
        "vernier_bg",
        "-m",
        runner_module("vernier_bg"),
        "--gt",
        str(workload.gt_path),  # type: ignore[union-attr] — smoke is InstanceWorkload
        "--dt",
        str(workload.dt_path),  # type: ignore[union-attr]
        "--iou-type",
        "bbox",
        "--workload-id",
        workload.workload_id,
        "--output",
        str(output),
        "--duration-s",
        "1.5",
    )
    proc = subprocess.run(
        cmd,
        env=uv_run_env(BENCH_ROOT, "vernier_bg"),
        check=False,
        capture_output=True,
    )
    assert proc.returncode == 0, (
        f"runner exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout.decode(errors='replace')}\n"
        f"stderr:\n{proc.stderr.decode(errors='replace')}\n"
    )

    payload = json.loads(output.read_text())
    assert payload["paradigm"] == "streaming"
    assert payload["impl"] == "vernier_bg"

    cdf_name = payload["artifact_paths"]["latency_cdf"]
    cdf = json.loads((output.parent / cdf_name).read_text())
    # All three queue depths populated; each carries the five-percentile
    # block plus a non-zero sample count (saturation actually fed).
    assert cdf["queue_capacities"] == [1, 8, 64]
    assert cdf["regression_threshold"] == 1.20
    for depth in ("1", "8", "64"):
        section = cdf["per_capacity"][depth]
        assert section["n_samples"] > 0, f"queue depth {depth} drained zero samples"
        for label in ("p50", "p90", "p99", "p999", "max"):
            assert label in section["percentiles_us"]
            assert section["percentiles_us"][label] >= 0.0


_ = env_dir
