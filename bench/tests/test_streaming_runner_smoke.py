"""End-to-end smoke for ``vernier_streaming_runner`` — invoke it as a
subprocess against the smoke fixture and validate the streaming-shaped
output bundle.

Skipped when ``bench/envs/vernier/.venv`` is missing (matches the
detection runner smoke test's gating). Mirrors
``tests/test_runner_vernier.py``'s structure.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from bench.harness.matrix import env_dir, runner_module, uv_run_argv, uv_run_env
from bench.harness.paths import BENCH_ROOT, REPO_ROOT
from bench.workloads import resolve
from tests.conftest import skip_if_no_env


def test_vernier_streaming_runner_smoke_throughput(tmp_path: Path) -> None:
    """``vernier_streaming`` runner against the perfect-match smoke
    fixture in throughput mode. Asserts the multi-artifact emission
    (stats.json + rss_curve.json) and a populated ``Summary.stats``."""
    skip_if_no_env("vernier_streaming")
    workload = resolve("smoke", REPO_ROOT)
    output = tmp_path / "vernier_streaming.json"

    cmd = uv_run_argv(
        BENCH_ROOT,
        "vernier_streaming",
        "-m",
        runner_module("vernier_streaming"),
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
        "--mode-flag",
        "throughput",
    )
    proc = subprocess.run(
        cmd,
        env=uv_run_env(BENCH_ROOT, "vernier_streaming"),
        check=False,
        capture_output=True,
    )
    assert proc.returncode == 0, (
        f"runner exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout.decode(errors='replace')}\n"
        f"stderr:\n{proc.stderr.decode(errors='replace')}\n"
    )

    assert output.exists()
    payload = json.loads(output.read_text())
    assert payload["schema_version"] == 2
    assert payload["paradigm"] == "streaming"
    assert payload["impl"] == "vernier_streaming"

    # Stages: load + init_streaming + update_per_image + finalize + total.
    for stage in ("load", "init_streaming", "update_per_image", "finalize", "total"):
        assert stage in payload["stages"], f"missing stage: {stage}"
        assert payload["stages"][stage]["wall_ns"] >= 0

    # Multi-artifact emission: stats + rss_curve, each with a path + sha.
    assert "summary" in payload["artifact_paths"]
    assert "rss_curve" in payload["artifact_paths"]
    assert payload["artifact_sha256"]["summary"]
    assert payload["artifact_sha256"]["rss_curve"]

    # The stats.json artifact lives next to the JSON output.
    stats_path = output.parent / payload["artifact_paths"]["summary"]
    rss_path = output.parent / payload["artifact_paths"]["rss_curve"]
    assert stats_path.exists()
    assert rss_path.exists()

    stats = json.loads(stats_path.read_text())
    # Smoke fixture is a single perfect match → AP=1.0 at stat_0.
    assert stats["summary_stats"]["stat_0"] == pytest.approx(1.0)

    rss = json.loads(rss_path.read_text())
    # ≥1 sample (the synchronous-at-enter one); psutil is in the bench env.
    assert len(rss["samples"]) >= 1


def test_vernier_streaming_summary_matches_batch(tmp_path: Path) -> None:
    """The streaming runner's ``Summary.stats`` must match what the
    detection (batch) runner produces on the same fixture — pins the
    parity claim end-to-end through the bench harness."""
    skip_if_no_env("vernier_streaming")
    skip_if_no_env("vernier")
    workload = resolve("smoke", REPO_ROOT)

    # Batch run.
    batch_json = tmp_path / "vernier.json"
    batch_npy = tmp_path / "vernier.npy"
    batch_cmd = uv_run_argv(
        BENCH_ROOT,
        "vernier",
        "-m",
        runner_module("vernier"),
        "--gt",
        str(workload.gt_path),  # type: ignore[union-attr]
        "--dt",
        str(workload.dt_path),  # type: ignore[union-attr]
        "--iou-type",
        "bbox",
        "--workload-id",
        workload.workload_id,
        "--output",
        str(batch_json),
        "--tensor-output",
        str(batch_npy),
    )
    proc = subprocess.run(
        batch_cmd, env=uv_run_env(BENCH_ROOT, "vernier"), check=False, capture_output=True
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    batch_stats = json.loads(batch_json.read_text())["summary_stats"]

    # Stream run.
    stream_json = tmp_path / "vernier_streaming.json"
    stream_cmd = uv_run_argv(
        BENCH_ROOT,
        "vernier_streaming",
        "-m",
        runner_module("vernier_streaming"),
        "--gt",
        str(workload.gt_path),  # type: ignore[union-attr]
        "--dt",
        str(workload.dt_path),  # type: ignore[union-attr]
        "--iou-type",
        "bbox",
        "--workload-id",
        workload.workload_id,
        "--output",
        str(stream_json),
        "--mode-flag",
        "throughput",
    )
    proc = subprocess.run(
        stream_cmd,
        env=uv_run_env(BENCH_ROOT, "vernier_streaming"),
        check=False,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    stream_payload = json.loads(stream_json.read_text())
    stats_path = stream_json.parent / stream_payload["artifact_paths"]["summary"]
    stream_stats = json.loads(stats_path.read_text())["summary_stats"]

    # Batch stats are named (AP, AP50, ...); stream stats are positional
    # (stat_0, stat_1, ...). Walk in lockstep to assert pointwise equality.
    batch_values = [batch_stats[k] for k in (
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_1",
        "AR_10",
        "AR_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    )]
    stream_values = [stream_stats[f"stat_{i}"] for i in range(len(batch_values))]
    for i, (b, s) in enumerate(zip(batch_values, stream_values, strict=True)):
        assert s == pytest.approx(b, rel=0, abs=1e-12), (
            f"stat[{i}] diverged: batch={b!r} stream={s!r}"
        )


# Avoid unused-import lint warnings for the helper that's only used in the
# subprocess-skip gate.
_ = env_dir
