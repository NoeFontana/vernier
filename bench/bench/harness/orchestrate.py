"""Subprocess fan-out and result assembly (ADR-0017 §"Runner contract").

M1 scope: single subprocess, single rep, no warmup, no parity coupling.
M2 fans out across runners; M3 adds parity; M5 adds rep loop, randomized
schedule, IQR gate, and machine fingerprint.

Every runner subprocess is invoked as
``uv run --directory <bench_root>/envs/<impl> python -m bench.runners.<impl>_runner ...``
so the runner sees its own pycocotools-flavored package and nothing else.
The orchestrator never imports vernier or any baseline.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bench import HARNESS_VERSION
from bench.harness import machine
from bench.harness.matrix import runner_module, uv_run_argv, uv_run_env
from bench.harness.schema import (
    BenchResult,
    IouType,
    Mode,
    RepResult,
    RunnerRepOutput,
)
from bench.runners._protocol import file_sha256


@dataclass(frozen=True)
class RunSpec:
    bench_root: Path
    repo_root: Path
    impl: str
    workload_id: str
    iou_type: IouType
    gt_path: Path
    dt_path: Path
    mode: Mode
    run_seed: int
    reps_count: int = 1
    warmup_discarded: int = 0


@dataclass(frozen=True)
class _SpawnResult:
    rep: RepResult
    runner_out: RunnerRepOutput
    tensor_path: Path


def _result_dir(spec: RunSpec, git_sha: str, machine_fp: str) -> Path:
    return spec.bench_root / "results" / git_sha / machine_fp / spec.workload_id / spec.iou_type


def _spawn_one_rep(
    spec: RunSpec,
    rep_index: int,
    intermediate_dir: Path,
    *,
    warmup: Literal[False] = False,
) -> _SpawnResult:
    rep_json = intermediate_dir / f"{spec.impl}-rep{rep_index}.json"
    rep_npy = intermediate_dir / f"{spec.impl}-rep{rep_index}.npy"
    cmd = uv_run_argv(
        spec.bench_root,
        spec.impl,
        "-m",
        runner_module(spec.impl),
        "--gt",
        str(spec.gt_path),
        "--dt",
        str(spec.dt_path),
        "--iou-type",
        spec.iou_type,
        "--workload-id",
        spec.workload_id,
        "--output",
        str(rep_json),
        "--tensor-output",
        str(rep_npy),
    )
    parent_start = time.perf_counter_ns()
    proc = subprocess.Popen(cmd, env=uv_run_env(spec.bench_root, spec.impl))
    _pid, status, rusage = os.wait4(proc.pid, 0)
    parent_wall_ns = time.perf_counter_ns() - parent_start
    if status != 0:
        raise RuntimeError(f"runner {spec.impl} exited with status {status}; cmd={cmd}")
    if not rep_json.exists() or not rep_npy.exists():
        raise RuntimeError(f"runner {spec.impl} succeeded but did not produce expected outputs")
    runner_out = RunnerRepOutput.model_validate_json(rep_json.read_bytes())

    rep_result = RepResult(
        rep=rep_index,
        warmup=warmup,
        stages=runner_out.stages,
        summary_stats=runner_out.summary_stats,
        # Linux ``ru_maxrss`` is reported in kilobytes; convert at capture.
        ru_maxrss_bytes=int(rusage.ru_maxrss) * 1024,
        parent_wall_ns=parent_wall_ns,
    )
    return _SpawnResult(rep=rep_result, runner_out=runner_out, tensor_path=rep_npy)


def run(spec: RunSpec) -> Path:
    """Execute one (impl, workload, iou_type) cell and persist its result."""
    git_sha = machine.git_sha(spec.repo_root)
    machine_fp = machine.fingerprint()

    out_dir = _result_dir(spec, git_sha, machine_fp)
    out_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir = out_dir / ".intermediate"
    intermediate_dir.mkdir(exist_ok=True)

    spawned: list[_SpawnResult] = [
        _spawn_one_rep(spec, rep_index, intermediate_dir) for rep_index in range(spec.reps_count)
    ]
    canonical = spawned[0]

    # Tensor of record is rep 0. M3 will assert every rep's tensor is
    # bit-equal before this — promoting one is then unambiguous.
    tensor_dst = out_dir / f"{spec.impl}.npy"
    shutil.copyfile(canonical.tensor_path, tensor_dst)
    tensor_sha256 = file_sha256(tensor_dst)
    if tensor_sha256 != canonical.runner_out.tensor_sha256:
        raise RuntimeError("tensor sha256 mismatch between runner output and orchestrator copy")

    result = BenchResult(
        impl=canonical.runner_out.impl,
        impl_version=canonical.runner_out.impl_version,
        iou_type=spec.iou_type,
        workload_id=spec.workload_id,
        git_sha=git_sha,
        machine_fingerprint=machine_fp,
        harness_version=HARNESS_VERSION,
        mode=spec.mode,
        run_seed=spec.run_seed,
        reps_count=spec.reps_count,
        warmup_discarded=spec.warmup_discarded,
        reps=[s.rep for s in spawned],
        aggregation=None,
        tensor_path=f"{spec.impl}.npy",
        tensor_sha256=tensor_sha256,
        warnings=list(canonical.runner_out.warnings),
    )

    out_json = out_dir / f"{spec.impl}.json"
    out_json.write_text(result.model_dump_json(indent=2))
    return out_json
