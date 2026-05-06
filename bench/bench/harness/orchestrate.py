"""Subprocess fan-out and result assembly (ADR-0017 §"Runner contract"
+ ADR-0033 paradigm-segmented path).

``run_cell`` is the cross-impl driver: it builds a schedule that
interleaves every ``(impl, rep)`` pair across the cell's impl list,
spawns each runner subprocess in randomized order per rep (deterministic
given ``run_seed``), assembles a per-impl ``BenchResult``, then runs the
three-tier parity check from ADR-0002. Release mode adds a governor
pre-flight and an IQR-relative-to-median gate on the ``total`` stage
(ADR-0017 §"Run modes").

Every runner subprocess is invoked as
``uv run --directory <bench_root>/envs/<env_name> python -m bench.runners.<impl>_runner ...``
so the runner sees its own paradigm-flavored package and nothing else.
The orchestrator never imports vernier or any baseline.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from coco_val_cache import file_sha256

from bench import HARNESS_VERSION
from bench.harness import machine
from bench.harness.matrix import runner_module, uv_run_argv, uv_run_env
from bench.harness.migrations.v1_to_v2 import TENSOR_KEY
from bench.harness.parity import CellParityReport, compare_cell, write_report
from bench.harness.schema import (
    Aggregation,
    BenchResult,
    IouType,
    IqrGateResult,
    Mode,
    Paradigm,
    RepResult,
    RunnerRepOutput,
)
from bench.harness.stats import (
    DEFAULT_IQR_RELATIVE_THRESHOLD,
    aggregate_memory,
    aggregate_reps,
    iqr_gate,
)

# (n_warmup, n_measurement) per ADR-0017 §"Run modes".
MODE_REPS: dict[Mode, tuple[int, int]] = {
    "dev": (0, 1),
    "release": (2, 10),
    "profile": (0, 1),
}


def mode_defaults(mode: Mode) -> tuple[int, int]:
    """Return ``(n_warmup, n_measurement)`` for ``mode``."""
    return MODE_REPS[mode]


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
    paradigm: Paradigm = "instance"


@dataclass(frozen=True)
class _SpawnResult:
    rep: RepResult
    runner_out: RunnerRepOutput
    tensor_path: Path


def result_dir(
    *,
    results_root: Path,
    git_sha: str,
    machine_fp: str,
    workload_id: str,
    iou_type: IouType,
    paradigm: Paradigm = "instance",
) -> Path:
    """Canonical cell directory:
    ``<results_root>/<sha>/<fp>/<paradigm>/<workload>/<metric>/`` (ADR-0033 v2).

    ``<metric>`` is the ex-``<iou>`` slot — bbox/segm/keypoints/
    boundary for instance, ``pq`` for panoptic, ``miou`` for semantic,
    throughput / p99 / rss for streaming. ``results_root`` is the
    parent of the per-sha buckets; in normal operation that's
    ``bench/results/``. Reused by reports + tests so the layout has
    one source of truth.

    ``paradigm`` defaults to ``"instance"`` so detection callers — and
    the existing tests that pre-date ADR-0033 — keep their ergonomic
    short signature; B-streams pass their paradigm explicitly.
    """
    return results_root / git_sha / machine_fp / paradigm / workload_id / iou_type


def _spawn_one_rep(
    *,
    bench_root: Path,
    impl: str,
    workload_id: str,
    iou_type: IouType,
    gt_path: Path,
    dt_path: Path,
    rep_index: int,
    intermediate_dir: Path,
    warmup: bool,
) -> _SpawnResult:
    rep_json = intermediate_dir / f"{impl}-rep{rep_index}.json"
    rep_npy = intermediate_dir / f"{impl}-rep{rep_index}.npy"
    cmd = uv_run_argv(
        bench_root,
        impl,
        "-m",
        runner_module(impl),
        "--gt",
        str(gt_path),
        "--dt",
        str(dt_path),
        "--iou-type",
        iou_type,
        "--workload-id",
        workload_id,
        "--output",
        str(rep_json),
        "--tensor-output",
        str(rep_npy),
    )
    parent_start = time.perf_counter_ns()
    proc = subprocess.Popen(cmd, env=uv_run_env(bench_root, impl))
    _pid, status, rusage = os.wait4(proc.pid, 0)
    parent_wall_ns = time.perf_counter_ns() - parent_start
    if status != 0:
        raise RuntimeError(f"runner {impl} exited with status {status}; cmd={cmd}")
    if not rep_json.exists() or not rep_npy.exists():
        raise RuntimeError(f"runner {impl} succeeded but did not produce expected outputs")
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


def _assemble_impl_result(
    *,
    impl: str,
    out_dir: Path,
    spawned: list[_SpawnResult],
    iou_type: IouType,
    workload_id: str,
    git_sha: str,
    machine_fp: str,
    mode: Mode,
    run_seed: int,
    reps_count: int,
    warmup_discarded: int,
    iqr_threshold: float,
    paradigm: Paradigm = "instance",
) -> tuple[Path, np.ndarray, str, IqrGateResult | None]:
    """Validate per-rep tensor bit-equality, promote rep 0, aggregate, write JSON.

    Detection runners produce a single tensor under the canonical
    ``"tensor"`` artifact slot; this assembler stays detection-shaped
    until B-streams land their own assemblers (panoptic snapshots,
    streaming summary+rss bundles).
    """
    canonical_sha = spawned[0].runner_out.artifact_sha256[TENSOR_KEY]
    for s in spawned[1:]:
        if s.runner_out.artifact_sha256[TENSOR_KEY] != canonical_sha:
            raise RuntimeError(
                f"per-rep tensor disagreement for impl {impl!r}: rep 0 sha "
                f"{canonical_sha[:12]} differs from rep {s.rep.rep} sha "
                f"{s.runner_out.artifact_sha256[TENSOR_KEY][:12]}"
            )

    canonical = spawned[0]
    tensor_dst = out_dir / f"{impl}.npy"
    shutil.copyfile(canonical.tensor_path, tensor_dst)
    tensor_sha256 = file_sha256(tensor_dst)
    if tensor_sha256 != canonical.runner_out.artifact_sha256[TENSOR_KEY]:
        raise RuntimeError("tensor sha256 mismatch between runner output and orchestrator copy")

    rep_results = [s.rep for s in spawned]
    aggregation: Aggregation | None = None
    gate_result: IqrGateResult | None = None
    if any(not r.warmup for r in rep_results):
        stages = aggregate_reps(rep_results)
        if mode == "release" and "total" in stages:
            gate_result = iqr_gate(stages, threshold=iqr_threshold)
        aggregation = Aggregation(
            stages=stages,
            iqr_gate=gate_result,
            memory=aggregate_memory(rep_results),
        )

    result = BenchResult(
        paradigm=paradigm,
        impl=canonical.runner_out.impl,
        impl_version=canonical.runner_out.impl_version,
        iou_type=iou_type,
        workload_id=workload_id,
        git_sha=git_sha,
        machine_fingerprint=machine_fp,
        harness_version=HARNESS_VERSION,
        mode=mode,
        run_seed=run_seed,
        reps_count=reps_count,
        warmup_discarded=warmup_discarded,
        reps=rep_results,
        aggregation=aggregation,
        artifact_paths={TENSOR_KEY: f"{impl}.npy"},
        artifact_sha256={TENSOR_KEY: tensor_sha256},
        warnings=list(canonical.runner_out.warnings),
    )

    out_json = out_dir / f"{impl}.json"
    out_json.write_text(result.model_dump_json(indent=2))
    return out_json, np.load(canonical.tensor_path), tensor_sha256, gate_result


def run(spec: RunSpec) -> Path:
    """Execute one impl's reps consecutively and persist the result.

    Single-impl convenience entry; ``run_cell`` is the cross-impl driver.
    Schedule is trivial here — ``warmup_discarded`` warmup reps then
    ``reps_count`` measurement reps, all consecutive.
    """
    git_sha = machine.git_sha(spec.repo_root)
    machine_fp = machine.fingerprint()

    out_dir = result_dir(
        results_root=spec.bench_root / "results",
        git_sha=git_sha,
        machine_fp=machine_fp,
        workload_id=spec.workload_id,
        iou_type=spec.iou_type,
        paradigm=spec.paradigm,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir = out_dir / ".intermediate"
    intermediate_dir.mkdir(exist_ok=True)

    spawned: list[_SpawnResult] = []
    for rep_index in range(spec.warmup_discarded + spec.reps_count):
        warmup = rep_index < spec.warmup_discarded
        spawned.append(
            _spawn_one_rep(
                bench_root=spec.bench_root,
                impl=spec.impl,
                workload_id=spec.workload_id,
                iou_type=spec.iou_type,
                gt_path=spec.gt_path,
                dt_path=spec.dt_path,
                rep_index=rep_index,
                intermediate_dir=intermediate_dir,
                warmup=warmup,
            )
        )

    out_json, _tensor, _sha, _gate = _assemble_impl_result(
        impl=spec.impl,
        out_dir=out_dir,
        spawned=spawned,
        iou_type=spec.iou_type,
        workload_id=spec.workload_id,
        git_sha=git_sha,
        machine_fp=machine_fp,
        mode=spec.mode,
        run_seed=spec.run_seed,
        reps_count=spec.reps_count,
        warmup_discarded=spec.warmup_discarded,
        iqr_threshold=DEFAULT_IQR_RELATIVE_THRESHOLD,
        paradigm=spec.paradigm,
    )
    return out_json


@dataclass(frozen=True)
class CellSpec:
    """One ``(paradigm, workload, metric)`` cell with the impls to fan
    out across.

    Caller is responsible for filtering ``impls`` to entries that
    support ``iou_type`` for the cell's paradigm (the CLI does this via
    the matrix module).
    """

    bench_root: Path
    repo_root: Path
    impls: list[str]
    workload_id: str
    iou_type: IouType
    gt_path: Path
    dt_path: Path
    mode: Mode
    run_seed: int
    paradigm: Paradigm = "instance"


@dataclass(frozen=True)
class CellRun:
    impl_jsons: dict[str, Path]
    parity: CellParityReport | None
    divergence_report_path: Path | None
    iqr_outcomes: dict[str, IqrGateResult]


def _build_schedule(
    impls: list[str], n_warmup: int, n_measurement: int, run_seed: int
) -> list[tuple[str, int, bool]]:
    """``[(impl, rep_index, is_warmup)]`` in execution order.

    Warmup reps come first; within each rep the impl order is a fresh
    permutation drawn from ``np.random.default_rng(run_seed)``. Two
    invocations with the same seed produce the same schedule.
    """
    if not impls:
        return []
    rng = np.random.default_rng(run_seed)
    schedule: list[tuple[str, int, bool]] = []
    for rep_idx in range(n_warmup + n_measurement):
        order = rng.permutation(len(impls))
        warmup = rep_idx < n_warmup
        schedule.extend((impls[int(slot)], rep_idx, warmup) for slot in order)
    return schedule


def run_cell(
    cell: CellSpec,
    *,
    parity: bool = True,
    iqr_threshold: float = DEFAULT_IQR_RELATIVE_THRESHOLD,
) -> CellRun:
    """Fan out across ``cell.impls`` per ``cell.mode``'s schedule, then
    run the cross-impl parity check.

    ``parity=False`` skips the comparison entirely (the ``--no-parity``
    escape hatch). When parity runs and fails, a ``divergence_report.json``
    lands next to the impl JSON files in the cell's result dir. Release
    mode also pre-flights the CPU governor and applies the IQR gate.
    """
    if cell.mode == "release":
        machine.ensure_performance_governor()

    n_warmup, n_measurement = mode_defaults(cell.mode)
    schedule = _build_schedule(cell.impls, n_warmup, n_measurement, cell.run_seed)
    total_reps = n_warmup + n_measurement

    git_sha = machine.git_sha(cell.repo_root)
    machine_fp = machine.fingerprint()

    out_dir = result_dir(
        results_root=cell.bench_root / "results",
        git_sha=git_sha,
        machine_fp=machine_fp,
        workload_id=cell.workload_id,
        iou_type=cell.iou_type,
        paradigm=cell.paradigm,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir = out_dir / ".intermediate"
    intermediate_dir.mkdir(exist_ok=True)

    # spawned[impl][rep_idx] = _SpawnResult
    spawned: dict[str, list[_SpawnResult | None]] = {
        impl: [None] * total_reps for impl in cell.impls
    }
    for impl_name, rep_idx, warmup in schedule:
        spawned[impl_name][rep_idx] = _spawn_one_rep(
            bench_root=cell.bench_root,
            impl=impl_name,
            workload_id=cell.workload_id,
            iou_type=cell.iou_type,
            gt_path=cell.gt_path,
            dt_path=cell.dt_path,
            rep_index=rep_idx,
            intermediate_dir=intermediate_dir,
            warmup=warmup,
        )

    impl_jsons: dict[str, Path] = {}
    impl_tensors: dict[str, np.ndarray] = {}
    impl_sha256: dict[str, str] = {}
    iqr_outcomes: dict[str, IqrGateResult] = {}
    for impl_name in cell.impls:
        results = [r for r in spawned[impl_name] if r is not None]
        out_json, tensor, tensor_sha, iqr_outcome = _assemble_impl_result(
            impl=impl_name,
            out_dir=out_dir,
            spawned=results,
            iou_type=cell.iou_type,
            workload_id=cell.workload_id,
            git_sha=git_sha,
            machine_fp=machine_fp,
            mode=cell.mode,
            run_seed=cell.run_seed,
            reps_count=n_measurement,
            warmup_discarded=n_warmup,
            iqr_threshold=iqr_threshold,
            paradigm=cell.paradigm,
        )
        impl_jsons[impl_name] = out_json
        impl_tensors[impl_name] = tensor
        impl_sha256[impl_name] = tensor_sha
        if iqr_outcome is not None:
            iqr_outcomes[impl_name] = iqr_outcome

    parity_report: CellParityReport | None = None
    divergence_report_path: Path | None = None
    if parity:
        parity_report = compare_cell(
            workload_id=cell.workload_id,
            iou_type=cell.iou_type,
            impl_tensors=impl_tensors,
            impl_sha256=impl_sha256,
        )
        if not parity_report.passed:
            divergence_report_path = write_report(parity_report, out_dir)

    return CellRun(
        impl_jsons=impl_jsons,
        parity=parity_report,
        divergence_report_path=divergence_report_path,
        iqr_outcomes=iqr_outcomes,
    )
