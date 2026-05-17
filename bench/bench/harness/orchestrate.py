"""Subprocess fan-out and result assembly (ADR-0017 §"Runner contract"
+ ADR-0033 paradigm-segmented path).

``run_cell`` is the cross-impl driver: it builds a schedule that
interleaves every ``(impl, rep)`` pair across the cell's impl list,
spawns each runner subprocess in randomized order per rep (deterministic
given ``run_seed``), assembles a per-impl ``BenchResult``, then runs the
parity comparator (per-paradigm cross-impl tolerance taxonomy in
``parity.py``; runtime ADR-0002 ``ParityMode`` is two-valued).
Release mode adds a governor
pre-flight and an IQR-relative-to-median gate on the ``total`` stage
(ADR-0017 §"Run modes").

Every runner subprocess is invoked as
``uv run --directory <bench_root>/envs/<env_name> python -m bench.runners.<impl>_runner ...``
so the runner sees its own paradigm-flavored package and nothing else.
The orchestrator never imports vernier or any baseline.

Per ADR-0033 §"Comparator registry", non-instance paradigms route the
cross-impl parity check through the comparator registry rather than
the legacy :func:`compare_cell` (which stays detection-only). The
spawn path is also paradigm-aware: panoptic cells use a
five-path argspec (``--gt-png-dir`` etc.) plus a two-artifact result
bundle (``snapshot.json`` + ``per_class.npy``). Detection paths are
byte-identical to v1.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from coco_val_cache import file_sha256

from bench import HARNESS_VERSION
from bench.harness import machine
from bench.harness.matrix import runner_module, uv_run_argv, uv_run_env
from bench.harness.migrations.v1_to_v2 import TENSOR_KEY
from bench.harness.parity import (
    CellParityReport,
    ConfusionMatrix,
    PanopticSnapshot,
    SemanticSnapshot,
    compare_cell,
    compare_lvis_cell,
    get_comparator,
    write_report,
)
from bench.harness.schema import (
    Aggregation,
    BenchResult,
    IouType,
    IqrGateResult,
    Metric,
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
from bench.runners._protocol import PANOPTIC_ARTIFACT_KEYS, SEMANTIC_ARTIFACT_KEYS

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
    # ``tensor_path`` is the legacy detection slot — populated for
    # detection runners (the canonical ``"tensor"`` artifact). Non-
    # instance paradigms produce multi-artifact bundles and populate
    # ``artifact_paths`` (a per-rep map of artifact-key → Path)
    # instead. Both shapes coexist; the assembler picks based on
    # paradigm.
    tensor_path: Path | None = None
    artifact_paths: dict[str, Path] = field(default_factory=dict)


def result_dir(
    *,
    results_root: Path,
    git_sha: str,
    machine_fp: str,
    workload_id: str,
    iou_type: Metric,
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

    The ``iou_type`` parameter is typed ``Metric`` (a superset of
    ``IouType``) so non-instance paradigms can pass their paradigm-
    specific metric ("pq", "miou", ...) directly.
    """
    return results_root / git_sha / machine_fp / paradigm / workload_id / iou_type


def _spawn_subprocess(
    *,
    bench_root: Path,
    impl: str,
    cmd: list[str],
) -> tuple[int, object, int]:
    """Run a runner subprocess and return ``(status, rusage, parent_wall_ns)``.

    Lifted out of :func:`_spawn_one_rep` so the panoptic spawn path
    can share the wait4 + parent-clock pattern without duplicating
    the pickle-glue.
    """
    parent_start = time.perf_counter_ns()
    proc = subprocess.Popen(cmd, env=uv_run_env(bench_root, impl))
    _pid, status, rusage = os.wait4(proc.pid, 0)
    parent_wall_ns = time.perf_counter_ns() - parent_start
    return status, rusage, parent_wall_ns


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
    num_threads: int | None = None,
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
    # ADR-0047 — append the optional threading axis when the cell pins
    # a thread count. Non-vernier runners accept the flag (it's wired
    # into the shared argspec) but ignore the value; vernier_runner
    # forwards it to ``Evaluator.evaluate``.
    if num_threads is not None:
        cmd.extend(["--num-threads", str(num_threads)])
    status, rusage, parent_wall_ns = _spawn_subprocess(bench_root=bench_root, impl=impl, cmd=cmd)
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
        ru_maxrss_bytes=int(rusage.ru_maxrss) * 1024,  # type: ignore[attr-defined]
        parent_wall_ns=parent_wall_ns,
    )
    return _SpawnResult(rep=rep_result, runner_out=runner_out, tensor_path=rep_npy)


def _validate_artifacts(
    *,
    impl: str,
    spawned: list[_SpawnResult],
    expected_keys: set[str],
) -> dict[str, str]:
    """Cross-rep artifact-sha256 invariant check (paradigm-agnostic).

    For each key in ``expected_keys``, every rep must produce the same
    sha256 — that's what proves the runner is deterministic and lets
    rep 0 be promoted as canonical. Returns the canonical (rep-0)
    sha256 dict so the caller can re-stamp it in the assembled result.

    Detection callers pass ``expected_keys={"tensor"}``. B-stream
    callers (when their own assemblers land) pass their own key set —
    e.g. panoptic ``{"snapshot", "per_class"}``, streaming
    ``{"summary", "rss_curve"}``. The function deliberately doesn't
    care which keys are which paradigm; it just checks every named
    key matches across reps.

    Idempotent on the empty key set (returns the empty dict). Raises
    ``RuntimeError`` on the first key whose sha disagrees across reps,
    naming the disagreeing rep so the diagnostic points at the
    offending subprocess.
    """
    canonical: dict[str, str] = {}
    for key in sorted(expected_keys):
        canonical_sha = spawned[0].runner_out.artifact_sha256[key]
        for s in spawned[1:]:
            other = s.runner_out.artifact_sha256[key]
            if other != canonical_sha:
                raise RuntimeError(
                    f"per-rep artifact disagreement for impl {impl!r}, "
                    f"key {key!r}: rep 0 sha {canonical_sha[:12]} differs "
                    f"from rep {s.rep.rep} sha {other[:12]}"
                )
        canonical[key] = canonical_sha
    return canonical


def _spawn_one_rep_panoptic(
    *,
    bench_root: Path,
    impl: str,
    workload_id: str,
    gt_png_dir: Path,
    gt_json: Path,
    dt_png_dir: Path,
    dt_json: Path,
    categories_json: Path,
    rep_index: int,
    intermediate_dir: Path,
    warmup: bool,
    num_threads: int | None = None,
) -> _SpawnResult:
    """Panoptic spawn path. Different argspec from
    :func:`_spawn_one_rep` (the four-path GT/DT family + categories
    JSON instead of single-path GT/DT) and a two-artifact result
    bundle (``snapshot.json`` + ``per_class.npy``).
    """
    rep_json = intermediate_dir / f"{impl}-rep{rep_index}.json"
    rep_snapshot = intermediate_dir / f"{impl}-rep{rep_index}-snapshot.json"
    rep_per_class = intermediate_dir / f"{impl}-rep{rep_index}-per_class.npy"
    cmd = uv_run_argv(
        bench_root,
        impl,
        "-m",
        runner_module(impl),
        "--gt-png-dir",
        str(gt_png_dir),
        "--gt-json",
        str(gt_json),
        "--dt-png-dir",
        str(dt_png_dir),
        "--dt-json",
        str(dt_json),
        "--categories-json",
        str(categories_json),
        "--workload-id",
        workload_id,
        "--paradigm",
        "panoptic",
        "--output",
        str(rep_json),
        "--snapshot-output",
        str(rep_snapshot),
        "--per-class-output",
        str(rep_per_class),
    )
    if num_threads is not None:
        cmd.extend(["--num-threads", str(num_threads)])
    status, rusage, parent_wall_ns = _spawn_subprocess(bench_root=bench_root, impl=impl, cmd=cmd)
    if status != 0:
        raise RuntimeError(f"runner {impl} exited with status {status}; cmd={cmd}")
    for required in (rep_json, rep_snapshot, rep_per_class):
        if not required.exists():
            raise RuntimeError(
                f"panoptic runner {impl} succeeded but did not produce {required.name}"
            )
    runner_out = RunnerRepOutput.model_validate_json(rep_json.read_bytes())

    rep_result = RepResult(
        rep=rep_index,
        warmup=warmup,
        stages=runner_out.stages,
        summary_stats=runner_out.summary_stats,
        ru_maxrss_bytes=int(rusage.ru_maxrss) * 1024,  # type: ignore[attr-defined]
        parent_wall_ns=parent_wall_ns,
    )
    return _SpawnResult(
        rep=rep_result,
        runner_out=runner_out,
        artifact_paths={"snapshot": rep_snapshot, "per_class": rep_per_class},
    )


def _spawn_one_rep_semantic(
    *,
    bench_root: Path,
    impl: str,
    workload_id: str,
    gt_label_map_dir: Path,
    dt_label_map_dir: Path,
    n_classes: int,
    ignore_label: int | None,
    rep_index: int,
    intermediate_dir: Path,
    warmup: bool,
    num_threads: int | None = None,
) -> _SpawnResult:
    """Semantic spawn path. Different argspec from
    :func:`_spawn_one_rep` (label-map dirs + n_classes + ignore_label
    instead of single-path GT/DT JSONs) and a two-artifact result
    bundle (``snapshot.json`` + ``per_class.npy``), mirroring panoptic.
    """
    rep_json = intermediate_dir / f"{impl}-rep{rep_index}.json"
    rep_snapshot = intermediate_dir / f"{impl}-rep{rep_index}-snapshot.json"
    rep_per_class = intermediate_dir / f"{impl}-rep{rep_index}-per_class.npy"
    rep_confusion = intermediate_dir / f"{impl}-rep{rep_index}-confusion.npy"
    cmd = uv_run_argv(
        bench_root,
        impl,
        "-m",
        runner_module(impl),
        "--gt-label-map-dir",
        str(gt_label_map_dir),
        "--dt-label-map-dir",
        str(dt_label_map_dir),
        "--n-classes",
        str(n_classes),
        # Wire ignore_label as -1 sentinel for "none"; the runner
        # decodes back to None.
        "--ignore-label",
        str(ignore_label if ignore_label is not None else -1),
        "--workload-id",
        workload_id,
        "--paradigm",
        "semantic",
        "--output",
        str(rep_json),
        "--snapshot-output",
        str(rep_snapshot),
        "--per-class-output",
        str(rep_per_class),
        "--confusion-output",
        str(rep_confusion),
    )
    if num_threads is not None:
        cmd.extend(["--num-threads", str(num_threads)])
    status, rusage, parent_wall_ns = _spawn_subprocess(bench_root=bench_root, impl=impl, cmd=cmd)
    if status != 0:
        raise RuntimeError(f"runner {impl} exited with status {status}; cmd={cmd}")
    for required in (rep_json, rep_snapshot, rep_per_class, rep_confusion):
        if not required.exists():
            raise RuntimeError(
                f"semantic runner {impl} succeeded but did not produce {required.name}"
            )
    runner_out = RunnerRepOutput.model_validate_json(rep_json.read_bytes())

    rep_result = RepResult(
        rep=rep_index,
        warmup=warmup,
        stages=runner_out.stages,
        summary_stats=runner_out.summary_stats,
        ru_maxrss_bytes=int(rusage.ru_maxrss) * 1024,  # type: ignore[attr-defined]
        parent_wall_ns=parent_wall_ns,
    )
    return _SpawnResult(
        rep=rep_result,
        runner_out=runner_out,
        artifact_paths={
            "snapshot": rep_snapshot,
            "per_class": rep_per_class,
            "confusion": rep_confusion,
        },
    )


def _aggregate_reps(
    *,
    rep_results: list[RepResult],
    mode: Mode,
    iqr_threshold: float,
) -> tuple[Aggregation | None, IqrGateResult | None]:
    """Across-rep aggregation; shared by detection and panoptic paths."""
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
    return aggregation, gate_result


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
    """Detection-shaped assembler: validate per-rep tensor bit-equality,
    promote rep 0, aggregate, write JSON.

    Panoptic cells route through :func:`_assemble_impl_result_panoptic`
    (two-artifact bundle); other paradigms add their own assemblers as
    they land.
    """
    _ = _validate_artifacts(impl=impl, spawned=spawned, expected_keys={TENSOR_KEY})

    canonical = spawned[0]
    if canonical.tensor_path is None:
        raise RuntimeError(
            f"detection assembler for impl {impl!r} requires tensor_path; "
            f"runner produced none — check the runner's write_outputs call."
        )
    tensor_dst = out_dir / f"{impl}.npy"
    shutil.copyfile(canonical.tensor_path, tensor_dst)
    tensor_sha256 = file_sha256(tensor_dst)
    if tensor_sha256 != canonical.runner_out.artifact_sha256[TENSOR_KEY]:
        raise RuntimeError("tensor sha256 mismatch between runner output and orchestrator copy")

    rep_results = [s.rep for s in spawned]
    aggregation, gate_result = _aggregate_reps(
        rep_results=rep_results, mode=mode, iqr_threshold=iqr_threshold
    )

    result = BenchResult(
        paradigm=paradigm,
        impl=canonical.runner_out.impl,
        impl_version=canonical.runner_out.impl_version,
        iou_type=iou_type,
        workload_id=workload_id,
        git_sha=git_sha,
        machine_fingerprint=machine_fp,
        cpu_model=machine.collect_inputs().cpu_model,
        cpu_arch=machine.cpu_arch(),
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


def _assemble_impl_result_panoptic(
    *,
    impl: str,
    out_dir: Path,
    spawned: list[_SpawnResult],
    workload_id: str,
    git_sha: str,
    machine_fp: str,
    mode: Mode,
    run_seed: int,
    reps_count: int,
    warmup_discarded: int,
    iqr_threshold: float,
) -> tuple[Path, PanopticSnapshot, dict[str, str], IqrGateResult | None]:
    """Panoptic assembler. Validates per-rep snapshot + per-class sha
    bit-equality (per the ADR-0033 strict-mode contract), promotes rep
    0's artifacts to the canonical ``<impl>.json`` / ``<impl>_per_class.npy``
    pair, aggregates rep timings, writes the ``BenchResult``.

    Returns ``(out_json, canonical_snapshot, sha_by_key, gate_result)``.
    """
    canonical_shas = _validate_artifacts(
        impl=impl, spawned=spawned, expected_keys=set(PANOPTIC_ARTIFACT_KEYS)
    )
    canonical = spawned[0]
    snapshot_src = canonical.artifact_paths.get("snapshot")
    per_class_src = canonical.artifact_paths.get("per_class")
    if snapshot_src is None or per_class_src is None:
        raise RuntimeError(
            f"panoptic spawn for impl {impl!r} missing artifact paths; "
            f"got: {sorted(canonical.artifact_paths)}"
        )

    snapshot_dst = out_dir / f"{impl}.json"
    per_class_dst = out_dir / f"{impl}_per_class.npy"
    shutil.copyfile(snapshot_src, snapshot_dst.with_suffix(".snapshot.json"))
    shutil.copyfile(per_class_src, per_class_dst)

    snapshot_sha = file_sha256(snapshot_dst.with_suffix(".snapshot.json"))
    per_class_sha = file_sha256(per_class_dst)
    if snapshot_sha != canonical_shas["snapshot"]:
        raise RuntimeError("panoptic snapshot sha mismatch between runner and orchestrator copy")
    if per_class_sha != canonical_shas["per_class"]:
        raise RuntimeError("panoptic per_class sha mismatch between runner and orchestrator copy")

    canonical_snapshot = PanopticSnapshot.model_validate_json(
        snapshot_dst.with_suffix(".snapshot.json").read_bytes()
    )

    rep_results = [s.rep for s in spawned]
    aggregation, gate_result = _aggregate_reps(
        rep_results=rep_results, mode=mode, iqr_threshold=iqr_threshold
    )

    # The result JSON carries the per-impl summary record; the
    # snapshot.json sits next to it as a separate artifact (the
    # comparator reads back from that path). Filename layout under
    # the cell dir: ``<impl>.json`` (BenchResult), ``<impl>.snapshot.json``
    # (PanopticSnapshot), ``<impl>_per_class.npy`` (uint64 N-by-3 table).
    result = BenchResult(
        paradigm="panoptic",
        impl=canonical.runner_out.impl,
        impl_version=canonical.runner_out.impl_version,
        # The schema's IouType slot still validates "bbox"/etc.; we
        # carry it for column compatibility — the panoptic metric is
        # encoded in the path segment via ``result_dir(... iou_type="pq",
        # paradigm="panoptic")``.
        iou_type=canonical.runner_out.iou_type,
        workload_id=workload_id,
        git_sha=git_sha,
        machine_fingerprint=machine_fp,
        cpu_model=machine.collect_inputs().cpu_model,
        cpu_arch=machine.cpu_arch(),
        harness_version=HARNESS_VERSION,
        mode=mode,
        run_seed=run_seed,
        reps_count=reps_count,
        warmup_discarded=warmup_discarded,
        reps=rep_results,
        aggregation=aggregation,
        artifact_paths={
            "snapshot": f"{impl}.snapshot.json",
            "per_class": per_class_dst.name,
        },
        artifact_sha256={
            "snapshot": snapshot_sha,
            "per_class": per_class_sha,
        },
        warnings=list(canonical.runner_out.warnings),
    )
    snapshot_dst.write_text(result.model_dump_json(indent=2))
    return (
        snapshot_dst,
        canonical_snapshot,
        {"snapshot": snapshot_sha, "per_class": per_class_sha},
        gate_result,
    )


def _assemble_impl_result_semantic(
    *,
    impl: str,
    out_dir: Path,
    spawned: list[_SpawnResult],
    workload_id: str,
    git_sha: str,
    machine_fp: str,
    mode: Mode,
    run_seed: int,
    reps_count: int,
    warmup_discarded: int,
    iqr_threshold: float,
) -> tuple[Path, SemanticSnapshot, ConfusionMatrix, dict[str, str], IqrGateResult | None]:
    """Semantic assembler. Mirrors :func:`_assemble_impl_result_panoptic`:
    validates per-rep snapshot + per-class sha bit-equality, promotes
    rep 0's artifacts to the canonical ``<impl>.json`` /
    ``<impl>_per_class.npy`` / ``<impl>_confusion.npy`` triple,
    aggregates rep timings, writes the ``BenchResult``, and loads the
    confusion-marginals into a :class:`ConfusionMatrix` artifact for
    the strict-tier parity comparator.

    Returns ``(out_json, canonical_snapshot, confusion_matrix,
    sha_by_key, gate_result)``.
    """
    canonical_shas = _validate_artifacts(
        impl=impl, spawned=spawned, expected_keys=set(SEMANTIC_ARTIFACT_KEYS)
    )
    canonical = spawned[0]
    snapshot_src = canonical.artifact_paths.get("snapshot")
    per_class_src = canonical.artifact_paths.get("per_class")
    confusion_src = canonical.artifact_paths.get("confusion")
    if snapshot_src is None or per_class_src is None or confusion_src is None:
        raise RuntimeError(
            f"semantic spawn for impl {impl!r} missing artifact paths; "
            f"got: {sorted(canonical.artifact_paths)}"
        )

    snapshot_dst = out_dir / f"{impl}.json"
    per_class_dst = out_dir / f"{impl}_per_class.npy"
    confusion_dst = out_dir / f"{impl}_confusion.npy"
    shutil.copyfile(snapshot_src, snapshot_dst.with_suffix(".snapshot.json"))
    shutil.copyfile(per_class_src, per_class_dst)
    shutil.copyfile(confusion_src, confusion_dst)

    snapshot_sha = file_sha256(snapshot_dst.with_suffix(".snapshot.json"))
    per_class_sha = file_sha256(per_class_dst)
    confusion_sha = file_sha256(confusion_dst)
    if snapshot_sha != canonical_shas["snapshot"]:
        raise RuntimeError("semantic snapshot sha mismatch between runner and orchestrator copy")
    if per_class_sha != canonical_shas["per_class"]:
        raise RuntimeError("semantic per_class sha mismatch between runner and orchestrator copy")
    if confusion_sha != canonical_shas["confusion"]:
        raise RuntimeError("semantic confusion sha mismatch between runner and orchestrator copy")

    canonical_snapshot = SemanticSnapshot.model_validate_json(
        snapshot_dst.with_suffix(".snapshot.json").read_bytes()
    )
    confusion_counts = np.load(confusion_dst, allow_pickle=False)
    confusion_artifact = ConfusionMatrix(
        n_classes=int(canonical_snapshot.n_classes),
        counts=confusion_counts,
        counts_sha256=confusion_sha,
    )

    rep_results = [s.rep for s in spawned]
    aggregation, gate_result = _aggregate_reps(
        rep_results=rep_results, mode=mode, iqr_threshold=iqr_threshold
    )

    result = BenchResult(
        paradigm="semantic",
        impl=canonical.runner_out.impl,
        impl_version=canonical.runner_out.impl_version,
        iou_type=canonical.runner_out.iou_type,
        workload_id=workload_id,
        git_sha=git_sha,
        machine_fingerprint=machine_fp,
        cpu_model=machine.collect_inputs().cpu_model,
        cpu_arch=machine.cpu_arch(),
        harness_version=HARNESS_VERSION,
        mode=mode,
        run_seed=run_seed,
        reps_count=reps_count,
        warmup_discarded=warmup_discarded,
        reps=rep_results,
        aggregation=aggregation,
        artifact_paths={
            "snapshot": f"{impl}.snapshot.json",
            "per_class": per_class_dst.name,
            "confusion": confusion_dst.name,
        },
        artifact_sha256={
            "snapshot": snapshot_sha,
            "per_class": per_class_sha,
            "confusion": confusion_sha,
        },
        warnings=list(canonical.runner_out.warnings),
    )
    snapshot_dst.write_text(result.model_dump_json(indent=2))
    return (
        snapshot_dst,
        canonical_snapshot,
        confusion_artifact,
        {
            "snapshot": snapshot_sha,
            "per_class": per_class_sha,
            "confusion": confusion_sha,
        },
        gate_result,
    )


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

    Detection cells populate ``gt_path`` / ``dt_path``; panoptic cells
    populate the four-path GT/DT family + categories JSON instead.
    """

    bench_root: Path
    repo_root: Path
    impls: list[str]
    workload_id: str
    iou_type: Metric  # widened from IouType for panoptic ("pq") + future paradigms.
    mode: Mode
    run_seed: int
    paradigm: Paradigm = "instance"
    gt_path: Path | None = None
    dt_path: Path | None = None
    # Panoptic four-path family. Populated only when paradigm="panoptic".
    gt_png_dir: Path | None = None
    gt_json: Path | None = None
    dt_png_dir: Path | None = None
    dt_json: Path | None = None
    categories_json: Path | None = None
    # Semantic family. Populated only when paradigm="semantic".
    gt_label_map_dir: Path | None = None
    dt_label_map_dir: Path | None = None
    n_classes: int | None = None
    ignore_label: int | None = None
    # ADR-0047 threading axis. ``None`` (the default) preserves the
    # pre-ADR-0047 single-threaded behavior at every callsite; an
    # explicit int forwards through the runner to
    # :meth:`vernier.instance.Evaluator.evaluate`'s ``num_threads``.
    # Non-vernier impls ignore the value (they have no rayon pool).
    num_threads: int | None = None


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
        if cell.paradigm == "panoptic":
            if (
                cell.gt_png_dir is None
                or cell.gt_json is None
                or cell.dt_png_dir is None
                or cell.dt_json is None
                or cell.categories_json is None
            ):
                raise ValueError(
                    "panoptic cell requires gt_png_dir/gt_json/dt_png_dir/dt_json/categories_json"
                )
            spawned[impl_name][rep_idx] = _spawn_one_rep_panoptic(
                bench_root=cell.bench_root,
                impl=impl_name,
                workload_id=cell.workload_id,
                gt_png_dir=cell.gt_png_dir,
                gt_json=cell.gt_json,
                dt_png_dir=cell.dt_png_dir,
                dt_json=cell.dt_json,
                categories_json=cell.categories_json,
                rep_index=rep_idx,
                intermediate_dir=intermediate_dir,
                warmup=warmup,
                num_threads=cell.num_threads,
            )
        elif cell.paradigm == "semantic":
            if (
                cell.gt_label_map_dir is None
                or cell.dt_label_map_dir is None
                or cell.n_classes is None
            ):
                raise ValueError(
                    "semantic cell requires gt_label_map_dir/dt_label_map_dir/n_classes"
                )
            spawned[impl_name][rep_idx] = _spawn_one_rep_semantic(
                bench_root=cell.bench_root,
                impl=impl_name,
                workload_id=cell.workload_id,
                gt_label_map_dir=cell.gt_label_map_dir,
                dt_label_map_dir=cell.dt_label_map_dir,
                n_classes=cell.n_classes,
                ignore_label=cell.ignore_label,
                rep_index=rep_idx,
                intermediate_dir=intermediate_dir,
                warmup=warmup,
                num_threads=cell.num_threads,
            )
        else:
            if cell.gt_path is None or cell.dt_path is None:
                raise ValueError(f"{cell.paradigm} cell requires gt_path and dt_path")
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
                num_threads=cell.num_threads,
            )

    impl_jsons: dict[str, Path] = {}
    impl_tensors: dict[str, np.ndarray] = {}
    impl_sha256: dict[str, str] = {}
    impl_panoptic_snapshots: dict[str, PanopticSnapshot] = {}
    impl_semantic_snapshots: dict[str, SemanticSnapshot] = {}
    impl_semantic_confusion: dict[str, ConfusionMatrix] = {}
    iqr_outcomes: dict[str, IqrGateResult] = {}
    for impl_name in cell.impls:
        results = [r for r in spawned[impl_name] if r is not None]
        if cell.paradigm == "panoptic":
            out_json, snapshot, _shas, iqr_outcome = _assemble_impl_result_panoptic(
                impl=impl_name,
                out_dir=out_dir,
                spawned=results,
                workload_id=cell.workload_id,
                git_sha=git_sha,
                machine_fp=machine_fp,
                mode=cell.mode,
                run_seed=cell.run_seed,
                reps_count=n_measurement,
                warmup_discarded=n_warmup,
                iqr_threshold=iqr_threshold,
            )
            impl_jsons[impl_name] = out_json
            impl_panoptic_snapshots[impl_name] = snapshot
        elif cell.paradigm == "semantic":
            (
                out_json,
                sem_snapshot,
                sem_confusion,
                _shas,
                iqr_outcome,
            ) = _assemble_impl_result_semantic(
                impl=impl_name,
                out_dir=out_dir,
                spawned=results,
                workload_id=cell.workload_id,
                git_sha=git_sha,
                machine_fp=machine_fp,
                mode=cell.mode,
                run_seed=cell.run_seed,
                reps_count=n_measurement,
                warmup_discarded=n_warmup,
                iqr_threshold=iqr_threshold,
            )
            impl_jsons[impl_name] = out_json
            impl_semantic_snapshots[impl_name] = sem_snapshot
            impl_semantic_confusion[impl_name] = sem_confusion
        else:
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
        if cell.paradigm == "instance":
            parity_report = compare_cell(
                workload_id=cell.workload_id,
                iou_type=cell.iou_type,
                impl_tensors=impl_tensors,
                impl_sha256=impl_sha256,
            )
        elif cell.paradigm == "lvis":
            # LVIS reuses the single-tensor parity surface but pairs
            # vernier_lvis vs lvis-api (not vernier vs pycocotools).
            # Same dispatch shape as instance, separate tier table.
            parity_report = compare_lvis_cell(
                workload_id=cell.workload_id,
                iou_type=cell.iou_type,
                impl_tensors=impl_tensors,
                impl_sha256=impl_sha256,
            )
        elif cell.paradigm == "panoptic":
            parity_report = get_comparator("panoptic").compare(
                workload_id=cell.workload_id,
                iou_type=cell.iou_type,
                impl_outputs=dict(impl_panoptic_snapshots),
            )
        elif cell.paradigm == "semantic":
            # The semantic strict tier compares ``ConfusionMatrix``
            # artifacts (the (4, n_classes) marginals — see
            # `_SemanticComparator` and ADR-0036). The CLI also
            # short-circuits parity when only one impl is registered.
            parity_report = get_comparator("semantic").compare(
                workload_id=cell.workload_id,
                iou_type=cell.iou_type,
                impl_outputs=dict(impl_semantic_confusion),
            )
        if parity_report is not None and not parity_report.passed:
            divergence_report_path = write_report(parity_report, out_dir)

    return CellRun(
        impl_jsons=impl_jsons,
        parity=parity_report,
        divergence_report_path=divergence_report_path,
        iqr_outcomes=iqr_outcomes,
    )
