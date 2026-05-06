"""``vernier-bench scale`` end-to-end on a synthetic result tree."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from bench.cli import main
from bench.harness.orchestrate import result_dir
from bench.harness.schema import (
    Aggregation,
    BenchResult,
    MemoryAggregation,
    RepResult,
    StageAggregation,
    StageTimings,
)

_FP = "fp9876543210"


def _write_cell(
    root: Path,
    *,
    workload_id: str,
    impl: str,
    median_ns: int,
    now: datetime,
) -> Path:
    git_sha = "deadbeefcafe0001"
    out_dir = result_dir(
        results_root=root,
        git_sha=git_sha,
        machine_fp=_FP,
        workload_id=workload_id,
        iou_type="bbox",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    result = BenchResult(
        paradigm="instance",
        impl=impl,
        impl_version="0.0.1",
        iou_type="bbox",
        workload_id=workload_id,
        git_sha=git_sha,
        machine_fingerprint=_FP,
        harness_version="0.0.0",
        mode="release",
        run_seed=0,
        reps_count=1,
        warmup_discarded=0,
        reps=[
            RepResult(
                rep=0,
                warmup=False,
                stages={"total": StageTimings(wall_ns=median_ns)},
                summary_stats={},
                ru_maxrss_bytes=150 * 1024 * 1024,
                parent_wall_ns=median_ns,
            )
        ],
        aggregation=Aggregation(
            stages={
                "total": StageAggregation(
                    median_ns=median_ns,
                    iqr_ns=median_ns // 100,
                    min_ns=median_ns,
                    max_ns=median_ns,
                )
            },
            memory=MemoryAggregation(
                median_bytes=150 * 1024 * 1024,
                min_bytes=150 * 1024 * 1024,
                max_bytes=150 * 1024 * 1024,
            ),
        ),
        artifact_paths={"tensor": f"{impl}.npy"},
        artifact_sha256={"tensor": "0" * 64},
    )
    json_path = out_dir / f"{impl}.json"
    json_path.write_text(result.model_dump_json())
    ts = now.timestamp()
    os.utime(json_path, (ts, ts))
    return json_path


@pytest.fixture
def scaling_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Three-point n_images ladder for vernier and pycocotools."""
    monkeypatch.setattr("bench.cli.BENCH_ROOT", tmp_path)
    results_root = tmp_path / "results"
    now = datetime(2026, 5, 6, tzinfo=timezone.utc)
    for n in (1000, 10000, 50000):
        wid = f"synthetic_n{n}_c80_g10_d30_s0"
        _write_cell(results_root, workload_id=wid, impl="vernier", median_ns=n * 100, now=now)
        _write_cell(results_root, workload_id=wid, impl="pycocotools", median_ns=n * 1500, now=now)
    return tmp_path


def test_scale_writes_md_and_svg(scaling_tree: Path) -> None:
    out_dir = scaling_tree / "scaling-out"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "scale",
            "--vary",
            "n_images",
            "--fix",
            "n_categories=80,gt_per_image=10,dt_per_image=30,seed=0",
            "--iou",
            "bbox",
            "--output-dir",
            str(out_dir),
        ],
    )
    if result.exit_code != 0:
        raise AssertionError(f"scale failed: {result.output}\n{result.exception}")

    md = (out_dir / "scaling.md").read_text()
    svg = (out_dir / "scaling.svg").read_text()
    assert "vernier" in md
    assert "pycocotools" in md
    assert "10k" in md
    assert "50k" in md
    assert svg.startswith("<svg")
    assert svg.count("<polyline") == 2


def test_scale_rejects_vary_in_fix(scaling_tree: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "scale",
            "--vary",
            "n_images",
            "--fix",
            "n_images=1000",
            "--iou",
            "bbox",
        ],
    )
    assert result.exit_code != 0
    assert "cannot also appear in --fix" in result.output


def test_scale_rejects_unknown_fix_axis(scaling_tree: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "scale",
            "--vary",
            "n_images",
            "--fix",
            "bogus=1",
            "--iou",
            "bbox",
        ],
    )
    assert result.exit_code != 0
    assert "unknown axis" in result.output


def test_scale_errors_when_no_cells_match(scaling_tree: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "scale",
            "--vary",
            "n_images",
            "--fix",
            "gt_per_image=999",
            "--iou",
            "bbox",
        ],
    )
    assert result.exit_code != 0
    assert "no synthetic cells matched" in result.output
