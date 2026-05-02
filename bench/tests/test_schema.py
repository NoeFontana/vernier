"""Schema-v1 round-trips. Catches accidental field additions/removals
and the extra=forbid contract."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from bench.harness.schema import (
    Aggregation,
    BenchResult,
    BenchWarning,
    RepResult,
    RunnerRepOutput,
    StageAggregation,
    StageTimings,
)


def _stages() -> dict[str, StageTimings]:
    return {
        "load": StageTimings(wall_ns=1_000_000),
        "evaluate": StageTimings(wall_ns=2_000_000, notes=["smoke"]),
        "accumulate": StageTimings(wall_ns=500_000),
        "summarize": StageTimings(wall_ns=10_000),
        "total": StageTimings(wall_ns=3_510_000),
    }


def _summary_stats() -> dict[str, float]:
    return {
        "AP": 1.0,
        "AP50": 1.0,
        "AP75": 1.0,
        "AP_small": -1.0,
        "AP_medium": -1.0,
        "AP_large": 1.0,
        "AR_1": 1.0,
        "AR_10": 1.0,
        "AR_100": 1.0,
        "AR_small": -1.0,
        "AR_medium": -1.0,
        "AR_large": 1.0,
    }


def _runner_rep_output() -> RunnerRepOutput:
    return RunnerRepOutput(
        impl="vernier",
        impl_version="0.0.1",
        iou_type="bbox",
        workload_id="smoke_perfect_match",
        stages=_stages(),
        summary_stats=_summary_stats(),
        tensor_sha256="0" * 64,
    )


def _bench_result() -> BenchResult:
    return BenchResult(
        impl="vernier",
        impl_version="0.0.1",
        iou_type="bbox",
        workload_id="smoke_perfect_match",
        git_sha="abcdef123456",
        machine_fingerprint="dev-unfp-m1",
        harness_version="0.0.0",
        mode="dev",
        run_seed=0,
        reps_count=1,
        warmup_discarded=0,
        reps=[
            RepResult(
                rep=0,
                warmup=False,
                stages=_stages(),
                summary_stats=_summary_stats(),
                ru_maxrss_bytes=42 * 1024,
                parent_wall_ns=4_000_000,
            )
        ],
        aggregation=Aggregation(
            stages={
                "total": StageAggregation(
                    median_ns=3_510_000,
                    iqr_ns=0,
                    min_ns=3_510_000,
                    max_ns=3_510_000,
                ),
            }
        ),
        tensor_path="vernier.npy",
        tensor_sha256="0" * 64,
        warnings=[BenchWarning(code="smoke", message="just a test")],
    )


def test_runner_rep_output_round_trip() -> None:
    obj = _runner_rep_output()
    serialized = obj.model_dump(mode="json")
    rebuilt = RunnerRepOutput.model_validate(serialized)
    assert rebuilt == obj


def test_bench_result_round_trip() -> None:
    obj = _bench_result()
    serialized = obj.model_dump(mode="json")
    rebuilt = BenchResult.model_validate(serialized)
    assert rebuilt == obj


def test_extra_field_is_rejected() -> None:
    payload = _bench_result().model_dump(mode="json")
    payload["__extra__"] = "should be rejected"
    with pytest.raises(ValidationError):
        BenchResult.model_validate(payload)


def test_schema_version_pinned_to_one() -> None:
    obj = _bench_result()
    assert obj.schema_version == 1
    serialized = json.loads(obj.model_dump_json())
    assert serialized["schema_version"] == 1
