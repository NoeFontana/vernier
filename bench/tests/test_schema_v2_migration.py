"""Schema v1 → v2 read-side compat shim (ADR-0033).

Asserts that:
- v1-shaped result JSON parses through ``upgrade()`` into a v2
  ``BenchResult``;
- v2 round-trips are idempotent (``upgrade()`` is a no-op);
- a v3+ claim is rejected (forward-compat refuses to silently
  drop unknown fields).
"""

from __future__ import annotations

import pytest

from bench.harness.migrations import LATEST, upgrade
from bench.harness.migrations.v1_to_v2 import TENSOR_KEY, migrate
from bench.harness.schema import BenchResult


def _v1_bench_result_dict() -> dict[str, object]:
    """A minimum-viable v1 ``BenchResult`` JSON dict — no paradigm
    field, single-tensor pair via ``tensor_path`` / ``tensor_sha256``.
    Mirrors what every detection cell wrote pre-ADR-0033."""
    return {
        "schema_version": 1,
        "impl": "vernier",
        "impl_version": "0.0.1",
        "iou_type": "bbox",
        "workload_id": "smoke_perfect_match",
        "git_sha": "abcdef123456",
        "machine_fingerprint": "dev-unfp-m1",
        "harness_version": "0.0.0",
        "mode": "dev",
        "run_seed": 0,
        "reps_count": 1,
        "warmup_discarded": 0,
        "reps": [
            {
                "rep": 0,
                "warmup": False,
                "stages": {"total": {"wall_ns": 1_000_000, "notes": []}},
                "summary_stats": {},
                "ru_maxrss_bytes": 100 * 1024 * 1024,
                "parent_wall_ns": 1_000_000,
            }
        ],
        "aggregation": None,
        "tensor_path": "vernier.npy",
        "tensor_sha256": "0" * 64,
        "warnings": [],
    }


def test_v1_dict_upgrades_to_v2_via_compat_shim() -> None:
    raw_v1 = _v1_bench_result_dict()
    upgraded = upgrade(raw_v1)
    assert upgraded["schema_version"] == 2
    assert upgraded["paradigm"] == "instance"
    # The v1 single-tensor pair lifts under the canonical "tensor" slot.
    assert upgraded["artifact_paths"] == {TENSOR_KEY: "vernier.npy"}
    assert upgraded["artifact_sha256"] == {TENSOR_KEY: "0" * 64}
    # The flat v1 fields no longer round-trip.
    assert "tensor_path" not in upgraded
    assert "tensor_sha256" not in upgraded


def test_upgraded_v1_dict_validates_as_v2_bench_result() -> None:
    """End-to-end: v1 dict → upgrade() → BenchResult.model_validate."""
    raw_v1 = _v1_bench_result_dict()
    result = BenchResult.model_validate(upgrade(raw_v1))
    assert result.schema_version == 2
    assert result.paradigm == "instance"
    assert result.artifact_paths[TENSOR_KEY] == "vernier.npy"
    assert result.artifact_sha256[TENSOR_KEY] == "0" * 64


def test_v2_dict_roundtrips_unchanged() -> None:
    """Idempotency: upgrade() of a v2 dict is a no-op."""
    v2 = {
        "schema_version": 2,
        "paradigm": "instance",
        "impl": "vernier",
        "artifact_paths": {"tensor": "vernier.npy"},
        "artifact_sha256": {"tensor": "0" * 64},
        # Other fields irrelevant to the migration; the shim never touches them.
    }
    out = upgrade(v2)
    assert out["schema_version"] == 2
    assert out == v2


def test_v3_claim_is_rejected() -> None:
    with pytest.raises(ValueError, match="schema_version 3"):
        upgrade({"schema_version": 3})


def test_missing_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing or non-integer"):
        upgrade({"impl": "vernier"})


def test_migrate_directly_is_idempotent_on_v2() -> None:
    """``migrate`` on a v2 dict returns a copy unchanged — it's the
    walker that decides whether to call it, but the function itself
    must be idempotent for the v2 case."""
    v2 = {
        "schema_version": 2,
        "paradigm": "instance",
        "artifact_paths": {"tensor": "vernier.npy"},
        "artifact_sha256": {"tensor": "0" * 64},
    }
    out = migrate(v2)
    assert out == v2
    # And the function returns a new dict so callers can't accidentally
    # mutate the input.
    assert out is not v2


def test_latest_constant_matches_current_version() -> None:
    """Sanity check: the migration framework's LATEST and the schema's
    pinned version must stay in lockstep."""
    assert LATEST == 2
