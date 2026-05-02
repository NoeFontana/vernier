"""Same seed and pinned params must yield byte-identical jittered DT
JSON. The workload identity is keyed only on (seed, JITTER_PARAMS_VERSION),
so any seed-independent drift (numpy version, dict-iteration order)
would break a downstream cache lookup."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.workloads import jittered_predictions, synthetic


@pytest.fixture
def tiny_synthetic_gt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VERNIER_BENCH_CACHE", str(tmp_path))
    gt, _ = synthetic.make_workload(
        n_images=4, n_categories=3, dt_per_image=2, gt_per_image=2, seed=0
    )
    return gt


def test_same_seed_yields_byte_identical_dt(tiny_synthetic_gt: Path) -> None:
    first = jittered_predictions.dt_path(gt_path=tiny_synthetic_gt, seed=42)
    first_bytes = first.read_bytes()
    first.unlink()

    second = jittered_predictions.dt_path(gt_path=tiny_synthetic_gt, seed=42)
    assert second.read_bytes() == first_bytes


def test_different_seeds_yield_different_dts(tiny_synthetic_gt: Path) -> None:
    a = jittered_predictions.dt_path(gt_path=tiny_synthetic_gt, seed=1)
    b = jittered_predictions.dt_path(gt_path=tiny_synthetic_gt, seed=2)
    assert a != b
    assert a.read_bytes() != b.read_bytes()


def test_cache_hit_skips_regeneration(tiny_synthetic_gt: Path) -> None:
    out = jittered_predictions.dt_path(gt_path=tiny_synthetic_gt, seed=7)
    mtime = out.stat().st_mtime_ns
    again = jittered_predictions.dt_path(gt_path=tiny_synthetic_gt, seed=7)
    assert again == out
    assert again.stat().st_mtime_ns == mtime
