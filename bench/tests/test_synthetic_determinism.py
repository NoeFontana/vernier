"""Synthetic workloads are pure functions of their parameters."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.workloads import synthetic


def test_same_params_yield_byte_identical_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VERNIER_BENCH_CACHE", str(tmp_path))
    params = dict(n_images=5, n_categories=3, dt_per_image=2, gt_per_image=2, seed=0)

    gt1, dt1 = synthetic.make_workload(**params)
    gt1_bytes, dt1_bytes = gt1.read_bytes(), dt1.read_bytes()
    gt1.unlink()
    dt1.unlink()

    gt2, dt2 = synthetic.make_workload(**params)
    assert gt2.read_bytes() == gt1_bytes
    assert dt2.read_bytes() == dt1_bytes


def test_different_seeds_yield_different_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VERNIER_BENCH_CACHE", str(tmp_path))
    common = dict(n_images=5, n_categories=3, dt_per_image=2, gt_per_image=2)

    gt_a, dt_a = synthetic.make_workload(seed=1, **common)
    gt_b, dt_b = synthetic.make_workload(seed=2, **common)
    assert gt_a.read_bytes() != gt_b.read_bytes()
    assert dt_a.read_bytes() != dt_b.read_bytes()


def test_workload_id_encodes_every_param() -> None:
    wid = synthetic.workload_id(
        n_images=10, n_categories=4, dt_per_image=5, gt_per_image=3, seed=42
    )
    assert wid == "synthetic_n10_c4_g3_d5_s42"
