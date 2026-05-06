"""``iscrowd_fraction`` extension to the synthetic generator."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from bench.harness.paths import REPO_ROOT
from bench.workloads import resolve, synthetic


def test_iscrowd_zero_is_legacy_workload_id() -> None:
    wid = synthetic.workload_id(
        n_images=10, n_categories=4, dt_per_image=5, gt_per_image=3, seed=42
    )
    legacy = synthetic.workload_id(
        n_images=10,
        n_categories=4,
        dt_per_image=5,
        gt_per_image=3,
        seed=42,
        iscrowd_fraction=0.0,
    )
    assert wid == legacy == "synthetic_n10_c4_g3_d5_s42"


def test_iscrowd_half_appends_x50_suffix() -> None:
    wid = synthetic.workload_id(
        n_images=10,
        n_categories=4,
        dt_per_image=5,
        gt_per_image=20,
        seed=42,
        iscrowd_fraction=0.5,
    )
    assert wid == "synthetic_n10_c4_g20_d5_x50_s42"


def test_iscrowd_zero_keeps_all_anns_iscrowd_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VERNIER_BENCH_CACHE", str(tmp_path))
    gt_path, _ = synthetic.make_workload(
        n_images=2, n_categories=3, dt_per_image=1, gt_per_image=4, seed=0
    )
    gt = json.loads(gt_path.read_text())
    assert all(a["iscrowd"] == 0 for a in gt["annotations"])


def test_iscrowd_half_flips_deterministic_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VERNIER_BENCH_CACHE", str(tmp_path))
    gt_path, _ = synthetic.make_workload(
        n_images=3,
        n_categories=2,
        dt_per_image=1,
        gt_per_image=20,
        seed=0,
        iscrowd_fraction=0.5,
    )
    gt = json.loads(gt_path.read_text())
    by_image: dict[int, list[int]] = {}
    for a in gt["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a["iscrowd"])
    for image_id, flags in by_image.items():
        assert len(flags) == 20, image_id
        assert sum(flags) == 10, image_id
        assert flags[:10] == [1] * 10
        assert flags[10:] == [0] * 10


def test_iscrowd_workload_id_carries_into_cache_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VERNIER_BENCH_CACHE", str(tmp_path))
    gt_path, dt_path = synthetic.make_workload(
        n_images=1,
        n_categories=2,
        dt_per_image=1,
        gt_per_image=4,
        seed=7,
        iscrowd_fraction=0.25,
    )
    assert "_x25_s7" in gt_path.name
    assert "_x25_s7" in dt_path.name


def test_parser_accepts_iscrowd_fraction_via_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VERNIER_BENCH_CACHE", str(tmp_path))
    w = resolve(
        "synthetic:n_images=2,seed=0,gt_per_image=4,iscrowd_fraction=0.5",
        REPO_ROOT,
    )
    assert re.search(r"_x50_s0$", w.workload_id)


def test_parser_rejects_iscrowd_above_one() -> None:
    with pytest.raises(ValueError, match=r"in \[0.0, 1.0\]"):
        resolve("synthetic:n_images=2,seed=0,iscrowd_fraction=1.5", REPO_ROOT)


def test_parser_rejects_iscrowd_negative() -> None:
    with pytest.raises(ValueError, match=r"in \[0.0, 1.0\]"):
        resolve("synthetic:n_images=2,seed=0,iscrowd_fraction=-0.1", REPO_ROOT)


def test_parser_rejects_iscrowd_non_numeric() -> None:
    with pytest.raises(ValueError, match="must be float"):
        resolve("synthetic:n_images=2,seed=0,iscrowd_fraction=cat", REPO_ROOT)
