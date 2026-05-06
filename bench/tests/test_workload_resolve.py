"""Workload string parsing — the CLI's only contact with the registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.harness.paths import REPO_ROOT
from bench.workloads import coco_val2017, resolve


def test_smoke_resolves_to_local_fixture() -> None:
    w = resolve("smoke", REPO_ROOT)
    assert w.workload_id == "smoke_perfect_match_segm"
    assert w.gt_path.exists()
    assert w.dt_path.exists()
    assert w.supported_iou_types == frozenset({"bbox", "segm", "boundary"})


def test_synthetic_minimal_args_uses_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VERNIER_BENCH_CACHE", str(tmp_path))
    w = resolve("synthetic:n_images=3,seed=7", REPO_ROOT)
    assert w.workload_id == "synthetic_n3_c80_g10_d30_s7"
    assert w.gt_path.exists()
    assert w.dt_path.exists()
    assert w.supported_iou_types == frozenset({"bbox"})


def test_synthetic_overrides_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERNIER_BENCH_CACHE", str(tmp_path))
    w = resolve(
        "synthetic:n_images=2,seed=0,n_categories=4,dt_per_image=1,gt_per_image=1",
        REPO_ROOT,
    )
    assert w.workload_id == "synthetic_n2_c4_g1_d1_s0"


def test_synthetic_rejects_missing_required() -> None:
    with pytest.raises(ValueError, match="missing required"):
        resolve("synthetic:n_categories=10", REPO_ROOT)


def test_synthetic_rejects_unknown_param() -> None:
    with pytest.raises(ValueError, match="unknown param"):
        resolve("synthetic:n_images=10,seed=0,bogus=1", REPO_ROOT)


def test_synthetic_rejects_non_int_value() -> None:
    with pytest.raises(ValueError, match="must be int"):
        resolve("synthetic:n_images=10,seed=cat", REPO_ROOT)


def test_synthetic_rejects_malformed_token() -> None:
    with pytest.raises(ValueError, match="not k=v"):
        resolve("synthetic:n_images=10,seed", REPO_ROOT)


def test_unknown_workload_lists_known_ids() -> None:
    with pytest.raises(ValueError, match="unknown workload"):
        resolve("does-not-exist", REPO_ROOT)


def test_coco_val2017_perfect_segm_resolves_when_inputs_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gt = tmp_path / "instances_val2017.json"
    gt.write_bytes(b"{}")
    dt = tmp_path / "perfect_dt_segm.json"
    dt.write_bytes(b"[]")
    # The GT loader sha256-verifies the real cached file; short-circuit it.
    monkeypatch.setattr(coco_val2017, "gt_path", lambda: gt)
    monkeypatch.setenv("VERNIER_COCO_DT_SEGM_PATH", str(dt))

    w = resolve("coco_val2017_perfect_segm", REPO_ROOT)
    assert w.workload_id == "coco_val2017_perfect_segm"
    assert w.gt_path == gt
    assert w.dt_path == dt
    assert w.supported_iou_types == frozenset({"segm", "boundary"})


def test_coco_val2017_jittered_serves_bbox_segm_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v2 jitter materializes segm RLE alongside bbox so the workload
    fans out to bbox + segm + boundary cells. Sha256-verified GT load
    is short-circuited the same way the perfect-segm test does it."""
    monkeypatch.setenv("VERNIER_BENCH_CACHE", str(tmp_path))
    gt = tmp_path / "instances_val2017.json"
    gt.write_text(
        '{"images":[{"id":1,"width":32,"height":32,"file_name":"1.jpg"}],'
        '"categories":[{"id":1,"name":"cat"}],'
        '"annotations":[{"id":1,"image_id":1,"category_id":1,'
        '"bbox":[4,4,8,8],"area":64,"iscrowd":0,'
        '"segmentation":[[4,4,12,4,12,12,4,12]]}]}'
    )
    monkeypatch.setattr(coco_val2017, "gt_path", lambda: gt)

    w = resolve("coco_val2017_jittered_seed42", REPO_ROOT)
    assert w.workload_id == "coco_val2017_jittered_seed42"
    assert w.gt_path == gt
    assert w.dt_path.exists()
    assert w.supported_iou_types == frozenset({"bbox", "segm", "boundary"})
