"""LVIS workload registry — name parsing, IoU support, env-overrides.

The cache-dependent reads (``ensure_gt`` → download) are gated by
``VERNIER_BENCH_DOWNLOAD_TESTS=1``; the default path-resolution tests
short-circuit the cache by setting ``VERNIER_LVIS_GT_PATH`` /
``VERNIER_LVIS_DT_SEGM_PATH`` so they run in the offline CI matrix.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bench.harness.paths import REPO_ROOT
from bench.workloads import lvis_v1, resolve

_DOWNLOAD_GATE = "VERNIER_BENCH_DOWNLOAD_TESTS"


def test_lvis_perfect_resolves_via_env_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``VERNIER_LVIS_GT_PATH`` and ``VERNIER_LVIS_DT_SEGM_PATH`` are
    the parity-cache convention; the bench resolver must honor them so
    a single populated cache serves both ``just test-parity-lvis-val``
    and ``vernier-bench run``."""
    gt = tmp_path / "lvis_v1_val.json"
    gt.write_bytes(b"{}")
    dt = tmp_path / "perfect_dt.json"
    dt.write_bytes(b"[]")
    monkeypatch.setenv("VERNIER_LVIS_GT_PATH", str(gt))
    monkeypatch.setenv("VERNIER_LVIS_DT_PATH", str(dt))

    w = resolve(lvis_v1.PERFECT_WORKLOAD_ID, REPO_ROOT)
    assert w.workload_id == "lvis_v1_val_perfect"
    assert w.paradigm == "lvis"
    assert w.gt_path == gt
    assert w.dt_path == dt
    # bbox-only at the vernier side until ``evaluate_segm_grid_with_dataset``
    # lands; the matrix entry pins this too.
    assert w.supported_iou_types == frozenset({"bbox"})


def test_lvis_jittered_resolves_via_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``lvis_v1_val_jittered_seed<N>`` runs the COCO jitter generator
    against the LVIS GT bytes. The cache filename is keyed off
    ``lvis_v1_val`` so a same-seed COCO jittered DT does not clobber it."""
    monkeypatch.setenv("VERNIER_BENCH_CACHE", str(tmp_path / "bench"))
    gt = tmp_path / "lvis_v1_val.json"
    # Minimal LVIS-ish GT: one image, one annotation. The jitter
    # generator only reads ``annotations`` / ``categories`` /
    # ``images`` keys; LVIS-specific ``not_exhaustive_category_ids``
    # are ignored at this layer (the runner consumes those
    # downstream).
    gt.write_text(
        '{"images":[{"id":1,"width":32,"height":32,"file_name":"1.jpg"}],'
        '"categories":[{"id":1,"name":"obj"}],'
        '"annotations":[{"id":1,"image_id":1,"category_id":1,'
        '"bbox":[4,4,8,8],"area":64,"iscrowd":0,'
        '"segmentation":[[4,4,12,4,12,12,4,12]]}]}'
    )
    monkeypatch.setenv("VERNIER_LVIS_GT_PATH", str(gt))

    w = resolve("lvis_v1_val_jittered_seed7", REPO_ROOT)
    assert w.workload_id == "lvis_v1_val_jittered_seed7"
    assert w.paradigm == "lvis"
    assert w.gt_path == gt
    # The cache filename uniquely identifies the LVIS workload — a
    # subsequent COCO jittered seed=7 must not collide.
    assert "lvis_v1_val_jittered_seed7" in w.dt_path.name
    assert w.dt_path.exists()
    assert w.supported_iou_types == frozenset({"bbox", "segm"})


def test_lvis_jittered_unknown_seed_format_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Names like ``lvis_v1_val_jittered_seedfoo`` are not workload ids;
    the resolver's catch-all ``unknown workload`` arm must fire so the
    CLI surfaces a helpful message instead of a stray match."""
    with pytest.raises(ValueError, match="unknown workload"):
        resolve("lvis_v1_val_jittered_seedfoo", REPO_ROOT)


@pytest.mark.skipif(
    os.environ.get(_DOWNLOAD_GATE) != "1",
    reason=f"set {_DOWNLOAD_GATE}=1 to exercise the LVIS cache fetch path.",
)
def test_lvis_perfect_populates_cache_when_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end cache-dependent path: no env override → falls through
    to ``lvis_val_cache.ensure_gt`` and synthesizes the perfect-DT.
    Gated by ``VERNIER_BENCH_DOWNLOAD_TESTS=1`` so offline CI skips it."""
    monkeypatch.setenv("VERNIER_LVIS_CACHE", str(tmp_path))
    monkeypatch.delenv("VERNIER_LVIS_GT_PATH", raising=False)
    monkeypatch.delenv("VERNIER_LVIS_DT_SEGM_PATH", raising=False)
    w = resolve(lvis_v1.PERFECT_WORKLOAD_ID, REPO_ROOT)
    assert w.gt_path.exists()
    assert w.dt_path.exists()
    assert w.workload_id == "lvis_v1_val_perfect"
