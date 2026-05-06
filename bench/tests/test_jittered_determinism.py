"""Same seed and pinned params must yield byte-identical jittered DT
JSON, including the segm RLE introduced in v2. The workload identity
is keyed only on (seed, JITTER_PARAMS_VERSION), so any seed-independent
drift (numpy version, dict-iteration order, scipy/pycocotools-internal
nondeterminism) would break a downstream cache lookup."""

from __future__ import annotations

import json
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


@pytest.fixture
def tiny_segm_gt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Hand-rolled GT with one polygon-segm annotation per image.

    The synthetic generator deliberately omits segmentation (its job is
    bbox scaling curves); segm determinism needs a fixture that exercises
    the polygon → mask → jitter → RLE pipeline.
    """
    monkeypatch.setenv("VERNIER_BENCH_CACHE", str(tmp_path))
    h, w = 64, 64
    images = [{"id": i, "width": w, "height": h, "file_name": f"{i}.jpg"} for i in range(4)]
    categories = [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
    annotations = []
    for i in range(4):
        offset = i * 8
        x0, y0, sz = 8 + offset, 8 + offset, 16
        annotations.append(
            {
                "id": i + 1,
                "image_id": i,
                "category_id": 1 + (i % 2),
                "bbox": [x0, y0, sz, sz],
                "area": sz * sz,
                "iscrowd": 0,
                "segmentation": [
                    [
                        x0,
                        y0,
                        x0 + sz,
                        y0,
                        x0 + sz,
                        y0 + sz,
                        x0,
                        y0 + sz,
                    ]
                ],
            }
        )
    out = tmp_path / "gt_with_segm.json"
    out.write_text(
        json.dumps({"images": images, "categories": categories, "annotations": annotations})
    )
    return out


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


def test_segm_rle_byte_identical_across_runs(tiny_segm_gt: Path) -> None:
    first = jittered_predictions.dt_path(gt_path=tiny_segm_gt, seed=42)
    first_bytes = first.read_bytes()
    first.unlink()

    second = jittered_predictions.dt_path(gt_path=tiny_segm_gt, seed=42)
    assert second.read_bytes() == first_bytes

    detections = json.loads(first_bytes)
    # Every detection (TP + FP) must carry segm/area in v2 so segm and
    # boundary runners can consume the same DT stream.
    assert all("segmentation" in d for d in detections), detections
    assert all("area" in d for d in detections), detections
    rles = [d["segmentation"] for d in detections]
    assert all(isinstance(r, dict) and isinstance(r["counts"], str) for r in rles)


def test_segm_rle_diverges_across_seeds(tiny_segm_gt: Path) -> None:
    a_path = jittered_predictions.dt_path(gt_path=tiny_segm_gt, seed=1)
    b_path = jittered_predictions.dt_path(gt_path=tiny_segm_gt, seed=2)
    a = json.loads(a_path.read_bytes())
    b = json.loads(b_path.read_bytes())
    a_rles = [d["segmentation"]["counts"] for d in a]
    b_rles = [d["segmentation"]["counts"] for d in b]
    assert a_rles != b_rles


def test_bbox_stream_independent_of_mask_stream(
    tiny_synthetic_gt: Path, tiny_segm_gt: Path
) -> None:
    """v2's mask jitter draws come from a side stream so the bbox/score/FP
    distribution at a given seed is unchanged from v1. We can't compare
    against v1 cache files directly (they live under a different
    JITTER_PARAMS_VERSION), but we can assert the property that lets v1→v2
    cross-snapshot comparisons stay honest: a GT with no segmentation
    produces the same bbox/score values as a GT whose segmentation we
    decode but whose mask-stream draws don't perturb the main rng.
    """
    no_segm_dt = jittered_predictions.dt_path(gt_path=tiny_synthetic_gt, seed=42)
    with_segm_dt = jittered_predictions.dt_path(gt_path=tiny_segm_gt, seed=42)
    no_segm = json.loads(no_segm_dt.read_bytes())
    with_segm = json.loads(with_segm_dt.read_bytes())
    # Same number of TPs (kept_mask) — both GTs have 4 anns.
    assert len([d for d in no_segm if d["score"] > 0]) == len(
        [d for d in with_segm if d["score"] > 0]
    )
    # Bboxes must round-trip through the same main-stream draws regardless
    # of whether mask jitter ran.
    assert [d["bbox"] for d in no_segm] == [d["bbox"] for d in with_segm]
    assert [d["score"] for d in no_segm] == [d["score"] for d in with_segm]
