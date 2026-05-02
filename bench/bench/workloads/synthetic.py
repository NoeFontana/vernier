"""Synthetic stress-test GT/DT generator (ADR-0017 §"Workloads").

Parametric ``(n_images, n_categories, dt_per_image, gt_per_image, seed)``.
Every parameter is part of the cache key, so the release-mode ladder
(10k / 50k / 100k images at fixed categories=80, dt_per_image=30)
materializes once and re-runs from the cache thereafter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bench.harness.paths import bench_cache_root
from bench.workloads.jittered_predictions import SCORE_BETA_ALPHA, SCORE_BETA_BETA

# Image canvas. Fixed to keep the workload identity tied only to the
# explicit param tuple — varying the canvas would force every cache
# key to grow.
IMAGE_W = 1024.0
IMAGE_H = 1024.0


def workload_id(
    *, n_images: int, n_categories: int, dt_per_image: int, gt_per_image: int, seed: int
) -> str:
    return (
        f"synthetic_n{n_images}_c{n_categories}"
        f"_g{gt_per_image}_d{dt_per_image}_s{seed}"
    )


def _cache_dir() -> Path:
    return bench_cache_root() / "synthetic"


def _random_bbox(rng: np.random.Generator) -> list[float]:
    w = float(rng.uniform(20.0, IMAGE_W / 4.0))
    h = float(rng.uniform(20.0, IMAGE_H / 4.0))
    x = float(rng.uniform(0.0, IMAGE_W - w))
    y = float(rng.uniform(0.0, IMAGE_H - h))
    return [x, y, w, h]


def _generate(
    *, n_images: int, n_categories: int, dt_per_image: int, gt_per_image: int, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    images = [
        {"id": i, "width": int(IMAGE_W), "height": int(IMAGE_H), "file_name": f"{i}.jpg"}
        for i in range(n_images)
    ]
    categories = [{"id": i + 1, "name": f"cat{i}"} for i in range(n_categories)]

    annotations: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    ann_id = 1
    for image in images:
        for _ in range(gt_per_image):
            bbox = _random_bbox(rng)
            cat_id = int(rng.integers(1, n_categories + 1))
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image["id"],
                    "category_id": cat_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
            ann_id += 1
        for _ in range(dt_per_image):
            bbox = _random_bbox(rng)
            cat_id = int(rng.integers(1, n_categories + 1))
            detections.append(
                {
                    "image_id": image["id"],
                    "category_id": cat_id,
                    "bbox": bbox,
                    "score": float(rng.beta(SCORE_BETA_ALPHA, SCORE_BETA_BETA)),
                }
            )

    gt = {"images": images, "categories": categories, "annotations": annotations}
    return gt, detections


def make_workload(
    *, n_images: int, n_categories: int, dt_per_image: int, gt_per_image: int, seed: int
) -> tuple[Path, Path]:
    """Return ``(gt_path, dt_path)``, materializing the cached pair if missing."""
    wid = workload_id(
        n_images=n_images,
        n_categories=n_categories,
        dt_per_image=dt_per_image,
        gt_per_image=gt_per_image,
        seed=seed,
    )
    cache = _cache_dir()
    gt_out = cache / f"{wid}_gt.json"
    dt_out = cache / f"{wid}_dt.json"
    if gt_out.exists() and dt_out.exists():
        return gt_out, dt_out

    gt, detections = _generate(
        n_images=n_images,
        n_categories=n_categories,
        dt_per_image=dt_per_image,
        gt_per_image=gt_per_image,
        seed=seed,
    )
    cache.mkdir(parents=True, exist_ok=True)
    with gt_out.open("w") as f:
        json.dump(gt, f, separators=(",", ":"))
    with dt_out.open("w") as f:
        json.dump(detections, f, separators=(",", ":"))
    return gt_out, dt_out
