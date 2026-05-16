"""Synthetic stress-test GT/DT generator (ADR-0017 §"Workloads").

Parametric ``(n_images, n_categories, dt_per_image, gt_per_image, seed)``.
Every parameter is part of the cache key, so the release-mode ladder
(10k / 50k / 100k images at fixed categories=80, dt_per_image=30)
materializes once and re-runs from the cache thereafter.
"""

from __future__ import annotations

import json
import re
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

# Int-typed parameters that scaling reports can pivot on. ``iscrowd_fraction``
# is excluded — it's the only float param and isn't a scaling axis today.
SCALING_AXES: tuple[str, ...] = (
    "n_images",
    "n_categories",
    "gt_per_image",
    "dt_per_image",
    "seed",
)

_WORKLOAD_ID_RE = re.compile(
    r"^synthetic"
    r"_n(?P<n_images>\d+)"
    r"_c(?P<n_categories>\d+)"
    r"_g(?P<gt_per_image>\d+)"
    r"_d(?P<dt_per_image>\d+)"
    r"(?:_x(?P<iscrowd_pct>\d+))?"
    r"_s(?P<seed>\d+)$"
)


def workload_id(
    *,
    n_images: int,
    n_categories: int,
    dt_per_image: int,
    gt_per_image: int,
    seed: int,
    iscrowd_fraction: float = 0.0,
) -> str:
    # iscrowd_fraction == 0.0 → no _x suffix, preserving legacy cache slots.
    base = f"synthetic_n{n_images}_c{n_categories}_g{gt_per_image}_d{dt_per_image}"
    if iscrowd_fraction > 0.0:
        return f"{base}_x{int(iscrowd_fraction * 100):02d}_s{seed}"
    return f"{base}_s{seed}"


def parse_workload_id(wid: str) -> dict[str, int | float] | None:
    """Inverse of :func:`workload_id`; ``None`` for non-synthetic ids."""
    m = _WORKLOAD_ID_RE.match(wid)
    if m is None:
        return None
    out: dict[str, int | float] = {
        "n_images": int(m["n_images"]),
        "n_categories": int(m["n_categories"]),
        "gt_per_image": int(m["gt_per_image"]),
        "dt_per_image": int(m["dt_per_image"]),
        "seed": int(m["seed"]),
    }
    if m["iscrowd_pct"] is not None:
        out["iscrowd_fraction"] = int(m["iscrowd_pct"]) / 100.0
    return out


def _cache_dir() -> Path:
    return bench_cache_root() / "synthetic"


def _random_bbox(rng: np.random.Generator) -> list[float]:
    w = float(rng.uniform(20.0, IMAGE_W / 4.0))
    h = float(rng.uniform(20.0, IMAGE_H / 4.0))
    x = float(rng.uniform(0.0, IMAGE_W - w))
    y = float(rng.uniform(0.0, IMAGE_H - h))
    return [x, y, w, h]


def _generate(
    *,
    n_images: int,
    n_categories: int,
    dt_per_image: int,
    gt_per_image: int,
    seed: int,
    iscrowd_fraction: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    images = [
        {"id": i, "width": int(IMAGE_W), "height": int(IMAGE_H), "file_name": f"{i}.jpg"}
        for i in range(n_images)
    ]
    categories = [{"id": i + 1, "name": f"cat{i}"} for i in range(n_categories)]

    # Crowd anns omit ``segmentation`` — pycocotools' crowd path keys off
    # ``iscrowd`` + the bbox alone for bbox eval.
    n_crowd_per_image = int(gt_per_image * iscrowd_fraction)

    annotations: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    ann_id = 1
    for image in images:
        for k in range(gt_per_image):
            bbox = _random_bbox(rng)
            cat_id = int(rng.integers(1, n_categories + 1))
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image["id"],
                    "category_id": cat_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 1 if k < n_crowd_per_image else 0,
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
    *,
    n_images: int,
    n_categories: int,
    dt_per_image: int,
    gt_per_image: int,
    seed: int,
    iscrowd_fraction: float = 0.0,
) -> tuple[Path, Path]:
    """Return ``(gt_path, dt_path)``, materializing the cached pair if missing."""
    wid = workload_id(
        n_images=n_images,
        n_categories=n_categories,
        dt_per_image=dt_per_image,
        gt_per_image=gt_per_image,
        seed=seed,
        iscrowd_fraction=iscrowd_fraction,
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
        iscrowd_fraction=iscrowd_fraction,
    )
    cache.mkdir(parents=True, exist_ok=True)
    with gt_out.open("w") as f:
        json.dump(gt, f, separators=(",", ":"))
    with dt_out.open("w") as f:
        json.dump(detections, f, separators=(",", ":"))
    return gt_out, dt_out


def make_workload_scaled(
    scale: int = 1,
    *,
    seed: int = 0,
    iscrowd_fraction: float = 0.0,
) -> tuple[Path, Path]:
    """5000*scale images, 80 categories, 100 dt/gt per image — dense regime."""
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    return make_workload(
        n_images=5000 * scale,
        n_categories=80,
        dt_per_image=100,
        gt_per_image=100,
        seed=seed,
        iscrowd_fraction=iscrowd_fraction,
    )


# ADR-0047 threading-scaling smoke fixture. Intentionally small —
# this exists to validate the plumbing of the ``num_threads`` axis end
# to end (workload → CellSpec → runner → result-store path), not to
# produce release-gated scaling numbers. The full sweep
# (val2017 / LVIS / panoptic / ADE20K) is its own operation. Keep this
# small enough that ``just bench-threads-smoke`` finishes in <60s on a
# laptop.
THREADS_SMOKE_WORKLOAD_ID = "synthetic_threads_smoke"
THREADS_SMOKE_N_IMAGES = 100
THREADS_SMOKE_N_CATEGORIES = 10
THREADS_SMOKE_SEED = 0
THREADS_SMOKE_NUM_THREADS: tuple[int, ...] = (1, 2, 4, 8)
# Match the registry defaults in :mod:`bench.workloads.__init__` so the
# smoke fixture's on-disk cache slot is shared with a hypothetical
# ``synthetic:n_images=100,n_categories=10,seed=0`` invocation.
_THREADS_SMOKE_DT_PER_IMAGE = 30
_THREADS_SMOKE_GT_PER_IMAGE = 10


def threads_smoke_paths() -> tuple[Path, Path]:
    """Materialize (or look up) the GT/DT pair backing the
    ``synthetic_threads_smoke`` workload."""
    return make_workload(
        n_images=THREADS_SMOKE_N_IMAGES,
        n_categories=THREADS_SMOKE_N_CATEGORIES,
        dt_per_image=_THREADS_SMOKE_DT_PER_IMAGE,
        gt_per_image=_THREADS_SMOKE_GT_PER_IMAGE,
        seed=THREADS_SMOKE_SEED,
    )
