"""Stress-test workloads spanning multiple scaling axes (Phase 3
robustness sweep, follow-up to the 10x scale).

Named regimes mirror the persona load of modern ML setups (DETR-style
high detection density, LVIS-crowded scenes, satellite / pathology
large-image segmentation). One-axis-at-a-time sweeps fix everything at
a baseline and vary a single parameter so envelopes are readable.

Generation is on-demand (no shared cache) because image dimensions are
parametric here; the existing `synthetic.py` cache pins dims at module
level by design. Regimes are small enough (≤5k images, segm regimes
≤500 images) that one materialization per run is cheap."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from bench.workloads.jittered_predictions import SCORE_BETA_ALPHA, SCORE_BETA_BETA

IouType = Literal["bbox", "segm"]


@dataclass(frozen=True)
class StressRegime:
    name: str
    n_images: int
    n_categories: int
    dt_per_image: int
    gt_per_image: int
    image_w: int
    image_h: int
    iou_type: IouType = "bbox"
    seed: int = 0


# Named regimes covering common ML workloads. Sized for ~minutes-scale
# end-to-end runs on a workstation, not hours.
REGIMES: tuple[StressRegime, ...] = (
    StressRegime("coco-baseline", 5000, 80, 100, 10, 640, 480, "bbox"),
    StressRegime("detr-output", 5000, 80, 500, 10, 640, 480, "bbox"),
    StressRegime("lvis-crowded", 2000, 1203, 100, 50, 1024, 1024, "segm"),
    StressRegime("open-images-cats", 1000, 10000, 50, 30, 1024, 1024, "bbox"),
    StressRegime("satellite-4k", 500, 50, 50, 50, 4096, 4096, "segm"),
    StressRegime("pathology-8k", 100, 20, 100, 100, 8000, 8000, "segm"),
)


# Per-axis sweeps fix the baseline and vary one knob. Used by the
# stress runner's --axis mode.
SWEEP_BASELINE = StressRegime(
    "sweep-baseline", 2000, 80, 100, 30, 640, 480, "bbox"
)

SWEEPS: dict[str, tuple[StressRegime, ...]] = {
    "dt": tuple(
        StressRegime(f"sweep-dt-{n}", 2000, 80, n, 30, 640, 480, "bbox")
        for n in (10, 100, 500)
    ),
    "gt": tuple(
        StressRegime(f"sweep-gt-{n}", 2000, 80, 100, n, 640, 480, "bbox")
        for n in (10, 100, 500)
    ),
    "cats": (
        StressRegime("sweep-cats-80", 2000, 80, 100, 30, 640, 480, "bbox"),
        StressRegime("sweep-cats-1203", 1000, 1203, 100, 30, 640, 480, "bbox"),
        StressRegime("sweep-cats-10000", 500, 10000, 50, 20, 640, 480, "bbox"),
    ),
    "dims": (
        StressRegime("sweep-dims-640", 500, 50, 50, 30, 640, 480, "segm"),
        StressRegime("sweep-dims-1920", 500, 50, 50, 30, 1920, 1080, "segm"),
        StressRegime("sweep-dims-4096", 500, 50, 50, 30, 4096, 4096, "segm"),
    ),
}


def _random_bbox(rng: np.random.Generator, w: int, h: int) -> list[float]:
    bw = float(rng.uniform(20.0, w / 4.0))
    bh = float(rng.uniform(20.0, h / 4.0))
    x = float(rng.uniform(0.0, w - bw))
    y = float(rng.uniform(0.0, h - bh))
    return [x, y, bw, bh]


def _bbox_polygon(bbox: list[float]) -> list[list[float]]:
    """Rectangle-shaped polygon matching the bbox. Exercises the
    polygon→RLE path at the regime's image scale without modeling
    realistic instance shapes — the codec is what's under test."""
    x, y, w, h = bbox
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def materialize(regime: StressRegime, out_dir: Path) -> tuple[Path, Path]:
    """Write GT/DT JSON for ``regime`` under ``out_dir``. Returns paths."""
    rng = np.random.default_rng(regime.seed)
    w, h = regime.image_w, regime.image_h

    images = [
        {"id": i, "width": w, "height": h, "file_name": f"{i}.jpg"}
        for i in range(regime.n_images)
    ]
    categories = [{"id": i + 1, "name": f"cat{i}"} for i in range(regime.n_categories)]

    annotations: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    ann_id = 1
    for image in images:
        for _ in range(regime.gt_per_image):
            bbox = _random_bbox(rng, w, h)
            cat_id = int(rng.integers(1, regime.n_categories + 1))
            ann: dict[str, Any] = {
                "id": ann_id,
                "image_id": image["id"],
                "category_id": cat_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            }
            if regime.iou_type == "segm":
                ann["segmentation"] = _bbox_polygon(bbox)
            annotations.append(ann)
            ann_id += 1
        for _ in range(regime.dt_per_image):
            bbox = _random_bbox(rng, w, h)
            cat_id = int(rng.integers(1, regime.n_categories + 1))
            det: dict[str, Any] = {
                "image_id": image["id"],
                "category_id": cat_id,
                "bbox": bbox,
                "score": float(rng.beta(SCORE_BETA_ALPHA, SCORE_BETA_BETA)),
            }
            if regime.iou_type == "segm":
                det["segmentation"] = _bbox_polygon(bbox)
            detections.append(det)

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_path = out_dir / f"{regime.name}_gt.json"
    dt_path = out_dir / f"{regime.name}_dt.json"
    with gt_path.open("w") as f:
        json.dump(
            {"images": images, "categories": categories, "annotations": annotations},
            f,
            separators=(",", ":"),
        )
    with dt_path.open("w") as f:
        json.dump(detections, f, separators=(",", ":"))
    return gt_path, dt_path
