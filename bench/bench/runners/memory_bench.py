"""Memory + GIL under training-loop load. Submits a synthetic detection
stream through `BackgroundEvaluator` while `bench.harness.rss.RSSSampler`
captures the RSS curve. Output CSV feeds the graph in
`docs/engineering/memory-under-training.md`."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import vernier
from vernier.instance import BackgroundEvaluator

from bench.harness.rss import RSSSampler

_N_IMAGES = 50
_N_CATEGORIES = 80
_DETS_PER_STEP = 5
_IMAGE_W = 640
_IMAGE_H = 480


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "memory" / "training-loop.csv",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _build_synthetic_gt() -> bytes:
    images = [
        {"id": i, "width": _IMAGE_W, "height": _IMAGE_H, "file_name": f"{i}.jpg"}
        for i in range(_N_IMAGES)
    ]
    categories = [{"id": i + 1, "name": f"cat{i}"} for i in range(_N_CATEGORIES)]
    annotations = [
        {
            "id": i + 1,
            "image_id": i,
            "category_id": (i % _N_CATEGORIES) + 1,
            "bbox": [10.0, 10.0, 40.0, 40.0],
            "area": 40.0 * 40.0,
            "iscrowd": 0,
        }
        for i in range(_N_IMAGES)
    ]
    gt = {"images": images, "categories": categories, "annotations": annotations}
    return json.dumps(gt, separators=(",", ":")).encode("utf-8")


def _fake_batch(rng: np.random.Generator, image_id: int) -> bytes:
    records: list[dict[str, Any]] = []
    cat_ids = rng.integers(1, _N_CATEGORIES + 1, size=_DETS_PER_STEP)
    scores = rng.random(size=_DETS_PER_STEP)
    for cat_id, score in zip(cat_ids, scores, strict=True):
        records.append(
            {
                "image_id": image_id,
                "category_id": int(cat_id),
                "bbox": [5.0, 5.0, 50.0, 50.0],
                "score": float(score),
            }
        )
    return json.dumps(records).encode("utf-8")


def main() -> None:
    args = _parse_args()
    gt_bytes = _build_synthetic_gt()
    rng = np.random.default_rng(args.seed)

    with BackgroundEvaluator(gt_bytes, iou_type="bbox") as bg, RSSSampler() as s:
        for _ in range(args.epochs):
            for step in range(args.steps_per_epoch):
                bg.submit(_fake_batch(rng, step % _N_IMAGES))
        bg.finalize()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_s", "rss_bytes"])
        for t, rss in s.samples:
            w.writerow([f"{t:.4f}", rss])
    print(f"[memory-bench] csv={args.output} vernier={vernier.__version__}")


if __name__ == "__main__":
    main()
