"""Platform-matrix smoke test (ADR-0006 streaming API). Used by slow.yml platform-matrix job."""

from __future__ import annotations

import json

import numpy as np

from vernier.instance import Bbox, CocoDataset, Evaluator


def _make_gt() -> bytes:
    """5 images x 2 categories x 3 GT annotations each."""
    rng = np.random.default_rng(0)
    images = [{"id": i, "width": 64, "height": 64} for i in range(1, 6)]
    categories = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    anns: list[dict] = []
    ann_id = 1
    for img in images:
        for _ in range(3):
            x = float(rng.integers(0, 32))
            y = float(rng.integers(0, 32))
            w = float(rng.integers(8, 24))
            h = float(rng.integers(8, 24))
            anns.append({
                "id": ann_id,
                "image_id": img["id"],
                "category_id": int(rng.choice([1, 2])),
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1
    return json.dumps({
        "images": images,
        "categories": categories,
        "annotations": anns,
    }).encode()


def _make_dt_batch(image_ids: list[int], seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    dets = []
    for img_id in image_ids:
        for _ in range(4):
            x = float(rng.integers(0, 32))
            y = float(rng.integers(0, 32))
            w = float(rng.integers(8, 24))
            h = float(rng.integers(8, 24))
            dets.append({
                "image_id": img_id,
                "category_id": int(rng.choice([1, 2])),
                "bbox": [x, y, w, h],
                "score": float(rng.random()),
            })
    return json.dumps(dets).encode()


def main() -> None:
    gt_bytes = _make_gt()
    dataset = CocoDataset.from_json(gt_bytes)
    evaluator = Evaluator(iou=Bbox())
    with evaluator.background(dataset) as bg:
        for batch_idx in range(5):
            bg.submit(_make_dt_batch([batch_idx + 1], seed=batch_idx + 1))
        summary = bg.finalize()
    for line in summary.pretty_lines():
        print(line)
    print("smoke-ok: AP=", summary.stats[0])


if __name__ == "__main__":
    main()
