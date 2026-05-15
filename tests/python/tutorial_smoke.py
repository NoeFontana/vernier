"""Platform-matrix smoke test (ADR-0030 array ingest via DLPack).

Mocks a batched detector — same calling convention as torchvision's
`DetectionModel(images) -> List[Dict[str, Tensor]]` — and submits each
batch directly to `BackgroundEvaluator`. vernier consumes the per-image
boxes / scores / labels arrays zero-copy through DLPack, so swapping
numpy for torch / jax / cupy at the model boundary needs no code
changes downstream.
"""

import json

import numpy as np

from vernier.instance import Bbox, CocoDataset, Detections, Evaluator

BATCH_SIZE = 2


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
            anns.append(
                {
                    "id": ann_id,
                    "image_id": img["id"],
                    "category_id": int(rng.choice([1, 2])),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return json.dumps(
        {
            "images": images,
            "categories": categories,
            "annotations": anns,
        }
    ).encode()


def _per_image(image_id: int) -> Detections:
    rng = np.random.default_rng(image_id)
    n_dets = 4
    boxes = np.empty((n_dets, 4), dtype=np.float64)
    boxes[:, 0] = rng.integers(0, 32, size=n_dets)
    boxes[:, 1] = rng.integers(0, 32, size=n_dets)
    boxes[:, 2] = rng.integers(8, 24, size=n_dets)
    boxes[:, 3] = rng.integers(8, 24, size=n_dets)
    return {
        "image_id": image_id,
        "boxes": boxes,
        "scores": rng.random(size=n_dets).astype(np.float64),
        "labels": rng.choice([1, 2], size=n_dets).astype(np.int64),
    }


def fake_model(image_ids: list[int]) -> list[Detections]:
    """Stand-in for `model(images) -> List[Dict[str, Tensor]]` — the
    torchvision detection-API shape. One Detections dict per image,
    each carrying boxes / scores / labels as arrays."""
    return [_per_image(i) for i in image_ids]


def main() -> None:
    gt_bytes = _make_gt()
    dataset = CocoDataset.from_json(gt_bytes)
    evaluator = Evaluator(iou=Bbox())
    all_ids = list(range(1, 6))
    batches = [all_ids[i : i + BATCH_SIZE] for i in range(0, len(all_ids), BATCH_SIZE)]
    with evaluator.background(dataset) as bg:
        for image_ids in batches:
            bg.submit(fake_model(image_ids))
        summary = bg.finalize()
    for line in summary.pretty_lines():
        print(line)
    print("smoke-ok: AP=", summary.stats[0])


if __name__ == "__main__":
    main()
