"""Deterministic jittered detections from a COCO GT JSON.

For every non-crowd GT annotation, emit a detection whose bbox is the
GT bbox + Gaussian noise on ``(x, y, w, h)``, with a confidence drawn
from a beta biased toward 1. A controlled fraction of GTs is dropped
(false negatives) and the same fraction is added at random positions
(false positives). The cache key is ``(seed, *jitter_params)`` — bump
:data:`JITTER_PARAMS_VERSION` if any default changes so old caches
self-invalidate.

Bbox-only for now. Segm/keypoints jittering follows when those iou
types ship in the runtime matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bench.harness.paths import bench_cache_root

# Synthetic detection scores are drawn from this beta. Single source of
# truth so jittered + synthetic stay aligned without silent drift.
SCORE_BETA_ALPHA = 5.0
SCORE_BETA_BETA = 1.0

# Single bump-on-default-change number, shared by the workload_id and
# cache filename so altering any default invalidates pre-existing
# detections without per-knob bookkeeping.
JITTER_PARAMS_VERSION = 1

# Easy preset (per ADR-0017 §"Workloads → COCO val2017") — small bbox
# jitter, modest FP/FN, scores biased toward correct so ranking is
# realistic but not pathologically tied.
BBOX_JITTER_SIGMA_PX = 5.0
FP_FRACTION = 0.05
FN_FRACTION = 0.05


def _jittered_dt_path(seed: int) -> Path:
    cache = bench_cache_root() / "jittered"
    return cache / f"coco_val2017_jittered_seed{seed}_v{JITTER_PARAMS_VERSION}.json"


def workload_id(seed: int) -> str:
    return f"coco_val2017_jittered_seed{seed}"


def _generate(*, gt_path: Path, seed: int, out: Path) -> None:
    with gt_path.open("rb") as f:
        gt = json.load(f)
    rng = np.random.default_rng(seed)

    images = gt.get("images", [])
    image_size_by_id: dict[int, tuple[float, float]] = {
        int(img["id"]): (float(img["width"]), float(img["height"])) for img in images
    }

    non_crowd = [a for a in gt["annotations"] if not a.get("iscrowd", 0)]
    keep_mask = rng.random(len(non_crowd)) >= FN_FRACTION

    detections: list[dict[str, object]] = []
    for ann, keep in zip(non_crowd, keep_mask, strict=True):
        if not keep:
            continue
        x, y, w, h = ann["bbox"]
        noise = rng.normal(0.0, BBOX_JITTER_SIGMA_PX, size=4)
        jx, jy, jw, jh = x + noise[0], y + noise[1], w + noise[2], h + noise[3]
        # Clamp to non-degenerate positive width/height so loadRes
        # doesn't reject. pycocotools tolerates floats here.
        jw = max(jw, 1.0)
        jh = max(jh, 1.0)
        score = float(rng.beta(SCORE_BETA_ALPHA, SCORE_BETA_BETA))
        detections.append(
            {
                "image_id": int(ann["image_id"]),
                "category_id": int(ann["category_id"]),
                "bbox": [float(jx), float(jy), float(jw), float(jh)],
                "score": score,
            }
        )

    n_fp = round(FP_FRACTION * len(non_crowd))
    if n_fp and image_size_by_id and gt.get("categories"):
        image_ids = list(image_size_by_id.keys())
        category_ids = [int(c["id"]) for c in gt["categories"]]
        fp_image_idx = rng.integers(0, len(image_ids), size=n_fp)
        fp_cat_idx = rng.integers(0, len(category_ids), size=n_fp)
        fp_scores = rng.beta(SCORE_BETA_ALPHA, SCORE_BETA_BETA, size=n_fp)
        for i in range(n_fp):
            img_id = image_ids[int(fp_image_idx[i])]
            iw, ih = image_size_by_id[img_id]
            w = float(rng.uniform(10.0, max(11.0, iw / 4.0)))
            h = float(rng.uniform(10.0, max(11.0, ih / 4.0)))
            x = float(rng.uniform(0.0, max(1.0, iw - w)))
            y = float(rng.uniform(0.0, max(1.0, ih - h)))
            detections.append(
                {
                    "image_id": img_id,
                    "category_id": category_ids[int(fp_cat_idx[i])],
                    "bbox": [x, y, w, h],
                    "score": float(fp_scores[i]),
                }
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(detections, f, separators=(",", ":"))


def dt_path(*, gt_path: Path, seed: int) -> Path:
    """Return the cached jittered DT, generating it if missing."""
    out = _jittered_dt_path(seed)
    if not out.exists():
        _generate(gt_path=gt_path, seed=seed, out=out)
    return out
