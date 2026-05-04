"""COCO panoptic val2017 whole-dataset parity smoke (ADR-0025).

Env-gated: requires :data:`VERNIER_PANOPTIC_GT_PATH` /
:data:`VERNIER_PANOPTIC_GT_PNG_DIR` / :data:`VERNIER_PANOPTIC_DT_PATH`
/ :data:`VERNIER_PANOPTIC_DT_PNG_DIR` to point at real artifacts.
Run ``python -m panoptic_val_cache`` to populate the canonical cache.

Skipped by default in CI (the panoptic GT + PNG bundle is ~250 MB
plus ~3 GB of images; we never commit the bytes per the COCO val
cache memory). Subsample defaults to 100 images for runtime;
override with ``VERNIER_PANOPTIC_VAL_SAMPLE_IMAGES``.

When the cache is provisioned with the perfect-DT (the
:func:`panoptic_val_cache.ensure_perfect_dt` flow), this smoke is a
sanity check: PQ=SQ=RQ=1.0 across every category. Real DTs from
upstream model zoos should be configured via the env vars and
produce non-trivial PQ values bit-equal to the oracle.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

import vernier

from .harness import (
    PanopticSnapshot,
    assert_snapshots_equal,
    pq_stat_to_snapshot,
    summary_to_snapshot,
)
from .panoptic_val_paths import require_artifacts, sample_image_count


def _decode_png_to_uint32(path: Path) -> np.ndarray:
    """Decode a panoptic PNG to a uint32 label map via Pillow + rgb2id.

    Mirrors panopticapi's eval-side decode (evaluation.py:86-89). The
    vernier path consumes uint32 ndarrays directly via
    PanopticDataset.from_arrays.
    """
    rgb = np.array(PILImage.open(path), dtype=np.uint32)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"non-RGB panoptic PNG: {path}")
    return rgb[:, :, 0] + 256 * rgb[:, :, 1] + 256 * 256 * rgb[:, :, 2]


def _build_label_maps(
    annotations: list[dict],
    png_dir: Path,
) -> dict[int, np.ndarray]:
    return {
        ann["image_id"]: _decode_png_to_uint32(png_dir / ann["file_name"]) for ann in annotations
    }


def _oracle_snapshot_full(
    gt_json: Path,
    gt_png_dir: Path,
    dt_json: Path,
    dt_png_dir: Path,
    n_images: int,
) -> tuple[PanopticSnapshot, list[dict], list[dict], dict[int, dict]]:
    """Run pq_compute_single_core on the (subsampled) val set."""
    from panopticapi.evaluation import pq_compute_single_core  # type: ignore[import-not-found]

    with gt_json.open() as f:
        gt = json.load(f)
    with dt_json.open() as f:
        dt = json.load(f)

    pred_by_image = {a["image_id"]: a for a in dt["annotations"]}
    matched: list[tuple[dict, dict]] = []
    for gt_ann in gt["annotations"]:
        if gt_ann["image_id"] in pred_by_image:
            matched.append((gt_ann, pred_by_image[gt_ann["image_id"]]))
    matched = matched[:n_images] if n_images >= 0 else matched

    cats_dict = {c["id"]: c for c in gt["categories"]}
    pq_stat = pq_compute_single_core(0, matched, str(gt_png_dir), str(dt_png_dir), cats_dict)

    snap = pq_stat_to_snapshot(pq_stat, cats_dict)
    gt_anns = [m[0] for m in matched]
    dt_anns = [m[1] for m in matched]
    return snap, gt_anns, dt_anns, cats_dict


def _vernier_snapshot_full(
    gt_anns: list[dict],
    gt_png_dir: Path,
    dt_anns: list[dict],
    dt_png_dir: Path,
    cats_dict: dict[int, dict],
) -> PanopticSnapshot:
    """Run the vernier pipeline on the same (subsampled) val set."""
    gt_label_maps = _build_label_maps(gt_anns, gt_png_dir)
    dt_label_maps = _build_label_maps(dt_anns, dt_png_dir)
    gt_segs = {ann["image_id"]: ann["segments_info"] for ann in gt_anns}
    dt_segs = {ann["image_id"]: ann["segments_info"] for ann in dt_anns}

    gt_segs_bytes = json.dumps({str(k): v for k, v in gt_segs.items()}).encode()
    dt_segs_bytes = json.dumps({str(k): v for k, v in dt_segs.items()}).encode()
    cats_bytes = json.dumps(list(cats_dict.values())).encode()

    gt = vernier.panoptic.Dataset.from_arrays(gt_label_maps, gt_segs_bytes, cats_bytes)
    dt = vernier.panoptic.Predictions.from_arrays(dt_label_maps, dt_segs_bytes)
    summary = vernier.panoptic.Evaluator(parity_mode="corrected").evaluate(gt, dt)
    return summary_to_snapshot(summary)


@pytest.mark.parity_panoptic_val
def test_panoptic_val2017_strict_bit_equal() -> None:
    """End-to-end strict-mode parity vs `pq_compute_single_core` on
    a (subsampled) panoptic val2017 corpus."""
    gt_json, gt_png_dir, dt_json, dt_png_dir = require_artifacts()
    n_images = sample_image_count()

    oracle, gt_anns, dt_anns, cats_dict = _oracle_snapshot_full(
        gt_json, gt_png_dir, dt_json, dt_png_dir, n_images
    )
    # Defensive: panoptic's per-image PqStat fold is structurally
    # lighter than the LVIS dense `Vec<Option<PerImageEval>>` grid,
    # but `gc.collect` between the two snapshots closes the oracle's
    # PQStat ref cycle before the vernier label-maps allocate.
    gc.collect()

    vsnap = _vernier_snapshot_full(gt_anns, gt_png_dir, dt_anns, dt_png_dir, cats_dict)
    gc.collect()

    assert_snapshots_equal(oracle, vsnap)
