"""COCO panoptic val2017 whole-dataset parity smoke (ADR-0025 PR-6).

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
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

import vernier

from .harness import PanopticSnapshot, assert_snapshots_equal
from .panoptic_val_paths import require_artifacts, sample_image_count

# Make the vendored panopticapi importable without going through pytest's
# conftest discovery (this file is invoked directly by `pytest -m parity_panoptic_val`).
_ORACLE_PATH = Path(__file__).parent / "oracle" / "panopticapi"
if str(_ORACLE_PATH) not in sys.path:
    sys.path.insert(0, str(_ORACLE_PATH))


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

    all_d, per_class = pq_stat.pq_average(cats_dict, isthing=None)
    things_d, _ = pq_stat.pq_average(cats_dict, isthing=True)
    stuff_d, _ = pq_stat.pq_average(cats_dict, isthing=False)

    snap = PanopticSnapshot(
        pq=all_d["pq"],
        sq=all_d["sq"],
        rq=all_d["rq"],
        n=all_d["n"],
        pq_things=things_d["pq"],
        sq_things=things_d["sq"],
        rq_things=things_d["rq"],
        n_things=things_d["n"],
        pq_stuff=stuff_d["pq"],
        sq_stuff=stuff_d["sq"],
        rq_stuff=stuff_d["rq"],
        n_stuff=stuff_d["n"],
        per_class={int(k): dict(v) for k, v in per_class.items()},
    )
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
    cats = list(cats_dict.values())

    gt_segs_bytes = json.dumps({str(k): v for k, v in gt_segs.items()}).encode()
    dt_segs_bytes = json.dumps({str(k): v for k, v in dt_segs.items()}).encode()
    cats_bytes = json.dumps(cats).encode()

    gt = vernier.PanopticDataset.from_arrays(gt_label_maps, gt_segs_bytes, cats_bytes)
    dt = vernier.PanopticPredictions.from_arrays(dt_label_maps, dt_segs_bytes)
    summary = vernier.PanopticEvaluator(parity_mode="corrected").evaluate(gt, dt)

    return PanopticSnapshot(
        pq=summary.pq,
        sq=summary.sq,
        rq=summary.rq,
        n=summary.n,
        pq_things=summary.pq_things if summary.pq_things is not None else 0.0,
        sq_things=summary.sq_things if summary.sq_things is not None else 0.0,
        rq_things=summary.rq_things if summary.rq_things is not None else 0.0,
        n_things=summary.n_things if summary.n_things is not None else 0,
        pq_stuff=summary.pq_stuff if summary.pq_stuff is not None else 0.0,
        sq_stuff=summary.sq_stuff if summary.sq_stuff is not None else 0.0,
        rq_stuff=summary.rq_stuff if summary.rq_stuff is not None else 0.0,
        n_stuff=summary.n_stuff if summary.n_stuff is not None else 0,
        per_class={
            int(cat): {"pq": row.pq, "sq": row.sq, "rq": row.rq}
            for cat, row in summary.per_class().items()
        },
    )


@pytest.mark.parity_panoptic_val
def test_panoptic_val2017_strict_bit_equal() -> None:
    """End-to-end strict-mode parity vs `pq_compute_single_core` on
    a (subsampled) panoptic val2017 corpus."""
    gt_json, gt_png_dir, dt_json, dt_png_dir = require_artifacts()
    n_images = sample_image_count()

    oracle, gt_anns, dt_anns, cats_dict = _oracle_snapshot_full(
        gt_json, gt_png_dir, dt_json, dt_png_dir, n_images
    )
    # Defensive memory hygiene: the LVIS rollout's PR-6 memory peak
    # was load-bearing on full val (22 GB dense grid); panoptic is
    # structurally lighter (per-image PqStat fold) but keep the
    # gc.collect to close the per-snapshot cycle before the second.
    gc.collect()

    vsnap = _vernier_snapshot_full(gt_anns, gt_png_dir, dt_anns, dt_png_dir, cats_dict)
    gc.collect()

    assert_snapshots_equal(oracle, vsnap)
