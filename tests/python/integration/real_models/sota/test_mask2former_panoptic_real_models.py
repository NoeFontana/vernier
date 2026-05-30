"""Real-prediction parity smoke for Mask2Former Swin-T panoptic vs panopticapi.

Sibling to ``test_detr_real_models.py``: real DT, real GT, strict-tier
parity claim. The model is ``facebook/mask2former-swin-tiny-coco-panoptic``;
predictions land under ``mask2former_panoptic_cache_dir()`` as one
rgb2id-encoded PNG per image + a ``panoptic_dt.json`` sidecar, all
keyed on the pinned hub commit SHA.

What this suite gates:

- **Strict bit-equality** on global PQ/SQ/RQ, Things and Stuff bucket
  PQ/SQ/RQ + counts, and the per-class PQ/SQ/RQ rows. Both the
  panopticapi oracle and vernier reduce the same integer
  intersect/union/tp totals via the same float arithmetic, so any
  drift in the integer accumulator surfaces here.

Skips cleanly when:

- ``real-models`` extra is missing (conftest's ``pytest.importorskip``).
- ``MASK2FORMER_PANOPTIC_REVISION`` is the ``_UNPINNED_REVISION``
  sentinel.
- COCO panoptic val2017 cache is not provisioned (i.e. user hasn't
  run ``python -m panoptic_val_cache``).
- COCO val2017 images are not present (``VERNIER_COCO_CACHE`` points
  at an incomplete layout).

First-time inference takes ~20-25h on an 8-core CPU (5000 images x
~14-18s/image; Mask2Former Swin-T runs 100 queries through 9
transformer decoder layers per image). Subsequent runs are seconds
once the prediction cache is on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ....parity_panoptic.harness import (
    PanopticSnapshot,
    assert_snapshots_equal,
    pq_stat_to_snapshot,
    summary_to_snapshot,
)

pytestmark = pytest.mark.real_models


def _build_label_maps(
    annotations: list[dict[str, Any]],
    png_dir: Path,
) -> dict[int, Any]:
    """Decode a list of (image_id, file_name) → uint32 label-map dict.

    Mirrors ``test_panoptic_val._build_label_maps``: rgb2id decode via
    Pillow + numpy. Pulled inline rather than imported to avoid
    coupling the SOTA harness to the parity_panoptic module's internal
    helpers (which are private to the val-smoke test).
    """
    import numpy as np
    from PIL import Image as PILImage

    out: dict[int, Any] = {}
    for ann in annotations:
        path = png_dir / ann["file_name"]
        rgb = np.array(PILImage.open(path), dtype=np.uint32)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"non-RGB panoptic PNG: {path}")
        out[int(ann["image_id"])] = rgb[:, :, 0] + 256 * rgb[:, :, 1] + 256 * 256 * rgb[:, :, 2]
    return out


def _oracle_snapshot(
    gt_dict: dict[str, Any],
    gt_png_dir: Path,
    dt_annotations: list[dict[str, Any]],
    dt_png_dir: Path,
) -> tuple[PanopticSnapshot, list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Run pq_compute_single_core on the (intersect of GT, DT) image set.

    Returns the snapshot plus the matched GT annotation list (so the
    vernier path uses the same image set) and the categories dict.
    """
    from panopticapi.evaluation import pq_compute_single_core  # type: ignore[import-not-found]

    dt_by_image = {a["image_id"]: a for a in dt_annotations}
    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for gt_ann in gt_dict["annotations"]:
        if gt_ann["image_id"] in dt_by_image:
            matched.append((gt_ann, dt_by_image[gt_ann["image_id"]]))

    cats_dict: dict[int, dict[str, Any]] = {c["id"]: c for c in gt_dict["categories"]}
    pq_stat = pq_compute_single_core(0, matched, str(gt_png_dir), str(dt_png_dir), cats_dict)
    snap = pq_stat_to_snapshot(pq_stat, cats_dict)
    gt_anns = [m[0] for m in matched]
    return snap, gt_anns, cats_dict


def _np_default(o: object) -> int | float | list[Any]:
    """JSON encoder for numpy scalars / arrays — same shape as the
    sibling :func:`tests.python.parity_panoptic.test_panoptic_val._np_default`.

    ``pq_compute_single_core`` mutates ``segments_info[*]['area']`` with
    a numpy int64 value derived from the rasterized PNG; re-encoding
    through vernier needs the cast back to native Python.
    """
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _vernier_snapshot(
    gt_anns: list[dict[str, Any]],
    gt_png_dir: Path,
    dt_annotations: list[dict[str, Any]],
    dt_png_dir: Path,
    cats_dict: dict[int, dict[str, Any]],
) -> PanopticSnapshot:
    """Run vernier-panoptic over the same matched image set."""
    import vernier

    dt_by_image = {a["image_id"]: a for a in dt_annotations}
    matched_dt = [dt_by_image[ann["image_id"]] for ann in gt_anns]

    gt_label_maps = _build_label_maps(gt_anns, gt_png_dir)
    dt_label_maps = _build_label_maps(matched_dt, dt_png_dir)
    gt_segs = {ann["image_id"]: ann["segments_info"] for ann in gt_anns}
    dt_segs = {ann["image_id"]: ann["segments_info"] for ann in matched_dt}

    gt_segs_bytes = json.dumps(
        {str(k): v for k, v in gt_segs.items()}, default=_np_default
    ).encode()
    dt_segs_bytes = json.dumps(
        {str(k): v for k, v in dt_segs.items()}, default=_np_default
    ).encode()
    cats_bytes = json.dumps(list(cats_dict.values()), default=_np_default).encode()

    gt = vernier.panoptic.Dataset.from_arrays(gt_label_maps, gt_segs_bytes, cats_bytes)
    dt = vernier.panoptic.Predictions.from_arrays(dt_label_maps, dt_segs_bytes)
    summary = vernier.panoptic.Evaluator(parity_mode="corrected").evaluate(gt, dt)
    return summary_to_snapshot(summary)


def test_mask2former_panoptic_parity_vs_panopticapi(
    coco_panoptic_gt: tuple[Path, Path, dict[str, Any]],
    mask2former_panoptic_cache_paths: tuple[Path, Path],
) -> None:
    """Strict-tier parity vs panopticapi on real Mask2Former predictions.

    The PQ/SQ/RQ reduction is integer-additive across images and
    floats only at the per-class average step; both sides apply that
    same float arithmetic. Drift would imply a real divergence in
    the per-image accumulator, not parser-level rounding (the
    DETR-R50 dtScores ULP drift doesn't surface in panoptic since
    panoptic doesn't carry the score-tensor pipeline).
    """
    _, gt_png_dir, gt_dict = coco_panoptic_gt
    dt_png_dir, dt_json_path = mask2former_panoptic_cache_paths
    dt_json = json.loads(dt_json_path.read_bytes())
    dt_annotations = dt_json["annotations"]

    oracle, gt_anns, cats_dict = _oracle_snapshot(gt_dict, gt_png_dir, dt_annotations, dt_png_dir)
    candidate = _vernier_snapshot(gt_anns, gt_png_dir, dt_annotations, dt_png_dir, cats_dict)

    assert_snapshots_equal(oracle, candidate)
