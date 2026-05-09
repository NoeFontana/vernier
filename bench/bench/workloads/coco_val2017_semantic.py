"""COCO val2017 semantic workload (ADR-0033 §B2 follow-up).

Real-image semantic-segmentation workload derived from the COCO
panoptic val2017 cache. The conversion (panoptic RGB segment_id
→ contiguous train-id label-map) lives in
:mod:`panoptic_val_cache` so the parity tests can reach the same
artifacts without depending on the bench package.

Class set: all 133 panoptic categories (80 thing + 53 stuff). The
train-id mapping is the ascending sort of upstream ``category_id`` —
deterministic across runs. Pixels with ``segment_id == 0`` (panoptic
"unlabeled" sentinel) map to ``ignore_label = 255`` (Cityscapes /
Pascal VOC convention).

The perfect-DT cell is GT-as-DT (symlink each GT PNG into the DT
dir). Both vernier and the mmsegmentation oracle should report
``mIoU = 1.0`` per present class.
"""

from __future__ import annotations

from pathlib import Path

from panoptic_val_cache import (
    SEMANTIC_IGNORE_LABEL as IGNORE_LABEL,  # re-exported for callers
)
from panoptic_val_cache import (
    ensure_semantic_gt,
    ensure_semantic_perfect_dt,
)

PERFECT_WORKLOAD_ID: str = "coco_val2017_semantic_perfect"


def perfect_workload_paths() -> tuple[Path, Path, int, dict[int, int]]:
    """Return ``(gt_label_maps, dt_label_maps, n_classes, train_id_to_category_id)``.

    Materializes the cached semantic GT + perfect DT if missing.
    Idempotent. Raises :class:`FileNotFoundError` (with an actionable
    hint) when the panoptic cache itself isn't provisioned.
    """
    gt_dir, n_classes, train_id_to_cat_id = ensure_semantic_gt()
    dt_dir = ensure_semantic_perfect_dt()
    return gt_dir, dt_dir, n_classes, train_id_to_cat_id


__all__ = [
    "IGNORE_LABEL",
    "PERFECT_WORKLOAD_ID",
    "perfect_workload_paths",
]
