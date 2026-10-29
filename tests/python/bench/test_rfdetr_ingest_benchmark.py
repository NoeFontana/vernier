"""Benchmark: ufcoco vs vernier on RF-DETR validation workloads (ADR-0066).

In RF-DETR (https://github.com/roboflow/rf-detr), model training runs validation at the
end of each epoch over 5,000 images x 300 unthresholded queries (~1.5 million detections
across 80 COCO categories).

ADR-0066 addresses five bottlenecks on this shape: the Python ``(N, 7)``
``column_stack`` copy (``DetectionsBatch``), FFI ingest, per-key index
vectors, per-cell allocation on empty-GT cells, and the boxed evaluation
grid. This benchmark times the whole path, so it tracks each stage as it
lands rather than pinning any one of them.
"""

from __future__ import annotations

import importlib
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from vernier._array_types import GtCategory
from vernier._core import CocoDataset, evaluate_bbox_grid, evaluate_bbox_summary
from vernier.adapters import DetectionColumns, TargetColumns, coco_inputs_from_columns

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    _uf_coco_mod = importlib.import_module("ultrafast_pycocotools.coco")
    _uf_eval_mod = importlib.import_module("ultrafast_pycocotools.cocoeval")
    COCO: Any = _uf_coco_mod.COCO
    UfCOCOeval: Any = _uf_eval_mod.COCOeval
    HAS_UFCOCO = True
except ImportError:
    COCO = None
    UfCOCOeval = None
    HAS_UFCOCO = False


def _generate_synthetic_rfdetr_state(
    n_images: int = 1000,
    dets_per_image: int = 100,
    gt_per_image: int = 7,
    n_classes: int = 80,
    seed: int = 42,
) -> tuple[CocoDataset, NDArray[np.float64], DetectionColumns, TargetColumns]:
    """Generate synthetic detection and ground-truth state matching RF-DETR's evaluation format."""
    rng = np.random.default_rng(seed)
    n_dets = n_images * dets_per_image
    n_gt = n_images * gt_per_image

    dt_image_ids = np.repeat(np.arange(n_images, dtype=np.float64), dets_per_image)
    dt_boxes = rng.uniform(0.0, 500.0, size=(n_dets, 4))
    dt_boxes[:, 2:] = rng.uniform(10.0, 120.0, size=(n_dets, 2))
    dt_scores = rng.uniform(0.01, 1.0, size=n_dets)
    dt_labels = rng.integers(0, n_classes, size=n_dets).astype(np.float64)

    matrix = np.column_stack([dt_image_ids, dt_boxes, dt_scores, dt_labels])

    det_columns: DetectionColumns = {
        "boxes": dt_boxes,
        "scores": dt_scores,
        "labels": dt_labels.astype(np.int64),
        "counts": np.full(n_images, dets_per_image, dtype=np.int64),
    }

    gt_boxes = rng.uniform(0.0, 500.0, size=(n_gt, 4))
    gt_boxes[:, 2:] = rng.uniform(10.0, 120.0, size=(n_gt, 2))
    gt_labels = rng.integers(0, n_classes, size=n_gt).astype(np.int64)
    target_columns: TargetColumns = {
        "boxes": gt_boxes,
        "labels": gt_labels,
        "iscrowd": np.zeros(n_gt, dtype=np.int64),
        "area": gt_boxes[:, 2] * gt_boxes[:, 3],
        "counts": np.full(n_images, gt_per_image, dtype=np.int64),
    }

    categories: list[GtCategory] = [{"id": i, "name": str(i)} for i in range(n_classes)]
    gt_dataset = CocoDataset.from_arrays(
        {
            "id": np.arange(n_images, dtype=np.int64),
            "height": np.full(n_images, 800, dtype=np.int64),
            "width": np.full(n_images, 800, dtype=np.int64),
        },
        {
            "id": np.arange(1, n_gt + 1, dtype=np.int64),
            "image_id": np.repeat(np.arange(n_images, dtype=np.int64), gt_per_image),
            "category_id": gt_labels,
            "bbox": gt_boxes,
            "area": gt_boxes[:, 2] * gt_boxes[:, 3],
            "iscrowd": np.zeros(n_gt, dtype=np.int64),
        },
        categories,
    )

    return gt_dataset, matrix, det_columns, target_columns


def _time_fn(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    res = fn(*args, **kwargs)
    return res, time.perf_counter() - start


@pytest.mark.benchmark
def test_rfdetr_ingest_ufcoco_vs_vernier() -> None:
    """Benchmark comparing ufcoco vs vernier on an RF-DETR detection state."""
    n_images = 1000
    dets_per_img = 100
    gt_dataset, matrix, det_cols, tgt_cols = _generate_synthetic_rfdetr_state(
        n_images=n_images, dets_per_image=dets_per_img
    )

    (gt_from_cols, dt_batch), t_vernier_adapter = _time_fn(
        coco_inputs_from_columns,
        det_cols,
        tgt_cols,
        box_format="xywh",
        categories=list(range(80)),
        area="auto",
    )
    assert isinstance(dt_batch, dict)
    assert "image_ids" in dt_batch

    summary_batch, t_vernier_batch_e2e = _time_fn(
        evaluate_bbox_summary,
        gt_from_cols,
        dt_batch,
        parity_mode="strict",
        max_dets=[1, 10, 100],
        use_cats=True,
    )
    summary_matrix, t_vernier_matrix_e2e = _time_fn(
        evaluate_bbox_summary,
        gt_dataset,
        matrix,
        parity_mode="strict",
        max_dets=[1, 10, 100],
        use_cats=True,
    )
    assert summary_batch.stats == summary_matrix.stats

    print(f"\n[RF-DETR Benchmark: {n_images} imgs, {n_images * dets_per_img} dets]")
    print(f"  vernier coco_inputs_from_columns:     {t_vernier_adapter * 1000:.2f} ms")
    print(f"  vernier e2e (DetectionsBatch, 1T):    {t_vernier_batch_e2e * 1000:.2f} ms")
    print(f"  vernier e2e ((N, 7) Matrix, 1T):      {t_vernier_matrix_e2e * 1000:.2f} ms")

    if HAS_UFCOCO:
        tgt_raw: dict[str, Any] = dict(tgt_cols)
        uf_coco_gt = COCO()
        uf_coco_gt.dataset = {
            "images": [{"id": i, "width": 800, "height": 800} for i in range(n_images)],
            "categories": [{"id": i, "name": str(i)} for i in range(80)],
            "annotations": [
                {
                    "id": idx + 1,
                    "image_id": idx // 7,
                    "category_id": int(tgt_raw["labels"][idx]),
                    "bbox": tgt_raw["boxes"][idx].tolist(),
                    "area": float(tgt_raw["area"][idx]),
                    "iscrowd": 0,
                }
                for idx in range(len(tgt_raw["labels"]))
            ],
        }
        uf_coco_gt.createIndex()

        def run_ufcoco() -> list[float]:
            uf_dt = uf_coco_gt.loadRes(matrix)
            ev = UfCOCOeval(uf_coco_gt, uf_dt, "bbox")
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            return list(ev.stats)

        uf_stats, t_ufcoco_e2e = _time_fn(run_ufcoco)
        print(f"  ufcoco e2e (loadRes+eval+acc+sum):    {t_ufcoco_e2e * 1000:.2f} ms")
        # Informational only: parity is pinned against pycocotools by the
        # parity suite, never against a third-party implementation.
        print(
            f"  max |stat delta| vs ufcoco:           "
            f"{max(abs(a - b) for a, b in zip(summary_batch.stats, uf_stats, strict=True)):.3g}"
        )


@pytest.mark.benchmark
@pytest.mark.slow
def test_rfdetr_ingest_scale_5000_images() -> None:
    """Full COCO val2017 scale RF-DETR benchmark: 5,000 images x 300 dets/img (1.5M dets)."""
    n_images = 5000
    dets_per_img = 300
    gt_dataset, matrix, det_cols, tgt_cols = _generate_synthetic_rfdetr_state(
        n_images=n_images, dets_per_image=dets_per_img
    )

    (gt_from_cols, dt_batch), t_adapter = _time_fn(
        coco_inputs_from_columns,
        det_cols,
        tgt_cols,
        box_format="xywh",
        categories=list(range(80)),
        area="auto",
    )

    _, t_grid_only = _time_fn(
        evaluate_bbox_grid,
        gt_from_cols,
        dt_batch,
        parity_mode="strict",
        max_dets_per_image=100,
        use_cats=True,
        num_threads=4,
    )
    summary_1t, t_e2e_1t = _time_fn(
        evaluate_bbox_summary,
        gt_from_cols,
        dt_batch,
        parity_mode="strict",
        max_dets=[1, 10, 100],
        use_cats=True,
    )
    summary_4t, t_e2e_4t = _time_fn(
        evaluate_bbox_summary,
        gt_from_cols,
        dt_batch,
        parity_mode="strict",
        max_dets=[1, 10, 100],
        use_cats=True,
        num_threads=4,
    )
    summary_mat_4t, t_mat_4t = _time_fn(
        evaluate_bbox_summary,
        gt_dataset,
        matrix,
        parity_mode="strict",
        max_dets=[1, 10, 100],
        use_cats=True,
        num_threads=4,
    )

    assert summary_1t.stats == summary_4t.stats
    assert summary_1t.stats == summary_mat_4t.stats

    print(f"\n[Scale 1.5M Detections: {n_images} imgs, {n_images * dets_per_img} dets]")
    print(f"  vernier coco_inputs_from_columns:     {t_adapter * 1000:.2f} ms")
    print(f"  vernier evaluate_bbox_grid (4T):      {t_grid_only * 1000:.2f} ms")
    print(f"  vernier e2e DetectionsBatch (1T):     {t_e2e_1t * 1000:.2f} ms")
    print(f"  vernier e2e DetectionsBatch (4T):     {t_e2e_4t * 1000:.2f} ms")
    print(f"  vernier e2e (N, 7) Matrix (4T):       {t_mat_4t * 1000:.2f} ms")
