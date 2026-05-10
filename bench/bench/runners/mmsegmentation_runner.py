"""mmsegmentation runner — vendored ``IoUMetric`` oracle for the
strict-tier semantic-mIoU parity pair (ADR-0033 §B2 / ADR-0036).

Loops the val2017 / synthetic-semantic GT-DT label-map pairs through
``IoUMetric.intersect_and_union``, sums the four per-class u64
marginals (intersect, union, area_pred, area_label) across images,
and emits the three-artifact bundle shared with
:mod:`vernier_semantic_runner`. The marginals are the parity surface
— mmseg's native output, and the shape vernier projects its NxN
confusion matrix to.

Stub installation goes through the vendored ``_loader.install_stubs``
so the oracle's import sequence matches the parity tests'. The pinned
SHA below mirrors ``ORACLE_MMSEGMENTATION_COMMIT_SHA`` in
``crates/vernier-semantic/src/parity.rs`` (cargo tripwire test
asserts the two stay in lockstep).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch  # pyright: ignore[reportMissingImports]
from PIL import Image

from bench.harness.parity import SemanticSnapshot
from bench.harness.paths import REPO_ROOT
from bench.harness.timing import StageTable
from bench.runners._protocol import (
    parse_semantic_runner_args,
    per_class_uint64_table,
    scan_label_map_dir,
    write_semantic_outputs,
)

# Mirror of ``ORACLE_MMSEGMENTATION_COMMIT_SHA`` in
# ``crates/vernier-semantic/src/parity.rs``; the cargo tripwire test
# enforces equality with the table in ``VENDORING.md``.
_ORACLE_COMMIT_SHA: str = "c685fe6767c4cadf6b051983ca6208f1b9d1ccb8"

_VENDORED_MMSEG_PATH: Path = (
    REPO_ROOT / "tests" / "python" / "parity_semantic" / "oracle" / "mmsegmentation"
)


def _install_mmseg_oracle() -> type:
    """Return the vendored ``IoUMetric`` class, installing stubs first."""
    if not _VENDORED_MMSEG_PATH.is_dir():
        raise FileNotFoundError(
            f"vendored mmsegmentation oracle not found at "
            f"{_VENDORED_MMSEG_PATH} — see ADR-0036 / VENDORING.md"
        )
    if str(_VENDORED_MMSEG_PATH) not in sys.path:
        sys.path.insert(0, str(_VENDORED_MMSEG_PATH))

    from _loader import install_stubs  # pyright: ignore[reportMissingImports]

    install_stubs()

    from mmseg.evaluation.metrics.iou_metric import (  # pyright: ignore[reportMissingImports]
        IoUMetric,
    )

    return IoUMetric


def _decode_png(path: Path) -> torch.Tensor:
    """Decode a single-channel label-map PNG into a torch tensor.

    Native dtype (uint8 typical) is preserved — ``IoUMetric`` does its
    own ``.float()`` promotion inside ``intersect_and_union``, so a
    pre-cast int64 would just double the working set for no parity
    benefit. ``np.array(..., copy=True)`` materializes a writable
    backing buffer; PIL's underlying mmap is read-only and torch
    warns when handed a read-only numpy array.
    """
    arr = np.array(Image.open(path), copy=True)
    if arr.ndim != 2:
        raise ValueError(
            f"mmsegmentation runner expects single-channel label-map PNGs; "
            f"got {arr.shape} from {path}"
        )
    return torch.from_numpy(arr)  # pyright: ignore[reportPrivateImportUsage]


def _project_to_snapshot(
    *,
    intersect: np.ndarray,
    union: np.ndarray,
    area_pred: np.ndarray,
    area_label: np.ndarray,
    n_classes: int,
) -> SemanticSnapshot:
    """Apply mmseg's NaN-on-zero division (quirk AL2) to per-class u64
    totals and pack into a :class:`SemanticSnapshot`.
    """
    label_sum = float(area_label.sum())
    pixel_accuracy = float("nan") if label_sum == 0.0 else float(intersect.sum()) / label_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = intersect.astype(np.float64) / union.astype(np.float64)
        accuracy = intersect.astype(np.float64) / area_label.astype(np.float64)
        precision = intersect.astype(np.float64) / area_pred.astype(np.float64)

    mean_iou = float(np.nanmean(iou)) if iou.size else float("nan")
    mean_accuracy = float(np.nanmean(accuracy)) if accuracy.size else float("nan")
    if label_sum == 0.0:
        fwiou = float("nan")
    else:
        weights = area_label.astype(np.float64) / label_sum
        fwiou = float(np.nansum(np.where(np.isnan(iou), 0.0, iou * weights)))

    per_class: dict[str, dict[str, float]] = {
        str(cls_id): {
            "iou": float(iou[cls_id]),
            "accuracy": float(accuracy[cls_id]),
            "precision": float(precision[cls_id]),
        }
        for cls_id in range(n_classes)
    }
    return SemanticSnapshot(
        miou=mean_iou,
        fwiou=fwiou,
        pixel_accuracy=pixel_accuracy,
        mean_accuracy=mean_accuracy,
        n_classes=int(n_classes),
        per_class=per_class,
    )


def main() -> int:
    args = parse_semantic_runner_args()
    stages = StageTable()
    iou_metric = _install_mmseg_oracle()

    n_classes = int(args.n_classes)
    # ``parse_semantic_runner_args`` encodes "no ignore label" as -1;
    # mmseg's intersect_and_union just compares against the label
    # tensor, so a value that never appears is the right pass-through.
    ignore_index = int(args.ignore_label)

    with stages.stage("load"):
        gt_paths = scan_label_map_dir(args.gt_label_map_dir)
        dt_paths = scan_label_map_dir(args.dt_label_map_dir)
        common = sorted(set(gt_paths) & set(dt_paths))
        if not common:
            raise RuntimeError(
                f"mmsegmentation runner found no overlapping image_ids between "
                f"{args.gt_label_map_dir} and {args.dt_label_map_dir}"
            )

    with stages.stage("evaluate"):
        intersect_total = np.zeros(n_classes, dtype=np.uint64)
        union_total = np.zeros(n_classes, dtype=np.uint64)
        pred_total = np.zeros(n_classes, dtype=np.uint64)
        label_total = np.zeros(n_classes, dtype=np.uint64)
        for image_id in common:
            gt_t = _decode_png(gt_paths[image_id])
            dt_t = _decode_png(dt_paths[image_id])
            intersect, union, area_pred, area_label = iou_metric.intersect_and_union(
                dt_t, gt_t, n_classes, ignore_index
            )
            intersect_total += intersect.numpy().astype(np.uint64)
            union_total += union.numpy().astype(np.uint64)
            pred_total += area_pred.numpy().astype(np.uint64)
            label_total += area_label.numpy().astype(np.uint64)

    with stages.stage("aggregate"):
        snap = _project_to_snapshot(
            intersect=intersect_total,
            union=union_total,
            area_pred=pred_total,
            area_label=label_total,
            n_classes=n_classes,
        )
        per_class_array = per_class_uint64_table(
            snap.per_class, columns=("iou", "accuracy", "precision")
        )
        confusion_marginals = np.stack(
            (intersect_total, union_total, pred_total, label_total), axis=0
        )
        snap_json = snap.model_dump_json().encode()

    stages.record("total", stages.total_so_far_ns())

    write_semantic_outputs(
        args=args,
        impl="mmsegmentation",
        impl_version=_ORACLE_COMMIT_SHA,
        stages=stages.to_dict(),
        snapshot_json_bytes=snap_json,
        per_class_array=per_class_array,
        confusion_array=confusion_marginals,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
