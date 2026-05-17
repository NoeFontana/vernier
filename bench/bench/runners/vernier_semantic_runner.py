"""vernier_semantic runner — invoked as a subprocess in
``bench/envs/vernier`` (ADR-0033 §B2).

Drives the val2017-scale semantic-mIoU cell through
:meth:`vernier.semantic.Evaluator.evaluate_from_pngs` (ADR-0037): the
fused libpng decode + confusion-matrix fold runs in Rust under
`py.detach`, no Pillow on the main thread, no `astype(uint32)` cast.
RSS is bounded by one decoded label-map per side at a time.

Emits a :class:`bench.harness.parity.SemanticSnapshot` JSON + a
per-class ``.npy`` table mirroring the panoptic two-artifact bundle.
"""

from __future__ import annotations

import sys

import numpy as np
import vernier

from bench.harness.parity import SemanticSnapshot
from bench.harness.timing import StageTable
from bench.runners._protocol import (
    parse_semantic_runner_args,
    per_class_uint64_table,
    scan_label_map_dir,
    write_semantic_outputs,
)


def _summary_to_snapshot(summary: vernier.semantic.Summary) -> SemanticSnapshot:
    """Project a :class:`vernier.semantic.Summary` into a
    :class:`SemanticSnapshot`. Per-class rows carry
    ``{iou, accuracy, precision}`` keyed by stringified class id.
    """
    per_class: dict[str, dict[str, float]] = {}
    for cls_id, row in summary.per_class().items():
        per_class[str(int(cls_id))] = {
            "iou": float(row.iou),
            "accuracy": float(row.accuracy),
            "precision": float(row.precision),
        }
    return SemanticSnapshot(
        miou=float(summary.miou),
        fwiou=float(summary.fwiou),
        pixel_accuracy=float(summary.pixel_accuracy),
        mean_accuracy=float(summary.mean_accuracy),
        n_classes=int(summary.confusion_matrix.n_classes),
        per_class=per_class,
    )


def _confusion_marginals(summary: vernier.semantic.Summary) -> np.ndarray:
    """Project vernier's NxN confusion matrix to the (4, N) marginals
    that are the cross-impl strict-tier parity surface.

    Convention (counts[gt_class, dt_class], same as
    :mod:`tests.python.parity_semantic.harness`):

    - ``intersect[i] = counts[i, i]`` — pixels where pred == gt == i.
    - ``area_label[i] = counts[i, :].sum()`` — total GT pixels in class i.
    - ``area_pred[i]  = counts[:, i].sum()`` — total predicted pixels in class i.
    - ``union[i]      = area_pred[i] + area_label[i] - intersect[i]``.

    These are exactly mmsegmentation ``IoUMetric.intersect_and_union``'s
    return values; equal marginals ⇒ equal mIoU under quirk AL2.
    """
    counts = summary.confusion_matrix.counts().astype(np.uint64, copy=False)
    intersect = np.diagonal(counts).astype(np.uint64, copy=True)
    area_label = counts.sum(axis=1, dtype=np.uint64)
    area_pred = counts.sum(axis=0, dtype=np.uint64)
    union = area_pred + area_label - intersect
    return np.stack((intersect, union, area_pred, area_label), axis=0)


def main() -> int:
    args = parse_semantic_runner_args()
    stages = StageTable()

    with stages.stage("load"):
        gt_paths = scan_label_map_dir(args.gt_label_map_dir)
        dt_paths = scan_label_map_dir(args.dt_label_map_dir)
        common = set(gt_paths) & set(dt_paths)
        if not common:
            raise RuntimeError(
                f"semantic runner found no overlapping image_ids between "
                f"{args.gt_label_map_dir} and {args.dt_label_map_dir}"
            )
        gt_for_ids = {iid: gt_paths[iid] for iid in common}
        dt_for_ids = {iid: dt_paths[iid] for iid in common}
        ignore_label = args.ignore_label if args.ignore_label >= 0 else None

    with stages.stage("evaluate_from_pngs"):
        # parity_mode="strict" pins bit-exact reproduction of
        # mmsegmentation's mIoU; the fused libpng path holds peak RSS
        # at one decoded label-map per side at a time.
        summary = vernier.semantic.Evaluator(parity_mode="strict").evaluate_from_pngs(
            gt_for_ids,
            dt_for_ids,
            n_classes=args.n_classes,
            ignore_label=ignore_label,
            num_threads=args.num_threads,
        )

    with stages.stage("aggregate"):
        snap = _summary_to_snapshot(summary)
        per_class_array = per_class_uint64_table(
            snap.per_class, columns=("iou", "accuracy", "precision")
        )
        confusion_array = _confusion_marginals(summary)
        snap_json = snap.model_dump_json().encode()

    stages.record("total", stages.total_so_far_ns())

    write_semantic_outputs(
        args=args,
        impl="vernier_semantic",
        impl_version=vernier.__version__,
        stages=stages.to_dict(),
        snapshot_json_bytes=snap_json,
        per_class_array=per_class_array,
        confusion_array=confusion_array,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
