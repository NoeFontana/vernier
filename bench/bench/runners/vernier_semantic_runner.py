"""Vernier semantic-segmentation runner — invoked as a subprocess in
``bench/envs/cityscapes`` (ADR-0033 §B2).

Calls :class:`vernier.semantic.Evaluator` against the trainId-space
label-map PNGs the workload provides; emits the accumulated NxN uint64
confusion matrix as ``<impl>.npy`` under the canonical ``"confusion"``
artifact slot.

Stages: ``load`` (decode PNGs), ``accumulate_confusion``
(``Evaluator.evaluate(...)`` — under the hood, the FFI bincount fold),
``derive_metrics`` (the Python wrapper exposes the four headline
floats + per-class breakdown directly off the Rust summary; we read
them out post-evaluate to keep the ``derive_metrics`` stage timer
meaningful), ``total``.

For the GT-as-DT smoke (``cityscapes_val_perfect``), the resulting
matrix is diagonal with all off-diagonal cells zero; the
``cityscapesscripts`` sibling runner (same env) produces the
bit-equal matrix and the strict-tier comparator validates equality.
"""

from __future__ import annotations

import sys

import numpy as np
import vernier
import vernier.semantic as vs

from bench.harness.timing import StageTable
from bench.runners._protocol_semantic import (
    CITYSCAPES_TRAIN_ID_NAMES,
    discover_image_pairs,
    load_label_map_png,
    parse_semantic_runner_args,
    write_semantic_outputs,
)


def main() -> int:
    args = parse_semantic_runner_args()
    stages = StageTable()

    with stages.stage("load"):
        pairs = discover_image_pairs(
            args.gt_label_maps_dir, args.dt_label_maps_dir, glob=args.png_glob
        )
        gt_label_maps: dict[int, np.ndarray] = {
            image_id: load_label_map_png(gt) for image_id, gt, _ in pairs
        }
        dt_label_maps: dict[int, np.ndarray] = {
            image_id: load_label_map_png(dt) for image_id, _, dt in pairs
        }

    n_classes: int = args.n_classes
    ignore_label: int | None = args.ignore_label

    # Build the Dataset / Predictions via the Cityscapes preset when the
    # workload's class count matches; the preset bakes the canonical
    # ``ignore_label=255`` and ``n_classes=19``. Otherwise fall back to
    # the generic ``from_arrays`` path (ADE20K / Pascal VOC will reuse
    # this runner with their own presets in S3-A/S3-B).
    if (
        n_classes == vs.CITYSCAPES_N_CLASSES
        and (ignore_label is None or ignore_label == vs.CITYSCAPES_IGNORE_LABEL)
    ):
        dataset = vs.Dataset.cityscapes({})
        # ``Dataset.cityscapes(...)`` expects PNG paths and decodes via
        # Pillow; we already decoded them in the ``load`` stage. Use
        # the lower-level constructor with the same n_classes /
        # ignore_label values to reuse the decoded arrays.
        dataset = vs.Dataset.from_arrays(
            gt_label_maps,
            n_classes=vs.CITYSCAPES_N_CLASSES,
            ignore_label=vs.CITYSCAPES_IGNORE_LABEL,
        )
    else:
        dataset = vs.Dataset.from_arrays(
            gt_label_maps, n_classes=n_classes, ignore_label=ignore_label
        )
    predictions = vs.Predictions.from_arrays(dt_label_maps)

    # Strict parity mode mirrors the cityscapesScripts oracle's NaN
    # disposition for zero-support classes — required for the strict
    # tier in the bench cell. Aligned/corrected modes diverge per
    # ADR-0028 §"Parity strategy".
    evaluator = vs.Evaluator(parity_mode="strict")

    with stages.stage("accumulate_confusion"):
        summary = evaluator.evaluate(dataset, predictions)

    with stages.stage("derive_metrics"):
        # The Rust summarize pass already populated the headline floats
        # — read them out + materialize the confusion matrix as a
        # NumPy view. Fast (the matrix is at most 150x150 cells per
        # ADR-0028).
        cm = np.asarray(summary.confusion_matrix.counts(), dtype=np.uint64)
        # Sanity: the counts() view above is fresh on every call;
        # asarray copies only if dtype mismatches. Force a copy to
        # detach from the FFI scratch buffer.
        cm = np.ascontiguousarray(cm)
        metrics: dict[str, float] = {
            "miou": float(summary.miou),
            "fwiou": float(summary.fwiou),
            "pixel_accuracy": float(summary.pixel_accuracy),
            "mean_accuracy": float(summary.mean_accuracy),
        }
        per_class = summary.per_class()
        for c in range(n_classes):
            row = per_class.get(c)
            metrics[f"iou_class_{c}"] = float(row.iou) if row is not None else 0.0

    stages.record("total", stages.total_so_far_ns())

    label_set = list(CITYSCAPES_TRAIN_ID_NAMES) if n_classes == len(CITYSCAPES_TRAIN_ID_NAMES) else []
    extra_summary: dict[str, object] = {
        "headline": {
            "miou": metrics["miou"],
            "fwiou": metrics["fwiou"],
            "pixel_accuracy": metrics["pixel_accuracy"],
            "mean_accuracy": metrics["mean_accuracy"],
        },
        "per_class": [
            {
                "class_id": c,
                "name": label_set[c] if c < len(label_set) else f"class_{c}",
                "iou": metrics[f"iou_class_{c}"],
            }
            for c in range(n_classes)
        ],
        "n_images": len(pairs),
        "n_classes": n_classes,
        "ignore_label": ignore_label,
    }
    extra_summary_path = args.confusion_output.with_name(
        args.confusion_output.stem + "_summary.json"
    )

    write_semantic_outputs(
        args=args,
        impl="vernier_semantic",
        impl_version=vernier.__version__,
        stages=stages.to_dict(),
        summary_stats=metrics,
        confusion=cm,
        extra_summary=extra_summary,
        extra_summary_path=extra_summary_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
