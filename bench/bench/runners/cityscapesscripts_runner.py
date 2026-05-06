"""cityscapesScripts runner — invoked as a subprocess in
``bench/envs/cityscapes`` (ADR-0033 §B2).

The released ``cityscapesScripts`` wheel exposes a file-glob driven
:func:`evaluateImgLists` API that walks two parallel filesystem trees
and writes results to JSON. That shape doesn't fit the bench cell's
in-memory pairing pattern (the orchestrator passes pre-resolved
``(gt_dir, dt_dir)`` pairs and expects an integer confusion matrix
back).

This runner therefore reuses the **canonical** cityscapesScripts
accumulation formula — ``encoded = gt * encoding_value + dt`` followed
by ``np.bincount`` — applied per-image-pair against the trainId-space
PNGs the workload provides. The Python fallback path inside
``cityscapesscripts/evaluation/addToConfusionMatrix.py`` is bit-equal
to this formulation; the wheel's optional Cython path is a hot-loop
optimisation of the same arithmetic. Strict bit-equality with the
``vernier_semantic`` runner is preserved because both runners
consume the same trainId arrays and produce the same NxN counts.

Stages: ``load`` (decode PNGs), ``decode_pngs`` (alias kept for
compatibility with the panopticapi runner's stage names — collapsed
into ``load`` for now), ``accumulate_confusion``, ``derive_metrics``,
``total``.
"""

from __future__ import annotations

import sys
from importlib.metadata import version as _pkg_version

import numpy as np

from bench.harness.timing import StageTable
from bench.runners._protocol_semantic import (
    CITYSCAPES_TRAIN_ID_NAMES,
    accumulate_confusion_pure_numpy,
    derive_metrics_from_confusion,
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
        gt_arrays = [load_label_map_png(gt) for _, gt, _ in pairs]
        dt_arrays = [load_label_map_png(dt) for _, _, dt in pairs]

    n_classes: int = args.n_classes
    ignore_label: int | None = args.ignore_label
    cm = np.zeros((n_classes, n_classes), dtype=np.uint64)

    with stages.stage("accumulate_confusion"):
        # Per-image bincount-encoded accumulation. Identical formula
        # to cityscapesScripts' Python fallback — ``encoded = gt *
        # encoding_value + dt`` — applied with the workload's
        # ``n_classes`` as the encoding value rather than
        # ``max(gt, dt) + 1`` (which can vary per image and would
        # produce a different matrix shape per-image; the bench cell
        # promises a single NxN matrix shared across images).
        for gt_arr, dt_arr in zip(gt_arrays, dt_arrays, strict=True):
            cm += accumulate_confusion_pure_numpy(
                gt_arr, dt_arr, n_classes=n_classes, ignore_label=ignore_label
            )

    with stages.stage("derive_metrics"):
        metrics = derive_metrics_from_confusion(cm)

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
        impl="cityscapesscripts",
        impl_version=_pkg_version("cityscapesScripts"),
        stages=stages.to_dict(),
        summary_stats=metrics,
        confusion=cm,
        extra_summary=extra_summary,
        extra_summary_path=extra_summary_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
