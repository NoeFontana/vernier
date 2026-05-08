"""vernier_semantic runner — invoked as a subprocess in
``bench/envs/vernier`` (ADR-0033 §B2).

Streams per-image GT/DT label-map PNG pairs through
:meth:`vernier.semantic.Evaluator.stream`: PNG decode runs on the main
thread and the Rust confusion-matrix fold runs synchronously per
image (the matrix is too cheap to warrant a worker thread; the
`background()` path is the option for users who want it). RSS stays
constant in the image count by construction — only one decoded
label-map array is in flight at a time.

Emits a :class:`bench.harness.parity.SemanticSnapshot` JSON + a
per-class ``.npy`` table mirroring the panoptic two-artifact bundle.
There is no oracle today (S3-B / mmsegmentation env pending), so the
parity comparator is a no-op for now; the cell still gates on the
runner contract (per-rep snapshot + per-class sha bit-equality).
"""

from __future__ import annotations

import sys
from pathlib import Path

import vernier

from bench.harness.parity import SemanticSnapshot
from bench.harness.timing import StageTable
from bench.runners._protocol import (
    parse_semantic_runner_args,
    per_class_uint64_table,
    write_semantic_outputs,
)


def _scan_label_map_dir(directory: Path) -> dict[int, Path]:
    """Index PNGs in ``directory`` by integer image_id parsed from the
    stem. Filenames must match ``<int>.png``.
    """
    out: dict[int, Path] = {}
    for entry in sorted(directory.iterdir()):
        if entry.suffix.lower() != ".png":
            continue
        try:
            image_id = int(entry.stem)
        except ValueError as e:
            raise ValueError(
                f"semantic runner expects label-map filenames of the form "
                f"'<int>.png'; got {entry.name!r} under {directory!s}."
            ) from e
        out[image_id] = entry
    return out


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


def main() -> int:
    args = parse_semantic_runner_args()
    stages = StageTable()

    with stages.stage("load"):
        gt_paths = _scan_label_map_dir(args.gt_label_map_dir)
        dt_paths = _scan_label_map_dir(args.dt_label_map_dir)
        common_ids = sorted(set(gt_paths) & set(dt_paths))
        if not common_ids:
            raise RuntimeError(
                f"semantic runner found no overlapping image_ids between "
                f"{args.gt_label_map_dir} and {args.dt_label_map_dir}"
            )
        ignore_label = args.ignore_label if args.ignore_label >= 0 else None

    with stages.stage("stream_miou"):
        # parity_mode="strict" pins bit-exact reproduction of
        # mmsegmentation's mIoU once the S3-B oracle lands. RSS stays
        # bounded by one decoded label map at a time.
        ev = vernier.semantic.Evaluator(parity_mode="strict").stream(
            n_classes=args.n_classes, ignore_label=ignore_label
        )
        for image_id in common_ids:
            gt_arr = vernier.semantic.decode_label_map_png(gt_paths[image_id])
            dt_arr = vernier.semantic.decode_label_map_png(dt_paths[image_id])
            ev.update(image_id, gt_arr, dt_arr)
        summary = ev.finalize()

    with stages.stage("aggregate"):
        snap = _summary_to_snapshot(summary)
        per_class_array = per_class_uint64_table(
            snap.per_class, columns=("iou", "accuracy", "precision")
        )
        snap_json = snap.model_dump_json().encode()

    stages.record("total", stages.total_so_far_ns())

    write_semantic_outputs(
        args=args,
        impl="vernier_semantic",
        impl_version=vernier.__version__,
        stages=stages.to_dict(),
        snapshot_json_bytes=snap_json,
        per_class_array=per_class_array,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
