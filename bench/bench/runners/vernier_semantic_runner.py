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
        )

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
