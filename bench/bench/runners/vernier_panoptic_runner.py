"""vernier_panoptic runner — invoked as a subprocess in
``bench/envs/panopticapi`` (ADR-0033 §B1).

Mirrors :func:`tests/python/parity_panoptic/harness.py:_vernier_snapshot`:
load the GT / DT PNG pairs into uint32 ndarrays, then call
``vernier.panoptic.Evaluator(parity_mode=...).evaluate(dataset,
predictions)``. The bench runner adds stage timers and emits a
:class:`bench.harness.parity.PanopticSnapshot` JSON instead of the
harness's frozen-dataclass return.

This runner shares the panopticapi env so the bench comparator can
run both impls in identical Python state. ``vernier`` itself is
installed via the vernier path-dep at ``bench/envs/panopticapi/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import vernier
from PIL import Image as PILImage

from bench.harness.parity import PanopticSnapshot
from bench.harness.timing import StageTable
from bench.runners._protocol import parse_panoptic_runner_args, write_panoptic_outputs


def _decode_panoptic_png_to_uint32(path: Path) -> np.ndarray:
    """Decode a panoptic PNG to a uint32 label map via PIL + rgb2id.

    Same decode as panopticapi's evaluation.py:86-89; vernier consumes
    the resulting uint32 ndarrays directly via
    :meth:`vernier.panoptic.Dataset.from_arrays`.
    """
    rgb = np.array(PILImage.open(path), dtype=np.uint32)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"non-RGB panoptic PNG: {path}")
    return rgb[:, :, 0] + 256 * rgb[:, :, 1] + 256 * 256 * rgb[:, :, 2]


def _summary_to_snapshot(summary: vernier.panoptic.Summary) -> PanopticSnapshot:
    """Project a :class:`vernier.panoptic.Summary` into a
    :class:`PanopticSnapshot`. Mirrors
    :func:`tests/python/parity_panoptic/harness.py:summary_to_snapshot`:
    coerces the ``Option<f64>`` / ``Option<usize>`` Things/Stuff
    bucket fields to ``0.0`` / ``0`` for the empty-bucket case so
    both impls emit the same JSON shape regardless of fixture.
    """
    return PanopticSnapshot(
        pq=float(summary.pq),
        sq=float(summary.sq),
        rq=float(summary.rq),
        n=int(summary.n),
        pq_things=float(summary.pq_things) if summary.pq_things is not None else 0.0,
        sq_things=float(summary.sq_things) if summary.sq_things is not None else 0.0,
        rq_things=float(summary.rq_things) if summary.rq_things is not None else 0.0,
        n_things=int(summary.n_things) if summary.n_things is not None else 0,
        pq_stuff=float(summary.pq_stuff) if summary.pq_stuff is not None else 0.0,
        sq_stuff=float(summary.sq_stuff) if summary.sq_stuff is not None else 0.0,
        rq_stuff=float(summary.rq_stuff) if summary.rq_stuff is not None else 0.0,
        n_stuff=int(summary.n_stuff) if summary.n_stuff is not None else 0,
        per_class={
            str(int(cat)): {"pq": float(row.pq), "sq": float(row.sq), "rq": float(row.rq)}
            for cat, row in summary.per_class().items()
        },
    )


def _per_class_table(snap: PanopticSnapshot) -> np.ndarray:
    """Build the per-class N×3 ``[pq, sq, rq]`` table.

    Same shape as the panopticapi runner's table (uint64 view of
    f64) so the per-class npy artifacts diff bit-equally between
    impls under the strict tier.
    """
    rows: list[tuple[int, float, float, float]] = sorted(
        ((int(k), v["pq"], v["sq"], v["rq"]) for k, v in snap.per_class.items()),
        key=lambda r: r[0],
    )
    if not rows:
        return np.zeros((0, 3), dtype=np.uint64)
    arr = np.array([[r[1], r[2], r[3]] for r in rows], dtype=np.float64)
    return arr.view(np.uint64).reshape(arr.shape)


def _build_label_maps(
    annotations: list[dict[str, Any]],
    png_dir: Path,
) -> dict[int, np.ndarray]:
    return {
        int(ann["image_id"]): _decode_panoptic_png_to_uint32(png_dir / ann["file_name"])
        for ann in annotations
    }


def main() -> int:
    args = parse_panoptic_runner_args()
    stages = StageTable()

    with stages.stage("load"):
        with args.gt_json.open() as f:
            gt = json.load(f)
        with args.dt_json.open() as f:
            dt = json.load(f)
        with args.categories_json.open() as f:
            categories = json.load(f)

        # Pair GT and DT annotations on image_id so every kept entry
        # has both a GT label-map and a DT label-map; matches the
        # oracle's pairing rule.
        pred_by_image = {a["image_id"]: a for a in dt["annotations"]}
        gt_anns: list[dict[str, Any]] = []
        dt_anns: list[dict[str, Any]] = []
        for gt_ann in gt["annotations"]:
            if gt_ann["image_id"] in pred_by_image:
                gt_anns.append(gt_ann)
                dt_anns.append(pred_by_image[gt_ann["image_id"]])

    with stages.stage("decode_pngs"):
        gt_label_maps = _build_label_maps(gt_anns, args.gt_png_dir)
        dt_label_maps = _build_label_maps(dt_anns, args.dt_png_dir)
        gt_segs = {ann["image_id"]: ann["segments_info"] for ann in gt_anns}
        dt_segs = {ann["image_id"]: ann["segments_info"] for ann in dt_anns}

        gt_segs_bytes = json.dumps({str(k): v for k, v in gt_segs.items()}).encode()
        dt_segs_bytes = json.dumps({str(k): v for k, v in dt_segs.items()}).encode()
        cats_bytes = json.dumps(list(categories)).encode()

        gt_dataset = vernier.panoptic.Dataset.from_arrays(
            gt_label_maps, gt_segs_bytes, cats_bytes
        )
        dt_predictions = vernier.panoptic.Predictions.from_arrays(
            dt_label_maps, dt_segs_bytes
        )

    with stages.stage("pq_compute"):
        # parity_mode="strict" so this runner reproduces panopticapi
        # bit-exactly under the strict-tier comparator. Aligned-tier
        # comparisons would relax to ``parity_mode="corrected"`` —
        # left as a Stage 3 follow-up (the strict run is the canonical
        # bench cell here).
        summary = vernier.panoptic.Evaluator(
            parity_mode="strict", things_stuff_split=True
        ).evaluate(gt_dataset, dt_predictions)

    with stages.stage("aggregate"):
        snap = _summary_to_snapshot(summary)
        per_class_array = _per_class_table(snap)
        snap_json = snap.model_dump_json().encode()

    stages.record("total", stages.total_so_far_ns())

    write_panoptic_outputs(
        args=args,
        impl="vernier_panoptic",
        impl_version=vernier.__version__,
        stages=stages.to_dict(),
        snapshot_json_bytes=snap_json,
        per_class_array=per_class_array,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
