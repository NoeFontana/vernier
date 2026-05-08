"""vernier_panoptic runner — invoked as a subprocess in
``bench/envs/panopticapi`` (ADR-0033 §B1).

Streams per-image GT/DT pairs through
:meth:`vernier.panoptic.Evaluator.background`: PNG decode runs on the
main thread while the Rust PQ kernel folds on a worker thread, with
bounded queue depth. Constant RSS by construction — only
``queue_capacity`` decoded label maps are in flight at any one time,
versus the prior eager-decode path that materialized all 5000 GT +
5000 DT uint32 arrays before invoking the kernel (~20 GiB at val2017
scale).

The kernel itself is unchanged; strict-tier parity vs panopticapi
still passes.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import vernier

from bench.harness.parity import PanopticSnapshot
from bench.harness.timing import StageTable
from bench.runners._protocol import (
    parse_panoptic_runner_args,
    per_class_uint64_table,
    write_panoptic_outputs,
)


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
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = [
            (gt_ann, pred_by_image[gt_ann["image_id"]])
            for gt_ann in gt["annotations"]
            if gt_ann["image_id"] in pred_by_image
        ]
        cats_bytes = json.dumps(list(categories)).encode()

    # The Background ``with`` block covers worker-thread shutdown if
    # ``submit_png`` raises; otherwise the worker would leak past the
    # runner's lifetime. parity_mode="strict" reproduces panopticapi
    # bit-exactly under the strict-tier comparator.
    #
    # ``submit_png`` hands raw PNG bytes to the Rust kernel, which
    # fuses libpng decode + RGB→id + (DT side) S3 area marginals in a
    # single pass. The prior path round-tripped through Pillow +
    # ``np.array(..., dtype=np.uint32)`` + Python-level ``R + 256·G +
    # 256²·B`` arithmetic on the main thread, which dominated wall
    # time on val2017 perfect-DT.
    with (
        stages.stage("stream_pq"),
        vernier.panoptic.Evaluator(parity_mode="strict", things_stuff_split=True).background(
            cats_bytes
        ) as ev,
    ):
        for gt_ann, dt_ann in pairs:
            gt_bytes = (args.gt_png_dir / gt_ann["file_name"]).read_bytes()
            dt_bytes = (args.dt_png_dir / dt_ann["file_name"]).read_bytes()
            ev.submit_png(
                int(gt_ann["image_id"]),
                gt_bytes,
                json.dumps(gt_ann["segments_info"]).encode(),
                dt_bytes,
                json.dumps(dt_ann["segments_info"]).encode(),
            )
        summary = ev.finalize()

    with stages.stage("aggregate"):
        snap = _summary_to_snapshot(summary)
        per_class_array = per_class_uint64_table(snap.per_class, columns=("pq", "sq", "rq"))
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
