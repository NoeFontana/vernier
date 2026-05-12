"""Vernier LVIS runner — invoked as a subprocess in ``bench/envs/vernier``.

Mirrors ``tests/python/parity_lvis/harness.py:_vernier_snapshot``: the
``CocoDataset.from_lvis_json → evaluate_bbox_grid_with_dataset →
accumulate → summarize_lvis`` chain, with stage timers around each
call. The parsed-once dataset path is mandatory for LVIS — the
JSON-bytes grid path strips federated metadata at GT load (it goes
through ``CocoDataset.from_json_bytes``), and the orchestrator's
AA3/AA4 federated branches don't fire without the
``not_exhaustive_category_ids`` / ``neg_category_ids`` per-image
filters.

LVIS canonical ``max_dets`` is 300 (AC1); the 13-entry plan keys
(``AP``, ``AP50``, ``AP75``, ``APs``, ``APm``, ``APl``, ``APr``,
``APc``, ``APf``, ``AR@300``, ``ARs@300``, ``ARm@300``, ``ARl@300``)
are produced by the ``lvis_default()`` summarize plan and pinned in
``crates/vernier-core/src/summarize.rs``.

The (T, R, K, A) precision tensor — no M-axis (AF5), K=1203 on full
val — is the strict-tier parity carrier vs the lvis-api oracle. The
13-entry stats live on ``RunnerRepOutput.summary_stats`` for the
report layer.
"""

from __future__ import annotations

import sys

import numpy as np
import vernier
from vernier._core import evaluate_bbox_grid_with_dataset
from vernier._types import PARITY_STRICT
from vernier.instance import CocoDataset

from bench.harness.timing import StageTable
from bench.runners._protocol import (
    lvis_stat_names,
    parse_lvis_runner_args,
    write_lvis_outputs,
)


def main() -> int:
    args = parse_lvis_runner_args()
    iou = args.iou_type
    if iou != "bbox":
        # Mirrors the matrix entry in ``bench/harness/matrix.py`` —
        # vernier_lvis is bbox-only until ``evaluate_segm_grid_with_dataset``
        # lands. Fail loud rather than silently degrade to the JSON-bytes
        # path that strips federated metadata.
        raise ValueError(
            f"vernier_lvis runner: iou_type={iou!r} not supported; "
            f"only 'bbox' is wired (segm needs evaluate_segm_grid_with_dataset)"
        )
    max_dets: int = int(args.max_dets)
    stages = StageTable()

    with stages.stage("load"):
        gt_bytes = args.gt.read_bytes()
        dt_bytes = args.dt.read_bytes()
        gt_dataset = CocoDataset.from_lvis_json(gt_bytes)

    with stages.stage("evaluate"):
        grid = evaluate_bbox_grid_with_dataset(
            gt_dataset,
            dt_bytes,
            PARITY_STRICT,
            max_dets,
            True,
        )

    with stages.stage("accumulate"):
        accum = grid.accumulate([max_dets])

    with stages.stage("summarize"):
        summary = accum.summarize_lvis(gt_dataset, [max_dets])

    # accum.precision is (T, R, K, A, M) where M=1; drop the trailing
    # axis to (T, R, K, A) — matches the lvis-api oracle's
    # ``ev.eval["precision"]`` shape and the parity contract (AF5).
    # The FFI getter already returns a fresh owned f64 array, so
    # ``asarray`` is sufficient; ``ascontiguousarray`` after the squeeze
    # gives ``np.save`` a contiguous buffer without an extra copy of
    # the 5D parent.
    precision = np.asarray(accum.precision)
    if precision.ndim == 5:
        if precision.shape[-1] != 1:
            raise AssertionError(
                f"vernier precision M-axis must be 1 at max_dets={max_dets}; "
                f"got {precision.shape}"
            )
        precision = np.ascontiguousarray(precision[..., 0])

    keys = lvis_stat_names(max_dets)
    raw_stats = [float(line) for line in summary.stats]
    if len(raw_stats) != len(keys):
        raise AssertionError(
            f"vernier lvis_default returned {len(raw_stats)} stats; "
            f"expected {len(keys)} (AF1)"
        )
    summary_stats: dict[str, float] = dict(zip(keys, raw_stats, strict=True))

    stages.record("total", stages.total_so_far_ns())

    write_lvis_outputs(
        args=args,
        impl="vernier_lvis",
        impl_version=vernier.__version__,
        stages=stages.to_dict(),
        summary_stats=summary_stats,
        precision_tensor=precision,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
