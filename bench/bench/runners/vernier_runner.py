"""Vernier runner — invoked as a subprocess in ``bench/envs/vernier``.

Mirrors ``tests/python/parity/harness.py:_run_vernier``: the same
``evaluate_*_grid → accumulate → summarize`` chain that the parity
harness uses, with stage timers around each call.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import vernier
import vernier._core as _vernier_core
from vernier._compat import DEFAULT_DILATION_RATIO, PARITY_STRICT

from bench.harness.timing import StageTable
from bench.runners._protocol import parse_runner_args, stat_names, write_outputs

_DEFAULT_MAX_DETS: tuple[int, ...] = (1, 10, 100)
_DEFAULT_KP_MAX_DETS: tuple[int, ...] = (20,)


def main() -> int:
    args = parse_runner_args()
    iou = args.iou_type
    stages = StageTable()

    with stages.stage("load"):
        gt_bytes = args.gt.read_bytes()
        dt_bytes = args.dt.read_bytes()

    max_dets = list(_DEFAULT_KP_MAX_DETS) if iou == "keypoints" else list(_DEFAULT_MAX_DETS)
    summarize_kwargs: dict[str, Any] = {"plan": "keypoints"} if iou == "keypoints" else {}
    md_top = max(max_dets)

    with stages.stage("evaluate"):
        if iou == "bbox":
            grid = _vernier_core.evaluate_bbox_grid(
                gt_bytes, dt_bytes, PARITY_STRICT, md_top, use_cats=True
            )
        elif iou == "segm":
            grid = _vernier_core.evaluate_segm_grid(
                gt_bytes, dt_bytes, PARITY_STRICT, md_top, use_cats=True
            )
        elif iou == "boundary":
            grid = _vernier_core.evaluate_boundary_grid(
                gt_bytes, dt_bytes, PARITY_STRICT, md_top, True, DEFAULT_DILATION_RATIO
            )
        elif iou == "keypoints":
            grid = _vernier_core.evaluate_keypoints_grid(
                gt_bytes, dt_bytes, PARITY_STRICT, md_top, True, {}
            )
        else:
            raise ValueError(f"unsupported iou_type {iou!r}")

    with stages.stage("accumulate"):
        acc = grid.accumulate(max_dets)
    with stages.stage("summarize"):
        summary = acc.summarize(max_dets, **summarize_kwargs)

    precision = np.asarray(acc.precision)
    raw_stats = np.asarray(summary.stats, dtype=np.float64)
    names = stat_names(iou)
    summary_stats: dict[str, float] = {name: float(raw_stats[i]) for i, name in enumerate(names)}

    # ``perf_counter`` at module top would also catch import time, which
    # is interpreter-state more than impl behavior.
    stages.record("total", stages.total_so_far_ns())

    write_outputs(
        args=args,
        impl="vernier",
        impl_version=vernier.__version__,
        stages=stages.to_dict(),
        summary_stats=summary_stats,
        precision_tensor=precision,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
