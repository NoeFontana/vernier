"""Vernier runner — invoked as a subprocess in ``bench/envs/vernier``.

Mirrors ``tests/python/parity/harness.py:_run_vernier`` (lines 141-181):
the same ``evaluate_*_grid → accumulate → summarize`` chain that the
parity harness uses. Stage timers wrap each call so the orchestrator
can attribute time to load / evaluate / accumulate / summarize.
"""

from __future__ import annotations

import sys

import numpy as np
import vernier
import vernier._core as _vernier_core

from bench.harness.timing import StageTable
from bench.runners._protocol import parse_runner_args, write_outputs

_DEFAULT_MAX_DETS: tuple[int, ...] = (1, 10, 100)
_DEFAULT_KP_MAX_DETS: tuple[int, ...] = (20,)
# Strict tier — the smoke workload runs the same parity harness routes.
_PARITY_MODE = "strict"

# Conventional COCO summary names. Order matches pycocotools'
# ``cocoeval.summarize`` output for bbox/segm and keypoints respectively.
_BBOX_STAT_NAMES: tuple[str, ...] = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
)
_KP_STAT_NAMES: tuple[str, ...] = (
    "AP",
    "AP50",
    "AP75",
    "AP_medium",
    "AP_large",
    "AR",
    "AR50",
    "AR75",
    "AR_medium",
    "AR_large",
)


def _stat_names(iou_type: str) -> tuple[str, ...]:
    if iou_type == "keypoints":
        return _KP_STAT_NAMES
    return _BBOX_STAT_NAMES


def main() -> int:
    args = parse_runner_args()
    if args.iou_type == "boundary":
        # Vernier's boundary surface lands here in M2's parity story; M1
        # only handles bbox/segm/keypoints to keep scope honest.
        print("vernier_runner: boundary IoU is not wired in M1", file=sys.stderr)
        return 2

    stages = StageTable()

    with stages.stage("load"):
        gt_bytes = args.gt.read_bytes()
        dt_bytes = args.dt.read_bytes()

    if args.iou_type == "keypoints":
        max_dets = list(_DEFAULT_KP_MAX_DETS)
        with stages.stage("evaluate"):
            grid = _vernier_core.evaluate_keypoints_grid(
                gt_bytes, dt_bytes, _PARITY_MODE, max(max_dets), True, {}
            )
        with stages.stage("accumulate"):
            acc = grid.accumulate(max_dets)
        with stages.stage("summarize"):
            summary = acc.summarize(max_dets, plan="keypoints")
    else:
        max_dets = list(_DEFAULT_MAX_DETS)
        grid_fn = (
            _vernier_core.evaluate_segm_grid
            if args.iou_type == "segm"
            else _vernier_core.evaluate_bbox_grid
        )
        with stages.stage("evaluate"):
            grid = grid_fn(gt_bytes, dt_bytes, _PARITY_MODE, max(max_dets), use_cats=True)
        with stages.stage("accumulate"):
            acc = grid.accumulate(max_dets)
        with stages.stage("summarize"):
            summary = acc.summarize(max_dets)

    precision = np.asarray(acc.precision).copy()
    raw_stats = np.asarray(summary.stats, dtype=np.float64).copy()
    names = _stat_names(args.iou_type)
    summary_stats: dict[str, float] = {
        name: float(raw_stats[i]) for i, name in enumerate(names)
    }

    # Total spans the bracketed work — perf_counter at module top would
    # also catch import time, which is interpreter-state more than impl
    # behavior. ``StageTable.total_so_far_ns()`` is the honest accounting.
    total_ns = stages.total_so_far_ns()
    stages.record("total", total_ns)

    impl_version = getattr(vernier, "__version__", "unknown")

    write_outputs(
        args=args,
        impl="vernier",
        impl_version=str(impl_version),
        stages=stages.to_dict(),
        summary_stats=summary_stats,
        precision_tensor=precision,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
