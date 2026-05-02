"""Shared CLI argspec and output helpers for all impl runners.

Every runner under ``bench.runners.*_runner`` accepts the same arguments
and writes the same JSON shape (``RunnerRepOutput``) plus a ``.npy``
precision tensor.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
from pathlib import Path
from typing import Any, get_args

import numpy as np

from bench.harness.schema import BenchWarning, IouType, RunnerRepOutput, StageTimings
from bench.harness.timing import StageTable

BBOX_STAT_NAMES: tuple[str, ...] = (
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
KP_STAT_NAMES: tuple[str, ...] = (
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


def stat_names(iou_type: IouType) -> tuple[str, ...]:
    return KP_STAT_NAMES if iou_type == "keypoints" else BBOX_STAT_NAMES


def parse_runner_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--dt", type=Path, required=True)
    p.add_argument("--iou-type", choices=list(get_args(IouType)), required=True)
    p.add_argument("--workload-id", type=str, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tensor-output", type=Path, required=True)
    return p.parse_args()


def file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_outputs(
    *,
    args: argparse.Namespace,
    impl: str,
    impl_version: str,
    stages: dict[str, StageTimings],
    summary_stats: dict[str, float],
    precision_tensor: np.ndarray,
    warnings: list[BenchWarning] | None = None,
) -> None:
    """Persist the tensor and the result JSON. The orchestrator re-checks
    the tensor sha256 after copying to the canonical result path."""
    tensor_path: Path = args.tensor_output
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(tensor_path, precision_tensor, allow_pickle=False)

    output = RunnerRepOutput(
        impl=impl,
        impl_version=impl_version,
        iou_type=args.iou_type,
        workload_id=args.workload_id,
        stages=stages,
        summary_stats=summary_stats,
        tensor_sha256=file_sha256(tensor_path),
        warnings=list(warnings or []),
    )
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output.model_dump_json(indent=2))


def run_cocoeval_pipeline(
    *,
    args: argparse.Namespace,
    impl: str,
    impl_version: str,
    coco_cls: type[Any],
    cocoeval_cls: type[Any],
) -> None:
    """Run the standard COCO/COCOeval load → evaluate → accumulate →
    summarize chain inside a stdout redirect, then persist outputs.

    Used by every runner that wraps a pycocotools-shaped surface
    (pycocotools, faster-coco-eval, boundary-iou-api). pycocotools and
    its drop-ins all print progress from inside ``COCO`` /
    ``loadRes`` / ``summarize``; one outer redirect is cleaner than
    sprinkling them.
    """
    stages = StageTable()
    with contextlib.redirect_stdout(io.StringIO()):
        with stages.stage("load"):
            gt = coco_cls(str(args.gt))
            dt = gt.loadRes(str(args.dt))
            cocoeval = cocoeval_cls(gt, dt, iouType=args.iou_type)
        with stages.stage("evaluate"):
            cocoeval.evaluate()
        with stages.stage("accumulate"):
            cocoeval.accumulate()
        with stages.stage("summarize"):
            cocoeval.summarize()

    precision = np.asarray(cocoeval.eval["precision"])
    raw_stats = np.asarray(cocoeval.stats, dtype=np.float64)
    names = stat_names(args.iou_type)
    summary_stats: dict[str, float] = {name: float(raw_stats[i]) for i, name in enumerate(names)}

    stages.record("total", stages.total_so_far_ns())

    write_outputs(
        args=args,
        impl=impl,
        impl_version=impl_version,
        stages=stages.to_dict(),
        summary_stats=summary_stats,
        precision_tensor=precision,
    )
