"""Shared CLI argspec and output helpers for all impl runners.

Every runner under ``bench.runners.*_runner`` accepts the same arguments
and writes the same JSON shape (``RunnerRepOutput``) plus a ``.npy``
precision tensor. Adding a new impl is one ~150-line module.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from bench.harness.schema import BenchWarning, RunnerRepOutput, StageTimings


def parse_runner_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--dt", type=Path, required=True)
    p.add_argument(
        "--iou-type",
        choices=["bbox", "segm", "keypoints", "boundary"],
        required=True,
    )
    p.add_argument("--workload-id", type=str, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tensor-output", type=Path, required=True)
    return p.parse_args()


def _file_sha256(path: Path) -> str:
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
        tensor_sha256=_file_sha256(tensor_path),
        warnings=list(warnings or []),
    )
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output.model_dump_json(indent=2))
