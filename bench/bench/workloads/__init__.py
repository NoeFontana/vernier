"""Workload registry. Each workload resolves to (gt_path, dt_path) plus
the set of IoU types it can serve.

M4 adds ``coco_val2017_jittered_seed*`` and ``synthetic:...``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bench.harness.schema import IouType
from bench.workloads import smoke


@dataclass(frozen=True)
class Workload:
    workload_id: str
    gt_path: Path
    dt_path: Path
    supported_iou_types: frozenset[IouType]


def resolve(workload_name: str, repo_root: Path) -> Workload:
    if workload_name == "smoke":
        gt, dt = smoke.paths(repo_root)
        return Workload(
            workload_id="smoke_perfect_match_segm",
            gt_path=gt,
            dt_path=dt,
            supported_iou_types=frozenset({"bbox", "segm", "boundary"}),
        )
    raise ValueError(f"unknown workload {workload_name!r}; M1/M2 only knows 'smoke'")
