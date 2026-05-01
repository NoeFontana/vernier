"""Workload registry. Each workload resolves to a (gt_path, dt_path) pair.

M1: only ``smoke``. M4 adds ``coco_val2017_jittered_seed*`` and
``synthetic:...``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bench.workloads import smoke


@dataclass(frozen=True)
class Workload:
    workload_id: str
    gt_path: Path
    dt_path: Path


def resolve(workload_name: str, repo_root: Path) -> Workload:
    if workload_name == "smoke":
        gt, dt = smoke.paths(repo_root)
        return Workload(workload_id="smoke_perfect_match", gt_path=gt, dt_path=dt)
    raise ValueError(
        f"unknown workload {workload_name!r}; M1 only knows 'smoke'"
    )
