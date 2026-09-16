"""Objects365 v2 val — the bench's scale workload.

GT is the consortium's val annotation JSON (CC BY 4.0), fetched and
sha256-pinned by the canonical :mod:`objects365_val_cache` package; the
bench keeps its copy under ``~/.cache/vernier-bench/objects365_val`` like
the COCO GT. Detections are the shared jitter generator run over the O365
GT (:func:`bench.workloads.jittered_predictions.objects365_dt_path`), so
the workload is bbox-only: Objects365 has no masks or keypoints.

Workload ids: ``objects365_val_jittered_seed<N>``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from objects365_val_cache import ensure_gt

from bench.harness.paths import bench_cache_root

_JITTERED_RE = re.compile(r"^objects365_val_jittered_seed(\d+)$")


def gt_path() -> Path:
    """Return a verified path to the Objects365 val GT JSON.

    ``VERNIER_OBJECTS365_GT_PATH`` short-circuits the cache (offline
    tests, a pre-populated copy elsewhere); otherwise the bench cache is
    populated and sha256-verified via :func:`objects365_val_cache.ensure_gt`.
    """
    env_override = os.environ.get("VERNIER_OBJECTS365_GT_PATH")
    if env_override:
        candidate = Path(env_override)
        if candidate.exists():
            return candidate
    return ensure_gt(cache=bench_cache_root() / "objects365_val")


def jittered_workload_id(seed: int) -> str:
    return f"objects365_val_jittered_seed{seed}"


def parse_jittered_seed(workload_name: str) -> int | None:
    """Parse the seed from ``objects365_val_jittered_seed<N>``, or ``None``."""
    m = _JITTERED_RE.match(workload_name)
    return int(m.group(1)) if m else None


__all__ = ["gt_path", "jittered_workload_id", "parse_jittered_seed"]
