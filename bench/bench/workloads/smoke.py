"""Smoke workload — points at the perfect-match-segm parity fixture.

The ``_segm`` variant carries polygons so it can serve bbox, segm, and
boundary cells from a single fixture. Real workloads land in M4 under
``~/.cache/vernier-bench/``.
"""

from __future__ import annotations

from pathlib import Path


def paths(repo_root: Path) -> tuple[Path, Path]:
    base = repo_root / "tests" / "python" / "parity" / "fixtures" / "perfect_match_segm"
    return base / "gt.json", base / "dt.json"
