"""Smoke workload — points at the perfect-match parity fixture.

This intentionally couples to ``tests/python/parity/fixtures/`` because the
smoke workload is dev-only: it exists so the harness's own tests have a
known-good (gt, dt) pair without round-tripping through the network or
the val2017 cache. Real workloads land in M4 under
``~/.cache/vernier-bench/``.
"""

from __future__ import annotations

from pathlib import Path


def paths(repo_root: Path) -> tuple[Path, Path]:
    base = repo_root / "tests" / "python" / "parity" / "fixtures" / "perfect_match"
    return base / "gt.json", base / "dt.json"
