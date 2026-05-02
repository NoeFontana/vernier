"""Filesystem anchors for the harness — single source of truth.

Resolved from the ``bench`` package's location, so any in-tree caller
(CLI, runner subprocess, pytest conftest) gets the same answers without
counting ``parent`` levels from its own ``__file__``.

Editable-install only. ``vernier-bench`` is a development-only package
(per ADR-0017): the ``envs/`` and ``results/`` siblings live next to the
source tree, not inside any installed wheel.
"""

from __future__ import annotations

import os
from pathlib import Path

import bench

BENCH_ROOT: Path = Path(bench.__file__).resolve().parent.parent
REPO_ROOT: Path = BENCH_ROOT.parent


def bench_cache_root() -> Path:
    """Root of the user-level cache for downloaded GTs and generated DTs.

    Honours ``VERNIER_BENCH_CACHE`` so tests and operator setups can
    redirect without touching ``$HOME``.
    """
    override = os.environ.get("VERNIER_BENCH_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "vernier-bench"
