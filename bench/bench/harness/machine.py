"""Machine fingerprint and git-sha helpers.

M1 stub. M5 replaces ``fingerprint()`` with the real
``sha256(cpu_model + n_cores + total_ram + os_release + glibc_version)[:12]``
per ADR-0017 §"Result store". Path-building code already calls into here
so M5 is a function-body change, not a path change.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def fingerprint() -> str:
    """First 12 chars of a stable per-machine hash. M1: literal stub."""
    return "dev-unfp-m1"


def git_sha(repo_root: Path) -> str:
    """Short git sha for the current HEAD of ``repo_root``. Empty string if
    this isn't a git checkout — the harness still runs, just under a
    placeholder bucket."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return out.decode().strip()
