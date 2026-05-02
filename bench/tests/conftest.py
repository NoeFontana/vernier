"""Shared fixtures for bench tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.harness.matrix import env_dir

BENCH_ROOT = Path(__file__).resolve().parent.parent


def skip_if_no_env(impl: str) -> None:
    """Skip the calling test if ``bench/envs/<impl>/.venv`` is missing."""
    if not (env_dir(BENCH_ROOT, impl) / ".venv").exists():
        pytest.skip(f"bench/envs/{impl}/.venv missing; run `just bench-sync` first")
