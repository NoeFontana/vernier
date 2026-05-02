"""Shared fixtures for bench tests."""

from __future__ import annotations

import pytest

from bench.harness.matrix import env_dir
from bench.harness.paths import BENCH_ROOT


def skip_if_no_env(impl: str) -> None:
    """Skip the calling test if ``bench/envs/<impl>/.venv`` is missing."""
    if not (env_dir(BENCH_ROOT, impl) / ".venv").exists():
        pytest.skip(f"bench/envs/{impl}/.venv missing; run `just bench-sync` first")
