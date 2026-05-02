"""Shared fixtures for bench tests."""

from __future__ import annotations

import numpy as np
import pytest

from bench.harness.matrix import env_dir
from bench.harness.paths import BENCH_ROOT


def skip_if_no_env(impl: str) -> None:
    """Skip the calling test if ``bench/envs/<impl>/.venv`` is missing."""
    if not (env_dir(BENCH_ROOT, impl) / ".venv").exists():
        pytest.skip(f"bench/envs/{impl}/.venv missing; run `just bench-sync` first")


@pytest.fixture
def zero_tensor() -> np.ndarray:
    """A precision tensor of zeros with the canonical (T, R, K, A, M) shape."""
    return np.zeros((10, 101, 1, 4, 3), dtype=np.float64)
