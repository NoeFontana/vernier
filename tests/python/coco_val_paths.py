"""Shared env-var contract + cache layout for the COCO val2017 parity smoke.

`tests/python/parity/test_coco_val.py` (bbox + segm) and
`tests/python/parity_boundary/test_coco_val.py` (boundary) both gate on
the same `VERNIER_COCO_*_PATH` env vars and the same `tools/fetch-coco-val.sh`
cache layout — the contract is single-sourced in
`docs/engineering/coco-val-parity.md`. Centralizing the helpers here
makes a fetcher-script rename a one-touch change instead of a
two-test-tree drift hazard.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

GT_ENV = "VERNIER_COCO_GT_PATH"
DT_ENV = "VERNIER_COCO_DT_PATH"
DT_SEGM_ENV = "VERNIER_COCO_DT_SEGM_PATH"
CACHE_ENV = "VERNIER_COCO_CACHE"

# Mirrors tools/fetch-coco-val.sh's default and its VERNIER_COCO_CACHE
# override, so perfect-DT tests find the artifacts the helper wrote.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "coco-val2017"


def cache_dir() -> Path:
    override = os.environ.get(CACHE_ENV)
    return Path(override).expanduser() if override else _DEFAULT_CACHE_DIR


def require_env_path(env: str) -> Path:
    value = os.environ.get(env)
    if not value:
        pytest.skip(f"{env} is unset; see docs/engineering/coco-val-parity.md")
    path = Path(value).expanduser()
    if not path.is_file():
        pytest.skip(f"{env}={value!r} does not point to a file")
    return path


def require_perfect_dt_artifacts(dt_filename: str) -> tuple[Path, Path]:
    """Locate the cached GT and a perfect-DT JSON, or skip the test.

    Both artifacts are produced by `tools/fetch-coco-val.sh`. Skipping
    when either is missing keeps `just test` green on a clean checkout.
    """
    cache = cache_dir()
    gt = cache / "instances_val2017.json"
    dt = cache / dt_filename
    if not gt.is_file() or not dt.is_file():
        pytest.skip(
            f"run ./tools/fetch-coco-val.sh to populate {cache}; "
            "see docs/engineering/coco-val-parity.md"
        )
    return gt, dt
