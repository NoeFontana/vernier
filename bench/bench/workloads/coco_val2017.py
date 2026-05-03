"""COCO val2017 GT — download, sha256-verify, cache locally.

Constants and the fetch+verify flow live in the canonical
:mod:`coco_val_cache` package (a `[tool.uv.sources]` path dep at
``tools/coco_val_cache/``). This module provides the bench-cache-rooted
wrapper: bench keeps its own cache at ``~/.cache/vernier-bench/`` so
``just test-coco-val`` and ``vernier-bench run`` can't fight over the
same file, but the canonical pin of "what file lives there" is shared.

``VERNIER_COCO_GT_PATH`` (the convention from
``docs/engineering/coco-val-parity.md``) is honoured as a preverified
fallback so users who already populated the parity cache don't pay the
download twice.
"""

from __future__ import annotations

import os
from pathlib import Path

from coco_val_cache import GT_FILENAME, GT_SHA256, ensure_gt, file_sha256

from bench.harness.paths import bench_cache_root

# Re-exported for callers that imported it from this module historically
# (e.g. the network-gated bench test suite).
EXPECTED_SHA256 = GT_SHA256


def gt_path() -> Path:
    """Return a verified path to the COCO val2017 GT JSON.

    Honors ``VERNIER_COCO_GT_PATH`` first (so users with a populated
    parity cache don't pay the download twice), then falls back to the
    bench cache. Raises ``RuntimeError`` on a sha256 mismatch.
    """
    env_override = os.environ.get("VERNIER_COCO_GT_PATH")
    if env_override:
        candidate = Path(env_override)
        if candidate.exists() and file_sha256(candidate) == GT_SHA256:
            return candidate

    return ensure_gt(cache=bench_cache_root() / "coco_val2017")


def perfect_dt_segm_path() -> Path:
    """Locate ``perfect_dt_segm.json`` (perfect-match DT with segmentation).

    The file is generated locally by the canonical
    :mod:`coco_val_cache` (which subprocesses
    ``tools/make-perfect-dt.py``); it isn't downloaded, so there's no
    sha256 to pin. Accepts ``VERNIER_COCO_DT_SEGM_PATH`` (the parity
    convention) and falls back to the parity cache at
    ``<repo>/.cache/coco-val2017/perfect_dt_segm.json``. Raises if
    neither is present so the harness's "missing workload input"
    error surfaces early instead of inside a runner subprocess.
    """
    env_override = os.environ.get("VERNIER_COCO_DT_SEGM_PATH")
    if env_override:
        candidate = Path(env_override)
        if candidate.exists():
            return candidate

    repo_root = Path(__file__).resolve().parents[3]
    cached = repo_root / ".cache" / "coco-val2017" / "perfect_dt_segm.json"
    if cached.exists():
        return cached

    raise RuntimeError(
        "perfect_dt_segm.json not found; set VERNIER_COCO_DT_SEGM_PATH or "
        "run ./tools/fetch-coco-val.sh to populate <repo>/.cache/coco-val2017/."
    )


__all__ = ["EXPECTED_SHA256", "GT_FILENAME", "gt_path", "perfect_dt_segm_path"]
