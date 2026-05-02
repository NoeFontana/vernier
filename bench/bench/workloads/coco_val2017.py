"""COCO val2017 GT — download, sha256-verify, cache locally.

Constants mirror ``tools/fetch-coco-val.sh`` (the parity-test cache).
The bench harness owns its own cache at ``~/.cache/vernier-bench/`` so
``just test-coco-val`` and ``vernier-bench run`` can't fight over the
same file. ``VERNIER_COCO_GT_PATH`` (the convention from
``docs/engineering/coco-val-parity.md``) is honoured as a preverified
fallback so users who already populated the parity cache don't pay the
download twice.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from bench.harness.paths import bench_cache_root
from bench.runners._protocol import file_sha256

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
GT_FILENAME = "instances_val2017.json"
EXPECTED_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"


def gt_path() -> Path:
    """Return a verified path to the COCO val2017 GT JSON.

    Raises ``RuntimeError`` on sha256 mismatch after download.
    """
    env_override = os.environ.get("VERNIER_COCO_GT_PATH")
    if env_override:
        candidate = Path(env_override)
        if candidate.exists() and file_sha256(candidate) == EXPECTED_SHA256:
            return candidate

    cache_dir = bench_cache_root() / "coco_val2017"
    gt = cache_dir / GT_FILENAME
    if gt.exists() and file_sha256(gt) == EXPECTED_SHA256:
        return gt

    cache_dir.mkdir(parents=True, exist_ok=True)
    gt.unlink(missing_ok=True)

    zip_path = cache_dir / "annotations_trainval2017.zip"
    try:
        with urllib.request.urlopen(ANNOTATIONS_URL) as response, zip_path.open("wb") as f:
            shutil.copyfileobj(response, f)
        with (
            zipfile.ZipFile(zip_path) as z,
            z.open(f"annotations/{GT_FILENAME}") as src,
            gt.open("wb") as dst,
        ):
            shutil.copyfileobj(src, dst)
    finally:
        zip_path.unlink(missing_ok=True)

    actual = file_sha256(gt)
    if actual != EXPECTED_SHA256:
        gt.unlink(missing_ok=True)
        raise RuntimeError(
            f"COCO val2017 GT sha256 mismatch: expected {EXPECTED_SHA256}, got {actual}"
        )
    return gt
