"""Single source of truth for the Objects365 v2 val annotation cache.

Objects365 is the scale workload of the bench harness: 80,000 images,
365 categories and 1,240,587 boxes on val, roughly 16 times the images and
34 times the boxes of COCO val2017. It exercises memory and throughput at a
size where COCO val2017 cannot.

Only the **annotation JSON** is fetched. The Objects365 Consortium
licenses its annotations under Creative Commons Attribution 4.0, which
permits use and adaptation with attribution. The images are
Flickr-copyrighted, are not owned by the consortium, and are never
downloaded or referenced by this cache. Nothing lands in the repository;
the file lives in a gitignored cache directory.

Library entry points (idempotent):

- :func:`cache_root` — resolve the cache directory (env-var aware).
- :func:`ensure_gt` — download + sha256-verify the val annotation JSON.

CLI entry point::

    python -m objects365_val_cache
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import urllib.request
from collections.abc import Sequence
from pathlib import Path

#: Hosted by BAAI (the Objects365 Consortium's data platform) — the same
#: object the Ultralytics and Detectron2 dataset configs fetch. Unchanged
#: since 2021-06-10 (``ETag c2c9ca669e8883daca1f45419b7a7f14``).
GT_URL = (
    "https://dorc.ks3-cn-beijing.ksyun.com/data-set/"
    "2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/zhiyuan_objv2_val.json"
)
GT_FILENAME = "zhiyuan_objv2_val.json"
#: Verified on 2026-09-15 against :data:`GT_URL`.
GT_SHA256 = "b5b6a043f3b36c1865240a3e23fc4cbbf627052d72b6a4428a993bf2549d4424"

#: Attribution required by CC BY 4.0; surfaced wherever results derived
#: from this file are published.
ATTRIBUTION = (
    "Objects365 annotations © Objects365 Consortium, licensed under CC BY 4.0 "
    "(https://creativecommons.org/licenses/by/4.0/); https://www.objects365.org/"
)

CACHE_ENV = "VERNIER_OBJECTS365_CACHE"

# Climb out of `objects365_val_cache/objects365_val_cache/` to repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "objects365-val"

_COPY_BUF_SIZE = 1 << 20


def cache_root(override: Path | None = None) -> Path:
    """Resolve the cache directory: explicit override, then env-var,
    then default. Mirrors :func:`lvis_val_cache.cache_root`.
    """
    if override is not None:
        return override
    env = os.environ.get(CACHE_ENV)
    if env:
        return Path(env).expanduser()
    return DEFAULT_CACHE_DIR


def file_sha256(path: Path) -> str:
    """Streaming SHA-256; 1 MiB chunks bound peak memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while buf := f.read(_COPY_BUF_SIZE):
            h.update(buf)
    return h.hexdigest()


def ensure_gt(*, cache: Path | None = None) -> Path:
    """Return a verified path to the val annotation JSON, downloading if
    necessary. Raises ``RuntimeError`` on a post-download SHA mismatch."""
    cache = cache_root(cache)
    cache.mkdir(parents=True, exist_ok=True)
    gt = cache / GT_FILENAME
    if gt.is_file() and file_sha256(gt) == GT_SHA256:
        return gt

    part = gt.with_suffix(gt.suffix + ".part")
    with urllib.request.urlopen(GT_URL) as response, part.open("wb") as f:
        shutil.copyfileobj(response, f, length=_COPY_BUF_SIZE)
    actual = file_sha256(part)
    if actual != GT_SHA256:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"Objects365 val GT SHA256 mismatch: expected {GT_SHA256}, got {actual}. "
            "Either the upstream bucket served a different artifact or the download "
            "was corrupted; rerun, and if the mismatch persists open an issue."
        )
    part.replace(gt)
    return gt


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m objects365_val_cache`` entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m objects365_val_cache",
        description="Populate the Objects365 v2 val annotation cache (annotations only).",
    )
    parser.parse_args(argv)
    cache = cache_root()
    print(f"Cache directory: {cache}")
    print(f"GT ready: {ensure_gt(cache=cache)}")
    print(ATTRIBUTION)
    return 0
