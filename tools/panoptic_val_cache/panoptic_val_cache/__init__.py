"""Single source of truth for the COCO panoptic val2017 dev cache contract.

Parallel to :mod:`coco_val_cache` (the COCO val2017 cache) and
:mod:`lvis_val_cache` (the LVIS v1 val cache); the panoptic
whole-dataset parity smoke (ADR-0025 PR-6) consumes the same
idempotent fetch+verify flow against the COCO panoptic GT JSON +
per-image PNG label maps.

Library entry points (all idempotent):

- :func:`cache_root` — resolve the cache directory (env-var aware).
- :func:`ensure_gt` — download + sha256-verify the GT zip, extract
  the JSON + PNG directory.
- :func:`ensure_perfect_dt` — synthesize a perfect-DT JSON +
  PNG directory from the GT (a parity sanity check that should
  produce ``PQ=1.0`` end-to-end against the oracle).

CLI entry point::

    python -m panoptic_val_cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

GT_URL = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"
#: Inner-zip relative paths we extract.
GT_INNER_JSON = "annotations/panoptic_val2017.json"
GT_INNER_PNG_DIR = "annotations/panoptic_val2017/"
GT_JSON_FILENAME = "panoptic_val2017.json"
GT_PNG_DIRNAME = "panoptic_val2017"
#: SHA-256 of the panoptic-annotations zip artifact at upstream
#: ``cocodataset.org`` as of 2026-05-03. Bumping is an ADR-level
#: decision per ADR-0025 §"Parity strategy" — every quirk vernier
#: reproduces is keyed to this exact byte sequence.
#:
#: NOTE: This SHA is **a placeholder** until the first developer
#: provisions the cache and publishes the verified hash; the cache
#: will refuse to use any download whose SHA does not match the
#: pinned value below. See PR-6 of the ADR-0025 rollout for the
#: refresh procedure once the pin is verified.
GT_ZIP_SHA256 = "0000000000000000000000000000000000000000000000000000000000000000"

CACHE_ENV = "VERNIER_PANOPTIC_CACHE"

# Climb out of `panoptic_val_cache/panoptic_val_cache/` to repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "panoptic-val2017"

PERFECT_DT_JSON_FILENAME = "perfect_dt.json"
PERFECT_DT_PNG_DIRNAME = "perfect_dt_pngs"

_COPY_BUF_SIZE = 1 << 20


def cache_root(override: Path | None = None) -> Path:
    """Resolve the cache directory: explicit override, then env-var,
    then default. Mirrors :func:`coco_val_cache.cache_root`.
    """
    if override is not None:
        return override
    env = os.environ.get(CACHE_ENV)
    if env:
        return Path(env).expanduser()
    return DEFAULT_CACHE_DIR


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes, hex-encoded. Streams in 1-MiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(_COPY_BUF_SIZE)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _atomic_download(url: str, dest: Path) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, part.open("wb") as f:
        shutil.copyfileobj(response, f, length=_COPY_BUF_SIZE)
    part.replace(dest)


def _verify_sha(zip_path: Path) -> None:
    if GT_ZIP_SHA256 == "0" * 64:
        # Bootstrapping: print the observed SHA so a developer can
        # paste it into this module to pin the artifact.
        actual = file_sha256(zip_path)
        print(
            f"WARNING: panoptic GT zip SHA-256 is unpinned. "
            f"Observed: {actual}\n"
            f"Update GT_ZIP_SHA256 in tools/panoptic_val_cache/panoptic_val_cache/__init__.py "
            f"to pin this artifact."
        )
        return
    actual = file_sha256(zip_path)
    if actual != GT_ZIP_SHA256:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"panoptic GT zip SHA256 mismatch: expected {GT_ZIP_SHA256}, got {actual}. "
            f"Either the upstream CDN served a different artifact or the "
            f"download was corrupted; rerun, and if the mismatch persists "
            f"open an issue."
        )


def ensure_gt(*, cache: Path | None = None) -> tuple[Path, Path]:
    """Return ``(json_path, png_dir_path)``, downloading + extracting
    if necessary. Idempotent: skips the download when the JSON
    already exists at the expected path with the expected size."""
    cache = cache_root(cache)
    cache.mkdir(parents=True, exist_ok=True)
    json_path = cache / GT_JSON_FILENAME
    png_dir = cache / GT_PNG_DIRNAME

    if json_path.is_file() and png_dir.is_dir() and any(png_dir.iterdir()):
        return json_path, png_dir

    zip_path = cache / "panoptic_annotations_trainval2017.zip"
    try:
        _atomic_download(GT_URL, zip_path)
        _verify_sha(zip_path)
        with zipfile.ZipFile(zip_path) as z:
            # Extract the val JSON.
            with z.open(GT_INNER_JSON) as src, json_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=_COPY_BUF_SIZE)
            # Extract the val PNG directory.
            png_dir.mkdir(exist_ok=True)
            for name in z.namelist():
                if not name.startswith(GT_INNER_PNG_DIR):
                    continue
                if name.endswith("/"):
                    continue
                target = png_dir / Path(name).name
                with z.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=_COPY_BUF_SIZE)
    finally:
        zip_path.unlink(missing_ok=True)

    return json_path, png_dir


def ensure_perfect_dt(*, cache: Path | None = None) -> tuple[Path, Path]:
    """Synthesize a perfect-DT JSON + PNG directory from the GT.

    A panoptic perfect DT is structurally a copy of the GT: same
    segment ids, same PNG bytes, same ``segments_info`` per image.
    Running this DT against the GT under any parity mode should
    produce ``PQ=1.0`` end-to-end (sanity check the kernel + parity
    harness on real data).

    Returns ``(json_path, png_dir_path)``. Idempotent: skips work if
    both already exist.
    """
    json_path, png_dir = ensure_gt(cache=cache)
    cache_dir = json_path.parent
    dt_json = cache_dir / PERFECT_DT_JSON_FILENAME
    dt_png_dir = cache_dir / PERFECT_DT_PNG_DIRNAME

    if dt_json.is_file() and dt_png_dir.is_dir() and any(dt_png_dir.iterdir()):
        return dt_json, dt_png_dir

    dt_png_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("rb") as f:
        gt_data = json.load(f)

    # DT has the same `annotations` shape but no `categories` key
    # (panopticapi's pq_compute reads categories from GT JSON only —
    # quirk S9). Mirror the GT but drop categories.
    dt_data = {
        "images": gt_data.get("images", []),
        "annotations": gt_data.get("annotations", []),
    }
    with dt_json.open("w") as f:
        json.dump(dt_data, f)

    # Symlink each GT PNG into the DT dir. Symlinks are zero-cost vs
    # full copies (~3 GB on COCO panoptic val2017) and the panopticapi
    # oracle reads via PIL.Image.open which follows symlinks.
    for png in png_dir.iterdir():
        if not png.is_file():
            continue
        target = dt_png_dir / png.name
        if target.is_symlink() or target.exists():
            continue
        target.symlink_to(png.resolve())

    return dt_json, dt_png_dir


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m panoptic_val_cache`` entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m panoptic_val_cache",
        description="Populate the COCO panoptic val2017 dev cache (GT + perfect-DT).",
    )
    parser.parse_args(argv)
    cache = cache_root()
    print(f"Cache directory: {cache}")
    json_path, png_dir = ensure_gt(cache=cache)
    print(f"GT JSON ready: {json_path}")
    print(f"GT PNG dir ready: {png_dir}")
    dt_json, dt_png_dir = ensure_perfect_dt(cache=cache)
    print(f"Perfect DT JSON: {dt_json}")
    print(f"Perfect DT PNG dir: {dt_png_dir}")
    print()
    print("Export the env vars and run `just test-parity-panoptic-val`:")
    print(f"  export VERNIER_PANOPTIC_GT_PATH={json_path}")
    print(f"  export VERNIER_PANOPTIC_GT_PNG_DIR={png_dir}")
    print(f"  export VERNIER_PANOPTIC_DT_PATH={dt_json}")
    print(f"  export VERNIER_PANOPTIC_DT_PNG_DIR={dt_png_dir}")
    return 0
