"""Single source of truth for the LVIS v1 val dev cache contract.

Parallel to :mod:`coco_val_cache` (the COCO val2017 cache); the LVIS
whole-dataset parity smoke (ADR-0026 PR-6) consumes the same
idempotent fetch+verify flow against the LVIS GT + a perfect-DT
synthesized from it. LVIS reuses COCO 2017 images by `coco_url`
reference; only the GT JSON and the perfect-DT JSON live here.

Library entry points (all idempotent):

- :func:`cache_root` — resolve the cache directory (env-var aware).
- :func:`ensure_gt` — download + sha256-verify the GT JSON.
- :func:`ensure_perfect_dts` — synthesize bbox + segm perfect-DTs
  via ``tools/make-perfect-dt.py``.

CLI entry point::

    python -m lvis_val_cache
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

GT_URL = "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"
#: Inner-zip path of the GT we extract.
GT_INNER_PATH = "lvis_v1_val.json"
GT_FILENAME = "lvis_v1_val.json"
#: Bumping is an ADR-level decision per ADR-0026 §"Parity strategy" —
#: every quirk vernier reproduces is keyed to this exact byte
#: sequence. Verified at vendor time on 2026-05-03 against the FAIR
#: public-files mirror; the LVIS v1 release is frozen since 2020-06.
GT_SHA256 = "2bf946b92c3037f53c172d80017f5b74ea035f00a21b20e0766b3b638b2363f9"

CACHE_ENV = "VERNIER_LVIS_CACHE"

# Climb out of `lvis_val_cache/lvis_val_cache/` to repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "lvis-val"

#: The perfect-DT synthesizer is identical to the COCO path's — LVIS
#: annotations match the COCO schema for `iscrowd`/`segmentation`/
#: `bbox`, so the same tool produces correct output when fed an LVIS
#: GT.
_MAKE_PERFECT_DT = _REPO_ROOT / "tools" / "make-perfect-dt.py"

PERFECT_DT_BBOX_FILENAME = "perfect_dt.json"
PERFECT_DT_SEGM_FILENAME = "perfect_dt_segm.json"

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
    """SHA-256 of a file's bytes, hex-encoded. Streams in 1-MiB
    chunks so the computation cost scales with throughput, not RSS.
    """
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


def ensure_gt(*, cache: Path | None = None) -> Path:
    """Return a verified path to the GT JSON, downloading if necessary.

    Idempotent. If the cached file matches :data:`GT_SHA256`, returns
    immediately. Otherwise (re)downloads from :data:`GT_URL`, extracts
    the inner JSON, verifies, returns the path. Raises ``RuntimeError``
    on a post-download SHA mismatch.
    """
    cache = cache_root(cache)
    cache.mkdir(parents=True, exist_ok=True)
    gt = cache / GT_FILENAME

    if gt.is_file() and file_sha256(gt) == GT_SHA256:
        return gt

    gt.unlink(missing_ok=True)
    zip_path = cache / "lvis_v1_val.json.zip"
    try:
        _atomic_download(GT_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z, z.open(GT_INNER_PATH) as src, gt.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=_COPY_BUF_SIZE)
    finally:
        zip_path.unlink(missing_ok=True)

    actual = file_sha256(gt)
    if actual != GT_SHA256:
        gt.unlink(missing_ok=True)
        raise RuntimeError(
            f"LVIS GT SHA256 mismatch: expected {GT_SHA256}, got {actual}. "
            f"Either the upstream CDN served a different artifact or the "
            f"download was corrupted; rerun, and if the mismatch persists "
            f"open an issue."
        )
    return gt


def ensure_perfect_dts(*, cache: Path | None = None) -> tuple[Path, Path]:
    """Return ``(bbox_dt_path, segm_dt_path)``, synthesizing if absent.

    The synthesizer is :mod:`make-perfect-dt`; we subprocess it to
    keep that script self-contained and CLI-usable in its own right.
    Calls :func:`ensure_gt` first.
    """
    gt = ensure_gt(cache=cache)
    cache_dir = gt.parent
    bbox_dt = cache_dir / PERFECT_DT_BBOX_FILENAME
    segm_dt = cache_dir / PERFECT_DT_SEGM_FILENAME

    def _synth(out: Path, *, segm: bool) -> None:
        if out.is_file():
            return
        cmd: list[str] = [sys.executable, str(_MAKE_PERFECT_DT)]
        if segm:
            cmd.append("--segm")
        cmd.extend([str(gt), str(out)])
        subprocess.run(cmd, check=True)

    _synth(bbox_dt, segm=False)
    _synth(segm_dt, segm=True)
    return bbox_dt, segm_dt


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m lvis_val_cache`` entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m lvis_val_cache",
        description="Populate the LVIS v1 val dev cache (GT JSON + perfect-DTs).",
    )
    parser.parse_args(argv)
    cache = cache_root()
    print(f"Cache directory: {cache}")
    gt = ensure_gt(cache=cache)
    print(f"GT ready: {gt}")
    bbox_dt, segm_dt = ensure_perfect_dts(cache=cache)
    print(f"Perfect bbox DT: {bbox_dt}")
    print(f"Perfect segm DT: {segm_dt}")
    print()
    print("Export the env vars (or eval the lines below) and run `just test-parity-lvis-val`:")
    print(f"  export VERNIER_LVIS_GT_PATH={gt}")
    print(f"  export VERNIER_LVIS_DT_PATH={bbox_dt}")
    print(f"  export VERNIER_LVIS_DT_SEGM_PATH={segm_dt}")
    return 0
