"""Single source of truth for the LVIS v1 val real-prediction cache contract.

Parallel to :mod:`coco_val_cache`, :mod:`panoptic_val_cache`, and
:mod:`ade20k_val_cache`; the LVIS v1 real-prediction parity smoke
(``tests/python/integration/real_models/sota/test_lvis_real_models.py``)
consumes the same idempotent fetch+verify flow.

LVIS v1 reuses the COCO val2017 image set — image records carry
``coco_url`` + ``file_name`` references to ``val2017/<image>.jpg``.
This module orchestrates a one-shot provisioner that yields both
sides:

- the LVIS GT JSON (downloaded from FAIR public-files, SHA-pinned;
  the pin is shared with :mod:`lvis_val_cache` so a future upstream
  re-host invalidates both caches in lockstep),
- the COCO val2017 image directory (delegated to
  :mod:`coco_val_cache.ensure_images` so we never duplicate the 778 MB
  zip on disk; a symlink under our cache root keeps the per-cache
  layout uniform).

Library entry points (all idempotent — re-running is a no-op when
artifacts are present and verified):

- :func:`cache_root` — resolve the cache directory (env-var aware).
- :func:`ensure_gt` — download + SHA-verify the GT JSON zip, extract.
- :func:`ensure_images` — provision the val2017 images (via
  :mod:`coco_val_cache`) and link them under our cache root.
- :func:`fetch` — one-shot ``ensure_gt`` + ``ensure_images``.
- :func:`image_path` — resolve a single image_id to its on-disk JPEG.

CLI entry point::

    python -m lvis_v1_val_cache fetch        # download + provision both sides
    python -m lvis_v1_val_cache --compute-sha # print SHA-256 of GT zip, exit
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

GT_URL = "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"
#: Inner-zip path of the GT we extract.
GT_INNER_PATH = "lvis_v1_val.json"
GT_FILENAME = "lvis_v1_val.json"

#: SHA-256 of the LVIS v1 val GT zip. Verified at vendor time on
#: 2026-05-03 against the FAIR public-files mirror and shared with
#: :mod:`lvis_val_cache.GT_SHA256` — the LVIS v1 release has been
#: frozen since 2020-06, so a divergence here would indicate the
#: upstream artifact changed. Bumping is an ADR-level decision
#: (per ADR-0026 §"Parity strategy") on the same change that bumps
#: :mod:`lvis_val_cache`.
GT_ZIP_SHA256 = "2bf946b92c3037f53c172d80017f5b74ea035f00a21b20e0766b3b638b2363f9"

#: LVIS v1 val totals — captured here so the test side can assert
#: provisioning produced the expected surface without re-parsing the
#: 890 MB JSON. The numbers come straight from the LVIS v1 release
#: notes (Gupta et al., 2019; v1 frozen 2020-06).
EXPECTED_N_IMAGES = 19_809
EXPECTED_N_CATEGORIES = 1_203

CACHE_ENV = "VERNIER_LVIS_V1_VAL_CACHE"

# Climb out of `lvis_v1_val_cache/lvis_v1_val_cache/` to repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "lvis-v1-val"

#: Symlinked image dir name under our cache root. Points at the
#: canonical :mod:`coco_val_cache` images directory so the 778 MB
#: of val2017 JPEGs aren't duplicated on disk.
VAL_IMG_DIRNAME = "val2017"

_COPY_BUF_SIZE = 1 << 20


def cache_root(override: Path | None = None) -> Path:
    """Resolve the cache directory.

    Precedence: explicit ``override`` arg, then ``$VERNIER_LVIS_V1_VAL_CACHE``,
    then :data:`DEFAULT_CACHE_DIR`. The directory is *not* created
    here — the ``ensure_*`` helpers do.
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


def ensure_gt(*, cache: Path | None = None) -> Path:
    """Return a verified path to the LVIS v1 val GT JSON, downloading if needed.

    Idempotent. If the cached file's full bytes were previously
    extracted, returns the path. Otherwise downloads the
    ``lvis_v1_val.json.zip`` artefact, SHA-verifies, extracts the
    inner JSON.

    Note: we verify the SHA of the *zip*, not of the extracted JSON.
    The zip is the published artefact (the JSON's byte sequence
    depends on zipfile's extraction order on macOS vs Linux); this
    matches the convention :mod:`lvis_val_cache` uses.
    """
    cache_dir = cache_root(cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    gt = cache_dir / GT_FILENAME

    if gt.is_file():
        # Cheap presence check: the load-bearing integrity surface is
        # the zip's SHA (verified at extract time). Re-verifying the
        # extracted JSON's bytes on every call would re-parse 890 MB.
        return gt

    zip_path = cache_dir / "lvis_v1_val.json.zip"
    try:
        _atomic_download(GT_URL, zip_path)
        actual = file_sha256(zip_path)
        if actual != GT_ZIP_SHA256:
            zip_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"LVIS v1 val GT zip SHA-256 mismatch: expected "
                f"{GT_ZIP_SHA256}, got {actual}. Either the upstream "
                f"artefact changed or the download was corrupted; "
                f"re-run, and if the mismatch persists open an issue."
            )
        with zipfile.ZipFile(zip_path) as z, z.open(GT_INNER_PATH) as src, gt.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=_COPY_BUF_SIZE)
    finally:
        zip_path.unlink(missing_ok=True)

    return gt


def ensure_images(*, cache: Path | None = None) -> Path:
    """Return the val2017 image directory, provisioning if needed.

    Delegates the image download to :mod:`coco_val_cache.ensure_images`
    (LVIS v1 val reuses the COCO val2017 image set), then plants a
    symlink under our cache root so a single ``VERNIER_LVIS_V1_VAL_CACHE``
    env var resolves both the GT JSON and the images without
    juggling a separate ``VERNIER_COCO_CACHE``.

    If symlinks aren't supported on the host filesystem (Windows
    without dev-mode, or a tarball-as-cache mount), we fall back to
    returning the canonical coco-val path directly — the parity
    test reads it through ``image_path`` either way.
    """
    from coco_val_cache import ensure_images as _ensure_coco_images

    cache_dir = cache_root(cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    coco_images_dir = _ensure_coco_images()
    link = cache_dir / VAL_IMG_DIRNAME
    if link.is_symlink() or link.exists():
        # Already provisioned; trust the existing pointer rather than
        # re-resolving (a manual override pointing at an external
        # mirror is a legitimate setup).
        return link
    try:
        link.symlink_to(coco_images_dir, target_is_directory=True)
    except OSError:
        # Filesystem refuses symlinks; the canonical coco-val path is
        # equally valid for the inference side.
        return coco_images_dir
    return link


def fetch(*, cache: Path | None = None) -> tuple[Path, Path]:
    """One-shot provisioner. Returns ``(gt_json_path, images_dir)``.

    Calls :func:`ensure_gt` then :func:`ensure_images`. The two
    sides are independent; an interruption between them is recovered
    by re-running ``fetch``.
    """
    gt = ensure_gt(cache=cache)
    images_dir = ensure_images(cache=cache)
    return gt, images_dir


def image_path(image_id: int, *, cache: Path | None = None) -> Path:
    """Resolve a single LVIS image_id to its val2017 JPEG path.

    Uses the COCO val2017 naming convention (``%012d.jpg`` zero-padded
    to 12 digits) — LVIS v1 image_ids ARE the COCO image_ids, so the
    join is by id without a separate lookup table. Does not check
    file existence; the inference loop's per-image
    ``iter_image_records`` does the I/O probe.
    """
    images_dir = cache_root(cache) / VAL_IMG_DIRNAME
    if not images_dir.exists():
        # Fall through to the canonical coco-val path; matches the
        # ``ensure_images`` fallback above.
        from coco_val_cache import IMAGES_DIRNAME
        from coco_val_cache import cache_root as _coco_cache_root

        images_dir = _coco_cache_root() / IMAGES_DIRNAME
    return images_dir / f"{image_id:012d}.jpg"


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m lvis_v1_val_cache`` entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m lvis_v1_val_cache",
        description=(
            "Provision the LVIS v1 val real-prediction cache "
            "(GT JSON + val2017 images via coco_val_cache)."
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["fetch"],
        default="fetch",
        help="Default action: download + provision both GT and images.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Override cache directory (defaults to env or repo .cache).",
    )
    parser.add_argument(
        "--compute-sha",
        action="store_true",
        help="Download the GT zip, print its SHA-256, delete it. "
        "Exit without extracting. Use after an upstream re-host to "
        "rotate the pin.",
    )
    args = parser.parse_args(argv)

    cache_dir = cache_root(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {cache_dir}")

    if args.compute_sha:
        zip_path = cache_dir / "lvis_v1_val.json.zip"
        try:
            _atomic_download(GT_URL, zip_path)
            print(f"SHA-256: {file_sha256(zip_path)}")
        finally:
            zip_path.unlink(missing_ok=True)
        return 0

    gt, images_dir = fetch(cache=args.cache)
    print(f"LVIS GT ready: {gt}  (sha256 verified at extract time)")
    print(f"COCO val2017 images ready: {images_dir}")
    return 0


__all__ = [
    "CACHE_ENV",
    "DEFAULT_CACHE_DIR",
    "EXPECTED_N_CATEGORIES",
    "EXPECTED_N_IMAGES",
    "GT_FILENAME",
    "GT_URL",
    "GT_ZIP_SHA256",
    "VAL_IMG_DIRNAME",
    "cache_root",
    "ensure_gt",
    "ensure_images",
    "fetch",
    "file_sha256",
    "image_path",
    "main",
]
