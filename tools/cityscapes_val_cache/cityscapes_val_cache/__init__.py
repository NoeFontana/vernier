"""Single source of truth for the Cityscapes val dev cache contract
(ADR-0028, ADR-0033 §B2).

Parallel to :mod:`coco_val_cache`, :mod:`panoptic_val_cache`, and
:mod:`lvis_val_cache`: same role (single home for "where does this
dataset live, what filename, what SHA"), different dataset.

Cityscapes is gated upstream: downloads require a registered account
on https://www.cityscapesdataset.com/ and agreement to the dataset
terms of use. There is no public unauthenticated URL for
``gtFine_trainvaltest.zip``. This cache therefore does not download
the dataset; it expects the user to pre-populate the cache with a
local copy of the zip (or to set the override env vars below).

The ``cityscapes_val_perfect`` bench workload (a GT-as-DT smoke; both
inputs point at the same trainId PNGs) is gated behind
``VERNIER_BENCH_CITYSCAPES=1`` so the bench harness default test loop
does not touch external systems.

Library entry points (all idempotent):

- :func:`cache_root` — resolve the cache directory (env-var aware).
- :func:`ensure_val_label_maps` — verify a populated cache and return
  the trainId PNG directory (extracts ``gtFine_trainvaltest.zip``
  in-place if the user has dropped it into the cache).

CLI entry point::

    python -m cityscapes_val_cache
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import zipfile
from collections.abc import Sequence
from pathlib import Path

#: Filename the user is expected to drop into the cache directory.
#: Sourced from ``https://www.cityscapesdataset.com/file-handling/?packageID=1``.
GT_ZIP_FILENAME = "gtFine_trainvaltest.zip"

#: SHA-256 of the official ``gtFine_trainvaltest.zip`` artifact.
#:
#: Cityscapes is a gated dataset; the official URL requires an
#: authenticated session. Until the first developer provisions the
#: cache and publishes the verified hash, this remains a placeholder —
#: the cache refuses any non-matching artifact once pinned. Bootstrap
#: mode raises with the observed hash so the developer can paste it
#: into this constant. Mirrors :mod:`panoptic_val_cache`'s placeholder
#: pattern.
GT_ZIP_SHA256 = "0" * 64

#: Inner-zip prefix containing val-split label maps in trainId space.
#: ``cityscapesscripts/preparation/createTrainIdLabelImgs.py`` writes
#: these alongside the upstream ``*_gtFine_labelIds.png`` files.
GT_INNER_DIR = "gtFine/val"

#: Glob pattern for the single-channel trainId PNGs that vernier and
#: cityscapesScripts both consume. Cityscapes 19-class evaluation
#: convention; pixel values are in ``[0, 18] ∪ {255}`` (255 is the
#: ignore label per ADR-0028).
GT_TRAIN_ID_PNG_GLOB = "*_gtFine_labelTrainIds.png"

CACHE_ENV = "VERNIER_CITYSCAPES_CACHE"

#: Override env vars: when the user has the dataset extracted somewhere
#: other than the canonical cache directory, point at it directly.
#: Mirrors ``VERNIER_COCO_GT_PATH`` from the coco-val cache.
GT_DIR_ENV = "VERNIER_CITYSCAPES_VAL_GT_DIR"
DT_DIR_ENV = "VERNIER_CITYSCAPES_VAL_DT_DIR"

# Climb out of `cityscapes_val_cache/cityscapes_val_cache/` to repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "cityscapes-val"

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


def _verify_sha(zip_path: Path) -> None:
    actual = file_sha256(zip_path)
    if GT_ZIP_SHA256 == "0" * 64:
        # Bootstrap mode: surface the observed hash so the first
        # developer can paste it into this constant. Mirrors
        # :func:`panoptic_val_cache._verify_sha`.
        raise RuntimeError(
            f"Cityscapes GT zip SHA-256 is unpinned (placeholder constant). "
            f"Observed SHA-256: {actual}\n"
            f"Bootstrap: paste this value into GT_ZIP_SHA256 in "
            f"tools/cityscapes_val_cache/cityscapes_val_cache/__init__.py "
            f"and rerun. The cache refuses any non-matching artifact "
            f"once pinned (per the parity-contract reproducibility claim)."
        )
    if actual != GT_ZIP_SHA256:
        raise RuntimeError(
            f"Cityscapes GT zip SHA256 mismatch: expected {GT_ZIP_SHA256}, "
            f"got {actual}. Either you pre-populated the cache with a "
            f"non-matching artifact (cityscapesdataset.com versions the "
            f"zip occasionally) or the file is corrupt."
        )


def _val_label_pngs(val_dir: Path) -> list[Path]:
    """Return every trainId PNG under ``val_dir`` (recursively). The
    Cityscapes val split has 500 images organized into 3 city
    subdirectories (frankfurt / lindau / munster).
    """
    return sorted(val_dir.rglob(GT_TRAIN_ID_PNG_GLOB))


def _val_dir_ok(val_dir: Path) -> bool:
    """Probe + count integrity check for ``gtFine/val/``.

    Cityscapes val carries 500 images. Mirrors the
    :func:`coco_val_cache._images_dir_ok` count check.
    """
    if not val_dir.is_dir():
        return False
    return len(_val_label_pngs(val_dir)) == 500


def ensure_val_label_maps(*, cache: Path | None = None) -> Path:
    """Return a verified path to the Cityscapes val trainId PNG dir.

    Three resolution paths in order of precedence:

    1. ``$VERNIER_CITYSCAPES_VAL_GT_DIR`` — if set and a non-empty
       directory, return it directly. The user has the data extracted
       somewhere else and does not want the cache to manage it.
    2. ``<cache>/gtFine/val/`` — if already populated and counts check
       out (500 trainId PNGs), return it.
    3. ``<cache>/gtFine_trainvaltest.zip`` — if the user has dropped
       the zip into the cache, sha256-verify and extract it in place.

    Raises ``RuntimeError`` if none of the three paths produces a
    valid val dir. Cityscapes is gated upstream — there is no
    automated download path.
    """
    env_override = os.environ.get(GT_DIR_ENV)
    if env_override:
        candidate = Path(env_override).expanduser()
        if _val_dir_ok(candidate):
            return candidate
        raise RuntimeError(
            f"{GT_DIR_ENV}={env_override!r} does not point at a populated "
            f"Cityscapes val dir (expected 500 ``*_gtFine_labelTrainIds.png`` "
            f"files under it)."
        )

    cache = cache_root(cache)
    val_dir = cache / GT_INNER_DIR

    if _val_dir_ok(val_dir):
        return val_dir

    zip_path = cache / GT_ZIP_FILENAME
    if zip_path.is_file():
        _verify_sha(zip_path)
        cache.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if not name.startswith(GT_INNER_DIR + "/"):
                    continue
                if name.endswith("/"):
                    continue
                target = cache / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=_COPY_BUF_SIZE)
        if _val_dir_ok(val_dir):
            return val_dir

    raise RuntimeError(
        "Cityscapes val cache is not populated. Cityscapes is a gated "
        "dataset (registration required at https://www.cityscapesdataset.com/); "
        "this module does not auto-download. Provision the cache by either:\n"
        f"  - downloading {GT_ZIP_FILENAME!r} from the dataset site and "
        f"placing it at {zip_path!s}, or\n"
        f"  - extracting it elsewhere and pointing at the extracted "
        f"trainId PNG dir via {GT_DIR_ENV}=<path-to-gtFine/val>."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m cityscapes_val_cache`` entry point.

    Reports cache status without attempting any download (Cityscapes
    is a gated dataset). Use this to confirm a populated cache before
    running the bench cell.
    """
    parser = argparse.ArgumentParser(
        prog="python -m cityscapes_val_cache",
        description="Probe the Cityscapes val dev cache (no automatic download).",
    )
    parser.parse_args(argv)

    cache = cache_root()
    print(f"Cache directory: {cache}")
    env_override = os.environ.get(GT_DIR_ENV)
    if env_override:
        print(f"{GT_DIR_ENV}={env_override}")

    try:
        val_dir = ensure_val_label_maps(cache=cache)
    except RuntimeError as e:
        print(f"NOT READY: {e}")
        return 1
    n_pngs = len(_val_label_pngs(val_dir))
    print(f"Val trainId PNG dir ready: {val_dir} ({n_pngs} pngs)")
    print()
    print("Export the env vars to run the bench cell against the cache:")
    print(f"  export VERNIER_BENCH_CITYSCAPES=1")
    print(f"  export {GT_DIR_ENV}={val_dir}")
    return 0
