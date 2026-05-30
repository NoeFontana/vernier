"""Single source of truth for the ADE20K SceneParse150 val dev cache contract.

Parallel to :mod:`panoptic_val_cache` and :mod:`coco_val_cache`. The
Mask2Former ADE-semantic real-prediction parity smoke consumes the
same idempotent fetch+verify flow against the SceneParse150 challenge
bundle.

Library entry points (all idempotent):

- :func:`cache_root` — resolve the cache directory (env-var aware).
- :func:`ensure_gt` — download + SHA-verify the challenge zip,
  extract the validation images + per-image semantic GT, materialize
  a converted train-id GT directory (0..149 + 255-ignore, mmseg
  ``reduce_zero_label=True`` convention).
- :func:`scan_label_map_pngs` — re-export of the shared helper.

CLI entry point::

    python -m ade20k_val_cache              # download + materialize
    python -m ade20k_val_cache --compute-sha # print SHA-256, exit
    python -m ade20k_val_cache --verify-only # verify pinned SHA, no extract

Label-space conversion: upstream PNGs are uint8 with values ``0..150``
where ``0`` is the unlabeled / background sentinel. mmsegmentation's
canonical ADE flow uses ``reduce_zero_label=True``, shifting to a
contiguous ``0..149`` train-id space and treating upstream ``0`` as
``ignore_label=255``. We do that conversion at materialize-time so
vernier's :func:`semantic.decode_label_map_png` consumes a uniform
shape across COCO-derived and ADE-derived workloads.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

GT_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"

#: SHA-256 of the SceneParse150 challenge zip. Verified at vendor
#: time on 2026-05-30 against the canonical MIT CSAIL mirror; a
#: future upstream re-host or revision would change this value and
#: the cache module must update atomically (same discipline as the
#: panoptic_val_cache GT_ZIP_SHA256).
GT_ZIP_SHA256: str | None = "7ff1be44964418441f542a7cc1e1a650e7dc0fc275f5d23252bc9bbdbc977b29"

_VAL_GT_INNER_PREFIX = "ADEChallengeData2016/annotations/validation/"
_VAL_IMG_INNER_PREFIX = "ADEChallengeData2016/images/validation/"
_VAL_GT_FILENAME_RE = re.compile(r"^ADE_val_(\d{8})\.png$")
_VAL_IMG_FILENAME_RE = re.compile(r"^ADE_val_(\d{8})\.jpg$")

CACHE_ENV = "VERNIER_ADE20K_CACHE"

# Climb out of `ade20k_val_cache/ade20k_val_cache/` to repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "ade20k-val"

VAL_GT_DIRNAME = "val_gt_train_ids"
VAL_IMG_DIRNAME = "val_images"

#: 150 semantic classes after the mmseg ``reduce_zero_label`` shift
#: (upstream ``1..150`` → contiguous train-id ``0..149``).
ADE20K_NUM_CLASSES = 150
#: Sentinel for unlabeled pixels (upstream ``segment_id == 0``).
#: Matches the Cityscapes / Pascal VOC / mmseg convention; safely
#: outside the ``0..149`` train-id range.
ADE20K_IGNORE_LABEL = 255

_COPY_BUF_SIZE = 1 << 20


def cache_root(override: Path | None = None) -> Path:
    """Resolve the cache directory: explicit override, then env-var,
    then default. Mirrors :func:`panoptic_val_cache.cache_root`."""
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
    """Verify the downloaded zip's SHA-256 against :data:`GT_ZIP_SHA256`.

    Unpinned (``GT_ZIP_SHA256 is None``): print the observed SHA and
    proceed. The user is expected to pin the constant via
    ``python -m ade20k_val_cache --compute-sha`` once and subsequent
    runs verify strictly. Print-without-fail is intentional on the
    first download — the alternative is forcing every new user
    through a two-step pin-then-download dance.

    Pinned: raise on mismatch and delete the zip (so a retry doesn't
    silently accept a corrupted cache).
    """
    actual = file_sha256(zip_path)
    if GT_ZIP_SHA256 is None:
        print(
            f"ade20k val zip SHA-256 (not yet pinned): {actual}. "
            f"To pin: edit `GT_ZIP_SHA256` in `ade20k_val_cache/__init__.py` "
            f"to this value. Subsequent runs will verify strictly."
        )
        return
    if actual != GT_ZIP_SHA256:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"ADE20K val zip SHA-256 mismatch: expected {GT_ZIP_SHA256}, "
            f"got {actual}. Either the upstream artifact changed or the "
            f"download was corrupted; rerun, and if the mismatch persists "
            f"open an issue."
        )


def _convert_raw_ade_png_to_train_ids(src_bytes: bytes, dst: Path) -> None:
    """Convert an ADE20K raw GT PNG (uint8 ``0..150``) to a train-id PNG.

    Upstream label ``0`` ("background/unlabeled") maps to
    :data:`ADE20K_IGNORE_LABEL`. Labels ``1..150`` shift down to
    contiguous train-ids ``0..149``. The mmseg ``reduce_zero_label=True``
    semantics — applied here at materialize time so vernier consumes a
    canonical 0..149 + 255 PNG without an ingest-side LUT.
    """
    import io

    import numpy as np
    from PIL import Image

    raw = np.asarray(Image.open(io.BytesIO(src_bytes)), dtype=np.uint8)
    if raw.ndim != 2:
        raise ValueError(f"ADE20K GT PNG must be single-channel uint8; got shape {raw.shape!r}")
    # Branchless LUT: 0 → 255, 1..150 → 0..149, anything ≥151 → 255.
    lut = np.full(256, ADE20K_IGNORE_LABEL, dtype=np.uint8)
    lut[1 : ADE20K_NUM_CLASSES + 1] = np.arange(ADE20K_NUM_CLASSES, dtype=np.uint8)
    Image.fromarray(lut[raw], mode="L").save(dst)


def _image_id_from_filename(name: str, pattern: re.Pattern[str]) -> int:
    """Parse the numeric image id out of an ADE val filename.

    Strict: filenames must match `ADE_val_<8-digit>.{png,jpg}`. A
    mismatched stray (rare, but ADE bundles have shipped with stray
    `.DS_Store` etc.) raises ValueError. The strictness defends
    against silently producing a partial cache when upstream layout
    shifts.
    """
    m = pattern.match(name)
    if m is None:
        raise ValueError(
            f"ADE20K val filename {name!r} doesn't match the expected pattern "
            f"{pattern.pattern!r}. If upstream has shipped a new naming "
            f"convention, the cache module needs updating."
        )
    return int(m.group(1))


def ensure_gt(*, cache: Path | None = None) -> tuple[Path, Path, int]:
    """Return ``(gt_dir, images_dir, n_classes)``, downloading + extracting
    if necessary.

    Idempotent: skips the network round-trip when both directories
    already exist and are non-empty. ``gt_dir`` contains the
    train-id-converted PNGs (one per validation image, named
    ``<image_id>.png``); ``images_dir`` contains the upstream JPEGs
    unmodified (for the inference side).
    """
    cache = cache_root(cache)
    cache.mkdir(parents=True, exist_ok=True)
    gt_dir = cache / VAL_GT_DIRNAME
    images_dir = cache / VAL_IMG_DIRNAME

    if (
        gt_dir.is_dir()
        and images_dir.is_dir()
        and any(gt_dir.iterdir())
        and any(images_dir.iterdir())
    ):
        return gt_dir, images_dir, ADE20K_NUM_CLASSES

    zip_path = cache / "ADEChallengeData2016.zip"
    try:
        _atomic_download(GT_URL, zip_path)
        _verify_sha(zip_path)
        gt_dir.mkdir(exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                if info.filename.startswith(_VAL_GT_INNER_PREFIX):
                    base = info.filename[len(_VAL_GT_INNER_PREFIX) :]
                    if not base:
                        continue
                    image_id = _image_id_from_filename(base, _VAL_GT_FILENAME_RE)
                    target = gt_dir / f"{image_id}.png"
                    if target.is_file():
                        continue
                    _convert_raw_ade_png_to_train_ids(z.read(info), target)
                elif info.filename.startswith(_VAL_IMG_INNER_PREFIX):
                    base = info.filename[len(_VAL_IMG_INNER_PREFIX) :]
                    if not base:
                        continue
                    image_id = _image_id_from_filename(base, _VAL_IMG_FILENAME_RE)
                    target = images_dir / f"{image_id}.jpg"
                    if target.is_file():
                        continue
                    with z.open(info) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=_COPY_BUF_SIZE)
    finally:
        zip_path.unlink(missing_ok=True)

    return gt_dir, images_dir, ADE20K_NUM_CLASSES


def scan_label_map_pngs(directory: Path) -> dict[int, Path]:
    """Index ``<int>.png`` files in ``directory`` by parsed image_id.

    Shared shape with :func:`panoptic_val_cache.scan_label_map_pngs` so
    the bench + parity tests use one helper across paradigms.
    """
    out: dict[int, Path] = {}
    for entry in sorted(directory.iterdir()):
        if entry.suffix.lower() != ".png":
            continue
        try:
            image_id = int(entry.stem)
        except ValueError as e:
            raise ValueError(
                f"label-map dir {directory!s} contains {entry.name!r}; expected '<int>.png'."
            ) from e
        out[image_id] = entry
    return out


def scan_image_jpgs(directory: Path) -> dict[int, Path]:
    """Index ``<int>.jpg`` files in ``directory`` by parsed image_id.

    Companion to :func:`scan_label_map_pngs` for the inference-side
    image directory.
    """
    out: dict[int, Path] = {}
    for entry in sorted(directory.iterdir()):
        if entry.suffix.lower() != ".jpg":
            continue
        try:
            image_id = int(entry.stem)
        except ValueError as e:
            raise ValueError(
                f"image dir {directory!s} contains {entry.name!r}; expected '<int>.jpg'."
            ) from e
        out[image_id] = entry
    return out


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="Download the zip, print its SHA-256, delete it. Exit without extracting.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Re-download the zip and verify against GT_ZIP_SHA256 without extracting. "
            "Useful after editing the pin to confirm strictness holds."
        ),
    )
    args = parser.parse_args(argv)

    cache = cache_root(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {cache}")

    if args.compute_sha or args.verify_only:
        zip_path = cache / "ADEChallengeData2016.zip"
        try:
            _atomic_download(GT_URL, zip_path)
            if args.compute_sha:
                sha = file_sha256(zip_path)
                print(f"SHA-256: {sha}")
            else:
                _verify_sha(zip_path)
                print("SHA-256 verified against pinned constant.")
        finally:
            zip_path.unlink(missing_ok=True)
        return 0

    gt_dir, images_dir, _ = ensure_gt(cache=args.cache)
    print(f"Validation GT ready: {gt_dir}  ({sum(1 for _ in gt_dir.iterdir())} PNGs)")
    print(f"Validation images ready: {images_dir}  ({sum(1 for _ in images_dir.iterdir())} JPGs)")
    return 0


__all__ = [
    "ADE20K_IGNORE_LABEL",
    "ADE20K_NUM_CLASSES",
    "CACHE_ENV",
    "DEFAULT_CACHE_DIR",
    "GT_URL",
    "GT_ZIP_SHA256",
    "VAL_GT_DIRNAME",
    "VAL_IMG_DIRNAME",
    "cache_root",
    "ensure_gt",
    "file_sha256",
    "main",
    "scan_image_jpgs",
    "scan_label_map_pngs",
]
