"""Single source of truth for the COCO val2017 dev cache contract.

Three places used to know the same constants (URL, filename, SHA256)
and re-implement the same fetch+verify flow: ``tools/fetch-coco-val.sh``
(bash), ``bench/bench/workloads/coco_val2017.py`` (Python), and
``tests/python/coco_val_paths.py`` (cache-root convention only). This
module is the canonical owner; the three call sites are thin wrappers.

The "what is the COCO val2017 cache" concept is one fact about the
world. Two copies could drift; one cannot. Image-related constants
moved here in lockstep with the ``--with-images`` capability.

Library entry points (all idempotent — re-running is a no-op when
artifacts are already present and verified):

- :func:`cache_root` — resolve the cache directory (env-var aware).
- :func:`ensure_gt` — download + sha256-verify the GT JSON.
- :func:`ensure_images` — download + extract ``val2017/`` images.
- :func:`ensure_perfect_dts` — synthesize the perfect-DT JSONs via
  ``make-perfect-dt.py``.

CLI entry point (replaces the old bash logic in
``tools/fetch-coco-val.sh``)::

    python -m coco_val_cache [--with-images]
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

GT_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
#: Inner-zip path of the GT we actually extract; the upstream zip
#: bundles train/val/test annotation files together.
GT_INNER_PATH = "annotations/instances_val2017.json"
GT_FILENAME = "instances_val2017.json"
#: Bumping is an ADR-level decision per
#: docs/engineering/coco-val-parity.md — every parity quirk we
#: reproduce is keyed to this exact byte sequence.
GT_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"

#: Keypoints GT — same upstream zip as the instances GT. Bumping is
#: the same ADR-level decision per docs/engineering/coco-val-parity.md;
#: OKS parity quirks (F1/F3/F4) are keyed to this byte sequence.
KP_GT_INNER_PATH = "annotations/person_keypoints_val2017.json"
KP_GT_FILENAME = "person_keypoints_val2017.json"
KP_GT_SHA256 = "788e2dae83c86bd547be7fab269d6399df5671063d29a61360cdb2cc370d2b14"

IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
IMAGES_DIRNAME = "val2017"
IMAGES_EXPECTED_COUNT = 5000
#: Lowest-numbered val2017 image; presence is the second half of the
#: integrity check. Pin + count catches "this directory has 5000 jpgs
#: from a different dataset" without committing to a full SHA pin
#: (image bytes don't affect parity claims; only the GT's do).
IMAGES_PROBE_FILENAME = "000000000139.jpg"

CACHE_ENV = "VERNIER_COCO_CACHE"

# Climb out of the `coco_val_cache/coco_val_cache/` package nesting to
# reach repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "coco-val2017"

#: The perfect-DT synthesizer (bbox + segm flavors). Kept as a
#: standalone CLI script so it remains usable independently; we
#: subprocess it from :func:`ensure_perfect_dts` below.
_MAKE_PERFECT_DT = _REPO_ROOT / "tools" / "make-perfect-dt.py"

PERFECT_DT_BBOX_FILENAME = "perfect_dt.json"
PERFECT_DT_SEGM_FILENAME = "perfect_dt_segm.json"

#: Buffer size for streaming downloads + extractions. 1 MiB amortizes
#: per-syscall overhead on the 778 MB image zip without ballooning
#: peak RSS.
_COPY_BUF_SIZE = 1 << 20


def cache_root(override: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the cache directory.

    Precedence: explicit ``override`` arg, then ``$VERNIER_COCO_CACHE``,
    then :data:`DEFAULT_CACHE_DIR`. The directory is *not* created here
    — the ``ensure_*`` helpers do the mkdir.
    """
    if override is not None:
        return Path(override).expanduser()
    env = os.environ.get(CACHE_ENV)
    return Path(env).expanduser() if env else DEFAULT_CACHE_DIR


def file_sha256(path: Path) -> str:
    """Streaming SHA256. 1 MiB chunks bound peak memory on large files."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while buf := f.read(_COPY_BUF_SIZE):
            h.update(buf)
    return h.hexdigest()


def _atomic_download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest``, atomic-on-success.

    Writes to ``<dest>.part`` then renames. A SIGINT mid-download
    leaves a stale ``.part`` the next run can overwrite cleanly; it
    cannot leave a half-written file at the canonical name that a
    later run would mistake for complete.
    """
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, part.open("wb") as f:
        shutil.copyfileobj(response, f, length=_COPY_BUF_SIZE)
    part.replace(dest)


def _images_dir_ok(images_dir: Path) -> bool:
    """Probe + count integrity check for ``val2017/``."""
    if not (images_dir / IMAGES_PROBE_FILENAME).is_file():
        return False
    n = sum(1 for p in images_dir.iterdir() if p.suffix == ".jpg" and p.is_file())
    return n == IMAGES_EXPECTED_COUNT


def _ensure_inner_zip_member(
    *, cache: Path | None, filename: str, inner_path: str, expected_sha: str
) -> Path:
    """Return a verified path to ``filename`` inside the upstream
    ``annotations_trainval2017.zip``. Idempotent under the SHA pin.
    """
    cache = cache_root(cache)
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / filename

    if dest.is_file() and file_sha256(dest) == expected_sha:
        return dest

    dest.unlink(missing_ok=True)
    zip_path = cache / "annotations_trainval2017.zip"
    try:
        _atomic_download(GT_URL, zip_path)
        with (
            zipfile.ZipFile(zip_path) as z,
            z.open(inner_path) as src,
            dest.open("wb") as dst,
        ):
            shutil.copyfileobj(src, dst, length=_COPY_BUF_SIZE)
    finally:
        zip_path.unlink(missing_ok=True)

    actual = file_sha256(dest)
    if actual != expected_sha:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"{filename} SHA256 mismatch: expected {expected_sha}, got {actual}.")
    return dest


def ensure_gt(*, cache: Path | None = None) -> Path:
    """Return a verified path to the instances GT JSON."""
    return _ensure_inner_zip_member(
        cache=cache,
        filename=GT_FILENAME,
        inner_path=GT_INNER_PATH,
        expected_sha=GT_SHA256,
    )


def ensure_kp_gt(*, cache: Path | None = None) -> Path:
    """Return a verified path to the keypoints GT JSON."""
    return _ensure_inner_zip_member(
        cache=cache,
        filename=KP_GT_FILENAME,
        inner_path=KP_GT_INNER_PATH,
        expected_sha=KP_GT_SHA256,
    )


def ensure_images(*, cache: Path | None = None) -> Path:
    """Return a verified path to ``val2017/``, downloading if necessary.

    Idempotent. Integrity check is the canonical-filename probe plus
    the 5000-jpg count. Raises ``RuntimeError`` if the post-extract
    check fails.
    """
    cache = cache_root(cache)
    cache.mkdir(parents=True, exist_ok=True)
    images_dir = cache / IMAGES_DIRNAME

    if _images_dir_ok(images_dir):
        return images_dir

    if images_dir.is_dir():
        # Stale / partial — clean slate before re-extracting. unzip's
        # incremental overwrite could leave orphan files from a prior
        # extraction; rm is the safe idempotent choice.
        shutil.rmtree(images_dir)

    zip_path = cache / Path(IMAGES_URL).name
    try:
        _atomic_download(IMAGES_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(cache)
    finally:
        zip_path.unlink(missing_ok=True)

    if not _images_dir_ok(images_dir):
        raise RuntimeError(
            f"post-extract integrity check failed for {images_dir}/: "
            f"expected {IMAGES_EXPECTED_COUNT} jpgs including "
            f"{IMAGES_PROBE_FILENAME}"
        )
    return images_dir


def ensure_perfect_dts(*, cache: Path | None = None) -> tuple[Path, Path]:
    """Return ``(bbox_dt_path, segm_dt_path)``, synthesizing if absent.

    Idempotent. The synthesizer is :mod:`make-perfect-dt`; we
    subprocess it to keep that script self-contained and CLI-usable
    in its own right. Calls :func:`ensure_gt` first.
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
    """``python -m coco_val_cache [--with-images]`` entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m coco_val_cache",
        description="Populate the COCO val2017 dev cache (GT JSON + perfect-DTs, "
        "optionally val2017/ images).",
    )
    parser.add_argument(
        "--with-images",
        action="store_true",
        help="Also download and extract val2017/ images (~778 MB zipped, "
        "~6.2 GB extracted). Required by inference harnesses (real-model "
        "TIDE validation); not needed by the parity smoke.",
    )
    args = parser.parse_args(argv)

    cache = cache_root()
    print(f"Cache directory: {cache}")
    gt = ensure_gt(cache=cache)
    print(f"GT ready: {gt}")
    bbox_dt, segm_dt = ensure_perfect_dts(cache=cache)
    print(f"Perfect bbox DT: {bbox_dt}")
    print(f"Perfect segm DT: {segm_dt}")
    if args.with_images:
        images_dir = ensure_images(cache=cache)
        print(f"val2017 images ready: {images_dir} ({IMAGES_EXPECTED_COUNT} jpgs)")
    print()
    print("Export the env vars (or eval the lines below) and run `just test-coco-val`:")
    print(f"  export VERNIER_COCO_GT_PATH={gt}")
    print(f"  export VERNIER_COCO_DT_PATH={bbox_dt}")
    print(f"  export VERNIER_COCO_DT_SEGM_PATH={segm_dt}")
    if args.with_images:
        print()
        print(
            "Images are picked up automatically by the real-model harness — "
            "no extra env var, just run:"
        )
        print("  uv run --extra real-models pytest -m real_models -v")
    return 0
