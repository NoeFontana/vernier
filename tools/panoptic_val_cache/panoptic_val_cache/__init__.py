"""Single source of truth for the COCO panoptic val2017 dev cache contract.

Parallel to :mod:`coco_val_cache` (the COCO val2017 cache) and
:mod:`lvis_val_cache` (the LVIS v1 val cache). The panoptic
whole-dataset parity smoke (ADR-0025) consumes the same idempotent
fetch+verify flow against the COCO panoptic GT JSON + per-image
PNG label maps.

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
import io
import json
import os
import shutil
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

GT_URL = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"
GT_INNER_JSON = "annotations/panoptic_val2017.json"
GT_INNER_PNG_ZIP = "annotations/panoptic_val2017.zip"
GT_NESTED_PNG_PREFIX = "panoptic_val2017/"
GT_JSON_FILENAME = "panoptic_val2017.json"
GT_PNG_DIRNAME = "panoptic_val2017"
#: SHA-256 of the panoptic-annotations zip artifact at upstream
#: ``cocodataset.org`` as observed 2026-05-07. Bumping is an
#: ADR-level decision per ADR-0025 §"Parity strategy" — every quirk
#: vernier reproduces is keyed to this exact byte sequence.
GT_ZIP_SHA256 = "c05f76d2129b6b561eb70efe16e7006df62f73fb92889132d373b9d90e31a370"

CACHE_ENV = "VERNIER_PANOPTIC_CACHE"

# Climb out of `panoptic_val_cache/panoptic_val_cache/` to repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "panoptic-val2017"

PERFECT_DT_JSON_FILENAME = "perfect_dt.json"
PERFECT_DT_PNG_DIRNAME = "perfect_dt_pngs"

# Semantic-derivation artifacts — produced by :func:`ensure_semantic_gt`
# and :func:`ensure_semantic_perfect_dt`. The semantic class set is the
# 133 panoptic categories (80 thing + 53 stuff) mapped to a contiguous
# 0..132 train-id by ascending sort of upstream ``category_id``.
SEMANTIC_GT_DIRNAME = "semantic_val2017_gt"
SEMANTIC_DT_DIRNAME = "semantic_val2017_perfect_dt"
SEMANTIC_MAPPING_FILENAME = "semantic_train_id_to_category_id.json"
#: Sentinel for unlabeled pixels (panoptic ``segment_id == 0``). Matches
#: the Cityscapes / Pascal VOC convention; safely outside the 0..132
#: train-id range.
SEMANTIC_IGNORE_LABEL = 255

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
            with z.open(GT_INNER_JSON) as src, json_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=_COPY_BUF_SIZE)
            nested_zip_bytes = z.read(GT_INNER_PNG_ZIP)
        png_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(nested_zip_bytes)) as inner:
            for name in inner.namelist():
                if not name.startswith(GT_NESTED_PNG_PREFIX) or name.endswith("/"):
                    continue
                target = png_dir / Path(name).name
                with inner.open(name) as src, target.open("wb") as dst:
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

    # Symlinks (vs full copies, ~3 GB on COCO panoptic val2017) — the
    # panopticapi oracle reads via PIL.Image.open, which follows them.
    _symlink_pngs(png_dir, dt_png_dir)

    return dt_json, dt_png_dir


def _symlink_pngs(src_dir: Path, dst_dir: Path) -> None:
    """Symlink every regular file in ``src_dir`` into ``dst_dir``.

    Idempotent: pre-existing entries (symlink or otherwise) are left
    untouched. Caller mkdirs ``dst_dir``.
    """
    for entry in src_dir.iterdir():
        if not entry.is_file():
            continue
        target = dst_dir / entry.name
        if target.is_symlink() or target.exists():
            continue
        target.symlink_to(entry.resolve())


def scan_label_map_pngs(directory: Path) -> dict[int, Path]:
    """Index ``<int>.png`` files in ``directory`` by parsed image_id.

    Single source of truth shared by parity tests and bench runners
    over the cached label-map dirs. Filenames not matching the
    ``<int>.png`` convention raise :class:`ValueError`.
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


def _build_semantic_train_id_map(categories: list[dict[str, object]]) -> dict[int, int]:
    """Map upstream ``category_id`` → contiguous ``train_id`` by ascending sort.

    The mapping is independent of JSON key ordering — the cache identity
    is the cache directory, so a re-sort of ``categories`` upstream must
    not change the train-id assignment.
    """
    cat_ids = sorted(int(c["id"]) for c in categories)  # type: ignore[arg-type]
    return {cat_id: train_id for train_id, cat_id in enumerate(cat_ids)}


def _convert_panoptic_to_semantic(
    panoptic_png: Path,
    segments_info: list[dict[str, object]],
    cat_id_to_train_id: dict[int, int],
) -> np.ndarray:
    """Decode a panoptic RGB PNG into a uint8 semantic label-map.

    ``segment_id == 0`` (unlabeled) maps to :data:`SEMANTIC_IGNORE_LABEL`.
    Other segment ids look up their ``category_id`` in ``segments_info``
    and convert to the contiguous train-id. Segment ids absent from
    ``segments_info`` get ignore_label too — fail-safe against
    malformed annotations.
    """
    import numpy as np
    from PIL import Image

    rgb = np.asarray(Image.open(panoptic_png), dtype=np.uint32)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(
            f"panoptic PNG {panoptic_png!s} must be RGB (3 channels); got shape {rgb.shape!r}"
        )
    segment_ids = rgb[..., 0] + rgb[..., 1] * 256 + rgb[..., 2] * 65536

    max_id = int(segment_ids.max(initial=0))
    lut = np.full(max_id + 1, SEMANTIC_IGNORE_LABEL, dtype=np.uint8)
    for seg in segments_info:
        seg_id = int(seg["id"])  # type: ignore[arg-type]
        if seg_id > max_id or seg_id == 0:
            continue
        cat_id = int(seg["category_id"])  # type: ignore[arg-type]
        train_id = cat_id_to_train_id.get(cat_id)
        if train_id is None:
            continue
        lut[seg_id] = np.uint8(train_id)
    return lut[segment_ids]


def ensure_semantic_gt(*, cache: Path | None = None) -> tuple[Path, int, dict[int, int]]:
    """Materialize the semantic GT label-map PNGs derived from panoptic GT.

    Returns ``(gt_dir, n_classes, train_id_to_category_id)``. The
    reverse map is the on-disk artifact for downstream consumers
    (parity tests + bench runners) that need to label per-class
    metric rows. Idempotent: subsequent calls skip work when the
    output dir is populated and the mapping JSON exists.

    Calls :func:`ensure_gt` to satisfy the panoptic GT cache before
    deriving — the caller does not need to call it directly.
    """
    from PIL import Image

    json_path, png_dir = ensure_gt(cache=cache)
    cache_dir = json_path.parent
    out_dir = cache_dir / SEMANTIC_GT_DIRNAME
    mapping_path = cache_dir / SEMANTIC_MAPPING_FILENAME

    if out_dir.is_dir() and mapping_path.is_file() and any(out_dir.iterdir()):
        with mapping_path.open() as f:
            mapping = {int(k): int(v) for k, v in json.load(f).items()}
        return out_dir, len(mapping), mapping

    with json_path.open() as f:
        gt = json.load(f)
    cat_id_to_train_id = _build_semantic_train_id_map(gt["categories"])
    train_id_to_cat_id = {tid: cid for cid, tid in cat_id_to_train_id.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    for ann in gt["annotations"]:
        image_id = int(ann["image_id"])
        src_png = png_dir / ann["file_name"]
        label_map = _convert_panoptic_to_semantic(src_png, ann["segments_info"], cat_id_to_train_id)
        Image.fromarray(label_map, mode="L").save(out_dir / f"{image_id}.png")

    mapping_path.write_text(
        json.dumps({str(k): v for k, v in train_id_to_cat_id.items()}, indent=2)
    )
    return out_dir, len(cat_id_to_train_id), train_id_to_cat_id


def ensure_semantic_perfect_dt(*, cache: Path | None = None) -> Path:
    """Symlink the semantic GT into a perfect-DT directory.

    Returns the DT directory path. Perfect DT == GT, so each output PNG
    is a symlink to its GT counterpart (zero-cost vs full copy on a
    val2017-scale 25 MB tree). Idempotent: skips work when ``dt_dir``
    is already populated.
    """
    gt_dir, _, _ = ensure_semantic_gt(cache=cache)
    cache_dir = gt_dir.parent
    dt_dir = cache_dir / SEMANTIC_DT_DIRNAME

    if dt_dir.is_dir() and any(dt_dir.iterdir()):
        return dt_dir

    dt_dir.mkdir(parents=True, exist_ok=True)
    _symlink_pngs(gt_dir, dt_dir)
    return dt_dir


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
