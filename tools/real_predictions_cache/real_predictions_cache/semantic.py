"""Real-prediction cache for semantic-segmentation models (Stage 3 / S3-A + S3-B).

Two model anchors:

- HRNet on Cityscapes val (S3-A) — strict-tier oracle path is
  cityscapesScripts (already in ``bench/envs/cityscapes/``).
- OCRNet on ADE20K val (S3-B) — needs the mmsegmentation env, which
  is the heavy ~5–8 min ``uv sync``. The cache adapter lands in
  Stage 3 even when the env is deferred so workload modules can plug
  in once both pieces are available.

Same shape as :mod:`real_predictions_cache.panoptic` — SHA-pinned URL
fetcher, idempotent on-disk cache. Stub URLs/SHAs (``None``) until
hosting lands.

The cached artifact is typically a ``.tar.gz`` containing per-image
class-id PNGs (one per val image, named by image id). Consumers
extract and feed the directory to the runner.
"""

from __future__ import annotations

from pathlib import Path

from coco_val_cache import _atomic_download, file_sha256

from real_predictions_cache import cache_root

# --- HRNet on Cityscapes val ----------------------------------------

HRNET_CITYSCAPES_BLOB_VERSION = "v1"
HRNET_CITYSCAPES_URL: str | None = None
HRNET_CITYSCAPES_SHA256: str | None = None

_HRNET_DATASET_ID = "cityscapes-val"


def hrnet_cityscapes_cache_filename() -> str:
    return (
        f"hrnet-w48-{HRNET_CITYSCAPES_BLOB_VERSION}-{_HRNET_DATASET_ID}.tar.gz"
    )


def hrnet_cityscapes_cache_path(*, cache: Path | None = None) -> Path:
    return cache_root(cache) / hrnet_cityscapes_cache_filename()


def ensure_hrnet_cityscapes(
    *,
    cache: Path | None = None,
    url: str | None = None,
    sha256: str | None = None,
) -> Path:
    """Return a verified path to the HRNet/Cityscapes prediction blob,
    downloading if necessary. Stub until the URL/SHA are pinned."""
    final_url = url if url is not None else HRNET_CITYSCAPES_URL
    final_sha = sha256 if sha256 is not None else HRNET_CITYSCAPES_SHA256
    if final_url is None or final_sha is None:
        raise RuntimeError(
            "HRNet/Cityscapes prediction blob URL/SHA256 not yet configured. "
            "Set HRNET_CITYSCAPES_URL and HRNET_CITYSCAPES_SHA256 in "
            "tools/real_predictions_cache/real_predictions_cache/semantic.py "
            "once the tarball is hosted, or pass url=/sha256= explicitly."
        )
    return _ensure(
        url=final_url,
        sha256=final_sha,
        filename=hrnet_cityscapes_cache_filename(),
        cache=cache,
        label="HRNet/Cityscapes",
    )


# --- OCRNet on ADE20K val -------------------------------------------

OCRNET_ADE20K_BLOB_VERSION = "v1"
OCRNET_ADE20K_URL: str | None = None
OCRNET_ADE20K_SHA256: str | None = None

_OCRNET_DATASET_ID = "ade20k-val"


def ocrnet_ade20k_cache_filename() -> str:
    return (
        f"ocrnet-hrnet-w48-{OCRNET_ADE20K_BLOB_VERSION}-{_OCRNET_DATASET_ID}.tar.gz"
    )


def ocrnet_ade20k_cache_path(*, cache: Path | None = None) -> Path:
    return cache_root(cache) / ocrnet_ade20k_cache_filename()


def ensure_ocrnet_ade20k(
    *,
    cache: Path | None = None,
    url: str | None = None,
    sha256: str | None = None,
) -> Path:
    """Return a verified path to the OCRNet/ADE20K prediction blob,
    downloading if necessary. Stub until the URL/SHA are pinned. Note
    that ADE20K parity also needs the mmseg env (S3-B); the cache
    works without it but the bench cell does not."""
    final_url = url if url is not None else OCRNET_ADE20K_URL
    final_sha = sha256 if sha256 is not None else OCRNET_ADE20K_SHA256
    if final_url is None or final_sha is None:
        raise RuntimeError(
            "OCRNet/ADE20K prediction blob URL/SHA256 not yet configured. "
            "Set OCRNET_ADE20K_URL and OCRNET_ADE20K_SHA256 in "
            "tools/real_predictions_cache/real_predictions_cache/semantic.py "
            "once the tarball is hosted, or pass url=/sha256= explicitly."
        )
    return _ensure(
        url=final_url,
        sha256=final_sha,
        filename=ocrnet_ade20k_cache_filename(),
        cache=cache,
        label="OCRNet/ADE20K",
    )


def _ensure(
    *,
    url: str,
    sha256: str,
    filename: str,
    cache: Path | None,
    label: str,
) -> Path:
    """Shared download+verify path. Both semantic anchors share this
    body; only the model name and dataset id differ."""
    cache_dir = cache_root(cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / filename

    if out.is_file() and file_sha256(out) == sha256:
        return out

    out.unlink(missing_ok=True)
    _atomic_download(url, out)
    actual = file_sha256(out)
    if actual != sha256:
        out.unlink(missing_ok=True)
        raise RuntimeError(
            f"{label} prediction blob SHA256 mismatch: expected {sha256}, got {actual}."
        )
    return out


__all__ = [
    "HRNET_CITYSCAPES_BLOB_VERSION",
    "HRNET_CITYSCAPES_SHA256",
    "HRNET_CITYSCAPES_URL",
    "OCRNET_ADE20K_BLOB_VERSION",
    "OCRNET_ADE20K_SHA256",
    "OCRNET_ADE20K_URL",
    "ensure_hrnet_cityscapes",
    "ensure_ocrnet_ade20k",
    "hrnet_cityscapes_cache_filename",
    "hrnet_cityscapes_cache_path",
    "ocrnet_ade20k_cache_filename",
    "ocrnet_ade20k_cache_path",
]
