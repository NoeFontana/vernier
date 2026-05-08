"""Real-prediction cache for panoptic models (Stage 3 / S3-A).

Mirrors the ``ensure_maskrcnn`` pattern from this package's
``__init__``: SHA-pinned URL fetcher with an idempotent on-disk
cache. Stub URLs/SHAs (``None``) until the prediction blobs are
hosted, at which point :func:`ensure_mask2former` swaps to a working
fetcher without code-path change.

Mask2Former on COCO panoptic val2017 is the v1 anchor; additional
models (e.g. OneFormer) follow the same shape with their own blob
version.

The cached artifact is whatever the upstream URL serves — typically
a ``.tar.gz`` containing the PNG segment dir + ``segments_info.json``.
This module verifies the SHA over the downloaded bytes; the caller
is responsible for extracting and consuming the bundle (panopticapi
expects an extracted directory layout).
"""

from __future__ import annotations

from pathlib import Path

from coco_val_cache import _atomic_download, file_sha256

from real_predictions_cache import cache_root

#: Cache-key version. Bump when the upstream model version, weights,
#: or post-processing changes; keeps a previously-cached blob from
#: silently masking a new fetch.
MASK2FORMER_BLOB_VERSION = "v1"

#: Pinned Mask2Former (Swin-L, COCO panoptic val2017) prediction URL.
#: ``None`` until the blob is hosted; :func:`ensure_mask2former`
#: raises a clear ``RuntimeError`` rather than silently succeeding.
MASK2FORMER_URL: str | None = None

#: SHA256 of the prediction blob at :data:`MASK2FORMER_URL`. Fill both
#: atomically when the upload lands.
MASK2FORMER_SHA256: str | None = None

_DATASET_ID = "coco-panoptic-val2017"


def mask2former_cache_filename() -> str:
    return f"mask2former-swinl-{MASK2FORMER_BLOB_VERSION}-{_DATASET_ID}.tar.gz"


def mask2former_cache_path(*, cache: Path | None = None) -> Path:
    return cache_root(cache) / mask2former_cache_filename()


def ensure_mask2former(
    *,
    cache: Path | None = None,
    url: str | None = None,
    sha256: str | None = None,
) -> Path:
    """Return a verified path to the Mask2Former prediction blob,
    downloading if necessary.

    Idempotent: a cached file matching ``sha256`` short-circuits
    without network I/O. ``url`` and ``sha256`` default to
    module-level :data:`MASK2FORMER_URL` / :data:`MASK2FORMER_SHA256`,
    both ``None`` until the upload lands.

    Raises ``RuntimeError`` if the URL/SHA aren't configured (clearer
    than a 404 when the bench tries to run a panoptic real-prediction
    cell pre-Stage-3).
    """
    final_url = url if url is not None else MASK2FORMER_URL
    final_sha = sha256 if sha256 is not None else MASK2FORMER_SHA256
    if final_url is None or final_sha is None:
        raise RuntimeError(
            "Mask2Former prediction blob URL/SHA256 not yet configured. "
            "Set MASK2FORMER_URL and MASK2FORMER_SHA256 in "
            "tools/real_predictions_cache/real_predictions_cache/panoptic.py "
            "once the tarball is hosted, or pass url=/sha256= explicitly."
        )

    cache_dir = cache_root(cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / mask2former_cache_filename()

    if out.is_file() and file_sha256(out) == final_sha:
        return out

    out.unlink(missing_ok=True)
    _atomic_download(final_url, out)
    actual = file_sha256(out)
    if actual != final_sha:
        out.unlink(missing_ok=True)
        raise RuntimeError(
            f"Mask2Former prediction blob SHA256 mismatch: expected {final_sha}, got {actual}."
        )
    return out


__all__ = [
    "MASK2FORMER_BLOB_VERSION",
    "MASK2FORMER_SHA256",
    "MASK2FORMER_URL",
    "ensure_mask2former",
    "mask2former_cache_filename",
    "mask2former_cache_path",
]
