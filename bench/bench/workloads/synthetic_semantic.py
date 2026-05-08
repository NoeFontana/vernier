"""Synthetic semantic-segmentation workload generator (ADR-0033 §B2).

Materializes deterministic per-image GT and DT label-map PNGs into
``bench/.cache/synthetic_semantic/<workload_id>/{gt,dt}/`` keyed by
``(n_images, n_classes, seed, jitter_rate)``. Idempotent: skips
generation when the per-image PNG count under the cache dir already
matches ``n_images`` for both sides.

This is the vernier-only baseline cell while the S3-B oracle (mmseg +
ADE20K) stays externally blocked. The perf signal is the streaming
``Evaluator.stream()`` path against a known label-map shape — the
kernel is correctness-bound under the Rust cargo bench, but the
Python harness is what reveals end-to-end (decode + FFI + fold)
characteristics on a realistic-sized image stack.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from bench.harness.paths import bench_cache_root

# Image canvas. Fixed (256x256) so the workload identity stays tied
# to the explicit param tuple; varying H/W would force every cache key
# to grow. 256x256 keeps generation sub-second for 200 images while
# being large enough that decode-time isn't a microbench artifact.
IMAGE_H = 256
IMAGE_W = 256

# Pixel-flip rate: fraction of GT pixels rewritten to a different
# class to produce the DT label map. Mirrors the Gaussian-jitter
# patten on instance synthetic, scaled to per-pixel granularity.
DEFAULT_JITTER_RATE = 0.1

# Default ignore label — matches the Cityscapes / Pascal VOC
# convention. A subset of GT pixels (5% by default) gets rewritten to
# this class so the ignore-handling path actually exercises.
DEFAULT_IGNORE_LABEL = 255
_IGNORE_FRACTION = 0.05

_WORKLOAD_ID_RE = re.compile(
    r"^synthetic_semantic"
    r"_n(?P<n_images>\d+)"
    r"_c(?P<n_classes>\d+)"
    r"(?:_j(?P<jitter_pct>\d+))?"
    r"_s(?P<seed>\d+)$"
)


def workload_id(
    *,
    n_images: int,
    n_classes: int,
    seed: int,
    jitter_rate: float = DEFAULT_JITTER_RATE,
) -> str:
    base = f"synthetic_semantic_n{n_images}_c{n_classes}"
    if jitter_rate != DEFAULT_JITTER_RATE:
        return f"{base}_j{int(jitter_rate * 100):02d}_s{seed}"
    return f"{base}_s{seed}"


def parse_workload_id(wid: str) -> dict[str, int | float] | None:
    """Inverse of :func:`workload_id`; ``None`` for non-matching ids."""
    m = _WORKLOAD_ID_RE.match(wid)
    if m is None:
        return None
    out: dict[str, int | float] = {
        "n_images": int(m["n_images"]),
        "n_classes": int(m["n_classes"]),
        "seed": int(m["seed"]),
    }
    if m["jitter_pct"] is not None:
        out["jitter_rate"] = int(m["jitter_pct"]) / 100.0
    return out


def _cache_dir() -> Path:
    return bench_cache_root() / "synthetic_semantic"


def _save_label_map_png(arr: np.ndarray, path: Path) -> None:
    """Write a single-channel uint8 PNG. Caller pins n_classes < 256."""
    from PIL import Image

    Image.fromarray(arr.astype(np.uint8, copy=False), mode="L").save(path)


def _generate_pair(
    rng: np.random.Generator,
    *,
    n_classes: int,
    jitter_rate: float,
    ignore_label: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one (gt, dt) pair. GT is a coarse blob field (non-uniform
    class assignment so per-class IoUs aren't all the same); DT flips
    a fraction of pixels to a random other class.
    """
    # Coarse blob field via downsample-then-upsample: 16x16 random
    # class draws upsampled to (H, W). Realistic semantic GT is spatially
    # smooth, not uniform random; this approximates that.
    coarse = rng.integers(0, n_classes, size=(16, 16), dtype=np.int32)
    gt = np.repeat(np.repeat(coarse, IMAGE_H // 16, axis=0), IMAGE_W // 16, axis=1)

    # Sprinkle ignore-label pixels (5%) so the ignore-handling path
    # gets exercised on every workload with an ignore label set.
    ignore_mask = rng.random((IMAGE_H, IMAGE_W)) < _IGNORE_FRACTION
    gt = np.where(ignore_mask, ignore_label, gt).astype(np.int32, copy=False)

    # DT: GT with `jitter_rate` of pixels flipped to a random other
    # class. Pixels masked as ignore in GT can hold any DT value;
    # the kernel ignores them anyway.
    flip_mask = rng.random((IMAGE_H, IMAGE_W)) < jitter_rate
    flips = rng.integers(0, n_classes, size=(IMAGE_H, IMAGE_W), dtype=np.int32)
    # Avoid self-assignment: bump same-class flips by 1 (mod n_classes).
    same_class = flips == gt
    flips = np.where(same_class, (flips + 1) % n_classes, flips)
    dt = np.where(flip_mask, flips, gt).astype(np.int32, copy=False)
    # DT must not contain the ignore label (semantic predictions are
    # in [0, n_classes); ignore is GT-only).
    dt = np.where(dt == ignore_label, 0, dt)

    return gt, dt


def make_workload(
    *,
    n_images: int,
    n_classes: int,
    seed: int,
    jitter_rate: float = DEFAULT_JITTER_RATE,
    ignore_label: int = DEFAULT_IGNORE_LABEL,
) -> tuple[Path, Path]:
    """Return ``(gt_dir, dt_dir)`` paths, materializing the cached pair
    if missing. Idempotent: re-running with the same params is a
    no-op.
    """
    if n_classes < 1 or n_classes > 255:
        raise ValueError(
            f"synthetic_semantic supports n_classes in [1, 255]; got {n_classes!r}. "
            f"Single-channel uint8 PNGs are the storage shape."
        )
    if not 0.0 <= jitter_rate <= 1.0:
        raise ValueError(f"jitter_rate must be in [0.0, 1.0]; got {jitter_rate!r}")

    wid = workload_id(n_images=n_images, n_classes=n_classes, seed=seed, jitter_rate=jitter_rate)
    base = _cache_dir() / wid
    gt_dir = base / "gt"
    dt_dir = base / "dt"
    # The marker is written last, so a partially-generated cache
    # (interrupted mid-loop) won't be mistaken for complete on the
    # next run. Counting PNGs would give false positives.
    done_marker = base / ".done"
    if done_marker.exists():
        return gt_dir, dt_dir

    gt_dir.mkdir(parents=True, exist_ok=True)
    dt_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for image_id in range(n_images):
        gt_arr, dt_arr = _generate_pair(
            rng,
            n_classes=n_classes,
            jitter_rate=jitter_rate,
            ignore_label=ignore_label,
        )
        _save_label_map_png(gt_arr, gt_dir / f"{image_id}.png")
        _save_label_map_png(dt_arr, dt_dir / f"{image_id}.png")
    done_marker.touch()
    return gt_dir, dt_dir
