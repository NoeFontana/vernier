"""Deterministic jittered detections from a COCO GT JSON.

For every non-crowd GT annotation, emit a detection whose bbox is the
GT bbox + Gaussian noise on ``(x, y, w, h)``, score drawn from a beta
biased toward 1, and segmentation produced by mask-space jitter on the
GT mask (dilate ``k_d ~ Poisson(λ=1)``, erode ``k_e ~ Poisson(λ=1)``,
both clipped to :data:`MASK_KERNEL_CLIP`; integer pixel translation
``(dy, dx) ~ N(0, sigma=2px)``). A controlled fraction of GTs is dropped
(false negatives) and the same fraction is added at random positions
(false positives — segm field on FPs is the filled bbox rectangle, so
segm/boundary runners can consume the same DT stream as bbox runners).

The cache key is ``(seed, JITTER_PARAMS_VERSION)`` — bump
:data:`JITTER_PARAMS_VERSION` if any default changes so old caches
self-invalidate.

Mask jitter draws come from an independent SeedSequence stream so the
v1→v2 upgrade preserves seed=N's bbox/score/FP byte-identity (anyone
who captured perf numbers under v1 sees the same bbox-row distribution
under v2; the segm/boundary rows are new). Encode/decode go through
pycocotools, not vernier-mask, since using the system under test as
the codec for its own test data would be circular.

Keypoint jitter (separate workload, separate cache file) applies
positional noise scaled per-keypoint by ``COCO sigmas * sqrt(area)``,
and flips visibility flags with low probability. Per ADR-0012 the COCO
17-person sigmas are the canonical OKS scale; the bench mirrors them
here from ``crates/vernier-core/src/similarity/oks.rs`` (post-divide-by-
10) so the jitter scale aligns with the OKS denominator and produces
realistic AP deltas. Keypoint jitter has its own SeedSequence stream so
its outputs are independent of the bbox/segm streams above.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy.ndimage as ndi
from pycocotools import mask as mask_utils

if TYPE_CHECKING:
    from pycocotools.mask import _EncodedRLE

from bench.harness.paths import bench_cache_root

# Synthetic detection scores are drawn from this beta. Single source of
# truth so jittered + synthetic stay aligned without silent drift.
SCORE_BETA_ALPHA = 5.0
SCORE_BETA_BETA = 1.0

# Single bump-on-default-change number, shared by the workload_id and
# cache filename so altering any default invalidates pre-existing
# detections without per-knob bookkeeping. v2 added mask-space jitter
# on a side stream — bbox/score/FP draws are identical to v1.
JITTER_PARAMS_VERSION = 2

# Easy preset (per ADR-0017 §"Workloads → COCO val2017") — small bbox
# jitter, modest FP/FN, scores biased toward correct so ranking is
# realistic but not pathologically tied.
BBOX_JITTER_SIGMA_PX = 5.0
FP_FRACTION = 0.05
FN_FRACTION = 0.05

# Mask-space jitter (v2): structurally resembles real Mask R-CNN errors
# (fattened/thinned/translated). Vertex-space jitter would push self-
# intersecting polygons through pycocotools' ``rleFrPoly`` path
# (boundary-iou-quirks H3), a parity disposition we keep out of the
# test distribution.
MASK_DILATE_LAMBDA = 1.0
MASK_ERODE_LAMBDA = 1.0
MASK_KERNEL_CLIP = 5
MASK_TRANSLATE_SIGMA_PX = 2.0

# Spawn key for the mask-jitter side stream. Pinned so v2 is reproducible
# across NumPy versions; treat as a one-time ABI choice.
_MASK_RNG_SPAWN_KEY = (0x6D61736B,)  # b"mask"

# COCO 17-keypoint sigmas (post-divide-by-10), mirroring
# ``crates/vernier-core/src/similarity/oks.rs::COCO_PERSON_SIGMAS``.
# Bench-side jitter scales the per-keypoint Gaussian by ``sigma_i *
# sqrt(area)`` so the perturbation magnitude maps onto the OKS
# denominator: a unit jitter at keypoint ``i`` corresponds to one
# OKS-equivalent on a unit-area object.
COCO_PERSON_SIGMAS: tuple[float, ...] = (
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
)

# Keypoint jitter knobs. ``KP_JITTER_SCALE`` sets the per-keypoint
# Gaussian std as ``KP_JITTER_SCALE * sigma_i * sqrt(area)``; with
# 1.0 the resulting OKS distribution centers around realistic
# detector-vs-GT errors. Visibility flips are rare so
# ``num_keypoints`` stays meaningful for the F3 surrogate branch.
KP_JITTER_SCALE = 1.5
KP_VISIBILITY_FLIP_PROB = 0.02

# Spawn key for the keypoint-jitter side stream. Independent of bbox
# and mask streams so seed=N's keypoint workload is reproducible
# without mass-perturbing other workloads' caches.
_KP_RNG_SPAWN_KEY = (0x6B70_6B70,)  # b"kpkp"


def _jittered_dt_path(seed: int) -> Path:
    cache = bench_cache_root() / "jittered"
    return cache / f"coco_val2017_jittered_seed{seed}_v{JITTER_PARAMS_VERSION}.json"


def workload_id(seed: int) -> str:
    return f"coco_val2017_jittered_seed{seed}"


def keypoints_workload_id(seed: int) -> str:
    return f"coco_val2017_keypoints_jittered_seed{seed}"


def _keypoints_jittered_dt_path(seed: int) -> Path:
    cache = bench_cache_root() / "jittered"
    return cache / f"coco_val2017_keypoints_jittered_seed{seed}_v{JITTER_PARAMS_VERSION}.json"


def _decode_segm_to_mask(seg: Any, *, height: int, width: int) -> np.ndarray | None:
    """Decode a COCO segmentation field to an HxW bool mask, or None if absent/empty."""
    if seg is None:
        return None
    rle: _EncodedRLE
    if isinstance(seg, list):
        if not seg:
            return None
        rle = mask_utils.merge(mask_utils.frPyObjects(seg, height, width))
    elif isinstance(seg, dict):
        counts = seg.get("counts")
        seg_rle = cast("_EncodedRLE", seg)
        if isinstance(counts, list):
            rle = cast("_EncodedRLE", mask_utils.frPyObjects(seg_rle, height, width))
        else:
            rle = seg_rle
    else:
        return None
    decoded = mask_utils.decode(rle)
    return np.asarray(decoded, dtype=bool)


def _apply_mask_jitter(
    mask: np.ndarray,
    *,
    k_dilate: int,
    k_erode: int,
    dy: int,
    dx: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Dilate (k_dilate iters), erode (k_erode iters), translate by (dy, dx) integer pixels."""
    out = mask
    if k_dilate > 0:
        out = ndi.binary_dilation(out, iterations=k_dilate)
    if k_erode > 0:
        out = ndi.binary_erosion(out, iterations=k_erode)
    if dy != 0 or dx != 0:
        translated = np.zeros_like(out)
        dst_ys, dst_ye = max(0, dy), min(height, height + dy)
        dst_xs, dst_xe = max(0, dx), min(width, width + dx)
        src_ys, src_ye = max(0, -dy), min(height, height - dy)
        src_xs, src_xe = max(0, -dx), min(width, width - dx)
        if dst_ye > dst_ys and dst_xe > dst_xs:
            translated[dst_ys:dst_ye, dst_xs:dst_xe] = out[src_ys:src_ye, src_xs:src_xe]
        out = translated
    return out


def _encode_mask(mask: np.ndarray) -> tuple[dict[str, Any], int]:
    """Encode a 2-D bool mask as a COCO RLE dict with ASCII counts; returns ``(rle, area)``."""
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    area = int(mask_utils.area(encoded))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": list(encoded["size"]), "counts": counts}, area


def _filled_bbox_mask(*, height: int, width: int, bbox: list[float]) -> np.ndarray:
    x, y, w, h = bbox
    out = np.zeros((height, width), dtype=bool)
    y0 = max(0, round(y))
    x0 = max(0, round(x))
    y1 = min(height, round(y + h))
    x1 = min(width, round(x + w))
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = True
    return out


def _generate(*, gt_path: Path, seed: int, out: Path) -> None:
    with gt_path.open("rb") as f:
        gt = json.load(f)
    rng = np.random.default_rng(seed)
    # Independent stream so v1 bbox/score/FP byte-identity is preserved
    # under the v2 schema bump.
    mask_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=_MASK_RNG_SPAWN_KEY))

    images = gt.get("images", [])
    image_hw_by_id: dict[int, tuple[int, int]] = {
        int(img["id"]): (int(img["height"]), int(img["width"])) for img in images
    }

    non_crowd = [a for a in gt["annotations"] if not a.get("iscrowd", 0)]
    keep_mask = rng.random(len(non_crowd)) >= FN_FRACTION

    detections: list[dict[str, object]] = []
    for ann, keep in zip(non_crowd, keep_mask, strict=True):
        if not keep:
            continue
        x, y, w, h = ann["bbox"]
        noise = rng.normal(0.0, BBOX_JITTER_SIGMA_PX, size=4)
        jx, jy, jw, jh = x + noise[0], y + noise[1], w + noise[2], h + noise[3]
        # Clamp to non-degenerate positive width/height so loadRes
        # doesn't reject. pycocotools tolerates floats here.
        jw = max(jw, 1.0)
        jh = max(jh, 1.0)
        score = float(rng.beta(SCORE_BETA_ALPHA, SCORE_BETA_BETA))
        bbox = [float(jx), float(jy), float(jw), float(jh)]
        det: dict[str, object] = {
            "image_id": int(ann["image_id"]),
            "category_id": int(ann["category_id"]),
            "bbox": bbox,
            "score": score,
        }

        # Always advance the mask RNG so a GT without segmentation
        # doesn't desync the side stream from a GT with segmentation.
        k_dilate = min(int(mask_rng.poisson(MASK_DILATE_LAMBDA)), MASK_KERNEL_CLIP)
        k_erode = min(int(mask_rng.poisson(MASK_ERODE_LAMBDA)), MASK_KERNEL_CLIP)
        dy = round(float(mask_rng.normal(0.0, MASK_TRANSLATE_SIGMA_PX)))
        dx = round(float(mask_rng.normal(0.0, MASK_TRANSLATE_SIGMA_PX)))

        hw = image_hw_by_id.get(int(ann["image_id"]))
        if hw is not None:
            gt_mask = _decode_segm_to_mask(ann.get("segmentation"), height=hw[0], width=hw[1])
            if gt_mask is not None:
                jittered = _apply_mask_jitter(
                    gt_mask,
                    k_dilate=k_dilate,
                    k_erode=k_erode,
                    dy=dy,
                    dx=dx,
                    height=hw[0],
                    width=hw[1],
                )
                rle, area = _encode_mask(jittered)
                det["segmentation"] = rle
                det["area"] = float(area)

        detections.append(det)

    n_fp = round(FP_FRACTION * len(non_crowd))
    if n_fp and image_hw_by_id and gt.get("categories"):
        image_ids = list(image_hw_by_id.keys())
        category_ids = [int(c["id"]) for c in gt["categories"]]
        fp_image_idx = rng.integers(0, len(image_ids), size=n_fp)
        fp_cat_idx = rng.integers(0, len(category_ids), size=n_fp)
        fp_scores = rng.beta(SCORE_BETA_ALPHA, SCORE_BETA_BETA, size=n_fp)
        for i in range(n_fp):
            img_id = image_ids[int(fp_image_idx[i])]
            ih, iw = image_hw_by_id[img_id]
            w = float(rng.uniform(10.0, max(11.0, iw / 4.0)))
            h = float(rng.uniform(10.0, max(11.0, ih / 4.0)))
            x = float(rng.uniform(0.0, max(1.0, iw - w)))
            y = float(rng.uniform(0.0, max(1.0, ih - h)))
            fp_bbox = [x, y, w, h]
            det = {
                "image_id": img_id,
                "category_id": category_ids[int(fp_cat_idx[i])],
                "bbox": fp_bbox,
                "score": float(fp_scores[i]),
            }
            # FP segmentation is the filled bbox rectangle: a structurally
            # honest "this is what an unrecognized region looks like" mask
            # that lets segm/boundary runners consume the same DT stream
            # without a separate FP-suppression pass.
            hw = image_hw_by_id.get(int(img_id))
            if hw is not None:
                fp_mask = _filled_bbox_mask(height=hw[0], width=hw[1], bbox=fp_bbox)
                rle, area = _encode_mask(fp_mask)
                det["segmentation"] = rle
                det["area"] = float(area)
            detections.append(det)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(detections, f, separators=(",", ":"))


def dt_path(*, gt_path: Path, seed: int) -> Path:
    """Return the cached jittered DT, generating it if missing."""
    out = _jittered_dt_path(seed)
    if not out.exists():
        _generate(gt_path=gt_path, seed=seed, out=out)
    return out


def lvis_dt_path(*, gt_path: Path, seed: int) -> Path:
    """LVIS-keyed cache sibling of :func:`dt_path`.

    The same generator runs against the LVIS GT bytes; the only
    difference is the cache filename, so a same-seed COCO and LVIS
    jittered DT live in different files instead of clobbering each
    other. Seed identity is therefore a property of (workload, seed),
    not of (paradigm, seed).
    """
    cache = bench_cache_root() / "jittered"
    out = cache / f"lvis_v1_val_jittered_seed{seed}_v{JITTER_PARAMS_VERSION}.json"
    if not out.exists():
        _generate(gt_path=gt_path, seed=seed, out=out)
    return out


def _generate_keypoints(*, gt_path: Path, seed: int, out: Path) -> None:
    """Emit a keypoints DT JSON: one detection per non-crowd GT with a
    keypoint vector. Per-keypoint Gaussian jitter is scaled by
    ``sigma_i * sqrt(area)``; visibility flips at probability
    :data:`KP_VISIBILITY_FLIP_PROB`. The bbox is the GT bbox (no bbox
    jitter — the OKS area normaliser comes from ``ann["area"]``, not
    the bbox).
    """
    with gt_path.open("rb") as f:
        gt = json.load(f)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=_KP_RNG_SPAWN_KEY))

    n_default_sigmas = len(COCO_PERSON_SIGMAS)
    detections: list[dict[str, object]] = []
    for ann in gt.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        kp = ann.get("keypoints")
        if not kp:
            continue
        n_triplets = len(kp) // 3
        if n_triplets == 0:
            continue
        # Sigmas length must match n_triplets; fall back to COCO-person
        # only when the count lines up. A non-17 keypoint vector with no
        # category override is a fixture bug — keypoint jitter is not
        # the place to silently invent a sigma table.
        if n_triplets != n_default_sigmas:
            raise ValueError(
                f"keypoint jitter expects {n_default_sigmas} keypoints "
                f"(COCO-person sigmas); GT ann {ann.get('id')} has {n_triplets}. "
                f"Per-category sigma override is an ADR-0012 follow-up."
            )

        area = float(ann.get("area", 0.0))
        scale = float(np.sqrt(max(area, 1.0)))

        kp_arr = np.asarray(kp, dtype=np.float64).reshape(n_triplets, 3)
        # One Gaussian draw per (x, y) per keypoint, scaled by
        # sigma_i * sqrt(area). Drawing the whole matrix at once keeps
        # the rng-stream order stable across numpy versions.
        sigmas = np.asarray(COCO_PERSON_SIGMAS, dtype=np.float64)
        noise = rng.normal(0.0, 1.0, size=(n_triplets, 2))
        per_kp_std = (sigmas * scale * KP_JITTER_SCALE)[:, None]
        kp_arr[:, :2] += noise * per_kp_std

        # Visibility flips: rare bit-flip on the v channel
        # (0 ↔ 2; v=1 is unused at the prediction boundary). Drawn
        # after the position noise so the position stream is stable
        # if KP_VISIBILITY_FLIP_PROB ever changes.
        flip_mask = rng.random(n_triplets) < KP_VISIBILITY_FLIP_PROB
        flipped = np.where(kp_arr[:, 2] > 0, 0.0, 2.0)
        kp_arr[:, 2] = np.where(flip_mask, flipped, kp_arr[:, 2])

        score = float(rng.beta(SCORE_BETA_ALPHA, SCORE_BETA_BETA))
        det = {
            "image_id": int(ann["image_id"]),
            "category_id": int(ann["category_id"]),
            "bbox": [float(v) for v in ann["bbox"]],
            "score": score,
            "keypoints": kp_arr.flatten().tolist(),
        }
        detections.append(det)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(detections, f, separators=(",", ":"))


def keypoints_dt_path(*, gt_path: Path, seed: int) -> Path:
    """Return the cached keypoints-jittered DT, generating it if missing."""
    out = _keypoints_jittered_dt_path(seed)
    if not out.exists():
        _generate_keypoints(gt_path=gt_path, seed=seed, out=out)
    return out
