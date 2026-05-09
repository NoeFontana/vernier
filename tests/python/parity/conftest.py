"""Pytest configuration for parity tests.

The parity suite double-runs the reference (pycocotools 2.0.8) and the
candidate (vernier) on the same fixtures and asserts every intermediate
matches. Today the candidate is a shim that delegates to pycocotools, so the
suite is a tautology — but the harness, fixture corpus, and CI plumbing are
real. As Rust evaluator pieces ship, the shim is replaced and the suite
becomes a load-bearing parity gate.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
from pycocotools import mask as pmask

from vernier._array_types import CompressedRLE, UncompressedRLE
from vernier.instance import Detections

FIXTURES_DIR = Path(__file__).parent / "fixtures"

IouType = Literal["bbox", "segm", "boundary", "keypoints"]


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


def _group_by_image(
    records: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """Bucket loadRes-shaped DT records by ``image_id``."""
    by_image: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        by_image.setdefault(int(r["image_id"]), []).append(r)
    return by_image


def loadres_to_detections(
    gt_records: dict[str, Any],
    dt_records: list[dict[str, Any]],
    iou_type: IouType,
) -> list[Detections]:
    """Convert a ``loadRes``-shaped DT JSON into per-image array
    ``Detections`` dicts that ADR-0030's ingest path consumes.

    Polygon DT segmentations are rasterized via pycocotools
    (``frPyObjects`` + union, matching quirk **K2**) and re-encoded as
    column-major uncompressed counts. RLE strings are decoded to a
    binary mask and re-encoded the same way.
    """
    image_dims = {
        int(im["id"]): (int(im["height"]), int(im["width"])) for im in gt_records["images"]
    }
    by_image = _group_by_image(dt_records)

    out: list[Detections] = []
    for image_id in sorted(by_image.keys()):
        dets = by_image[image_id]
        boxes = np.asarray([[float(x) for x in d["bbox"]] for d in dets], dtype=np.float64)
        scores = np.asarray([float(d["score"]) for d in dets], dtype=np.float64)
        labels = np.asarray([int(d["category_id"]) for d in dets], dtype=np.int64)
        payload: Detections = {
            "image_id": image_id,
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }
        if iou_type in ("segm", "boundary"):
            h, w = image_dims[image_id]
            payload["rles"] = [_segmentation_to_rle(d["segmentation"], h, w) for d in dets]
        if iou_type == "keypoints":
            kps_per_det = [d["keypoints"] for d in dets]
            n = len(dets)
            if not kps_per_det:
                payload["keypoints"] = np.zeros((0, 0, 3), dtype=np.float64)
            else:
                k = len(kps_per_det[0]) // 3
                payload["keypoints"] = np.asarray(kps_per_det, dtype=np.float64).reshape(n, k, 3)
        out.append(payload)
    return out


def _decode_seg_to_binary(seg: object, h: int, w: int) -> np.ndarray:
    """Render any COCO segmentation (polygon list, uncompressed-list dict,
    or compressed-string dict) into a binary 2-D mask via pycocotools."""
    if isinstance(seg, dict):
        counts = seg["counts"]
        seg_h, seg_w = int(seg["size"][0]), int(seg["size"][1])
        if isinstance(counts, list):
            rle_list = pmask.frPyObjects(
                {"counts": counts, "size": [seg_h, seg_w]},  # type: ignore[arg-type]
                seg_h,
                seg_w,
            )
            return np.asarray(pmask.decode(rle_list))
        encoded = {"counts": counts.encode("ascii"), "size": list(seg["size"])}
        return np.asarray(pmask.decode(encoded))  # type: ignore[arg-type]
    encoded_list = pmask.frPyObjects([list(p) for p in seg], h, w)  # type: ignore[arg-type]
    return np.asarray(pmask.decode(pmask.merge(encoded_list)))


def _segmentation_to_rle(seg: object, h: int, w: int) -> UncompressedRLE:
    if isinstance(seg, dict) and isinstance(seg["counts"], list):
        return {
            "counts": np.asarray(seg["counts"], dtype=np.uint32),
            "size": (int(seg["size"][0]), int(seg["size"][1])),
        }
    binary = _decode_seg_to_binary(seg, h, w)
    return {
        "counts": np.asarray(_binary_mask_to_runs(binary), dtype=np.uint32),
        "size": (binary.shape[0], binary.shape[1]),
    }


def _segmentation_to_compressed_rle(seg: object, h: int, w: int) -> CompressedRLE:
    """Form-2 ingest shape: ``{counts: bytes, size: (h, w)}`` where
    ``counts`` is the COCO 6-bit ASCII compressed payload emitted by
    :func:`pycocotools.mask.encode`.
    """
    binary = _decode_seg_to_binary(seg, h, w)
    compressed = pmask.encode(np.asfortranarray(binary.astype(np.uint8)))
    raw_counts = compressed["counts"]
    if isinstance(raw_counts, bytes):
        counts_bytes = raw_counts
    elif isinstance(raw_counts, str):
        counts_bytes = raw_counts.encode("ascii")
    else:
        counts_bytes = bytes(raw_counts)
    return {
        "counts": counts_bytes,
        "size": (binary.shape[0], binary.shape[1]),
    }


def _segmentation_to_bitmask(
    seg: object, h: int, w: int, order: Literal["C", "F"] = "C"
) -> np.ndarray:
    """Form-3 ingest shape: 2-D ``uint8`` array of shape ``(h, w)``."""
    binary_u8 = _decode_seg_to_binary(seg, h, w).astype(np.uint8, copy=False)
    if order == "F":
        return np.asfortranarray(binary_u8)
    return np.ascontiguousarray(binary_u8)


def _binary_mask_to_runs(binary: np.ndarray) -> list[int]:
    flat = binary.flatten("F")
    runs: list[int] = []
    cur = 0
    cur_val: np.uint8 = np.uint8(0)
    for v in flat:
        if v == cur_val:
            cur += 1
        else:
            runs.append(int(cur))
            cur = 1
            cur_val = v
    runs.append(int(cur))
    return runs


def shard_dt_bytes(dt_path: Path, n_shards: int, seed: int) -> list[bytes]:
    """Split DT records by image_id into ``n_shards`` disjoint payloads.

    Splitting by image_id (not by record) avoids the streaming /
    background evaluator's duplicate-image-id rejection: the same image
    can never appear in two batches. Empty shards (when ``n_shards`` >
    number of images) are returned as ``b"[]"`` so callers can still
    issue an ``update()`` for them without special-casing.
    """
    records = json.loads(dt_path.read_text())
    by_image = _group_by_image(records)
    image_ids = sorted(by_image.keys())
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    shards: list[list[dict]] = [[] for _ in range(n_shards)]
    for i, img_id in enumerate(image_ids):
        shards[i % n_shards].extend(by_image[img_id])
    return [json.dumps(s).encode("utf-8") for s in shards]
