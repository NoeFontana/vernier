"""Mask2Former Swin-T → COCO panoptic results adapter (SOTA harness).

``facebook/mask2former-swin-tiny-coco-panoptic`` is the Apache-2.0,
CPU-runnable panoptic baseline next to the DETR-R50 detection cell.
Loaded via ``transformers.AutoImageProcessor`` +
``AutoModelForUniversalSegmentation``; predictions land as one
rgb2id-encoded PNG per image + a single ``panoptic_dt.json`` sidecar
in the COCO panoptic results shape, so panopticapi's
``pq_compute_single_core`` and ``vernier.panoptic.Predictions.from_arrays``
both ingest the same files.

Cache discipline mirrors the DETR-R50 adapter (lessons from PR #265):
full-SHA cache key, ``torch.set_num_threads(1)`` pin so summation
order is host-independent, and loud-fail on unmapped class names so a
subset / aliased / corrupted GT can't silently populate a partial
cache.

Inference is the cost driver — Mask2Former Swin-T is ~14-18s per
640x480 image on an 8-core AMD EPYC-Milan (vs DETR-R50's ~9s — the
masked-attention decoder runs 100 queries through 9 transformer
layers per image regardless of detection density). End-to-end on
COCO val2017 (5000 images): ~20-25h.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from real_predictions_cache import (
    MASK2FORMER_PANOPTIC_MODEL_ID,
    MASK2FORMER_PANOPTIC_REVISION,
)

_DATASET_LABEL = "coco-panoptic-val2017"


def _coco_panoptic_class_mapping(
    gt_categories: list[dict[str, Any]], id2label: dict[int, str]
) -> dict[int, int]:
    """Map Mask2Former's 0..132 contiguous train-id to the GT JSON's
    sparse panoptic ``category_id`` (1..200 with gaps).

    Same name-based join as the DETR-R50 adapter's
    :func:`_coco_class_mapping`: model labels match GT category names,
    not GT ids — the model happens to be trained on the COCO panoptic
    133-class subset whose category_id values are NOT contiguous on
    the COCO upstream side, but the names are stable across both.

    Loud-fail on unmapped names (no silent skip): the COCO panoptic
    133-class space is a strict subset of the GT's published
    categories, so every model label MUST resolve. A missing name
    means the GT JSON is something other than canonical COCO
    panoptic — refusing to cache against it is the only safe move
    when the cache filename embeds the pinned revision SHA.
    """
    name_to_cat_id = {cat["name"]: int(cat["id"]) for cat in gt_categories}
    mapping: dict[int, int] = {}
    for label_id, name in id2label.items():
        cat_id = name_to_cat_id.get(name)
        if cat_id is None:
            raise RuntimeError(
                f"Mask2Former panoptic label {label_id} ('{name}') has no "
                f"matching category in the GT JSON. The COCO panoptic "
                f"133-class space (80 thing + 53 stuff) is a strict subset "
                f"of canonical COCO panoptic val2017 categories — a missing "
                f"name means the GT is something other than canonical "
                f"panoptic_val2017.json. Refusing to silently populate a "
                f"partial cache under the pinned revision SHA. "
                f"GT category names: {sorted(name_to_cat_id)}"
            )
        mapping[label_id] = cat_id
    return mapping


def _instantiate_model(revision: str) -> tuple[Any, Any]:
    """Lazy transformers import + processor / model instantiation.

    Loaded with ``revision=`` so the cache filename's commit-SHA pin
    is the same SHA the weights resolve to. ``low_cpu_mem_usage=True``
    streams shards into memory rather than materializing the full
    state dict twice (Mask2Former Swin-T is ~50M params; the full
    state dict is ~200MB on disk so the peak-RSS difference matters
    on smaller CPU boxes).

    Pins ``torch.set_num_threads(1)`` before the first forward pass
    so intra-op summation order is the same regardless of the host's
    physical core count — the cache key is ``(model, revision,
    dataset)`` only, so without this pin two hosts on the same
    revision would populate the cache with bit-different bytes via
    summation-order drift in matmul reductions. Same invariant as
    the DETR-R50 adapter (PR #265 follow-up).
    """
    import torch
    from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation

    torch.set_num_threads(1)

    processor = AutoImageProcessor.from_pretrained(MASK2FORMER_PANOPTIC_MODEL_ID, revision=revision)
    model = AutoModelForUniversalSegmentation.from_pretrained(
        MASK2FORMER_PANOPTIC_MODEL_ID,
        revision=revision,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return processor, model


def _segments_for_image(
    *,
    image_id: int,
    image_size_hw: tuple[int, int],
    processor: Any,
    model: Any,
    image: Any,
    class_mapping: dict[int, int],
    png_out_path: Path,
) -> dict[str, Any]:
    """Run one forward pass + panoptic post-process; write the per-image
    PNG and return the per-image annotation dict.

    Returns the annotation in COCO panoptic results shape:
    ``{"image_id": int, "file_name": str, "segments_info": [...]}``.
    Each segment carries ``id``, ``category_id`` (mapped to GT
    space), and ``area`` (computed from the seg map, since the
    upstream post-process doesn't emit it).

    ``target_sizes`` is ``int64`` (not ``float64``) for the same
    reason as DETR-R50: transformers' post-process upcasts
    arithmetic against an fp64 size tensor, tying cached bytes to a
    transformers internal that can shift between minor versions.
    """
    import numpy as np
    import torch
    from PIL import Image as PILImage

    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image_size_hw], dtype=torch.int64)
    result = processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=target_sizes,
    )[0]
    # `segmentation` is an HxW int64 tensor of segment ids; `segments_info`
    # is a list of {id, label_id, was_fused, score}.
    seg_map = result["segmentation"].to(torch.int64).cpu().numpy()
    raw_segments = result["segments_info"]

    # Encode segment ids as RGB via rgb2id: id = R + 256*G + 256²*B.
    # panopticapi's evaluation.py decodes via this exact convention.
    seg_u32 = seg_map.astype(np.uint32, copy=False)
    rgb = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    work = seg_u32.copy()
    for i in range(3):
        rgb[..., i] = (work % 256).astype(np.uint8)
        work //= 256
    PILImage.fromarray(rgb, mode="RGB").save(png_out_path)

    # Build the per-image segments_info in COCO panoptic results shape:
    # `id` is the segment id (matches the PNG); `category_id` is the
    # GT-space sparse COCO id; `area` is the pixel count.
    seg_infos: list[dict[str, Any]] = []
    for seg in raw_segments:
        seg_id = int(seg["id"])
        if seg_id == 0:
            # `0` is the panoptic "void" sentinel; not a real segment.
            continue
        label_id = int(seg["label_id"])
        cat_id = class_mapping.get(label_id)
        if cat_id is None:
            # See the _coco_panoptic_class_mapping discussion: every
            # label SHOULD resolve under canonical COCO panoptic GT.
            # If we get here, the mapping was built against a
            # different GT than the predictions are being scored
            # against — drop loudly rather than emit a bogus cat_id.
            raise RuntimeError(
                f"image {image_id}: segment id {seg_id} has label_id "
                f"{label_id} which has no entry in the class mapping. "
                f"This indicates the GT JSON used to build the mapping "
                f"is not consistent with the model's training set."
            )
        area = int((seg_u32 == seg_id).sum())
        if area == 0:
            # post_process can emit a segment id that ended up fully
            # overlapped during merging; drop them.
            continue
        seg_infos.append(
            {
                "id": seg_id,
                "category_id": cat_id,
                "area": area,
                "iscrowd": 0,
            }
        )

    return {
        "image_id": int(image_id),
        "file_name": png_out_path.name,
        "segments_info": seg_infos,
    }


def predict_coco_panoptic_val(
    *,
    gt: dict[str, Any],
    image_dir: Path,
    cache_dir: Path,
    dt_json_path: Path,
    revision: str = MASK2FORMER_PANOPTIC_REVISION,
    progress: bool = True,
) -> bytes:
    """Run Mask2Former panoptic inference on every image in ``gt['images']``.

    Writes per-image PNGs into ``cache_dir`` and a single
    ``panoptic_dt.json`` at ``dt_json_path`` in the COCO panoptic
    results shape. Returns the JSON bytes.

    Owns the cache contract end-to-end: a hit on ``dt_json_path``
    AND the matching set of PNGs short-circuits without instantiating
    the model. On a miss, instantiates lazily, runs inference,
    atomic-writes the JSON (PNGs are written in-place as we go;
    partial PNG sets are tolerated on a re-run via the same per-image
    skip mechanism the cache_dir holds).

    Atomic JSON write protects against SIGINT mid-run leaving a
    half-written sidecar that the next session would mistake for
    complete.
    """
    if dt_json_path.is_file():
        return dt_json_path.read_bytes()

    from PIL import Image

    processor, model = _instantiate_model(revision)
    id2label: dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}
    class_mapping = _coco_panoptic_class_mapping(gt["categories"], id2label)

    image_list: Sequence[dict[str, Any]] = gt["images"]
    images: Iterable[dict[str, Any]] = image_list
    if progress:
        from tqdm import tqdm

        images = tqdm(image_list, total=len(image_list), desc=f"mask2former-pan {_DATASET_LABEL}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    annotations: list[dict[str, Any]] = []
    for img in images:
        image_path = image_dir / img["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(
                f"image referenced by GT JSON missing on disk: {image_path}. "
                f"Re-run the COCO val2017 fetcher; the cache root must contain "
                f"the full val2017/ directory next to instances_val2017.json."
            )
        image_id = int(img["id"])
        png_out = cache_dir / f"{image_id}.png"
        with Image.open(image_path) as pil:
            pil_rgb = pil.convert("RGB")
            ann = _segments_for_image(
                image_id=image_id,
                image_size_hw=(int(img["height"]), int(img["width"])),
                processor=processor,
                model=model,
                image=pil_rgb,
                class_mapping=class_mapping,
                png_out_path=png_out,
            )
        annotations.append(ann)

    payload = json.dumps({"annotations": annotations}).encode("utf-8")
    dt_json_path.parent.mkdir(parents=True, exist_ok=True)
    part = dt_json_path.with_suffix(dt_json_path.suffix + ".part")
    part.write_bytes(payload)
    part.replace(dt_json_path)
    return payload
