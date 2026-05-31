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
full-SHA cache key, thread pin so summation order is host-independent,
and loud-fail on unmapped class names so a subset / aliased /
corrupted GT can't silently populate a partial cache. The cross-cell
scaffolding (thread pin, model load, atomic write, name-based class
join) lives in :mod:`._harness_common`.

Inference is the cost driver — Mask2Former Swin-T is ~14-18s per
640x480 image on an 8-core AMD EPYC-Milan (vs DETR-R50's ~9s — the
masked-attention decoder runs 100 queries through 9 transformer
layers per image regardless of detection density). End-to-end on
COCO val2017 (5000 images): ~20-25h.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from real_predictions_cache import (
    MASK2FORMER_PANOPTIC_MODEL_ID,
    MASK2FORMER_PANOPTIC_REVISION,
)

from ._harness_common import (
    atomic_write_bytes,
    iter_image_records,
    load_processor_and_model,
    name_based_class_mapping,
)

_DATASET_LABEL = "coco-panoptic-val2017"


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
    """
    import numpy as np
    import torch
    from PIL import Image as PILImage

    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)

    # ``target_sizes`` is a Python list-of-tuples (not an int64 tensor):
    # the panoptic post-process passes ``target_sizes[idx]`` straight to
    # ``torch.nn.functional.interpolate(size=...)``, which rejects a
    # tensor argument in current PyTorch. The int64-tensor discipline
    # exists for the detection path's box-scale multiplication, not here.
    result = processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=[image_size_hw],
    )[0]
    # ``segmentation`` is an HxW int64 tensor of segment ids;
    # ``segments_info`` is a list of ``{id, label_id, was_fused, score}``.
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
    # Atomic ``.part`` → rename so a SIGINT mid-write doesn't leave a
    # half-encoded PNG that the per-image skip below (or a downstream
    # panopticapi consumer) would silently treat as complete.
    png_part = png_out_path.with_suffix(png_out_path.suffix + ".part")
    PILImage.fromarray(rgb, mode="RGB").save(png_part)
    png_part.replace(png_out_path)

    # Build the per-image segments_info in COCO panoptic results shape:
    # ``id`` is the segment id (matches the PNG); ``category_id`` is
    # the GT-space sparse COCO id; ``area`` is the pixel count.
    seg_infos: list[dict[str, Any]] = []
    for seg in raw_segments:
        seg_id = int(seg["id"])
        if seg_id == 0:
            # ``0`` is the panoptic "void" sentinel; not a real segment.
            continue
        label_id = int(seg["label_id"])
        cat_id = class_mapping.get(label_id)
        if cat_id is None:
            # name_based_class_mapping would already have raised if any
            # model label is missing from the GT; reaching this branch
            # means the model emitted a label_id outside its own
            # id2label space — a load-time guard would have caught it.
            raise RuntimeError(
                f"image {image_id}: segment id {seg_id} has label_id "
                f"{label_id} which is outside the model's id2label space. "
                f"Indicates a transformers version drift; refusing to "
                f"emit a bogus cat_id under the pinned revision SHA."
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

    Writes per-image PNGs and per-image JSON sidecars into ``cache_dir``
    and a single aggregated ``panoptic_dt.json`` at ``dt_json_path`` in
    the COCO panoptic results shape. Returns the aggregated JSON bytes.

    Resume contract: a hit on ``dt_json_path`` short-circuits without
    instantiating the model. On a miss, every per-image ``{id}.png`` +
    ``{id}.json`` pair already on disk is loaded verbatim and skipped
    by the inference loop — so a SIGINT mid-run resumes cheaply on
    next invocation. Both per-image writes are atomic
    (``.part`` → rename), so a SIGINT in the middle of either write
    cannot leave a half-encoded artefact the skip-on-hit branch would
    silently consume. If every expected pair is already present, the
    model load itself is skipped and the aggregated JSON is built
    directly from the sidecars.
    """
    if dt_json_path.is_file():
        return dt_json_path.read_bytes()

    cache_dir.mkdir(parents=True, exist_ok=True)
    sorted_images = sorted(gt["images"], key=lambda i: int(i["id"]))
    sidecar_paths = {int(img["id"]): cache_dir / f"{int(img['id'])}.json" for img in sorted_images}
    png_paths = {int(img["id"]): cache_dir / f"{int(img['id'])}.png" for img in sorted_images}

    # Full-coverage short-circuit: every (png, sidecar) already on disk
    # means a prior run completed inference but died before writing the
    # aggregate JSON. Avoid the multi-hundred-MB model load and assemble
    # the aggregate from the sidecars directly.
    if all(png_paths[iid].is_file() and sidecar_paths[iid].is_file() for iid in sidecar_paths):
        annotations = [json.loads(sidecar_paths[iid].read_bytes()) for iid in sidecar_paths]
        payload = json.dumps({"annotations": annotations}).encode("utf-8")
        atomic_write_bytes(dt_json_path, payload)
        return payload

    from PIL import Image

    processor, model = load_processor_and_model(
        MASK2FORMER_PANOPTIC_MODEL_ID,
        revision,
        model_cls_name="AutoModelForUniversalSegmentation",
    )
    id2label: dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}
    # Loud-fail with no documented drop set: the COCO panoptic 133-class
    # space is a strict subset of canonical COCO panoptic val2017
    # categories. A missing name means the GT is something other than
    # canonical panoptic_val2017.json.
    class_mapping = name_based_class_mapping(
        id2label,
        gt["categories"],
        context=f"mask2former-pan ({_DATASET_LABEL})",
    )

    annotations = []
    for img, image_path in iter_image_records(
        sorted_images,
        image_dir,
        desc=f"mask2former-pan {_DATASET_LABEL}",
        progress=progress,
    ):
        image_id = int(img["id"])
        png_out = png_paths[image_id]
        sidecar_out = sidecar_paths[image_id]
        if png_out.is_file() and sidecar_out.is_file():
            annotations.append(json.loads(sidecar_out.read_bytes()))
            continue
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
        atomic_write_bytes(sidecar_out, json.dumps(ann).encode("utf-8"))
        annotations.append(ann)

    payload = json.dumps({"annotations": annotations}).encode("utf-8")
    atomic_write_bytes(dt_json_path, payload)
    return payload
