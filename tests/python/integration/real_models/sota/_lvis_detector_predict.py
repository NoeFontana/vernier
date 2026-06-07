"""Deformable DETR (LVIS box-supervised) → LVIS results JSON adapter.

``facebook/deformable-detr-box-supervised`` is the Apache-2.0,
CPU-runnable, LVIS v1-trained baseline anchoring the federated-
evaluation real-prediction parity smoke (ADR-0026, PR-3 of the real-
prediction roadmap). The checkpoint is "Box-Supervised_DeformDETR_R50_4x"
from the original Detic release, re-hosted on the Hugging Face hub
with a transformers-compatible ``DeformableDetrForObjectDetection``
config (300-query head, 1203 id2label entries covering the full LVIS
v1 category set).

The model is loaded via ``transformers.AutoImageProcessor`` +
``AutoModelForObjectDetection``; predictions are written in the LVIS
results JSON shape — same flat ``list[dict]`` layout that
``lvis-api``'s ``LVISResults`` constructor and vernier's federated
bbox grid both ingest. The cache discipline mirrors the DETR-R50 cell
(:mod:`_detr_predict`) with one tightening: LVIS GT categories cover
the model's full output space, so the loud-fail on unmapped names has
NO documented drop set — every model label must resolve.

Inference is the cost driver — LVIS v1 val is 19,809 images at
~640x480 vs COCO val2017's 5,000, and the Deformable-DETR forward is
heavier per image than DETR-R50 (300-query head + 6 deformable-
attention decoder layers vs DETR-R50's 100 queries / 6 standard-
attention layers). Budget ~48-72h end-to-end on an 8-core AMD
EPYC-Milan; linear scaling with image count, mostly independent of
detection density (Deformable-DETR's 300-query head fires the same
forward graph regardless).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from real_predictions_cache import LVIS_DETECTOR_REVISION

from ._harness_common import (
    atomic_write_bytes,
    iter_image_records,
    load_processor_and_model,
    make_detection_target_sizes,
    name_based_class_mapping,
)

_MODEL_ID = "facebook/deformable-detr-box-supervised"
_DATASET_LABEL = "lvis-v1-val"

#: Permissive score floor. LVIS evaluation uses ``max_dets=300`` per
#: image (vs COCO's 100), so the long tail of low-confidence detections
#: carries the same per-detection cost it does on the COCO cell. The
#: 0.05 floor mirrors the DETR-R50 / rf-detr cells, keeps the
#: ``max_dets`` trim semantically meaningful (lvis-api's
#: :class:`LVISResults` reorders by score descending and clamps to
#: 300), and never drops a detection the AP / per-category federated
#: machinery would otherwise read.
_SCORE_THRESHOLD = 0.05


def _xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def _file_name_from_image_record(img: dict[str, Any]) -> str:
    """Resolve the val2017-relative filename for an LVIS image record.

    LVIS v1 image records ship with both ``file_name`` (the COCO-style
    zero-padded ``000000000139.jpg``) and ``coco_url`` (the public
    image URL). Prefer the explicit ``file_name`` field; fall back to
    the last path segment of ``coco_url`` so a future LVIS GT release
    that drops the redundant ``file_name`` field still resolves.
    The Detectron2 LVIS loader documents this same fallback policy.
    """
    fn = img.get("file_name")
    if isinstance(fn, str) and fn:
        return fn
    url = img.get("coco_url")
    if isinstance(url, str) and url:
        return url.rsplit("/", 1)[-1]
    raise KeyError(
        f"LVIS image record {img.get('id')!r} has neither 'file_name' "
        f"nor 'coco_url'; cannot resolve val2017 image path."
    )


def _records_for_image(
    *,
    image_id: int,
    image_size_hw: tuple[int, int],
    processor: Any,
    model: Any,
    image: Any,
    class_mapping: dict[int, int],
) -> list[dict[str, Any]]:
    """Run one forward pass + post-process; emit LVIS results records.

    ``post_process_object_detection`` returns ``boxes`` in xyxy form
    in the original image's pixel coordinates (the ``target_sizes``
    arg is the (H, W) it scales back into); LVIS results — like COCO
    — want xywh.
    """
    import torch

    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=make_detection_target_sizes(image_size_hw),
        threshold=_SCORE_THRESHOLD,
    )[0]

    scores = results["scores"].tolist()
    labels = results["labels"].tolist()
    boxes = results["boxes"].tolist()
    records: list[dict[str, Any]] = []
    for i, label_id in enumerate(labels):
        # ``class_mapping`` was built from the GT's full 1203-category
        # name set; the loud-fail at build time guarantees every model
        # label resolved. A miss here would mean the model's id2label
        # space drifted between load and inference — which it cannot,
        # because we pin the revision. Keep the guard as a sanity
        # check; raising is correct (vs silently dropping the box).
        cat_id = class_mapping[int(label_id)]
        records.append(
            {
                "image_id": int(image_id),
                "category_id": int(cat_id),
                "bbox": _xyxy_to_xywh(boxes[i]),
                "score": float(scores[i]),
            }
        )
    return records


def predict_lvis_val(
    *,
    gt: dict[str, Any],
    image_dir: Path,
    cache_path: Path,
    revision: str = LVIS_DETECTOR_REVISION,
    progress: bool = True,
) -> bytes:
    """Run Deformable-DETR inference on every image in ``gt['images']``.

    Emits an LVIS results JSON (the flat list shape ``lvis-api``'s
    :class:`LVISResults` constructor consumes). Owns the cache
    contract end-to-end: a hit on ``cache_path`` returns the bytes
    without instantiating the model (which would download weights).
    On a miss, instantiates the model lazily, runs inference, atomic-
    writes the cache, returns the bytes.

    The name-based class mapping passes NO documented drop set: LVIS
    GT covers the model's full 1203-category id2label space, so any
    unmapped name is a load-time error (a subset / aliased /
    corrupted GT). Same loud-fail invariant the DETR-R50 cell uses
    with the COCO-91 → COCO-80 drop set.
    """
    if cache_path.is_file():
        return cache_path.read_bytes()

    from PIL import Image

    processor, model = load_processor_and_model(
        _MODEL_ID, revision, model_cls_name="AutoModelForObjectDetection"
    )
    id2label: dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}
    class_mapping = name_based_class_mapping(
        id2label,
        gt["categories"],
        # Empty drop set — LVIS GT is a strict superset of the model's
        # label space. The loud-fail message will list the GT
        # categories so a partial-GT-cache mismatch surfaces directly.
        dropped_names=frozenset(),
        na_marker=None,
        context=f"deformable-detr-lvis ({_DATASET_LABEL})",
    )

    records: list[dict[str, Any]] = []
    for img, image_path in _iter_lvis_images(
        gt["images"],
        image_dir,
        progress=progress,
    ):
        with Image.open(image_path) as pil:
            pil_rgb = pil.convert("RGB")
            records.extend(
                _records_for_image(
                    image_id=int(img["id"]),
                    image_size_hw=(int(img["height"]), int(img["width"])),
                    processor=processor,
                    model=model,
                    image=pil_rgb,
                    class_mapping=class_mapping,
                )
            )

    payload = json.dumps(records).encode("utf-8")
    atomic_write_bytes(cache_path, payload)
    return payload


def _iter_lvis_images(
    images: list[dict[str, Any]],
    image_dir: Path,
    *,
    progress: bool,
) -> Any:
    """LVIS-flavored wrapper over :func:`iter_image_records`.

    The DETR-R50 cell uses COCO val2017 GT, whose ``file_name`` field
    is the canonical bare-filename. LVIS GT ships both ``file_name``
    and ``coco_url`` but the on-disk layout is the same
    ``val2017/<file>.jpg`` shape because LVIS reuses COCO 2017
    images.  We pre-normalise so every record has a usable
    ``file_name`` before handing to the shared iterator — this lets us
    keep the loud-fail "image missing on disk" error in one place
    (the shared helper) without copy-pasting.
    """
    normalised: list[dict[str, Any]] = []
    for img in images:
        fn = _file_name_from_image_record(img)
        # Shallow-copy + overwrite so the upstream GT dict isn't
        # mutated (the caller may parse it once and keep a reference).
        normalised.append({**img, "file_name": fn})
    yield from iter_image_records(
        normalised,
        image_dir,
        desc=f"deformable-detr-lvis {_DATASET_LABEL}",
        progress=progress,
    )
