"""DETR-R50 → COCO results JSON adapter for the SOTA validation harness.

``facebook/detr-resnet-50`` is the Apache-2.0, CPU-runnable, COCO-80
bbox baseline anchoring the first wave of the Hugging Face SOTA harness
(panoptic / semantic / keypoints follow). The model is loaded via
``transformers.AutoImageProcessor`` + ``AutoModelForObjectDetection``;
predictions are written in the same COCO ``loadRes`` shape that the
rf-detr / Mask R-CNN cached blobs already use, so the bench harness's
``InstanceWorkload`` runners pick them up unchanged.

Cache discipline mirrors the rf-detr adapter (``tide/_rfdetr_predict.py``):
keying on ``(model_name, revision_sha, dataset_id)`` so a weights bump
on the hub invalidates the cache by construction rather than silently
producing stale numbers. The cross-cell scaffolding (thread pin, model
load, atomic write, name-based class join) lives in
:mod:`._harness_common`.

Inference is the cost driver — ~9s per 640x480 image with PyTorch's
intra-op threading on an 8-core AMD EPYC-Milan; ~12-15h end-to-end on
COCO val2017 (5000 images). Linear scaling with image count and
mostly independent of detection density (DETR's 100-query head fires
the same forward graph regardless).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from real_predictions_cache import DETR_RESNET50_REVISION

from ._harness_common import (
    atomic_write_bytes,
    iter_image_records,
    load_processor_and_model,
    make_detection_target_sizes,
    name_based_class_mapping,
)

_MODEL_ID = "facebook/detr-resnet-50"
_DATASET_LABEL = "coco-val2017"

#: Permissive score floor — same value rf-detr's TIDE adapter uses.
#: vernier sees the full PR curve regardless, so cutting at 0.5 would
#: just throw away signal the AP / TIDE machinery is designed to absorb.
_SCORE_THRESHOLD = 0.05

#: The 11 COCO-2014 categories DETR's id2label publishes that COCO
#: val2017 GT (80-category subset) drops. Skipping these silently is
#: correct; any OTHER unmapped name must fail loud (subset / aliased /
#: corrupted GT) — same loud-fail invariant the rfdetr adapter enforces.
_COCO91_DROPPED_NAMES = frozenset(
    {
        "street sign",
        "hat",
        "shoe",
        "eye glasses",
        "plate",
        "mirror",
        "window",
        "desk",
        "door",
        "blender",
        "hair brush",
    }
)


def _xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def _records_for_image(
    *,
    image_id: int,
    image_size_hw: tuple[int, int],
    processor: Any,
    model: Any,
    image: Any,
    class_mapping: dict[int, int],
) -> list[dict[str, Any]]:
    """Run one forward pass + post-process; emit COCO records.

    ``post_process_object_detection`` returns ``boxes`` in xyxy form in
    the original image's pixel coordinates (the ``target_sizes`` arg is
    the (H, W) it scales back into); COCO results want xywh.
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
        cat_id = class_mapping.get(int(label_id))
        if cat_id is None:
            # DETR can emit a "N/A" slot above threshold on a noisy
            # image; the COCO oracle would treat it as an unknown
            # category, so we drop it rather than feed bogus IDs.
            continue
        records.append(
            {
                "image_id": int(image_id),
                "category_id": int(cat_id),
                "bbox": _xyxy_to_xywh(boxes[i]),
                "score": float(scores[i]),
            }
        )
    return records


def predict_coco_val(
    *,
    gt: dict[str, Any],
    image_dir: Path,
    cache_path: Path,
    revision: str = DETR_RESNET50_REVISION,
    progress: bool = True,
) -> bytes:
    """Run DETR-R50 inference on every image in ``gt['images']``, emit COCO JSON.

    Owns the cache contract end-to-end: a hit on ``cache_path`` returns
    the bytes without instantiating the model (which would download
    weights). On a miss, instantiates the model lazily, runs inference,
    atomic-writes the cache, and returns the bytes.
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
        dropped_names=_COCO91_DROPPED_NAMES,
        na_marker="N/A",
        context=f"detr-r50 ({_DATASET_LABEL})",
    )

    records: list[dict[str, Any]] = []
    for img, image_path in iter_image_records(
        gt["images"],
        image_dir,
        desc=f"detr-r50 {_DATASET_LABEL}",
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
