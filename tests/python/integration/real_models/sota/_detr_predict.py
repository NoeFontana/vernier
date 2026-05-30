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
producing stale numbers.

Inference is the cost driver — ~9s per 640x480 image with PyTorch's
intra-op threading on an 8-core AMD EPYC-Milan; ~12-15h end-to-end on
COCO val2017 (5000 images). Linear scaling with image count and
mostly independent of detection density (DETR's 100-query head fires
the same forward graph regardless).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from real_predictions_cache import DETR_RESNET50_REVISION

_MODEL_ID = "facebook/detr-resnet-50"
_DATASET_LABEL = "coco-val2017"

#: Permissive score floor — same value rf-detr's TIDE adapter uses.
#: vernier sees the full PR curve regardless, so cutting at 0.5 would
#: just throw away signal the AP / TIDE machinery is designed to absorb.
_SCORE_THRESHOLD = 0.05

#: The 11 COCO-2014 categories DETR's id2label publishes that COCO
#: val2017 GT (80-category subset) drops. Skipping these silently is
#: correct; any OTHER unmapped name should fail loud (subset / aliased /
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


def _coco_class_mapping(gt: dict[str, Any], id2label: dict[int, str]) -> dict[int, int]:
    """Map DETR's hub-config label id (0..91, with COCO sparse-id gaps
    marked ``"N/A"``) to the GT JSON's ``category_id`` (1..90 with the
    same gaps on canonical COCO).

    Resolved by name match against the GT JSON's ``categories`` list,
    not by id-equals-id fallback: DETR happens to publish labels in the
    COCO sparse-id space, but matching by name is the convention rf-detr
    established (``tide/_rfdetr_predict.py::_coco_class_mapping``).

    DETR is COCO-2014-classes-aware (91 categories) while COCO val2017
    GT uses the 80-category subset; the 11 dropped names are listed in
    :data:`_COCO91_DROPPED_NAMES` and silent-skipped here. Any OTHER
    missing name raises loud — that's the failure mode rfdetr's adapter
    explicitly defends against (a subset / aliased / corrupted GT JSON
    that would otherwise produce a partially-empty cache under a pinned
    revision SHA).
    """
    name_to_cat_id = {cat["name"]: int(cat["id"]) for cat in gt["categories"]}
    mapping: dict[int, int] = {}
    for label_id, name in id2label.items():
        if name == "N/A":
            continue
        cat_id = name_to_cat_id.get(name)
        if cat_id is None:
            if name in _COCO91_DROPPED_NAMES:
                continue
            raise RuntimeError(
                f"DETR label {label_id} ('{name}') has no matching category "
                f"in the GT JSON, and it isn't one of the 11 documented "
                f"COCO-91→COCO-80 drops ({sorted(_COCO91_DROPPED_NAMES)}). "
                f"Likely a subset / aliased / corrupted GT — refusing to "
                f"silently produce a partial cache. GT category names: "
                f"{sorted(name_to_cat_id)}"
            )
        mapping[label_id] = cat_id
    return mapping


def _xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def _instantiate_model(revision: str) -> tuple[Any, Any]:
    """Lazy transformers import + processor/model instantiation.

    Loaded with ``revision=`` so the cache filename's commit-SHA pin is
    the same SHA the weights resolve to. ``low_cpu_mem_usage=True``
    streams shards into memory rather than materializing the full
    state dict twice; harmless on CPU but cuts peak RSS during load.

    Pins ``torch.set_num_threads(1)`` before the first forward pass so
    intra-op summation order is the same regardless of the host's
    physical core count — the cache key is ``(model, revision, dataset)``
    only, so without this pin two machines on the same revision would
    populate the cache with bit-different bytes via summation-order
    drift in matmul reductions.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    torch.set_num_threads(1)

    processor = AutoImageProcessor.from_pretrained(_MODEL_ID, revision=revision)
    model = AutoModelForObjectDetection.from_pretrained(
        _MODEL_ID,
        revision=revision,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return processor, model


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
    the (H, W) it scales back into); COCO results want xywh, which is
    a one-liner in :func:`_xyxy_to_xywh`.
    """
    import torch

    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)

    # int64 (not float64) target_sizes: transformers' post-process builds
    # scale_fct from this tensor and multiplies the float32 boxes against
    # it. fp64 here silently upcasts the box arithmetic to fp64, making
    # cached bytes sensitive to a transformers internal that could shift
    # between minor versions. int → boxes stay at their native dtype.
    target_sizes = torch.tensor([image_size_hw], dtype=torch.int64)
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
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

    Atomic write protects against SIGINT mid-run leaving a half-written
    JSON the next session would mistake for complete — same pattern as
    ``coco_val_cache._atomic_download``.
    """
    if cache_path.is_file():
        return cache_path.read_bytes()

    from PIL import Image

    processor, model = _instantiate_model(revision)
    id2label: dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}
    class_mapping = _coco_class_mapping(gt, id2label)

    # `Sequence` (not `Iterable`) so `len()` below is type-honest. The
    # GT JSON loader always returns a list; pinning the annotation here
    # prevents a future generator-shaped caller from silently slipping
    # past type-checking and crashing on `len()` mid-run.
    image_list: Sequence[dict[str, Any]] = gt["images"]
    images: Iterable[dict[str, Any]] = image_list
    if progress:
        from tqdm import tqdm

        images = tqdm(image_list, total=len(image_list), desc=f"detr-r50 {_DATASET_LABEL}")

    records: list[dict[str, Any]] = []
    for img in images:
        image_path = image_dir / img["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(
                f"image referenced by GT JSON missing on disk: {image_path}. "
                f"Re-run the COCO val2017 fetcher; the cache root must contain "
                f"the full val2017/ directory next to instances_val2017.json."
            )
        with Image.open(image_path) as pil:
            pil_rgb = pil.convert("RGB")
            image_h = int(img["height"])
            image_w = int(img["width"])
            records.extend(
                _records_for_image(
                    image_id=int(img["id"]),
                    image_size_hw=(image_h, image_w),
                    processor=processor,
                    model=model,
                    image=pil_rgb,
                    class_mapping=class_mapping,
                )
            )

    payload = json.dumps(records).encode("utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    part = cache_path.with_suffix(cache_path.suffix + ".part")
    part.write_bytes(payload)
    part.replace(cache_path)
    return payload
