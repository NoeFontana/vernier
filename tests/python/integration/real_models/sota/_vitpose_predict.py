"""ViTPose-base-simple → COCO keypoints results JSON adapter.

``usyd-community/vitpose-base-simple`` is the Apache-2.0, CPU-runnable
COCO-person 17-keypoint baseline anchoring the keypoints cell of the
Hugging Face SOTA harness. ViTPose is a *top-down* estimator: it
consumes a person box crop and outputs a 17-joint heatmap per crop —
unlike the DETR / Mask2Former cells (bottom-up, one forward per
image). For a clean parity surface we feed it the **GT person boxes**
from ``person_keypoints_val2017.json``: this isolates the keypoint
head's numerics from any detector quirks, mirrors the canonical
"GT-boxes" eval mode mmpose / MMCV use for ViTPose's published
numbers, and means the cache content is fully determined by the
weights pin + the (already SHA-pinned) GT JSON.

Cache discipline mirrors the DETR / Mask2Former cells: the filename
embeds the FULL 40-hex revision SHA (see
:func:`real_predictions_cache.vitpose_cache_filename`) so a weights
bump invalidates by construction. Cross-cell scaffolding (thread pin,
revision-pinned load, atomic write) lives in :mod:`._harness_common`.

Inference cost: ViTPose is light (~256x192 input, ~86M params); first
end-to-end run takes ~2-3 h on 8-core CPU across the ~11k GT person
annotations in val2017 (NOT 5000 images — top-down iteration is
per-box). Subsequent runs read from disk and skip the model.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from real_predictions_cache import VITPOSE_REVISION

from ._harness_common import (
    atomic_write_bytes,
    load_processor_and_model,
    tqdm_or_passthrough,
)

_MODEL_ID = "usyd-community/vitpose-base-simple"
_DATASET_LABEL = "coco-val2017"

#: COCO uses category_id=1 for ``person`` in the keypoints GT. ViTPose
#: emits a single category (its 17-joint head is COCO-person specific),
#: so we hard-pin the join rather than walking ``id2label`` — the
#: model's id2label is JOINT names ("Nose", "L_Eye", ...), not category
#: names. The category-name sanity check is enforced separately at
#: load-time below.
_PERSON_CATEGORY_ID = 1

#: Per-keypoint score floor for the visibility flag emitted in the COCO
#: JSON. pycocotools' OKS evaluator ignores ``v`` on the DT side (only
#: the GT's ``v`` gates which keypoints contribute to the score, per
#: ADR-0012 quirk F5), so this is purely informational — mmpose /
#: downstream viewers read it but the parity numbers do not. We hard-
#: pin DT-side ``v`` to {1, 2} (never 0): "not labelled" is a GT-side
#: concept, and any DT-side test asserting "we saw both 1 and 2" would
#: be tautological on a working model rather than exercising F5. The
#: F5 quirk surface is exercised by the GT fixtures in the parity
#: suite, not by these real-model DT outputs.
_KP_VIS_THRESHOLD = 0.3

#: COCO's 17-joint name set. Used as the documented expected set for
#: the model's id2label sanity check at load time; a mismatch fails
#: loudly rather than silently writing a cache against a different
#: skeleton topology. ViTPose-base-simple ships these exact names
#: (some with underscore prefixes); we normalise case-insensitively.
_COCO_KEYPOINT_NAMES_LOWER = frozenset(
    {
        "nose",
        "l_eye",
        "r_eye",
        "l_ear",
        "r_ear",
        "l_shoulder",
        "r_shoulder",
        "l_elbow",
        "r_elbow",
        "l_wrist",
        "r_wrist",
        "l_hip",
        "r_hip",
        "l_knee",
        "r_knee",
        "l_ankle",
        "r_ankle",
    }
)


def _iter_person_annotations(
    gt: dict[str, Any],
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield ``(image_record, annotation_record)`` for each GT person
    annotation with a usable bounding box.

    Skips annotations with ``iscrowd=1`` (COCO convention: crowd boxes
    are exclusion regions, not single-person crops) and zero-area
    boxes (would crash the processor's coordinate-warp at inference).
    Joined against the GT ``images`` table once so the caller can read
    the original image-size H/W without a per-annotation lookup.
    """
    images_by_id: dict[int, dict[str, Any]] = {int(img["id"]): img for img in gt["images"]}
    for ann in gt["annotations"]:
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        bbox = ann.get("bbox")
        if bbox is None or float(bbox[2]) <= 0.0 or float(bbox[3]) <= 0.0:
            continue
        img = images_by_id.get(int(ann["image_id"]))
        if img is None:
            # GT internally inconsistent — refuse to silently drop.
            # Same loud-fail discipline as `iter_image_records`.
            raise RuntimeError(
                f"keypoints GT references image_id={ann['image_id']} "
                f"that's not in the `images` table. Refusing to populate "
                f"a partial cache under a pinned revision SHA."
            )
        yield img, ann


def _assert_coco_keypoint_topology(id2label: dict[int, str]) -> None:
    """Loud-fail if the model's joint names don't match COCO's 17.

    Same loud-fail invariant the name-based class mapping enforces on
    the detection cells: a topology mismatch (e.g. someone re-uploads
    the model with MPII's 16-joint head) would silently produce a
    cache against the wrong skeleton, surfacing weeks later as a
    silent OKS regression. Names are normalised case-insensitively
    because the upstream model card capitalises some entries.
    """
    if len(id2label) != 17:
        raise RuntimeError(
            f"ViTPose joint head has {len(id2label)} entries; expected 17 "
            f"(COCO-person topology). id2label: {id2label}"
        )
    model_names_lower = {name.lower() for name in id2label.values()}
    missing = _COCO_KEYPOINT_NAMES_LOWER - model_names_lower
    extra = model_names_lower - _COCO_KEYPOINT_NAMES_LOWER
    if missing or extra:
        raise RuntimeError(
            f"ViTPose joint names don't match COCO's 17 ({_DATASET_LABEL}). "
            f"missing={sorted(missing)} extra={sorted(extra)}. Refusing to "
            f"populate a partial cache under a pinned revision SHA."
        )


def _record_for_box(
    *,
    image_id: int,
    bbox_xywh: list[float],
    processor: Any,
    model: Any,
    image: Any,
) -> dict[str, Any]:
    """Run one forward pass on one person crop; emit a COCO keypoints record.

    ``post_process_pose_estimation`` returns ``keypoints`` in the
    original image's pixel coordinates and ``scores`` as the per-joint
    confidence (the heatmap peak value). COCO results want a flat
    51-element ``[x, y, v] * 17`` list plus a single per-instance
    ``score`` (we use the mean per-joint confidence as the instance
    score — the convention mmpose uses for ViTPose's published
    numbers).
    """
    import torch

    # boxes is a list[per_image][per_box][4]; we feed one image, one box.
    boxes = [[bbox_xywh]]
    inputs = processor(images=[image], boxes=boxes, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)

    # Do NOT pass `target_sizes` here. Despite its docstring claiming
    # `(height, width)` semantics, transformers' VitPoseImageProcessor
    # at v4.x actually treats `target_sizes[i]` as a `(W, H)` scale
    # factor and multiplies the box by `[W, H, W, H]`, ASSUMING the box
    # is normalized to [0,1]. Our boxes are already in original-image
    # pixel space (COCO xywh, sourced directly from GT person
    # annotations), so passing `target_sizes` scales them by the image
    # dimensions into nonsense (~480x the intended values on COCO
    # val2017) and the resulting keypoints land outside any sane
    # coordinate space — pycocotools then reports 0 mAP.
    # The default (None) path skips the scale step and the inverse
    # box-to-center-and-scale transform on the heatmap side already
    # returns keypoints in original-image pixel space.
    per_image_results = processor.post_process_pose_estimation(
        outputs,
        boxes=boxes,
    )
    assert len(per_image_results) == 1, (
        f"ViTPose post_process_pose_estimation returned "
        f"{len(per_image_results)} per-image entries; expected 1 "
        f"(one image per forward pass)"
    )
    per_box_results = per_image_results[0]
    assert len(per_box_results) == 1, (
        f"ViTPose post_process_pose_estimation returned "
        f"{len(per_box_results)} per-box entries; expected 1 "
        f"(one box per forward pass — any future per-image batching of "
        f"multiple boxes must remove this assert and revisit the cache "
        f"contract)"
    )
    result = per_box_results[0]
    kp_xy = result["keypoints"].tolist()  # shape (17, 2)
    kp_scores = result["scores"].tolist()  # shape (17,)

    flat: list[float] = []
    for (x, y), s in zip(kp_xy, kp_scores, strict=True):
        # COCO v in {0,1,2}; for predictions we map score-floor to
        # 2 (visible) vs 1 (labelled-but-low-conf). pycocotools' OKS
        # eval ignores DT-side v but downstream viewers honour it.
        v = 2 if float(s) >= _KP_VIS_THRESHOLD else 1
        flat.extend([float(x), float(y), int(v)])

    # Per-instance score is the mean across the 17 joint confidences —
    # the same reduction mmpose uses for COCO keypoints submission.
    instance_score = float(sum(kp_scores) / len(kp_scores))

    return {
        "image_id": int(image_id),
        "category_id": _PERSON_CATEGORY_ID,
        # Echoed verbatim from the GT person box. pycocotools accepts
        # bbox-less keypoint DT entries (falls back to GT area); the
        # vernier-side JSON parser declares the field non-optional.
        # Top-down ViTPose already has the box on hand, so emit it
        # rather than make the parser path looser.
        "bbox": list(bbox_xywh),
        "keypoints": flat,
        "score": instance_score,
    }


def predict_coco_val(
    *,
    gt: dict[str, Any],
    image_dir: Path,
    cache_path: Path,
    revision: str = VITPOSE_REVISION,
    progress: bool = True,
) -> bytes:
    """Run ViTPose inference on every GT person annotation; emit COCO keypoints JSON.

    Owns the cache contract end-to-end: a hit on ``cache_path`` returns
    the bytes without instantiating the model (which would download
    weights). On a miss, instantiates the model lazily, runs inference
    per GT person box, atomic-writes the cache, and returns the bytes.

    Top-down iteration: the loop is over GT annotations, not images,
    because ViTPose needs a person crop. Images with no GT person
    annotations contribute nothing to the cache (no detector to
    surface false positives on background) — symmetric with how the
    canonical mmpose / MMCV ViTPose eval is configured.
    """
    if cache_path.is_file():
        return cache_path.read_bytes()

    from PIL import Image

    processor, model = load_processor_and_model(
        _MODEL_ID, revision, model_cls_name="VitPoseForPoseEstimation"
    )
    id2label: dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}
    _assert_coco_keypoint_topology(id2label)

    # Materialize the annotation iterator so tqdm's bar gets a stable
    # total. The generator above is a lightweight join over the GT
    # tables; full realisation is ~11k entries (val2017 person count),
    # negligible vs the inference cost.
    annotations = list(_iter_person_annotations(gt))

    records: list[dict[str, Any]] = []
    # Cache the PIL Image across consecutive annotations on the same
    # image_id — val2017 person annotations are clustered by image, so
    # this avoids re-decoding the same JPEG dozens of times. Open with
    # a single context manager per image to keep file descriptor
    # discipline.
    current_image_id: int | None = None
    current_rgb: Any = None
    for ann_idx, (img, ann) in enumerate(
        tqdm_or_passthrough(
            annotations,
            desc=f"vitpose {_DATASET_LABEL}",
            progress=progress,
        )
    ):
        image_id = int(img["id"])
        if image_id != current_image_id:
            image_path = image_dir / img["file_name"]
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"image referenced by keypoints GT JSON missing on "
                    f"disk: {image_path}. Re-run the dataset fetcher; "
                    f"the cache root must contain the full images/ "
                    f"directory next to the GT JSON."
                )
            with Image.open(image_path) as pil:
                current_rgb = pil.convert("RGB")
            current_image_id = image_id

        bbox_xywh = [float(v) for v in ann["bbox"]]
        records.append(
            _record_for_box(
                image_id=image_id,
                bbox_xywh=bbox_xywh,
                processor=processor,
                model=model,
                image=current_rgb,
            )
        )
        # Silence pyright's unused-loop-var: ann_idx documents the
        # cache resume granularity (per-annotation, not per-image) for
        # any future incremental-write refactor.
        _ = ann_idx

    payload = json.dumps(records).encode("utf-8")
    atomic_write_bytes(cache_path, payload)
    return payload
