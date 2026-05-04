"""rf-detr → COCO JSON adapter for the TIDE validation harness.

The :class:`rfdetr.RFDETRNano` and :class:`rfdetr.RFDETRSegNano` models
emit a :class:`supervision.Detections` object per image; vernier's
``error_decomposition`` wants a JSON byte payload in COCO's
``COCO.loadRes`` shape. This module is the single adapter — the harness
delegates here, the cache stores the COCO-shaped output, and the test
code never touches rfdetr's native types.

Cache discipline: predictions are keyed on ``(model_name,
model_version, dataset_id)``. Re-running the harness with the same pin
hits the cache and skips inference; bumping the rfdetr pin (an
ADR-level operation per the vendoring policy) invalidates the cache by
construction.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np
    from supervision import Detections


_RFDETR_VERSION = "1.6.5.post0"
_DATASET_ID = "coco-val2017"

#: rf-detr model variants the harness exercises. ``nano`` is the
#: bbox-only RFDETRNano; ``segnano`` is the instance-seg RFDETRSegNano
#: (which also produces masks usable by the boundary kernel).
ModelName = Literal["nano", "segnano"]


def cache_filename(model_name: ModelName) -> str:
    """Stable filename for cached predictions.

    Versioned + dataset-tagged so a pin bump or dataset swap can't
    silently reuse stale predictions.
    """
    return f"rfdetr-{model_name}-{_RFDETR_VERSION}-{_DATASET_ID}.json"


def predictions_cache_root() -> Path:
    """Per-user cache for model predictions (machine-local).

    Resolves via :func:`platformdirs.user_cache_dir` so the path is
    XDG-correct on Linux, ``~/Library/Caches/...`` on macOS, and
    ``%LOCALAPPDATA%\\...`` on Windows. Predictions are large and
    slow to recompute — keying them per ``(model, version, dataset)``
    lets a re-run of the harness skip inference entirely.
    """
    import platformdirs

    root = Path(platformdirs.user_cache_dir("vernier")) / "real-models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _coco_class_mapping(gt: dict[str, Any], dense_class_names: list[str]) -> dict[int, int]:
    """Map rfdetr's dense ``class_id`` (0..79) to COCO's sparse
    ``category_id`` (1..90 with gaps).

    Resolved by name match against the GT JSON's ``categories`` list,
    not by sorted-position fallback: matching by index works on
    canonical COCO splits but breaks silently on any subset that drops
    a class. Name-match fails loudly, which is what we want.
    """
    name_to_cat_id = {cat["name"]: int(cat["id"]) for cat in gt["categories"]}
    mapping: dict[int, int] = {}
    for dense_idx, name in enumerate(dense_class_names):
        if name not in name_to_cat_id:
            raise RuntimeError(
                f"rfdetr COCO class {dense_idx} ('{name}') has no matching "
                f"category in the GT JSON; the harness can't map class ids. "
                f"GT category names: {sorted(name_to_cat_id)}"
            )
        mapping[dense_idx] = name_to_cat_id[name]
    return mapping


def _xyxy_to_xywh(xyxy: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    return [x1, y1, x2 - x1, y2 - y1]


def _mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """Boolean mask (H, W) → COCO RLE.

    ``np.asfortranarray(mask, dtype=np.uint8)`` does the dtype cast and
    Fortran-order enforcement in a single pass — the encoder needs both.
    The ``counts`` field comes back as bytes; decoding to ``ascii`` is
    required so the JSON layer doesn't choke on non-UTF-8 bytes.
    """
    import numpy as np
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}


def _detections_to_records(
    detections: Detections,
    image_id: int,
    class_mapping: dict[int, int],
    *,
    include_masks: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    n = len(detections.xyxy)
    if n == 0:
        return records
    confidences = detections.confidence
    class_ids = detections.class_id
    masks = detections.mask if include_masks else None
    if confidences is None or class_ids is None:
        raise RuntimeError(
            "rfdetr returned a Detections object without confidence/class_id; "
            "the harness assumed a populated detection set"
        )
    if include_masks and masks is None:
        raise RuntimeError("expected segmentation masks on a seg model output, got None")

    for i in range(n):
        dense_class = int(class_ids[i])
        if dense_class not in class_mapping:
            continue
        rec: dict[str, Any] = {
            "image_id": int(image_id),
            "category_id": int(class_mapping[dense_class]),
            "bbox": _xyxy_to_xywh(detections.xyxy[i]),
            "score": float(confidences[i]),
        }
        if include_masks and masks is not None:
            rec["segmentation"] = _mask_to_rle(masks[i])
        records.append(rec)
    return records


def _instantiate_model(model_name: ModelName) -> tuple[Any, bool]:
    """Lazy rfdetr import + model instantiation. Returns ``(model, include_masks)``."""
    import rfdetr

    if model_name == "nano":
        return rfdetr.RFDETRNano(), False
    return rfdetr.RFDETRSegNano(), True


def predict_coco_val(
    *,
    model_name: ModelName,
    gt: dict[str, Any],
    image_dir: Path,
    cache_path: Path,
    threshold: float = 0.05,
    progress: bool = True,
) -> bytes:
    """Run rfdetr inference on every image in ``gt['images']``, emit COCO JSON.

    Owns the cache contract end-to-end: a hit on ``cache_path`` returns
    the bytes without instantiating a model (which would download
    weights). On a miss, instantiates the model lazily, runs inference,
    writes the cache, and returns the bytes. Either way the bytes are
    in the same shape ``vernier.instance.error_decomposition`` consumes.

    ``threshold=0.05`` is deliberately permissive — TIDE rewards keeping
    low-confidence FPs visible (they populate the Bkg / Bkg+Cls bins);
    cutting them at 0.5 would distort the decomposition. The mAP
    accumulator sees the full PR curve regardless.
    """
    if cache_path.is_file():
        return cache_path.read_bytes()

    from rfdetr.assets.coco_classes import COCO_CLASSES

    model, include_masks = _instantiate_model(model_name)
    class_mapping = _coco_class_mapping(gt, list(COCO_CLASSES))
    images: Iterable[dict[str, Any]] = gt["images"]

    if progress:
        from tqdm import tqdm

        images = tqdm(list(images), desc=f"rfdetr-{model_name} val2017")

    records: list[dict[str, Any]] = []
    for img in images:
        image_path = image_dir / img["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(
                f"image referenced by GT JSON missing on disk: {image_path}. "
                f"Re-run the COCO val2017 fetcher; the cache root must contain "
                f"the full val2017/ directory next to instances_val2017.json."
            )
        detections = model.predict(str(image_path), threshold=threshold)
        records.extend(
            _detections_to_records(
                detections,
                image_id=int(img["id"]),
                class_mapping=class_mapping,
                include_masks=include_masks,
            )
        )

    payload = json.dumps(records).encode("utf-8")
    cache_path.write_bytes(payload)
    return payload
