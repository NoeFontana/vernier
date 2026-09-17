"""COCO-JSON normalizers shared by the shim and the array grid.

These are the small, fiddly conversions between a COCO dictionary
built for `pycocotools` and the COCO JSON vernier's Rust core parses.
They exist because `pycocotools.cocoeval.COCOeval` reads less of the
dataset than vernier's schema requires — it never touches an image's
size unless a mask on that image is converted — so a dataset assembled
for `COCOeval` can legitimately omit sizes vernier insists on, and can
carry `bytes` RLE counts straight out of `pycocotools.mask.encode`
that :func:`json.dumps` refuses.

Re-exported from :mod:`vernier.adapters`, which is the canonical
import path — this module is private so the drop-in in
:mod:`vernier._compat` can share it without an import cycle through
the adapters package. See ADR-0055.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "coco_json_default",
    "to_coco_json",
    "with_mask_image_sizes",
    "with_placeholder_image_sizes",
]


def with_placeholder_image_sizes(dataset: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fill every missing image ``width`` / ``height`` with ``0``.

    For the IoU types that never read an image size — ``bbox`` and
    ``keypoints``, whose kernels work in annotation coordinates —
    `pycocotools` reads ``images[].width`` / ``height`` only from
    ``annToRLE``, so a dataset assembled for those types routinely
    omits them (TorchMetrics' ``_get_coco_format`` is one such
    producer). vernier's GT schema requires both on every image, and a
    ``0`` placeholder is inert for these kernels.

    Do not use this for ``segm`` or ``boundary``: a ``0x0`` size
    rasterizes a polygon to nothing and would score silently. Use
    :func:`with_mask_image_sizes` there.

    Returns ``dataset`` unchanged when every image is already sized;
    otherwise a shallow copy. The caller's dictionaries are never
    mutated.
    """
    images = dataset.get("images", [])
    if all("width" in image and "height" in image for image in images):
        return dataset
    return {
        **dataset,
        "images": [{"width": 0, "height": 0, **image} for image in images],
    }


def with_mask_image_sizes(
    dataset: Mapping[str, Any],
    detection_sizes: Mapping[Any, tuple[int, int] | None],
) -> Mapping[str, Any]:
    """Fill missing image sizes for a mask dataset, where it is safe to.

    Under ``segm`` / ``boundary``, `pycocotools`' ``_prepare`` calls
    ``annToRLE`` on every ground-truth annotation against ``cocoGt.imgs``
    and on every detection against ``cocoDt.imgs``, and ``annToRLE``
    reads the image's ``height`` / ``width`` before it looks at the
    segmentation. A size is therefore read only for an image some
    annotation points at, and from the side that annotation belongs to.
    Producers that mirror that — TorchMetrics sizes an image only when
    *that* side has masks on it — leave gaps `pycocotools` never
    notices.

    vernier's GT schema needs a size on every image and checks each
    detection's RLE size against it, so this fills a gap only where
    `pycocotools` would not have failed:

    - an image nothing points at, on either side, becomes ``0x0`` — no
      kernel reads it;
    - an image only detections point at takes the size the detection
      side knows;
    - every other gap is left in place, and surfaces as vernier's schema
      error exactly where `pycocotools` raises ``KeyError``.

    ``detection_sizes`` is keyed by the image ids the *detections* point
    at — that key set is what distinguishes the first two cases — and
    maps each to that side's ``(height, width)``, or to ``None`` when the
    detection side does not know it either. Build it from the
    detections' ``images`` entries for a `pycocotools` ``COCO`` object,
    or from the first detection mask's ``size`` for a caller driving the
    array grid.

    Returns ``dataset`` unchanged when every image is already sized;
    otherwise a shallow copy. The caller's dictionaries are never
    mutated.
    """
    images = dataset.get("images", [])
    if all("width" in image and "height" in image for image in images):
        return dataset
    annotated = {ann["image_id"] for ann in dataset.get("annotations", [])}

    def sized(image: Mapping[str, Any]) -> Mapping[str, Any]:
        image_id = image["id"]
        if ("width" in image and "height" in image) or image_id in annotated:
            return image
        if image_id in detection_sizes:
            size = detection_sizes[image_id]
            if size is None:
                return image
            height, width = size
            return {**image, "height": height, "width": width}
        return {"width": 0, "height": 0, **image}

    return {**dataset, "images": [sized(image) for image in images]}


def coco_json_default(obj: Any) -> Any:
    """``json.dumps(default=...)`` hook for `pycocotools`-shaped values.

    Handles the two non-JSON types a COCO dictionary assembled against
    `pycocotools` carries:

    - ``bytes`` RLE ``counts``, straight out of
      ``pycocotools.mask.encode``. COCO RLE is ASCII (quirk **K3**), so
      they decode as ASCII.
    - NumPy scalars, which ``pycocotools.mask.area`` and ``loadRes``
      leave on ``area`` / ``score`` fields.

    Anything else raises :class:`TypeError`, as :mod:`json` would.
    """
    if isinstance(obj, bytes):
        return obj.decode("ascii")
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")


def to_coco_json(obj: Mapping[str, Any] | Iterable[Any]) -> bytes:
    """Serialize a COCO dataset or annotation list to the bytes vernier parses.

    :func:`json.dumps` with :func:`coco_json_default` and a UTF-8
    encode — the one-liner every caller handing a `pycocotools`-shaped
    dictionary to a vernier grid would otherwise write itself.
    """
    return json.dumps(obj, default=coco_json_default).encode()
