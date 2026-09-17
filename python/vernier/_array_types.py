"""ADR-0030 array-ingest payload shapes.

Runtime TypedDict definitions used by the streaming and background
evaluators when callers hand in numpy / DLPack arrays instead of JSON
bytes. The stubs in :mod:`vernier._core` import the same names so the
``update`` / ``submit`` signatures resolve cleanly under pyright.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray


class UncompressedRLE(TypedDict):
    """COCO RLE shape on the array-ingest path (uncompressed counts).

    ``counts`` is the uncompressed run-length array (uint32, contiguous).
    ``size`` is ``(height, width)`` in COCO order.
    """

    counts: NDArray[np.uint32]
    size: tuple[int, int]


class CompressedRLE(TypedDict):
    """COCO compressed RLE shape (6-bit ASCII bytes, as emitted by
    ``pycocotools.mask.encode``).

    ``counts`` is the compressed bytes payload, validated as UTF-8 ASCII at ingest.
    ``size`` is ``(height, width)`` in COCO order.
    """

    counts: bytes
    size: tuple[int, int]


#: Per-item shape accepted by ``Detections.rles``. A 2-D ``bool`` or
#: ``uint8`` array is treated as a bitmask of shape ``(H, W)``; C- and
#: F-order are both accepted (C-order incurs a single column-major copy
#: at ingest).
RLEInput: TypeAlias = UncompressedRLE | CompressedRLE | NDArray[np.bool_] | NDArray[np.uint8]


class JsonRLE(TypedDict):
    """The RLE shape a COCO results *file* carries, as ``json.load``
    leaves it.

    ``counts`` is either the compressed 6-bit string (quirk **K3**: JSON
    has no bytes type, so the same payload arrives as ``str``) or the
    uncompressed run lengths as a plain list of ints. ``size`` is
    ``(height, width)`` in COCO order.

    Accepted on :attr:`ResultAnnotation.segmentation` only — the
    columnar :attr:`Detections.rles` is an in-memory array surface and
    takes :data:`RLEInput`.
    """

    counts: str | Sequence[int]
    size: Sequence[int]


#: COCO polygon segmentation: one flat ``[x0, y0, x1, y1, …]`` list per
#: polygon, nested one level. Sub-polygons are unioned into a single mask
#: (quirk **K2**).
PolygonSegmentation: TypeAlias = Sequence[Sequence[float]]

#: Per-annotation ``segmentation`` shape on the result-dict route. It is
#: everything a results file can hold (:data:`PolygonSegmentation`,
#: :class:`JsonRLE`) plus the in-memory forms ADR-0030 added
#: (:data:`RLEInput`), so the route accepts every payload the file route
#: does and then some.
SegmentationInput: TypeAlias = RLEInput | JsonRLE | PolygonSegmentation


class Detections(TypedDict, total=False):
    """One per-image detection batch in array form.

    Fields are gated by ``iou_type``:

    - ``bbox``: ``image_id``, ``boxes``, ``scores``, ``labels``.
    - ``segm`` / ``boundary``: above plus ``rles``.
    - ``keypoints``: ``image_id``, ``boxes``, ``scores``, ``labels``,
      ``keypoints``.

    Required dtypes (no silent promotion — opt in via
    ``cast_inputs=True``):

    - ``boxes``: ``float64`` ``(N, 4)`` C-contiguous, xywh.
    - ``scores``: ``float64`` ``(N,)``.
    - ``labels``: ``int64`` ``(N,)``.
    - ``rles[i]`` (uncompressed dict): ``counts: uint32`` 1-D contiguous, ``size: (h, w)``.
    - ``rles[i]`` (compressed dict): ``counts: bytes`` (COCO 6-bit ASCII), ``size: (h, w)``.
    - ``rles[i]`` (bitmask): 2-D ``bool`` or ``uint8``, shape ``(H, W)``, C- or F-order.
    - ``keypoints``: ``float64`` ``(N, K, 3)`` C-contiguous.
    """

    image_id: int
    boxes: NDArray[np.float64]
    scores: NDArray[np.float64]
    labels: NDArray[np.int64]
    rles: Sequence[RLEInput]
    keypoints: NDArray[np.float64]


#: Union of legal forms for ``StreamingEvaluator.update`` / ``BackgroundEvaluator.submit``.
class ResultAnnotation(TypedDict, total=False):
    """One COCO *result* annotation — the shape ``loadRes`` consumes.

    This is the per-annotation dict a pycocotools- or TorchMetrics-style
    caller already has in hand. Passing the list directly skips the
    ``json.dumps`` / parse round trip the bytes route would otherwise
    pay for the same data.

    ``image_id``, ``category_id``, ``bbox`` and ``score`` are always
    required. ``keypoints`` is required under ``iou_type='keypoints'``.

    ``segmentation`` is optional on every ``iou_type``, exactly as it is
    in a results *file*: under ``'segm'`` / ``'boundary'`` an annotation
    without one is governed by quirk **J2** (``parity_mode='strict'``
    synthesizes the bbox rectangle pycocotools synthesizes;
    ``'corrected'`` refuses and names the detection). It takes
    :data:`SegmentationInput` — polygons, either RLE ``counts``
    encoding, or a 2-D bitmask.

    ``area`` and ``iscrowd`` are accepted and ignored: area is derived
    (quirk **J3**) and detections are never crowd (quirks **E2**/**J4**).
    An explicit ``id`` is preserved; an absent one is auto-assigned
    ``1..N`` by position (quirk **J1**).
    """

    image_id: int
    category_id: int
    bbox: Sequence[float]
    score: float
    id: int
    segmentation: SegmentationInput
    keypoints: Sequence[float]
    num_keypoints: int


#: ``(N, 7)`` C-contiguous float64 detection matrix, laid out as
#: ``image_id, x, y, w, h, score, category_id``. The ``image_id`` and
#: ``category_id`` columns must hold exact integers within 2^53; a
#: fractional or oversized value is rejected rather than truncated.
DetectionMatrix: TypeAlias = NDArray[np.float64]

DetectionsInput: TypeAlias = (
    bytes
    | Detections
    | Sequence[Detections]
    | ResultAnnotation
    | Sequence[ResultAnnotation]
    | DetectionMatrix
)


__all__ = [
    "CompressedRLE",
    "DetectionMatrix",
    "Detections",
    "DetectionsInput",
    "JsonRLE",
    "PolygonSegmentation",
    "RLEInput",
    "ResultAnnotation",
    "SegmentationInput",
    "UncompressedRLE",
]
