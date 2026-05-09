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

#: Back-compat alias: ``RLE`` is still exported and equals the original
#: uncompressed shape.
RLE: TypeAlias = UncompressedRLE


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
DetectionsInput: TypeAlias = bytes | Detections | Sequence[Detections]


__all__ = [
    "RLE",
    "CompressedRLE",
    "Detections",
    "DetectionsInput",
    "RLEInput",
    "UncompressedRLE",
]
