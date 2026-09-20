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

    ``area`` is **carried**, exactly as a results *file* carries it, and
    what it does is decided downstream by ``dt_area`` (quirk **J3**):
    under the ``"bbox"`` default it is ignored and the area is derived
    from the box, and under ``dt_area="supplied"`` it is the area that
    buckets the detection into small / medium / large. Dropping it here
    would make this route score differently from the file route for the
    same payload, so it is not dropped.

    ``iscrowd`` is accepted and ignored: a detection is never a crowd
    (quirks **E2**/**J4**).

    ``rbox`` is required under ``iou_type='rotated_box'`` and ``quad``
    under ``'quad'`` (ADR-0063). They are deliberately separate keys
    rather than a longer ``bbox``: detectron2 overloads ``bbox`` with a
    fifth angle value, and a length-5 ``bbox`` read as ``[x, y, w, h]``
    evaluates a box at the wrong place with nothing raised. ``bbox``
    stays the axis-aligned envelope.

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
    area: float
    iscrowd: int
    rbox: Sequence[float]
    quad: Sequence[float]


class GtCategory(TypedDict, total=False):
    """One COCO ``categories`` entry (ADR-0060).

    The ``categories`` section is the one part of a GT document this
    surface does not express columnar-ly: ``name`` is a string, which has
    no array form, and the section is O(K) with K small (80 on COCO,
    1203 on LVIS) rather than O(N) in annotations.
    """

    id: int
    name: str
    supercategory: str


class GtImages(TypedDict, total=False):
    """The ``images`` section of a GT document, as columns (ADR-0060).

    ``id``, ``width`` and ``height`` are required ``(M,)`` ``int64``
    arrays. ``width`` and ``height`` are range-checked into ``u32``; a
    negative or oversized dimension is rejected, never truncated.

    ``file_name`` is a per-image sequence of ``str`` (or ``None``), not
    an array — nothing in the evaluation reads it, but it participates
    in :attr:`vernier.CocoDataset.dataset_hash`, so a caller
    reproducing a JSON document through this route can carry it.
    """

    id: NDArray[np.int64]
    width: NDArray[np.int64]
    height: NDArray[np.int64]
    file_name: Sequence[str | None]


class GtAnnotations(TypedDict, total=False):
    """The ``annotations`` section of a GT document, as columns (ADR-0060).

    Required, all length ``N``:

    - ``id``: ``int64``. Ground-truth ids are **supplied, never
      assigned** — the mirror image of quirk **J1** on the detection
      side — and are observable through ``evalImgs['gtIds']``.
    - ``image_id``, ``category_id``: ``int64``.
    - ``bbox``: ``float64`` ``(N, 4)`` C-contiguous, xywh.
    - ``area``: ``float64`` ``(N,)``. Read **verbatim**. Unlike a
      detection's (quirk **J3**, derived from the box by default), a
      ground truth's area is the number COCO recorded and the number the
      small / medium / large bucketing reads, so it is required rather
      than derivable.
    - ``iscrowd``: ``bool`` / ``uint8`` / ``int64`` ``(N,)``. Required,
      because it drives the ignore resolution (quirk **D1**) and the
      crowd IoA denominator (quirk **E1**).

    Optional:

    - ``ignore``: ``bool`` / ``uint8`` / ``int64`` ``(N,)``.
    - ``segmentation``: a length-``N`` sequence whose entries are
      ``None`` or any :data:`SegmentationInput` shape. A DataFrame
      column (an ``object``-dtype array, or a ``Series``) is accepted;
      a *numeric* array is not, because a stacked ``(N, H, W)`` bitmask
      would otherwise be walked into ``N`` planes.
    - ``keypoints``: ``float64`` ``(N, K, 3)`` C-contiguous.
    - ``num_keypoints``: ``int64`` ``(N,)``.

    **Absent versus zero.** A column has no null, but ``ignore`` and
    ``num_keypoints`` are genuinely optional *per annotation* in COCO
    JSON and mean something different absent than present-and-zero —
    under quirk **D1**, an absent ``ignore`` lets ``parity_mode``
    ``"corrected"`` fall back to ``iscrowd``, while a present ``0`` pins
    it false. So: pass such a column as a **signed** array, in which a
    **negative entry means the field was absent on that annotation**;
    omitting the column means absent on every annotation. A ``bool`` or
    ``uint8`` column has no negative and therefore means present
    everywhere. Neither field has a meaningful negative value otherwise.

    The rule is scoped to those two columns. ``iscrowd`` is *required*,
    so there is no "absent" for a negative entry to mean, and a negative
    one is rejected naming the annotation rather than read as false —
    which would hand back a different crowd set than was passed, and no
    diagnostic.

    ``keypoints`` is the one field that is all-or-nothing per document
    rather than per annotation: an ``(N, K, 3)`` array cannot say
    "absent on row i". A GT document that carries keypoints on some
    annotations and not others has to use the JSON route.

    Non-finite values in ``bbox``, ``area`` or ``keypoints`` are
    rejected. JSON has no ``NaN`` or ``Infinity`` literal, so a GT
    *file* cannot carry one either; refusing them is not a divergence
    from the file route but a refusal of input it could never produce.
    """

    id: NDArray[np.int64]
    image_id: NDArray[np.int64]
    category_id: NDArray[np.int64]
    bbox: NDArray[np.float64]
    area: NDArray[np.float64]
    iscrowd: NDArray[np.bool_] | NDArray[np.uint8] | NDArray[np.int64]
    ignore: NDArray[np.bool_] | NDArray[np.uint8] | NDArray[np.int64]
    segmentation: Sequence[SegmentationInput | None]
    keypoints: NDArray[np.float64]
    num_keypoints: NDArray[np.int64]


#: ``(N, 7)`` C-contiguous float64 detection matrix, laid out as
#: ``image_id, x, y, w, h, score, category_id``. The ``image_id`` and
#: ``category_id`` columns must hold exact integers within 2^53; a
#: fractional or oversized value is rejected rather than truncated.
DetectionMatrix: TypeAlias = NDArray[np.float64]

#: Union of legal forms for ``StreamingEvaluator.update`` /
#: ``BackgroundEvaluator.submit`` and for the ``dt=`` argument of every
#: ``Evaluator.evaluate`` / ``evaluate_*_grid`` entry point.
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
    "GtAnnotations",
    "GtCategory",
    "GtImages",
    "JsonRLE",
    "PolygonSegmentation",
    "RLEInput",
    "ResultAnnotation",
    "SegmentationInput",
    "UncompressedRLE",
]
