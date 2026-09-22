"""Per-sample ingest route: framework-shaped records to vernier's inputs.

ADR-0063. A training loop holds predictions and ground truth as one
record per image, in whatever array type its framework uses. vernier
takes columnar ground truth (ADR-0060) and one of three detection
routes (ADR-0030, ADR-0057). This module is the conversion between
them, and it is the only place that conversion should be written.

The rules it encodes are the reason it exists. Each descends from a
pycocotools quirk vernier already dispositions, and most are silent
when violated rather than loud:

* every image gets an ``images`` entry, including one with no
  annotations — the evaluation counts images, not annotations;
* annotation ids start at **1**; COCOeval's results are wrong from 0;
* ``area`` falls back **per element**, to the mask's area under
  ``segm`` and the box's under ``bbox``, matching what COCOeval
  derives (quirk **J3**) — a framework that never recorded an area
  stores zeros, so the fallback is load-bearing, not a corner case;
* ``iscrowd`` is widened to ``int64``: vernier reads any non-zero as a
  crowd, so a ``uint8`` column wraps 256 to 0 (quirks **D1**, **E1**);
* image sizes resolve as :func:`vernier.adapters.with_mask_image_sizes`
  resolves them — the image's own first mask, else the size its
  detections carry, else the ``0x0`` nothing reads;
* an empty per-image box array may arrive ``(1, 0)`` rather than
  ``(0, 4)`` (TorchMetrics' ``_fix_empty_tensors`` shapes it that way to
  avoid a DDP all-reduce hang), which breaks both concatenation and
  per-image counting unless it is normalized first;
* the fastest detection route differs by IoU type — the ``(N, 7)``
  matrix carries bbox state with no Python object per detection, while
  ``segm`` needs the columnar route, the only one that carries a mask
  *and* keeps the arrays whole.

**No framework is imported or named.** Arrays are read through DLPack
or :func:`numpy.asarray`, and a device tensor is moved by duck-typed
``.detach()`` / ``.cpu()``. ``tests/python/test_no_framework_imports.py``
is the gate.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from vernier import _core
from vernier._array_types import (
    Detections,
    DetectionsInput,
    GtCategory,
    Prediction,
    RLEInput,
    Target,
)
from vernier._types import ParityMode

#: Layout ``boxes`` is read as. ``xywh`` is COCO-native and vernier-native.
BoxFormat = Literal["xywh", "xyxy", "cxcywh"]

#: How a ground truth's ``area`` column is filled.
AreaPolicy = Literal["auto", "supplied", "box", "mask"]

#: IoU types this route builds inputs for.
SampleIouType = Literal["bbox", "segm"]


def _as_array(value: Any, field: str) -> NDArray[Any]:
    """Return ``value`` as a numpy array without naming its framework.

    ``.detach()`` drops an autograd graph and ``.cpu()`` moves a device
    tensor; both are probed by attribute, so torch, jax and anything
    else offering them work, and numpy (offering neither) falls straight
    through. The final :func:`numpy.asarray` is zero-copy for any CPU
    buffer exporting the array or DLPack protocol.
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    try:
        array: NDArray[Any] = np.asarray(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError(f"{field}: cannot read as an array ({exc})") from exc
    return array


def _cast_f64(array: NDArray[Any], field: str, *, cast_inputs: bool) -> NDArray[np.float64]:
    """Convert ``array`` to ``dtype``, or refuse when casting is off.

    ADR-0004 pins ``f64`` at the array boundary and ADR-0030 refuses
    ``f32`` rather than promoting it silently, because a silent
    promotion resurfaces as parity drift. That reasoning holds where the
    caller chose the dtype. It does not hold here: a model emits
    ``float32``, so refusing it would make every caller write the
    conversion this route exists to delete. ``cast_inputs=False``
    restores the strict boundary for a caller who wants it.
    """
    if array.dtype != np.float64 and not cast_inputs:
        raise TypeError(
            f"{field}: expected dtype float64, got {array.dtype}. "
            "Pass cast_inputs=True to convert, or convert it yourself."
        )
    return np.asarray(array, dtype=np.float64)


def _boxes(
    value: Any, box_format: BoxFormat, field: str, *, cast_inputs: bool
) -> NDArray[np.float64]:
    """Return one image's boxes as ``(N, 4)`` float64 in COCO ``xywh``.

    ``reshape(-1, 4)`` is a no-op on an already-``(N, 4)`` array and
    turns the ``(1, 0)`` empty TorchMetrics produces back into
    ``(0, 4)``, which is the shape concatenation and per-image counting
    both need.
    """
    array = _as_array(value, field)
    # An empty image may arrive `(1, 0)` (TorchMetrics' `_fix_empty_tensors`
    # shapes it that way to dodge a DDP all-reduce hang) or `(0, 4)`; both
    # mean zero boxes. Anything else must be `(N, 4)` on the nose — checking
    # only `size % 4` would accept a transposed `(4, N)` array, whose rows
    # then interleave coordinates from different boxes with no diagnostic.
    if array.size:
        if array.ndim != 2 or array.shape[1] != 4:
            raise ValueError(
                f"{field}: expected an (N, 4) array, got shape {array.shape}. "
                "A transposed (4, N) layout is not accepted — pass boxes row-major."
            )
    elif array.ndim > 2:
        raise ValueError(f"{field}: expected an (N, 4) array, got shape {array.shape}")
    casted = _cast_f64(array, field, cast_inputs=cast_inputs)
    boxes = np.reshape(casted, (-1, 4))
    if box_format == "xywh":
        return np.ascontiguousarray(boxes)
    converted = np.empty_like(boxes)
    if box_format == "xyxy":
        converted[:, 0:2] = boxes[:, 0:2]
        converted[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]
    elif box_format == "cxcywh":
        converted[:, 2:4] = boxes[:, 2:4]
        converted[:, 0:2] = boxes[:, 0:2] - boxes[:, 2:4] / 2.0
    else:
        # Never fall through to a default layout. ADR-0063 refuses to
        # auto-detect `box_format` because a wrong guess yields plausible,
        # wrong AP rather than an error; silently treating an unrecognised
        # spelling as one of the three would reintroduce exactly that.
        raise ValueError(
            f"{field}: unknown box_format {box_format!r}; expected 'xywh', 'xyxy' or 'cxcywh'"
        )
    return converted


def _record_boxes(
    record: Prediction | Target,
    field: str,
    labels: NDArray[np.int64],
    box_format: BoxFormat,
    *,
    masked: bool,
    cast_inputs: bool,
) -> NDArray[np.float64]:
    """One record's boxes, which a mask-only pipeline need not carry.

    Under ``segm`` the box column is required by the ground-truth schema
    (ADR-0060) but **no kernel reads it**: matching is on mask IoU, the
    area bucketing reads the ``area`` column, and detection areas come
    from the mask (``dt_area="mask"``). Verified by construction — true,
    zeroed and deliberately wrong boxes all produce identical ``segm``
    metrics. So an instance-segmentation pipeline that never materializes
    boxes, which TorchMetrics also permits under ``iou_type="segm"``, can
    omit them and gets a zero column it cannot observe.

    This is the same move :func:`vernier.adapters.with_mask_image_sizes`
    already makes for an unsized image (ADR-0055): fill where nothing
    looks, rather than demand a value the caller does not have.

    Under ``bbox`` the column is load-bearing and stays required — which
    includes each pass of a two-IoU-type run, since the bbox pass builds
    its own inputs.
    """
    if "boxes" not in record and masked:
        return np.zeros((len(labels), 4), dtype=np.float64)
    return _boxes(
        _required(record, "boxes", field), box_format, f"{field}.boxes", cast_inputs=cast_inputs
    )


def _labels(value: Any, field: str) -> NDArray[np.int64]:
    """Return one image's class labels as int64, refusing fractional ones.

    A float label array is checked before the cast rather than
    truncated: ``2.7`` silently becoming class 2 is a wrong evaluation
    with no diagnostic.
    """
    array = _as_array(value, field)
    if array.dtype.kind == "f":
        if array.size and not np.array_equal(array, np.floor(array)):
            raise ValueError(f"{field}: class labels must be integral, got fractional values")
    elif array.dtype.kind not in "iub":
        raise TypeError(f"{field}: expected an integer array, got {array.dtype}")
    labels: NDArray[np.int64] = np.asarray(array, dtype=np.int64)
    return np.reshape(labels, (-1,))


def _rles(record: Prediction | Target, field: str, count: int) -> list[RLEInput]:
    """Return one image's masks in a shape vernier's RLE ingest accepts.

    Takes either ``rles`` (already encoded) or ``masks`` (bitmasks).
    A ``(size, counts)`` pair is normalized to the dict form: that pair
    is how a TorchMetrics metric state carries a mask, and it is not one
    of the shapes :data:`RLEInput` names.
    """
    if "rles" in record:
        items = list(record["rles"])
    elif "masks" in record:
        masks = _as_array(record["masks"], f"{field}.masks")
        items = list(masks) if masks.ndim == 3 else ([masks] if masks.size else [])
    else:
        raise KeyError(f"{field}: iou_type='segm' needs either 'rles' or 'masks'")

    normalized: list[RLEInput] = []
    for item in items:
        pair = cast("tuple[Sequence[int], Any] | None", item if isinstance(item, tuple) else None)
        if pair is not None and len(pair) == 2:
            # A TorchMetrics metric state carries a mask as this pair,
            # which is not one of the shapes `RLEInput` names.
            normalized.append(
                cast("RLEInput", {"size": (int(pair[0][0]), int(pair[0][1])), "counts": pair[1]})
            )
        else:
            normalized.append(cast("RLEInput", item))
    if len(normalized) != count:
        raise ValueError(
            f"{field}: {len(normalized)} masks for {count} labels; every annotation needs one"
        )
    return normalized


def _image_ids(predictions: Sequence[Prediction], targets: Sequence[Target]) -> NDArray[np.int64]:
    """Resolve each record pair's image id, defaulting to its position.

    A trainer that never assigns ids gets positional ones, which is what
    ``0..M-1`` means everywhere else in this route. When both sides
    carry an id they must agree — a mismatch means the two sequences are
    not aligned, which no downstream check would catch.
    """
    ids: list[int] = []
    for i, (prediction, target) in enumerate(zip(predictions, targets, strict=True)):
        target_id = target.get("image_id")
        prediction_id = prediction.get("image_id")
        if target_id is not None and prediction_id is not None and target_id != prediction_id:
            raise ValueError(
                f"samples[{i}]: predictions and targets disagree on image_id "
                f"({prediction_id} vs {target_id}); the two sequences must be aligned per image"
            )
        resolved = target_id if target_id is not None else prediction_id
        ids.append(i if resolved is None else int(resolved))
    if len(set(ids)) != len(ids):
        raise ValueError("image_id must be unique across samples")
    return np.asarray(ids, dtype=np.int64)


def gt_image_sizes(
    gt_rles: Sequence[Sequence[RLEInput] | None],
    dt_rles: Sequence[Sequence[RLEInput] | None] | None = None,
    supplied: Sequence[tuple[int, int] | None] | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """The ``height`` / ``width`` columns a columnar ground truth needs.

    The columnar mirror of :func:`vernier.adapters.detection_image_sizes`.
    ADR-0055 published the dict-shaped resolver and explicitly left this
    case to the caller — "the `cocoDt`-less array-grid caller still
    builds the mapping itself, from the first detection mask's ``size``
    per image" — which made it the one rule every columnar consumer
    reimplements.

    Resolution order per image: a caller-supplied size wins, else the
    image's own first mask, else the size its detections carry, else
    ``0x0`` — the last of which nothing reads, because no annotation
    points at that image. The mask-derived part matches
    :func:`vernier.adapters.with_mask_image_sizes`.

    Args:
        gt_rles: One entry per image — that image's ground-truth masks,
            or ``None`` / empty when it has none.
        dt_rles: The same for the detection side, when known.
        supplied: One entry per image — the size the caller pinned for
            it (``Target.size``), or ``None``. Takes precedence over
            both mask sources.

    Returns:
        ``(height, width)``, each an ``(M,)`` int64 array aligned with
        ``gt_rles``.
    """
    heights: list[int] = []
    widths: list[int] = []
    for i, gt in enumerate(gt_rles):
        size = supplied[i] if supplied is not None and i < len(supplied) else None
        if size is None:
            size = _first_size(gt)
        if size is None and dt_rles is not None and i < len(dt_rles):
            size = _first_size(dt_rles[i])
        height, width = (0, 0) if size is None else size
        heights.append(height)
        widths.append(width)
    return np.asarray(heights, dtype=np.int64), np.asarray(widths, dtype=np.int64)


def _first_size(rles: Sequence[RLEInput] | None) -> tuple[int, int] | None:
    """``(height, width)`` of the first mask, or ``None`` when there is none."""
    if not rles:
        return None
    first = rles[0]
    if isinstance(first, dict):
        size = cast("Mapping[str, Any]", first).get("size")
        if size is None:
            return None
        return int(size[0]), int(size[1])
    array = np.asarray(first)
    if array.ndim != 2:
        return None
    return int(array.shape[0]), int(array.shape[1])


def _with_ids(entries: Sequence[GtCategory]) -> list[tuple[int, GtCategory]]:
    """Pair each category with its id, which :func:`_categories` guarantees."""
    return [(int(entry.get("id", 0)), entry) for entry in entries]


def _categories(
    categories: Sequence[int] | Sequence[GtCategory] | None,
    label_columns: Sequence[NDArray[np.int64]],
) -> list[GtCategory]:
    """Resolve the ``categories`` section.

    ``None`` takes the union of every label seen on **both** sides: a
    class that appears only in predictions is still a category, and
    dropping it would silently stop scoring its false positives.
    """
    if categories is not None:
        resolved = list(categories)
        entries: list[GtCategory] = []
        for position, value in enumerate(resolved):
            if isinstance(value, dict):
                if "id" not in value:
                    raise KeyError(f"categories[{position}] has no 'id'")
                entries.append(dict(value))  # type: ignore[arg-type]
            else:
                entries.append({"id": int(value), "name": str(int(value))})
    else:
        seen = np.unique(np.concatenate(label_columns)) if label_columns else np.empty(0, np.int64)
        entries = [{"id": int(value), "name": str(int(value))} for value in seen]
    # Sorted by id, always. The evaluation's category axis is sorted
    # (`vernier-core/src/evaluate.rs`), so a caller-supplied order would
    # leave `classes` and the per-class vectors silently transposed
    # relative to each other.
    return [entry for _, entry in sorted(_with_ids(entries), key=lambda pair: pair[0])]


def _required(record: Prediction | Target, key: str, field: str) -> Any:
    """One required field off a record, or a ``KeyError`` naming it.

    Defaulting a missing field to an empty array would turn a misspelled
    key into an annotation-less image or a detection-free prediction —
    an evaluation that runs to completion and reports a plausible,
    meaningless number. An image genuinely without annotations says so
    with an explicit empty array (ADR-0057: refuse, never repair).
    """
    if key not in record:
        raise KeyError(
            f"{field}.{key} is required; pass an explicit empty array for an image with none"
        )
    return record[key]  # type: ignore[literal-required]


def _column(
    record: Prediction | Target, key: str, field: str, *, cast_inputs: bool
) -> NDArray[Any]:
    """One required float64 column off a record."""
    array = _as_array(_required(record, key, field), f"{field}.{key}")
    column = _cast_f64(array, f"{field}.{key}", cast_inputs=cast_inputs)
    return np.reshape(column, (-1,))


def coco_inputs(
    predictions: Sequence[Prediction],
    targets: Sequence[Target],
    *,
    iou_type: SampleIouType = "bbox",
    box_format: BoxFormat = "xywh",
    categories: Sequence[int] | Sequence[GtCategory] | None = None,
    area: AreaPolicy = "auto",
    cast_inputs: bool = True,
) -> tuple[_core.CocoDataset, DetectionsInput]:
    """Build vernier's evaluation inputs from per-image records (ADR-0063).

    ``predictions[i]`` and ``targets[i]`` describe the same image; the
    two sequences must be the same length and aligned.

    The pair is returned rather than a metric, because the same inputs
    drive every vernier surface — :class:`vernier.instance.Evaluator`,
    TIDE, LRP, result tables, calibration, custom grids (ADR-0040) and
    the partitioned/DDP path. :func:`coco_metrics` is the convenience
    wrapper for the AP case.

    Three of this function's decisions need both sides, which is why it
    is one call and not two: a class seen only in predictions must still
    become a category; image sizes fall back from the ground truth's
    masks to the detections'; and the detection route is chosen per IoU
    type.

    Args:
        predictions: One record per image. See :class:`Prediction`.
        targets: One record per image. See :class:`Target`.
        iou_type: ``"bbox"`` or ``"segm"``. Under ``"segm"`` every
            record needs masks; ``boxes`` become optional there, since
            nothing reads them (see :func:`_record_boxes`).
        box_format: Layout ``boxes`` is read as. Never auto-detected: a
            ``(N, 4)`` array is ambiguous between ``xywh`` and ``xyxy``,
            and a wrong guess yields plausible, wrong AP rather than an
            error.
        categories: The COCO ``categories`` section, as ids or as
            entries. ``None`` takes the union of every label on both
            sides.
        area: How the ground truth's ``area`` column is filled.
            ``"auto"`` reads a supplied positive area and otherwise
            falls back per element — to the mask's area under ``segm``,
            the box's under ``bbox``.
        cast_inputs: Convert array dtypes rather than refusing them.
            On by default here, unlike the rest of vernier's array
            surface; see :func:`_cast`.

    Returns:
        The parsed ground truth and the detections, ready for any
        ``evaluate_*`` entry point.

    Raises:
        ValueError: If the sequences differ in length or are misaligned,
            if a record's columns disagree in length, or if a label is
            fractional.
        KeyError: If a required field is absent.
        TypeError: If a value cannot be read as an array, or its dtype
            is wrong under ``cast_inputs=False``.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions and targets must describe the same images: "
            f"got {len(predictions)} and {len(targets)}"
        )
    # Validated rather than pattern-matched with a fall-through default: an
    # unrecognised spelling must not quietly select one of the behaviours.
    if iou_type not in ("bbox", "segm"):
        raise ValueError(f"unknown iou_type {iou_type!r}; expected 'bbox' or 'segm'")
    if area not in ("auto", "supplied", "box", "mask"):
        raise ValueError(f"unknown area {area!r}; expected 'auto', 'supplied', 'box' or 'mask'")
    if box_format not in ("xywh", "xyxy", "cxcywh"):
        raise ValueError(f"unknown box_format {box_format!r}; expected 'xywh', 'xyxy' or 'cxcywh'")
    masked = iou_type == "segm"
    image_ids = _image_ids(predictions, targets)

    gt_boxes: list[NDArray[np.float64]] = []
    gt_labels: list[NDArray[np.int64]] = []
    gt_crowds: list[NDArray[np.int64]] = []
    gt_supplied: list[NDArray[np.float64]] = []
    gt_rles: list[list[RLEInput]] = []
    for i, target in enumerate(targets):
        field = f"targets[{i}]"
        # Labels first: they define the annotation count, which is what an
        # omitted box column is sized against.
        labels = _labels(_required(target, "labels", field), f"{field}.labels")
        boxes = _record_boxes(
            target, field, labels, box_format, masked=masked, cast_inputs=cast_inputs
        )
        if len(boxes) != len(labels):
            raise ValueError(f"{field}: {len(boxes)} boxes for {len(labels)} labels")
        crowds = (
            _as_array(target["iscrowd"], f"{field}.iscrowd")
            if "iscrowd" in target
            else np.zeros(len(labels))
        )
        supplied = (
            _column(target, "area", field, cast_inputs=cast_inputs)
            if "area" in target
            else np.zeros(len(labels))
        )
        for name, column in (("iscrowd", crowds), ("area", supplied)):
            if len(column) != len(labels):
                raise ValueError(f"{field}.{name}: {len(column)} entries for {len(labels)} labels")
        gt_boxes.append(boxes)
        gt_labels.append(labels)
        # int64, never uint8: vernier reads any non-zero as a crowd, so a
        # uint8 column wraps 256 to 0 and un-crowds that annotation.
        crowd_column: NDArray[np.int64] = np.asarray(crowds, dtype=np.int64)
        gt_crowds.append(np.reshape(crowd_column, (-1,)))
        gt_supplied.append(supplied)
        gt_rles.append(_rles(target, field, len(labels)) if masked else [])

    dt_boxes: list[NDArray[np.float64]] = []
    dt_labels: list[NDArray[np.int64]] = []
    dt_scores: list[NDArray[np.float64]] = []
    dt_rles: list[list[RLEInput]] = []
    for i, prediction in enumerate(predictions):
        field = f"predictions[{i}]"
        labels = _labels(_required(prediction, "labels", field), f"{field}.labels")
        boxes = _record_boxes(
            prediction, field, labels, box_format, masked=masked, cast_inputs=cast_inputs
        )
        scores = _column(prediction, "scores", field, cast_inputs=cast_inputs)
        if not (len(boxes) == len(labels) == len(scores)):
            raise ValueError(
                f"{field}: boxes/labels/scores disagree "
                f"({len(boxes)}, {len(labels)}, {len(scores)})"
            )
        dt_boxes.append(boxes)
        dt_labels.append(labels)
        dt_scores.append(scores)
        dt_rles.append(_rles(prediction, field, len(labels)) if masked else [])

    counts = np.asarray([len(labels) for labels in gt_labels], dtype=np.int64)
    total = int(counts.sum())
    all_boxes = np.concatenate(gt_boxes) if total else np.zeros((0, 4), dtype=np.float64)
    all_labels = np.concatenate(gt_labels) if total else np.zeros((0,), dtype=np.int64)

    heights, widths = (
        gt_image_sizes(gt_rles, dt_rles, [target.get("size") for target in targets])
        if masked
        else (np.zeros(len(targets), np.int64), np.zeros(len(targets), np.int64))
    )

    images = {"id": image_ids, "height": heights, "width": widths}
    annotations: dict[str, Any] = {
        # From 1, never 0: COCOeval's results are wrong for a zero id.
        "id": np.arange(1, total + 1, dtype=np.int64),
        "image_id": np.repeat(image_ids, counts),
        "category_id": all_labels,
        "bbox": np.ascontiguousarray(all_boxes),
        "area": _gt_area(all_boxes, gt_supplied, gt_rles, area=area, masked=masked, total=total),
        "iscrowd": np.concatenate(gt_crowds) if total else np.zeros((0,), np.int64),
    }
    if masked:
        annotations["segmentation"] = [rle for image in gt_rles for rle in image]

    dataset = _core.CocoDataset.from_arrays(
        images,  # type: ignore[arg-type]
        annotations,  # type: ignore[arg-type]
        _categories(categories, [all_labels, *(labels for labels in dt_labels if len(labels))]),
    )
    return dataset, _detections(image_ids, dt_boxes, dt_scores, dt_labels, dt_rles, masked=masked)


def _gt_area(
    all_boxes: NDArray[np.float64],
    supplied_columns: Sequence[NDArray[np.float64]],
    gt_rles: Sequence[Sequence[RLEInput]],
    *,
    area: AreaPolicy,
    masked: bool,
    total: int,
) -> NDArray[np.float64]:
    """Fill the ground truth's ``area`` column.

    ADR-0060 makes this required and read verbatim, because it is what
    the small / medium / large bucketing reads — vernier will not derive
    it, since "silently substituting ``w * h`` would re-bucket every
    polygon GT". So the choice is made here, explicitly.

    ``"auto"`` mirrors COCOeval: a positive supplied area wins, and
    anything else falls back **per element**. The per-element part
    matters — a framework that records areas for some annotations and
    zeros for the rest is the common case, not a corner one.
    """
    supplied = np.concatenate(supplied_columns) if total else np.zeros((0,), np.float64)
    if area == "supplied":
        return supplied
    if area in ("box", "mask") and (area == "mask") != masked:
        raise ValueError(
            f"area={area!r} is not available under iou_type={'segm' if masked else 'bbox'!r}"
        )
    # `auto` with every area already positive never reads the fallback, and
    # deriving it is the single most expensive step of a masked ingest — so
    # do not derive it. `np.where` would evaluate both arms regardless.
    if area == "auto" and bool(np.all(supplied > 0)):
        return supplied
    computed = _mask_areas(gt_rles) if masked else all_boxes[:, 2] * all_boxes[:, 3]
    if area in ("box", "mask"):
        return computed
    return np.where(supplied > 0, supplied, computed)


def _mask_areas(gt_rles: Sequence[Sequence[RLEInput]]) -> NDArray[np.float64]:
    """Foreground pixel count per ground-truth mask.

    A bitmask is summed in NumPy and a pre-encoded RLE goes to the FFI.
    Both give the same number — bit-identical, since a foreground count
    is exact in f64 — but the round trip through the RLE codec costs
    ~19x more than the sum, and it re-rasterizes masks that
    ``CocoDataset.from_arrays`` rasterizes again a moment later.
    Splitting by form keeps the codec for the payloads that need
    decoding and leaves the common training-loop case to NumPy.
    """
    areas = np.empty(sum(len(image) for image in gt_rles), dtype=np.float64)
    encoded: list[RLEInput] = []
    encoded_at: list[int] = []
    position = 0
    for i, image in enumerate(gt_rles):
        for item in image:
            if isinstance(item, dict):
                encoded.append(item)
                encoded_at.append(position)
            else:
                bitmask = np.asarray(item)
                if bitmask.ndim != 2:
                    raise ValueError(
                        f"targets[{i}]: expected a 2-D bitmask, got shape {bitmask.shape}"
                    )
                areas[position] = float(np.count_nonzero(bitmask))
            position += 1
    if encoded:
        try:
            decoded = _core.rle_area(encoded)
        except (TypeError, ValueError) as exc:
            # `rle_area` indexes the compacted list it was handed, which is
            # not the caller's numbering. Re-raise against the image the
            # mask belongs to, matching the field paths the rest of the
            # module builds.
            raise type(exc)(f"{_attribute(exc, gt_rles, encoded_at)}") from exc
        areas[encoded_at] = np.asarray(decoded, dtype=np.float64)
    return areas


def _attribute(
    exc: Exception, gt_rles: Sequence[Sequence[RLEInput]], encoded_at: Sequence[int]
) -> str:
    """Rewrite an ``rles[j]`` message to name the image the mask is on."""
    message = str(exc)
    match = re.match(r"rles\[(\d+)\]:?\s*(.*)", message, re.DOTALL)
    if match is None:
        return message
    flat = encoded_at[int(match.group(1))]
    consumed = 0
    for image, masks in enumerate(gt_rles):
        if flat < consumed + len(masks):
            return f"targets[{image}].masks[{flat - consumed}]: {match.group(2)}"
        consumed += len(masks)
    return message


def _detections(
    image_ids: NDArray[np.int64],
    boxes: Sequence[NDArray[np.float64]],
    scores: Sequence[NDArray[np.float64]],
    labels: Sequence[NDArray[np.int64]],
    rles: Sequence[Sequence[RLEInput]],
    *,
    masked: bool,
) -> DetectionsInput:
    """Pick the fastest detection route that can express these detections.

    ``bbox`` takes the ``(N, 7)`` matrix — the only route that hands the
    whole state over with no Python object per detection. What it cannot
    carry is exactly what a bbox pass does not use: a segmentation, an
    explicit id, a supplied area.

    ``segm`` takes the columnar route, the only one that carries a mask
    *and* keeps the arrays whole. The third route, a list of COCO result
    dicts, costs more per-detection Python than the rest of the ingest
    put together.
    """
    if masked:
        return [
            Detections(
                image_id=int(image_id),
                boxes=np.ascontiguousarray(image_boxes),
                scores=image_scores,
                labels=image_labels,
                rles=image_rles,
            )
            for image_id, image_boxes, image_scores, image_labels, image_rles in zip(
                image_ids, boxes, scores, labels, rles, strict=True
            )
        ]
    counts = np.asarray([len(image_scores) for image_scores in scores], dtype=np.int64)
    if not int(counts.sum()):
        return np.zeros((0, 7), dtype=np.float64)
    # One fresh C-contiguous float64 buffer laid out as
    # image_id, x, y, w, h, score, category_id — the layout the matrix
    # route requires of the caller rather than copying silently.
    # `column_stack` promotes the int64 id columns against the float64
    # boxes and scores, so the result is already float64 and C-contiguous;
    # a further `astype` would memcpy N*56 bytes for nothing.
    matrix = np.column_stack(
        [
            np.repeat(image_ids, counts),
            np.concatenate(list(boxes)),
            np.concatenate(list(scores)),
            np.concatenate(list(labels)),
        ]
    )
    # int64 id columns against float64 boxes and scores promote to float64;
    # pyright only sees the union of the inputs' dtypes.
    return cast("NDArray[np.float64]", matrix)


#: The twelve COCO summary statistics, in the canonical order
#: :attr:`vernier.instance.Summary.stats` reports them
#: (``docs/reference/coco-summary-stats.md``). The names are the
#: conventional ones the wider ecosystem logs, not vernier's own
#: (``AP``, ``AP@.50``, ``AR_1``); the borrow is confined to this one
#: mapping so the output is directly loggable.
_STAT_KEYS: tuple[str, ...] = (
    "map",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
    "mar_{0}",
    "mar_{1}",
    "mar_{2}",
    "mar_small",
    "mar_medium",
    "mar_large",
)


def coco_metrics(
    predictions: Sequence[Prediction],
    targets: Sequence[Target],
    *,
    iou_type: SampleIouType | Sequence[SampleIouType] = "bbox",
    box_format: BoxFormat = "xywh",
    categories: Sequence[int] | Sequence[GtCategory] | None = None,
    area: AreaPolicy = "auto",
    class_metrics: bool = False,
    max_dets: Sequence[int] = (1, 10, 100),
    iou_thresholds: Sequence[float] | None = None,
    recall_thresholds: Sequence[float] | None = None,
    parity_mode: ParityMode = "corrected",
    num_threads: int | None = None,
    cast_inputs: bool = True,
) -> dict[str, float | NDArray[Any]]:
    """Evaluate per-image records and return the COCO metrics (ADR-0063).

    The convenience wrapper over :func:`coco_inputs` for the AP case.
    Anything else — TIDE, LRP, result tables, calibration, a custom grid
    — takes :func:`coco_inputs`' pair directly.

    Values are Python floats and numpy arrays. Converting them to a
    framework tensor is the caller's one line (``torch.as_tensor(...)``,
    zero-copy off numpy); vernier returns no framework type and imports
    no framework.

    ``parity_mode`` defaults to ``"corrected"``, matching
    :class:`vernier.instance.Evaluator`. Under ``"strict"``, ``map``
    reports pycocotools' ``-1`` whenever 100 is not among ``max_dets``,
    which is a genuine pycocotools behaviour and rarely what a training
    run wants to log.

    Args:
        predictions: One record per image. See :class:`Prediction`.
        targets: One record per image. See :class:`Target`.
        iou_type: One IoU type, or several. With more than one, each
            metric key is prefixed (``bbox_map``, ``segm_map``).
        box_format: Layout ``boxes`` is read as.
        categories: See :func:`coco_inputs`.
        area: See :func:`coco_inputs`.
        class_metrics: Also return per-class AP and AR vectors, aligned
            with the resolved category ids. Off by default: each vector
            materializes a fresh array across the FFI.
        max_dets: The COCO detection-count ladder. Three entries, since
            the summary reports ``mar`` at each.
        iou_thresholds: IoU grid, defaulting to COCO's ``.50:.05:.95``.
        recall_thresholds: Recall grid, defaulting to COCO's 101 points.
        parity_mode: ``"strict"`` or ``"corrected"`` (ADR-0002).
        num_threads: Thread budget. ``None`` evaluates sequentially
            (ADR-0047), which is the DDP-safe default — a rank sharing a
            node should not claim every core.
        cast_inputs: See :func:`coco_inputs`.

    Returns:
        The twelve COCO statistics under the conventional metric names,
        plus ``classes`` and, under ``class_metrics``, the per-class
        vectors.

    Raises:
        ValueError: If ``max_dets`` does not hold exactly three entries,
            or for any reason :func:`coco_inputs` raises.
    """
    if len(max_dets) != 3:
        raise ValueError(f"max_dets takes exactly three entries, got {len(max_dets)}")
    iou_types: tuple[SampleIouType, ...] = (
        (iou_type,) if isinstance(iou_type, str) else tuple(iou_type)
    )
    ladder = [int(value) for value in max_dets]
    # Resolved once: `_categories` is order-defining (it sorts by id), and
    # recomputing it per IoU type would re-read every label column.
    resolved = _categories(categories, _label_columns(predictions, targets))
    results: dict[str, float | NDArray[Any]] = {}
    for kind in iou_types:
        prefix = "" if len(iou_types) == 1 else f"{kind}_"
        dataset, detections = coco_inputs(
            predictions,
            targets,
            iou_type=kind,
            box_format=box_format,
            categories=resolved,
            area=area,
            cast_inputs=cast_inputs,
        )
        shared: dict[str, Any] = {
            "parity_mode": parity_mode,
            "max_dets_per_image": ladder[-1],
            "use_cats": True,
            "iou_thresholds": None if iou_thresholds is None else list(iou_thresholds),
            "recall_thresholds": None if recall_thresholds is None else list(recall_thresholds),
            "num_threads": num_threads,
        }
        # `dt_area` names a different Literal on each grid, and a detection's
        # area is derived rather than supplied (quirk J3): from the box under
        # bbox, from the mask under segm.
        grid = (
            _core.evaluate_bbox_grid(dataset, detections, dt_area="bbox", **shared)
            if kind == "bbox"
            else _core.evaluate_segm_grid(dataset, detections, dt_area="mask", **shared)
        )
        accumulated = grid.accumulate(ladder)
        stats = accumulated.summarize(ladder).stats
        for key, value in zip(_STAT_KEYS, stats, strict=True):
            results[prefix + key.format(*ladder)] = float(value)
        if class_metrics:
            results[f"{prefix}map_per_class"] = _per_class(accumulated.precision)
            results[f"{prefix}mar_{ladder[-1]}_per_class"] = _per_class(accumulated.recall)
    # Same list, same order as the per-class vectors' category axis.
    results["classes"] = np.asarray(
        [category_id for category_id, _ in _with_ids(resolved)], dtype=np.int64
    )
    return results


def _label_columns(
    predictions: Sequence[Prediction], targets: Sequence[Target]
) -> list[NDArray[np.int64]]:
    """Every label column on both sides, for resolving ``categories``."""
    columns: list[NDArray[np.int64]] = []
    for i, record in enumerate(targets):
        if "labels" in record:
            columns.append(_labels(record["labels"], f"targets[{i}].labels"))
    for i, record in enumerate(predictions):
        if "labels" in record:
            columns.append(_labels(record["labels"], f"predictions[{i}].labels"))
    return [column for column in columns if len(column)]


def _per_class(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """Reduce a precision or recall tensor to one value per category.

    Mirrors what vernier reports in its own per-class table
    (``crates/vernier-core/src/tables.rs``): the mean over the IoU axis
    — and, for precision, the recall axis — at the **all-areas** bucket
    and the **largest** maxDets cap. Averaging over the area and maxDets
    axes instead would fold AP_small/medium/large, and AR@1/AR@10, into
    a number matching neither ``map`` nor the ``ap`` column of
    ``EvalResult.per_class`` for the same run.

    Precision is ``(T, R, K, A, M)`` and recall is ``(T, K, A, M)``, so
    after selecting ``A`` and ``M`` the category axis is last in both.
    ``-1`` is the empty-cell sentinel (quirk **C5**); the mean covers the
    defined entries only, so a category present in no image reports
    ``-1`` rather than dragging the average down.
    """
    # Area index 0 is the all-areas bucket; the summary reads the last
    # maxDets cap. Selecting both leaves (T, R, K) or (T, K).
    selected = tensor[..., 0, -1]
    values = np.reshape(np.moveaxis(selected, -1, 0), (selected.shape[-1], -1))
    defined = values > -1
    counts = defined.sum(axis=1)
    totals = np.where(defined, values, 0.0).sum(axis=1)
    return np.where(counts > 0, totals / np.maximum(counts, 1), -1.0)
