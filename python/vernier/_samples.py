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
* ``area`` falls back **per element**, to the mask's area whenever the
  annotation carries a mask and to the box's when it does not, which is
  what COCOeval buckets by (quirk **J3**) — a framework that never
  recorded an area stores zeros, so the fallback is load-bearing;
* ``iscrowd`` is widened to ``int64``: vernier reads any non-zero as a
  crowd, so a ``uint8`` column wraps 256 to 0 (quirks **D1**, **E1**);
* image sizes resolve as :func:`vernier.adapters.with_mask_image_sizes`
  resolves them — the image's own first mask, else the size its
  detections carry, else the ``0x0`` nothing reads;
* an empty per-image box array may arrive ``(1, 0)`` rather than
  ``(0, 4)`` (TorchMetrics' ``_fix_empty_tensors`` shapes it that way to
  avoid a DDP all-reduce hang), which breaks both concatenation and
  per-image counting unless it is normalized first;
* the fastest detection route depends on whether masks are present —
  the ``(N, 7)`` matrix carries box state with no Python object per
  detection, while a mask needs the columnar route, the only one that
  carries one *and* keeps the arrays whole.

**No framework is imported or named.** Arrays are read through DLPack
or :func:`numpy.asarray`, and a device tensor is moved by duck-typed
``.detach()`` / ``.cpu()``. ``tests/python/test_no_framework_imports.py``
is the gate.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from vernier import _core
from vernier._array_types import (
    DetectionColumns,
    Detections,
    DetectionsInput,
    GtCategory,
    Prediction,
    RLEInput,
    Target,
    TargetColumns,
)
from vernier._types import ParityMode

#: Layout ``boxes`` is read as. ``xywh`` is COCO-native and vernier-native.
BoxFormat = Literal["xywh", "xyxy", "cxcywh"]

#: How a ground truth's ``area`` column is filled.
AreaPolicy = Literal["auto", "supplied", "box", "mask"]

#: IoU types this route builds inputs for.
SampleIouType = Literal["bbox", "segm"]

#: Any of the four input shapes, for the helpers that only read a key.
_Columnish: TypeAlias = "Prediction | Target | DetectionColumns | TargetColumns"


def _as_array(value: Any, field: str, *, cast_inputs: bool = True) -> NDArray[Any]:
    """Return ``value`` as a numpy array without naming its framework.

    ``.detach()`` drops an autograd graph and ``.cpu()`` moves a device
    tensor; both are probed by attribute, so torch, jax and anything
    else offering them work, and numpy (offering neither) falls straight
    through. The final :func:`numpy.asarray` is zero-copy for any CPU
    buffer exporting the array or DLPack protocol; only a dtype it
    refuses costs anything more.
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    try:
        array: NDArray[Any] = np.asarray(value)
    except (TypeError, ValueError) as exc:
        array = _as_f64_unrepresentable(value, field, exc, cast_inputs=cast_inputs)
    return array


def _as_f64_unrepresentable(
    value: Any, field: str, exc: Exception, *, cast_inputs: bool
) -> NDArray[Any]:
    """Retry a dtype numpy has no equivalent for, through the value's own f64 conversion.

    An autocast training loop holds ``bfloat16``, which numpy cannot
    represent, so :func:`numpy.asarray` refuses the tensor outright and
    the caller sees a dtype complaint naming a library vernier never
    mentions. ``.double()`` is probed by attribute like ``.detach()``
    above, and is exact -- ``bfloat16`` is ``float32`` with a truncated
    mantissa, so this invents no precision the value did not have.

    Converting is what ``cast_inputs`` gates (ADR-0004, ADR-0030), so
    ``cast_inputs=False`` refuses here rather than in
    :func:`_cast_f64`, which would never see this value.
    """
    if not cast_inputs:
        raise TypeError(
            f"{field}: expected dtype float64, got a dtype numpy cannot represent ({exc}). "
            "Pass cast_inputs=True to convert, or convert it yourself."
        ) from exc
    if hasattr(value, "double"):
        try:
            return np.asarray(value.double())
        except (TypeError, ValueError):
            pass
    raise TypeError(f"{field}: cannot read as an array ({exc})") from exc


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
    array = _as_array(value, field, cast_inputs=cast_inputs)
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

    A bbox grid over that zero column reports 0, so
    :func:`coco_metrics` refuses ``iou_type="bbox"`` when a record omits
    its boxes, and a caller driving a grid directly owns the same check.
    vernier will not derive the box from the mask: ADR-0057 refuses to
    repair an input the caller did not supply.
    """
    if "boxes" not in record and _has_masks(record):
        return np.zeros((len(labels), 4), dtype=np.float64)
    return _boxes(
        _required(record, "boxes", field), box_format, f"{field}.boxes", cast_inputs=cast_inputs
    )


def _integers(value: Any, field: str) -> NDArray[np.int64]:
    """Return an integer column as ``(N,)`` int64, refusing fractional values.

    A float column is checked before the cast rather than truncated. For
    ``labels``, ``2.7`` silently becoming class 2 is a wrong evaluation
    with no diagnostic; for ``iscrowd``, ``0.5`` truncates to 0 and
    un-crowds the annotation — the same silent outcome the int64
    widening exists to prevent.
    """
    array = _as_array(value, field)
    if array.dtype.kind == "f":
        if array.size and not np.array_equal(array, np.floor(array)):
            raise ValueError(f"{field}: must be integral, got fractional values")
    elif array.dtype.kind not in "iub":
        raise TypeError(f"{field}: expected an integer array, got {array.dtype}")
    integers: NDArray[np.int64] = np.asarray(array, dtype=np.int64)
    return np.reshape(integers, (-1,))


def _has_masks(record: _Columnish) -> bool:
    """Whether a record carries any mask, in either accepted spelling.

    A present-but-empty column counts as *no* masks. A caller assembling
    columns generically sets the key unconditionally, and reading that as
    "zero masks for N annotations" would reject a perfectly good
    bbox-only run.
    """
    entry = cast("Mapping[str, Any]", record)
    for key in ("rles", "masks"):
        value = entry.get(key)
        if value is not None and len(value):
            return True
    return False


def _rles(record: _Columnish, field: str, count: int) -> list[RLEInput]:
    """Return one image's masks in a shape vernier's RLE ingest accepts.

    Takes either ``rles`` (already encoded) or ``masks`` (bitmasks).
    A ``(size, counts)`` pair is normalized to the dict form: that pair
    is how a TorchMetrics metric state carries a mask, and it is not one
    of the shapes :data:`RLEInput` names.
    """
    # The first *non-empty* column wins, which is how `_has_masks` reads
    # them. Keying on presence instead would let a caller that sets both
    # keys unconditionally — `rles: []` beside a real `masks` array — be
    # read as "zero masks for N annotations" and refused.
    entry = cast("Mapping[str, Any]", record)
    rles, masks = entry.get("rles"), entry.get("masks")
    items: list[Any]
    if rles is not None and len(rles):
        items = list(rles)
    elif masks is not None and len(masks):
        array = _as_array(masks, f"{field}.masks")
        items = list(array) if array.ndim == 3 else ([array] if array.size else [])
    elif rles is not None or masks is not None:
        items = []
    else:
        raise KeyError(f"{field}: a mask column must be spelled 'rles' or 'masks'")

    # Hot: this runs once per mask, and at validation scale that is hundreds
    # of thousands of iterations. The common case is a list that is already in
    # an accepted shape, so it is recognised in one scan and handed over
    # untouched -- no per-item work, and one `cast` for the list rather than
    # one per element (`typing.cast` is a real call at runtime).
    normalized: list[RLEInput]
    if not any(isinstance(item, tuple) for item in items):
        normalized = cast("list[RLEInput]", items)
    else:
        # A TorchMetrics metric state carries a mask as a `(size, counts)`
        # pair, which is not one of the shapes `RLEInput` names.
        normalized = []
        for item in items:
            pair = cast(
                "tuple[Sequence[int], Any] | None", item if isinstance(item, tuple) else None
            )
            if pair is None:
                normalized.append(cast("RLEInput", item))
            else:
                size, counts = pair
                normalized.append(
                    cast("RLEInput", {"size": (int(size[0]), int(size[1])), "counts": counts})
                )
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
    :func:`vernier.adapters.with_mask_image_sizes`. Both optional
    sequences must cover every image; a short one is refused rather than
    read as "no size for the rest".

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
    return _sizes_from_firsts(
        [image[0] if image else None for image in gt_rles],
        None if dt_rles is None else [image[0] if image else None for image in dt_rles],
        supplied,
    )


def _sizes_from_firsts(
    gt_first: Sequence[RLEInput | None],
    dt_first: Sequence[RLEInput | None] | None,
    supplied: Sequence[tuple[int, int] | None] | None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """The resolution order itself, over one candidate mask per image.

    Only the *first* mask of an image is ever read, so both callers
    reduce to this — :func:`gt_image_sizes` from per-image sequences and
    :func:`_flat_image_sizes` through per-image bounds. One copy of the
    rule, because a fix to it has to land everywhere at once.

    A sequence that does not cover every image is refused rather than
    read as "no size for the rest" (ADR-0057: refuse, never repair).
    """
    count = len(gt_first)
    for name, sequence in (("sizes", supplied), ("detection masks", dt_first)):
        if sequence is not None and len(sequence) != count:
            raise ValueError(f"{name}: {len(sequence)} entries for {count} images")
    heights: NDArray[np.int64] = np.zeros(count, dtype=np.int64)
    widths: NDArray[np.int64] = np.zeros(count, dtype=np.int64)
    for i in range(count):
        size = None if supplied is None else supplied[i]
        first = gt_first[i]
        if size is None and first is not None:
            size = _size_of(first)
        if size is None and dt_first is not None:
            candidate = dt_first[i]
            if candidate is not None:
                size = _size_of(candidate)
        if size is not None:
            heights[i], widths[i] = size
    return heights, widths


def _size_of(first: RLEInput) -> tuple[int, int] | None:
    """``(height, width)`` a single mask declares, or ``None`` if it declares none."""
    if isinstance(first, dict):
        size = cast("Mapping[str, Any]", first).get("size")
        if size is None:
            return None
        return int(size[0]), int(size[1])
    array = _as_array(first, "rles")
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


def _required(record: _Columnish, key: str, field: str) -> Any:
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


def _column(record: _Columnish, key: str, field: str, *, cast_inputs: bool) -> NDArray[Any]:
    """One required float64 column off a record."""
    array = _as_array(_required(record, key, field), f"{field}.{key}", cast_inputs=cast_inputs)
    column = _cast_f64(array, f"{field}.{key}", cast_inputs=cast_inputs)
    return np.reshape(column, (-1,))


def coco_inputs(
    predictions: Sequence[Prediction],
    targets: Sequence[Target],
    *,
    box_format: BoxFormat = "xywh",
    categories: Sequence[int] | Sequence[GtCategory] | None = None,
    area: AreaPolicy = "auto",
    cast_inputs: bool = True,
) -> tuple[_core.CocoDataset, DetectionsInput]:
    """Build vernier's evaluation inputs from per-image records (ADR-0063).

    ``predictions[i]`` and ``targets[i]`` describe the same image; the
    two sequences must be the same length and aligned.

    **There is no ``iou_type``.** The masks decide: records that carry
    them produce inputs a ``segm`` grid can read, and records that do
    not produce bbox-only inputs. Naming the IoU type here as well as at
    the grid would be two places to say one thing and a chance for them
    to disagree — and it bought nothing measurable, since attaching
    segmentation costs ~2% of the conversion and the detection route it
    would select is within 0.3% of the alternative on a real grid. One
    set of inputs therefore serves both passes of a two-IoU-type run.

    The pair is returned rather than a metric, because it is what every
    surface that takes a parsed ground truth reads:
    :class:`vernier.instance.Evaluator`, the ``evaluate_*_grid`` and
    ``evaluate_*_summary`` entry points, custom grids (ADR-0040),
    calibration through ``cells_from_grid``, and the partitioned/DDP
    path. :func:`coco_metrics` is the convenience wrapper for the AP
    case.

    TIDE, LRP, the confusion matrix and the ``tables=`` path do **not**
    take it yet — each refuses a :class:`CocoDataset` handle and asks
    for GT JSON bytes, a limitation that predates this route and is
    theirs to lift. When they lift it these inputs work unchanged, which
    is the point of returning them rather than a metric.

    Two of this function's decisions need both sides, which is why it is
    one call and not two: a class seen only in predictions must still
    become a category, and image sizes fall back from the ground truth's
    masks to the detections'.

    Args:
        predictions: One record per image. See :class:`Prediction`.
        targets: One record per image. See :class:`Target`.
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
    dataset, detections, _ = _records_to_inputs(
        predictions,
        targets,
        box_format=box_format,
        categories=categories,
        area=area,
        cast_inputs=cast_inputs,
    )
    return dataset, detections


def _records_to_inputs(
    predictions: Sequence[Prediction],
    targets: Sequence[Target],
    *,
    box_format: BoxFormat,
    categories: Sequence[int] | Sequence[GtCategory] | None,
    area: AreaPolicy,
    cast_inputs: bool,
) -> tuple[_core.CocoDataset, DetectionsInput, list[GtCategory]]:
    """:func:`coco_inputs`, also returning the resolved ``categories``.

    :func:`coco_metrics` needs that order for its ``classes`` key and
    for the per-class vectors' axis. Taking it from the resolution the
    build already performs is what keeps the metric wrapper from reading
    every label column a second time.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions and targets must describe the same images: "
            f"got {len(predictions)} and {len(targets)}"
        )
    _validate_literals(area=area, box_format=box_format)
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
        labels = _integers(_required(target, "labels", field), f"{field}.labels")
        boxes = _record_boxes(target, field, labels, box_format, cast_inputs=cast_inputs)
        if len(boxes) != len(labels):
            raise ValueError(f"{field}: {len(boxes)} boxes for {len(labels)} labels")
        crowds = (
            _integers(target["iscrowd"], f"{field}.iscrowd")
            if "iscrowd" in target
            else np.zeros(len(labels), np.int64)
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
        gt_crowds.append(crowds)
        gt_supplied.append(supplied)
        gt_rles.append(_rles(target, field, len(labels)) if _has_masks(target) else [])

    dt_boxes: list[NDArray[np.float64]] = []
    dt_labels: list[NDArray[np.int64]] = []
    dt_scores: list[NDArray[np.float64]] = []
    dt_rles: list[list[RLEInput]] = []
    for i, prediction in enumerate(predictions):
        field = f"predictions[{i}]"
        labels = _integers(_required(prediction, "labels", field), f"{field}.labels")
        boxes = _record_boxes(prediction, field, labels, box_format, cast_inputs=cast_inputs)
        scores = _column(prediction, "scores", field, cast_inputs=cast_inputs)
        if not (len(boxes) == len(labels) == len(scores)):
            raise ValueError(
                f"{field}: boxes/labels/scores disagree "
                f"({len(boxes)}, {len(labels)}, {len(scores)})"
            )
        dt_boxes.append(boxes)
        dt_labels.append(labels)
        dt_scores.append(scores)
        dt_rles.append(_rles(prediction, field, len(labels)) if _has_masks(prediction) else [])

    flat_gt_rles = [rle for image in gt_rles for rle in image]
    flat_dt_rles = [rle for image in dt_rles for rle in image]
    gt_counts = np.asarray([len(labels) for labels in gt_labels], dtype=np.int64)
    dt_counts = np.asarray([len(labels) for labels in dt_labels], dtype=np.int64)
    return _build(
        gt_boxes=_join(gt_boxes, (0, 4), np.float64),
        gt_labels=_join(gt_labels, (0,), np.int64),
        gt_crowds=_join(gt_crowds, (0,), np.int64),
        gt_area=_join(gt_supplied, (0,), np.float64),
        gt_counts=gt_counts,
        gt_rles=flat_gt_rles,
        dt_boxes=_join(dt_boxes, (0, 4), np.float64),
        dt_scores=_join(dt_scores, (0,), np.float64),
        dt_labels=_join(dt_labels, (0,), np.int64),
        dt_counts=dt_counts,
        dt_rles=flat_dt_rles,
        image_ids=image_ids,
        sizes=_declared_sizes(targets),
        categories=categories,
        area=area,
    )


def _declared_sizes(targets: Sequence[Target]) -> list[tuple[int, int] | None] | None:
    """``Target.size`` per image, or ``None`` when no record pinned one."""
    declared = [target.get("size") for target in targets]
    return declared if any(size is not None for size in declared) else None


def coco_inputs_from_columns(
    detections: DetectionColumns,
    targets: TargetColumns,
    *,
    box_format: BoxFormat = "xywh",
    categories: Sequence[int] | Sequence[GtCategory] | None = None,
    area: AreaPolicy = "auto",
    image_ids: Any | None = None,
    cast_inputs: bool = True,
) -> tuple[_core.CocoDataset, DetectionsInput]:
    """Build vernier's evaluation inputs from whole columns (ADR-0063).

    The columnar spelling of :func:`coco_inputs`. Same rules, same
    result -- both land on one builder, and
    ``test_columnar_and_per_sample_agree`` pins the two to the identical
    ``dataset_hash``. What differs is only how the caller already holds
    its state.

    Prefer this when the state is *already* concatenated, which is the
    case for a TorchMetrics-shaped metric: splitting it into per-image
    records for :func:`coco_inputs` only to have vernier concatenate it
    again costs a Python-level pass per image per field, and at
    validation scale that is the most expensive thing on the path. A
    caller with genuine per-image records should use :func:`coco_inputs`
    and not synthesize columns.

    ``counts`` carries the image structure -- entry ``i`` is image
    ``i``'s row count -- so the columns themselves need no image
    grouping. Ground truth and detections have independent counts, both
    of length ``M``.

    Args:
        detections: Every detection, as columns. See
            :class:`DetectionColumns`.
        targets: Every ground-truth annotation, as columns. See
            :class:`TargetColumns`.
        box_format: Layout ``boxes`` is read as; never auto-detected.
        categories: The COCO ``categories`` section, as ids or entries.
            ``None`` takes the union of every label on both sides.
        area: How the ground truth's ``area`` column is filled.
        image_ids: ``(M,)`` image ids, defaulting to ``0..M-1``.
        cast_inputs: Convert array dtypes rather than refusing them.

    Returns:
        The parsed ground truth and the detections, ready for any
        ``evaluate_*`` entry point.

    Raises:
        ValueError: If a ``counts`` column does not sum to its columns'
            length, if the two ``counts`` differ in length, or for any
            reason :func:`coco_inputs` raises.
        KeyError: If a required column is absent.
        TypeError: If a value cannot be read as an array, or its dtype is
            wrong under ``cast_inputs=False``.
    """
    _validate_literals(area=area, box_format=box_format)
    gt_counts = _counts(targets, "targets")
    dt_counts = _counts(detections, "detections")
    if len(gt_counts) != len(dt_counts):
        raise ValueError(
            f"targets.counts and detections.counts must describe the same images: "
            f"got {len(gt_counts)} and {len(dt_counts)}"
        )
    n_images = len(gt_counts)
    ids = (
        np.arange(n_images, dtype=np.int64)
        if image_ids is None
        else np.reshape(np.asarray(_as_array(image_ids, "image_ids"), dtype=np.int64), (-1,))
    )
    if len(ids) != n_images:
        raise ValueError(f"image_ids: {len(ids)} entries for {n_images} images")
    # Duplicates would point two images' annotations at one id, which the
    # per-sample spelling already refuses in `_image_ids`; the two routes
    # must accept exactly the same inputs.
    if len(np.unique(ids)) != n_images:
        raise ValueError("image_ids must be unique across images")

    gt_labels = _integers(_required(targets, "labels", "targets"), "targets.labels")
    gt_boxes = _columns_boxes(targets, "targets", gt_labels, box_format, cast_inputs=cast_inputs)
    dt_labels = _integers(_required(detections, "labels", "detections"), "detections.labels")
    dt_boxes = _columns_boxes(
        detections, "detections", dt_labels, box_format, cast_inputs=cast_inputs
    )
    dt_scores = _column(detections, "scores", "detections", cast_inputs=cast_inputs)

    for name, counts, rows in (
        ("targets", gt_counts, len(gt_labels)),
        ("detections", dt_counts, len(dt_labels)),
    ):
        if int(counts.sum()) != rows:
            raise ValueError(f"{name}.counts sums to {int(counts.sum())} but there are {rows} rows")
    if len(dt_scores) != len(dt_labels):
        raise ValueError(f"detections: labels/scores disagree ({len(dt_labels)}, {len(dt_scores)})")

    crowds = (
        _integers(targets["iscrowd"], "targets.iscrowd")
        if "iscrowd" in targets
        else np.zeros(len(gt_labels), np.int64)
    )
    supplied = (
        _column(targets, "area", "targets", cast_inputs=cast_inputs)
        if "area" in targets
        else np.zeros(len(gt_labels), np.float64)
    )
    for name, column in (("iscrowd", crowds), ("area", supplied)):
        if len(column) != len(gt_labels):
            raise ValueError(f"targets.{name}: {len(column)} entries for {len(gt_labels)} labels")

    sizes: Sequence[tuple[int, int] | None] | None = None
    if "sizes" in targets:
        declared = np.reshape(
            np.asarray(_as_array(targets["sizes"], "targets.sizes"), np.int64), (-1, 2)
        )
        if len(declared) != n_images:
            raise ValueError(f"targets.sizes: {len(declared)} entries for {n_images} images")
        sizes = [(int(h), int(w)) for h, w in declared]
    dataset, detections_input, _ = _build(
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        gt_crowds=crowds,
        gt_area=supplied,
        gt_counts=gt_counts,
        gt_rles=_flat_rles(targets, "targets", len(gt_labels)),
        dt_boxes=dt_boxes,
        dt_scores=dt_scores,
        dt_labels=dt_labels,
        dt_counts=dt_counts,
        dt_rles=_flat_rles(detections, "detections", len(dt_labels)),
        image_ids=ids,
        sizes=sizes,
        categories=categories,
        area=area,
    )
    return dataset, detections_input


def _counts(columns: DetectionColumns | TargetColumns, field: str) -> NDArray[np.int64]:
    """The per-image row counts, which is what assigns rows to images."""
    raw = _as_array(_required(columns, "counts", field), f"{field}.counts")
    return np.reshape(np.asarray(raw, dtype=np.int64), (-1,))


def _columns_boxes(
    columns: DetectionColumns | TargetColumns,
    field: str,
    labels: NDArray[np.int64],
    box_format: BoxFormat,
    *,
    cast_inputs: bool,
) -> NDArray[np.float64]:
    """The box column, zero-filled when absent (see :func:`_record_boxes`)."""
    if "boxes" not in columns:
        return np.zeros((len(labels), 4), dtype=np.float64)
    boxes = _boxes(columns["boxes"], box_format, f"{field}.boxes", cast_inputs=cast_inputs)
    if len(boxes) != len(labels):
        raise ValueError(f"{field}: {len(boxes)} boxes for {len(labels)} labels")
    return boxes


def _flat_rles(columns: DetectionColumns | TargetColumns, field: str, count: int) -> list[RLEInput]:
    """The flat mask column, or empty when the caller carries none."""
    if not _has_masks(columns):
        return []
    return _rles(columns, field, count)


def _validate_literals(*, area: str, box_format: str, iou_types: Sequence[str] = ()) -> None:
    """Refuse an unrecognised option rather than falling through to a default.

    Every one of these selects a behaviour that is silently wrong if
    mis-selected, so a typo must raise instead of picking an arm — an
    unrecognised ``iou_type`` would otherwise reach the ``else`` arm and
    report one IoU type's metrics under another's keys.
    """
    if area not in ("auto", "supplied", "box", "mask"):
        raise ValueError(f"unknown area {area!r}; expected 'auto', 'supplied', 'box' or 'mask'")
    if box_format not in ("xywh", "xyxy", "cxcywh"):
        raise ValueError(f"unknown box_format {box_format!r}; expected 'xywh', 'xyxy' or 'cxcywh'")
    for kind in iou_types:
        if kind not in ("bbox", "segm"):
            raise ValueError(f"unknown iou_type {kind!r}; expected 'bbox' or 'segm'")


def _join(
    columns: Sequence[NDArray[Any]], empty_shape: tuple[int, ...], dtype: type[Any]
) -> NDArray[Any]:
    """Concatenate per-image columns, with a correctly-shaped empty for no rows."""
    for column in columns:
        if len(column):
            return np.concatenate(columns)
    return np.zeros(empty_shape, dtype=dtype)


def _build(
    *,
    gt_boxes: NDArray[np.float64],
    gt_labels: NDArray[np.int64],
    gt_crowds: NDArray[np.int64],
    gt_area: NDArray[np.float64],
    gt_counts: NDArray[np.int64],
    gt_rles: Sequence[RLEInput],
    dt_boxes: NDArray[np.float64],
    dt_scores: NDArray[np.float64],
    dt_labels: NDArray[np.int64],
    dt_counts: NDArray[np.int64],
    dt_rles: Sequence[RLEInput],
    image_ids: NDArray[np.int64],
    sizes: Sequence[tuple[int, int] | None] | None,
    categories: Sequence[int] | Sequence[GtCategory] | None,
    area: AreaPolicy,
) -> tuple[_core.CocoDataset, DetectionsInput, list[GtCategory]]:
    """Assemble vernier's inputs from whole columns.

    The single builder both public spellings land on, so a per-sample
    call and a columnar one cannot drift: everything above this point is
    a way of producing these arrays, and everything the evaluator sees
    is decided here.
    """
    # Masks decide, not a parameter: carrying them means the inputs support
    # `segm`, and the caller picks the grid. See `coco_inputs`.
    masked = len(gt_rles) > 0 or len(dt_rles) > 0
    total = int(gt_counts.sum())
    # All-or-nothing, per side. A mask column covering only some of a side's
    # rows is indexed through `counts` as if it covered all of them: image
    # `i` then reads another image's mask, and the area column broadcasts
    # rather than refusing. Checked here because this is the one point both
    # spellings pass through.
    for name, rles, rows in (
        ("targets", gt_rles, total),
        ("detections", dt_rles, int(dt_counts.sum())),
    ):
        if len(rles) not in (0, rows):
            raise ValueError(
                f"{name}: {len(rles)} masks for {rows} rows; a mask column is "
                "all-or-nothing — every row carries one, or none does"
            )
    heights, widths = (
        _flat_image_sizes(gt_rles, gt_counts, dt_rles, dt_counts, sizes)
        if masked or sizes is not None
        else (np.zeros(len(gt_counts), np.int64), np.zeros(len(gt_counts), np.int64))
    )
    annotations: dict[str, Any] = {
        # From 1, never 0: COCOeval's results are wrong for a zero id.
        "id": np.arange(1, total + 1, dtype=np.int64),
        "image_id": np.repeat(image_ids, gt_counts),
        "category_id": gt_labels,
        "bbox": np.ascontiguousarray(gt_boxes),
        "area": _gt_area(gt_boxes, gt_area, gt_rles, area=area),
        "iscrowd": gt_crowds,
    }
    if len(gt_rles):
        annotations["segmentation"] = list(gt_rles)
    resolved = _categories(categories, [gt_labels, dt_labels])
    dataset = _core.CocoDataset.from_arrays(
        {"id": image_ids, "height": heights, "width": widths},
        annotations,  # type: ignore[arg-type]
        resolved,
    )
    detections = _detections(
        image_ids, dt_boxes, dt_scores, dt_labels, dt_counts, dt_rles, masked=masked
    )
    return dataset, detections, resolved


def _offsets(counts: NDArray[np.int64]) -> NDArray[np.int64]:
    """Row bounds per image: ``offsets[i]:offsets[i + 1]`` is image ``i``."""
    bounds: NDArray[np.int64] = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=bounds[1:])
    return bounds


def _flat_image_sizes(
    gt_rles: Sequence[RLEInput],
    gt_counts: NDArray[np.int64],
    dt_rles: Sequence[RLEInput],
    dt_counts: NDArray[np.int64],
    supplied: Sequence[tuple[int, int] | None] | None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """:func:`gt_image_sizes`, reading flat mask lists through per-image bounds."""
    return _sizes_from_firsts(_firsts(gt_rles, gt_counts), _firsts(dt_rles, dt_counts), supplied)


def _firsts(rles: Sequence[RLEInput], counts: NDArray[np.int64]) -> list[RLEInput | None]:
    """Each image's first mask, the only one a size is ever read from.

    An empty flat list means the side carries no masks at all — a bbox
    ground truth beside masked detections, say — so every image reports
    ``None`` rather than indexing past the end.
    """
    if not len(rles):
        return [None] * len(counts)
    at = _offsets(counts)
    return [rles[start] if count else None for start, count in zip(at[:-1], counts, strict=True)]


def _gt_area(
    boxes: NDArray[np.float64],
    supplied: NDArray[np.float64],
    gt_rles: Sequence[RLEInput],
    *,
    area: AreaPolicy,
) -> NDArray[np.float64]:
    """Fill the ground truth's ``area`` column.

    ADR-0060 makes this required and read verbatim, because it is what
    the small / medium / large bucketing reads — vernier will not derive
    it, since "silently substituting ``w * h`` would re-bucket every
    polygon GT". So the choice is made here, explicitly.

    ``"auto"`` mirrors COCO: a positive supplied area wins, and anything
    else falls back **per element** to the *mask's* area when the record
    carries a mask, and to the box's only when it does not. The
    per-element part matters — a framework that records areas for some
    annotations and zeros for the rest is the common case, not a corner
    one.

    The mask takes precedence **regardless of ``iou_type``**, because a
    COCO ground truth has one ``area`` per annotation and it is the
    segmentation's: ``COCOeval`` under ``iouType="bbox"`` buckets by
    that same field rather than recomputing ``w * h``. Deriving the box
    area for a bbox pass would silently disagree with every
    pycocotools-shaped evaluator for any object whose two areas straddle
    ``32**2`` or ``96**2`` — visible only as AP moving between the
    small and medium buckets, with no error anywhere.
    """
    if area == "supplied":
        return supplied
    has_masks = len(gt_rles) > 0
    if area == "mask" and not has_masks:
        raise ValueError("area='mask' needs masks on the ground-truth records")
    # `auto` with every area already positive never reads the fallback, and
    # deriving it is the single most expensive step of a masked ingest — so
    # do not derive it. `np.where` would evaluate both arms regardless.
    if area == "auto" and bool(np.all(supplied > 0)):
        return supplied
    from_mask = has_masks and area != "box"
    computed = _mask_areas(gt_rles) if from_mask else boxes[:, 2] * boxes[:, 3]
    if area in ("box", "mask"):
        return computed
    return np.where(supplied > 0, supplied, computed)


def _mask_areas(gt_rles: Sequence[RLEInput]) -> NDArray[np.float64]:
    """Foreground pixel count per ground-truth mask.

    A bitmask is summed in NumPy and a pre-encoded RLE goes to the FFI.
    Both give the same number — bit-identical, since a foreground count
    is exact in f64 — but the round trip through the RLE codec costs
    ~19x more than the sum, and it re-rasterizes masks that
    ``CocoDataset.from_arrays`` rasterizes again a moment later.

    The FFI hands back a buffer rather than a list of floats, so the
    all-encoded case — every mask pre-encoded, which is what a
    TorchMetrics-shaped state holds — is a single memcpy with no
    per-mask Python object anywhere on the path.
    """
    areas: NDArray[np.float64] = np.empty(len(gt_rles), dtype=np.float64)
    encoded: list[RLEInput] = []
    encoded_at: list[int] = []
    for position, item in enumerate(gt_rles):
        if isinstance(item, dict):
            encoded.append(item)
            encoded_at.append(position)
        else:
            bitmask = _as_array(item, f"masks[{position}]")
            if bitmask.ndim != 2:
                raise ValueError(
                    f"masks[{position}]: expected a 2-D bitmask, got shape {bitmask.shape}"
                )
            areas[position] = float(np.count_nonzero(bitmask))
    if encoded:
        try:
            decoded = _core.rle_area(encoded)
        except (TypeError, ValueError) as exc:
            # `rle_area` indexes the compacted list it was handed, which is
            # not the caller's numbering.
            raise type(exc)(_attribute(exc, encoded_at)) from exc
        values = np.frombuffer(decoded, dtype=np.float64)
        # Every mask encoded is the common case; assigning through a
        # scatter index would convert `encoded_at` to an array first.
        if len(encoded) == len(gt_rles):
            areas[:] = values
        else:
            areas[encoded_at] = values
    return areas


def _attribute(exc: Exception, encoded_at: Sequence[int]) -> str:
    """Rewrite an ``rles[j]`` message to the caller's own mask numbering."""
    message = str(exc)
    # The extractor appends its own field to the root (`rles[0].counts: ...`),
    # so consuming the separator here would splice the prefix back together
    # as `masks[0]: .counts: ...`.
    match = re.match(r"rles\[(\d+)\]", message)
    if match is None:
        return message
    return f"masks[{encoded_at[int(match.group(1))]}]{message[match.end() :]}"


def _detections(
    image_ids: NDArray[np.int64],
    boxes: NDArray[np.float64],
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    counts: NDArray[np.int64],
    rles: Sequence[RLEInput],
    *,
    masked: bool,
) -> DetectionsInput:
    """Pick the fastest detection route that can express these detections.

    ``bbox`` takes the ``(N, 7)`` matrix -- the only route that hands the
    whole state over with no Python object per detection. What it cannot
    carry is exactly what a bbox pass does not use: a segmentation, an
    explicit id, a supplied area.

    ``segm`` takes the columnar route, the only one that carries a mask
    *and* keeps the arrays whole; its per-image entries are **views**
    into the columns, not copies. The third route, a list of COCO result
    dicts, costs more per-detection Python than the rest of the ingest
    put together.
    """
    if not masked:
        if not len(scores):
            return np.zeros((0, 7), dtype=np.float64)
        # `column_stack` promotes the int64 id columns against the float64
        # boxes and scores, so the result is already float64 and
        # C-contiguous; a further `astype` would memcpy N*56 bytes for nothing.
        matrix = np.column_stack([np.repeat(image_ids, counts), boxes, scores, labels])
        return cast("NDArray[np.float64]", matrix)
    at = _offsets(counts)
    return [
        Detections(
            image_id=int(image_id),
            boxes=np.ascontiguousarray(boxes[start:stop]),
            scores=scores[start:stop],
            labels=labels[start:stop],
            rles=rles[start:stop],
        )
        for image_id, start, stop in zip(image_ids, at[:-1], at[1:], strict=True)
    ]


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
    Anything else — a custom grid, calibration, the partitioned path —
    takes :func:`coco_inputs`' pair directly.

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
            metric key is prefixed (``bbox_map``, ``segm_map``). This is
            the one place the IoU type is named, because it is the one
            that calls a kernel; :func:`coco_inputs` builds inputs that
            serve whichever grid the masks allow.
        box_format: Layout ``boxes`` is read as.
        categories: See :func:`coco_inputs`.
        area: See :func:`coco_inputs`.
        class_metrics: Also return per-class AP and AR vectors, aligned
            with the resolved category ids. Off by default: each vector
            materializes a fresh array across the FFI.
        max_dets: The COCO detection-count ladder — three increasing
            positive caps, since the summary reports ``mar`` at each and
            caps every image at the last.
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
        ValueError: If ``iou_type`` or ``max_dets`` is not one of the
            accepted values, or for any reason :func:`coco_inputs`
            raises.
        KeyError: If ``iou_type`` includes ``"bbox"`` and a record omits
            its boxes.
    """
    iou_types: tuple[SampleIouType, ...] = (
        (iou_type,) if isinstance(iou_type, str) else tuple(iou_type)
    )
    _validate_literals(area=area, box_format=box_format, iou_types=iou_types)
    ladder = [int(value) for value in max_dets]
    if len(ladder) != 3 or ladder[0] < 1 or not ladder[0] < ladder[1] < ladder[2]:
        raise ValueError(
            f"max_dets must be three increasing positive caps, got {tuple(max_dets)}. "
            "Every image is capped at the last, so a descending ladder truncates the "
            "run to the smallest cap while the keys still read `mar_100`; a repeated "
            "one collapses two `mar_{n}` keys into one."
        )
    if "bbox" in iou_types:
        _require_boxes(predictions, targets)
    # Built once for every IoU type: the inputs do not depend on which grid
    # reads them, so a two-IoU-type run converts the state once. `_categories`
    # is order-defining (it sorts by id) and the build already resolves it, so
    # taking it back from there reads every label column exactly once.
    dataset, detections, resolved = _records_to_inputs(
        predictions,
        targets,
        box_format=box_format,
        categories=categories,
        area=area,
        cast_inputs=cast_inputs,
    )
    if "segm" in iou_types and not isinstance(detections, list):
        raise ValueError("iou_type='segm' needs masks on the records; none were passed")
    results: dict[str, float | NDArray[Any]] = {}
    for kind in iou_types:
        prefix = "" if len(iou_types) == 1 else f"{kind}_"
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


def _require_boxes(predictions: Sequence[Prediction], targets: Sequence[Target]) -> None:
    """Refuse a bbox evaluation over records that omit their boxes.

    :func:`_record_boxes` zero-fills the column for a mask-only
    pipeline, which a segm grid cannot observe and a bbox grid reports
    as ``0``. The check belongs here because this is the first point
    that knows which grid will read the inputs.
    """
    for name, records in (("targets", targets), ("predictions", predictions)):
        for i, record in enumerate(records):
            if "boxes" not in record:
                raise KeyError(
                    f"{name}[{i}].boxes is required for iou_type='bbox'; a record "
                    "carrying only masks gets a zero box column, which scores 0"
                )


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
