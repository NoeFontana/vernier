"""The per-sample ingest route (ADR-0063).

Two things are asserted, and they are different claims.

**Same document.** :func:`vernier.adapters.coco_inputs` must build the
*same* ground truth the COCO-JSON route builds from the same records —
not merely one that scores the same. ``CocoDataset.dataset_hash`` is the
oracle, following ``test_gt_ingest_route_equivalence.py``. A
metrics-only check would pass while the document diverged in a way AP
happens not to read today.

**Same numbers.** The metrics must equal what ``pycocotools`` produces
for the same state, compared at ``float32`` because that is the dtype
TorchMetrics reports and the only honest common precision.

The rules in ADR-0063 §"Context" are each covered by a case that fails
without them — an annotation-less image, a ``(1, 0)`` empty box array, a
crowd with a supplied area, an ``iscrowd`` value above 255, a
per-element area fallback, and a mask-only-sized image.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from vernier import _core
from vernier._array_types import CompressedRLE
from vernier.adapters import (
    DetectionColumns,
    Prediction,
    Target,
    TargetColumns,
    coco_inputs,
    coco_inputs_from_columns,
    coco_metrics,
    gt_image_sizes,
    to_coco_json,
)


def _dataset_from_json(
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    categories: list[dict[str, Any]],
) -> _core.CocoDataset:
    """The same document via the JSON route, for a `dataset_hash` comparison."""
    return _core.CocoDataset.from_json(
        to_coco_json({"images": images, "annotations": annotations, "categories": categories})
    )


def test_columnar_ground_truth_is_the_same_document() -> None:
    """`coco_inputs` and the JSON route converge on one `CocoDataset`.

    Every field the hash covers is exercised: an image with no
    annotations (image 2), a crowd, a supplied area that wins over the
    computed one, and a computed area that fills in for a zero.
    """
    targets: list[Target] = [
        {
            "boxes": np.array([[10.0, 10.0, 20.0, 30.0], [0.0, 0.0, 4.0, 5.0]]),
            "labels": np.array([1, 2]),
            "iscrowd": np.array([0, 1]),
            "area": np.array([0.0, 99.0]),
        },
        {"boxes": np.zeros((0, 4)), "labels": np.zeros((0,), dtype=np.int64)},
    ]
    predictions: list[Prediction] = [
        {
            "boxes": np.array([[10.0, 10.0, 20.0, 30.0]]),
            "scores": np.array([0.9]),
            "labels": np.array([1]),
        },
        {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, dtype=np.int64)},
    ]
    dataset, _ = coco_inputs(predictions, targets, categories=[1, 2])

    expected = _dataset_from_json(
        # Every image gets an entry, including the one with no annotations.
        [{"id": 0, "height": 0, "width": 0}, {"id": 1, "height": 0, "width": 0}],
        [
            # Annotation ids start at 1; a zero id makes COCOeval wrong.
            # area: 0.0 supplied -> falls back to 20*30; 99.0 supplied -> kept.
            {
                "id": 1,
                "image_id": 0,
                "category_id": 1,
                "bbox": [10.0, 10.0, 20.0, 30.0],
                "area": 600.0,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 0,
                "category_id": 2,
                "bbox": [0.0, 0.0, 4.0, 5.0],
                "area": 99.0,
                "iscrowd": 1,
            },
        ],
        [{"id": 1, "name": "1"}, {"id": 2, "name": "2"}],
    )
    assert dataset.dataset_hash == expected.dataset_hash


def test_empty_boxes_shaped_one_by_zero_are_normalized() -> None:
    """A ``(1, 0)`` empty box array counts as zero detections, not one.

    TorchMetrics' ``_fix_empty_tensors`` shapes an empty per-image box
    tensor that way to avoid a DDP all-reduce hang. Left alone it breaks
    concatenation, and its ``len()`` miscounts the image as holding one
    detection.
    """
    targets: list[Target] = [
        {"boxes": np.array([[1.0, 1.0, 5.0, 5.0]]), "labels": np.array([3])},
        {"boxes": np.zeros((1, 0)), "labels": np.zeros((0,), dtype=np.int64)},
    ]
    predictions: list[Prediction] = [
        {
            "boxes": np.array([[1.0, 1.0, 5.0, 5.0]]),
            "scores": np.array([0.8]),
            "labels": np.array([3]),
        },
        {"boxes": np.zeros((1, 0)), "scores": np.zeros(0), "labels": np.zeros(0, dtype=np.int64)},
    ]
    dataset, detections = coco_inputs(predictions, targets)
    assert dataset.num_annotations == 1
    assert dataset.num_images == 2
    assert np.asarray(detections).shape == (1, 7)


def test_iscrowd_above_255_stays_a_crowd() -> None:
    """``iscrowd`` is widened to int64, so 256 does not wrap to 0.

    vernier reads any non-zero value as a crowd. A ``uint8`` column
    would wrap 256 to 0 and evaluate the annotation as a normal ground
    truth — silently, and only for the annotations that matter most.

    The assertion is semantic rather than structural: 256 must score
    like 1 and unlike 0. The COCO-JSON route cannot stand in as the
    oracle here, because it refuses a non-boolean ``iscrowd`` outright
    ("expected 0 or 1 for COCO bool field") where the columnar route
    accepts it — so the document hash is not a shared reference for
    this case.
    """

    def metrics(crowd: int) -> float:
        targets: list[Target] = [
            {
                "boxes": np.array([[0.0, 0.0, 10.0, 10.0]]),
                "labels": np.array([1]),
                "iscrowd": np.array([crowd], dtype=np.int64),
                "area": np.array([100.0]),
            }
        ]
        predictions: list[Prediction] = [
            {
                "boxes": np.array([[0.0, 0.0, 4.0, 4.0]]),
                "scores": np.array([0.9]),
                "labels": np.array([1]),
            }
        ]
        return float(coco_metrics(predictions, targets, categories=[1])["map"])

    assert metrics(256) == metrics(1)
    assert metrics(256) != metrics(0)


def test_categories_include_a_prediction_only_class() -> None:
    """A class seen only in predictions is still a category.

    Dropping it would stop scoring its false positives, which changes AP
    without any error.
    """
    targets: list[Target] = [{"boxes": np.array([[0.0, 0.0, 4.0, 4.0]]), "labels": np.array([1])}]
    predictions: list[Prediction] = [
        {
            "boxes": np.array([[0.0, 0.0, 4.0, 4.0]]),
            "scores": np.array([0.5]),
            "labels": np.array([7]),
        }
    ]
    dataset, _ = coco_inputs(predictions, targets)
    assert dataset.num_categories == 2


@pytest.mark.parametrize(
    ("box_format", "boxes"),
    [
        ("xywh", [[10.0, 20.0, 30.0, 40.0]]),
        ("xyxy", [[10.0, 20.0, 40.0, 60.0]]),
        ("cxcywh", [[25.0, 40.0, 30.0, 40.0]]),
    ],
)
def test_box_formats_converge_on_xywh(box_format: str, boxes: list[list[float]]) -> None:
    """All three layouts describe the same box, so all three hash alike."""
    targets: list[Target] = [{"boxes": np.asarray(boxes), "labels": np.array([1])}]
    predictions: list[Prediction] = [
        {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, dtype=np.int64)}
    ]
    dataset, _ = coco_inputs(predictions, targets, box_format=box_format, categories=[1])  # type: ignore[arg-type]
    expected = _dataset_from_json(
        [{"id": 0, "height": 0, "width": 0}],
        [
            {
                "id": 1,
                "image_id": 0,
                "category_id": 1,
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "area": 1200.0,
                "iscrowd": 0,
            }
        ],
        [{"id": 1, "name": "1"}],
    )
    assert dataset.dataset_hash == expected.dataset_hash


def test_fractional_labels_are_refused() -> None:
    """A float label is checked, not truncated: 2.7 must not become class 2."""
    targets: list[Target] = [{"boxes": np.array([[0.0, 0.0, 4.0, 4.0]]), "labels": np.array([2.7])}]
    predictions: list[Prediction] = [
        {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, dtype=np.int64)}
    ]
    with pytest.raises(ValueError, match="integral"):
        coco_inputs(predictions, targets)


def test_misaligned_image_ids_are_refused() -> None:
    """Disagreeing ids mean the two sequences are not aligned per image."""
    targets: list[Target] = [
        {"boxes": np.zeros((0, 4)), "labels": np.zeros(0, dtype=np.int64), "image_id": 5}
    ]
    predictions: list[Prediction] = [
        {
            "boxes": np.zeros((0, 4)),
            "scores": np.zeros(0),
            "labels": np.zeros(0, dtype=np.int64),
            "image_id": 6,
        }
    ]
    with pytest.raises(ValueError, match="image_id"):
        coco_inputs(predictions, targets)


def test_mismatched_sequence_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="same images"):
        coco_inputs([], [{"boxes": np.zeros((0, 4)), "labels": np.zeros(0, dtype=np.int64)}])


def test_cast_inputs_false_refuses_float32() -> None:
    """The strict ADR-0004 boundary is restorable for a caller who wants it."""
    targets: list[Target] = [{"boxes": np.zeros((0, 4)), "labels": np.zeros(0, dtype=np.int64)}]
    predictions: list[Prediction] = [
        {
            "boxes": np.zeros((1, 4), dtype=np.float32),
            "scores": np.zeros(1, dtype=np.float32),
            "labels": np.zeros(1, dtype=np.int64),
        }
    ]
    with pytest.raises(TypeError, match="float64"):
        coco_inputs(predictions, targets, cast_inputs=False)


def test_gt_image_sizes_resolution_order() -> None:
    """Own mask, else the detections' mask, else ``0x0``."""
    gt = [[{"size": (7, 9), "counts": b"x"}], None, None]
    dt = [None, [{"size": (3, 4), "counts": b"y"}], None]
    heights, widths = gt_image_sizes(gt, dt)
    assert list(heights) == [7, 3, 0]
    assert list(widths) == [9, 4, 0]


def test_per_class_vectors_align_with_classes() -> None:
    """One entry per class, not per area range.

    ``precision`` is ``(T, R, K, A, M)`` and ``recall`` is
    ``(T, K, A, M)``: the category axis is ``-3`` in both only *after*
    the area and maxDets axes are selected. Reducing over the wrong one
    returns four values — one per area bucket — which still looks like a
    plausible vector and silently mislabels every class.
    """
    targets: list[Target] = [
        {"boxes": np.array([[0.0, 0.0, 10.0, 10.0]]), "labels": np.array([3])},
        {"boxes": np.array([[5.0, 5.0, 20.0, 20.0]]), "labels": np.array([7])},
    ]
    predictions: list[Prediction] = [
        {
            "boxes": np.array([[0.0, 0.0, 10.0, 10.0]]),
            "scores": np.array([0.9]),
            "labels": np.array([3]),
        },
        {
            "boxes": np.array([[6.0, 6.0, 20.0, 20.0]]),
            "scores": np.array([0.7]),
            "labels": np.array([7]),
        },
    ]
    result = coco_metrics(predictions, targets, class_metrics=True, categories=[3, 7])
    assert len(np.asarray(result["classes"])) == 2
    assert len(np.asarray(result["map_per_class"])) == 2
    assert len(np.asarray(result["mar_100_per_class"])) == 2


@pytest.mark.parametrize("order", [[3, 7], [7, 3]])
def test_categories_are_sorted_by_id_whatever_the_caller_passed(order: list[int]) -> None:
    """The evaluation's category axis is id-sorted, so ``classes`` must be too.

    Preserving the caller's order would leave ``classes`` and the
    per-class vectors transposed relative to each other, reporting one
    class's score under another's id with nothing to signal it.
    """
    targets: list[Target] = [
        {"boxes": np.array([[0.0, 0.0, 10.0, 10.0]]), "labels": np.array([3])},
        {"boxes": np.array([[5.0, 5.0, 20.0, 20.0]]), "labels": np.array([7])},
    ]
    predictions: list[Prediction] = [
        {
            "boxes": np.array([[0.0, 0.0, 10.0, 10.0]]),
            "scores": np.array([0.9]),
            "labels": np.array([3]),
        },
        {
            "boxes": np.array([[6.0, 6.0, 20.0, 20.0]]),
            "scores": np.array([0.7]),
            "labels": np.array([7]),
        },
    ]
    result = coco_metrics(predictions, targets, class_metrics=True, categories=order)
    assert list(np.asarray(result["classes"])) == [3, 7]
    # Category 3's detection is exact and 7's is offset, so the first
    # entry must be the larger — which pins the pairing, not just the order.
    per_class = np.asarray(result["map_per_class"])
    assert per_class[0] > per_class[1]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"box_format": "XYXY"}, "box_format"),
        ({"area": "bogus"}, "area"),
    ],
)
def test_unknown_literals_are_refused(kwargs: dict[str, Any], match: str) -> None:
    """No option falls through to a default behaviour.

    ADR-0063 refuses to auto-detect ``box_format`` because a wrong guess
    yields plausible, wrong AP rather than an error. A bare ``else``
    over the three spellings would reintroduce exactly that for a typo.
    """
    targets: list[Target] = [{"boxes": np.zeros((0, 4)), "labels": np.zeros(0, dtype=np.int64)}]
    predictions: list[Prediction] = [
        {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, dtype=np.int64)}
    ]
    with pytest.raises(ValueError, match=match):
        coco_inputs(predictions, targets, **kwargs)


@pytest.mark.parametrize(
    "record",
    [
        {"bbox": np.zeros((1, 4)), "labels": np.zeros(1, dtype=np.int64)},
        {"boxes": np.zeros((1, 4)), "label": np.zeros(1, dtype=np.int64)},
    ],
)
def test_misspelled_required_fields_raise(record: dict[str, Any]) -> None:
    """A missing field is an error, never an annotation-less image.

    Defaulting to an empty array turns a typo into an evaluation that
    runs to completion and reports a plausible, meaningless number.
    """
    predictions: list[Prediction] = [
        {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, dtype=np.int64)}
    ]
    with pytest.raises(KeyError, match="required"):
        coco_inputs(predictions, [record])  # type: ignore[list-item]


def test_transposed_boxes_are_refused() -> None:
    """A ``(4, N)`` array has a multiple-of-four size but interleaved rows.

    Checking only ``size % 4`` accepts it, and the per-record
    ``len(boxes) == len(labels)`` guard passes too, so the annotations
    come out scrambled with no diagnostic anywhere.
    """
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 20.0, 20.0]])
    targets: list[Target] = [{"boxes": np.ascontiguousarray(boxes.T), "labels": np.array([3, 7])}]
    predictions: list[Prediction] = [
        {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, dtype=np.int64)}
    ]
    with pytest.raises(ValueError, match=r"\(N, 4\)"):
        coco_inputs(predictions, targets)


def test_supplied_image_size_wins_over_the_masks() -> None:
    """``Target.size`` is documented as the size vernier checks against."""
    heights, widths = gt_image_sizes(
        [[{"size": (7, 9), "counts": b"x"}], None],
        [None, [{"size": (3, 4), "counts": b"y"}]],
        [(11, 13), None],
    )
    assert list(heights) == [11, 3]
    assert list(widths) == [13, 4]


def test_bitmask_and_encoded_areas_agree() -> None:
    """The NumPy bitmask sum and the RLE codec give the same area.

    ``_mask_areas`` splits by form because summing a bitmask is ~19x
    cheaper than a round trip through the codec. That is only legitimate
    if the two agree exactly, which a foreground count in f64 is. Both
    are checked against ``pycocotools.mask.area``, the same oracle the
    rest of the parity suite uses.

    The comparison is on the areas, not on ``dataset_hash``: the two
    forms store *different segmentation payloads*, so the documents
    differ by construction even when every area matches.
    """
    from pycocotools import mask as mask_utils

    from vernier._samples import _mask_areas

    rng = np.random.default_rng(11)
    bitmasks = [(rng.random((20, 24)) > 0.7).astype(np.uint8) for _ in range(3)]
    raw = [mask_utils.encode(np.asfortranarray(m)) for m in bitmasks]
    oracle = np.asarray([float(mask_utils.area(item)) for item in raw])
    # pycocotools spells `size` as a list; `CompressedRLE` declares the
    # tuple. Both are read the same at the boundary, but the declared
    # shape is what a caller is told to pass.
    encoded: list[CompressedRLE] = [
        {
            "size": (int(item["size"][0]), int(item["size"][1])),
            # `encode` returns bytes; the stub widens it to `str | bytes`
            # because the same type covers the JSON spelling (quirk K3).
            "counts": cast("bytes", item["counts"]),
        }
        for item in raw
    ]

    from_bitmasks = _mask_areas([np.asfortranarray(m.astype(bool)) for m in bitmasks])
    from_encoded = _mask_areas(encoded)

    assert np.array_equal(from_bitmasks, oracle)
    assert np.array_equal(from_encoded, oracle)
    # Mixed within one call: the encoded entries are scattered back into
    # position, so an off-by-one in the index bookkeeping would show here.
    mixed = _mask_areas([np.asfortranarray(bitmasks[0].astype(bool)), encoded[1], encoded[2]])
    assert np.array_equal(mixed, oracle)


def _discs() -> tuple[list[Any], list[Any]]:
    """One predicted and one ground-truth mask, overlapping but not equal."""
    rows, columns = np.ogrid[:40, :50]

    def disc(center_x: int, center_y: int, radius: int) -> Any:
        return ((rows - center_y) ** 2 + (columns - center_x) ** 2 <= radius**2).astype(np.uint8)

    return [np.stack([disc(12, 12, 7)])], [np.stack([disc(13, 12, 7)])]


def test_segm_accepts_records_without_boxes() -> None:
    """A mask-only pipeline need not materialize boxes it does not have.

    Under ``segm`` the box column is required by the ground-truth schema
    but no kernel reads it, so demanding it would reject a legitimate
    instance-segmentation pipeline — one TorchMetrics itself permits
    under ``iou_type="segm"``.
    """
    predicted, actual = _discs()
    targets: list[Target] = [{"labels": np.array([0]), "masks": masks} for masks in actual]
    predictions: list[Prediction] = [
        {"labels": np.array([0]), "scores": np.array([0.9]), "masks": masks} for masks in predicted
    ]
    result = coco_metrics(predictions, targets, iou_type="segm", parity_mode="strict")
    assert float(result["map"]) > 0


def test_segm_metrics_do_not_depend_on_the_boxes() -> None:
    """The zero-fill is inert, which is what makes omitting boxes safe.

    Pins the property the optionality rests on: under ``segm``, true,
    zeroed and deliberately wrong boxes all evaluate identically. If a
    future change starts reading the box column on this path, this
    fails rather than silently changing everyone's numbers.
    """
    predicted, actual = _discs()

    def run(boxes: Any | None) -> dict[str, Any]:
        targets: list[Target] = []
        predictions: list[Prediction] = []
        for gt_masks, dt_masks in zip(actual, predicted, strict=True):
            target: Target = {"labels": np.array([0]), "masks": gt_masks}
            prediction: Prediction = {
                "labels": np.array([0]),
                "scores": np.array([0.9]),
                "masks": dt_masks,
            }
            if boxes is not None:
                target["boxes"] = boxes
                prediction["boxes"] = boxes
            targets.append(target)
            predictions.append(prediction)
        return coco_metrics(predictions, targets, iou_type="segm", parity_mode="strict")

    omitted = run(None)
    truthful = run(np.array([[6.0, 5.0, 15.0, 15.0]]))
    nonsense = run(np.array([[999.0, 999.0, 3.0, 3.0]]))
    for key in ("map", "map_50", "map_small", "mar_100"):
        assert float(omitted[key]) == float(truthful[key]) == float(nonsense[key])


def test_boxes_are_required_without_masks() -> None:
    """A record with no mask must carry boxes: nothing else can size the annotation."""
    targets: list[Target] = [{"labels": np.array([0])}]
    predictions: list[Prediction] = [{"labels": np.array([0]), "scores": np.array([0.9])}]
    with pytest.raises(KeyError, match="boxes is required"):
        coco_inputs(predictions, targets)


def test_auto_area_reads_the_mask_not_the_box() -> None:
    """A COCO ground truth's ``area`` is the segmentation's whenever it has one.

    ``COCOeval`` buckets by the annotation's ``area`` field under either
    IoU type, and in a COCO file that field is the segmentation's area --
    it never recomputes ``w * h``. Deriving the box area would disagree
    with every pycocotools-shaped evaluator for any object whose two
    areas straddle ``32**2`` or ``96**2``, visible only as AP moving
    between the small and medium buckets.

    A disc of radius 17 straddles it: box ``34 x 34 = 1156`` (medium),
    mask ``~901`` (small).
    """
    rows, columns = np.ogrid[:128, :128]
    mask = ((rows - 64) ** 2 + (columns - 64) ** 2 <= 17**2).astype(np.uint8)
    assert 34 * 34 > 32**2  # the box is medium
    assert int(mask.sum()) < 32**2  # the mask is small

    def built(area: Any = None) -> bytes:
        target: Target = {
            "boxes": np.array([[47.0, 47.0, 34.0, 34.0]]),
            "labels": np.array([1]),
            "masks": np.asfortranarray(mask.astype(bool))[None],
        }
        if area is not None:
            target["area"] = area
        predictions: list[Prediction] = [
            {
                "boxes": np.zeros((0, 4)),
                "scores": np.zeros(0),
                "labels": np.zeros(0, dtype=np.int64),
            }
        ]
        return coco_inputs(predictions, [target], categories=[1])[0].dataset_hash

    # `auto` lands on the mask's area, which is what supplying it gives...
    assert built() == built(np.array([float(mask.sum())]))
    # ... and not on the box's.
    assert built() != built(np.array([34.0 * 34.0]))


@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
def test_columnar_and_per_sample_agree(iou_type: Any) -> None:
    """The two spellings are one conversion, so they must produce one document.

    ``coco_inputs`` and ``coco_inputs_from_columns`` differ only in how
    the caller already holds its state; both land on the same builder.
    ``dataset_hash`` proves the ground truth is identical rather than
    equivalent, and the detections are compared route-for-route.
    """
    rng = np.random.default_rng(31)
    rows, columns_ = np.ogrid[:32, :32]
    per_image = [3, 0, 2, 1]

    def masks_for(count: int) -> list[Any]:
        return [
            np.asfortranarray(
                (rows - rng.integers(8, 24)) ** 2 + (columns_ - rng.integers(8, 24)) ** 2 <= 36
            )
            for _ in range(count)
        ]

    predictions: list[Prediction] = []
    targets: list[Target] = []
    for count in per_image:
        boxes = np.column_stack(
            [
                rng.uniform(0, 20, count),
                rng.uniform(0, 20, count),
                rng.uniform(1, 9, count),
                rng.uniform(1, 9, count),
            ]
        )
        labels = rng.integers(0, 3, count)
        target: Target = {"boxes": boxes, "labels": labels, "iscrowd": rng.integers(0, 2, count)}
        prediction: Prediction = {
            "boxes": boxes + 0.5,
            "labels": labels,
            "scores": rng.uniform(0.1, 0.9, count),
        }
        if iou_type == "segm":
            target["masks"] = masks_for(count)
            prediction["masks"] = masks_for(count)
        targets.append(target)
        predictions.append(prediction)

    def joined(key: str, records: list[Any], empty: Any) -> Any:
        parts = [np.asarray(r[key]) for r in records if len(np.asarray(r[key]))]
        return np.concatenate(parts) if parts else empty

    counts = np.asarray(per_image, dtype=np.int64)
    target_columns: TargetColumns = {
        "boxes": joined("boxes", targets, np.zeros((0, 4))),
        "labels": joined("labels", targets, np.zeros(0, np.int64)),
        "iscrowd": joined("iscrowd", targets, np.zeros(0, np.int64)),
        "counts": counts,
    }
    detection_columns: DetectionColumns = {
        "boxes": joined("boxes", predictions, np.zeros((0, 4))),
        "labels": joined("labels", predictions, np.zeros(0, np.int64)),
        "scores": joined("scores", predictions, np.zeros(0)),
        "counts": counts,
    }
    if iou_type == "segm":
        target_columns["rles"] = [m for record in targets for m in record.get("masks", [])]
        detection_columns["rles"] = [m for record in predictions for m in record.get("masks", [])]

    by_record, record_detections = coco_inputs(predictions, targets)
    by_column, column_detections = coco_inputs_from_columns(detection_columns, target_columns)

    assert by_record.dataset_hash == by_column.dataset_hash
    assert by_record.num_annotations == by_column.num_annotations > 0
    if iou_type == "bbox":
        assert np.array_equal(np.asarray(record_detections), np.asarray(column_detections))
    else:
        assert len(record_detections) == len(column_detections) == len(per_image)
        for a, b in zip(record_detections, column_detections, strict=True):
            left, right = cast("dict[str, Any]", a), cast("dict[str, Any]", b)
            assert left["image_id"] == right["image_id"]
            for key in ("boxes", "scores", "labels"):
                assert np.array_equal(np.asarray(left[key]), np.asarray(right[key]))
            assert len(left["rles"]) == len(right["rles"])


def _box_records() -> tuple[list[Prediction], list[Target]]:
    """One image, one box each side, overlapping exactly."""
    box = np.array([[10.0, 10.0, 20.0, 30.0]])
    targets: list[Target] = [{"boxes": box, "labels": np.array([1])}]
    predictions: list[Prediction] = [
        {"boxes": box, "scores": np.array([0.9]), "labels": np.array([1])}
    ]
    return predictions, targets


def _mask_records() -> tuple[list[Prediction], list[Target]]:
    """The same, carrying masks and no boxes at all."""
    predicted, actual = _discs()
    targets: list[Target] = [{"labels": np.array([0]), "masks": masks} for masks in actual]
    predictions: list[Prediction] = [
        {"labels": np.array([0]), "scores": np.array([0.9]), "masks": masks} for masks in predicted
    ]
    return predictions, targets


def test_a_mask_column_is_all_or_nothing() -> None:
    """Masking only some records misreads every image after the first.

    The flat mask list is indexed through ``counts``, so a column
    covering part of a side hands image ``i`` another image's mask, and
    the area column broadcasts instead of refusing. The columnar
    spelling already rejected this shape; both must, since they are one
    conversion.
    """
    predicted, actual = _discs()
    box = np.zeros((1, 4))
    targets: list[Target] = [
        {"boxes": box, "labels": np.array([0]), "masks": actual[0]},
        {"boxes": box, "labels": np.array([0])},
    ]
    predictions: list[Prediction] = [
        {"boxes": box, "scores": np.array([0.9]), "labels": np.array([0]), "masks": predicted[0]},
        {"boxes": box, "scores": np.array([0.9]), "labels": np.array([0])},
    ]
    with pytest.raises(ValueError, match="all-or-nothing"):
        coco_inputs(predictions, targets)


def test_bbox_metrics_refuse_records_without_boxes() -> None:
    """The zero box column a mask-only record gets scores 0 under ``bbox``.

    :func:`coco_inputs` fills it so a segm pipeline need not carry boxes
    no segm kernel reads. A bbox grid does read it, and reports ``0`` —
    a plausible number with no error — so the one function that names
    the grid refuses the combination rather than returning it.
    """
    predictions, targets = _mask_records()
    assert float(coco_metrics(predictions, targets, iou_type="segm")["map"]) > 0
    with pytest.raises(KeyError, match="boxes is required"):
        coco_metrics(predictions, targets, iou_type="bbox")
    with pytest.raises(KeyError, match="boxes is required"):
        coco_metrics(predictions, targets, iou_type=("bbox", "segm"))


def test_unknown_iou_type_is_refused() -> None:
    """A typo would fall through to the segm grid.

    The run then reports one IoU type's metrics under the other's keys —
    the silent mis-selection ``_validate_literals`` exists to prevent,
    and the reason ``iou_type`` is checked beside ``area`` and
    ``box_format``.
    """
    predictions, targets = _mask_records()
    with pytest.raises(ValueError, match="unknown iou_type"):
        coco_metrics(predictions, targets, iou_type=cast("Any", "boundary"))


@pytest.mark.parametrize("max_dets", [(100, 10, 1), (1, 1, 1), (0, 10, 100), (1, 10)])
def test_max_dets_must_be_three_increasing_caps(max_dets: tuple[int, ...]) -> None:
    """Every image is capped at the last entry, whatever the keys read.

    A descending ladder truncates the run to the smallest cap while the
    keys still say ``mar_100``; a repeated one collapses two ``mar_{n}``
    keys into one, so the dict quietly holds eleven statistics instead
    of twelve.
    """
    predictions, targets = _box_records()
    with pytest.raises(ValueError, match="max_dets"):
        coco_metrics(predictions, targets, max_dets=max_dets)


def test_fractional_iscrowd_is_refused() -> None:
    """``0.5`` truncates to ``0`` and un-crowds the annotation.

    The same silent outcome as a ``uint8`` column wrapping 256 to 0
    (quirks **D1**, **E1**), reached from the other side of the cast.
    """
    box = np.array([[10.0, 10.0, 20.0, 30.0]])
    predictions, targets = _box_records()
    with pytest.raises(ValueError, match="integral"):
        coco_inputs(predictions, [{**targets[0], "iscrowd": np.array([0.5])}])
    detection_columns: DetectionColumns = {
        "boxes": box,
        "scores": np.array([0.9]),
        "labels": np.array([1]),
        "counts": np.array([1]),
    }
    target_columns: TargetColumns = {
        "boxes": box,
        "labels": np.array([1]),
        "iscrowd": np.array([0.5]),
        "counts": np.array([1]),
    }
    with pytest.raises(ValueError, match="integral"):
        coco_inputs_from_columns(detection_columns, target_columns)


def test_columnar_image_ids_must_be_unique() -> None:
    """Duplicates point two images' annotations at one id.

    ``_image_ids`` already refuses the identical mistake in the
    per-sample spelling, and the two spellings must accept exactly the
    same inputs.
    """
    columns: Any = {
        "boxes": np.zeros((2, 4)),
        "scores": np.ones(2),
        "labels": np.ones(2, np.int64),
        "counts": np.array([1, 1]),
    }
    with pytest.raises(ValueError, match="unique"):
        coco_inputs_from_columns(columns, columns, image_ids=np.array([5, 5]))


def test_an_empty_rles_column_does_not_hide_the_masks() -> None:
    """A caller that sets both mask keys unconditionally still gets its masks.

    ``_has_masks`` reads whichever column is non-empty, so keying the
    read on mere presence made ``rles: []`` beside a real ``masks``
    array look like zero masks for N annotations.
    """
    predictions, targets = _mask_records()
    both: list[Prediction] = [{**predictions[0], "rles": []}]
    assert float(coco_metrics(both, targets, iou_type="segm")["map"]) > 0


def test_a_supplied_size_survives_a_bbox_only_run() -> None:
    """``Target.size`` is the caller's, whether or not masks are present.

    Resolving sizes only for a masked build silently returned ``0x0``
    for an image whose size the caller had pinned.
    """
    predictions, targets = _box_records()
    sized, bare = (
        coco_inputs(predictions, [{**targets[0], "size": (480, 640)}])[0],
        (coco_inputs(predictions, targets)[0]),
    )
    assert sized.dataset_hash != bare.dataset_hash


def test_a_sizes_column_must_cover_every_image() -> None:
    """A short column is refused, not read as "no size for the rest"."""
    columns: Any = {
        "boxes": np.zeros((2, 4)),
        "scores": np.ones(2),
        "labels": np.ones(2, np.int64),
        "counts": np.array([1, 1]),
    }
    sized = cast("Any", {**columns, "sizes": np.array([[4, 5]])})
    with pytest.raises(ValueError, match="sizes"):
        coco_inputs_from_columns(columns, sized)
    with pytest.raises(ValueError, match="sizes"):
        gt_image_sizes([[{"size": (7, 9), "counts": b"x"}], None], None, [(11, 13)])


def test_masked_ground_truth_beside_box_only_detections() -> None:
    """A real COCO ground truth carries segmentation even for a box model.

    The masked build must not demand a mask the detector never
    produced — only that each side is internally whole.
    """
    _, actual = _discs()
    box = np.array([[6.0, 6.0, 13.0, 13.0]])
    targets: list[Target] = [{"boxes": box, "labels": np.array([0]), "masks": actual[0]}]
    predictions: list[Prediction] = [
        {"boxes": box, "scores": np.array([0.9]), "labels": np.array([0])}
    ]
    assert float(coco_metrics(predictions, targets, iou_type="bbox")["map"]) > 0


def test_a_mask_error_names_the_callers_index() -> None:
    """The rewritten prefix keeps the field the extractor appended to it.

    ``rle_area`` numbers the compacted list it was handed, so the index
    is remapped; consuming the separator as well spliced the prefix back
    together as ``masks[1]: .counts:``.
    """
    from vernier._samples import _mask_areas

    with pytest.raises(ValueError, match=r"^masks\[1\]\.counts: "):
        _mask_areas([np.zeros((4, 4), np.uint8), {"size": (4, 4), "counts": b"\xff\xff"}])


def test_the_pair_drives_the_surfaces_the_docs_promise() -> None:
    """ADR-0063's case for returning inputs instead of a metric.

    The claim is only worth making if it is true, and when this test was
    written it was narrower than "everything": TIDE, LRP, the confusion
    matrix, the FP-IoU histogram and the ``tables=`` / ``manifest=``
    paths each refused a :class:`CocoDataset` handle and asked for GT
    JSON bytes. ADR-0064 lifted that, so the refusal half of this test
    became a works half — every instance surface now reads the pair.

    This stays the *breadth* check (one conversion reaching each
    surface). The per-surface equality against the JSON route lives in
    ``tests/python/test_diagnostic_surfaces_take_the_pair.py``.
    """
    from vernier import _core
    from vernier.instance import (
        Bbox,
        Evaluator,
        confusion_matrix,
        error_decomposition,
        evaluate_bbox_grid,
        fp_iou_histogram,
        optimal_lrp,
    )

    predictions, targets = _box_records()
    ground_truth, detections = coco_inputs(predictions, targets)

    summary = Evaluator(iou=Bbox()).evaluate(ground_truth, detections)
    assert summary.stats[0] > 0

    grid = evaluate_bbox_grid(
        ground_truth, detections, parity_mode="corrected", max_dets_per_image=100, use_cats=True
    )
    accumulated = grid.accumulate([1, 10, 100])
    assert accumulated.summarize([1, 10, 100]).stats[0] == summary.stats[0]
    assert _core.cells_from_grid(grid) is not None

    report = error_decomposition(ground_truth, detections)
    assert report.baseline_map == pytest.approx(summary.stats[0])
    assert fp_iou_histogram(ground_truth, detections).n_total_dts == 1
    assert optimal_lrp(ground_truth, detections).per_class

    pytest.importorskip("polars", reason="`vernier[tables]` extra not installed")
    assert confusion_matrix(ground_truth, detections).height > 0

    tabled = Evaluator(iou=Bbox()).evaluate(ground_truth, detections, tables="all")
    assert tabled.summary is not None
    assert tabled.summary.stats == summary.stats
    assert tabled.per_class.height > 0

    # `coco_inputs` numbers images from 0, so the manifest keys it —
    # a mismatched key warns and skips rather than raising, which would
    # make this an assertion about nothing.
    manifest = {"manifest_version": "1", "key_kind": "image_id", "rows": [{"key": 0, "split": "a"}]}
    partitioned = Evaluator(iou=Bbox()).evaluate(ground_truth, detections, manifest=manifest)
    assert partitioned.summary is not None
    assert partitioned.summary.stats == summary.stats
    assert partitioned.slices.height > 0
