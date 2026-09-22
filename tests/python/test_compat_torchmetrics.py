"""Consumer contract: TorchMetrics' ``MeanAveragePrecision`` under the drop-in.

TorchMetrics is the widest real consumer of
``pycocotools.cocoeval.COCOeval``'s mutable surface: it rebinds
``params.iouThrs`` / ``recThrs`` / ``maxDets`` on every ``compute()``,
reads ``coco_eval.ious`` and ``coco_eval.eval`` back out under
``extended_summary=True``, and drives one evaluator around a per-class
loop by assigning ``params.catIds = [class_id]`` under
``class_metrics=True`` (``torchmetrics/detection/helpers.py``). Each of
those is a way the ADR-0007 drop-in can silently stop being a drop-in.

The assertion is conformance against the real library, in the style of
``tests/python/test_compat.py``: run the *same* metric state through
``MeanAveragePrecision`` twice — once with
:func:`vernier.adapters.patched_pycocotools`, once without — and require
the two result dictionaries to be equal, element for element. Nothing
here mirrors what vernier is expected to produce; upstream pycocotools
decides, and vernier has to agree.

Fast (synthetic tensors, no checkpoints), so deliberately not marked
``slow``. ``torchmetrics`` ships in the ``real-models`` extra because it
needs torch; the module skips cleanly without it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
_detection = pytest.importorskip("torchmetrics.detection")

from vernier.adapters import patched_pycocotools  # noqa: E402

MeanAveragePrecision: Any = _detection.MeanAveragePrecision

# Two images of boxes that overlap their targets closely but not exactly,
# across two classes, plus a target with no prediction and a prediction
# with no target. That is enough to exercise every branch the result
# dictionary reports: a matched pair at several IoU thresholds, a false
# positive, a false negative, and a class whose per-class AP is defined
# only on one of the two images.
_PRED_BOXES: tuple[tuple[tuple[float, ...], ...], ...] = (
    ((10.0, 10.0, 60.0, 60.0), (70.0, 70.0, 110.0, 120.0), (5.0, 200.0, 25.0, 220.0)),
    ((5.0, 5.0, 45.0, 55.0), (120.0, 30.0, 190.0, 100.0)),
)
_PRED_SCORES: tuple[tuple[float, ...], ...] = ((0.9, 0.7, 0.2), (0.4, 0.8))
_PRED_LABELS: tuple[tuple[int, ...], ...] = ((0, 1, 0), (1, 0))
_TARGET_BOXES: tuple[tuple[tuple[float, ...], ...], ...] = (
    ((11.0, 11.0, 61.0, 61.0), (70.0, 72.0, 112.0, 120.0)),
    ((6.0, 6.0, 44.0, 54.0), (130.0, 35.0, 185.0, 95.0), (300.0, 300.0, 340.0, 340.0)),
)
_TARGET_LABELS: tuple[tuple[int, ...], ...] = ((0, 1), (1, 0, 0))


def _predictions() -> list[dict[str, Any]]:
    return [
        {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "scores": torch.tensor(scores, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        for boxes, scores, labels in zip(_PRED_BOXES, _PRED_SCORES, _PRED_LABELS)
    ]


def _targets() -> list[dict[str, Any]]:
    return [
        {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        for boxes, labels in zip(_TARGET_BOXES, _TARGET_LABELS)
    ]


def _compute(*, patched: bool, **kwargs: Any) -> dict[str, Any]:
    metric = MeanAveragePrecision(**kwargs)
    metric.update(_predictions(), _targets())
    if not patched:
        return dict(metric.compute())
    # The patch has to be live across `compute()` only: `update()` just
    # accumulates tensors, and the COCOeval symbol is looked up inside
    # the backend at compute time.
    with patched_pycocotools("strict"):
        return dict(metric.compute())


def _assert_equal(reference: Any, candidate: Any, path: str) -> None:
    if isinstance(reference, dict):
        assert isinstance(candidate, dict), f"{path}: dict vs {type(candidate).__name__}"
        assert set(reference) == set(candidate), f"{path}: key sets differ"
        for key in reference:
            _assert_equal(reference[key], candidate[key], f"{path}[{key!r}]")
        return
    # `ious` values are `[]` for an (image, category) pair with nothing on
    # one side (quirk F5) and a tensor otherwise; TorchMetrics converts
    # only the latter. The types have to match, or a downstream
    # `.shape` / `.numel()` breaks on one path and not the other.
    assert type(reference) is type(candidate), (
        f"{path}: {type(reference).__name__} vs {type(candidate).__name__}"
    )
    if torch.is_tensor(reference):
        assert reference.dtype == candidate.dtype, f"{path}: dtype"
        assert reference.shape == candidate.shape, f"{path}: shape"
        assert torch.equal(reference, candidate), f"{path}: values"
    else:
        assert reference == candidate, f"{path}: values"


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="defaults"),
        pytest.param({"extended_summary": True}, id="extended_summary"),
        pytest.param({"class_metrics": True}, id="class_metrics"),
        pytest.param(
            {"extended_summary": True, "class_metrics": True},
            id="extended_summary+class_metrics",
        ),
    ],
)
def test_mean_average_precision_matches_unpatched(kwargs: dict[str, Any]) -> None:
    reference = _compute(patched=False, **kwargs)
    candidate = _compute(patched=True, **kwargs)
    _assert_equal(reference, candidate, "result")


def test_extended_summary_reports_a_populated_iou_map() -> None:
    # Guards the assertion above against passing vacuously: if `ious`
    # ever came back empty on both paths, the equality check would still
    # hold. `extended_summary` is the only reader of `COCOeval.ious`, and
    # it must see real matrices for the pairs that have both sides.
    ious = _compute(patched=True, extended_summary=True)["ious"]
    reference = _compute(patched=False, extended_summary=True)["ious"]
    assert isinstance(ious, dict)
    populated = {
        key: value
        for key, value in ious.items()
        if torch.is_tensor(value) and value.ndim == 2 and value.numel()
    }
    assert populated, "extended_summary reported no non-empty IoU matrix"
    # Same pairs carry a matrix on both paths, and the matrices agree —
    # the shape check is what the equality test above cannot see, since
    # two empty maps would compare equal just as happily.
    assert set(populated) == {
        key
        for key, value in reference.items()
        if torch.is_tensor(value) and value.ndim == 2 and value.numel()
    }
    for key, matrix in populated.items():
        assert matrix.shape == reference[key].shape


def test_class_metrics_reports_one_entry_per_class() -> None:
    # Same anti-vacuity guard for the `params.catIds = [class_id]` loop:
    # the per-class vectors must be as long as the class list, not the
    # `[-1]` sentinel `class_metrics=False` reports.
    result = _compute(patched=True, class_metrics=True)
    classes = result["classes"]
    assert classes.numel() == 2
    assert result["map_per_class"].numel() == classes.numel()
    assert result["mar_100_per_class"].numel() == classes.numel()


# ---------------------------------------------------------------------------
# The per-sample ingest route (ADR-0063)
# ---------------------------------------------------------------------------
#
# The drop-in above proves vernier agrees with pycocotools when it stands in
# for `COCOeval`. This section proves the *other* route agrees too: the same
# metric state, handed to `vernier.adapters.coco_metrics` as per-image records
# instead of through the shim. That route bypasses the COCO dictionaries
# entirely, so nothing above covers it.
#
# Comparison is at `float32`. TorchMetrics reports `float32` tensors
# (`CocoBackend._coco_stats_to_tensor_dict`) while vernier returns `float64`;
# rounding both to the narrower type is the only honest common precision, and
# it still pins every bit TorchMetrics' own consumers can observe.

_STAT_KEYS: tuple[str, ...] = (
    "map",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
    "mar_1",
    "mar_10",
    "mar_100",
    "mar_small",
    "mar_medium",
    "mar_large",
)


def _assert_stats_equal(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    for key in _STAT_KEYS:
        expected = np.float32(float(reference[key]))
        actual = np.float32(float(candidate[key]))
        assert expected == actual, f"{key}: {expected!r} != {actual!r}"


@pytest.mark.parametrize("box_format", ["xyxy", "xywh", "cxcywh"])
def test_coco_metrics_matches_pycocotools_on_bbox(box_format: str) -> None:
    """The route agrees with pycocotools for the state TorchMetrics holds."""
    from vernier.adapters import coco_metrics

    predictions, targets = _predictions(), _targets()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="pycocotools")
    metric.update(predictions, targets)
    reference = dict(metric.compute())

    converted_predictions = [
        {**record, "boxes": _convert_boxes(record["boxes"], box_format)} for record in predictions
    ]
    converted_targets = [
        {**record, "boxes": _convert_boxes(record["boxes"], box_format)} for record in targets
    ]
    candidate = coco_metrics(
        converted_predictions,  # type: ignore[arg-type]
        converted_targets,  # type: ignore[arg-type]
        iou_type="bbox",
        box_format=box_format,  # type: ignore[arg-type]
        parity_mode="strict",
    )
    _assert_stats_equal(reference, candidate)
    assert list(np.asarray(candidate["classes"])) == list(reference["classes"].numpy())


def _convert_boxes(boxes: Any, box_format: str) -> Any:
    """Re-express ``xyxy`` boxes in ``box_format``, so all three describe one box."""
    if box_format == "xyxy":
        return boxes
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    if box_format == "xywh":
        return torch.stack([boxes[:, 0], boxes[:, 1], widths, heights], dim=1)
    return torch.stack(
        [boxes[:, 0] + widths / 2, boxes[:, 1] + heights / 2, widths, heights], dim=1
    )


def test_coco_metrics_matches_pycocotools_on_segm() -> None:
    """Same, under ``segm``.

    Exercises the columnar detection route and the mask-area fallback:
    TorchMetrics stores no ground-truth area, so every annotation's area
    is derived from its mask. That the small / medium / large split
    agrees is what proves the fallback ran and ran correctly.
    """
    from vernier.adapters import coco_metrics

    height, width = 40, 50
    rows, columns = np.ogrid[:height, :width]

    def disc(center_x: int, center_y: int, radius: int) -> Any:
        return ((rows - center_y) ** 2 + (columns - center_x) ** 2 <= radius**2).astype(np.uint8)

    def boxes_of(masks: Any) -> Any:
        corners = []
        for mask in masks:
            ys, xs = np.nonzero(mask)
            corners.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        return torch.tensor(np.asarray(corners, dtype=np.float64))

    predicted = [np.stack([disc(12, 12, 7), disc(35, 25, 6)]), np.stack([disc(20, 20, 9)])]
    actual = [np.stack([disc(13, 12, 7), disc(34, 25, 6)]), np.stack([disc(21, 20, 9)])]

    predictions = [
        {
            "boxes": boxes_of(masks),
            "scores": torch.tensor([0.9, 0.6][: len(masks)]),
            "labels": torch.tensor([0, 1][: len(masks)]),
            "masks": torch.tensor(masks.astype(bool)),
        }
        for masks in predicted
    ]
    targets = [
        {
            "boxes": boxes_of(masks),
            "labels": torch.tensor([0, 1][: len(masks)]),
            "masks": torch.tensor(masks.astype(bool)),
        }
        for masks in actual
    ]

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="segm", backend="pycocotools")
    metric.update(predictions, targets)
    reference = dict(metric.compute())

    candidate = coco_metrics(
        predictions,  # type: ignore[arg-type]
        targets,  # type: ignore[arg-type]
        iou_type="segm",
        box_format="xyxy",
        parity_mode="strict",
    )
    _assert_stats_equal(reference, candidate)
    # Anti-vacuity: every statistic being -1 would compare equal just as
    # happily. The discs are small, so the small bucket must be populated
    # and the large one must not — which is the mask-area fallback's doing.
    assert float(candidate["map_small"]) > 0
    assert float(candidate["map_large"]) == -1


def test_coco_metrics_matches_pycocotools_on_both_iou_types() -> None:
    """A two-IoU-type run prefixes its keys and emits the full set.

    One conversion serves both passes — the inputs do not depend on
    which grid reads them — so this pins that the shared ground truth
    still scores each IoU type exactly as a single-type run does. It is
    also the shape a training run that logs both actually uses.
    """
    from vernier.adapters import coco_metrics

    rows, columns = np.ogrid[:40, :50]

    def disc(center_x: int, center_y: int, radius: int) -> Any:
        return ((rows - center_y) ** 2 + (columns - center_x) ** 2 <= radius**2).astype(np.uint8)

    def boxes_of(masks: Any) -> Any:
        corners = []
        for mask in masks:
            ys, xs = np.nonzero(mask)
            corners.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        return torch.tensor(np.asarray(corners, dtype=np.float64))

    predicted = [np.stack([disc(12, 12, 7), disc(35, 25, 6)]), np.stack([disc(20, 20, 9)])]
    actual = [np.stack([disc(13, 12, 7), disc(34, 25, 6)]), np.stack([disc(21, 20, 9)])]
    predictions = [
        {
            "boxes": boxes_of(masks),
            "scores": torch.tensor([0.9, 0.6][: len(masks)]),
            "labels": torch.tensor([0, 1][: len(masks)]),
            "masks": torch.tensor(masks.astype(bool)),
        }
        for masks in predicted
    ]
    targets = [
        {
            "boxes": boxes_of(masks),
            "labels": torch.tensor([0, 1][: len(masks)]),
            "masks": torch.tensor(masks.astype(bool)),
        }
        for masks in actual
    ]

    metric = MeanAveragePrecision(
        box_format="xyxy", iou_type=("bbox", "segm"), backend="pycocotools"
    )
    metric.update(predictions, targets)
    reference = dict(metric.compute())

    candidate = coco_metrics(
        predictions,  # type: ignore[arg-type]
        targets,  # type: ignore[arg-type]
        iou_type=("bbox", "segm"),
        box_format="xyxy",
        parity_mode="strict",
    )
    for iou_type in ("bbox", "segm"):
        _assert_stats_equal(
            {key: reference[f"{iou_type}_{key}"] for key in _STAT_KEYS},
            {key: candidate[f"{iou_type}_{key}"] for key in _STAT_KEYS},
        )
    # Emits every aggregate key the oracle does — a prefix typo would
    # otherwise pass the loop above by comparing a key to itself.
    expected = {f"{iou_type}_{key}" for iou_type in ("bbox", "segm") for key in _STAT_KEYS}
    assert expected <= set(candidate)
    assert expected <= set(reference)
