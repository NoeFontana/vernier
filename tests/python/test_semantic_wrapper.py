"""Wrapper-layer tests for the ADR-0028 semantic-segmentation surface.

The Rust kernel is exercised by ``crates/vernier-semantic`` unit tests
(`kernel.rs`, `summarize.rs`); the FFI is exercised by
``test_semantic_smoke.py``. This file's job is narrower: prove the
Python wrapper layer (`vernier.semantic.{Evaluator, Dataset,
Predictions}`) routes inputs to the FFI correctly and that the
preset constructors bake the right `(n_classes, ignore_label)`
constants.

Per-dataset presets that decode PNG fixtures live in
``test_semantic_presets.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vernier.semantic import (
    ADE20K_IGNORE_LABEL,
    ADE20K_N_CLASSES,
    CITYSCAPES_IGNORE_LABEL,
    CITYSCAPES_N_CLASSES,
    PASCAL_VOC_IGNORE_LABEL,
    PASCAL_VOC_N_CLASSES,
    Dataset,
    Evaluator,
    Predictions,
    Summary,
)


def _toy_pair(n_classes: int = 3, h: int = 4, w: int = 4) -> tuple[Dataset, Predictions]:
    arr = np.tile(np.arange(n_classes, dtype=np.uint32), (h * w // n_classes) + 1)[: h * w]
    arr = arr.reshape(h, w)
    gt = Dataset.from_arrays({1: arr}, n_classes=n_classes)
    dt = Predictions.from_arrays({1: arr.copy()})
    return gt, dt


def test_evaluator_perfect_match_through_wrapper() -> None:
    gt, dt = _toy_pair()
    s = Evaluator().evaluate(gt, dt)
    assert isinstance(s, Summary)
    assert s.miou == pytest.approx(1.0)
    assert s.pixel_accuracy == pytest.approx(1.0)


def test_dataset_from_arrays_upcasts_dtype() -> None:
    # uint8 input is upcast to uint32 by the wrapper.
    arr_u8 = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    gt = Dataset.from_arrays({1: arr_u8}, n_classes=2)
    assert gt.label_maps[1].dtype == np.uint32


def test_dataset_validates_n_classes() -> None:
    with pytest.raises(ValueError, match="n_classes"):
        Dataset(label_maps={}, n_classes=0)


def test_dataset_validates_ignore_label() -> None:
    with pytest.raises(ValueError, match="ignore_label"):
        Dataset(label_maps={}, n_classes=3, ignore_label=-1)


def test_evaluator_propagates_ignore_label() -> None:
    # Cityscapes-style ignore=255; verify the wrapper plumbs it through
    # to the FFI.
    gt = Dataset.from_arrays(
        {1: np.array([[0, 255], [1, 1]], dtype=np.uint32)},
        n_classes=2,
        ignore_label=255,
    )
    dt = Predictions.from_arrays({1: np.array([[0, 0], [1, 1]], dtype=np.uint32)})
    s = Evaluator().evaluate(gt, dt)
    assert s.miou == pytest.approx(1.0)
    # Ignore pixel was excluded; only 3 pixels in the matrix.
    assert s.confusion_matrix.total == 3


def test_evaluator_propagates_label_remap() -> None:
    # DT in raw class space {7, 8, 9}; remap to eval space {0, 1, 2}.
    gt = Dataset.from_arrays(
        {1: np.array([[0, 0, 1], [1, 2, 2]], dtype=np.uint32)},
        n_classes=3,
    )
    dt_raw = Predictions.from_arrays(
        {1: np.array([[7, 7, 8], [8, 9, 9]], dtype=np.uint32)},
    )
    s = Evaluator(label_remap={7: 0, 8: 1, 9: 2}).evaluate(gt, dt_raw)
    assert s.miou == pytest.approx(1.0)


def test_evaluator_strict_yields_nan_for_zero_support() -> None:
    gt = Dataset.from_arrays({1: np.array([[0, 1]], dtype=np.uint32)}, n_classes=3)
    dt = Predictions.from_arrays({1: np.array([[0, 1]], dtype=np.uint32)})
    s = Evaluator(parity_mode="strict").evaluate(gt, dt)
    rows = s.per_class()
    assert np.isnan(rows[2].iou)


def test_evaluator_is_immutable() -> None:
    e = Evaluator()
    with pytest.raises(AttributeError):
        e.parity_mode = "strict"  # pyright: ignore[reportAttributeAccessIssue]


def test_unknown_parity_mode_rejected_by_ffi() -> None:
    gt, dt = _toy_pair()
    e = Evaluator(parity_mode="aligned")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="parity_mode"):
        e.evaluate(gt, dt)


def test_predictions_from_binary_masks_argmax() -> None:
    # Two classes, 2x2 pixels:
    # class 0 mask: [[1,0],[0,1]]  → pixels 0 and 3 are class 0
    # class 1 mask: [[0,1],[1,0]]  → pixels 1 and 2 are class 1
    # No overlap; argmax → unique class per pixel.
    masks = np.stack(
        [
            np.array([[1, 0], [0, 1]], dtype=np.uint8),
            np.array([[0, 1], [1, 0]], dtype=np.uint8),
        ],
        axis=0,
    )
    dt = Predictions.from_binary_masks({1: masks}, merge="argmax", unlabeled_class=255)
    expected = np.array([[0, 1], [1, 0]], dtype=np.uint32)
    np.testing.assert_array_equal(dt.label_maps[1], expected)


def test_predictions_from_binary_masks_unlabeled_pixels() -> None:
    # All-zero mask stack → every pixel is unlabeled_class.
    masks = np.zeros((3, 2, 2), dtype=np.uint8)
    dt = Predictions.from_binary_masks({1: masks}, merge="argmax", unlabeled_class=255)
    assert (dt.label_maps[1] == 255).all()


def test_predictions_from_binary_masks_highest_class_id() -> None:
    # Pixel where class 0, 1, and 2 all fire. argmax → 0; first → 0;
    # highest_class_id → 2.
    masks = np.ones((3, 1, 1), dtype=np.uint8)
    dt_argmax = Predictions.from_binary_masks({1: masks}, merge="argmax", unlabeled_class=255)
    dt_highest = Predictions.from_binary_masks(
        {1: masks}, merge="highest_class_id", unlabeled_class=255
    )
    assert dt_argmax.label_maps[1][0, 0] == 0
    assert dt_highest.label_maps[1][0, 0] == 2


def test_predictions_from_binary_masks_rejects_2d() -> None:
    with pytest.raises(ValueError, match="3-D"):
        Predictions.from_binary_masks({1: np.zeros((2, 2), dtype=np.uint8)})


def test_predictions_from_binary_masks_unknown_merge_rule() -> None:
    masks = np.zeros((1, 2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="merge rule"):
        Predictions.from_binary_masks({1: masks}, merge="bogus")  # type: ignore[arg-type]


def test_preset_constants_match_pinned_values() -> None:
    # Tripwire: the Python-side constants mirror the Rust-side
    # `vernier_semantic::parity::*`. If the Rust pin moves, the
    # mirroring constant here moves with it; both update atomically.
    assert CITYSCAPES_IGNORE_LABEL == 255
    assert CITYSCAPES_N_CLASSES == 19
    assert ADE20K_IGNORE_LABEL == 0
    assert ADE20K_N_CLASSES == 150
    assert PASCAL_VOC_IGNORE_LABEL == 255
    assert PASCAL_VOC_N_CLASSES == 21
