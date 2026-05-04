"""Per-dataset preset tests for ``vernier.semantic`` (ADR-0028).

Synthesizes single-channel PNG label maps via Pillow and round-trips
them through ``Dataset.cityscapes`` / ``ade20k`` / ``pascal_voc`` /
``Predictions.from_files``. The presets bake the canonical
``(n_classes, ignore_label)`` constants; this file's job is to verify
the bake, not to validate end-to-end against the dataset bytes (those
parity tests live in ``tests/python/parity_semantic/`` once PR-B6
lands the mmsegmentation oracle).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

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
)


def _write_png(path: Path, arr: np.ndarray) -> Path:
    """Write a 2-D array as a single-channel PNG. Pillow's `mode='L'`
    (8-bit) for uint8, `mode='I;16'` for uint16, otherwise `mode='I'`
    (32-bit signed)."""
    if arr.dtype == np.uint8:
        Image.fromarray(arr, mode="L").save(path)
    elif arr.dtype == np.uint16:
        Image.fromarray(arr, mode="I;16").save(path)
    else:
        Image.fromarray(arr.astype(np.int32), mode="I").save(path)
    return path


def test_cityscapes_preset_bakes_constants(tmp_path: Path) -> None:
    # Cityscapes-style: uint8 PNG with class ids in [0, 19) plus 255.
    arr = np.array([[0, 1, 2], [255, 18, 0]], dtype=np.uint8)
    path = _write_png(tmp_path / "munich_000000.png", arr)
    gt = Dataset.cityscapes({1: path})
    assert gt.n_classes == CITYSCAPES_N_CLASSES == 19
    assert gt.ignore_label == CITYSCAPES_IGNORE_LABEL == 255
    assert gt.label_maps[1].dtype == np.uint32
    np.testing.assert_array_equal(gt.label_maps[1], arr.astype(np.uint32))


def test_ade20k_preset_bakes_constants(tmp_path: Path) -> None:
    # ADE20K-style: uint8 (or uint16 for >256 classes) with 0 = ignore.
    arr = np.array([[0, 1, 50], [149, 0, 1]], dtype=np.uint8)
    path = _write_png(tmp_path / "ade_val_00000001.png", arr)
    gt = Dataset.ade20k({1: path})
    assert gt.n_classes == ADE20K_N_CLASSES == 150
    assert gt.ignore_label == ADE20K_IGNORE_LABEL == 0


def test_pascal_voc_preset_bakes_constants(tmp_path: Path) -> None:
    arr = np.array([[0, 1, 20], [255, 0, 5]], dtype=np.uint8)
    path = _write_png(tmp_path / "2007_000027.png", arr)
    gt = Dataset.pascal_voc({1: path})
    assert gt.n_classes == PASCAL_VOC_N_CLASSES == 21
    assert gt.ignore_label == PASCAL_VOC_IGNORE_LABEL == 255


def test_dataset_from_files_rejects_rgb_png(tmp_path: Path) -> None:
    # Multi-channel PNG belongs in vernier.panoptic.Dataset; the
    # semantic surface rejects it with a typed message.
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb_path = tmp_path / "rgb.png"
    Image.fromarray(rgb, mode="RGB").save(rgb_path)
    with pytest.raises(ValueError, match="single-channel"):
        Dataset.from_files({1: rgb_path}, n_classes=3)


def test_predictions_from_files_round_trip(tmp_path: Path) -> None:
    arr = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    path = _write_png(tmp_path / "pred.png", arr)
    dt = Predictions.from_files({1: path})
    np.testing.assert_array_equal(dt.label_maps[1], arr.astype(np.uint32))


def test_cityscapes_preset_end_to_end(tmp_path: Path) -> None:
    # GT and DT identical → mIoU = 1.0; ignore_label respected.
    arr = np.array([[0, 1, 2], [255, 17, 18]], dtype=np.uint8)
    gt_path = _write_png(tmp_path / "gt.png", arr)
    dt_path = _write_png(tmp_path / "dt.png", arr)
    gt = Dataset.cityscapes({1: gt_path})
    dt = Predictions.from_files({1: dt_path})
    s = Evaluator().evaluate(gt, dt)
    assert s.miou == pytest.approx(1.0)
    # The ignore pixel was excluded; 5 pixels in the matrix.
    assert s.confusion_matrix.total == 5


def test_from_files_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, OSError)):
        Dataset.from_files({1: tmp_path / "does_not_exist.png"}, n_classes=3)
