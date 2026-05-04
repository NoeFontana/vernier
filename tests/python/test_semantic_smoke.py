"""Smoke tests for the ADR-0028 semantic-segmentation FFI surface.

End-to-end verification that the Rust kernel + summarize pass behave
correctly when driven from Python via :func:`vernier._core.evaluate_semantic_from_arrays`.
The Rust side has hand-computed unit tests in
``crates/vernier-semantic/src/{kernel,summarize}.rs``; this file's job
is narrower — prove the FFI surface carries the values through
verbatim and that the pyclass getters expose the expected fields.

The Python public wrapper (``vernier.semantic.Evaluator``, presets,
etc.) lands in PR-B5; until then, calls go directly through
``vernier._core``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vernier._core import (
    ClassSemanticStats,
    ConfusionMatrix,
    SemanticSummary,
    evaluate_semantic_from_arrays,
)


def _toy_perfect_match(
    n_classes: int = 3, h: int = 10, w: int = 10
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """One image, GT == DT, with every class represented at least once."""
    arr = np.tile(np.arange(n_classes, dtype=np.uint32), (h * w // n_classes) + 1)[: h * w]
    arr = arr.reshape(h, w)
    return {1: arr}, {1: arr.copy()}


def test_perfect_match_yields_unit_metrics() -> None:
    gt, dt = _toy_perfect_match(n_classes=3)
    s = evaluate_semantic_from_arrays(gt, dt, n_classes=3, parity_mode="corrected")
    assert isinstance(s, SemanticSummary)
    assert s.miou == pytest.approx(1.0)
    assert s.fwiou == pytest.approx(1.0)
    assert s.pixel_accuracy == pytest.approx(1.0)
    assert s.mean_accuracy == pytest.approx(1.0)


def test_per_class_is_dict_of_class_stats() -> None:
    gt, dt = _toy_perfect_match(n_classes=3)
    s = evaluate_semantic_from_arrays(gt, dt, n_classes=3, parity_mode="corrected")
    rows = s.per_class()
    assert set(rows) == {0, 1, 2}
    for cls, row in rows.items():
        assert isinstance(row, ClassSemanticStats)
        assert row.class_id == cls
        assert row.iou == pytest.approx(1.0)
        assert row.accuracy == pytest.approx(1.0)
        assert row.precision == pytest.approx(1.0)
        assert row.n_gt_pixels > 0
        assert row.n_dt_pixels == row.n_gt_pixels


def test_confusion_matrix_exposes_numpy_view() -> None:
    gt, dt = _toy_perfect_match(n_classes=3, h=4, w=3)
    s = evaluate_semantic_from_arrays(gt, dt, n_classes=3, parity_mode="corrected")
    cm = s.confusion_matrix
    assert isinstance(cm, ConfusionMatrix)
    assert cm.n_classes == 3
    counts = cm.counts()
    assert counts.shape == (3, 3)
    assert counts.dtype == np.uint64
    # Perfect match → diagonal-only matrix.
    assert int(np.trace(counts)) == 4 * 3
    assert int(counts.sum()) == 4 * 3
    assert cm.total == 4 * 3
    assert cm.trace == 4 * 3


def test_one_off_diagonal_pixel_drives_metric_drop() -> None:
    # Two classes, four pixels: gt=[0,0,1,1], dt=[0,1,1,1].
    # class 0: TP=1, FP=0, FN=1 → IoU=0.5
    # class 1: TP=2, FP=1, FN=0 → IoU=2/3
    # mIoU ≈ 0.5833
    gt = {1: np.array([[0, 0], [1, 1]], dtype=np.uint32)}
    dt = {1: np.array([[0, 1], [1, 1]], dtype=np.uint32)}
    s = evaluate_semantic_from_arrays(gt, dt, n_classes=2, parity_mode="corrected")
    assert s.miou == pytest.approx((0.5 + 2.0 / 3.0) / 2.0, rel=0, abs=1e-12)
    assert s.pixel_accuracy == pytest.approx(0.75)


def test_ignore_label_excludes_pixels() -> None:
    # gt=[0, 255, 1, 1]; dt=[0, 0, 1, 1]; ignore=255 (Cityscapes-style).
    # After mask: 3 pixels, all diagonal → mIoU=1.0.
    gt = {1: np.array([[0, 255], [1, 1]], dtype=np.uint32)}
    dt = {1: np.array([[0, 0], [1, 1]], dtype=np.uint32)}
    s = evaluate_semantic_from_arrays(
        gt, dt, n_classes=2, parity_mode="corrected", ignore_label=255
    )
    assert s.miou == pytest.approx(1.0)
    # Confusion matrix carries 3 pixels (the ignore was dropped).
    assert s.confusion_matrix.total == 3


def test_label_remap_applies_to_dt_before_bincount() -> None:
    # Predictions arrive in raw class space {7, 8, 9}; remap to eval
    # space {0, 1, 2}. After remap, GT (already in eval space) and
    # DT line up for a perfect match.
    gt = {1: np.array([[0, 0, 1], [1, 2, 2]], dtype=np.uint32)}
    dt_raw = {1: np.array([[7, 7, 8], [8, 9, 9]], dtype=np.uint32)}
    s = evaluate_semantic_from_arrays(
        gt,
        dt_raw,
        n_classes=3,
        parity_mode="corrected",
        label_remap={7: 0, 8: 1, 9: 2},
    )
    assert s.miou == pytest.approx(1.0)
    assert s.pixel_accuracy == pytest.approx(1.0)


def test_strict_mode_yields_nan_for_zero_support_class() -> None:
    # Class 2 is never in GT and never predicted. Under Strict the
    # per-class IoU is NaN; mIoU is the unweighted mean over the two
    # supported classes, so still 1.0.
    gt = {1: np.array([[0, 1]], dtype=np.uint32)}
    dt = {1: np.array([[0, 1]], dtype=np.uint32)}
    s = evaluate_semantic_from_arrays(gt, dt, n_classes=3, parity_mode="strict")
    rows = s.per_class()
    assert np.isnan(rows[2].iou)
    assert s.miou == pytest.approx(1.0)


def test_corrected_mode_yields_zero_for_zero_support_class() -> None:
    gt = {1: np.array([[0, 1]], dtype=np.uint32)}
    dt = {1: np.array([[0, 1]], dtype=np.uint32)}
    s = evaluate_semantic_from_arrays(gt, dt, n_classes=3, parity_mode="corrected")
    rows = s.per_class()
    assert rows[2].iou == 0.0
    assert s.miou == pytest.approx(1.0)


def test_missing_dt_image_raises() -> None:
    gt = {1: np.zeros((2, 2), dtype=np.uint32), 2: np.zeros((2, 2), dtype=np.uint32)}
    dt = {1: np.zeros((2, 2), dtype=np.uint32)}  # image_id=2 missing
    with pytest.raises(ValueError, match="missing prediction"):
        evaluate_semantic_from_arrays(gt, dt, n_classes=3, parity_mode="corrected")


def test_shape_mismatch_raises() -> None:
    gt = {1: np.zeros((2, 2), dtype=np.uint32)}
    dt = {1: np.zeros((3, 3), dtype=np.uint32)}
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate_semantic_from_arrays(gt, dt, n_classes=3, parity_mode="corrected")


def test_unknown_parity_mode_raises() -> None:
    gt = {1: np.zeros((2, 2), dtype=np.uint32)}
    dt = {1: np.zeros((2, 2), dtype=np.uint32)}
    with pytest.raises(ValueError, match="parity_mode"):
        evaluate_semantic_from_arrays(gt, dt, n_classes=3, parity_mode="aligned")


def test_n_classes_zero_rejected() -> None:
    with pytest.raises(ValueError, match="n_classes"):
        evaluate_semantic_from_arrays({}, {}, n_classes=0, parity_mode="corrected")


def test_empty_dataset_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        evaluate_semantic_from_arrays({}, {}, n_classes=3, parity_mode="corrected")


def test_multi_image_accumulation() -> None:
    # Two images, both perfect match.
    gt = {
        1: np.array([[0, 1]], dtype=np.uint32),
        2: np.array([[0, 1, 2]], dtype=np.uint32),
    }
    dt = {1: gt[1].copy(), 2: gt[2].copy()}
    s = evaluate_semantic_from_arrays(gt, dt, n_classes=3, parity_mode="corrected")
    assert s.miou == pytest.approx(1.0)
    # Five pixels total across both images, all on diagonal.
    assert s.confusion_matrix.total == 5
    assert s.confusion_matrix.trace == 5
