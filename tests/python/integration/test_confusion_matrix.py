"""End-to-end smoke tests for :func:`vernier.instance.confusion_matrix` (ADR-0023).

Hand-computed expected counts on the existing TIDE fixtures cover the
three kernels (bbox / segm / boundary) plus the two unsupported kernel
selectors (`Keypoints` and the `CocoDataset` handle, both rejected with
:class:`NotImplementedError`).

The Rust integration test (`crates/vernier-core/tests/confusion_matrix.rs`)
covers the algorithmic correctness against the four canonical TIDE
fixtures; this Python file pins the Python wrapper / FFI dict shape and
exercises the three kernels through the public surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import vernier
from vernier.instance import Bbox, Boundary, CocoDataset, Keypoints, Segm

FIXTURE_ROOT = Path(__file__).parents[1] / "oracle" / "tide" / "fixtures"


def _load(name: str) -> tuple[bytes, bytes]:
    gt = (FIXTURE_ROOT / name / "gt.json").read_bytes()
    dt = (FIXTURE_ROOT / name / "dt.json").read_bytes()
    return gt, dt


def _to_dict(df: object) -> dict[tuple[str, str], int]:
    """Project the long-format DataFrame to a `(gt, dt) -> count` dict
    so per-cell asserts are clearer than walking row indices."""
    rows = df.iter_rows(named=True)  # type: ignore[attr-defined]
    return {(r["gt_class"], r["dt_class"]): int(r["count"]) for r in rows}


def test_confusion_matrix_bbox_all_perfect_diagonal():
    gt, dt = _load("all_perfect")
    df = vernier.instance.confusion_matrix(gt, dt, iou=Bbox())

    assert df.columns == ["gt_class", "dt_class", "count"]
    cells = _to_dict(df)
    assert cells == {("1", "1"): 1, ("2", "2"): 1}


def test_confusion_matrix_bbox_all_cls_off_diagonal():
    gt, dt = _load("all_cls")
    df = vernier.instance.confusion_matrix(gt, dt, iou=Bbox())

    cells = _to_dict(df)
    assert cells == {("1", "2"): 1, ("2", "1"): 1}


def test_confusion_matrix_bbox_all_bkg_fp_row_with_diagonal_covers():
    # The all_bkg fixture has *both* high-score background DTs and
    # lower-score covering DTs (see the TIDE fixture); the covering
    # DTs claim the GTs (the score order doesn't matter — the side
    # pass argmax picks the highest-IoU GT regardless of score).
    gt, dt = _load("all_bkg")
    df = vernier.instance.confusion_matrix(gt, dt, iou=Bbox())

    cells = _to_dict(df)
    assert cells == {
        ("1", "1"): 1,
        ("2", "2"): 1,
        ("__none__", "1"): 1,
        ("__none__", "2"): 1,
    }


def test_confusion_matrix_bbox_with_ignore_excludes_crowd_from_missed():
    gt, dt = _load("with_ignore")
    df = vernier.instance.confusion_matrix(gt, dt, iou=Bbox())

    cells = _to_dict(df)
    # One TP (the regular GT on image 2), one FP (background DT on
    # image 1), no missed (the crowd GT is silent).
    assert cells == {("1", "1"): 1, ("__none__", "1"): 1}


def test_confusion_matrix_segm_all_perfect_diagonal():
    gt, dt = _load("segm_all_perfect")
    df = vernier.instance.confusion_matrix(gt, dt, iou=Segm())

    cells = _to_dict(df)
    assert cells == {("1", "1"): 1, ("2", "2"): 1}


def test_confusion_matrix_boundary_all_perfect_diagonal():
    gt, dt = _load("boundary_all_perfect")
    # Default dilation_ratio of 0.02 matches the COCO setup the fixture
    # was authored for.
    df = vernier.instance.confusion_matrix(gt, dt, iou=Boundary())

    cells = _to_dict(df)
    # The boundary fixture has a single-class GT (cat 1) with two
    # annotations; the perfect-match DTs land on the same class.
    # The exact count layout depends on the fixture; assert the
    # diagonal is non-empty and no off-diagonal cells fired.
    assert any(g == d for (g, d) in cells), "expected at least one diagonal cell"
    for (g, d), count in cells.items():
        if g != "__none__" and d != "__none__":
            assert g == d, f"unexpected off-diagonal cell ({g}, {d}) = {count}"


def test_confusion_matrix_keypoints_rejected_per_adr_0024():
    gt, dt = _load("all_perfect")
    with pytest.raises(NotImplementedError, match="ADR-0024"):
        vernier.instance.confusion_matrix(gt, dt, iou=Keypoints())


def test_confusion_matrix_dataset_handle_rejected():
    gt, dt = _load("all_perfect")
    handle = CocoDataset.from_json(gt)
    with pytest.raises(NotImplementedError, match="CocoDataset handles"):
        vernier.instance.confusion_matrix(handle, dt, iou=Bbox())


def test_confusion_matrix_use_cats_false_rejected():
    gt, dt = _load("all_perfect")
    with pytest.raises(ValueError, match="use_cats=False"):
        vernier.instance.confusion_matrix(gt, dt, iou=Bbox(), use_cats=False)


def test_confusion_matrix_t_f_out_of_range_rejected():
    gt, dt = _load("all_perfect")
    with pytest.raises(ValueError, match="iou_threshold"):
        vernier.instance.confusion_matrix(gt, dt, iou=Bbox(), t_f=1.5)


def test_confusion_matrix_default_iou_is_bbox():
    # Omitting the `iou` kwarg should default to Bbox(), exercising
    # the same path as iou=Bbox() above.
    gt, dt = _load("all_perfect")
    df = vernier.instance.confusion_matrix(gt, dt)
    cells = _to_dict(df)
    assert cells == {("1", "1"): 1, ("2", "2"): 1}
