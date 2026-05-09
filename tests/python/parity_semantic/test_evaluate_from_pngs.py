"""Parity tests for `Evaluator.evaluate_from_pngs` (ADR-0037).

Pins bit-equality between the fused libpng-decode path and the
existing array-input path. The val2017 perfect-DT smoke
(``test_parity_semantic_real``) covers the whole-dataset claim against
the mmsegmentation oracle; these tests cover the smaller "fused-PNG
matches array-path" claim that doesn't need the panoptic cache to run,
so it can land in the default ``just test`` loop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vernier.semantic as vsem


def _write_grayscale8(path: Path, arr: np.ndarray) -> None:
    """Write a uint8 (H, W) ndarray to ``path`` as an 8-bit grayscale PNG."""
    from PIL import Image

    if arr.dtype != np.uint8:
        raise TypeError(f"_write_grayscale8 requires uint8 input; got {arr.dtype}")
    Image.fromarray(arr, mode="L").save(path, format="PNG")


def _make_fixture(
    rng: np.random.Generator, n_images: int, height: int, width: int
) -> dict[int, np.ndarray]:
    """Generate `n_images` random uint8 (H, W) label maps in [0, 4) with
    a sprinkling of ignore=255 pixels."""
    out: dict[int, np.ndarray] = {}
    for iid in range(n_images):
        arr = rng.integers(0, 4, size=(height, width), dtype=np.uint8)
        ignore_mask = rng.random(size=(height, width)) < 0.05
        arr[ignore_mask] = 255
        out[iid] = arr
    return out


@pytest.mark.parity_semantic
def test_evaluate_from_pngs_matches_evaluate_from_arrays(tmp_path: Path) -> None:
    rng = np.random.default_rng(0xDEADBEEF)
    n_classes = 4
    ignore_label = 255

    gt_arrays = _make_fixture(rng, n_images=8, height=32, width=48)
    dt_arrays = _make_fixture(rng, n_images=8, height=32, width=48)

    gt_dir = tmp_path / "gt"
    dt_dir = tmp_path / "dt"
    gt_dir.mkdir()
    dt_dir.mkdir()
    gt_paths: dict[int, Path] = {}
    dt_paths: dict[int, Path] = {}
    for iid, arr in gt_arrays.items():
        p = gt_dir / f"{iid}.png"
        _write_grayscale8(p, arr)
        gt_paths[iid] = p
    for iid, arr in dt_arrays.items():
        p = dt_dir / f"{iid}.png"
        _write_grayscale8(p, arr)
        dt_paths[iid] = p

    evaluator = vsem.Evaluator(parity_mode="strict")

    array_summary = evaluator.evaluate(
        vsem.Dataset.from_arrays(gt_arrays, n_classes=n_classes, ignore_label=ignore_label),
        vsem.Predictions.from_arrays(dt_arrays),
    )
    fused_summary = evaluator.evaluate_from_pngs(
        gt_paths, dt_paths, n_classes=n_classes, ignore_label=ignore_label
    )

    array_counts = array_summary.confusion_matrix.counts()
    fused_counts = fused_summary.confusion_matrix.counts()
    np.testing.assert_array_equal(
        fused_counts,
        array_counts,
        err_msg="evaluate_from_pngs must produce a bit-equal confusion matrix",
    )
    assert fused_summary.miou == array_summary.miou
    assert fused_summary.fwiou == array_summary.fwiou
    assert fused_summary.pixel_accuracy == array_summary.pixel_accuracy


@pytest.mark.parity_semantic
def test_evaluate_from_pngs_rejects_rgb_png(tmp_path: Path) -> None:
    from PIL import Image

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    p = tmp_path / "rgb.png"
    Image.fromarray(rgb, mode="RGB").save(p, format="PNG")

    with pytest.raises(ValueError, match="unsupported PNG format"):
        vsem.Evaluator(parity_mode="strict").evaluate_from_pngs(
            {0: p}, {0: p}, n_classes=3, ignore_label=255
        )


@pytest.mark.parity_semantic
def test_evaluate_from_pngs_rejects_label_remap(tmp_path: Path) -> None:
    p = tmp_path / "x.png"
    _write_grayscale8(p, np.zeros((4, 4), dtype=np.uint8))

    with pytest.raises(NotImplementedError, match="label_remap"):
        vsem.Evaluator(parity_mode="strict", label_remap={0: 1}).evaluate_from_pngs(
            {0: p}, {0: p}, n_classes=3
        )


@pytest.mark.parity_semantic
def test_submit_png_matches_submit_array(tmp_path: Path) -> None:
    """`BackgroundEvaluator.submit_png` must produce a confusion matrix
    bit-equal to the array-input `submit` path on the same fixtures
    (ADR-0037)."""
    rng = np.random.default_rng(0xCAFEBABE)
    n_classes = 4
    ignore_label = 255

    gt_arrays = _make_fixture(rng, n_images=8, height=32, width=48)
    dt_arrays = _make_fixture(rng, n_images=8, height=32, width=48)

    array_evaluator = vsem.Evaluator(parity_mode="strict")
    with array_evaluator.background(n_classes, ignore_label=ignore_label) as bg_array:
        for iid in sorted(gt_arrays):
            bg_array.submit(
                iid,
                gt_arrays[iid].astype(np.uint32, copy=False),
                dt_arrays[iid].astype(np.uint32, copy=False),
            )
        array_summary = bg_array.finalize()

    gt_dir = tmp_path / "gt"
    dt_dir = tmp_path / "dt"
    gt_dir.mkdir()
    dt_dir.mkdir()
    gt_bytes_map: dict[int, bytes] = {}
    dt_bytes_map: dict[int, bytes] = {}
    for iid, arr in gt_arrays.items():
        p = gt_dir / f"{iid}.png"
        _write_grayscale8(p, arr)
        gt_bytes_map[iid] = p.read_bytes()
    for iid, arr in dt_arrays.items():
        p = dt_dir / f"{iid}.png"
        _write_grayscale8(p, arr)
        dt_bytes_map[iid] = p.read_bytes()

    png_evaluator = vsem.Evaluator(parity_mode="strict")
    with png_evaluator.background(n_classes, ignore_label=ignore_label) as bg_png:
        for iid in sorted(gt_bytes_map):
            bg_png.submit_png(iid, gt_bytes_map[iid], dt_bytes_map[iid])
        png_summary = bg_png.finalize()

    np.testing.assert_array_equal(
        png_summary.confusion_matrix.counts(),
        array_summary.confusion_matrix.counts(),
        err_msg="submit_png must produce a bit-equal confusion matrix vs submit(array)",
    )
    assert png_summary.miou == array_summary.miou


@pytest.mark.parity_semantic
def test_submit_png_rejects_rgb_png(tmp_path: Path) -> None:
    from PIL import Image

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    p = tmp_path / "rgb.png"
    Image.fromarray(rgb, mode="RGB").save(p, format="PNG")
    bytes_ = p.read_bytes()

    with (
        vsem.Evaluator(parity_mode="strict").background(3, ignore_label=255) as bg,
        pytest.raises(ValueError, match="unsupported PNG format"),
    ):
        bg.submit_png(0, bytes_, bytes_)
