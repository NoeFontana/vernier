"""Parity tests for the loosened multi-dtype `submit(arr)` /
`evaluate(arr)` FFI surface (ADR-0037).

The kernel walks at native dtype; pins that `uint8`, `uint16`, and
`uint32` array inputs all produce a bit-equal confusion matrix on the
same fixture. Regression-protects against an accidental re-introduction
of an upcast at the wrapper or FFI boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

import vernier.semantic as vsem


def _make_fixture(
    rng: np.random.Generator, n_images: int, height: int, width: int
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for iid in range(n_images):
        arr = rng.integers(0, 4, size=(height, width), dtype=np.uint8)
        ignore = rng.random(size=(height, width)) < 0.05
        arr[ignore] = 255
        out[iid] = arr
    return out


@pytest.mark.parity_semantic
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_evaluate_matches_across_dtypes(dtype: type) -> None:
    rng = np.random.default_rng(0xFADE7AB1E)
    n_classes = 4
    ignore_label = 255

    gt_u8 = _make_fixture(rng, n_images=8, height=24, width=32)
    dt_u8 = _make_fixture(rng, n_images=8, height=24, width=32)
    gt = {iid: arr.astype(dtype, copy=False) for iid, arr in gt_u8.items()}
    dt = {iid: arr.astype(dtype, copy=False) for iid, arr in dt_u8.items()}

    summary = vsem.Evaluator(parity_mode="strict").evaluate(
        vsem.Dataset.from_arrays(gt, n_classes=n_classes, ignore_label=ignore_label),
        vsem.Predictions.from_arrays(dt),
    )
    assert summary.confusion_matrix.n_classes == n_classes
    # u8 is the canonical reference (no widening on the kernel walk).
    if dtype is np.uint8:
        return
    reference = vsem.Evaluator(parity_mode="strict").evaluate(
        vsem.Dataset.from_arrays(gt_u8, n_classes=n_classes, ignore_label=ignore_label),
        vsem.Predictions.from_arrays(dt_u8),
    )
    np.testing.assert_array_equal(
        summary.confusion_matrix.counts(),
        reference.confusion_matrix.counts(),
        err_msg=f"{dtype.__name__} path must match uint8 confusion matrix",
    )


@pytest.mark.parity_semantic
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_submit_matches_across_dtypes(dtype: type) -> None:
    """`BackgroundEvaluator.submit(arr)` must accept any of the three
    natural class-id widths and produce a bit-equal confusion matrix."""
    rng = np.random.default_rng(0xFADE7AB1F)
    n_classes = 4
    ignore_label = 255

    gt_u8 = _make_fixture(rng, n_images=8, height=24, width=32)
    dt_u8 = _make_fixture(rng, n_images=8, height=24, width=32)

    def run(arrays_gt: dict[int, np.ndarray], arrays_dt: dict[int, np.ndarray]) -> np.ndarray:
        evaluator = vsem.Evaluator(parity_mode="strict")
        with evaluator.background(n_classes, ignore_label=ignore_label) as bg:
            for iid in sorted(arrays_gt):
                bg.submit(iid, arrays_gt[iid], arrays_dt[iid])
            summary = bg.finalize()
        return summary.confusion_matrix.counts()

    reference = run(gt_u8, dt_u8)
    candidate = run(
        {iid: arr.astype(dtype, copy=False) for iid, arr in gt_u8.items()},
        {iid: arr.astype(dtype, copy=False) for iid, arr in dt_u8.items()},
    )
    np.testing.assert_array_equal(
        candidate,
        reference,
        err_msg=f"submit({dtype.__name__}) must match submit(uint8) confusion matrix",
    )


@pytest.mark.parity_semantic
def test_submit_rejects_float_dtype() -> None:
    rng = np.random.default_rng(0xFADE7AB20)
    arr = rng.integers(0, 4, size=(8, 8), dtype=np.uint8).astype(np.float32)
    with (
        vsem.Evaluator(parity_mode="strict").background(4, ignore_label=255) as bg,
        pytest.raises(ValueError, match="uint8/uint16/uint32"),
    ):
        bg.submit(0, arr, arr)  # pyright: ignore[reportArgumentType]
