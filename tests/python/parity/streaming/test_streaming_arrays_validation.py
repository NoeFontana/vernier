"""ADR-0030 boundary contract on the array path.

Pins:

- dtype/contiguity rejection (no silent promotion).
- ``cast_inputs=True`` opt-in promotes f32 / i32 with a one-shot warning.
- per-iou_type required-field enforcement.
- the GPU rejection greppable string.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vernier.instance import Detections, StreamingEvaluator

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _gt_bbox_bytes() -> bytes:
    return (FIXTURES / "perfect_match" / "gt.json").read_bytes()


def _bbox_payload(
    *,
    dtype_boxes: Any = np.float64,
    dtype_labels: Any = np.int64,
) -> dict[str, Any]:
    """Build a one-detection bbox payload. Returns a plain dict so the
    test sites that intentionally violate the `Detections` dtype
    contract still type-check at the call site."""
    return {
        "image_id": 1,
        "boxes": np.array([[10.0, 10.0, 50.0, 50.0]], dtype=dtype_boxes),
        "scores": np.array([0.9], dtype=np.float64),
        "labels": np.array([1], dtype=dtype_labels),
    }


@pytest.mark.parity
def test_f32_boxes_rejected_with_cast_hint() -> None:
    ev = StreamingEvaluator(_gt_bbox_bytes(), iou_type="bbox")
    with pytest.raises(TypeError, match="cast_inputs=True"):
        ev.update(_bbox_payload(dtype_boxes=np.float32))  # type: ignore[arg-type]


@pytest.mark.parity
def test_i32_labels_rejected() -> None:
    ev = StreamingEvaluator(_gt_bbox_bytes(), iou_type="bbox")
    with pytest.raises(TypeError, match="expected int64"):
        ev.update(_bbox_payload(dtype_labels=np.int32))  # type: ignore[arg-type]


@pytest.mark.parity
def test_non_contiguous_boxes_rejected() -> None:
    ev = StreamingEvaluator(_gt_bbox_bytes(), iou_type="bbox")
    full = np.zeros((1, 8), dtype=np.float64)
    full[0, :4] = [10.0, 10.0, 50.0, 50.0]
    payload: dict[str, Any] = {
        "image_id": 1,
        "boxes": full[:, :4],  # strided slice; deliberately non-contiguous
        "scores": np.array([0.9], dtype=np.float64),
        "labels": np.array([1], dtype=np.int64),
    }
    with pytest.raises(TypeError, match="ascontiguousarray"):
        ev.update(payload)  # type: ignore[arg-type]


@pytest.mark.parity
def test_missing_required_field_lists_field_name() -> None:
    ev = StreamingEvaluator(_gt_bbox_bytes(), iou_type="bbox")
    incomplete = _bbox_payload()
    del incomplete["scores"]
    with pytest.raises(ValueError, match="scores"):
        ev.update(incomplete)  # type: ignore[arg-type]


@pytest.mark.parity
def test_segm_requires_rles() -> None:
    gt_segm = (FIXTURES / "perfect_match_segm" / "gt.json").read_bytes()
    ev = StreamingEvaluator(gt_segm, iou_type="segm")
    with pytest.raises(ValueError, match="rles"):
        ev.update(_bbox_payload())  # type: ignore[arg-type]


@pytest.mark.parity
def test_keypoints_requires_keypoints_field() -> None:
    gt_kp = (FIXTURES / "keypoints_perfect_match" / "gt.json").read_bytes()
    ev = StreamingEvaluator(gt_kp, iou_type="keypoints")
    with pytest.raises(ValueError, match="keypoints"):
        ev.update(_bbox_payload())  # type: ignore[arg-type]


@pytest.mark.parity
def test_cast_inputs_true_accepts_f32_and_emits_one_warning() -> None:
    """The one-shot latch: many casts on the same evaluator emit at most
    one warning. Each `update` hands a *different* image_id so the
    streaming-side duplicate-image guard does not trip — the test is
    about the cast latch alone.
    """
    # Use a 2-image GT so we can submit two batches without tripping the
    # duplicate-image_id guard.
    gt_bytes = (FIXTURES / "missing_dt_image" / "gt.json").read_bytes()
    ev = StreamingEvaluator(gt_bytes, iou_type="bbox", cast_inputs=True)
    payload_a: dict[str, Any] = {
        "image_id": 1,
        "boxes": np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32),
        "scores": np.array([0.9], dtype=np.float64),
        "labels": np.array([1], dtype=np.int64),
    }
    payload_b: dict[str, Any] = {
        "image_id": 2,
        "boxes": np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32),
        "scores": np.array([0.8], dtype=np.float64),
        "labels": np.array([1], dtype=np.int32),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ev.update(payload_a)  # type: ignore[arg-type]
        ev.update(payload_b)  # type: ignore[arg-type]
    msgs = [str(w.message) for w in caught if "cast_inputs" in str(w.message)]
    assert len(msgs) == 1, f"expected exactly one cast warning, got {msgs!r}"


class _FakeGPUArray:
    """Stand-in for a torch CUDA tensor: reports ``device_type=2``
    (kCUDA) via ``__dlpack_device__`` so the rejection path fires
    without needing a real GPU.
    """

    def __dlpack_device__(self) -> tuple[int, int]:
        return (2, 0)

    def __dlpack__(self, *_: object, **__: object) -> object:
        raise AssertionError("__dlpack__ should not be called once device is rejected")


@pytest.mark.parity
def test_gpu_dlpack_rejected_with_greppable_message() -> None:
    ev = StreamingEvaluator(_gt_bbox_bytes(), iou_type="bbox")
    # `Detections.boxes` is typed as `NDArray[float64]`; the FakeGPUArray
    # stand-in covers the `__dlpack__` protocol so the FFI dispatcher
    # receives the rejection at the device-screen step. The payload is
    # deliberately untyped so this test isn't accidentally weakened by
    # the type system erasing the GPU-handle stand-in.
    payload: dict[str, Any] = {
        "image_id": 1,
        "boxes": _FakeGPUArray(),
        "scores": np.array([0.9], dtype=np.float64),
        "labels": np.array([1], dtype=np.int64),
    }
    with pytest.raises(TypeError, match="vernier-0030 does not accept GPU-resident detections"):
        ev.update(payload)  # type: ignore[arg-type]


@pytest.mark.parity
def test_legacy_bytes_path_still_works() -> None:
    """Regression check: routing through `DetectionsArg::extract` must
    leave the bytes path bit-identical."""
    gt = _gt_bbox_bytes()
    dt = (FIXTURES / "perfect_match" / "dt.json").read_bytes()
    ev1 = StreamingEvaluator(gt, iou_type="bbox", parity_mode="strict")
    ev1.update(dt)
    s1 = ev1.finalize().stats

    payload: Detections = {
        "image_id": 1,
        "boxes": np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float64),
        "scores": np.array([0.9], dtype=np.float64),
        "labels": np.array([1], dtype=np.int64),
    }
    ev2 = StreamingEvaluator(gt, iou_type="bbox", parity_mode="strict")
    ev2.update(payload)
    s2 = ev2.finalize().stats
    assert s1 == pytest.approx(s2, rel=0, abs=1e-12)
