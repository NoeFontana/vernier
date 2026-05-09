"""ADR-0030 amendment (2026-05-09): validation contracts for the
broadened ``Detections.rles`` ingest.

Pins the rejection paths for the two new forms:

- bitmask: dtype must be ``bool`` or ``uint8``; ndim must be 2; GPU
  tensors rejected with the existing ``vernier-0030`` greppable string.
- compressed dict: ``counts`` must be valid UTF-8 bytes; ``size`` is
  still required.
- top-level dispatch error names all three accepted forms when an item
  is none of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from vernier.instance import BackgroundEvaluator, Detections

from .test_streaming_arrays_validation import _FakeGPUArray

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _gt_segm_bytes() -> bytes:
    return (FIXTURES / "perfect_match_segm" / "gt.json").read_bytes()


def _segm_payload(rles: list[Any]) -> Detections:
    """One-detection segm payload with a configurable ``rles`` field."""
    return cast(
        Detections,
        {
            "image_id": 1,
            "boxes": np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float64),
            "scores": np.array([0.9], dtype=np.float64),
            "labels": np.array([1], dtype=np.int64),
            "rles": rles,
        },
    )


@pytest.mark.parity
def test_bitmask_wrong_dtype_rejected() -> None:
    ev = BackgroundEvaluator(_gt_segm_bytes(), iou_type="segm")
    bad = np.zeros((100, 100), dtype=np.float32)
    with pytest.raises(TypeError, match="bool or uint8"):
        ev.submit(_segm_payload([bad]))


@pytest.mark.parity
def test_bitmask_wrong_ndim_rejected() -> None:
    ev = BackgroundEvaluator(_gt_segm_bytes(), iou_type="segm")
    bad_3d = np.zeros((4, 100, 100), dtype=np.uint8)
    with pytest.raises((TypeError, ValueError), match=r"2-D|2D|ndim"):
        ev.submit(_segm_payload([bad_3d]))


@pytest.mark.parity
def test_bitmask_non_contiguous_rejected() -> None:
    """A sliced 3-D tensor produces a 2-D view that is neither C- nor
    F-contiguous; the bitmask path must reject it with a clear hint."""
    ev = BackgroundEvaluator(_gt_segm_bytes(), iou_type="segm")
    full = np.zeros((4, 100, 100), dtype=np.uint8)
    sliced = full[0, ::2, :]  # strided
    assert not sliced.flags["C_CONTIGUOUS"]
    assert not sliced.flags["F_CONTIGUOUS"]
    with pytest.raises(TypeError, match="contiguous"):
        ev.submit(_segm_payload([sliced]))


@pytest.mark.parity
def test_compressed_rle_non_utf8_bytes_rejected() -> None:
    ev = BackgroundEvaluator(_gt_segm_bytes(), iou_type="segm")
    bad = {"counts": b"\xff\xfe\xfd", "size": (100, 100)}
    with pytest.raises(ValueError, match="UTF-8"):
        ev.submit(_segm_payload([bad]))


@pytest.mark.parity
def test_compressed_rle_missing_size_rejected() -> None:
    ev = BackgroundEvaluator(_gt_segm_bytes(), iou_type="segm")
    bad = {"counts": b"PPYo`0"}
    with pytest.raises(ValueError, match="size"):
        ev.submit(_segm_payload([bad]))


@pytest.mark.parity
def test_bitmask_gpu_dlpack_rejected() -> None:
    """The existing GPU device screen at the DLPack boundary fires
    regardless of which form the bitmask path was attempted as."""
    ev = BackgroundEvaluator(_gt_segm_bytes(), iou_type="segm")
    with pytest.raises(TypeError, match="vernier-0030 does not accept GPU-resident detections"):
        ev.submit(_segm_payload([_FakeGPUArray()]))


@pytest.mark.parity
def test_dispatch_error_lists_accepted_forms() -> None:
    """An item that is neither a dict nor a 2-D array (e.g. a bare int)
    hits the top-level dispatch with a message naming the accepted forms."""
    ev = BackgroundEvaluator(_gt_segm_bytes(), iou_type="segm")
    with pytest.raises(TypeError, match=r"RLE dict|bitmask|2-D"):
        ev.submit(_segm_payload([42]))
