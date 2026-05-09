"""ADR-0030 array path on ``BackgroundEvaluator.submit``.

Mirrors the streaming array-vs-JSON oracle on the background surface:
both ingest forms must produce byte-identical ``Summary.stats``. Also
pins that ``QueueFullError`` semantics carry through the array path
unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vernier._impl import StreamingEvaluator
from vernier.instance import BackgroundEvaluator, Detections, QueueFullError

from ..parity.conftest import loadres_to_detections

FIXTURES = Path(__file__).parent.parent / "parity" / "fixtures"


@pytest.mark.parity
def test_submit_arrays_matches_streaming_bytes() -> None:
    fixture = "missing_dt_image"
    gt_bytes = (FIXTURES / fixture / "gt.json").read_bytes()
    dt_bytes = (FIXTURES / fixture / "dt.json").read_bytes()
    gt_records = json.loads(gt_bytes)
    dt_records = json.loads(dt_bytes)

    # Streaming JSON path as the reference (per the existing async-equals-
    # sync convention in test_background_async_equals_sync.py).
    streaming = StreamingEvaluator(gt_bytes, iou_type="bbox", parity_mode="strict")
    streaming.update(dt_bytes)
    s_stream = streaming.finalize().stats

    bg = BackgroundEvaluator(gt_bytes, iou_type="bbox", parity_mode="strict")
    bg.submit(loadres_to_detections(gt_records, dt_records, "bbox"))
    s_bg = bg.finalize().stats

    assert len(s_stream) == len(s_bg)
    for i, (s, b) in enumerate(zip(s_stream, s_bg, strict=True)):
        assert b == pytest.approx(s, rel=0, abs=1e-12), (
            f"stat[{i}] diverged: streaming={s!r} bg_arrays={b!r}"
        )


@pytest.mark.parity
def test_queue_full_error_carries_through_arrays() -> None:
    """Same backpressure semantics as the JSON path. Capacity=1 + a
    blocked first send (no draining) means the second attempt with
    timeout=0 raises ``QueueFullError``.
    """
    gt_bytes = (FIXTURES / "missing_dt_image" / "gt.json").read_bytes()
    bg = BackgroundEvaluator(
        gt_bytes,
        iou_type="bbox",
        parity_mode="strict",
        queue_capacity=1,
    )
    payload_a: Detections = {
        "image_id": 1,
        "boxes": np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float64),
        "scores": np.array([0.9], dtype=np.float64),
        "labels": np.array([1], dtype=np.int64),
    }
    payload_b: Detections = {
        "image_id": 2,
        "boxes": np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float64),
        "scores": np.array([0.8], dtype=np.float64),
        "labels": np.array([1], dtype=np.int64),
    }
    bg.submit(payload_a)
    # Drive the queue full in the worst-case ordering: many quick sends
    # each with timeout=0; a fixed-cardinality loop is enough to expose
    # backpressure deterministically without resorting to sleep games.
    raised = False
    for _ in range(64):
        try:
            bg.submit(payload_b, timeout=0.0)
        except QueueFullError:
            raised = True
            break
    bg.finalize()
    assert raised, "expected QueueFullError on a saturated queue with timeout=0"
