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

from vernier.instance import BackgroundEvaluator, Bbox, Detections, Evaluator, QueueFullError

from ..parity.conftest import loadres_to_detections


@pytest.mark.parity
def test_submit_arrays_matches_streaming_bytes(fixtures_dir: Path) -> None:
    fixture = "missing_dt_image"
    gt_bytes = (fixtures_dir / fixture / "gt.json").read_bytes()
    dt_bytes = (fixtures_dir / fixture / "dt.json").read_bytes()
    gt_records = json.loads(gt_bytes)
    dt_records = json.loads(dt_bytes)

    # Batch JSON path as the reference: same kernel as the background
    # path, just one shot.
    s_batch = Evaluator(iou=Bbox(), parity_mode="strict").evaluate(gt_bytes, dt_bytes).stats

    bg = BackgroundEvaluator(gt_bytes, iou_type="bbox", parity_mode="strict")
    bg.submit(loadres_to_detections(gt_records, dt_records, "bbox"))
    s_bg = bg.finalize().stats

    assert len(s_batch) == len(s_bg)
    for i, (s, b) in enumerate(zip(s_batch, s_bg, strict=True)):
        assert b == pytest.approx(s, rel=0, abs=1e-12), (
            f"stat[{i}] diverged: batch={s!r} bg_arrays={b!r}"
        )


@pytest.mark.parity
def test_queue_full_error_carries_through_arrays(fixtures_dir: Path) -> None:
    """Same backpressure semantics as the JSON path. Capacity=1 + a
    blocked first send (no draining) means a later attempt with
    timeout=0 raises ``QueueFullError``.

    The first send is heavy enough to keep the worker busy while the
    later, microsecond-scale sends fill the queue, so saturation does not
    depend on scheduling. Each later send still carries a fresh
    ``image_id``: a send the worker does drain must not resubmit an
    image, which ``update()`` rejects (no silent merge).
    """
    gt_bytes = (fixtures_dir / "missing_dt_image" / "gt.json").read_bytes()
    bg = BackgroundEvaluator(
        gt_bytes,
        iou_type="bbox",
        parity_mode="strict",
        queue_capacity=1,
        # Lowest worker priority, as in the JSON-path backpressure test.
        worker_nice=19,
    )
    # A heavy first payload keeps the worker busy for tens of milliseconds
    # (index build + top-k over 300k detections), while each submit below
    # takes microseconds: the capacity-1 queue fills deterministically
    # rather than by winning a scheduling race against a fast worker.
    n_heavy = 300_000
    payload_a: Detections = {
        "image_id": 1,
        "boxes": np.tile(np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float64), (n_heavy, 1)),
        "scores": np.linspace(0.1, 0.9, n_heavy, dtype=np.float64),
        "labels": np.ones(n_heavy, dtype=np.int64),
    }

    def payload_b(image_id: int) -> Detections:
        return {
            "image_id": image_id,
            "boxes": np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float64),
            "scores": np.array([0.8], dtype=np.float64),
            "labels": np.array([1], dtype=np.int64),
        }

    bg.submit(payload_a)
    # Fresh image ids, so a send the worker does reach is never a repeat.
    raised = False
    for attempt in range(64):
        try:
            bg.submit(payload_b(2 + attempt), timeout=0.0)
        except QueueFullError:
            raised = True
            break
    bg.finalize()
    assert raised, "expected QueueFullError on a saturated queue with timeout=0"
