"""Streaming finalize equals batch evaluate.

`StreamingEvaluator.update(...)+finalize()` over any number of shards
must produce the same `Summary.stats` as
`Evaluator.evaluate(gt, dt)` over the union of those shards. We pin
this contract in `parity_mode="strict"` over the full bbox + segm
fixture corpus.

Keypoints fixtures are excluded for now: the Phase D scope brief calls
out detection-style fixtures only (the parity surface is the same, but
the Streaming/batch comparator path is straightforward to extend in a
follow-up).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from vernier._impl import StreamingEvaluator
from vernier.instance import Bbox, Evaluator, IouKind, Segm

from ..conftest import shard_dt_bytes
from ..test_parity import BBOX_FIXTURES, SEGM_FIXTURES

IouType = Literal["bbox", "segm", "boundary", "keypoints"]

FIXTURES = Path(__file__).parent.parent / "fixtures"

# `heterogeneous_dt_segm` is intentionally omitted: it is a corrected-mode
# rejection fixture (quirk J6) that raises in both batch and streaming
# paths under strict parity, so it does not exercise the streaming-vs-batch
# parity claim this module pins. Every other entry from BBOX_FIXTURES /
# SEGM_FIXTURES is in scope; new fixtures added there are picked up
# automatically.
_SEGM_FIXTURES = [f for f in SEGM_FIXTURES if f != "heterogeneous_dt_segm"]

_PARITY_CASES: list[tuple[str, IouType]] = [
    *((f, "bbox") for f in BBOX_FIXTURES),
    *((f, "segm") for f in _SEGM_FIXTURES),
]


def _iou_kernel(iou_type: IouType) -> IouKind:
    if iou_type == "bbox":
        return Bbox()
    if iou_type == "segm":
        return Segm()
    raise AssertionError(f"unhandled iou_type: {iou_type}")


@pytest.mark.parity
@pytest.mark.parametrize(("fixture", "iou_type"), _PARITY_CASES)
@pytest.mark.parametrize("n_shards", [1, 4])
def test_streaming_finalize_equals_batch(fixture: str, iou_type: IouType, n_shards: int) -> None:
    gt_path = FIXTURES / fixture / "gt.json"
    dt_path = FIXTURES / fixture / "dt.json"
    gt_bytes = gt_path.read_bytes()
    dt_bytes = dt_path.read_bytes()

    batch_summary = Evaluator(iou=_iou_kernel(iou_type), parity_mode="strict").evaluate(
        gt_bytes, dt_bytes
    )

    shards = shard_dt_bytes(dt_path, n_shards=n_shards, seed=0xC0C0)
    ev = StreamingEvaluator(gt_bytes, iou_type=iou_type, parity_mode="strict")
    for shard in shards:
        ev.update(shard)
    stream_summary = ev.finalize()

    assert len(batch_summary.stats) == len(stream_summary.stats)
    for i, (b, s) in enumerate(zip(batch_summary.stats, stream_summary.stats, strict=True)):
        # Strict-mode bit-equality is the contract; tiny `abs=1e-12`
        # absorbs ULP wobble that crops up when accumulate sees the
        # same cells in a different (k, a, i) iteration order. If this
        # later proves too tight for some kernel/shard combination,
        # bump to `abs=1e-6` and add a TODO citing
        # ADR-0013 §Determinism.
        assert s == pytest.approx(b, rel=0, abs=1e-12), (
            f"stat[{i}] diverged: batch={b!r} stream={s!r} "
            f"(fixture={fixture}, iou_type={iou_type}, n_shards={n_shards})"
        )
