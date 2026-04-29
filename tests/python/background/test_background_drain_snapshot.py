"""``BackgroundEvaluator.snapshot()`` after partial submit equals streaming.

Submit half the fixture's shards via background, ``drain_until_idle`` to
let the worker catch up, then ``snapshot()`` and compare to a streaming
evaluator processing the same half. Bit-equal in strict mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

import vernier

from ..parity.conftest import shard_dt_bytes
from .conftest import drain_until_idle

IouType = Literal["bbox", "segm", "boundary", "keypoints"]

FIXTURES = Path(__file__).parent.parent / "parity" / "fixtures"

# Same coverage scheme as test_background_async_equals_sync.py:
# representative slice; streaming has the broader corpus.
_SNAPSHOT_CASES: list[tuple[str, IouType]] = [
    ("perfect_match", "bbox"),
    ("missing_dt_image", "bbox"),
    ("crowd_region", "bbox"),
    ("perfect_match_segm", "segm"),
]


@pytest.mark.parity
@pytest.mark.parametrize(("fixture", "iou_type"), _SNAPSHOT_CASES)
def test_background_snapshot_matches_streaming_on_prefix(fixture: str, iou_type: IouType) -> None:
    gt_path = FIXTURES / fixture / "gt.json"
    dt_path = FIXTURES / fixture / "dt.json"
    gt_bytes = gt_path.read_bytes()

    # 2-shard partition gives us a well-defined "first half" — shard 0.
    shards = shard_dt_bytes(dt_path, n_shards=2, seed=0xBAB1)
    prefix_shards = shards[: max(1, len(shards) // 2)]

    streaming_ev = vernier.StreamingEvaluator(gt_bytes, iou_type=iou_type, parity_mode="strict")
    for s in prefix_shards:
        streaming_ev.update(s)
    streaming_snap = streaming_ev.snapshot()

    bg_ev = vernier.BackgroundEvaluator(gt_bytes, iou_type=iou_type, parity_mode="strict")
    for s in prefix_shards:
        bg_ev.submit(s)
    drain_until_idle(bg_ev)
    bg_snap = bg_ev.snapshot()

    assert len(streaming_snap.stats) == len(bg_snap.stats)
    for i, (s, b) in enumerate(zip(streaming_snap.stats, bg_snap.stats, strict=True)):
        assert b == pytest.approx(s, rel=0, abs=1e-12), (
            f"snapshot stat[{i}] diverged: streaming={s!r} background={b!r}"
        )

    bg_ev.finalize()
