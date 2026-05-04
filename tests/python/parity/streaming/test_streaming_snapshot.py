"""Streaming snapshot equals batch evaluate-of-prefix.

After submitting a subset of the fixture's DTs, `snapshot()` must
match `Evaluator.evaluate(gt, subset)` on the same subset.

The "subset" is defined by the same image-id sharding the streaming
contract requires (no detection for an image submitted in shard A may
appear in shard B). We submit shards 0 .. n//2-1 of an n-shard split
and compare against the batch evaluator run over the union of those
same shards' records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

from vernier.instance import Bbox, Evaluator, IouKind, Segm, StreamingEvaluator

from ..conftest import shard_dt_bytes

FIXTURES = Path(__file__).parent.parent / "fixtures"

IouType = Literal["bbox", "segm", "boundary", "keypoints"]


_SNAPSHOT_CASES: list[tuple[str, IouType]] = [
    # Single-image fixtures collapse into "snapshot all of it"; we still
    # exercise them to pin that snapshot is callable. The behaviour
    # difference shows up on `missing_dt_image` (2 GT images, ~half).
    ("perfect_match", "bbox"),
    ("zero_overlap", "bbox"),
    ("crowd_region", "bbox"),
    ("missing_dt_image", "bbox"),
    ("score_ties", "bbox"),
    ("perfect_match_segm", "segm"),
    ("missing_dt_image_segm", "segm"),
]


def _iou_kernel(iou_type: IouType) -> IouKind:
    if iou_type == "bbox":
        return Bbox()
    if iou_type == "segm":
        return Segm()
    raise AssertionError(f"unhandled iou_type: {iou_type}")


def _concat_shards(shards: list[bytes]) -> bytes:
    """Merge a list of JSON-array payloads back into one JSON array."""
    merged: list[dict] = []
    for s in shards:
        merged.extend(json.loads(s))
    return json.dumps(merged).encode("utf-8")


@pytest.mark.parity
@pytest.mark.parametrize(("fixture", "iou_type"), _SNAPSHOT_CASES)
def test_streaming_snapshot_matches_batch_on_prefix(fixture: str, iou_type: IouType) -> None:
    gt_path = FIXTURES / fixture / "gt.json"
    dt_path = FIXTURES / fixture / "dt.json"
    gt_bytes = gt_path.read_bytes()

    # Use 2-shard partition so that "first half" is well-defined: the
    # first shard is exactly the prefix we feed to streaming, and the
    # batch comparison uses the same shard concatenated back to JSON.
    shards = shard_dt_bytes(dt_path, n_shards=2, seed=0xBAB1)
    prefix_shards = shards[: max(1, len(shards) // 2)]
    prefix_dt = _concat_shards(prefix_shards)

    batch_summary = Evaluator(iou=_iou_kernel(iou_type), parity_mode="strict").evaluate(
        gt_bytes, prefix_dt
    )

    ev = StreamingEvaluator(gt_bytes, iou_type=iou_type, parity_mode="strict")
    for s in prefix_shards:
        ev.update(s)
    snap = ev.snapshot()

    assert len(batch_summary.stats) == len(snap.stats)
    for i, (b, s) in enumerate(zip(batch_summary.stats, snap.stats, strict=True)):
        assert s == pytest.approx(b, rel=0, abs=1e-12), (
            f"snapshot stat[{i}] diverged: batch={b!r} streaming={s!r}"
        )
