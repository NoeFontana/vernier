"""``BackgroundEvaluator`` end-to-end equals ``StreamingEvaluator``.

The background evaluator is a thread-pool wrapper around the streaming
evaluator (per ADR-0014); its output across any number of shards must
bit-equal the streaming evaluator processing the same shards in the
same order. We pin that against the streaming baseline (NOT the batch
evaluator) on the principle that streaming is the established baseline
in this repo, and matching it proves the wrapper preserves semantics
without re-litigating the streaming-vs-batch parity that lives in
``tests/python/parity/streaming/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from vernier._impl import StreamingEvaluator
from vernier.instance import BackgroundEvaluator

from ..parity.conftest import shard_dt_bytes

IouType = Literal["bbox", "segm", "boundary", "keypoints"]

FIXTURES = Path(__file__).parent.parent / "parity" / "fixtures"


# A representative slice of the streaming corpus: bbox + segm, mixing
# perfect-match, missing-image, and crowd cases. Full coverage already
# lives in tests/python/parity/streaming/test_streaming_finalize_equals_batch.py.
_PARITY_CASES: list[tuple[str, IouType]] = [
    ("perfect_match", "bbox"),
    ("missing_dt_image", "bbox"),
    ("crowd_region", "bbox"),
    ("perfect_match_segm", "segm"),
]


@pytest.mark.parity
@pytest.mark.parametrize(("fixture", "iou_type"), _PARITY_CASES)
@pytest.mark.parametrize("n_shards", [1, 4])
def test_background_finalize_equals_streaming(
    fixture: str, iou_type: IouType, n_shards: int
) -> None:
    gt_path = FIXTURES / fixture / "gt.json"
    dt_path = FIXTURES / fixture / "dt.json"
    gt_bytes = gt_path.read_bytes()

    shards = shard_dt_bytes(dt_path, n_shards=n_shards, seed=0xC0C0)

    streaming_ev = StreamingEvaluator(gt_bytes, iou_type=iou_type, parity_mode="strict")
    for shard in shards:
        streaming_ev.update(shard)
    streaming_summary = streaming_ev.finalize()

    bg_ev = BackgroundEvaluator(gt_bytes, iou_type=iou_type, parity_mode="strict")
    for shard in shards:
        bg_ev.submit(shard)
    bg_summary = bg_ev.finalize()

    assert len(streaming_summary.stats) == len(bg_summary.stats)
    for i, (s, b) in enumerate(zip(streaming_summary.stats, bg_summary.stats, strict=True)):
        # Background composes around the same streaming kernel; we expect
        # bit-equality. ``abs=1e-12`` swallows pure-FP noise; bump to
        # 1e-9 with a TODO if a fixture/kernel pair breaks this.
        assert b == pytest.approx(s, rel=0, abs=1e-12), (
            f"stat[{i}] diverged: streaming={s!r} background={b!r} "
            f"(fixture={fixture}, iou_type={iou_type}, n_shards={n_shards})"
        )
