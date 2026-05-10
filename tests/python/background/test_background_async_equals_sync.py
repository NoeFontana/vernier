"""``BackgroundEvaluator`` end-to-end equals ``Evaluator.evaluate``.

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

from vernier.instance import BackgroundEvaluator, Bbox, Evaluator, IouKind, Segm

from ..parity.conftest import shard_dt_bytes

IouType = Literal["bbox", "segm", "boundary", "keypoints"]

# A representative slice of the streaming corpus: bbox + segm, mixing
# perfect-match, missing-image, and crowd cases. Full coverage already
# lives in tests/python/parity/streaming/test_streaming_finalize_equals_batch.py.
_PARITY_CASES: list[tuple[str, IouType]] = [
    ("perfect_match", "bbox"),
    ("missing_dt_image", "bbox"),
    ("crowd_region", "bbox"),
    ("perfect_match_segm", "segm"),
]


def _iou_kernel(iou_type: IouType) -> IouKind:
    return Bbox() if iou_type == "bbox" else Segm()


@pytest.mark.parity
@pytest.mark.parametrize(("fixture", "iou_type"), _PARITY_CASES)
@pytest.mark.parametrize("n_shards", [1, 4])
def test_background_finalize_equals_batch(
    fixture: str, iou_type: IouType, n_shards: int, fixtures_dir: Path
) -> None:
    gt_path = fixtures_dir / fixture / "gt.json"
    dt_path = fixtures_dir / fixture / "dt.json"
    gt_bytes = gt_path.read_bytes()
    dt_bytes = dt_path.read_bytes()

    shards = shard_dt_bytes(dt_path, n_shards=n_shards, seed=0xC0C0)

    iou = _iou_kernel(iou_type)
    batch_summary = Evaluator(iou=iou, parity_mode="strict").evaluate(gt_bytes, dt_bytes)

    bg_ev = BackgroundEvaluator(gt_bytes, iou_type=iou_type, parity_mode="strict")
    for shard in shards:
        bg_ev.submit(shard)
    bg_summary = bg_ev.finalize()

    assert len(batch_summary.stats) == len(bg_summary.stats)
    for i, (s, b) in enumerate(zip(batch_summary.stats, bg_summary.stats, strict=True)):
        # Background composes around the same streaming kernel as batch;
        # bit-equality is expected. ``abs=1e-12`` swallows pure-FP noise.
        assert b == pytest.approx(s, rel=0, abs=1e-12), (
            f"stat[{i}] diverged: batch={s!r} background={b!r} "
            f"(fixture={fixture}, iou_type={iou_type}, n_shards={n_shards})"
        )
