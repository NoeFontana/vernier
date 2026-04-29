"""Phase D fixtures and helpers for `StreamingEvaluator` parity tests.

The helpers in this module shard a fixture's `dt.json` payload into
multiple disjoint batches that obey the streaming evaluator's
single-batch-per-image rule. They are pure data-shuffle utilities — the
test logic itself lives in the sibling `test_streaming_*` modules.

## Skipped per Phase D scope (follow-up tickets)

- **Order-invariance** — needs the `(score, stream_position)`
  tiebreak which is currently deferred (ADR-0013 §Determinism). Until
  the strict-mode tiebreak lands, two orderings of the same DTs can
  produce a small ULP wobble in tied-score regions, so we cannot pin
  the bit-equality contract.
- **Checkpoint round-trip** — `StreamingEvaluator.checkpoint` /
  `restore` raise `NotImplementedError` in v0; the rkyv-based
  implementation is its own ADR.
- **Crash mid-stream** — needs the Phase E (`BackgroundEvaluator`)
  worker / panic-recovery surface.
- **`snapshot_running` regression** — v0 `snapshot_running` delegates
  to `snapshot()`, so there is nothing to regress against until the
  fast-path lands.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def shard_dt_bytes(dt_path: Path, n_shards: int, seed: int) -> list[bytes]:
    """Split DT records by image_id into `n_shards` disjoint payloads.

    Splitting by image_id (not by record) avoids the
    `StreamingEvaluator`'s duplicate-image-id rejection: the same image
    can never appear in two batches.

    Empty shards (when `n_shards > number of images`) are returned as
    `b"[]"` so that callers can still issue an `update()` for them
    without special-casing.
    """
    records = json.loads(dt_path.read_text())
    by_image: dict[int, list[dict]] = {}
    for r in records:
        by_image.setdefault(r["image_id"], []).append(r)
    image_ids = sorted(by_image.keys())
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    shards: list[list[dict]] = [[] for _ in range(n_shards)]
    for i, img_id in enumerate(image_ids):
        shards[i % n_shards].extend(by_image[img_id])
    return [json.dumps(s).encode("utf-8") for s in shards]
