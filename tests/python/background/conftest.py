"""Phase G fixtures and helpers (ADR-0014 BackgroundEvaluator).

The shard helper is a verbatim copy of
``tests/python/parity/streaming/conftest.py``'s ``shard_dt_bytes`` so this
directory stays self-contained — the streaming test tree is the source of
truth for the same partition rules; copying lets the background tests
move independently if the streaming layout changes later.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any


def shard_dt_bytes(dt_path: Path, n_shards: int, seed: int) -> list[bytes]:
    """Split DT records by image_id into ``n_shards`` disjoint payloads.

    Splitting by image_id (not by record) avoids the streaming /
    background evaluator's duplicate-image-id rejection: the same image
    can never appear in two batches.
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


def drain_until_idle(ev: Any, timeout: float = 5.0) -> None:
    """Spin until ``queue_depth`` reads 0 and counters stop changing.

    The background worker is asynchronous — ``submit()`` returns
    immediately. Tests that need to see counter state after a submit must
    wait until the worker has processed the queue. We require three
    consecutive identical readings (with ``queue_depth == 0``) before
    declaring the worker idle, which dodges the case where ``queue_depth``
    is momentarily 0 between two pending submits.
    """
    deadline = time.monotonic() + timeout
    last: tuple[int, int, int] = (-1, -1, -1)
    stable = 0
    while time.monotonic() < deadline:
        cur: tuple[int, int, int] = (
            ev.images_seen,
            ev.detections_seen,
            ev.queue_depth,
        )
        if cur == last and cur[2] == 0:
            stable += 1
            if stable >= 3:
                return
        else:
            stable = 0
        last = cur
        time.sleep(0.01)
    raise TimeoutError(f"BackgroundEvaluator did not idle within {timeout}s; last={last}")
