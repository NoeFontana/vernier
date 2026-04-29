"""Pytest configuration for parity tests.

The parity suite double-runs the reference (pycocotools 2.0.8) and the
candidate (vernier) on the same fixtures and asserts every intermediate
matches. Today the candidate is a shim that delegates to pycocotools, so the
suite is a tautology — but the harness, fixture corpus, and CI plumbing are
real. As Rust evaluator pieces ship, the shim is replaced and the suite
becomes a load-bearing parity gate.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


def shard_dt_bytes(dt_path: Path, n_shards: int, seed: int) -> list[bytes]:
    """Split DT records by image_id into ``n_shards`` disjoint payloads.

    Splitting by image_id (not by record) avoids the streaming /
    background evaluator's duplicate-image-id rejection: the same image
    can never appear in two batches. Empty shards (when ``n_shards`` >
    number of images) are returned as ``b"[]"`` so callers can still
    issue an ``update()`` for them without special-casing.
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
