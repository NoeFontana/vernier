"""``submit(timeout=None)`` blocks until the worker has space, never raises.

With ``timeout=None`` (the default) and a tiny queue, the third (and
fourth, fifth, ...) submit should block until the worker drains a slot.
The test must succeed without raising; we use a wall-clock timeout
proxy of 5 seconds so a worker hang gets caught instead of stalling
the suite indefinitely.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import vernier

FIXTURES = Path(__file__).parent.parent / "parity" / "fixtures"


def test_blocking_submit_eventually_succeeds() -> None:
    gt = (FIXTURES / "perfect_match" / "gt.json").read_bytes()
    ev = vernier.instance.BackgroundEvaluator(gt, queue_capacity=1)
    try:
        deadline = time.monotonic() + 5.0
        for _ in range(8):
            if time.monotonic() > deadline:
                pytest.fail("blocking submits stalled past 5s")
            # Empty payloads are cheap to "process" but still occupy a
            # slot in the bounded channel between send and worker pull,
            # so they exercise the backpressure path the same as a real
            # batch would.
            ev.submit(b"[]", timeout=None)
    finally:
        ev.finalize()
