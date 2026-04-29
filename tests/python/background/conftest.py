"""Phase G fixtures and helpers (ADR-0014 BackgroundEvaluator).

The shard helper lives in `tests/python/parity/conftest.py`; this
module exports the worker-quiescence helper that is unique to the
background suite.
"""

from __future__ import annotations

import time
from typing import Any


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
