"""``submit(timeout=0.0)`` raises a structured ``QueueFullError`` on saturation.

The ``timeout=0.0`` (try-send) backpressure mode is documented to raise
``vernier.instance.QueueFullError`` immediately if the channel has no space. The
exception carries the configured ``queue_capacity`` and the request's
``timeout`` so callers can build retry policy on top.

Forcing the queue full is racy: even with ``queue_capacity=1`` the
worker may pull a message between two submits. We loop ``submit(timeout=0.0)``
up to 50 times — under any reasonable scheduler the queue saturates well
before the loop runs out — and assert the structured exception fires at
least once.
"""

from __future__ import annotations

from pathlib import Path

import vernier

FIXTURES = Path(__file__).parent.parent / "parity" / "fixtures"


def test_queue_full_raises_with_structured_attrs() -> None:
    gt = (FIXTURES / "perfect_match" / "gt.json").read_bytes()
    # `worker_nice=19` (lowest priority) gives the calling thread a
    # better chance at filling the queue before the worker drains it.
    ev = vernier.instance.BackgroundEvaluator(gt, queue_capacity=1, worker_nice=19)
    captured: vernier.instance.QueueFullError | None = None
    try:
        for _ in range(50):
            try:
                ev.submit(b"[]", timeout=0.0)
            except vernier.instance.QueueFullError as e:
                captured = e
                break
    finally:
        ev.finalize()

    assert captured is not None, "expected QueueFullError under saturation"
    # Inspect the structured attrs outside the `except` block so ruff's
    # PT017 (no-asserts-in-except) stays happy and the failure message
    # is plain.
    assert captured.queue_capacity == 1, (
        f"expected queue_capacity=1 on QueueFullError, got {captured.queue_capacity}"
    )
    assert captured.timeout == 0.0, (
        f"expected timeout=0.0 on QueueFullError, got {captured.timeout}"
    )
