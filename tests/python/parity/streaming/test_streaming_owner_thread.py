"""Single-writer rule: only the thread that first wrote may write again.

Per ADR-0013, `StreamingEvaluator` is single-writer. The first
`update()` call records the calling thread as owner; subsequent
`update()` calls from any other thread raise `RuntimeError` with
`"single-writer"` in the message. The owning thread can keep writing
indefinitely.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

import vernier

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.mark.parity
def test_streaming_evaluator_rejects_other_thread() -> None:
    gt_bytes = (FIXTURES / "perfect_match" / "gt.json").read_bytes()
    ev = vernier.instance.StreamingEvaluator(gt_bytes)

    # Establish the owning thread.
    ev.update(b"[]")

    captured: list[str] = []

    def submit_from_other_thread() -> None:
        try:
            ev.update(b"[]")
        except RuntimeError as e:
            captured.append(str(e))

    t = threading.Thread(target=submit_from_other_thread)
    t.start()
    t.join()

    assert captured, "expected the cross-thread update to raise"
    assert "single-writer" in captured[0], (
        f"expected 'single-writer' in error message, got: {captured[0]!r}"
    )

    # Owner thread can keep submitting unaffected — the rejection is
    # not a one-way kill switch.
    ev.update(b"[]")
