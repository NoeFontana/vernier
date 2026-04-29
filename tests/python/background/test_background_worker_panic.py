"""Worker panic recovery — gated on the ``test-poison`` Cargo feature.

When the FFI is built with ``--features test-poison`` the
``BackgroundEvaluator`` exposes a hidden ``_inject_poison_for_tests``
that posts a ``WorkerMessage::Poison`` to the worker. The worker
panics on receipt; ``JoinHandle::join`` returns ``Err(payload)`` and
the FFI surfaces a ``RuntimeError`` to the caller.

If the wheel under test was built without the feature, the hidden
method is absent and the entire module skips.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import vernier
import vernier._core

TEST_POISON = hasattr(vernier._core.BackgroundEvaluator, "_inject_poison_for_tests")
pytestmark = pytest.mark.skipif(
    not TEST_POISON,
    reason="requires --features test-poison build of vernier-ffi",
)

FIXTURES = Path(__file__).parent.parent / "parity" / "fixtures"


def test_worker_panic_surfaces_as_runtime_error() -> None:
    gt = (FIXTURES / "perfect_match" / "gt.json").read_bytes()
    ev = vernier.BackgroundEvaluator(gt)
    # Inject the poison message; the worker pulls it off the channel and
    # panics. The channel is then closed (sender side is fine; the worker
    # holding the receiver is gone).
    ev._inject_poison_for_tests()  # type: ignore[attr-defined]

    # Give the worker a moment to actually take the poison off the queue
    # and panic. Without this brief wait the next submit may race and
    # land before the channel disconnects.
    deadline = time.monotonic() + 2.0
    raised: Exception | None = None
    while time.monotonic() < deadline:
        try:
            # Either submit, snapshot, or finalize must surface the
            # panic. We try finalize first because it joins the worker
            # handle and is guaranteed to observe a panic payload.
            ev.finalize()
        except RuntimeError as e:
            raised = e
            break
        except Exception as e:  # pragma: no cover — diagnostic
            raised = e
            break
        time.sleep(0.05)

    assert raised is not None, "expected an exception after worker poisoned"
    msg = str(raised).lower()
    assert "panic" in msg or "poison" in msg or "no longer" in msg, (
        f"expected panic/poison-related error, got: {raised!r}"
    )

    # The worker thread must be gone. We give it a short grace period so
    # the OS can finish reaping a freshly-panicked thread.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        workers = [t for t in threading.enumerate() if "vernier-bg-worker" in t.name]
        if not workers:
            break
        time.sleep(0.05)
    workers = [t for t in threading.enumerate() if "vernier-bg-worker" in t.name]
    assert not workers, f"worker thread should have exited after panic, still present: {workers}"
