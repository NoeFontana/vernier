"""``with vernier.instance.BackgroundEvaluator(...) as ev:`` cleans up on exit.

The training-loop persona: ``with`` block, simulated trainer crash
inside, evaluator must shut its worker down so we don't leak threads
across test runs.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import vernier


def test_context_manager_cleans_up_worker(fixtures_dir: Path) -> None:
    gt = (fixtures_dir / "perfect_match" / "gt.json").read_bytes()
    try:
        with vernier.instance.BackgroundEvaluator(gt) as ev:
            ev.submit(b"[]")
            raise ValueError("simulated trainer crash")
    except ValueError:
        pass

    # `__exit__` runs ``shutdown()`` synchronously, but the worker still
    # needs a moment to break out of its `recv()` and unwind its stack.
    # Give it up to 2s before declaring a leak.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        workers = [t for t in threading.enumerate() if "vernier-bg-worker" in t.name]
        if not workers:
            break
        time.sleep(0.05)
    workers = [t for t in threading.enumerate() if "vernier-bg-worker" in t.name]
    assert not workers, f"context-manager exit should have shut the worker: {workers}"
