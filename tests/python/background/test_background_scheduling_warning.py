"""Best-effort scheduling: bad config → exactly one ``UserWarning``.

When the worker's startup scheduling adjustment (nice / affinity) fails
— e.g. the requested CPU core doesn't exist, or the user lacks
``CAP_SYS_NICE`` for a negative nice value — the worker stamps its
``Err(...)`` outcome on the shared state. The FFI reads that once and
emits a single ``UserWarning`` mentioning "scheduling".

This test forces failure with both an out-of-range affinity AND a
nice value that requires elevated privileges, so at least one of the
two scheduling syscalls fails and the warning fires. We assert
exactly one warning so we don't regress into spamming the logs.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import vernier


def test_invalid_affinity_warns_once(fixtures_dir: Path) -> None:
    gt = (fixtures_dir / "perfect_match" / "gt.json").read_bytes()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ev = vernier.instance.BackgroundEvaluator(gt, worker_affinity=999_999, worker_nice=-20)
        ev.finalize()

    sched = [w for w in caught if "scheduling" in str(w.message).lower()]
    assert len(sched) == 1, (
        f"expected exactly one scheduling warning, got {len(sched)}: "
        f"{[str(w.message) for w in sched]}"
    )
