"""``BackgroundEvaluator.snapshot(peek=True)`` — placeholder for v0.

Per ADR-0013 §"Fast snapshot mode", ``snapshot_running`` (the cheaper,
possibly-stale snapshot path) currently delegates to the regular
``snapshot()`` — the optimization is deferred. The test is structured
but module-level ``pytest.skip`` is in force; the next iteration that
implements the fast path flips this on.
"""

from __future__ import annotations

import pytest

pytest.skip(
    "ADR-0013 §Fast snapshot mode: peek delegates to snapshot in v0",
    allow_module_level=True,
)
