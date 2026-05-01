"""Stage timing — a context manager that records a wall-clock duration in ns.

Used by every runner to bracket the load / evaluate / accumulate /
summarize stages. The Pydantic ``StageTimings`` model is constructed
*after* the timer stops so model validation stays out of the hot path.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from bench.harness.schema import StageTimings


class StageTable:
    """A name → StageTimings collector."""

    def __init__(self) -> None:
        self._stages: dict[str, StageTimings] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - start
            self._stages[name] = StageTimings(wall_ns=elapsed)

    def record(self, name: str, wall_ns: int, notes: list[str] | None = None) -> None:
        self._stages[name] = StageTimings(wall_ns=wall_ns, notes=list(notes or []))

    def total_so_far_ns(self) -> int:
        return sum(s.wall_ns for s in self._stages.values())

    def to_dict(self) -> dict[str, StageTimings]:
        return dict(self._stages)
