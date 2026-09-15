"""Stage timing — a context manager that records a wall-clock duration in ns.

Used by every runner to bracket the load / evaluate / accumulate /
summarize stages. The Pydantic ``StageTimings`` model is constructed
*after* the timer stops so model validation stays out of the hot path.

Each stage also records process CPU time (all threads), so
``cpu_ns / wall_ns`` exposes an impl's effective parallelism — the
evidence that a cell's CPU budget was actually respected.
"""

from __future__ import annotations

import os
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
        cpu_start = time.process_time_ns()
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - start
            cpu = time.process_time_ns() - cpu_start
            self._stages[name] = StageTimings(wall_ns=elapsed, cpu_ns=cpu)

    def record(self, name: str, wall_ns: int, notes: list[str] | None = None) -> None:
        self._stages[name] = StageTimings(wall_ns=wall_ns, notes=list(notes or []))

    def record_total(self) -> None:
        """Record the ``total`` stage as the sum of the stages so far,
        annotated with the CPU set the runner actually executed on."""
        stages = list(self._stages.values())
        cpus = ",".join(str(c) for c in sorted(os.sched_getaffinity(0)))
        self._stages["total"] = StageTimings(
            wall_ns=sum(s.wall_ns for s in stages),
            cpu_ns=sum(s.cpu_ns or 0 for s in stages),
            notes=[f"cpu_affinity={cpus}"],
        )

    def to_dict(self) -> dict[str, StageTimings]:
        return dict(self._stages)
