"""Stage timing — a context manager that records a wall-clock duration in ns.

Used by every runner to bracket the load / evaluate / accumulate /
summarize stages. The Pydantic ``StageTimings`` model is constructed
*after* the timer stops so model validation stays out of the hot path.

Each stage also records:

- process CPU time (all threads), so ``cpu_ns / wall_ns`` exposes an
  impl's effective parallelism — the evidence that a cell's CPU budget
  was actually respected;
- RSS at stage start and the exact RSS high-water mark during the
  stage. The kernel's ``VmHWM`` is reset through
  ``/proc/self/clear_refs`` before each stage, so the peak is exact
  rather than sampled, and per-stage rather than process-lifetime.

The ``/proc`` reads and the reset happen outside the timed span.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from bench.harness.schema import StageTimings

_STATUS = Path("/proc/self/status")
_CLEAR_REFS = Path("/proc/self/clear_refs")
_KB_FIELD = re.compile(r"^(VmRSS|VmHWM):\s+(\d+)\s+kB$", re.MULTILINE)


def _rss_fields() -> dict[str, int]:
    """``{"VmRSS": bytes, "VmHWM": bytes}``; empty where ``/proc`` is unavailable."""
    try:
        text = _STATUS.read_text()
    except OSError:
        return {}
    return {name: int(kb) * 1024 for name, kb in _KB_FIELD.findall(text)}


def _reset_peak_rss() -> bool:
    """Reset ``VmHWM`` to the current RSS (``echo 5 > clear_refs``)."""
    try:
        _CLEAR_REFS.write_text("5")
    except OSError:
        return False
    return True


class StageTable:
    """A name → StageTimings collector."""

    def __init__(self) -> None:
        self._stages: dict[str, StageTimings] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        peak_reset = _reset_peak_rss()
        rss_start = _rss_fields().get("VmRSS")
        cpu_start = time.process_time_ns()
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - start
            cpu = time.process_time_ns() - cpu_start
            # Without a reset, VmHWM is the process-lifetime peak and
            # would misattribute import-time spikes to this stage.
            peak = _rss_fields().get("VmHWM") if peak_reset else None
            self._stages[name] = StageTimings(
                wall_ns=elapsed, cpu_ns=cpu, rss_start_bytes=rss_start, peak_rss_bytes=peak
            )

    def record(self, name: str, wall_ns: int, notes: list[str] | None = None) -> None:
        self._stages[name] = StageTimings(wall_ns=wall_ns, notes=list(notes or []))

    def record_total(self) -> None:
        """Record the ``total`` stage as the aggregate of the stages so
        far — summed wall/CPU time, the first stage's starting RSS, the
        highest per-stage peak — annotated with the CPU set the runner
        actually executed on."""
        stages = list(self._stages.values())
        peaks = [s.peak_rss_bytes for s in stages if s.peak_rss_bytes is not None]
        cpus = ",".join(str(c) for c in sorted(os.sched_getaffinity(0)))
        self._stages["total"] = StageTimings(
            wall_ns=sum(s.wall_ns for s in stages),
            cpu_ns=sum(s.cpu_ns or 0 for s in stages),
            rss_start_bytes=stages[0].rss_start_bytes if stages else None,
            peak_rss_bytes=max(peaks) if peaks else None,
            notes=[f"cpu_affinity={cpus}"],
        )

    def to_dict(self) -> dict[str, StageTimings]:
        return dict(self._stages)
