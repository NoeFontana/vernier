"""Process-RSS sampler — 100ms-interval polling of psutil's
``Process().memory_info().rss``.

Used by the streaming runners (B3) to capture an RSS curve while the
runner is doing work. The naive-Python baseline pattern accumulates
predictions in a list and ``cocoeval.evaluate()`` over the union — its
RSS curve grows ~linearly with image count. The vernier streaming
path's curve is bounded by the per-image working set; the difference
is what the bench cell exposes.

Lift-and-adapt of the psutil pattern at
``tests/python/integration/real_models/tide/run.py:_peak_rss_mb``.
That helper returns one scalar at call time; the sampler here keeps a
running list so callers can serialize the full curve.

When psutil isn't importable (lock-skinny environments) the sampler is
a no-op: ``samples`` returns an empty list and ``peak_rss_bytes()``
returns ``-1``. Same disposition as the upstream tide helper: the
metric is just absent, the runner still serializes.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType

# psutil is an optional runtime dep. The sampler is a no-op without it
# rather than import-time failing — every B3 runner imports this module
# at module load, but env-skin tests (no psutil) still want a clean
# import path.
try:
    import psutil as _psutil

    _PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by monkeypatch test
    _psutil = None  # type: ignore[assignment]
    _PSUTIL_AVAILABLE = False


class RSSSampler:
    """Background-thread RSS sampler. 100ms cadence by default.

    Use as a context manager::

        with RSSSampler() as s:
            ... do work ...
        peak = s.peak_rss_bytes()
        curve = s.samples  # list[(wall_s, rss_bytes)]

    The sampler thread is a ``daemon`` so the worst-case interpreter-
    exit path doesn't hang on it. ``__exit__`` joins normally.
    """

    def __init__(self, interval_s: float = 0.1) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s must be positive, got {interval_s!r}")
        self._interval_s = interval_s
        self._samples: list[tuple[float, int]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0: float | None = None

    def __enter__(self) -> "RSSSampler":
        if not _PSUTIL_AVAILABLE:
            # No-op path; thread is never started so __exit__ is also a
            # no-op. ``samples`` stays empty; ``peak_rss_bytes`` returns -1.
            return self
        self._t0 = time.perf_counter()
        # Capture the initial sample synchronously so a very short
        # context window still produces ≥1 entry.
        proc = _psutil.Process()
        self._samples.append((0.0, int(proc.memory_info().rss)))
        self._thread = threading.Thread(
            target=self._run, name="RSSSampler", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: "TracebackType | None",
    ) -> None:
        self._stop_event.set()
        if self._thread is not None:
            # The interval is 100ms; a 2× timeout is generous. We don't
            # propagate exceptions out of here — the work the sampler is
            # wrapping has already finished or raised; this is cleanup.
            self._thread.join(timeout=self._interval_s * 5)

    def _run(self) -> None:
        # ``_psutil`` is set (we wouldn't have started the thread
        # otherwise); shadowing into a local avoids one attribute lookup
        # per sample.
        proc = _psutil.Process()
        t0 = self._t0
        assert t0 is not None  # __enter__ sets this before starting the thread
        while not self._stop_event.wait(self._interval_s):
            t = time.perf_counter() - t0
            self._samples.append((t, int(proc.memory_info().rss)))

    @property
    def samples(self) -> list[tuple[float, int]]:
        """List of ``(wall_time_s, rss_bytes)`` samples in capture order.

        Returned as a fresh list so callers can mutate/serialize without
        affecting an in-flight sampler. Empty when psutil is unavailable.
        """
        return list(self._samples)

    def peak_rss_bytes(self) -> int:
        """Peak RSS observed across captured samples, in bytes.

        Returns ``-1`` when no samples were captured (psutil missing or
        sampler never entered).
        """
        if not self._samples:
            return -1
        return max(rss for _, rss in self._samples)
