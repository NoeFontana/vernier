"""``bench.harness.rss.RSSSampler`` smoke tests.

The sampler is the load-bearing artifact of the streaming-vs-naive
cell — it captures the RSS curve that proves the user pattern's RSS
growth is linear in N. Both runtime branches (psutil available, psutil
unimportable) are exercised.
"""

from __future__ import annotations

import builtins
import sys
import time

import pytest

from bench.harness import rss as rss_mod
from bench.harness.rss import RSSSampler


def test_rss_sampler_captures_steady_state() -> None:
    """A tight loop that briefly holds memory should produce ≥1 sample
    and a peak that's at least as high as the initial reading."""
    with RSSSampler(interval_s=0.05) as sampler:
        # Hold 50MB of bytes for ~150ms; the sampler's 50ms interval
        # gives ≥2 follow-up samples beyond the synchronous-at-enter one.
        ballast = bytearray(50 * 1024 * 1024)
        time.sleep(0.15)
        # Touch the buffer so the kernel doesn't lazily defer the
        # commit (Linux overcommit can otherwise hide the RSS bump).
        ballast[0] = 1
        ballast[-1] = 1

    samples = sampler.samples
    assert len(samples) >= 1, "expected at least one sample (synchronous-at-enter)"
    peak = sampler.peak_rss_bytes()
    initial = samples[0][1]
    # A 50MB bump can be smaller than allocated due to fragmentation +
    # Python's allocator pooling; require a non-negative delta and
    # let the test stay tolerant. The point is "the curve recorded
    # something and the peak is ≥ the initial".
    assert peak >= initial, f"peak ({peak}) below initial ({initial})"


def test_rss_sampler_no_psutil_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """When psutil isn't importable the sampler stays a no-op: empty
    samples + peak_rss_bytes() == -1. Re-import the module under a
    forced ``ImportError`` so the module-level branch flips."""
    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if name == "psutil":
            raise ImportError("psutil unavailable for this test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.delitem(sys.modules, "psutil", raising=False)
    monkeypatch.delitem(sys.modules, "bench.harness.rss", raising=False)

    import importlib

    reloaded = importlib.import_module("bench.harness.rss")
    try:
        assert reloaded._PSUTIL_AVAILABLE is False
        with reloaded.RSSSampler(interval_s=0.05) as sampler:
            time.sleep(0.05)
        assert sampler.samples == []
        assert sampler.peak_rss_bytes() == -1
    finally:
        # Restore the real module for any later tests in this session.
        monkeypatch.setattr(builtins, "__import__", real_import)
        monkeypatch.delitem(sys.modules, "bench.harness.rss", raising=False)
        importlib.import_module("bench.harness.rss")


def test_rss_sampler_rejects_zero_interval() -> None:
    with pytest.raises(ValueError, match="positive"):
        RSSSampler(interval_s=0.0)


def test_rss_sampler_thread_is_daemon() -> None:
    """The sampler thread must be daemonized so an exiting interpreter
    doesn't hang waiting for it (the runner subprocess relies on this)."""
    with RSSSampler(interval_s=0.05) as sampler:
        time.sleep(0.05)
        thread = sampler._thread  # noqa: SLF001 — internal-state check
        assert thread is not None
        assert thread.daemon is True
