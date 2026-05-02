"""Linux gate (ADR-0017 §G1) — non-Linux invocations must raise the
exact error message, since the documentation cross-links to it."""

from __future__ import annotations

import pytest

from bench.harness.platform_check import (
    UnsupportedPlatformError,
    _format_message,
    ensure_linux,
)


def test_linux_passes() -> None:
    """The CI / test runner is Linux; ``ensure_linux`` must be a no-op."""
    ensure_linux()


def test_non_linux_raises_with_documented_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.release", lambda: "23.4.0")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    with pytest.raises(UnsupportedPlatformError) as exc_info:
        ensure_linux()
    msg = str(exc_info.value)
    assert "vernier-bench requires Linux" in msg
    assert "Darwin 23.4.0 (arm64)" in msg
    assert "governor inspection, taskset, and perf stat" in msg


def test_format_message_is_pure() -> None:
    """The message builder is platform-independent so docs can quote it."""
    msg = _format_message("Windows", "10", "AMD64")
    assert "Windows 10 (AMD64)" in msg
    assert "vernier-bench requires Linux" in msg
