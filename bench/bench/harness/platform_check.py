"""Linux-only gate (ADR-0017 §G1).

The harness assumes ``/proc``, ``/sys``, ``taskset``, and ``perf`` exist;
on macOS or Windows those have no equivalents the rigor mode trusts. The
exact error message is normative — keep it in sync with ADR-0017
§"Linux-only justification".
"""

from __future__ import annotations

import platform


class UnsupportedPlatformError(RuntimeError):
    """Raised when ``vernier-bench`` is invoked on a non-Linux host."""


def _format_message(system: str, release: str, machine: str) -> str:
    return (
        "ERROR: vernier-bench requires Linux for rigor-mode execution.\n"
        f"       Detected: {system} {release} ({machine}).\n"
        "       Reason: governor inspection, taskset, and perf stat have no\n"
        "       portable equivalents on macOS that meet the harness's\n"
        "       reproducibility requirements.\n"
        '       To unblock dev-mode work on macOS, see ADR-0017 §"What this\n'
        '       ADR explicitly does not decide".'
    )


def ensure_linux() -> None:
    """Raise :class:`UnsupportedPlatformError` if ``platform.system()`` isn't Linux."""
    system = platform.system()
    if system == "Linux":
        return
    raise UnsupportedPlatformError(_format_message(system, platform.release(), platform.machine()))
