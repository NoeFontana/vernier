"""Linux-only gate (ADR-0017 §G1) + streaming-cell sensitivity warnings
(ADR-0033 §"Streaming hardening").

The harness assumes ``/proc``, ``/sys``, ``taskset``, and ``perf`` exist;
on macOS or Windows those have no equivalents the rigor mode trusts. The
exact error message is normative — keep it in sync with ADR-0017
§"Linux-only justification".

Streaming cells (B3) measure peak RSS and per-image latency CDFs; these
are sensitive to two kernel knobs that have no effect on AP-fold-bound
detection cells:

- **transparent hugepages** — ``always`` policies cause large
  allocations to coalesce, perturbing the per-second RSS sampler.
- **swappiness** — high values let the kernel page out cold pages
  during a long-running stream, which inflates peak RSS spuriously.

These warnings only fire when the streaming paradigm is being
benchmarked; instance / panoptic / semantic cells are unaffected.
"""

from __future__ import annotations

import platform
from pathlib import Path

from bench.harness.schema import Paradigm

# THP enable file. ``[never]`` is the recommended setting for benches
# that measure RSS curves; ``always`` (the default on most distros) is
# the canonical perturber. The brackets indicate the *active* policy.
_THP_ENABLED_PATH: Path = Path("/sys/kernel/mm/transparent_hugepage/enabled")
_THP_PREFERRED_ACTIVE: str = "never"

# Swappiness file. The kernel default is 60 on most distros and matches
# what most benches see; non-default values warrant a heads-up because
# they shift the RSS curve under load.
_SWAPPINESS_PATH: Path = Path("/proc/sys/vm/swappiness")
_DEFAULT_SWAPPINESS: int = 60


class UnsupportedPlatformError(RuntimeError):
    """Raised when ``vernier-bench`` is invoked on a non-Linux host."""


class StreamingPlatformWarning(UserWarning):
    """Soft warning raised when a streaming bench runs against a kernel
    whose THP / swappiness settings are likely to perturb RSS curves.

    Used as a discriminator for tests; the runtime path emits the
    warning via :mod:`warnings` so users can filter or upgrade per
    site policy.
    """


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


def _read_thp_active(path: Path = _THP_ENABLED_PATH) -> str | None:
    """Parse ``/sys/kernel/mm/transparent_hugepage/enabled`` content.

    The file format is space-separated policy names with the active
    one wrapped in brackets, e.g. ``always madvise [never]``. Returns
    the active policy (without brackets), or ``None`` if the file is
    missing or unparseable.
    """
    try:
        text = path.read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    for token in text.split():
        if token.startswith("[") and token.endswith("]"):
            return token[1:-1]
    return None


def _read_swappiness(path: Path = _SWAPPINESS_PATH) -> int | None:
    """Parse ``/proc/sys/vm/swappiness`` (one int per line).

    Returns ``None`` if the file is missing or unparseable.
    """
    try:
        text = path.read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    try:
        return int(text)
    except ValueError:
        return None


def streaming_sensitivity_warnings(
    paradigm: Paradigm,
    *,
    thp_path: Path = _THP_ENABLED_PATH,
    swappiness_path: Path = _SWAPPINESS_PATH,
) -> list[str]:
    """Return human-readable warnings for non-default knobs that affect
    streaming-cell measurements. Empty list for non-streaming paradigms.

    Returned messages are also emitted via :mod:`warnings` with category
    :class:`StreamingPlatformWarning` — callers that want both display
    formats (CLI echo + Python warning machinery) can iterate the
    return value.
    """
    if paradigm != "streaming":
        return []

    messages: list[str] = []

    thp = _read_thp_active(thp_path)
    if thp is not None and thp != _THP_PREFERRED_ACTIVE:
        messages.append(
            f"transparent_hugepage policy is {thp!r}; "
            f"streaming RSS curves are sensitive — consider "
            f'`echo {_THP_PREFERRED_ACTIVE} > {thp_path}` for stable measurements.'
        )

    swappiness = _read_swappiness(swappiness_path)
    if swappiness is not None and swappiness != _DEFAULT_SWAPPINESS:
        messages.append(
            f"vm.swappiness is {swappiness} (kernel default is {_DEFAULT_SWAPPINESS}); "
            f"streaming peak-RSS measurements may include cold-page eviction artifacts."
        )

    if messages:
        import warnings

        for msg in messages:
            warnings.warn(msg, StreamingPlatformWarning, stacklevel=2)

    return messages
