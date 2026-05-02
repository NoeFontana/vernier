"""Machine fingerprint and governor checks (ADR-0017 §"Result store").

The fingerprint is ``sha256(cpu_model + n_cores + total_ram + os_release +
glibc_version)`` truncated to 12 hex chars. Stable across reboots,
distinct between dev boxes, short enough to land in a result path. All
five inputs are sourced from Linux-only conventions (``/proc``,
``os.uname``, ``platform.libc_ver``); the platform gate runs first so
this module doesn't have to worry about non-Linux fallbacks.

``ensure_performance_governor`` is the release-mode pre-flight: every
online CPU's cpufreq governor must be ``performance`` or the run aborts
with a copy-pasteable ``cpupower`` fix-it line.
"""

from __future__ import annotations

import functools
import hashlib
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_PROC_CPUINFO = Path("/proc/cpuinfo")
_PROC_MEMINFO = Path("/proc/meminfo")
_CPUFREQ_GLOB = "sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
_REQUIRED_GOVERNOR = "performance"


@dataclass(frozen=True)
class MachineInputs:
    """Ordered tuple that feeds the fingerprint hash. Field order is
    load-bearing: reordering would re-bucket every existing result."""

    cpu_model: str
    n_cores: int
    total_ram_kb: int
    os_release: str
    glibc_version: str

    def hash_input(self) -> str:
        return (
            f"{self.cpu_model}|{self.n_cores}|{self.total_ram_kb}|"
            f"{self.os_release}|{self.glibc_version}"
        )


class GovernorError(RuntimeError):
    """Raised when one or more CPUs aren't on the ``performance`` governor."""


def _read_cpu_model(cpuinfo: str) -> str:
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            _, _, value = line.partition(":")
            return value.strip()
    return "unknown"


def _read_n_cores(cpuinfo: str) -> int:
    return sum(1 for line in cpuinfo.splitlines() if line.startswith("processor"))


def _read_total_ram_kb(meminfo: str) -> int:
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            match = re.search(r"(\d+)", line)
            if match:
                return int(match.group(1))
    return 0


@functools.cache
def collect_inputs() -> MachineInputs:
    """Read the five-tuple of fingerprint inputs.

    Cached for the life of the process: the inputs don't change, and the
    fingerprint determinism test calls this 100x — no reason to re-read
    ``/proc`` and re-shell ``platform.libc_ver`` on every call.
    """
    cpuinfo = _PROC_CPUINFO.read_text() if _PROC_CPUINFO.exists() else ""
    meminfo = _PROC_MEMINFO.read_text() if _PROC_MEMINFO.exists() else ""
    libc_name, libc_version = platform.libc_ver()
    glibc = libc_version or libc_name or "unknown"
    return MachineInputs(
        cpu_model=_read_cpu_model(cpuinfo),
        n_cores=_read_n_cores(cpuinfo),
        total_ram_kb=_read_total_ram_kb(meminfo),
        os_release=os.uname().release,
        glibc_version=glibc,
    )


@functools.cache
def fingerprint() -> str:
    """First 12 hex chars of sha256 over the five-tuple."""
    digest = hashlib.sha256(collect_inputs().hash_input().encode()).hexdigest()
    return digest[:12]


def _governor_paths() -> list[Path]:
    return sorted(Path("/").glob(_CPUFREQ_GLOB))


def ensure_performance_governor() -> None:
    """Raise :class:`GovernorError` if any online CPU isn't on ``performance``.

    Quiet on machines without cpufreq exposed (e.g., container hosts) —
    the assumption is that release mode runs on bare metal where the
    governor file is present, and we'd rather a missing file fall through
    to a noisy IQR than block on a config we can't inspect.
    """
    paths = _governor_paths()
    if not paths:
        return

    offenders: list[tuple[str, str]] = []
    for path in paths:
        try:
            value = path.read_text().strip()
        except OSError:
            continue
        if value != _REQUIRED_GOVERNOR:
            offenders.append((path.parent.parent.name, value))

    if offenders:
        cores = ", ".join(f"{name}={gov}" for name, gov in offenders)
        raise GovernorError(
            f"CPU governor must be {_REQUIRED_GOVERNOR!r} on every core; "
            f"found {cores}.\n"
            f"Fix: sudo cpupower frequency-set -g {_REQUIRED_GOVERNOR}"
        )


def git_sha(repo_root: Path) -> str:
    """Short git sha for the current HEAD of ``repo_root``. Empty string if
    this isn't a git checkout — the harness still runs, just under a
    placeholder bucket."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return out.decode().strip()
