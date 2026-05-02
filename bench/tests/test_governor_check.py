"""Governor pre-flight (ADR-0017 §"Run modes") — performance is required
on every online core in release mode.

Real ``/sys`` reads are platform state-dependent; the test mocks the
glob+read path so unit tests don't depend on the runner's CPU governor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.harness import machine
from bench.harness.machine import GovernorError


def _stage_governor_files(tmp_path: Path, governors: dict[str, str]) -> Path:
    """Write fake ``cpufreq/scaling_governor`` files; return the prefix."""
    for name, value in governors.items():
        cpu = tmp_path / "sys" / "devices" / "system" / "cpu" / name / "cpufreq"
        cpu.mkdir(parents=True, exist_ok=True)
        (cpu / "scaling_governor").write_text(f"{value}\n")
    return tmp_path


def test_passes_when_every_core_is_performance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _stage_governor_files(tmp_path, {"cpu0": "performance", "cpu1": "performance"})
    monkeypatch.setattr(
        machine,
        "_governor_paths",
        lambda: sorted(root.glob("sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")),
    )
    machine.ensure_performance_governor()


def test_fails_when_any_core_is_powersave(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _stage_governor_files(tmp_path, {"cpu0": "performance", "cpu1": "powersave"})
    monkeypatch.setattr(
        machine,
        "_governor_paths",
        lambda: sorted(root.glob("sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")),
    )
    with pytest.raises(GovernorError) as exc_info:
        machine.ensure_performance_governor()
    msg = str(exc_info.value)
    assert "cpu1=powersave" in msg
    assert "cpupower frequency-set -g performance" in msg


def test_quiet_when_cpufreq_not_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Container hosts often hide ``cpufreq``; we'd rather fall through
    to the IQR gate than refuse to run on a config we can't inspect."""
    monkeypatch.setattr(machine, "_governor_paths", lambda: [])
    machine.ensure_performance_governor()
