"""Streaming-cell sensitivity warnings (ADR-0033 §"Streaming hardening").

Streaming RSS curves and per-image latency CDFs are sensitive to two
kernel knobs that detection cells don't notice:

- ``/sys/kernel/mm/transparent_hugepage/enabled`` — the active policy
  (between brackets) must be ``never`` for stable RSS measurements.
- ``/proc/sys/vm/swappiness`` — non-default values shift the peak-RSS
  curve under saturation.

The warnings only fire for the ``streaming`` paradigm; instance /
panoptic / semantic cells skip them entirely.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from bench.harness.platform_check import (
    StreamingPlatformWarning,
    _read_swappiness,
    _read_thp_active,
    streaming_sensitivity_warnings,
)


@pytest.fixture
def thp_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Synthetic THP and swappiness sysfiles for monkey-patching the
    reads. Returns (thp_path, swappiness_path)."""
    thp = tmp_path / "thp_enabled"
    swap = tmp_path / "swappiness"
    return thp, swap


def test_thp_never_emits_no_warning(thp_paths: tuple[Path, Path]) -> None:
    thp, swap = thp_paths
    thp.write_text("always madvise [never]\n")
    swap.write_text("60\n")
    msgs = streaming_sensitivity_warnings(
        "streaming", thp_path=thp, swappiness_path=swap
    )
    assert msgs == []


def test_thp_always_emits_warning(thp_paths: tuple[Path, Path]) -> None:
    thp, swap = thp_paths
    thp.write_text("[always] madvise never\n")
    swap.write_text("60\n")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", StreamingPlatformWarning)
        msgs = streaming_sensitivity_warnings(
            "streaming", thp_path=thp, swappiness_path=swap
        )
    assert len(msgs) == 1
    assert "transparent_hugepage" in msgs[0]
    assert "'always'" in msgs[0]
    streaming_warns = [w for w in captured if issubclass(w.category, StreamingPlatformWarning)]
    assert len(streaming_warns) == 1


def test_thp_madvise_emits_warning(thp_paths: tuple[Path, Path]) -> None:
    """``madvise`` is also non-default for benchmarking; warn so the
    user sees it. Anything-but-``never`` triggers the warning."""
    thp, swap = thp_paths
    thp.write_text("always [madvise] never\n")
    swap.write_text("60\n")
    msgs = streaming_sensitivity_warnings(
        "streaming", thp_path=thp, swappiness_path=swap
    )
    assert any("'madvise'" in m for m in msgs)


def test_non_default_swappiness_emits_warning(thp_paths: tuple[Path, Path]) -> None:
    thp, swap = thp_paths
    thp.write_text("always madvise [never]\n")
    swap.write_text("100\n")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", StreamingPlatformWarning)
        msgs = streaming_sensitivity_warnings(
            "streaming", thp_path=thp, swappiness_path=swap
        )
    assert len(msgs) == 1
    assert "swappiness" in msgs[0]
    assert "100" in msgs[0]
    streaming_warns = [w for w in captured if issubclass(w.category, StreamingPlatformWarning)]
    assert len(streaming_warns) == 1


def test_both_non_default_emits_two_warnings(thp_paths: tuple[Path, Path]) -> None:
    thp, swap = thp_paths
    thp.write_text("[always] madvise never\n")
    swap.write_text("10\n")
    msgs = streaming_sensitivity_warnings(
        "streaming", thp_path=thp, swappiness_path=swap
    )
    assert len(msgs) == 2


@pytest.mark.parametrize("paradigm", ["instance", "panoptic", "semantic"])
def test_non_streaming_paradigms_skip_check(
    paradigm: str, thp_paths: tuple[Path, Path]
) -> None:
    """For non-streaming paradigms the warning logic is a no-op even
    when the sysfiles look bad — detection / panoptic / semantic
    cells are insensitive to these knobs."""
    thp, swap = thp_paths
    thp.write_text("[always] madvise never\n")
    swap.write_text("100\n")
    msgs = streaming_sensitivity_warnings(
        paradigm,  # type: ignore[arg-type]
        thp_path=thp,
        swappiness_path=swap,
    )
    assert msgs == []


def test_missing_sysfiles_are_silent(thp_paths: tuple[Path, Path]) -> None:
    """If ``/sys/...`` reads fail (kernel without THP, or container
    without ``/proc/sys`` mounted), the function falls back to no
    warning rather than crashing the bench."""
    thp, swap = thp_paths
    # Don't write the files — they'll be missing.
    msgs = streaming_sensitivity_warnings(
        "streaming", thp_path=thp, swappiness_path=swap
    )
    assert msgs == []


def test_read_thp_active_parses_brackets(thp_paths: tuple[Path, Path]) -> None:
    """Direct test of the parser — the brackets indicate the active
    policy regardless of order."""
    thp, _ = thp_paths
    thp.write_text("always [madvise] never\n")
    assert _read_thp_active(thp) == "madvise"
    thp.write_text("[never] always madvise\n")
    assert _read_thp_active(thp) == "never"


def test_read_swappiness_parses_int(thp_paths: tuple[Path, Path]) -> None:
    _, swap = thp_paths
    swap.write_text("60\n")
    assert _read_swappiness(swap) == 60
    swap.write_text("invalid\n")
    assert _read_swappiness(swap) is None
