"""Per-paradigm report-fragment registry (ADR-0033 §"Report fragments").

B-streams (B1 panoptic, B2 semantic, B3 streaming, B5 BG p99) register
their paradigm-specific report fragments through this registry. The
test asserts:

- a stub fragment can be registered under each paradigm;
- the rendered output contains a section for each;
- re-registering the same ``(paradigm, name)`` pair replaces in
  place (idempotent under reload);
- distinct names with the same paradigm coexist (multiple fragments
  per paradigm — used by B3 for separate throughput + RSS-curve
  fragments).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from bench.harness.schema import BenchResult, Paradigm
from bench.reports.registry import (
    ReportFragment,
    fragments_for,
    register_report_fragment,
    render_paradigm,
)


class _StubFragment:
    """Minimal protocol-compliant fragment.

    The class-level paradigm/name pair makes this look like a
    real B-stream registration site; the render returns a
    discriminating string so tests can assert ordering.
    """

    def __init__(self, paradigm: Paradigm, name: str, payload: str) -> None:
        # Pyright doesn't allow ClassVar override per-instance, so
        # shadow with regular attributes — the protocol's runtime
        # check only cares the attributes exist.
        self.paradigm: Paradigm = paradigm
        self.name: str = name
        self._payload = payload

    def render(self, cells: Sequence[BenchResult]) -> str:  # noqa: ARG002
        return self._payload


def _restore(paradigm: Paradigm, originals: list[ReportFragment]) -> None:
    """Helper: restore a paradigm's fragment list after a test mutates it."""
    from bench.reports.registry import _REGISTRY

    _REGISTRY[paradigm].clear()
    _REGISTRY[paradigm].extend(originals)


def test_stub_fragment_registers_under_each_paradigm() -> None:
    """Register one stub per paradigm, then assert each paradigm's
    list contains it."""
    paradigms: tuple[Paradigm, ...] = ("instance", "panoptic", "semantic", "streaming")
    snapshots = {p: list(fragments_for(p)) for p in paradigms}
    try:
        for p in paradigms:
            register_report_fragment(_StubFragment(p, f"{p}_stub", f"<{p} payload>"))
        for p in paradigms:
            names = [f.name for f in fragments_for(p)]
            assert f"{p}_stub" in names
    finally:
        for p, originals in snapshots.items():
            _restore(p, originals)


def test_render_paradigm_includes_stub_payload() -> None:
    snapshot = list(fragments_for("panoptic"))
    try:
        register_report_fragment(_StubFragment("panoptic", "per_class_pq", "PQ table\n| ... |"))
        rendered = render_paradigm("panoptic", [])
        assert "PQ table" in rendered
    finally:
        _restore("panoptic", snapshot)


def test_register_idempotent_on_same_name() -> None:
    """Re-registering the same ``(paradigm, name)`` replaces in place;
    distinct names append. This keeps reloads (test fixtures /
    ``importlib.reload``) from double-rendering."""
    snapshot = list(fragments_for("streaming"))
    try:
        register_report_fragment(_StubFragment("streaming", "rss_curve", "v1"))
        n_after_first = len(fragments_for("streaming"))
        register_report_fragment(_StubFragment("streaming", "rss_curve", "v2"))
        n_after_second = len(fragments_for("streaming"))
        assert n_after_second == n_after_first
        rendered = render_paradigm("streaming", [])
        assert "v2" in rendered
        assert "v1" not in rendered
    finally:
        _restore("streaming", snapshot)


def test_register_distinct_names_coexist() -> None:
    """Two fragments with different names under the same paradigm
    both render. B3 uses this for separate throughput-delta and
    RSS-curve fragments."""
    snapshot = list(fragments_for("streaming"))
    try:
        register_report_fragment(_StubFragment("streaming", "throughput_delta", "TPUT"))
        register_report_fragment(_StubFragment("streaming", "rss_curve", "RSS"))
        rendered = render_paradigm("streaming", [])
        assert "TPUT" in rendered
        assert "RSS" in rendered
    finally:
        _restore("streaming", snapshot)


def test_built_in_instance_fragment_is_pre_populated() -> None:
    """A-thick pre-registers a built-in fragment for the ``instance``
    paradigm so an integrator iterating the registry sees the
    detection paradigm contribute, instead of mysteriously coming
    up empty."""
    fragments = fragments_for("instance")
    names = [f.name for f in fragments]
    assert "instance_cells_summary" in names


def test_register_unknown_paradigm_rejected() -> None:
    import pytest

    class _Bogus:
        paradigm: ClassVar[str] = "future_paradigm"
        name: ClassVar[str] = "x"

        def render(self, cells: Sequence[BenchResult]) -> str:  # noqa: ARG002
            return ""

    with pytest.raises(ValueError, match="future_paradigm"):
        register_report_fragment(_Bogus())  # type: ignore[arg-type]
