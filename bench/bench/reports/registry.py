"""Report-fragment registry — paradigm-keyed, additive.

A "fragment" is a pluggable piece of a paradigm's report: the per-cell
delta table for instance, the per-class PQ table for panoptic, the
confusion-matrix divergence visualization for semantic, the RSS-curve
plot for streaming. The registry lets each paradigm plug its
fragments in without touching the core compare/report code.

Lifecycle:

- This module defines the protocol, populates
  ``"instance"`` with the existing detection compare-table fragment
  refactored out of ``compare.py`` / ``render.py``.
- Per-paradigm modules import :func:`register_report_fragment` at module
  import time and append their fragments to the appropriate
  paradigm's list. Multiple fragments per paradigm are supported —
  the renderer concatenates them in registration order.
- The integration phase composes fragments into one master report by
  iterating paradigms and rendering each paradigm's fragment list.

Why a list per paradigm rather than one fragment: B1 may register a
``per_class_pq`` fragment and a separate ``boundary_pq`` fragment when
the upstream fork lands; B3 may register both a throughput-delta
fragment and an RSS-curve fragment. The registry is the multiplexer
so each fragment owns its render shape independently.

Convention for B-streams:

    # bench/runners/<paradigm>_runner.py — at module import:
    from bench.harness.schema import Paradigm
    from bench.reports.registry import (
        ReportFragment,
        register_report_fragment,
    )

    class _MyFragment:
        paradigm: ClassVar[Paradigm] = "panoptic"
        name: ClassVar[str] = "per_class_pq"

        def render(self, cells: list[BenchResult]) -> str:
            ...

    register_report_fragment(_MyFragment())

Idempotency: registering a fragment with the same ``(paradigm, name)``
twice replaces the first registration rather than appending. This
keeps reloads (test fixtures, ``importlib.reload``) from
double-rendering. Distinct names with the same paradigm coexist.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Protocol, runtime_checkable

from bench.harness.schema import BenchResult, Paradigm


@runtime_checkable
class ReportFragment(Protocol):
    """One renderable section of a paradigm's report.

    Implementations are typically lightweight stateless classes; the
    ``render`` call receives the list of ``BenchResult`` cells that
    matched the report's filters (e.g., a base-vs-head cross-walk
    yielded these rows for ``paradigm`` after grouping).

    Fragment classes carry two ClassVars:

    - ``paradigm`` — the registry key. The dispatcher uses it; a
      fragment that returns rows from a different paradigm is a
      registration error.
    - ``name`` — a fragment-local identifier. Used as the
      idempotency key so re-registering replaces rather than
      appends. Convention: snake-case, descriptive (``per_class_pq``,
      ``rss_curve``, ``confusion_divergence``).
    """

    paradigm: ClassVar[Paradigm]
    name: ClassVar[str]

    def render(self, cells: Sequence[BenchResult]) -> str: ...


# Per-paradigm list of fragments in registration order. Lookups iterate
# this list; the registry itself is mutable so B-streams can extend it.
# Keys mirror the ``Paradigm`` literal — every entry is initialized so
# callers don't have to special-case empty paradigms.
_REGISTRY: dict[Paradigm, list[ReportFragment]] = {
    "instance": [],
    "panoptic": [],
    "semantic": [],
    "streaming": [],
}


def register_report_fragment(fragment: ReportFragment) -> None:
    """Append ``fragment`` to its paradigm's list (idempotent on name).

    If a fragment with the same ``(paradigm, name)`` is already
    registered, replace it in-place rather than appending. This keeps
    the rendered report stable across reloads (test fixtures use
    ``importlib.reload``; without idempotency a re-imported runner
    would double-register its fragment).
    """
    paradigm = fragment.paradigm
    if paradigm not in _REGISTRY:
        raise ValueError(
            f"unknown paradigm {paradigm!r}; registered fragment must use "
            f"one of {list(_REGISTRY)}"
        )
    name = fragment.name
    fragments = _REGISTRY[paradigm]
    for i, existing in enumerate(fragments):
        if existing.name == name:
            fragments[i] = fragment
            return
    fragments.append(fragment)


def fragments_for(paradigm: Paradigm) -> list[ReportFragment]:
    """Return the registered fragments for ``paradigm`` in registration order.

    Returns a *copy* of the internal list so callers can iterate
    without worrying about further mutation. Empty list when no
    fragments have been registered yet — typical for a fresh process
    that hasn't imported any B-stream runner module.
    """
    return list(_REGISTRY.get(paradigm, []))


def render_paradigm(paradigm: Paradigm, cells: Sequence[BenchResult]) -> str:
    """Render every fragment for ``paradigm`` and concatenate.

    Fragments are joined with a blank line so each owns its own
    layout. An empty paradigm (no registered fragments) yields the
    empty string — the caller decides whether to surface a "no
    fragments registered" message in that case.
    """
    parts = [fragment.render(cells) for fragment in fragments_for(paradigm)]
    return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Built-in fragments
# ---------------------------------------------------------------------------
#
# Pre-populated for the instance paradigm with the detection
# compare-table fragment that the existing ``compare.py`` /
# ``render.py`` already implement. The fragment is a thin wrapper that
# delegates to the legacy renderer; B-streams compose around it
# without rewriting it.
#
# Why a wrapper: the legacy renderer takes ``CompareRow`` (the result
# of ``compare_shas``), but the registry's ``render(cells)`` signature
# is ``BenchResult``-shaped. The wrapper is constructed at compare-
# time with the rows it needs and registered lazily — see
# ``compose_compare_sections`` in ``compare.py``.


class InstanceCellsFragment:
    """Default fragment for the instance paradigm.

    Renders a one-line "N detection cell(s) covered" descriptor — the
    actual delta table is built directly by ``compare`` because it
    needs the base/head SHA pair, which a fragment-only signature
    doesn't carry. Listed here so an integrator iterating the
    registry sees the instance paradigm contribute a fragment instead
    of mysteriously coming up empty.
    """

    paradigm: ClassVar[Paradigm] = "instance"
    name: ClassVar[str] = "instance_cells_summary"

    def render(self, cells: Sequence[BenchResult]) -> str:
        if not cells:
            return ""
        n = len(cells)
        workloads = sorted({c.workload_id for c in cells})
        ious = sorted({c.iou_type for c in cells})
        # Markdown summary line; the per-cell delta table is rendered
        # by ``compare`` because it carries SHA-pair context the
        # fragment shape doesn't.
        return (
            f"_{n} instance cell(s); workloads: {', '.join(workloads)}; "
            f"iou-types: {', '.join(ious)}._"
        )


# Register the built-in instance fragment. B-streams add their own at
# import time (``bench/runners/<paradigm>_runner.py`` calls
# ``register_report_fragment`` at module top-level).
register_report_fragment(InstanceCellsFragment())
