"""Schema migrations (ADR-0017 §F2; extended by ADR-0033). Applied
lazily on read.

Each migration is a module ``v{N}_to_v{N+1}.py`` exposing
``FROM_VERSION = N`` and ``def migrate(d: dict) -> dict``. ``upgrade()``
walks them in order until the dict reports ``schema_version == LATEST``.

The full v1 → v2 forward migration of on-disk result trees (re-emit
each cell under the paradigm-segmented path) is the A-thick
deliverable. The read-side compat shim in ``v1_to_v2`` lets readers
keep parsing v1 result JSON without that migration having run yet.
"""

from __future__ import annotations

from typing import Any

from bench.harness.migrations import v1_to_v2

LATEST: int = 2

# Each entry is (from_version, migrate_fn). Append new migrations in order.
_MIGRATIONS: list[tuple[int, Any]] = [
    (v1_to_v2.FROM_VERSION, v1_to_v2.migrate),
]


def upgrade(d: dict[str, Any]) -> dict[str, Any]:
    """Walk the migration chain until ``d`` is at LATEST. Idempotent."""
    current = d.get("schema_version")
    if not isinstance(current, int):
        raise ValueError(f"missing or non-integer schema_version: {current!r}")
    if current > LATEST:
        raise ValueError(
            f"file claims schema_version {current}; this build only knows "
            f"versions up to {LATEST}. Update vernier-bench."
        )
    for from_version, fn in _MIGRATIONS:
        if d["schema_version"] == from_version:
            d = fn(d)
    return d
