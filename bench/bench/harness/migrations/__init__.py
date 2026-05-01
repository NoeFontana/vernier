"""Schema migrations (ADR-0017 §F2). Applied lazily on read.

Each migration is a module ``v{N}_to_v{N+1}.py`` exposing
``FROM_VERSION = N`` and ``def migrate(d: dict) -> dict``. ``upgrade()``
walks them in order until the dict reports ``schema_version == LATEST``.

v1 has no migrations yet. The framework still ships so v2 doesn't
need a special-cased cutover.
"""

from __future__ import annotations

from typing import Any

LATEST: int = 1

# Each entry is (from_version, migrate_fn). Append new migrations in order.
_MIGRATIONS: list[tuple[int, Any]] = []


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
