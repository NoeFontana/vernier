"""Read-side compat shim for v1 result JSON (ADR-0033).

v1 results were detection-only and stored a single tensor pair
(``tensor_path`` + ``tensor_sha256``); v2 generalizes to
``artifact_paths`` / ``artifact_sha256`` dicts and adds a required
``paradigm`` field. This shim only handles the read path: any v1 dict
is upgraded in-memory so the v2 model parses it cleanly.

The forward migration of on-disk result trees (rewriting v1 result
JSON files to v2 under the paradigm-segmented path) is the A-thick
deliverable; that tool will live next to this module as
``migrate_tree.py`` and walk every ``results/<sha>/<fp>/<workload>/<iou>/...``
that's still v1.

This module is deliberately the only place that knows the
``"tensor"`` slot key — once detection writes go through the v2
schema, every reader sees the same dict shape regardless of
paradigm.
"""

from __future__ import annotations

from typing import Any

# v1 → v2 migration. Append further migrations in sibling modules.
FROM_VERSION: int = 1
TO_VERSION: int = 2

# Canonical artifact key for the v1 single-tensor pair. Every v1 reader
# that hands a v2 model a v1 dict ends up with the tensor slot under
# this key; B-stream runners that emit non-tensor artifacts pick their
# own keys (``"snapshot"``, ``"per_class"``, ``"summary"``,
# ``"rss_curve"``).
TENSOR_KEY: str = "tensor"


def migrate(d: dict[str, Any]) -> dict[str, Any]:
    """Lift a v1 result dict into v2-shape. Idempotent for v2 input.

    - ``schema_version: 1`` → ``2``
    - ``tensor_path`` lifts into ``artifact_paths = {"tensor": ...}``
    - ``tensor_sha256`` lifts into ``artifact_sha256 = {"tensor": ...}``
    - ``paradigm`` defaults to ``"instance"`` (v1 was detection-only)

    The function returns a *new* dict so callers don't have to worry
    about whether they own ``d``.
    """
    if d.get("schema_version") == TO_VERSION:
        return dict(d)
    if d.get("schema_version") != FROM_VERSION:
        raise ValueError(
            f"v1_to_v2.migrate expected schema_version=={FROM_VERSION}, "
            f"got {d.get('schema_version')!r}"
        )

    out = dict(d)
    out["schema_version"] = TO_VERSION
    out.setdefault("paradigm", "instance")

    tensor_path = out.pop("tensor_path", None)
    tensor_sha = out.pop("tensor_sha256", None)
    artifact_paths: dict[str, str] = dict(out.pop("artifact_paths", {}))
    artifact_sha256: dict[str, str] = dict(out.pop("artifact_sha256", {}))
    if tensor_path is not None:
        artifact_paths.setdefault(TENSOR_KEY, str(tensor_path))
    if tensor_sha is not None:
        artifact_sha256.setdefault(TENSOR_KEY, str(tensor_sha))
    out["artifact_paths"] = artifact_paths
    out["artifact_sha256"] = artifact_sha256
    return out
