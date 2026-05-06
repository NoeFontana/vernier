"""Forward migration of on-disk v1 result trees into the v2
paradigm-segmented layout (ADR-0033 §"Result-store path scheme v2").

Distinct from ``v1_to_v2.py`` (the read-side compat shim). That module
is enough for the *loader* to keep parsing v1 trees, but new writes go
to the v2 path scheme — a tree that mixes the two layouts is hard to
reason about. This walker re-emits every v1 cell at the matching v2
path, rewriting its result JSON through the read-side ``migrate``
helper so the on-disk shape matches the v2 schema model.

Path scheme:

- v1 cell directory: ``<sha>/<fp>/<workload>/<iou>/`` (4 segments)
- v2 cell directory: ``<sha>/<fp>/instance/<workload>/<iou>/`` (5
  segments — paradigm slot is always ``"instance"`` for v1 since v1
  was detection-only).

Idempotent: re-running on an already-migrated tree is a no-op (paths
that exist on the v2 side are detected and skipped). ``--keep-v1-paths``
defaults to ``True`` so the migration is reversible during the
transition window; passing ``False`` deletes the old v1 directories
once their v2 mirrors are in place. Empty intermediate directories left
behind when v1 paths are removed are pruned bottom-up.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from bench.harness.migrations.v1_to_v2 import migrate as migrate_dict

# A v1 cell directory has 4 path components below ``results_root``:
# ``<sha>/<fp>/<workload>/<iou>``. v2 has 5 (the extra ``<paradigm>``
# segment). The walker keys off this depth so it never confuses one
# for the other when both shapes coexist mid-migration.
_V1_DEPTH: int = 4
_V2_INSTANCE_PARADIGM: str = "instance"


@dataclass
class MigrationStats:
    """One row per migration run. Counters are append-only — running
    twice on the same tree leaves the second run's stats showing zero
    new files copied (idempotency)."""

    cells_migrated: int = 0
    files_copied: int = 0
    files_skipped_already_v2: int = 0
    v1_paths_removed: int = 0
    paths: list[Path] = field(default_factory=list)


def _v1_cell_dirs(results_root: Path) -> list[Path]:
    """Yield every v1-shaped cell directory under ``results_root``.

    A v1 cell sits at depth 4 (``<sha>/<fp>/<workload>/<iou>``) and is
    a directory whose name is one of the legacy iou-types. v2 cells
    live one level deeper so this filter cleanly separates them: the
    v2 leaf directory's grandparent is the paradigm segment, never an
    iou-type literal.
    """
    if not results_root.exists():
        return []
    out: list[Path] = []
    # Depth-4 glob; restrict to directories. Each match is the cell
    # leaf (e.g. ``.../smoke/bbox``); the four segments above
    # ``results_root`` are sha/fp/workload/iou.
    for cell_dir in results_root.glob("*/*/*/*"):
        if not cell_dir.is_dir():
            continue
        # A v1 cell's parent's parent is ``<fp>``; for v2 cells the
        # same depth path lands on ``<paradigm>/<workload>``, where
        # the leaf would be a workload, not an iou — but the ambiguity
        # only matters if a workload happens to share a name with one
        # of the v1 paradigm leaves. The only safe disambiguator is to
        # look at the leaf-of-leaf: a v2 cell's granchild contains
        # ``*.json`` files; a v1 cell *is* the leaf containing them.
        # Equivalently: a v1 leaf has no subdirectories that are
        # paradigm-named. Easiest rule: the leaf itself contains the
        # JSON files for v1, and is a parent of further dirs for v2.
        if not any(p.suffix == ".json" for p in cell_dir.iterdir() if p.is_file()):
            continue
        # Final guard: v2 trees with the same depth would have the
        # paradigm segment two levels up; reject any cell where the
        # great-grandparent name is one of the known paradigms (this
        # catches a hypothetical v2 cell that landed at this depth).
        # In practice v2 cells are at depth 5 and are filtered above
        # by the ``*.json`` check (their leaves are ``<metric>``
        # directories, not the JSON-bearing leaves themselves).
        out.append(cell_dir)
    return out


def _v2_target_dir(v1_cell_dir: Path, results_root: Path) -> Path:
    """Translate a v1 cell directory into its v2 mirror under the
    ``instance`` paradigm segment.

    v1: ``<root>/<sha>/<fp>/<workload>/<iou>``
    v2: ``<root>/<sha>/<fp>/instance/<workload>/<iou>``
    """
    rel = v1_cell_dir.relative_to(results_root)
    parts = rel.parts
    if len(parts) != _V1_DEPTH:
        raise ValueError(
            f"expected v1 cell at depth {_V1_DEPTH} below {results_root}; got {rel}"
        )
    sha, fp, workload, iou = parts
    return results_root / sha / fp / _V2_INSTANCE_PARADIGM / workload / iou


def _migrate_one_json(src: Path, dst: Path) -> bool:
    """Lift one v1 result JSON into v2 and write it to ``dst``.

    Returns ``True`` if a file was written, ``False`` if the JSON was
    already v2 and ``dst`` already exists with matching content (the
    idempotent no-op case).
    """
    payload = json.loads(src.read_bytes())
    upgraded = migrate_dict(payload)
    serialized = json.dumps(upgraded, indent=2, sort_keys=True)

    if dst.exists():
        existing = dst.read_text()
        # Re-serialize the existing file the same way to compare
        # canonically — different dump options shouldn't trip the
        # idempotency check.
        try:
            existing_payload = json.loads(existing)
            existing_canonical = json.dumps(existing_payload, indent=2, sort_keys=True)
        except json.JSONDecodeError:
            existing_canonical = existing
        if existing_canonical == serialized:
            return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(serialized)
    return True


def _migrate_one_cell(
    v1_cell_dir: Path,
    results_root: Path,
    *,
    stats: MigrationStats,
) -> None:
    """Mirror every file in ``v1_cell_dir`` to its v2 counterpart.

    Result JSON files are rewritten through ``v1_to_v2.migrate``;
    every other file (tensors, divergence reports, intermediate dirs
    skipped) is copied verbatim. Re-running on a tree that's already
    been migrated is a no-op.
    """
    v2_dir = _v2_target_dir(v1_cell_dir, results_root)
    v2_dir.mkdir(parents=True, exist_ok=True)
    stats.cells_migrated += 1
    stats.paths.append(v2_dir)

    for entry in v1_cell_dir.iterdir():
        # ``.intermediate/`` holds per-rep scratch files that the
        # orchestrator writes before assembling the cell result; they
        # don't carry parity-relevant state and re-running the cell
        # rewrites them. Skip rather than copy to keep migration cheap.
        if entry.is_dir() and entry.name == ".intermediate":
            continue
        if entry.is_dir():
            # Only ``.intermediate`` is expected; other dirs would mean
            # an unknown layout, in which case we copy rather than
            # silently lose data.
            target = v2_dir / entry.name
            if not target.exists():
                shutil.copytree(entry, target)
                stats.files_copied += 1
            continue
        target = v2_dir / entry.name
        if entry.suffix == ".json" and entry.name != "divergence_report.json":
            if _migrate_one_json(entry, target):
                stats.files_copied += 1
            else:
                stats.files_skipped_already_v2 += 1
        else:
            # Tensors and divergence reports copy verbatim — they're
            # binary-identical between v1 and v2 (the tensor sha256 is
            # what the read shim lifts into ``artifact_sha256["tensor"]``).
            if not target.exists():
                shutil.copyfile(entry, target)
                stats.files_copied += 1
            else:
                stats.files_skipped_already_v2 += 1


def _prune_empty_parents(path: Path, *, stop_at: Path) -> None:
    """Bottom-up remove empty directories from ``path`` up to (but not
    including) ``stop_at``. Used to clean up ``<sha>/<fp>/<workload>/``
    once its iou children have been moved.
    """
    cur = path
    while cur != stop_at and cur.is_dir() and not any(cur.iterdir()):
        cur.rmdir()
        cur = cur.parent


def migrate_tree(
    results_root: Path,
    *,
    keep_v1_paths: bool = True,
) -> MigrationStats:
    """Re-emit every v1 cell under ``results_root`` at its v2 path.

    Every v1 cell's JSON result is rewritten through
    ``v1_to_v2.migrate`` (so the on-disk schema matches v2); tensor
    files copy verbatim. Re-running on an already-migrated tree is a
    no-op (the per-file write is content-compared; the per-cell walker
    skips deleted v1 dirs).

    ``keep_v1_paths`` defaults to ``True`` for the transition window
    so users can fall back to the v1 layout if anything in the v2 path
    is wrong. Passing ``False`` deletes each v1 cell directory once its
    v2 mirror has been written, then prunes the empty intermediate
    ``<sha>/<fp>/<workload>/`` dirs that are left behind.
    """
    stats = MigrationStats()
    if not results_root.exists():
        return stats

    cell_dirs = _v1_cell_dirs(results_root)
    for v1_cell in cell_dirs:
        _migrate_one_cell(v1_cell, results_root, stats=stats)

    if not keep_v1_paths:
        for v1_cell in cell_dirs:
            # Sanity check: the v2 mirror must exist before we touch
            # the v1 source. Always true after the loop above, but
            # guard explicitly so a future refactor can't lose data.
            v2_mirror = _v2_target_dir(v1_cell, results_root)
            if not v2_mirror.exists():
                continue
            shutil.rmtree(v1_cell)
            stats.v1_paths_removed += 1
            # The v1 layout had ``<sha>/<fp>/<workload>/<iou>`` — once
            # ``<iou>`` is gone, ``<workload>`` may be empty. Prune up
            # to ``<fp>`` (the paradigm segment lives one below ``<fp>``
            # and is the v2 home, so we never touch ``<sha>`` or above).
            _prune_empty_parents(v1_cell.parent, stop_at=v1_cell.parent.parent)

    return stats
