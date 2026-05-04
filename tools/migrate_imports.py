"""One-shot import migration script for ADR-0029 namespace restructure.

Usage::

    python tools/migrate_imports.py path/to/file.py [more.py ...]
    python tools/migrate_imports.py --tree tests/

Rewrites flat-root vernier imports into the per-paradigm submodules
mandated by ADR-0029. Idempotent: running twice produces the same
output as running once. Pre-1.0 migration aid; expected to be deleted
in a follow-up patch once external 0.0.x users have replayed it.

Scope: single-line ``from vernier import X, Y`` statements and
``vernier.X`` qualified accesses. Multi-line wrapped imports
(``from vernier import (\n    X,\n    Y,\n)``) are not rewritten; the
38 importer files migrated for ADR-0029 had none, so the limitation
was not load-bearing.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from pathlib import Path

# Symbols that move to vernier.instance under their existing names.
INSTANCE_NAMES = frozenset(
    {
        "BackgroundEvaluator",
        "Bbox",
        "Boundary",
        "Dataset",
        "EvalResult",
        "Evaluator",
        "FpIouHistogram",
        "IouKind",
        "Keypoints",
        "MemoryBudgetWarning",
        "OutOfBudgetError",
        "QueueFullError",
        "Segm",
        "StreamingEvaluator",
        "Summary",
        "TableName",
        "TablesConfig",
        "TideConfig",
        "TideReport",
        "confusion_matrix",
        "error_decomposition",
        "fp_iou_histogram",
    }
)

# Symbols that move to vernier.panoptic, with rename: drop the
# "Panoptic" prefix from the type names (PanopticEvaluator → Evaluator,
# etc.). The dict maps old root name to new submodule name.
PANOPTIC_RENAMES: dict[str, str] = {
    "PanopticEvaluator": "Evaluator",
    "PanopticDataset": "Dataset",
    "PanopticPredictions": "Predictions",
    "PanopticSummary": "Summary",
    "ClassPanopticStats": "ClassPanopticStats",
}

# Symbols that stay at the root.
ROOT_NAMES = frozenset({"COCOeval", "Frequency", "ParityMode", "version"})


def _split_destinations(names: list[str]) -> dict[str, list[str]]:
    """Group names by destination: instance, panoptic (with rename), or root."""
    buckets: dict[str, list[str]] = {"instance": [], "panoptic": [], "root": []}
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        if name in INSTANCE_NAMES:
            buckets["instance"].append(name)
        elif name in PANOPTIC_RENAMES:
            buckets["panoptic"].append(name)
        elif name in ROOT_NAMES:
            buckets["root"].append(name)
        else:
            # Unknown symbol — keep it at root (let the import fail loudly
            # rather than silently dropping it).
            buckets["root"].append(name)
    return buckets


def _rewrite_from_imports(line: str) -> list[str]:
    """Rewrite a single ``from vernier import ...`` line into one or more
    submodule imports. Returns a list of replacement lines (preserving
    the original leading whitespace)."""
    match = re.match(r"^(\s*)from vernier import (.+?)\s*$", line)
    if not match:
        return [line]
    indent, payload = match.group(1), match.group(2)
    # Strip parens / trailing commas if present (e.g., wrapped imports).
    payload = payload.strip().strip("()").rstrip(",")
    raw_names = [n.strip() for n in payload.split(",") if n.strip()]
    buckets = _split_destinations(raw_names)
    out: list[str] = []
    if buckets["instance"]:
        names = ", ".join(sorted(buckets["instance"]))
        out.append(f"{indent}from vernier.instance import {names}\n")
    if buckets["panoptic"]:
        # Rename Panoptic-prefixed types to their unprefixed submodule names.
        # Use ``as`` aliases so existing references in the file keep working
        # without a follow-up sweep.
        renames = sorted(buckets["panoptic"])
        clauses = []
        for old in renames:
            new = PANOPTIC_RENAMES[old]
            if new == old:
                clauses.append(new)
            else:
                clauses.append(f"{new} as {old}")
        out.append(f"{indent}from vernier.panoptic import {', '.join(clauses)}\n")
    if buckets["root"]:
        names = ", ".join(sorted(buckets["root"]))
        out.append(f"{indent}from vernier import {names}\n")
    return out or [line]


# Order matters: panoptic first so PanopticEvaluator wins over Evaluator.
QUALIFIED_REPLACEMENTS: list[tuple[str, str]] = [
    *((rf"\bvernier\.{old}\b", f"vernier.panoptic.{new}") for old, new in PANOPTIC_RENAMES.items()),
    *((rf"\bvernier\.{name}\b", f"vernier.instance.{name}") for name in INSTANCE_NAMES),
]


def _rewrite_qualified(text: str) -> str:
    """Rewrite ``vernier.X`` qualified accesses for moved symbols."""
    for pattern, replacement in QUALIFIED_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    return text


def migrate_file(path: Path) -> bool:
    """Rewrite a single file in place. Returns True if changed."""
    original = path.read_text()
    lines = original.splitlines(keepends=True)
    rewritten: list[str] = []
    for line in lines:
        rewritten.extend(_rewrite_from_imports(line))
    text = "".join(rewritten)
    text = _rewrite_qualified(text)
    if text == original:
        return False
    path.write_text(text)
    return True


_SKIP_DIRS = frozenset({".venv", "venv", "__pycache__", ".git", "site-packages"})


def _iter_python(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            yield root
        elif root.is_dir():
            for path in root.rglob("*.py"):
                if any(part in _SKIP_DIRS for part in path.parts):
                    continue
                yield path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", type=Path, help="Files or directories")
    p.add_argument("--tree", action="store_true", help="Recurse into directories")
    args = p.parse_args()
    targets: Iterable[Path] = _iter_python(args.paths) if args.tree else args.paths
    changed = 0
    for path in targets:
        if migrate_file(path):
            changed += 1
            print(f"migrated: {path}")
    print(f"{changed} file(s) migrated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
