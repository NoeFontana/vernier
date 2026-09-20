#!/usr/bin/env python3
"""Validate the `Wired-by` citations in the quirks surveys.

Every quirk row cites the code that implements or pins it. Those
citations used to carry `file:line` coordinates, which go stale on
every refactor that shifts a line -- silently, because nothing read
them back. This check enforces the replacement contract:

    a citation is `<path>::<item>` -- a file path plus a *named*
    definition in it, and never a line number.

Names move only when someone renames or deletes the thing, and this
check turns that into a lint failure in the same PR.

Run: `just lint-citations` (part of `just lint`), or directly:
    python3 tools/check_quirk_citations.py [FILE ...]

Cheap by construction: it reads the surveyed markdown plus the handful
of source files actually cited, resolves each path against the git
index, and greps for a definition. No build, no import, no network.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DEFAULT_SURVEYS = (
    "docs/engineering/pycocotools-quirks.md",
    "docs/engineering/obb-quirks.md",
)

#: Header of the column whose backticked spans are checked.
CITATION_COLUMN = "wired-by"

#: A backticked span is a citation when it names a source file. Anything
#: else in the column (fixture names, type names, prose) is left alone.
CITATION_RE = re.compile(r"^(?P<path>[\w./-]+\.(?:rs|py|pyi))(?:::(?P<item>[\w:.]+))?$")

#: Rejected outright: a trailing `:123` or `:123-456` coordinate.
LINE_NUMBER_RE = re.compile(r":\d+(?:-\d+)?$")

#: `<!-- citation-root: crates/vernier-core/src -->` in the survey lets
#: rows cite `matching.rs` rather than the full path, while keeping the
#: resolution unambiguous.
ROOT_DIRECTIVE_RE = re.compile(r"<!--\s*citation-root:\s*(?P<root>[\w./-]+)\s*-->")

SOURCE_SUFFIXES = (".rs", ".py", ".pyi")

SKIP_DIRS = {".git", "target", ".venv", "site", "node_modules", "__pycache__"}


def repo_sources() -> list[str]:
    """Every tracked source file, as repo-relative posix paths."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout
        paths = [p for p in out.split("\0") if p]
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - fallback
        paths = [
            str(p.relative_to(REPO))
            for p in REPO.rglob("*")
            if p.is_file() and not SKIP_DIRS & set(p.relative_to(REPO).parts)
        ]
    return [p for p in paths if p.endswith(SOURCE_SUFFIXES)]


def resolve_path(cited: str, sources: list[str], roots: list[str]) -> tuple[str | None, list[str]]:
    """Resolve a (possibly abbreviated) cited path to one tracked file.

    Order: exact repo-relative path, then each declared `citation-root`
    in turn, then a unique suffix match. Anything that still matches
    more than one file is reported so the row can lengthen its path.
    """
    index = set(sources)
    if cited in index:
        return cited, [cited]
    for root in roots:
        candidate = f"{root.rstrip('/')}/{cited}"
        if candidate in index:
            return candidate, [candidate]
    suffix = "/" + cited
    hits = [p for p in sources if p.endswith(suffix)]
    if len(hits) == 1:
        return hits[0], hits
    return None, hits


def rust_defines(text: str, name: str) -> bool:
    n = re.escape(name)
    patterns = (
        # fn / struct / enum / trait / type / mod / union / const / static
        rf"\b(?:fn|struct|enum|trait|type|mod|union|const|static)\s+{n}\b",
        # macro-free enum variant or associated item at line start
        rf"(?m)^\s*{n}\s*(?:[,({{]|=>|:)",
        # tuple-struct / unit variant declared inline
        rf"\b{n}\s*=>",
    )
    return any(re.search(p, text) for p in patterns)


def python_defines(text: str, name: str) -> bool:
    n = re.escape(name)
    patterns = (
        rf"\b(?:def|class)\s+{n}\b",
        rf"(?m)^\s*{n}\s*(?::[^=]*)?=",
    )
    return any(re.search(p, text) for p in patterns)


def defines(path: str, text: str, item: str) -> bool:
    # `Type::method` / `Class.method` -- the last segment is the name we
    # can find; the qualifier is for the reader.
    leaf = re.split(r"::|\.", item)[-1]
    if path.endswith(".rs"):
        return rust_defines(text, leaf)
    return python_defines(text, leaf)


def citation_cells(lines: list[str]) -> list[tuple[int, str]]:
    """Yield (line_number, cell) for the citation column of every row."""
    col: int | None = None
    out: list[tuple[int, str]] = []
    for lineno, line in enumerate(lines, 1):
        if not line.lstrip().startswith("|"):
            col = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(set(c) <= {"-", ":", " "} and c for c in cells):
            continue  # separator row
        lowered = [c.lower() for c in cells]
        if CITATION_COLUMN in lowered:
            col = lowered.index(CITATION_COLUMN)
            continue
        if col is not None and col < len(cells):
            out.append((lineno, cells[col]))
    return out


def check(survey: Path, sources: list[str], cache: dict[str, str]) -> list[str]:
    errors: list[str] = []
    rel = survey.relative_to(REPO) if survey.is_absolute() else survey
    lines = survey.read_text(encoding="utf-8").splitlines()
    roots = [m.group("root") for line in lines for m in [ROOT_DIRECTIVE_RE.search(line)] if m]
    checked = 0

    for lineno, cell in citation_cells(lines):
        for span in re.findall(r"`([^`]+)`", cell):
            where = f"{rel}:{lineno}"
            if LINE_NUMBER_RE.search(span):
                errors.append(
                    f"{where}: citation `{span}` carries a line number. "
                    "Cite `<path>::<name>` instead -- names survive refactors."
                )
                continue
            m = CITATION_RE.match(span)
            if m is None:
                continue
            checked += 1
            cited_path = m.group("path")
            resolved, hits = resolve_path(cited_path, sources, roots)
            if resolved is None:
                if hits:
                    errors.append(
                        f"{where}: citation `{span}` is ambiguous -- "
                        f"`{cited_path}` matches {len(hits)} files "
                        f"({', '.join(sorted(hits)[:4])}...). Lengthen the path."
                    )
                else:
                    errors.append(
                        f"{where}: citation `{span}` names no tracked file (`{cited_path}`)."
                    )
                continue
            item = m.group("item")
            if item is None:
                errors.append(
                    f"{where}: citation `{span}` names a file but no item. Cite `<path>::<name>`."
                )
                continue
            text = cache.get(resolved)
            if text is None:
                text = (REPO / resolved).read_text(encoding="utf-8", errors="replace")
                cache[resolved] = text
            if not defines(resolved, text, item):
                errors.append(
                    f"{where}: citation `{span}` -- no definition of "
                    f"`{item.split('::')[-1].split('.')[-1]}` found in {resolved}."
                )

    if not errors:
        print(f"{rel}: {checked} citations OK")
    return errors


def main(argv: list[str]) -> int:
    targets = argv[1:] or list(DEFAULT_SURVEYS)
    sources = repo_sources()
    cache: dict[str, str] = {}
    errors: list[str] = []
    for t in targets:
        p = Path(t)
        if not p.is_absolute():
            p = REPO / p
        if not p.exists():
            errors.append(f"{t}: no such file")
            continue
        errors.extend(check(p, sources, cache))

    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    if errors:
        print(f"\n{len(errors)} stale citation(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
