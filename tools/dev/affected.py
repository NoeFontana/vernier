"""Pick the test suites a branch's changes can affect.

    python tools/dev/affected.py [--base REF] [--run] [PYTEST_ARGS...]

Diffs the working tree (untracked files included) against the merge base
with REF (default: origin/main), prints the reason for each relevant
change and the resulting commands, and with --run executes them. The rules
are coarse on purpose and err towards running more; anything no rule
recognises runs everything.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]

# Changes that never affect a test outcome.
INERT_PREFIXES = ("docs/", ".github/", ".claude/", "tools/dev/", "LICENSE", ".gitignore")
INERT_SUFFIXES = (".md",)
# Inputs of every build.
GLOBAL_FILES = {"Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "pyproject.toml", "uv.lock"}

# The Rust suite takes seconds, so it always runs whole.
RUST = ["just", "test-rust"]
BENCH = ["just", "bench-test"]
PYTEST = ["uv", "run", "pytest"]


def git(*args: str) -> list[str]:
    out = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return [line for line in out.stdout.splitlines() if line]


def changed_files(base: str) -> list[str]:
    merge_base = git("merge-base", base, "HEAD")[0]
    tracked = git("diff", "--name-only", merge_base)
    untracked = git("ls-files", "--others", "--exclude-standard")
    return sorted(set(tracked) | set(untracked))


def workspace() -> tuple[dict[str, str], set[str]]:
    """Map each crate directory to its crate; name the crates linked into `_core`."""
    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    dirs: dict[str, str] = {}
    deps: dict[str, set[str]] = {}
    for pkg in meta["packages"]:
        dirs[str(Path(pkg["manifest_path"]).parent.relative_to(ROOT))] = pkg["name"]
        deps[pkg["name"]] = {d["name"] for d in pkg["dependencies"] if d.get("kind") is None}
    extension: set[str] = set()
    pending = ["vernier-ffi"]
    while pending:
        name = pending.pop()
        if name in deps and name not in extension:
            extension.add(name)
            pending.extend(deps[name])
    return dirs, extension


def cli_tests() -> list[str]:
    """Python test files that drive the CLI binary (the `vernier_bin` fixture)."""
    return sorted(
        str(p.relative_to(ROOT))
        for p in (ROOT / "tests" / "python").rglob("test_*.py")
        if "vernier_bin" in p.read_text()
    )


def plan(files: list[str], pytest_args: list[str]) -> tuple[list[str], list[list[str]]]:
    files = [f for f in files if not (f.startswith(INERT_PREFIXES) or f.endswith(INERT_SUFFIXES))]
    if not files:
        return [], []
    dirs, extension = workspace()
    reasons: list[str] = []
    rust = python_all = bench = False
    py_files: set[str] = set()

    for f in files:
        path = PurePosixPath(f)
        if f in GLOBAL_FILES:
            reasons.append(f"{f}: input of every build -> everything")
            rust = python_all = True
        elif crate := next((n for d, n in dirs.items() if path.is_relative_to(d)), None):
            rust = True
            if crate in extension:
                reasons.append(f"{f}: {crate} is linked into the extension -> Rust + Python")
                python_all = True
            elif crate == "vernier-cli":
                reasons.append(f"{f}: {crate} -> Rust + the Python tests driving the CLI")
                py_files.update(cli_tests())
            else:
                reasons.append(f"{f}: {crate} -> Rust")
        elif f.startswith("python/"):
            reasons.append(f"{f}: Python package -> Python")
            python_all = True
        elif f.startswith("tests/python/"):
            if path.match("test_*.py"):
                reasons.append(f"{f}: test file -> run it")
                py_files.add(f)
            else:
                reasons.append(f"{f}: shared test helper or fixture -> Python")
                python_all = True
        elif f.startswith("bench/"):
            reasons.append(f"{f}: bench harness -> bench tests")
            bench = True
        else:
            reasons.append(f"{f}: no rule matches -> everything")
            rust = python_all = bench = True

    commands: list[list[str]] = []
    if rust:
        commands.append(RUST)
    existing = sorted(p for p in py_files if (ROOT / p).exists())
    if python_all or existing:
        commands.append([*PYTEST, *([] if python_all else existing), *pytest_args])
    if bench:
        commands.append(BENCH)
    return reasons, commands


def main() -> int:
    parser = argparse.ArgumentParser(description="Pick the tests a branch's changes can affect.")
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--run", action="store_true", help="execute the commands")
    args, pytest_args = parser.parse_known_args()

    reasons, commands = plan(changed_files(args.base), pytest_args)
    for reason in reasons or ["no test-relevant changes"]:
        print(f"  {reason}")
    for command in commands:
        print("$", shlex.join(command))
        if args.run and (code := subprocess.run(command, cwd=ROOT).returncode):
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
