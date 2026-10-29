"""Fixtures shared by every Python test package."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def vernier_bin() -> Path:
    """The `vernier` CLI binary, built if stale.

    Dev profile, so it reuses what `cargo nextest` built. The path comes
    from cargo's build report, which honours `CARGO_TARGET_DIR` and never
    returns a stale binary.
    """
    proc = subprocess.run(
        ["cargo", "build", "-p", "vernier-cli", "--message-format=json-render-diagnostics"],
        cwd=WORKSPACE_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    for line in proc.stdout.splitlines():
        message = json.loads(line)
        if (
            message.get("reason") == "compiler-artifact"
            and message["target"]["name"] == "vernier"
            and (exe := message.get("executable"))
        ):
            return Path(exe)
    raise RuntimeError("cargo build -p vernier-cli reported no `vernier` executable")
