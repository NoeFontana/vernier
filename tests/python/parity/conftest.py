"""Pytest configuration for parity tests.

The parity suite double-runs the reference (pycocotools 2.0.8) and the
candidate (vernier) on the same fixtures and asserts every intermediate
matches. Today the candidate is a shim that delegates to pycocotools, so the
suite is a tautology — but the harness, fixture corpus, and CI plumbing are
real. As Rust evaluator pieces ship, the shim is replaced and the suite
becomes a load-bearing parity gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR
