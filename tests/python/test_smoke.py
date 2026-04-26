"""Smoke tests: the FFI bridge loads and exposes the public surface.

These tests are intentionally trivial. Their job is to fail loudly when the
build is broken — wrong wheel ABI, missing symbols, dynamic linker problems —
not to cover algorithmic behavior.
"""

from __future__ import annotations

import vernier


def test_package_imports() -> None:
    assert vernier is not None


def test_version_is_nonempty_string() -> None:
    v = vernier.version()
    assert isinstance(v, str)
    assert v


def test_dunder_version_matches() -> None:
    assert vernier.__version__ == vernier.version()
