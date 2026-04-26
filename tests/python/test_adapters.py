"""Tests for the pycocotools monkey-patch adapter (ADR-0007).

The adapter replaces ``pycocotools.cocoeval.COCOeval`` with vernier's
drop-in. These tests verify:

- The patch is observable through the canonical
  ``from pycocotools.cocoeval import COCOeval`` import path.
- Unpatch restores the original class and is idempotent.
- The context-manager variant nests via LIFO (per ADR-0007
  §"Reentrancy").
- Calling the patch helper without pycocotools installed raises
  ``ImportError`` instead of silently registering vernier's class.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pycocotools.cocoeval as cocoeval_mod
import pytest

from vernier._compat import PycocotoolsCOCOeval
from vernier.adapters import patch_pycocotools, patched_pycocotools

ORIGINAL_COCOEVAL = cocoeval_mod.COCOeval


@pytest.fixture(autouse=True)
def _restore_after_test() -> Iterator[None]:
    yield
    # Defensive teardown: even if a test forgets to unpatch, the next
    # test starts from the pristine pycocotools class. Mutating module
    # state is the whole point of this adapter, so an explicit
    # restore-loop is cheaper than per-test bookkeeping.
    cocoeval_mod.COCOeval = ORIGINAL_COCOEVAL
    PycocotoolsCOCOeval.DEFAULT_PARITY_MODE = "strict"


def test_patch_replaces_cocoeval_in_sys_modules() -> None:
    unpatch = patch_pycocotools()
    try:
        # The canonical migration import path.
        from pycocotools.cocoeval import COCOeval as PatchedCOCOeval

        assert PatchedCOCOeval is PycocotoolsCOCOeval
        assert cocoeval_mod.COCOeval is PycocotoolsCOCOeval
    finally:
        unpatch()
    assert cocoeval_mod.COCOeval is ORIGINAL_COCOEVAL


def test_patch_default_mode_is_strict() -> None:
    # Per ADR-0007: default is "strict" because migration intent is
    # bit-exactness. Adapter mutates the class-var so unflagged
    # construction picks the patched mode up.
    unpatch = patch_pycocotools()
    try:
        assert PycocotoolsCOCOeval.DEFAULT_PARITY_MODE == "strict"
    finally:
        unpatch()


def test_patch_propagates_corrected_mode() -> None:
    unpatch = patch_pycocotools(parity_mode="corrected")
    try:
        assert PycocotoolsCOCOeval.DEFAULT_PARITY_MODE == "corrected"
    finally:
        unpatch()
    assert PycocotoolsCOCOeval.DEFAULT_PARITY_MODE == "strict"


def test_unpatch_is_idempotent() -> None:
    unpatch = patch_pycocotools()
    unpatch()
    assert cocoeval_mod.COCOeval is ORIGINAL_COCOEVAL
    # Second call must not pop another (unrelated) patch off the stack.
    unpatch()
    assert cocoeval_mod.COCOeval is ORIGINAL_COCOEVAL


def test_context_manager_restores_on_exit() -> None:
    with patched_pycocotools():
        assert cocoeval_mod.COCOeval is PycocotoolsCOCOeval
    assert cocoeval_mod.COCOeval is ORIGINAL_COCOEVAL


def test_context_manager_restores_on_exception() -> None:
    def _enter_then_raise() -> None:
        with patched_pycocotools():
            assert cocoeval_mod.COCOeval is PycocotoolsCOCOeval
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _enter_then_raise()
    assert cocoeval_mod.COCOeval is ORIGINAL_COCOEVAL


def test_nested_context_managers_lifo_restore_inner_then_outer() -> None:
    # Per ADR-0007 §"Reentrancy": inner exit restores outer state;
    # outer exit restores the original class.
    with patched_pycocotools(parity_mode="strict"):
        assert PycocotoolsCOCOeval.DEFAULT_PARITY_MODE == "strict"
        with patched_pycocotools(parity_mode="corrected"):
            assert PycocotoolsCOCOeval.DEFAULT_PARITY_MODE == "corrected"
        # Inner exit restored "strict" (outer's mode), not the
        # pre-patch class.
        assert PycocotoolsCOCOeval.DEFAULT_PARITY_MODE == "strict"
        assert cocoeval_mod.COCOeval is PycocotoolsCOCOeval
    assert cocoeval_mod.COCOeval is ORIGINAL_COCOEVAL


def test_double_patch_then_unwind_restores_original() -> None:
    # ADR-0007 §"Idempotency": "Calling patch_pycocotools twice without
    # unpatching returns an unpatch handle that, when called, restores
    # the *original* pycocotools class — not the first-patch state."
    # The second unpatch must collapse to the original because the
    # adapter records the original on first patch only.
    unpatch_a = patch_pycocotools(parity_mode="strict")
    unpatch_b = patch_pycocotools(parity_mode="corrected")
    unpatch_b()
    unpatch_a()
    assert cocoeval_mod.COCOeval is ORIGINAL_COCOEVAL
    assert PycocotoolsCOCOeval.DEFAULT_PARITY_MODE == "strict"


def test_patch_raises_when_pycocotools_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Per ADR-0007 §"What we deliberately do not do": no silent
    # fallthrough. Simulate an environment without pycocotools by
    # masking the relevant entries in sys.modules and forcing an
    # ImportError on re-import.
    monkeypatch.setitem(sys.modules, "pycocotools", None)
    monkeypatch.setitem(sys.modules, "pycocotools.cocoeval", None)
    with pytest.raises(ImportError, match="pycocotools"):
        patch_pycocotools()
