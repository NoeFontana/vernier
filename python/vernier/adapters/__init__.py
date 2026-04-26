"""Framework adapters for vernier.

This subpackage holds optional integrations with external frameworks
(PyTorch tensors via DLPack, etc.) and migration helpers. Each adapter
is importable without its underlying framework being installed; gate
framework-specific imports behind ``try``/``except ImportError``.

The pycocotools migration helper (:func:`patch_pycocotools`) is the
sanctioned entry point for swapping
``pycocotools.cocoeval.COCOeval`` with vernier's drop-in. Policy and
rationale are in ADR-0007.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator
from typing import Literal

from vernier._compat import PycocotoolsCOCOeval

ParityMode = Literal["strict", "corrected"]

# LIFO stack of (saved_class, saved_default_parity_mode) entries — one
# per outstanding patch_pycocotools call. The bottom entry is the
# original pycocotools class; nested patches stack on top so unwinding
# in reverse order restores intermediate states for context-manager
# reentrancy (per ADR-0007 §"Reentrancy").
_PATCH_STACK: list[tuple[type, ParityMode]] = []


def patch_pycocotools(parity_mode: ParityMode = "strict") -> Callable[[], None]:
    """Replace ``pycocotools.cocoeval.COCOeval`` with vernier's drop-in.

    Returns an idempotent unpatch callable that restores the previous
    state of ``sys.modules["pycocotools.cocoeval"].COCOeval`` (the
    original pycocotools class on the outermost unpatch). Per ADR-0007:

    - Default ``parity_mode`` is ``"strict"`` because migration intent
      is bit-exactness with pycocotools.
    - Raises :class:`ImportError` if pycocotools is not installed —
      silent fallthrough would let downstream test setups think the
      patch took effect when it did not.
    - Nested calls stack via a module-level LIFO; unpatch handles are
      idempotent (second call is a no-op).

    Not thread-safe — call once at process or test setup, unwind at
    teardown.
    """
    try:
        import pycocotools.cocoeval as cocoeval_mod
    except ImportError as exc:
        raise ImportError(
            "patch_pycocotools requires pycocotools to be installed; "
            "install it with `pip install pycocotools`, or use "
            "`vernier.COCOeval` directly without the patch."
        ) from exc

    saved_class = cocoeval_mod.COCOeval
    saved_default = PycocotoolsCOCOeval.DEFAULT_PARITY_MODE
    _PATCH_STACK.append((saved_class, saved_default))

    PycocotoolsCOCOeval.DEFAULT_PARITY_MODE = parity_mode
    cocoeval_mod.COCOeval = PycocotoolsCOCOeval

    expected_depth = len(_PATCH_STACK)
    unpatched = False

    def unpatch() -> None:
        nonlocal unpatched
        if unpatched:
            return
        unpatched = True
        # Pop the entry this call pushed. If callers unwind out of
        # order (inner unpatch after outer), the LIFO discipline is
        # broken — we still restore *some* prior state so the suite
        # can continue, but we do not try to re-thread the stack.
        if len(_PATCH_STACK) >= expected_depth and _PATCH_STACK:
            prev_class, prev_default = _PATCH_STACK.pop()
            cocoeval_mod.COCOeval = prev_class
            PycocotoolsCOCOeval.DEFAULT_PARITY_MODE = prev_default

    return unpatch


@contextlib.contextmanager
def patched_pycocotools(parity_mode: ParityMode = "strict") -> Generator[None, None, None]:
    """Context-manager variant of :func:`patch_pycocotools`.

    Nests correctly: an outer ``with`` followed by an inner ``with``
    restores the outer-patch state on inner exit and the original
    pycocotools class on outer exit (per ADR-0007 §"Reentrancy").
    """
    unpatch = patch_pycocotools(parity_mode)
    try:
        yield
    finally:
        unpatch()


__all__ = ["patch_pycocotools", "patched_pycocotools"]
