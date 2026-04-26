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

from vernier._compat import ParityMode, PycocotoolsCOCOeval

# The original pycocotools class, captured on the first patch and
# released only when the patch count returns to zero. Per ADR-0007
# §"Idempotency": "The helper records the original on first call only."
_original_cocoeval: type | None = None

# Stack of saved DEFAULT_PARITY_MODE values, one per outstanding patch.
# Pop-on-unpatch gives reentrant restoration of the class-var so nested
# context managers (per ADR-0007 §"Reentrancy") see the outer's mode
# again on inner exit.
_parity_mode_stack: list[ParityMode] = []


def patch_pycocotools(parity_mode: ParityMode = "strict") -> Callable[[], None]:
    """Replace ``pycocotools.cocoeval.COCOeval`` with vernier's drop-in.

    Returns an idempotent unpatch callable. The outermost unpatch
    restores the original pycocotools class; intermediate unpatches
    pop the parity-mode stack so nested
    :func:`patched_pycocotools` exits restore the surrounding patch
    state. Per ADR-0007:

    - Default ``parity_mode`` is ``"strict"`` because migration intent
      is bit-exactness with pycocotools.
    - Raises :class:`ImportError` if pycocotools is not installed —
      silent fallthrough would let downstream test setups think the
      patch took effect when it did not.
    - Calling ``patch_pycocotools`` repeatedly without unpatching
      stacks the parity-mode state but never overwrites the saved
      original class, so the final unpatch always restores the real
      pycocotools class.

    Not thread-safe — call once at process or test setup, unwind at
    teardown.
    """
    global _original_cocoeval

    try:
        import pycocotools.cocoeval as cocoeval_mod
    except ImportError as exc:
        raise ImportError(
            "patch_pycocotools requires pycocotools to be installed; "
            "install it with `pip install pycocotools`, or use "
            "`vernier.COCOeval` directly without the patch."
        ) from exc

    if _original_cocoeval is None:
        _original_cocoeval = cocoeval_mod.COCOeval

    _parity_mode_stack.append(PycocotoolsCOCOeval.DEFAULT_PARITY_MODE)
    PycocotoolsCOCOeval.DEFAULT_PARITY_MODE = parity_mode
    cocoeval_mod.COCOeval = PycocotoolsCOCOeval

    unpatched = False

    def unpatch() -> None:
        nonlocal unpatched
        global _original_cocoeval
        if unpatched:
            return
        unpatched = True
        if _parity_mode_stack:
            PycocotoolsCOCOeval.DEFAULT_PARITY_MODE = _parity_mode_stack.pop()
        if not _parity_mode_stack and _original_cocoeval is not None:
            cocoeval_mod.COCOeval = _original_cocoeval
            _original_cocoeval = None

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
