"""Pytest configuration for the LVIS parity tree.

Adds the vendored `lvis-dataset/lvis-api` checkout to ``sys.path`` so
``from lvis import LVIS, LVISResults, LVISEval`` resolves to the
verbatim upstream at the SHA recorded in
``oracle/VENDORING.md``. The oracle is also installed as a PyPI
distribution (`lvis==0.5.3` in ``[dependency-groups].dev``), so two
import sources coexist: the dev-install satisfies tooling that
imports `lvis` at module load, while the vendored copy is the
authoritative parity oracle. The path insertion below biases imports
toward the vendored tree at test time.

Per ADR-0026 §"Parity strategy" the oracle is *not* edited; any
runtime-side compat shims (e.g., NumPy deprecations) live in this
conftest, not in the vendored source.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ORACLE_PATH = Path(__file__).parent / "oracle" / "lvis_api"
if str(_ORACLE_PATH) not in sys.path:
    sys.path.insert(0, str(_ORACLE_PATH))


def _restore_numpy_float_alias() -> None:
    # The vendored 0.5.3 release predates NumPy 1.20's removal of
    # `np.float`. Modern NumPy raises `AttributeError` on access; the
    # upstream master HEAD has the fix (commit `d5a663fb`, Feb 2024)
    # but no PyPI release carries it. Re-binding the alias keeps the
    # oracle verbatim per the no-modifications invariant; if a future
    # vendoring bumps the SHA past the fix, this shim becomes a no-op.
    import numpy as _np

    if not hasattr(_np, "float"):
        _np.float = float  # type: ignore[attr-defined]


_restore_numpy_float_alias()
