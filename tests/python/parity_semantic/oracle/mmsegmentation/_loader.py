"""Reusable stub-injection helper for the vendored mmsegmentation oracle.

Lifted out of ``conftest.py`` so the bench runner (which lives outside
the pytest tree) can use the same code path. Calling
:func:`install_stubs` is idempotent and must precede any
``from mmseg.evaluation.metrics.iou_metric import IoUMetric``.

The stubs satisfy ``iou_metric.py``'s top-level imports without
pulling mmcv / mmengine / the mmsegmentation package itself
(~3 GB transitive). See ``VENDORING.md`` and ADR-0036 for the
parity contract.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ORACLE_PATH: Path = Path(__file__).parent


def install_stubs() -> None:
    """Make the vendored ``IoUMetric`` importable in the current process.

    Inserts ``ORACLE_PATH`` into :data:`sys.path` (so the in-tree
    ``mmseg`` package shadowed at that path resolves) and registers
    minimal stubs for ``mmengine.{dist,evaluator,logging,utils}``,
    ``mmseg.registry`` and ``prettytable`` in :data:`sys.modules`.

    Idempotent: re-calling is a no-op for already-installed modules.
    """
    if str(ORACLE_PATH) not in sys.path:
        sys.path.insert(0, str(ORACLE_PATH))

    # The stub itself lives at oracle/_mmengine_stub.py; importing
    # works once ORACLE_PATH is on sys.path.
    import _mmengine_stub  # pyright: ignore[reportMissingImports]

    _install_stub_module("mmengine", {})
    _install_stub_module("mmengine.dist", {"is_main_process": _mmengine_stub.is_main_process})
    _install_stub_module("mmengine.evaluator", {"BaseMetric": _mmengine_stub.BaseMetric})
    _install_stub_module(
        "mmengine.logging",
        {"MMLogger": _mmengine_stub.MMLogger, "print_log": _mmengine_stub.print_log},
    )
    _install_stub_module("mmengine.utils", {"mkdir_or_exist": _mmengine_stub.mkdir_or_exist})
    _install_stub_module("mmseg.registry", {"METRICS": _mmengine_stub.METRICS})
    _install_stub_module("prettytable", {"PrettyTable": _mmengine_stub.PrettyTable})


def _install_stub_module(name: str, attrs: dict[str, object]) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
