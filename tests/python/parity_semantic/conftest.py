"""Pytest configuration for the semantic-segmentation parity tree.

Adds the vendored mmsegmentation oracle to ``sys.path`` and registers
the minimal `mmengine` / `mmseg.registry` stubs from
``_mmengine_stub.py`` in ``sys.modules``, so that

    from mmseg.evaluation.metrics.iou_metric import IoUMetric

resolves the verbatim upstream copy without pulling mmcv / mmengine /
the mmsegmentation package itself (~3 GB transitive). See
``oracle/mmsegmentation/VENDORING.md`` for the parity contract and
ADR-0036 for the vendoring decision.

`torch` is a real test dependency: ``IoUMetric.intersect_and_union``
calls ``torch.histc`` for label binning, which numpy does not
replicate bit-exactly. The pin lives in ``pyproject.toml`` and
mirrors into ``crates/vernier-semantic/src/parity.rs::ORACLE_TORCH_PIN``.
Tests skip cleanly if torch is absent (parity tests are gated on
``torch`` being installed; ``just lint`` does not require it).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_ORACLE_PATH = Path(__file__).parent / "oracle" / "mmsegmentation"

# Skip collection of every test in this tree if torch is unavailable.
# Mirrors the `real_models` extra pattern: parity tests are opt-in and
# do not gate the default `just test` loop on the heavy torch wheel.
torch = pytest.importorskip(
    "torch",
    reason=(
        "mmsegmentation IoUMetric.intersect_and_union calls torch.histc; "
        "install with `uv sync --group dev` (torch is in dev-deps)."
    ),
)

if str(_ORACLE_PATH) not in sys.path:
    sys.path.insert(0, str(_ORACLE_PATH))

# Stubs replace mmengine.{dist,evaluator,logging,utils} and
# mmseg.registry before iou_metric.py is imported. The vendored file
# is byte-equal to upstream; the stubs satisfy its top-level imports
# without pulling the real packages.
import _mmengine_stub  # noqa: E402  -- after sys.path mutation  # pyright: ignore[reportMissingImports]


def _install_stub_module(name: str, attrs: dict[str, object]) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


# Parents must exist before children, so set up `mmengine` then its
# submodules. The same applies to `mmseg.registry` (the `mmseg`
# namespace package itself lives in ``oracle/mmsegmentation/mmseg/``
# and is reached through sys.path; we only need to inject `registry`).
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
