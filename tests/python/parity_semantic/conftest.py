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
from pathlib import Path

import pytest

# Skip collection of every test in this tree if torch is unavailable.
# Mirrors the `real_models` extra pattern: parity tests are opt-in and
# do not gate the default `just test` loop on the heavy torch wheel.
pytest.importorskip(
    "torch",
    reason=(
        "mmsegmentation IoUMetric.intersect_and_union calls torch.histc; "
        "install with `uv sync --group dev` (torch is in dev-deps)."
    ),
)

# Ensure the loader module is reachable, then delegate stub setup to it
# so the bench runner can use the same code path.
_ORACLE_PATH = Path(__file__).parent / "oracle" / "mmsegmentation"
if str(_ORACLE_PATH) not in sys.path:
    sys.path.insert(0, str(_ORACLE_PATH))

from _loader import install_stubs  # noqa: E402  # pyright: ignore[reportMissingImports]

install_stubs()
