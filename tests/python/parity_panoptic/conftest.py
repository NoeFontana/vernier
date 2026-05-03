"""Pytest configuration for the panoptic parity tree.

Adds the vendored `cocodataset/panopticapi` checkout to ``sys.path``
so ``from panopticapi.evaluation import pq_compute_single_core``
resolves to the verbatim upstream copy. The oracle is not pip-installed
(see ADR-0025 §"Parity strategy" and ``oracle/VENDORING.md``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ORACLE_PATH = Path(__file__).parent / "oracle" / "panopticapi"
if str(_ORACLE_PATH) not in sys.path:
    sys.path.insert(0, str(_ORACLE_PATH))
