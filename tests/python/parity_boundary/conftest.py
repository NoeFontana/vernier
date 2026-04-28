"""Pytest configuration for the boundary-IoU parity tree.

Adds the vendored `bowenc0221/boundary-iou-api` checkout to ``sys.path``
so ``from boundary_iou.utils.boundary_utils import mask_to_boundary``
resolves to the verbatim upstream copy. The oracle is not pip-installed
(see ADR-0010 §"Oracle (E2 + E3)" and ``oracle/VENDORING.md``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ORACLE_PATH = Path(__file__).parent / "oracle" / "boundary_iou_api"
if str(_ORACLE_PATH) not in sys.path:
    sys.path.insert(0, str(_ORACLE_PATH))
