"""Pytest configuration for the boundary-IoU parity tree.

Adds the vendored `bowenc0221/boundary-iou-api` checkout to ``sys.path``
so ``from boundary_iou.utils.boundary_utils import mask_to_boundary``
resolves to the verbatim upstream copy. The oracle is not pip-installed
(see ADR-0010 §"Oracle (E2 + E3)" and ``oracle/VENDORING.md``).
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

_ORACLE_PATH = Path(__file__).parent / "oracle" / "boundary_iou_api"
if str(_ORACLE_PATH) not in sys.path:
    sys.path.insert(0, str(_ORACLE_PATH))


def _stub_matplotlib() -> None:
    # The vendored oracle's `coco.py` imports `matplotlib` at module
    # scope for `showAnns`-style visualization helpers we never call.
    # Pulling matplotlib into our test deps just to satisfy a top-level
    # import is heavyweight; install minimal stand-ins so the eval path
    # imports cleanly. If a future oracle bump actually exercises the
    # plotting helpers, these stubs raise on attribute access via
    # `ModuleType`'s default `__getattr__` and surface the gap loudly.
    if "matplotlib" in sys.modules:
        return
    for name in ("matplotlib", "matplotlib.pyplot"):
        sys.modules[name] = types.ModuleType(name)
    for sub, member in (("collections", "PatchCollection"), ("patches", "Polygon")):
        mod = types.ModuleType(f"matplotlib.{sub}")
        setattr(mod, member, type(member, (), {}))
        sys.modules[f"matplotlib.{sub}"] = mod


def _force_single_process_boundary_augment() -> None:
    # The oracle's `augment_annotations_with_boundary_multi_core` shards
    # annotations across `multiprocessing.Pool(cpu_count())`. On Python
    # 3.14 the pool surfaces `ConnectionResetError` under our test
    # harness; the single-core variant is functionally identical (the
    # multi-core path just splits + concatenates). Routing the public
    # entry point through the single-core helper sidesteps the failure
    # without diverging from upstream output.
    # Touch the public package so its `__init__.py` imports the
    # multi-core symbol; afterward both modules are in `sys.modules`
    # and the patch can swap the binding via `setattr`.
    __import__("boundary_iou.utils.boundary_utils")
    utils_pkg = sys.modules["boundary_iou.utils"]
    boundary_utils = sys.modules["boundary_iou.utils.boundary_utils"]

    def _single(
        annotations: list[Any],
        ann_to_mask: Callable[[Any], Any],
        dilation_ratio: float = 0.02,
    ) -> list[Any]:
        return boundary_utils.augment_annotations_with_boundary_single_core(
            0, annotations, ann_to_mask, dilation_ratio
        )

    setattr(boundary_utils, "augment_annotations_with_boundary_multi_core", _single)
    setattr(utils_pkg, "augment_annotations_with_boundary_multi_core", _single)


def _restore_numpy_float_alias() -> None:
    # The vendored oracle's `cocoeval.py:434` does
    # `.astype(dtype=np.float)` — `np.float` was the builtin alias
    # removed in NumPy 1.20+. Re-binding the alias keeps the oracle
    # verbatim per ADR-0010 §"Oracle (E2 + E3)" instead of vendoring a
    # patched copy.
    import numpy as _np

    if not hasattr(_np, "float"):
        _np.float = float  # type: ignore[attr-defined]


_stub_matplotlib()
_force_single_process_boundary_augment()
_restore_numpy_float_alias()
