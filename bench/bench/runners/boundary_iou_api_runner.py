"""boundary-iou-api runner — invoked as a subprocess in
``bench/envs/boundary-iou-api``.

The oracle is a verbatim symlinked checkout at ``./oracle/`` (see
``VENDORING.md`` next to this file). The orchestrator puts that
directory on ``PYTHONPATH`` via ``uv_run_env`` so the runner doesn't
have to mutate ``sys.path`` itself; the bootstrap section below covers
the remaining import-time fixups (matplotlib stub, single-core
multiprocessing, ``np.float`` alias) that mirror
``tests/python/parity_boundary/conftest.py``.
"""

from __future__ import annotations

# --- bootstrap (must come first) ------------------------------------------
import sys
import types
from collections.abc import Callable
from typing import Any


def _stub_matplotlib() -> None:
    if "matplotlib" in sys.modules:
        return
    for name in ("matplotlib", "matplotlib.pyplot"):
        sys.modules[name] = types.ModuleType(name)
    for sub, member in (("collections", "PatchCollection"), ("patches", "Polygon")):
        mod = types.ModuleType(f"matplotlib.{sub}")
        setattr(mod, member, type(member, (), {}))
        sys.modules[f"matplotlib.{sub}"] = mod


def _force_single_process_boundary_augment() -> None:
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

    boundary_utils.augment_annotations_with_boundary_multi_core = _single
    utils_pkg.augment_annotations_with_boundary_multi_core = _single


def _restore_numpy_float_alias() -> None:
    import numpy as _np

    if not hasattr(_np, "float"):
        _np.float = float  # type: ignore[attr-defined]


_stub_matplotlib()
_restore_numpy_float_alias()
_force_single_process_boundary_augment()
# --- /bootstrap -----------------------------------------------------------

from boundary_iou.coco_instance_api.coco import COCO  # noqa: E402
from boundary_iou.coco_instance_api.cocoeval import COCOeval  # noqa: E402

from bench.runners._protocol import parse_runner_args, run_cocoeval_pipeline  # noqa: E402

# Mirrors ``ORACLE_COMMIT_SHA`` in
# ``crates/vernier-core/src/boundary_parity.rs`` and the table in
# ``tests/python/parity_boundary/oracle/VENDORING.md``; all three
# update atomically when the oracle is refreshed.
_ORACLE_SHA_PREFIX = "37d25586a677"


def main() -> int:
    args = parse_runner_args()
    if args.iou_type != "boundary":
        print(
            f"boundary_iou_api_runner: --iou-type {args.iou_type} is not supported; "
            "this runner only serves the boundary cell.",
            file=sys.stderr,
        )
        return 2

    run_cocoeval_pipeline(
        args=args,
        impl="boundary-iou-api",
        impl_version=_ORACLE_SHA_PREFIX,
        coco_cls=COCO,
        cocoeval_cls=COCOeval,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
