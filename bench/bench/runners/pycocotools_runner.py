"""Pycocotools runner — invoked as a subprocess in ``bench/envs/pycocotools``.

Mirrors ``tests/python/parity/harness.py:_run_pycocotools`` so timings
and the precision tensor come from the same path the parity suite
already validates against.
"""

from __future__ import annotations

import sys
from importlib.metadata import version as _pkg_version

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from bench.runners._protocol import parse_runner_args, run_cocoeval_pipeline


def main() -> int:
    args = parse_runner_args()
    if args.iou_type == "boundary":
        print("pycocotools_runner: boundary IoU is not a pycocotools surface", file=sys.stderr)
        return 2

    run_cocoeval_pipeline(
        args=args,
        impl="pycocotools",
        impl_version=_pkg_version("pycocotools"),
        coco_cls=COCO,
        cocoeval_cls=COCOeval,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
