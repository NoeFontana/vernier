"""hotcoco runner — invoked as a subprocess in ``bench/envs/hotcoco``.

hotcoco (Rust/PyO3) exposes pycocotools-shaped ``COCO`` / ``COCOeval``
classes, so the cell runs through the same
:func:`run_cocoeval_pipeline` as pycocotools and faster-coco-eval: the
timed span is identical (JSON path on disk → ``summarize()``), and the
``load`` stage covers JSON parse + index build + ``COCOeval``
construction for all three.

Threading: hotcoco evaluates and accumulates on rayon's global pool,
which defaults to every visible CPU. ``RAYON_NUM_THREADS`` is set to the
cell's CPU budget *before* ``hotcoco`` is imported (rayon reads it once,
when the global pool is first built), so the pool size matches the CPU
set the orchestrator pinned this process to.
"""

from __future__ import annotations

import os
import sys
from importlib.metadata import version as _pkg_version

from bench.harness.cpu_affinity import granted_cpu_count
from bench.runners._protocol import parse_runner_args, run_cocoeval_pipeline


def main() -> int:
    args = parse_runner_args()
    if args.iou_type == "boundary":
        print("hotcoco_runner: boundary IoU is not a hotcoco surface", file=sys.stderr)
        return 2

    os.environ["RAYON_NUM_THREADS"] = str(granted_cpu_count())
    import hotcoco

    run_cocoeval_pipeline(
        args=args,
        impl="hotcoco",
        impl_version=_pkg_version("hotcoco"),
        coco_cls=hotcoco.COCO,
        cocoeval_cls=hotcoco.COCOeval,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
