"""ultrafast-pycocotools runner — invoked as a subprocess in ``bench/envs/ufcoco``.

ufcoco (Rust/PyO3) exposes pycocotools-shaped ``COCO`` / ``COCOeval``
classes, so the cell runs through the same
:func:`run_cocoeval_pipeline` as pycocotools, faster-coco-eval and
hotcoco: the timed span is identical (JSON path on disk →
``summarize()``), and the ``load`` stage covers JSON parse + index build
+ ``COCOeval`` construction.

Threading: ufcoco evaluates on rayon's global pool, which defaults to
every visible CPU. ``RAYON_NUM_THREADS`` is set to the cell's CPU budget
*before* ``ultrafast_pycocotools`` is imported (rayon reads it once, when
the global pool is first built), so the pool size matches the CPU set
the orchestrator pinned this process to.

Boundary IoU is ufcoco's own extension of ``COCOeval``
(``iouType="boundary"``); its ``boundary_dilation_ratio`` default is the
boundary-iou-api reference's 0.02, the same as vernier's.
Its boundary output nonetheless differs from that reference's, so the
parity matrix pairs ufcoco only on bbox / segm / keypoints.
"""

from __future__ import annotations

import os
import sys
from importlib.metadata import version as _pkg_version

from bench.harness.cpu_affinity import granted_cpu_count
from bench.runners._protocol import parse_runner_args, run_cocoeval_pipeline


def main() -> int:
    args = parse_runner_args()
    os.environ["RAYON_NUM_THREADS"] = str(granted_cpu_count())
    from ultrafast_pycocotools.coco import COCO
    from ultrafast_pycocotools.cocoeval import COCOeval

    run_cocoeval_pipeline(
        args=args,
        impl="ufcoco",
        impl_version=_pkg_version("ultrafast-pycocotools"),
        coco_cls=COCO,
        cocoeval_cls=COCOeval,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
