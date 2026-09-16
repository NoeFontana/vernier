"""faster-coco-eval runner — invoked as a subprocess in
``bench/envs/faster-coco-eval``.

Bootstrap order is load-bearing: faster-coco-eval registers its drop-in
``COCO`` / ``COCOeval`` by mutating the ``pycocotools`` namespace. That
must run *before* any ``from pycocotools...`` import — otherwise the
unmodified pycocotools names get bound and the runner silently
benchmarks pycocotools instead.

Supported iouType set spans bbox / segm / keypoints / boundary —
faster-coco-eval (≥1.6) ships its own boundary surface alongside the
COCOeval drop-in, with the ``boundary_dilation_ratio`` default tracking
the boundary-iou-api 0.02 reference. Numerical agreement with
``boundary-iou-api`` at the parity tensor level is not asserted here;
the cell is timing-only.

Threading (≥1.8): faster-coco-eval parallelizes RLE IoU on a Python
thread pool (``rle_iou_max_workers``), boundary preparation
(``boundary_cpu_count``), and the C++ image-evaluation / accumulation
loops, which size themselves from ``std::thread::hardware_concurrency()``
with no knob. The runner forwards the cell's CPU budget to both knobs;
the C++ pools are bounded only by the CPU affinity the orchestrator pins
this process to (they may still spawn more threads than CPUs, which is
the library's own behavior on a constrained host).
"""

from __future__ import annotations

# --- bootstrap (must come first) ------------------------------------------
import faster_coco_eval

faster_coco_eval.init_as_pycocotools()
# --- /bootstrap -----------------------------------------------------------

import sys  # noqa: E402
from importlib.metadata import version as _pkg_version  # noqa: E402
from typing import Any  # noqa: E402

from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from bench.harness.cpu_affinity import granted_cpu_count  # noqa: E402
from bench.runners._protocol import parse_runner_args, run_cocoeval_pipeline  # noqa: E402


def main() -> int:
    args = parse_runner_args()
    budget = granted_cpu_count()
    cocoeval_kwargs: dict[str, Any] = {"rle_iou_max_workers": budget}
    if args.iou_type == "boundary":
        cocoeval_kwargs["boundary_cpu_count"] = budget
    run_cocoeval_pipeline(
        args=args,
        impl="faster-coco-eval",
        impl_version=_pkg_version("faster-coco-eval"),
        coco_cls=COCO,
        cocoeval_cls=COCOeval,
        cocoeval_kwargs=cocoeval_kwargs,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
