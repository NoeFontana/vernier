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

from bench.runners._protocol import parse_runner_args, run_cocoeval_pipeline  # noqa: E402


def main() -> int:
    args = parse_runner_args()
    # faster-coco-eval exposes parallelism only on boundary IoU via the
    # ``boundary_cpu_count`` constructor kwarg — it controls the
    # ``calculateRleForAllAnnotations`` step that boundary `_prepare()`
    # runs. bbox / segm / keypoints have no thread knob in
    # faster-coco-eval, so the ADR-0047 ``--num-threads`` axis is a
    # no-op there and is intentionally not forwarded.
    cocoeval_kwargs: dict[str, Any] = {}
    if args.iou_type == "boundary" and args.num_threads is not None:
        cocoeval_kwargs["boundary_cpu_count"] = args.num_threads
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
