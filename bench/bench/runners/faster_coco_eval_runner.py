"""faster-coco-eval runner — invoked as a subprocess in
``bench/envs/faster-coco-eval``.

Bootstrap order is load-bearing: faster-coco-eval registers its drop-in
``COCO`` / ``COCOeval`` by mutating the ``pycocotools`` namespace. That
must run *before* any ``from pycocotools...`` import — otherwise the
unmodified pycocotools names get bound and the runner silently
benchmarks pycocotools instead.
"""

from __future__ import annotations

# --- bootstrap (must come first) ------------------------------------------
import faster_coco_eval

faster_coco_eval.init_as_pycocotools()
# --- /bootstrap -----------------------------------------------------------

import sys  # noqa: E402
from importlib.metadata import version as _pkg_version  # noqa: E402

from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from bench.runners._protocol import parse_runner_args, run_cocoeval_pipeline  # noqa: E402


def main() -> int:
    args = parse_runner_args()
    if args.iou_type == "boundary":
        print(
            "faster_coco_eval_runner: boundary IoU is not a faster-coco-eval surface",
            file=sys.stderr,
        )
        return 2

    run_cocoeval_pipeline(
        args=args,
        impl="faster-coco-eval",
        impl_version=_pkg_version("faster-coco-eval"),
        coco_cls=COCO,
        cocoeval_cls=COCOeval,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
