"""hotcoco LVIS runner — invoked as a subprocess in ``bench/envs/hotcoco``.

Drives hotcoco's lvis-api-shaped surface (``LVIS`` / ``LVISResults`` /
``LVISEval``, i.e. ``COCOeval(..., lvis_style=True)``) through the same
``load → evaluate → accumulate → summarize`` span the ``lvis-api``
runner times, so the three LVIS impls share one timed boundary.

hotcoco's LVIS ``get_results()`` keys are the 13-entry plan
(``AP`` … ``APf``, ``AR@300`` … ``ARl@300``) verbatim, so they map 1:1
onto :func:`lvis_stat_names`. The precision tensor carries a trailing
M-axis of length 1 at ``max_dets=300``; it is squeezed to the
``(T, R, K, A)`` shape the LVIS comparator expects (AF5).

Threading follows :mod:`bench.runners.hotcoco_runner`: the rayon pool
is sized to the cell's CPU budget before ``hotcoco`` is imported.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from importlib.metadata import version as _pkg_version

import numpy as np

from bench.harness.cpu_affinity import cpu_budget
from bench.harness.timing import StageTable
from bench.runners._protocol import lvis_stat_names, parse_lvis_runner_args, write_lvis_outputs


def main() -> int:
    args = parse_lvis_runner_args()
    os.environ["RAYON_NUM_THREADS"] = str(cpu_budget(args.num_threads))
    import hotcoco

    max_dets: int = int(args.max_dets)
    stages = StageTable()
    with contextlib.redirect_stdout(io.StringIO()):
        with stages.stage("load"):
            lvis_gt = hotcoco.LVIS(str(args.gt))
            lvis_dt = hotcoco.LVISResults(lvis_gt, str(args.dt), max_dets=max_dets)
            ev = hotcoco.LVISEval(lvis_gt, lvis_dt, iou_type=args.iou_type)
            ev.params.max_dets = [max_dets]
        with stages.stage("evaluate"):
            ev.evaluate()
        with stages.stage("accumulate"):
            ev.accumulate()
        with stages.stage("summarize"):
            ev.summarize()

    precision = np.asarray(ev.eval["precision"], dtype=np.float64)
    if precision.ndim == 5:
        if precision.shape[-1] != 1:
            raise AssertionError(
                f"hotcoco precision M-axis must be 1 at max_dets={max_dets}; got {precision.shape}"
            )
        precision = np.ascontiguousarray(precision[..., 0])

    results = ev.get_results()
    keys = lvis_stat_names(max_dets)
    missing = [k for k in keys if k not in results]
    if missing:
        raise AssertionError(f"hotcoco LVIS results missing {missing}; got {sorted(results)}")
    summary_stats = {k: float(results[k]) for k in keys}

    stages.record_total()
    write_lvis_outputs(
        args=args,
        impl="hotcoco_lvis",
        impl_version=_pkg_version("hotcoco"),
        stages=stages.to_dict(),
        summary_stats=summary_stats,
        precision_tensor=precision,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
