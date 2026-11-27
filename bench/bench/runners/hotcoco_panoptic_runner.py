"""hotcoco panoptic runner — invoked as a subprocess in ``bench/envs/hotcoco``.

Drives ``hotcoco.panoptic.PanopticEval`` (added in hotcoco 1.3.0) over the
four-path COCO panoptic format (GT/DT PNG dirs + JSONs) so it shares the
timed boundary with ``vernier_panoptic`` and ``panopticapi``.

Threading: ``PanopticEval.evaluate()`` evaluates images in parallel on
hotcoco's rayon global pool, so ``RAYON_NUM_THREADS`` is set to the cell's
CPU budget *before* ``hotcoco`` is imported (matching ``hotcoco_runner`` and
``hotcoco_lvis_runner``).
"""

from __future__ import annotations

import os
import sys
from importlib.metadata import version as _pkg_version
from typing import Any

from bench.harness.cpu_affinity import granted_cpu_count
from bench.harness.parity import PanopticSnapshot
from bench.harness.timing import StageTable
from bench.runners._protocol import (
    parse_panoptic_runner_args,
    per_class_uint64_table,
    write_panoptic_outputs,
)


def _bucket_vals(bucket: dict[str, Any]) -> tuple[float, float, float, int]:
    """Extract ``(pq, sq, rq, n)`` from a hotcoco ``results()`` bucket,
    normalizing empty-split ``-1.0`` sentinels to ``0.0`` to match
    ``vernier_panoptic_runner``."""
    n = int(bucket["n"])
    if n == 0:
        return (0.0, 0.0, 0.0, 0)
    return (
        max(0.0, float(bucket["pq"])),
        max(0.0, float(bucket["sq"])),
        max(0.0, float(bucket["rq"])),
        n,
    )


def _results_to_snapshot(results: dict[str, Any]) -> PanopticSnapshot:
    """Project hotcoco ``PanopticEval.results()`` into ``PanopticSnapshot``.

    hotcoco's ``per_class`` includes every GT category, using ``-1.0``
    for categories with ``tp == fp == fn == 0``; ``panopticapi`` and
    ``vernier_panoptic`` omit unevaluated categories from ``per_class``,
    so filter those zero-count rows out here.
    """
    pq, sq, rq, n = _bucket_vals(results["All"])
    pq_things, sq_things, rq_things, n_things = _bucket_vals(results["Things"])
    pq_stuff, sq_stuff, rq_stuff, n_stuff = _bucket_vals(results["Stuff"])

    per_class: dict[str, dict[str, float]] = {}
    for cat_id, row in results.get("per_class", {}).items():
        if int(row.get("tp", 0)) == 0 and int(row.get("fp", 0)) == 0 and int(row.get("fn", 0)) == 0:
            continue
        per_class[str(int(cat_id))] = {
            "pq": float(row["pq"]),
            "sq": float(row["sq"]),
            "rq": float(row["rq"]),
        }

    return PanopticSnapshot(
        pq=pq,
        sq=sq,
        rq=rq,
        n=n,
        pq_things=pq_things,
        sq_things=sq_things,
        rq_things=rq_things,
        n_things=n_things,
        pq_stuff=pq_stuff,
        sq_stuff=sq_stuff,
        rq_stuff=rq_stuff,
        n_stuff=n_stuff,
        per_class=per_class,
    )


def main() -> int:
    args = parse_panoptic_runner_args()
    os.environ["RAYON_NUM_THREADS"] = str(granted_cpu_count())
    import hotcoco

    stages = StageTable()
    with stages.stage("load"):
        ev = hotcoco.panoptic.PanopticEval(
            str(args.gt_json),
            str(args.dt_json),
            gt_folder=str(args.gt_png_dir),
            pred_folder=str(args.dt_png_dir),
        )

    with stages.stage("evaluate"):
        ev.evaluate()

    with stages.stage("aggregate"):
        results = ev.results()
        snap = _results_to_snapshot(results)
        per_class_array = per_class_uint64_table(snap.per_class, columns=("pq", "sq", "rq"))
        snap_json = snap.model_dump_json().encode()

    stages.record_total()

    write_panoptic_outputs(
        args=args,
        impl="hotcoco_panoptic",
        impl_version=_pkg_version("hotcoco"),
        stages=stages.to_dict(),
        snapshot_json_bytes=snap_json,
        per_class_array=per_class_array,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
