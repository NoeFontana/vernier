"""Naive-Python baseline runner — runs in ``bench/envs/pycocotools``.

This runner is the user-pattern baseline ADR-0033 §"streaming" pins
the streaming evaluator's RSS-shape claim against:

    predictions = []
    for batch in loader:
        for det in batch:
            predictions.append(det)
    cocoeval = COCOeval(gt, dt_loadres(predictions), iou_type)
    cocoeval.evaluate(); cocoeval.accumulate(); cocoeval.summarize()

The list-grow-then-evaluate loop's RSS scales linearly with prediction
count; vernier's streaming surface holds constant. This runner emits
the same multi-artifact bundle (``stats.json`` + ``rss_curve.json``)
as the vernier streaming runner so the comparator can do an apples-
to-apples wall-time delta + RSS-curve diff.

Stages: ``load``, ``accumulate_in_python`` (the
``predictions.append(...)`` loop), ``cocoeval_evaluate``,
``cocoeval_accumulate``, ``cocoeval_summarize``, ``total``. The
``accumulate_in_python`` stage is what the streaming surface is
trying to obviate; nameing it explicitly here makes the comparator's
delta target legible.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import time
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from bench.harness.rss import RSSSampler
from bench.harness.schema import BenchWarning, RunnerRepOutput, StageTimings
from bench.harness.timing import StageTable

_STATS_KEY = "summary"
_RSS_CURVE_KEY = "rss_curve"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--dt", type=Path, required=True)
    p.add_argument(
        "--iou-type",
        choices=["bbox", "segm", "boundary", "keypoints"],
        default="bbox",
    )
    p.add_argument("--workload-id", type=str, required=True)
    p.add_argument("--output", type=Path, required=True)
    # Accept the same --mode-flag the streaming runner does so the
    # orchestrator's spawn argv shape is uniform; only ``vs_naive`` is
    # currently a meaningful value here.
    p.add_argument(
        "--mode-flag",
        choices=["throughput", "vs_naive", "dlpack"],
        required=True,
    )
    return p.parse_args()


def _emit_outputs(
    *,
    output_path: Path,
    workload_id: str,
    iou_type: str,
    impl: str,
    impl_version: str,
    stages: dict[str, StageTimings],
    summary_stats: dict[str, float],
    rss_samples: list[tuple[float, int]],
    warnings: list[BenchWarning] | None = None,
) -> None:
    """Mirror of the vernier streaming runner's output emitter — same
    ``artifact_paths`` slot names so the comparator pipes through one
    code path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_paths: dict[str, str] = {}
    artifact_sha256: dict[str, str] = {}

    stats_path = output_path.parent / f"{impl}-stats.json"
    stats_payload = json.dumps(
        {"summary_stats": summary_stats, "iou_type": iou_type},
        sort_keys=True,
        indent=2,
    ).encode("utf-8")
    stats_path.write_bytes(stats_payload)
    artifact_paths[_STATS_KEY] = stats_path.name
    artifact_sha256[_STATS_KEY] = hashlib.sha256(stats_payload).hexdigest()

    rss_path = output_path.parent / f"{impl}-rss.json"
    rss_payload = json.dumps(
        {
            "samples": [{"t_s": t, "rss_bytes": rss} for t, rss in rss_samples],
            "interval_s": 0.1,
        },
        indent=2,
    ).encode("utf-8")
    rss_path.write_bytes(rss_payload)
    artifact_paths[_RSS_CURVE_KEY] = rss_path.name
    artifact_sha256[_RSS_CURVE_KEY] = hashlib.sha256(rss_payload).hexdigest()

    rep_output = RunnerRepOutput(
        paradigm="streaming",
        impl=impl,
        impl_version=impl_version,
        iou_type=iou_type,  # type: ignore[arg-type]
        workload_id=workload_id,
        stages=stages,
        summary_stats=summary_stats,
        artifact_paths=artifact_paths,
        artifact_sha256=artifact_sha256,
        warnings=list(warnings or []),
    )
    output_path.write_text(rep_output.model_dump_json(indent=2))


def main() -> int:
    args = _parse_args()
    stages = StageTable()

    with RSSSampler() as sampler, stages.stage("total"):
        with stages.stage("load"):
            gt_bytes = args.gt.read_bytes()
            dt_bytes = args.dt.read_bytes()

        # The user pattern is "append per-image to a list, then evaluate
        # the union" — we reproduce the shape by JSON-decoding into a
        # list and re-encoding once. The list lives in Python until the
        # ``cocoeval_evaluate`` stage starts; the RSS curve captures
        # the steady-state grow.
        with stages.stage("accumulate_in_python"):
            predictions: list[Any] = []
            # Explicitly iterate so the ``append`` is a real RSS-grower
            # rather than a one-shot list literal copy. Mirrors the user
            # pattern in the plan.
            for det in json.loads(dt_bytes):
                predictions.append(det)
            dt_payload = json.dumps(predictions).encode("utf-8")
            del predictions

        with contextlib.redirect_stdout(io.StringIO()):
            with stages.stage("cocoeval_evaluate"):
                # ``COCO`` parses GT JSON from disk; pass the path so the
                # cost is the same one a user-pattern caller would pay.
                gt = COCO(str(args.gt))
                # ``loadRes`` accepts a list/np.ndarray as well as a file
                # path; use the in-memory list to skip a disk round-trip
                # that the streaming runner doesn't pay either.
                dt = gt.loadRes(json.loads(dt_payload))
                cocoeval = COCOeval(gt, dt, iouType=args.iou_type)
                cocoeval.evaluate()
            with stages.stage("cocoeval_accumulate"):
                cocoeval.accumulate()
            with stages.stage("cocoeval_summarize"):
                cocoeval.summarize()

        # Extract Summary.stats positionally — the streaming-vs-naive
        # comparator compares stat_<i> on both sides, so don't rely on
        # the impl-specific stat name table here.
        raw_stats = list(cocoeval.stats)
        summary_stats = {f"stat_{i}": float(v) for i, v in enumerate(raw_stats)}

    # ``gt_bytes`` is held only to make the load stage non-trivial; it's
    # released to the GC at function exit. The naive-Python user
    # pattern's RSS peak is dominated by the prediction list, not GT
    # JSON, so this is fine.
    del gt_bytes

    _emit_outputs(
        output_path=args.output,
        workload_id=args.workload_id,
        iou_type=args.iou_type,
        impl="naive_python",
        impl_version=_pkg_version("pycocotools"),
        stages=stages.to_dict(),
        summary_stats=summary_stats,
        rss_samples=sampler.samples,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
