"""BackgroundEvaluator p99 latency runner — streaming-paradigm cell.

Spins up a saturation feeder thread that pushes detections continuously
into ``vernier.instance.BackgroundEvaluator`` (constructed with
``record_latency_samples=True``) for a fixed duration, then drains the
per-submit nanosecond samples and emits an HDR-histogram-style CDF
(p50/p90/p99/p999/max) per queue depth.

Three queue capacities are exercised per run: ``{1, 8, 64}``. The
artifact is a single ``latency_cdf.json`` with one section per depth.

The cell's parity tier is *informational* — there is no oracle for tail
latency. The streaming comparator records ``parity_tier="informational"``
on the cell metadata; regression detection (warn when ``current p99 >
1.20 * prior p99``) is layered above the artifact in the report layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from collections.abc import Sequence
from itertools import cycle
from pathlib import Path
from typing import Any

import vernier
from vernier.instance import BackgroundEvaluator

from bench.harness.schema import BenchWarning, RunnerRepOutput, StageTimings
from bench.harness.timing import StageTable

_LATENCY_CDF_KEY = "latency_cdf"
_QUEUE_CAPACITIES: tuple[int, ...] = (1, 8, 64)
_PERCENTILES: tuple[float, ...] = (0.5, 0.9, 0.99, 0.999, 1.0)
_DEFAULT_DURATION_S = 60.0


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
    p.add_argument(
        "--duration-s",
        type=float,
        default=_DEFAULT_DURATION_S,
        help=(
            f"Saturation duration per queue depth. Default: {_DEFAULT_DURATION_S}s. "
            "Override with a small value for smoke / dev runs."
        ),
    )
    return p.parse_args()


def _group_dt_by_image(dt_records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_img: dict[int, list[dict[str, Any]]] = {}
    for det in dt_records:
        by_img.setdefault(int(det["image_id"]), []).append(det)
    return by_img


def _records_to_per_image_bytes(
    by_img: dict[int, list[dict[str, Any]]],
) -> list[bytes]:
    """Serialize each image's DT list to a JSON ``bytes`` payload —
    the ``BackgroundEvaluator.submit`` shape that round-trips through
    the JSON ingest path. Pre-serializing keeps the feeder thread's
    hot path a constant-cost ``submit`` call."""
    return [json.dumps(records).encode("utf-8") for records in by_img.values()]


def _percentiles(samples_ns: Sequence[int]) -> dict[str, float]:
    """Return ``{p50, p90, p99, p999, max}`` in microseconds.

    Microseconds keep the JSON readable; the raw ns vector is preserved
    in the histogram artifact for downstream HDR analysis.
    """
    if not samples_ns:
        return {label: 0.0 for label in ("p50", "p90", "p99", "p999", "max")}
    sorted_ns = sorted(samples_ns)
    n = len(sorted_ns)
    out: dict[str, float] = {}
    for q, label in zip(_PERCENTILES, ("p50", "p90", "p99", "p999", "max"), strict=True):
        idx = min(int(q * n), n - 1)
        out[label] = sorted_ns[idx] / 1000.0  # ns → µs
    return out


def _saturate_one_queue(
    *,
    gt_bytes: bytes,
    iou_type: str,
    queue_capacity: int,
    payloads: list[bytes],
    duration_s: float,
) -> tuple[list[int], dict[str, float]]:
    """Run the saturation loop at one queue depth. Returns
    ``(latency_samples_ns, percentile_dict)``."""
    bg = BackgroundEvaluator(
        gt_bytes,
        iou_type=iou_type,  # type: ignore[arg-type]
        queue_capacity=queue_capacity,
        record_latency_samples=True,
    )
    stop_event = threading.Event()
    submission_count = [0]

    def _feeder() -> None:
        for payload in cycle(payloads):
            if stop_event.is_set():
                return
            try:
                bg.submit(payload)
            except Exception:
                # The evaluator may shut down while a submit is in
                # flight; bail rather than crashing the feeder thread.
                return
            submission_count[0] += 1

    feeder = threading.Thread(target=_feeder, daemon=True)
    feeder.start()
    try:
        time.sleep(duration_s)
    finally:
        stop_event.set()
        feeder.join(timeout=2.0)

    samples = bg.drain_latency_samples_ns()
    bg.__exit__(None, None, None)
    return list(samples), _percentiles(samples)


def _emit_outputs(
    *,
    output_path: Path,
    workload_id: str,
    iou_type: str,
    impl: str,
    impl_version: str,
    stages: dict[str, StageTimings],
    cdf_per_capacity: dict[int, dict[str, Any]],
    warnings: list[BenchWarning] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, str] = {}
    artifact_sha256: dict[str, str] = {}

    cdf_path = output_path.parent / f"{impl}-latency_cdf.json"
    cdf_payload = json.dumps(
        {
            "queue_capacities": list(cdf_per_capacity.keys()),
            "per_capacity": {str(q): cdf_per_capacity[q] for q in cdf_per_capacity},
            "regression_threshold": 1.20,
        },
        sort_keys=True,
        indent=2,
    ).encode("utf-8")
    cdf_path.write_bytes(cdf_payload)
    artifact_paths[_LATENCY_CDF_KEY] = cdf_path.name
    artifact_sha256[_LATENCY_CDF_KEY] = hashlib.sha256(cdf_payload).hexdigest()

    summary_stats: dict[str, float] = {}
    for q, section in cdf_per_capacity.items():
        for label, value in section["percentiles_us"].items():
            summary_stats[f"q{q}_{label}_us"] = float(value)
        summary_stats[f"q{q}_n_samples"] = float(section["n_samples"])

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

    with stages.stage("load"):
        gt_bytes = args.gt.read_bytes()
        dt_records = json.loads(args.dt.read_bytes())
        by_img = _group_dt_by_image(dt_records)
        payloads = _records_to_per_image_bytes(by_img)
        if not payloads:
            print(
                "[vernier_bg_p99_runner] dt has no records — saturation feeder would idle",
                file=sys.stderr,
            )
            return 1

    cdf_per_capacity: dict[int, dict[str, Any]] = {}
    for queue_capacity in _QUEUE_CAPACITIES:
        with stages.stage(f"saturate_q{queue_capacity}"):
            samples_ns, percentiles_us = _saturate_one_queue(
                gt_bytes=gt_bytes,
                iou_type=args.iou_type,
                queue_capacity=queue_capacity,
                payloads=payloads,
                duration_s=args.duration_s,
            )
        cdf_per_capacity[queue_capacity] = {
            "n_samples": len(samples_ns),
            "duration_s": args.duration_s,
            "percentiles_us": percentiles_us,
        }

    _emit_outputs(
        output_path=args.output,
        workload_id=args.workload_id,
        iou_type=args.iou_type,
        impl="vernier_bg",
        impl_version=vernier.__version__,
        stages=stages.to_dict(),
        cdf_per_capacity=cdf_per_capacity,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
