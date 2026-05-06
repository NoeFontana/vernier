"""Vernier streaming runner — runs in ``bench/envs/vernier`` (no new env).

Three sub-modes gated by ``--mode-flag``:

- ``throughput`` — measures per-image ``StreamingEvaluator.update()``
  + ``finalize()`` over the GT/DT pair; emits the bit-equal
  ``Summary.stats`` (asserted against batch by the comparator) and
  per-stage timings. The hot stage is ``update_per_image``.
- ``vs_naive`` — emits the same artifact bundle as ``throughput``;
  the result-store path segment ``vs_naive`` distinguishes the two
  cells. Comparator pairs the produced summary against the
  ``naive_python`` runner's batch summary.
- ``dlpack`` — runs the streaming evaluator twice on the same DT
  records: once via JSON bytes, once via the array-form ``Detections``
  path (ADR-0030). Wall-time delta is the artifact; the comparator
  enforces byte-equal ``Summary.stats`` between paths.

Every run wraps work in an ``RSSSampler`` and emits ``rss_curve.json``
alongside ``stats.json``. The B-stream design pins multi-artifact
emission as the per-paradigm shape (per ADR-0033 §"artifact_paths").

Stages: ``load``, ``init_streaming``, ``update_per_image`` (sums of all
per-image updates), ``finalize``, ``total``. ``dlpack`` mode adds
``update_array_path`` + ``finalize_array_path`` (the second-pass timings).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import vernier
from vernier._array_types import RLE, Detections
from vernier.instance import StreamingEvaluator

from bench.harness.rss import RSSSampler
from bench.harness.schema import BenchWarning, RunnerRepOutput, StageTimings
from bench.harness.timing import StageTable

ModeFlag = Literal["throughput", "vs_naive", "dlpack"]

_STATS_KEY = "summary"
_RSS_CURVE_KEY = "rss_curve"
_DLPACK_DELTA_KEY = "dlpack_delta"


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
        "--mode-flag",
        choices=["throughput", "vs_naive", "dlpack"],
        required=True,
    )
    return p.parse_args()


def _group_dt_by_image(dt_records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Bucket loadRes-shaped DT records by ``image_id`` (mirror of the
    parity conftest's helper, lifted to keep the runner standalone)."""
    by_image: dict[int, list[dict[str, Any]]] = {}
    for r in dt_records:
        by_image.setdefault(int(r["image_id"]), []).append(r)
    return by_image


def _records_to_per_image_bytes(
    dt_records: list[dict[str, Any]],
    gt_image_ids: list[int],
) -> list[bytes]:
    """Produce one JSON-bytes shard per GT image (in sorted GT order).

    Empty shards (no DT for an image) are returned as ``b"[]"`` so the
    caller can still issue the ``update()`` for shape symmetry.
    """
    by_image = _group_dt_by_image(dt_records)
    out: list[bytes] = []
    for img_id in gt_image_ids:
        out.append(json.dumps(by_image.get(img_id, [])).encode("utf-8"))
    return out


def _records_to_array_detections(
    gt_records: dict[str, Any],
    dt_records: list[dict[str, Any]],
    iou_type: str,
) -> list[Detections]:
    """ADR-0030 array-form translation: per-image ``Detections`` dicts.

    Lift-and-adapt of ``loadres_to_detections`` from
    ``tests/python/parity/conftest.py``. Pulled in here rather than
    importing from the test tree so the runner subprocess doesn't pick
    up pytest as a dep.
    """
    image_dims = {
        int(im["id"]): (int(im["height"]), int(im["width"]))
        for im in gt_records["images"]
    }
    by_image = _group_dt_by_image(dt_records)
    out: list[Detections] = []
    for image_id in sorted(by_image.keys()):
        dets = by_image[image_id]
        boxes = np.asarray([[float(x) for x in d["bbox"]] for d in dets], dtype=np.float64)
        scores = np.asarray([float(d["score"]) for d in dets], dtype=np.float64)
        labels = np.asarray([int(d["category_id"]) for d in dets], dtype=np.int64)
        payload: Detections = {
            "image_id": image_id,
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }
        if iou_type in ("segm", "boundary"):
            h, w = image_dims[image_id]
            payload["rles"] = [_segmentation_to_rle(d["segmentation"], h, w) for d in dets]
        out.append(payload)
    return out


def _segmentation_to_rle(seg: object, h: int, w: int) -> RLE:
    # Used only on segm/boundary cells. The bbox cells (today) skip this
    # branch; the runner imports ``pycocotools.mask`` lazily so the
    # bbox-only path doesn't pay the import cost.
    from pycocotools import mask as pmask  # noqa: PLC0415

    if isinstance(seg, dict):
        counts = seg["counts"]
        if isinstance(counts, list):
            return {
                "counts": np.asarray(counts, dtype=np.uint32),
                "size": (int(seg["size"][0]), int(seg["size"][1])),
            }
        encoded = {"counts": counts.encode("ascii"), "size": list(seg["size"])}
        binary = np.asarray(pmask.decode(encoded))  # type: ignore[arg-type]
    else:
        encoded_list = pmask.frPyObjects([list(p) for p in seg], h, w)  # type: ignore[arg-type]
        binary = np.asarray(pmask.decode(pmask.merge(encoded_list)))
    flat = binary.flatten("F")
    runs: list[int] = []
    cur = 0
    cur_val: np.uint8 = np.uint8(0)
    for v in flat:
        if v == cur_val:
            cur += 1
        else:
            runs.append(int(cur))
            cur = 1
            cur_val = v
    runs.append(int(cur))
    return {
        "counts": np.asarray(runs, dtype=np.uint32),
        "size": (binary.shape[0], binary.shape[1]),
    }


def _stream_per_image(
    gt_bytes: bytes,
    iou_type: str,
    shards: Sequence[bytes | Detections],
    stages: StageTable,
    *,
    update_stage: str,
    finalize_stage: str,
    init_stage: str,
) -> list[float]:
    """Per-image ``update()`` + ``finalize()`` loop with stage timing.

    Returns the resulting ``Summary.stats`` list. Splits ``update``
    timing into one ``StageTimings`` entry whose ``wall_ns`` is the sum
    across all per-image updates — keeps the schema's stage dict shape
    while still exposing the hot loop's aggregate cost.
    """
    with stages.stage(init_stage):
        ev = StreamingEvaluator(gt_bytes, iou_type=iou_type, parity_mode="strict")  # type: ignore[arg-type]

    update_total_ns = 0
    for shard in shards:
        t0 = time.perf_counter_ns()
        ev.update(shard)
        update_total_ns += time.perf_counter_ns() - t0
    stages.record(update_stage, update_total_ns)

    with stages.stage(finalize_stage):
        summary = ev.finalize()
    return [float(s) for s in summary.stats]


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
    extra_artifacts: dict[str, dict[str, Any]] | None = None,
    warnings: list[BenchWarning] | None = None,
) -> None:
    """Write streaming-shaped outputs: stats.json, rss_curve.json,
    optional dlpack_delta.json + the cell ``RunnerRepOutput`` JSON.

    The orchestrator will copy these into the canonical cell dir; sha256
    over each artifact lets the v2 schema's ``artifact_sha256`` carry the
    parity carrier the comparator consumes.
    """
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

    for key, payload in (extra_artifacts or {}).items():
        path = output_path.parent / f"{impl}-{key}.json"
        bytes_payload = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        path.write_bytes(bytes_payload)
        artifact_paths[key] = path.name
        artifact_sha256[key] = hashlib.sha256(bytes_payload).hexdigest()

    rep_output = RunnerRepOutput(
        paradigm="streaming",
        impl=impl,
        impl_version=impl_version,
        # iou_type is a Literal["bbox","segm","keypoints","boundary"];
        # the bench schema's metric/path slot is separate from the
        # IouType discriminator the streaming evaluator needs, but the
        # ``RunnerRepOutput`` field carries the IoU kernel choice.
        iou_type=iou_type,  # type: ignore[arg-type]
        workload_id=workload_id,
        stages=stages,
        summary_stats=summary_stats,
        artifact_paths=artifact_paths,
        artifact_sha256=artifact_sha256,
        warnings=list(warnings or []),
    )
    output_path.write_text(rep_output.model_dump_json(indent=2))


def _summary_stats_to_dict(stats: list[float]) -> dict[str, float]:
    """Name the stats positionally as ``stat_<i>`` — mirror of the
    detection runner's stat_names trick, but streaming surfaces every
    iou kernel through the same evaluator and the canonical names live
    in ``bench.runners._protocol.stat_names``. Keep the streaming
    runner self-contained: position-keyed dicts are good enough for
    the comparator (which compares positionally anyway)."""
    return {f"stat_{i}": v for i, v in enumerate(stats)}


def _gt_image_ids(gt_records: dict[str, Any]) -> list[int]:
    return sorted({int(im["id"]) for im in gt_records["images"]})


def _run_throughput_or_vs_naive(args: argparse.Namespace) -> None:
    stages = StageTable()

    with RSSSampler() as sampler, stages.stage("total"):
        with stages.stage("load"):
            gt_bytes = args.gt.read_bytes()
            dt_bytes = args.dt.read_bytes()
            gt_records = json.loads(gt_bytes)
            dt_records = json.loads(dt_bytes)

        image_ids = _gt_image_ids(gt_records)
        shards = _records_to_per_image_bytes(dt_records, image_ids)

        stats = _stream_per_image(
            gt_bytes,
            args.iou_type,
            shards,
            stages,
            init_stage="init_streaming",
            update_stage="update_per_image",
            finalize_stage="finalize",
        )

    _emit_outputs(
        output_path=args.output,
        workload_id=args.workload_id,
        iou_type=args.iou_type,
        impl="vernier_streaming",
        impl_version=vernier.__version__,
        stages=stages.to_dict(),
        summary_stats=_summary_stats_to_dict(stats),
        rss_samples=sampler.samples,
    )


def _run_dlpack(args: argparse.Namespace) -> None:
    stages = StageTable()

    with RSSSampler() as sampler, stages.stage("total"):
        with stages.stage("load"):
            gt_bytes = args.gt.read_bytes()
            dt_bytes = args.dt.read_bytes()
            gt_records = json.loads(gt_bytes)
            dt_records = json.loads(dt_bytes)

        image_ids = _gt_image_ids(gt_records)
        json_shards: list[bytes | Detections] = list(
            _records_to_per_image_bytes(dt_records, image_ids)
        )
        array_shards: list[bytes | Detections] = list(
            _records_to_array_detections(gt_records, dt_records, args.iou_type)
        )

        # Pass 1: JSON ingest path.
        json_t0 = time.perf_counter_ns()
        json_stats = _stream_per_image(
            gt_bytes,
            args.iou_type,
            json_shards,
            stages,
            init_stage="init_streaming",
            update_stage="update_per_image",
            finalize_stage="finalize",
        )
        json_elapsed_ns = time.perf_counter_ns() - json_t0

        # Pass 2: array (DLPack-equivalent) ingest path. Same evaluator
        # config, same DT bytes provenance — the array path is the
        # ADR-0030 alternate ingest.
        array_t0 = time.perf_counter_ns()
        array_stats = _stream_per_image(
            gt_bytes,
            args.iou_type,
            array_shards,
            stages,
            init_stage="init_streaming_array",
            update_stage="update_array_path",
            finalize_stage="finalize_array_path",
        )
        array_elapsed_ns = time.perf_counter_ns() - array_t0

    if len(json_stats) != len(array_stats):
        raise RuntimeError(
            f"dlpack: json/array Summary.stats length mismatch ({len(json_stats)} "
            f"vs {len(array_stats)})"
        )
    delta_payload: dict[str, Any] = {
        "json_wall_ns": json_elapsed_ns,
        "array_wall_ns": array_elapsed_ns,
        "ratio": (
            float(array_elapsed_ns) / float(json_elapsed_ns) if json_elapsed_ns > 0 else None
        ),
        "stats_json": json_stats,
        "stats_array": array_stats,
    }

    _emit_outputs(
        output_path=args.output,
        workload_id=args.workload_id,
        iou_type=args.iou_type,
        impl="vernier_streaming",
        impl_version=vernier.__version__,
        stages=stages.to_dict(),
        # The summary the comparator reads is the JSON-path one;
        # bit-equality between paths is asserted via the dlpack_delta
        # artifact's ``stats_array`` entry (the ADR-0030 oracle says they
        # should be byte-equal).
        summary_stats=_summary_stats_to_dict(json_stats),
        rss_samples=sampler.samples,
        extra_artifacts={_DLPACK_DELTA_KEY: delta_payload},
    )


def main() -> int:
    args = _parse_args()
    if args.mode_flag in ("throughput", "vs_naive"):
        _run_throughput_or_vs_naive(args)
    elif args.mode_flag == "dlpack":
        _run_dlpack(args)
    else:  # pragma: no cover - argparse choices already constrain this
        raise ValueError(f"unsupported mode flag {args.mode_flag!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
