"""Shared CLI argspec and output helpers for all impl runners.

Every detection runner under ``bench.runners.*_runner`` accepts the
same arguments and writes the same JSON shape (``RunnerRepOutput``)
plus a ``.npy`` precision tensor. ADR-0033 generalizes the schema to
``artifact_paths`` / ``artifact_sha256`` dicts so non-instance
paradigms can emit multi-artifact result bundles; detection sticks to
the canonical single-tensor pair under the ``"tensor"`` slot.

ADR-0033 §"Runner CLI" extends the protocol with paradigm-specific
flags. Detection runners keep :func:`parse_runner_args` unchanged
(strictly additive — the argspec contract test still passes).
Panoptic runners use :func:`parse_panoptic_runner_args`, which
swaps ``--gt`` / ``--dt`` for the panoptic four-path family and
adds ``--paradigm panoptic`` for explicit dispatch.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
from typing import Any, get_args

import numpy as np
from coco_val_cache import file_sha256

from bench.harness.migrations.v1_to_v2 import TENSOR_KEY
from bench.harness.schema import BenchWarning, IouType, RunnerRepOutput, StageTimings
from bench.harness.timing import StageTable

BBOX_STAT_NAMES: tuple[str, ...] = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
)
KP_STAT_NAMES: tuple[str, ...] = (
    "AP",
    "AP50",
    "AP75",
    "AP_medium",
    "AP_large",
    "AR",
    "AR50",
    "AR75",
    "AR_medium",
    "AR_large",
)


def stat_names(iou_type: IouType) -> tuple[str, ...]:
    return KP_STAT_NAMES if iou_type == "keypoints" else BBOX_STAT_NAMES


def parse_runner_args() -> argparse.Namespace:
    """Detection-runner argspec. Untouched by ADR-0033 so the legacy
    detection runners ship unchanged. Panoptic / semantic / streaming
    runners use their own ``parse_*_runner_args`` siblings.

    A ``--paradigm`` flag is accepted optionally for forward-compat —
    when omitted, defaults to ``"instance"`` (the v1 implicit value).
    The flag is ignored by detection runners; the orchestrator passes
    it through so non-instance dispatchers can verify the paradigm
    matches their expectation.
    """
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--dt", type=Path, required=True)
    p.add_argument("--iou-type", choices=list(get_args(IouType)), required=True)
    p.add_argument("--workload-id", type=str, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tensor-output", type=Path, required=True)
    p.add_argument(
        "--paradigm",
        type=str,
        default="instance",
        help=(
            "Evaluation paradigm. Optional; detection runners default to "
            "'instance' for v1 compatibility. Non-instance paradigms use "
            "paradigm-specific argspec (parse_panoptic_runner_args, etc.)."
        ),
    )
    return p.parse_args()


def parse_panoptic_runner_args() -> argparse.Namespace:
    """Panoptic-runner argspec (ADR-0033 §B1).

    The four-path panoptic family (GT PNG dir + GT segments_info JSON,
    DT PNG dir + DT segments_info JSON, plus the categories JSON)
    replaces ``--gt`` / ``--dt``. Output flags stay shared:

    - ``--output`` — the per-rep ``RunnerRepOutput`` JSON (same as
      detection; the orchestrator parses and stitches it identically).
    - ``--snapshot-output`` — the per-rep ``PanopticSnapshot`` JSON
      under the ``"snapshot"`` artifact slot.
    - ``--per-class-output`` — the per-rep per-class ``.npy`` table
      under the ``"per_class"`` artifact slot.

    ``--iou-type`` is accepted (defaults to ``"bbox"``) so the
    orchestrator's per-rep argv can pass it without forking the
    invocation; panoptic runners ignore it. The semantic carrier of
    the cell is the workload + the ``"pq"`` metric, recorded
    separately on the orchestrator side.
    """
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--gt-png-dir", type=Path, required=True)
    p.add_argument("--gt-json", type=Path, required=True)
    p.add_argument("--dt-png-dir", type=Path, required=True)
    p.add_argument("--dt-json", type=Path, required=True)
    p.add_argument("--categories-json", type=Path, required=True)
    p.add_argument("--workload-id", type=str, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--snapshot-output", type=Path, required=True)
    p.add_argument("--per-class-output", type=Path, required=True)
    p.add_argument(
        "--paradigm",
        type=str,
        default="panoptic",
        choices=["panoptic"],
        help="Evaluation paradigm. Pinned to 'panoptic' for this runner family.",
    )
    p.add_argument(
        "--iou-type",
        type=str,
        default="bbox",
        help=(
            "Detection-IouType slot carried by RunnerRepOutput. Ignored "
            "by panoptic runners; the metric for the cell is 'pq'."
        ),
    )
    return p.parse_args()


def write_panoptic_outputs(
    *,
    args: argparse.Namespace,
    impl: str,
    impl_version: str,
    stages: dict[str, StageTimings],
    snapshot_json_bytes: bytes,
    per_class_array: np.ndarray,
    warnings: list[BenchWarning] | None = None,
) -> None:
    """Persist a panoptic runner's two-artifact bundle.

    - ``snapshot.json`` — the ``PanopticSnapshot`` Pydantic JSON; the
      comparator reads this back and dispatches per-tier comparisons.
    - ``per_class.npy`` — uint64 ``N×3`` array (rows sorted by
      category id) holding ``[pq, sq, rq]`` per category as f64
      bit-cast into uint64. ``np.save`` with ``allow_pickle=False``;
      readers cast back via ``arr.view(np.float64)``.

    The ``RunnerRepOutput`` records both artifacts under the canonical
    ``"snapshot"`` and ``"per_class"`` slots.

    The ``summary_stats`` slot of ``RunnerRepOutput`` carries the
    All-bucket pq/sq/rq plus the bucket counts so downstream report
    aggregation has scalar columns without re-parsing the snapshot.
    """
    from bench.harness.parity import PanopticSnapshot

    snap = PanopticSnapshot.model_validate_json(snapshot_json_bytes)

    snapshot_path: Path = args.snapshot_output
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(snapshot_json_bytes)

    per_class_path: Path = args.per_class_output
    per_class_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(per_class_path, per_class_array, allow_pickle=False)

    summary_stats: dict[str, float] = {
        "PQ": float(snap.pq),
        "SQ": float(snap.sq),
        "RQ": float(snap.rq),
        "PQ_things": float(snap.pq_things),
        "SQ_things": float(snap.sq_things),
        "RQ_things": float(snap.rq_things),
        "PQ_stuff": float(snap.pq_stuff),
        "SQ_stuff": float(snap.sq_stuff),
        "RQ_stuff": float(snap.rq_stuff),
        "n": float(snap.n),
        "n_things": float(snap.n_things),
        "n_stuff": float(snap.n_stuff),
    }

    output = RunnerRepOutput(
        paradigm="panoptic",
        impl=impl,
        impl_version=impl_version,
        # The schema's IouType field carries an instance-shaped value
        # for cross-paradigm column compatibility; the metric for
        # panoptic cells is "pq" (recorded by the orchestrator at the
        # cell level — RunnerRepOutput carries iou_type, not metric).
        iou_type=args.iou_type,
        workload_id=args.workload_id,
        stages=stages,
        summary_stats=summary_stats,
        artifact_paths={
            "snapshot": snapshot_path.name,
            "per_class": per_class_path.name,
        },
        artifact_sha256={
            "snapshot": file_sha256(snapshot_path),
            "per_class": file_sha256(per_class_path),
        },
        warnings=list(warnings or []),
    )
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output.model_dump_json(indent=2))


def write_outputs(
    *,
    args: argparse.Namespace,
    impl: str,
    impl_version: str,
    stages: dict[str, StageTimings],
    summary_stats: dict[str, float],
    precision_tensor: np.ndarray,
    warnings: list[BenchWarning] | None = None,
) -> None:
    """Persist the tensor and the result JSON. The orchestrator re-checks
    the tensor sha256 after copying to the canonical result path.

    Detection runners produce a single precision tensor; under v2 it is
    written under the canonical ``"tensor"`` slot of the artifact
    dicts. B-stream runners that emit multi-artifact bundles (panoptic
    snapshot + per-class table; streaming summary + RSS curve) build
    their ``RunnerRepOutput`` directly without going through this
    helper.
    """
    tensor_path: Path = args.tensor_output
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(tensor_path, precision_tensor, allow_pickle=False)

    output = RunnerRepOutput(
        paradigm="instance",
        impl=impl,
        impl_version=impl_version,
        iou_type=args.iou_type,
        workload_id=args.workload_id,
        stages=stages,
        summary_stats=summary_stats,
        artifact_paths={TENSOR_KEY: tensor_path.name},
        artifact_sha256={TENSOR_KEY: file_sha256(tensor_path)},
        warnings=list(warnings or []),
    )
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output.model_dump_json(indent=2))


def run_cocoeval_pipeline(
    *,
    args: argparse.Namespace,
    impl: str,
    impl_version: str,
    coco_cls: type[Any],
    cocoeval_cls: type[Any],
) -> None:
    """Run the standard COCO/COCOeval load → evaluate → accumulate →
    summarize chain inside a stdout redirect, then persist outputs.

    Used by every runner that wraps a pycocotools-shaped surface
    (pycocotools, faster-coco-eval, boundary-iou-api). pycocotools and
    its drop-ins all print progress from inside ``COCO`` /
    ``loadRes`` / ``summarize``; one outer redirect is cleaner than
    sprinkling them.
    """
    stages = StageTable()
    with contextlib.redirect_stdout(io.StringIO()):
        with stages.stage("load"):
            gt = coco_cls(str(args.gt))
            dt = gt.loadRes(str(args.dt))
            cocoeval = cocoeval_cls(gt, dt, iouType=args.iou_type)
        with stages.stage("evaluate"):
            cocoeval.evaluate()
        with stages.stage("accumulate"):
            cocoeval.accumulate()
        with stages.stage("summarize"):
            cocoeval.summarize()

    precision = np.asarray(cocoeval.eval["precision"])
    raw_stats = np.asarray(cocoeval.stats, dtype=np.float64)
    names = stat_names(args.iou_type)
    summary_stats: dict[str, float] = {name: float(raw_stats[i]) for i, name in enumerate(names)}

    stages.record("total", stages.total_so_far_ns())

    write_outputs(
        args=args,
        impl=impl,
        impl_version=impl_version,
        stages=stages.to_dict(),
        summary_stats=summary_stats,
        precision_tensor=precision,
    )
