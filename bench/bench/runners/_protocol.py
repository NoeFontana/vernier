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


# LVIS 13-entry plan, mirrored from
# ``crates/vernier-core/src/summarize.rs::lvis_default`` and the oracle's
# ``LVISEval.results`` keys. The ``@{max_dets}`` suffix on the AR rows is
# resolved at write time (LVIS canonical max_dets is 300, AC1).
LVIS_STAT_NAMES_TEMPLATE: tuple[str, ...] = (
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "APr",
    "APc",
    "APf",
    "AR@{max_dets}",
    "ARs@{max_dets}",
    "ARm@{max_dets}",
    "ARl@{max_dets}",
)


def lvis_stat_names(max_dets: int) -> tuple[str, ...]:
    """Resolve the 13 LVIS plan keys for a given ``max_dets``. Used by
    both the ``vernier_lvis`` and ``lvis-api`` runners so the
    ``summary_stats`` dict comes out keyed identically across impls
    (the cross-impl comparator zips over keys)."""
    return tuple(k.format(max_dets=max_dets) for k in LVIS_STAT_NAMES_TEMPLATE)


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


# Artifact-slot names emitted by the panoptic + semantic ``write_*_outputs``
# helpers below. Single source of truth — the orchestrator's per-paradigm
# assemblers import these so the producer and consumer sides can't drift.
PANOPTIC_ARTIFACT_KEYS: frozenset[str] = frozenset({"snapshot", "per_class"})
SEMANTIC_ARTIFACT_KEYS: frozenset[str] = frozenset({"snapshot", "per_class", "confusion"})
# LVIS reuses the detection single-tensor ``"tensor"`` slot for the
# ``(T, R, K, A)`` precision tensor — same parity surface shape as
# instance, just with K=1203 categories on val. The 13-entry summary
# plan rides ``RunnerRepOutput.summary_stats`` (no separate snapshot
# artifact — the precision tensor is the load-bearing parity carrier).
LVIS_ARTIFACT_KEYS: frozenset[str] = frozenset({"tensor"})


def scan_label_map_dir(directory: Path) -> dict[int, Path]:
    """Index PNGs in ``directory`` by integer image_id parsed from the
    stem. Filenames must match ``<int>.png``. Shared by every semantic
    runner — both vernier_semantic and the mmsegmentation oracle index
    label-map directories the same way.
    """
    out: dict[int, Path] = {}
    for entry in sorted(directory.iterdir()):
        if entry.suffix.lower() != ".png":
            continue
        try:
            image_id = int(entry.stem)
        except ValueError as e:
            raise ValueError(
                f"semantic runner expects label-map filenames of the form "
                f"'<int>.png'; got {entry.name!r} under {directory!s}."
            ) from e
        out[image_id] = entry
    return out


def per_class_uint64_table(
    per_class: dict[str, dict[str, float]],
    columns: tuple[str, ...],
) -> np.ndarray:
    """Build an ``N x len(columns)`` ``uint64`` view of an ``f64``
    per-class table from a snapshot's ``per_class`` dict.

    Rows are sorted by integer class id so the artifact diffs
    bit-equally between impls (uint64 view of f64 sidesteps the
    text-rounding ``np.save`` would otherwise apply to a float-text
    column).
    """
    rows: list[tuple[int, tuple[float, ...]]] = sorted(
        ((int(k), tuple(float(v[c]) for c in columns)) for k, v in per_class.items()),
        key=lambda r: r[0],
    )
    if not rows:
        return np.zeros((0, len(columns)), dtype=np.uint64)
    arr = np.array([list(r[1]) for r in rows], dtype=np.float64)
    return arr.view(np.uint64).reshape(arr.shape)


def parse_semantic_runner_args() -> argparse.Namespace:
    """Semantic-runner argspec (ADR-0033 §B2).

    The label-map family (GT label-map dir + DT label-map dir, plus
    ``--n-classes`` and an optional ``--ignore-label``) replaces
    ``--gt`` / ``--dt``. Output flags follow the panoptic two-artifact
    shape:

    - ``--output`` — the per-rep ``RunnerRepOutput`` JSON.
    - ``--snapshot-output`` — the per-rep ``SemanticSnapshot`` JSON
      under the ``"snapshot"`` artifact slot.
    - ``--per-class-output`` — the per-rep per-class ``.npy`` table
      under the ``"per_class"`` artifact slot.

    ``--ignore-label`` is encoded as a non-negative int; pass
    ``-1`` to mean "no ignore label" (matches the
    :class:`vernier.semantic.Evaluator` ``ignore_label=None`` shape).
    """
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--gt-label-map-dir", type=Path, required=True)
    p.add_argument("--dt-label-map-dir", type=Path, required=True)
    p.add_argument("--n-classes", type=int, required=True)
    p.add_argument(
        "--ignore-label",
        type=int,
        default=-1,
        help="Ignore-label class id; -1 means no ignore label.",
    )
    p.add_argument("--workload-id", type=str, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--snapshot-output", type=Path, required=True)
    p.add_argument("--per-class-output", type=Path, required=True)
    p.add_argument(
        "--confusion-output",
        type=Path,
        required=True,
        help=(
            "Path for the per-class strict-tier parity surface — a "
            "(4, n_classes) uint64 array with rows "
            "(intersect, union, area_pred, area_label). Both the "
            "vendored mmsegmentation oracle and vernier_semantic emit "
            "this shape so the comparator's `np.array_equal` check is "
            "well-defined across impls (ADR-0036)."
        ),
    )
    p.add_argument(
        "--paradigm",
        type=str,
        default="semantic",
        choices=["semantic"],
        help="Evaluation paradigm. Pinned to 'semantic' for this runner family.",
    )
    p.add_argument(
        "--iou-type",
        type=str,
        default="bbox",
        help=(
            "Detection-IouType slot carried by RunnerRepOutput. Ignored "
            "by semantic runners; the metric for the cell is 'miou'."
        ),
    )
    return p.parse_args()


def write_semantic_outputs(
    *,
    args: argparse.Namespace,
    impl: str,
    impl_version: str,
    stages: dict[str, StageTimings],
    snapshot_json_bytes: bytes,
    per_class_array: np.ndarray,
    confusion_array: np.ndarray,
    warnings: list[BenchWarning] | None = None,
) -> None:
    """Persist a semantic runner's three-artifact bundle.

    - ``snapshot.json`` — the ``SemanticSnapshot`` Pydantic JSON; the
      report layer reads this back for the headline scalars + the
      per-class derived floats.
    - ``per_class.npy`` — uint64 ``N x 4`` array (rows sorted by class id)
      holding ``[iou, accuracy, precision, support]`` as f64 bit-cast
      into uint64. ``np.save`` with ``allow_pickle=False``; readers cast
      back via ``arr.view(np.float64)``.
    - ``confusion.npy`` — the strict-tier parity surface — uint64
      ``(4, n_classes)`` array with rows
      ``(intersect, union, area_pred, area_label)``. The four marginals
      are mmsegmentation ``IoUMetric.intersect_and_union``'s native
      output; vernier_semantic projects its NxN confusion matrix to
      the same shape so cross-impl ``np.array_equal`` is well-defined.
      Equal marginals ⇒ equal mIoU / FWIoU / pixel_accuracy /
      mean_accuracy by ADR-0028's quirk-AL2 (bit-deterministic float
      arithmetic on integer totals).

    The ``RunnerRepOutput`` records all three artifacts under the
    canonical ``"snapshot"`` / ``"per_class"`` / ``"confusion"`` slots.
    The ``summary_stats`` slot carries the four headline scalars
    (mIoU / FWIoU / pixel_accuracy / mean_accuracy) so downstream
    report aggregation has scalar columns without re-parsing the
    snapshot.
    """
    from bench.harness.parity import SemanticSnapshot

    snap = SemanticSnapshot.model_validate_json(snapshot_json_bytes)

    snapshot_path: Path = args.snapshot_output
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(snapshot_json_bytes)

    per_class_path: Path = args.per_class_output
    per_class_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(per_class_path, per_class_array, allow_pickle=False)

    confusion_path: Path = args.confusion_output
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    if confusion_array.dtype != np.uint64:
        raise TypeError(f"semantic confusion array must be uint64; got {confusion_array.dtype}")
    if confusion_array.ndim != 2 or confusion_array.shape[0] != 4:
        raise ValueError(
            f"semantic confusion array must be shape (4, n_classes); got {confusion_array.shape}"
        )
    np.save(confusion_path, confusion_array, allow_pickle=False)

    summary_stats: dict[str, float] = {
        "mIoU": float(snap.miou),
        "FWIoU": float(snap.fwiou),
        "pixel_accuracy": float(snap.pixel_accuracy),
        "mean_accuracy": float(snap.mean_accuracy),
        "n_classes": float(snap.n_classes),
    }

    output = RunnerRepOutput(
        paradigm="semantic",
        impl=impl,
        impl_version=impl_version,
        # IouType slot kept for cross-paradigm column compatibility;
        # the metric for semantic cells is "miou", recorded by the
        # orchestrator separately.
        iou_type=args.iou_type,
        workload_id=args.workload_id,
        stages=stages,
        summary_stats=summary_stats,
        artifact_paths={
            "snapshot": snapshot_path.name,
            "per_class": per_class_path.name,
            "confusion": confusion_path.name,
        },
        artifact_sha256={
            "snapshot": file_sha256(snapshot_path),
            "per_class": file_sha256(per_class_path),
            "confusion": file_sha256(confusion_path),
        },
        warnings=list(warnings or []),
    )
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output.model_dump_json(indent=2))


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
    - ``per_class.npy`` — uint64 ``N x 3`` array (rows sorted by
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


def parse_lvis_runner_args() -> argparse.Namespace:
    """LVIS-runner argspec (ADR-0026 + ADR-0033).

    Same on-disk shape as the detection argspec — LVIS GT/DT are JSON
    pairs — but with two LVIS-specific knobs:

    - ``--paradigm`` is pinned to ``"lvis"`` so the orchestrator
      routes through the LVIS comparator (vernier_lvis vs lvis-api,
      not vernier vs pycocotools).
    - ``--max-dets`` defaults to ``300`` (LVIS canonical, AC1) instead
      of detection's ``100``. The flag is per-rep argv so future cells
      can sweep.

    ``--iou-type`` is restricted to ``"bbox"`` until the
    ``evaluate_segm_grid_with_dataset`` FFI lands; both runners reject
    other choices loudly so a misrouted cell fails at the runner rather
    than silently mis-evaluating.
    """
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--dt", type=Path, required=True)
    p.add_argument("--iou-type", choices=["bbox"], required=True)
    p.add_argument("--workload-id", type=str, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tensor-output", type=Path, required=True)
    p.add_argument(
        "--max-dets",
        type=int,
        default=300,
        help="LVIS canonical max_dets (AC1 = 300). Per-rep argv override allowed.",
    )
    p.add_argument(
        "--paradigm",
        type=str,
        default="lvis",
        choices=["lvis"],
        help="Evaluation paradigm. Pinned to 'lvis' for this runner family.",
    )
    return p.parse_args()


def write_lvis_outputs(
    *,
    args: argparse.Namespace,
    impl: str,
    impl_version: str,
    stages: dict[str, StageTimings],
    summary_stats: dict[str, float],
    precision_tensor: np.ndarray,
    warnings: list[BenchWarning] | None = None,
) -> None:
    """Persist an LVIS runner's tensor + result JSON.

    Same on-disk shape as detection (``"tensor"`` slot single-array
    plus the 13-entry summary stats), with ``paradigm="lvis"`` pinned
    on the ``RunnerRepOutput`` so the orchestrator routes the result
    to the LVIS comparator. The summary keys must be in the
    :func:`lvis_stat_names` shape — the cross-impl comparator zips
    over keys positionally.
    """
    tensor_path: Path = args.tensor_output
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(tensor_path, precision_tensor, allow_pickle=False)

    output = RunnerRepOutput(
        paradigm="lvis",
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
