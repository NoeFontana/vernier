"""Shared CLI argspec + output helpers for semantic-segmentation
runners (ADR-0033 §B2).

Every semantic runner under ``bench.runners.*_runner`` for the
``semantic`` paradigm accepts the same arguments and writes the same
JSON shape (``RunnerRepOutput``) plus a single ``.npy`` artifact under
the ``"confusion"`` slot of ``artifact_paths`` / ``artifact_sha256``.

The instance-runner sibling at ``_protocol.py`` keeps the detection-
shaped ``parse_runner_args`` (``--gt`` / ``--dt`` / ``--iou-type``)
and the ``"tensor"`` slot. Splitting these by paradigm avoids forcing
``parse_runner_args`` to grow a discriminated argparse and keeps each
runner's CLI surface readable.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from coco_val_cache import file_sha256

from bench.harness.schema import BenchWarning, RunnerRepOutput, StageTimings
from bench.harness.timing import StageTable

#: Canonical artifact key for the semantic confusion matrix. Mirrors
#: ``TENSOR_KEY`` in ``bench/bench/harness/migrations/v1_to_v2.py``;
#: every reader that handles a semantic ``BenchResult`` looks up
#: ``artifact_paths[CONFUSION_KEY]``.
CONFUSION_KEY: str = "confusion"

#: PNG glob the runners use to enumerate per-image label maps. The
#: Cityscapes preset writes ``*_gtFine_labelTrainIds.png``; other
#: presets (ADE20K / Pascal VOC) reuse the same protocol module with
#: their own glob.
DEFAULT_TRAIN_ID_PNG_GLOB: str = "*_gtFine_labelTrainIds.png"

#: Cityscapes 19-class trainId names — pinned in
#: ``cityscapesscripts/helpers/labels.py`` (the canonical 0-indexed
#: trainId-to-name mapping for the evaluation 19-class subset).
CITYSCAPES_TRAIN_ID_NAMES: tuple[str, ...] = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)


def parse_semantic_runner_args(argv: list[str] | None = None) -> argparse.Namespace:
    """``vernier_semantic`` and ``cityscapesscripts`` runner argspec.

    The orchestrator builds this argv once per-rep; runners parse it
    via ``argparse``. Splitting from ``parse_runner_args`` (the
    instance shape) keeps each paradigm's CLI flat and self-documenting.
    """
    p = argparse.ArgumentParser(add_help=True)
    # Either the GT/DT directories or a single PNG glob; the runners
    # auto-pair on filename basenames.
    p.add_argument("--gt-label-maps-dir", type=Path, required=True)
    p.add_argument("--dt-label-maps-dir", type=Path, required=True)
    p.add_argument("--n-classes", type=int, required=True)
    p.add_argument("--ignore-label", type=int, default=None)
    p.add_argument("--paradigm", choices=["semantic"], default="semantic")
    p.add_argument(
        "--png-glob",
        type=str,
        default=DEFAULT_TRAIN_ID_PNG_GLOB,
        help=(
            "Glob pattern for label-map PNGs under each directory. "
            "Default matches the Cityscapes trainId convention."
        ),
    )
    p.add_argument("--workload-id", type=str, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--confusion-output", type=Path, required=True)
    return p.parse_args(argv)


def discover_image_pairs(
    gt_dir: Path,
    dt_dir: Path,
    *,
    glob: str = DEFAULT_TRAIN_ID_PNG_GLOB,
) -> list[tuple[int, Path, Path]]:
    """Walk ``gt_dir`` and pair each PNG with its sibling under
    ``dt_dir``.

    The Cityscapes val split organizes its 500 images into 3 city
    subdirectories; the recursive walk handles that without baking in
    the city names. Returns a list of ``(image_id, gt_path, dt_path)``
    tuples sorted by image_id (a stable hash of the GT relative path)
    for run-to-run determinism.

    Image IDs are derived from the relative path under ``gt_dir`` —
    this matters for the bench cell because both runners must agree
    on which DT pairs with which GT, and a stable derivation removes
    a class of "the runners disagreed" bugs.
    """
    if not gt_dir.is_dir():
        raise ValueError(f"--gt-label-maps-dir {gt_dir!s} is not a directory")
    if not dt_dir.is_dir():
        raise ValueError(f"--dt-label-maps-dir {dt_dir!s} is not a directory")

    pairs: list[tuple[int, Path, Path]] = []
    for gt_path in sorted(gt_dir.rglob(glob)):
        rel = gt_path.relative_to(gt_dir)
        dt_path = dt_dir / rel
        if not dt_path.is_file():
            # Fallback: search for a basename match anywhere under
            # ``dt_dir``. Real datasets sometimes flatten the city
            # subdirectories.
            candidates = list(dt_dir.rglob(gt_path.name))
            if len(candidates) == 1:
                dt_path = candidates[0]
            else:
                raise FileNotFoundError(
                    f"no DT label map for GT {gt_path!s}: looked at {dt_path!s} "
                    f"and found {len(candidates)} basename match(es)."
                )
        # Image id is the (sorted) index — stable as long as the GT
        # walk is sorted, which it is.
        image_id = len(pairs)
        pairs.append((image_id, gt_path, dt_path))
    if not pairs:
        raise RuntimeError(
            f"no label-map PNGs found under {gt_dir!s} for glob {glob!r}"
        )
    return pairs


def write_semantic_outputs(
    *,
    args: argparse.Namespace,
    impl: str,
    impl_version: str,
    stages: dict[str, StageTimings],
    summary_stats: dict[str, float],
    confusion: np.ndarray,
    warnings: list[BenchWarning] | None = None,
    extra_summary_path: Path | None = None,
    extra_summary: Mapping[str, Any] | None = None,
) -> None:
    """Persist the confusion matrix and the result JSON for a semantic
    runner.

    ``confusion`` must be a `(n_classes, n_classes)` ``uint64`` array.
    Lands under the canonical ``"confusion"`` slot of ``artifact_paths``
    / ``artifact_sha256``.

    ``extra_summary`` (when provided) is dumped to ``extra_summary_path``
    as JSON — used by the runners to ship per-class IoU + headline
    floats alongside the integer matrix; the orchestrator copies the
    file to the cell dir but the comparator only reads the matrix.
    """
    if confusion.dtype != np.uint64:
        raise ValueError(
            f"semantic confusion matrix must be uint64; got dtype={confusion.dtype}"
        )
    n = args.n_classes
    if confusion.shape != (n, n):
        raise ValueError(
            f"semantic confusion matrix must be ({n}, {n}); got shape={confusion.shape}"
        )

    confusion_path: Path = args.confusion_output
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(confusion_path, confusion, allow_pickle=False)

    artifact_paths: dict[str, str] = {CONFUSION_KEY: confusion_path.name}
    artifact_sha256: dict[str, str] = {CONFUSION_KEY: file_sha256(confusion_path)}

    if extra_summary is not None and extra_summary_path is not None:
        extra_summary_path.parent.mkdir(parents=True, exist_ok=True)
        extra_summary_path.write_text(json.dumps(extra_summary, indent=2, sort_keys=True))

    # Semantic uses ``iou_type="bbox"`` as a placeholder on the result
    # JSON because the v2 schema's ``iou_type`` field is still typed
    # ``IouType`` (the bbox/segm/keypoints/boundary literal). The
    # paradigm-segmented result path uses ``metric="miou"`` instead;
    # the placeholder here is consumed only by readers that filter on
    # paradigm first (the comparator + the report layer both do this).
    output = RunnerRepOutput(
        paradigm="semantic",
        impl=impl,
        impl_version=impl_version,
        iou_type="bbox",
        workload_id=args.workload_id,
        stages=stages,
        summary_stats=summary_stats,
        artifact_paths=artifact_paths,
        artifact_sha256=artifact_sha256,
        warnings=list(warnings or []),
    )
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output.model_dump_json(indent=2))


def derive_metrics_from_confusion(
    confusion: np.ndarray,
    *,
    eps: float = 0.0,
) -> dict[str, float]:
    """Reduce an NxN ``uint64`` confusion matrix to the four headline
    semantic-segmentation float metrics.

    Identical formula to vernier-semantic's ``summarize.rs`` and
    cityscapesScripts' :func:`evaluateImgLists`: per-class IoU =
    ``tp / (tp + fp + fn)``, mIoU = mean of per-class IoUs (excluding
    classes with zero support), pixel-accuracy = ``trace / total``,
    mean-accuracy = mean per-class recall.

    For the bench cell, both impls' float metrics are derived from
    bit-equal integer counts so the floats inherit the bit-equality;
    we still emit them as ``summary_stats`` because the report layer
    aggregates them into the longitudinal view.

    ``eps`` is added to denominators only when the support is
    non-zero — matches the ZeroDivisionError-avoidance pattern
    documented in ``crates/vernier-semantic/src/summarize.rs``.
    """
    cm = confusion.astype(np.int64, copy=False)
    n = cm.shape[0]
    diag = np.diag(cm).astype(np.float64)
    gt_sum = cm.sum(axis=1).astype(np.float64)
    dt_sum = cm.sum(axis=0).astype(np.float64)
    union = gt_sum + dt_sum - diag
    total = float(cm.sum())

    iou_per_class = np.full(n, np.nan, dtype=np.float64)
    acc_per_class = np.full(n, np.nan, dtype=np.float64)
    valid_iou_mask = union > 0
    iou_per_class[valid_iou_mask] = diag[valid_iou_mask] / (union[valid_iou_mask] + eps)
    valid_acc_mask = gt_sum > 0
    acc_per_class[valid_acc_mask] = diag[valid_acc_mask] / (gt_sum[valid_acc_mask] + eps)

    miou = float(np.nanmean(iou_per_class)) if valid_iou_mask.any() else 0.0
    mean_accuracy = float(np.nanmean(acc_per_class)) if valid_acc_mask.any() else 0.0
    pixel_accuracy = float(diag.sum() / total) if total > 0 else 0.0
    weighted_iou = (
        float(np.nansum(iou_per_class * (gt_sum / total))) if total > 0 else 0.0
    )

    out: dict[str, float] = {
        "miou": miou,
        "fwiou": weighted_iou,
        "pixel_accuracy": pixel_accuracy,
        "mean_accuracy": mean_accuracy,
    }
    for c in range(n):
        out[f"iou_class_{c}"] = float(iou_per_class[c]) if not np.isnan(iou_per_class[c]) else 0.0
    return out


def accumulate_confusion_pure_numpy(
    gt: np.ndarray,
    dt: np.ndarray,
    *,
    n_classes: int,
    ignore_label: int | None,
) -> np.ndarray:
    """Build a single-image NxN ``uint64`` confusion matrix.

    Uses the ``np.bincount`` formula every reasonable confusion-matrix
    accumulator converges on (cityscapesScripts' Python fallback path
    in ``addToConfusionMatrix.py`` is structurally identical):

        encoded = gt * n_classes + dt
        counts = np.bincount(encoded, minlength=n_classes**2)
        cm = counts.reshape(n_classes, n_classes)

    Pixels with ``gt == ignore_label`` (when set) are dropped before
    the bincount (quirk **AJ2** in the sem-seg quirks survey).
    Non-ignore-label pixels with a class id outside ``[0, n_classes)``
    are a data error — surfaced as a ``ValueError`` so the runner's
    upstream cache provisioning gets a clear signal.

    The runner uses this for its ``cityscapesscripts``-style
    accumulator (the released wheel only exposes the file-based
    ``evaluateImgLists`` API, which doesn't fit the bench cell's
    in-memory pairing pattern); the same formula appears in the
    ``vernier`` runner's per-image fold path under the hood.
    """
    if gt.shape != dt.shape:
        raise ValueError(
            f"GT/DT shape mismatch: {gt.shape} vs {dt.shape}"
        )
    gt_flat = gt.reshape(-1).astype(np.int64, copy=False)
    dt_flat = dt.reshape(-1).astype(np.int64, copy=False)

    if ignore_label is not None:
        keep = gt_flat != ignore_label
        gt_flat = gt_flat[keep]
        dt_flat = dt_flat[keep]

    # Range-check after filtering — pixels with class id == ignore_label
    # in DT are still considered (the dataset's ignore convention is
    # GT-only) but other out-of-range values are an error.
    if gt_flat.size > 0:
        gt_max = int(gt_flat.max())
        dt_max = int(dt_flat.max())
        if gt_max >= n_classes:
            raise ValueError(
                f"GT pixel value {gt_max} out of range for n_classes={n_classes}"
            )
        if dt_max >= n_classes:
            raise ValueError(
                f"DT pixel value {dt_max} out of range for n_classes={n_classes}"
            )

    encoded = gt_flat * n_classes + dt_flat
    counts = np.bincount(encoded, minlength=n_classes * n_classes).astype(np.uint64, copy=False)
    return counts.reshape(n_classes, n_classes)


def load_label_map_png(path: Path) -> np.ndarray:
    """Decode a single-channel label-map PNG via Pillow.

    Same shape contract as
    :func:`vernier.semantic._decode_png_to_uint32`: returns a 2-D
    array (the runners then upcast to ``int64`` for the bincount).
    """
    from PIL import Image

    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError(
            f"label-map PNG {path!s} must be single-channel (2-D); got shape {arr.shape!r}"
        )
    return arr


def env_or_attr(args: argparse.Namespace, attr: str, env: str) -> Any:
    """Helper: prefer ``argparse`` value, fall back to env var."""
    val = getattr(args, attr, None)
    return val if val is not None else os.environ.get(env)
