"""Parity harness for the semantic-segmentation evaluation surface
(ADR-0028, ADR-0036).

Two runners share the :class:`SemanticSnapshot` shape:

- :func:`run_mmsegmentation` — the verbatim-vendored ``IoUMetric``
  (oracle). Calls the static methods ``IoUMetric.intersect_and_union``
  and ``IoUMetric.total_area_to_metrics`` directly (bypassing
  ``BaseMetric.process`` and its DataLoader sample shape — incidental
  to the parity claim).
- :func:`run_streaming_pair` — single-pass streaming variant for
  val2017-scale workloads, interleaving the oracle accumulator with a
  vernier ``BackgroundEvaluator``. u64-additive across images, so
  peak memory is one decoded label-map per side, not the whole
  dataset.

The strict-mode parity claim is bit-equality on the **integer
confusion-matrix totals** (intersect/union/pred/label, all u64).
Derived float scalars (aAcc, mIoU, per-class IoU/Acc) follow trivially
from the same inputs — both runners apply the same float arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# `from mmseg.evaluation.metrics.iou_metric import IoUMetric` only resolves
# after :mod:`conftest` (or the bench loader) installs the mmengine stubs.
from mmseg.evaluation.metrics.iou_metric import IoUMetric  # pyright: ignore[reportMissingImports]
from numpy.typing import NDArray

import vernier.semantic as vsem
from vernier._types import PARITY_STRICT


@dataclass(frozen=True, slots=True)
class SemanticSnapshot:
    """Shared snapshot shape for oracle and candidate runners.

    Carries the four per-class u64 totals (the bit-equality surface)
    plus the headline scalar ``aacc`` and the derived per-class IoU /
    Acc arrays. Two snapshots from different runners are bit-equal
    iff their confusion-matrix totals agree.
    """

    aacc: float
    iou: NDArray[np.float64]
    acc: NDArray[np.float64]
    intersect: NDArray[np.uint64]
    union: NDArray[np.uint64]
    pred: NDArray[np.uint64]
    label: NDArray[np.uint64]


def _project_totals(
    intersect: NDArray[np.uint64],
    union: NDArray[np.uint64],
    pred: NDArray[np.uint64],
    label: NDArray[np.uint64],
) -> SemanticSnapshot:
    """Apply the mmsegmentation float arithmetic to per-class u64 totals.

    The same computation runs on both sides; running it once here keeps
    the projection in one place so the two runners are guaranteed to
    use the same NaN-handling and division semantics.

    The per-class arrays are produced with ``intersect / union`` and
    ``intersect / label`` — divisions by zero produce ``nan`` per
    NumPy default, matching mmsegmentation's strict semantics (quirk
    AL2). ``aAcc`` is computed in float64 to mirror torch's
    ``.float()`` promotion in ``IoUMetric.total_area_to_metrics``.
    """
    label_sum = float(label.sum())
    aacc = float("nan") if label_sum == 0.0 else float(intersect.sum()) / label_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = intersect.astype(np.float64) / union.astype(np.float64)
        acc = intersect.astype(np.float64) / label.astype(np.float64)
    return SemanticSnapshot(
        aacc=aacc,
        iou=iou,
        acc=acc,
        intersect=intersect,
        union=union,
        pred=pred,
        label=label,
    )


def run_mmsegmentation(
    pred: NDArray[np.integer[Any]],
    gt: NDArray[np.integer[Any]],
    num_classes: int,
    ignore_index: int,
) -> SemanticSnapshot:
    """Oracle runner: vendored ``IoUMetric`` on a single (pred, gt) pair.

    ``pred`` and ``gt`` are 2-D integer arrays. They get cast to
    ``int64`` (matching the dtype the upstream library promotes to
    inside ``intersect_and_union``) and passed to the static method
    as torch tensors.
    """
    pred_t = torch.from_numpy(np.ascontiguousarray(pred).astype(np.int64))  # pyright: ignore[reportPrivateImportUsage]
    gt_t = torch.from_numpy(np.ascontiguousarray(gt).astype(np.int64))  # pyright: ignore[reportPrivateImportUsage]

    intersect, union, area_pred, area_label = IoUMetric.intersect_and_union(
        pred_t, gt_t, num_classes, ignore_index
    )

    return _project_totals(
        intersect=intersect.numpy().astype(np.uint64),
        union=union.numpy().astype(np.uint64),
        pred=area_pred.numpy().astype(np.uint64),
        label=area_label.numpy().astype(np.uint64),
    )


def _summary_to_snapshot(summary: vsem.Summary) -> SemanticSnapshot:
    """Project a vernier :class:`Summary` into the shared
    :class:`SemanticSnapshot` via the per-class u64 totals.

    Convention: ``counts[gt_class, dt_class]`` (verified against the
    kernel implementation in vernier-core::kernel — see ADR-0028).
    """
    counts = summary.confusion_matrix.counts().astype(np.uint64, copy=False)
    intersect = np.diagonal(counts).astype(np.uint64, copy=True)
    label = counts.sum(axis=1, dtype=np.uint64)
    pred = counts.sum(axis=0, dtype=np.uint64)
    union = pred + label - intersect
    return _project_totals(intersect=intersect, union=union, pred=pred, label=label)


def run_streaming_pair(
    pairs: Iterable[tuple[int, NDArray[np.integer[Any]], NDArray[np.integer[Any]]]],
    *,
    num_classes: int,
    ignore_index: int,
) -> tuple[SemanticSnapshot, SemanticSnapshot]:
    """Interleaved single-pass streaming runner. Returns
    ``(oracle, candidate)`` snapshots after consuming the iterator
    exactly once — both runners see the same decoded arrays.

    Memory ceiling: one decoded label-map per side plus the
    background worker's queue (default capacity 8). Suitable for
    val2017-scale workloads where materializing the full dataset
    would peak at tens of GB.
    """
    totals: tuple[NDArray[np.uint64], ...] = (
        np.zeros(num_classes, dtype=np.uint64),
        np.zeros(num_classes, dtype=np.uint64),
        np.zeros(num_classes, dtype=np.uint64),
        np.zeros(num_classes, dtype=np.uint64),
    )
    evaluator = vsem.Evaluator(parity_mode=PARITY_STRICT)
    with evaluator.background(num_classes, ignore_label=ignore_index) as bg:
        for image_id, gt_arr, dt_arr in pairs:
            oracle_snap = run_mmsegmentation(dt_arr, gt_arr, num_classes, ignore_index)
            totals = (
                totals[0] + oracle_snap.intersect,
                totals[1] + oracle_snap.union,
                totals[2] + oracle_snap.pred,
                totals[3] + oracle_snap.label,
            )
            bg.submit(
                image_id,
                gt_arr.astype(np.uint32, copy=False),
                dt_arr.astype(np.uint32, copy=False),
            )
        candidate_summary = bg.finalize()
    oracle = _project_totals(*totals)
    candidate = _summary_to_snapshot(candidate_summary)
    return oracle, candidate
