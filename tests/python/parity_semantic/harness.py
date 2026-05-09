"""Parity harness for the semantic-segmentation evaluation surface
(ADR-0028, ADR-0036).

Imports the verbatim-vendored `mmseg.evaluation.metrics.iou_metric`
through the stub-injected `mmengine` / `mmseg.registry` /
`prettytable` modules wired in :mod:`conftest`. The oracle calls
``IoUMetric.intersect_and_union`` and ``IoUMetric.total_area_to_metrics``
as static methods — bypassing ``BaseMetric.process`` and its DataLoader
sample shape, which is incidental to the parity claim.

For now this harness exposes only the oracle side: vernier-side
comparison wiring lives in a follow-up PR (the Rust evaluator emits
the same per-class confusion-matrix rows; cross-checking is
mechanical once the Python facade lands). The single test in
``test_parity_semantic.py`` asserts the vendored bytes + stubs
produce the expected metrics on a hand-built fixture, which is what
proves the vendor is correctly wired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# Imported for its side-effect: conftest must have been loaded so the
# stub modules are registered before this import resolves.
from mmseg.evaluation.metrics.iou_metric import IoUMetric  # pyright: ignore[reportMissingImports]
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SemanticSnapshot:
    """Per-class semantic-segmentation metrics returned by the oracle.

    Shape mirrors ``IoUMetric.total_area_to_metrics``'s output for
    ``iou_metrics=['mIoU']`` — the four per-class arrays plus the
    scalar overall accuracy. Future per-quirk fixtures may add
    ``Dice`` / ``Fscore`` columns; the dataclass is extended in
    lock-step with the harness call site.
    """

    aacc: float
    iou: NDArray[np.float64]
    acc: NDArray[np.float64]
    intersect: NDArray[np.float64]
    union: NDArray[np.float64]
    pred: NDArray[np.float64]
    label: NDArray[np.float64]


def run_mmsegmentation(
    pred: NDArray[np.integer[Any]],
    gt: NDArray[np.integer[Any]],
    num_classes: int,
    ignore_index: int,
) -> SemanticSnapshot:
    """Run the vendored mmsegmentation IoUMetric on a single
    prediction / ground-truth pair.

    The oracle's static methods accept torch tensors; we pass numpy
    arrays through ``torch.from_numpy``. The returned arrays are
    converted back to numpy for the snapshot.
    """
    pred_t = torch.from_numpy(np.ascontiguousarray(pred).astype(np.int64))  # pyright: ignore[reportPrivateImportUsage]
    gt_t = torch.from_numpy(np.ascontiguousarray(gt).astype(np.int64))  # pyright: ignore[reportPrivateImportUsage]

    intersect, union, area_pred, area_label = IoUMetric.intersect_and_union(
        pred_t, gt_t, num_classes, ignore_index
    )

    metrics = IoUMetric.total_area_to_metrics(intersect, union, area_pred, area_label, ["mIoU"])

    return SemanticSnapshot(
        aacc=float(metrics["aAcc"]),
        iou=np.asarray(metrics["IoU"], dtype=np.float64),
        acc=np.asarray(metrics["Acc"], dtype=np.float64),
        intersect=intersect.numpy().astype(np.float64),
        union=union.numpy().astype(np.float64),
        pred=area_pred.numpy().astype(np.float64),
        label=area_label.numpy().astype(np.float64),
    )
