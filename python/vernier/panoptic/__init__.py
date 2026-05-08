"""Panoptic-quality (PQ) evaluation surface (ADR-0025).

Per ADR-0029, the panoptic-segmentation evaluation paradigm lives under
``vernier.panoptic``. Sibling to :mod:`vernier.instance` and
:mod:`vernier.semantic`. The Rust kernel ships in the
``vernier-panoptic`` crate; this module is a thin Python wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from vernier._core import (
    BackgroundPanopticEvaluator as BackgroundEvaluator,
)

# Re-export the five distributed-eval exception types under the
# panoptic namespace so callers catching `vernier.panoptic.PartialFormatMismatch`
# match the same class object as `vernier.instance.PartialFormatMismatch`
# (ADR-0032: shared paradigm-agnostic error classes).
from vernier._core import (
    ClassPanopticStats,
    PartialDatasetMismatch,
    PartialFormatMismatch,
    PartialParamsMismatch,
    PartialPartitionOverlap,
    PartialRankCollision,
    evaluate_panoptic,
)
from vernier._core import (
    PanopticDataset as Dataset,
)
from vernier._core import (
    PanopticPredictions as Predictions,
)
from vernier._core import (
    PanopticSummary as Summary,
)
from vernier._core import (
    StreamingPanopticEvaluator as StreamingEvaluator,
)
from vernier._types import ParityMode

__all__ = [
    "BackgroundEvaluator",
    "ClassPanopticStats",
    "Dataset",
    "Evaluator",
    "ParityMode",
    "PartialDatasetMismatch",
    "PartialFormatMismatch",
    "PartialParamsMismatch",
    "PartialPartitionOverlap",
    "PartialRankCollision",
    "Predictions",
    "StreamingEvaluator",
    "Summary",
    "decode_label_map_png",
]


def decode_label_map_png(path: str | Path) -> NDArray[np.uint32]:
    """Decode a panoptic RGB PNG into a `(H, W)` ``uint32`` segment-id
    label map via the ``rgb2id`` convention (panopticapi/evaluation.py
    rgb2id: ``r + 256*g + 256²*b``).

    Lazy-imports Pillow; raises a structured :class:`ImportError` if it
    isn't installed. Three channels are required — a non-RGB PNG is
    rejected with :class:`ValueError`. Single-channel class-id label
    maps (semantic-segmentation) belong in
    :func:`vernier.semantic.Dataset.from_files`, which has its own
    decoder.
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Pillow is required for `vernier.panoptic.decode_label_map_png`; "
            "install via `pip install Pillow` (or include it in your dev "
            "environment)."
        ) from e
    rgb = np.array(Image.open(path), dtype=np.uint32)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(
            f"panoptic PNG {path!s} must be RGB (3 channels); got shape {rgb.shape!r}. "
            f"Single-channel semantic label maps belong in `vernier.semantic`."
        )
    return rgb[:, :, 0] + 256 * rgb[:, :, 1] + 256 * 256 * rgb[:, :, 2]


@dataclass(frozen=True, slots=True)
class Evaluator:
    """Panoptic-quality (PQ) evaluator (ADR-0025).

    Sibling to :class:`vernier.instance.Evaluator`. Per category,
    ``PQ_c`` is computed directly as
    ``iou_c / (TP_c + 0.5*FP_c + 0.5*FN_c)`` (panopticapi form, quirk
    **W1**) — algebraically equal to ``SQ_c * RQ_c`` but f64
    non-associative, so the direct form is what holds bit-equality.
    Global ``PQ`` / ``SQ`` / ``RQ`` are unweighted means over the
    present-categories subset (W2, W3, W7); things / stuff buckets are
    independent unweighted means over their subsets (W4).

    Defaults match panopticapi's ``pq_compute`` shape, except for
    ``parity_mode``, which defaults to ``"corrected"`` (the ADR-0002
    recommendation for net-new users); migrating users wanting
    bit-exact panopticapi behavior should set ``parity_mode="strict"``.

    ``boundary=True`` raises :class:`NotImplementedError`. Boundary PQ
    is deferred to a follow-up ADR (ADR-0025 §"explicitly does not
    decide" Q3 / Z1). The composition rule in the bowenc0221 fork is
    not the instance-case ``min(mask_iou, boundary_iou)`` and resolving
    it requires its own pass.
    """

    parity_mode: ParityMode = "corrected"
    things_stuff_split: bool = True
    boundary: bool = False

    def __post_init__(self) -> None:
        if self.boundary:
            raise NotImplementedError(
                "boundary panoptic-quality is deferred to a follow-up ADR "
                "(ADR-0025 §'explicitly does not decide' Q3 / Z1). "
                "The composition rule in the bowenc0221 fork is not the "
                "instance-case min(mask_iou, boundary_iou) and resolving it "
                "requires its own pass."
            )

    def evaluate(
        self,
        gt: Dataset,
        dt: Predictions,
    ) -> Summary:
        """Run the panoptic-quality evaluation.

        ``gt`` and ``dt`` must have been built via
        :meth:`Dataset.from_arrays` / :meth:`Predictions.from_arrays`
        (file-loading helpers ship in a follow-up).
        """
        return evaluate_panoptic(gt, dt, self.parity_mode, self.things_stuff_split)

    def stream(
        self,
        categories: bytes,
        *,
        retain_per_image_deltas: bool = False,
        rank_id: int | None = None,
    ) -> StreamingEvaluator:
        """Build a :class:`StreamingEvaluator` that shares this
        evaluator's ``parity_mode`` and ``things_stuff_split``.

        ``categories`` is a JSON byte string of the form
        ``[{"id": int, "isthing": bool}, ...]`` — same shape as the
        ``categories`` argument to :meth:`Dataset.from_arrays`.

        ``retain_per_image_deltas=True`` enables strict-mode bit-
        equality across distributed-eval ranks (ADR-0032 PR-E
        §"Determinism") at ~2× streaming memory cost. Default off
        keeps the single-rank path lean.

        ``rank_id``, when set, identifies this evaluator's rank in
        a multi-process eval. Required for strict-mode cross-rank
        merge via :meth:`StreamingEvaluator.from_partials`.
        """
        return StreamingEvaluator(
            categories,
            self.parity_mode,
            things_stuff_split=self.things_stuff_split,
            retain_per_image_deltas=retain_per_image_deltas,
            rank_id=rank_id,
        )

    def background(
        self,
        categories: bytes,
        *,
        retain_per_image_deltas: bool = False,
        rank_id: int | None = None,
        queue_capacity: int = 8,
        worker_affinity: int | None = None,
        worker_nice: int = 5,
        shutdown_timeout_seconds: float = 5.0,
    ) -> BackgroundEvaluator:
        """Build a :class:`BackgroundEvaluator` (ADR-0014 + ADR-0032)
        that shares this evaluator's ``parity_mode`` and
        ``things_stuff_split``.

        The returned wrapper owns a single dedicated worker thread
        running a :class:`StreamingEvaluator` of the same shape;
        :meth:`BackgroundEvaluator.submit` enqueues per-image
        ``(gt_label_map, gt_segments_info, dt_label_map,
        dt_segments_info)`` tuples and returns immediately. Use this
        when the panoptic-quality kernel measurably stalls the
        training loop — the per-image attribute pass is the dominant
        cost at COCO-panoptic scale.

        ``retain_per_image_deltas=True`` enables strict-mode bit-
        equality across distributed-eval ranks (ADR-0032
        §"Determinism") at ~2× streaming memory cost. The five
        queueing / scheduling knobs mirror
        :class:`vernier.instance.Evaluator.background`.
        """
        return BackgroundEvaluator(
            categories,
            self.parity_mode,
            things_stuff_split=self.things_stuff_split,
            retain_per_image_deltas=retain_per_image_deltas,
            rank_id=rank_id,
            queue_capacity=queue_capacity,
            worker_affinity=worker_affinity,
            worker_nice=worker_nice,
            shutdown_timeout_seconds=shutdown_timeout_seconds,
        )
