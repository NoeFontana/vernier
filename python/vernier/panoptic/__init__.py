"""Panoptic-quality (PQ) evaluation surface (ADR-0025).

Per ADR-0029, the panoptic-segmentation evaluation paradigm lives under
``vernier.panoptic``. Sibling to :mod:`vernier.instance` and
:mod:`vernier.semantic`. The Rust kernel ships in the
``vernier-panoptic`` crate; this module is a thin Python wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass

from vernier._core import (
    ClassPanopticStats,
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
from vernier._types import ParityMode

__all__ = [
    "ClassPanopticStats",
    "Dataset",
    "Evaluator",
    "Predictions",
    "Summary",
]


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
