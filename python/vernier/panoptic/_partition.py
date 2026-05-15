"""Panoptic ADR-0046 partitioned-eval glue.

The panoptic substrate runs a per-image matching + attribution pass
that yields per-image deltas (one `HashMap<CategoryId, PqStat>` per
image); these deltas are u64-counter + f64 IoU-sum payloads that are
cheap to retain and image-id-filterable at summarize time. The
:func:`vernier._core.evaluate_panoptic_partitioned` FFI runs the
matching pass **exactly once** to populate the per-image delta vec
and then folds + summarizes that vec under (a) no filter for
``overall`` and (b) each slice's image-id set for the per-slice rows
— the C3 axiom of ADR-0046, image-axis analogue of ADR-0026's
K-axis subset-at-summarize-time.

The ``overall`` summary is bit-identical to a non-partitioned
:meth:`Evaluator.evaluate` over the same handles — ADR-0046's load-
bearing parity claim — because the un-filtered fold + summarize
reproduces the canonical aggregation step verbatim.

The earlier C1 fallback (one ``evaluate_panoptic`` call per slice)
remains structurally available but is no longer the default path;
LVIS-scale callers see ~20x speedups for ~20 slices because the
matching pass dominates wall time.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vernier._core import (
    PanopticDataset,
    PanopticPredictions,
    PanopticSummary,
    evaluate_panoptic_partitioned,
)

if TYPE_CHECKING:  # pragma: no cover — type-checker only
    from vernier._types import ParityMode


def evaluate_partitioned(
    gt: PanopticDataset,
    dt: PanopticPredictions,
    *,
    parity_mode: ParityMode,
    things_stuff_split: bool,
    boundary: bool,
    dilation_ratio: float,
    manifest: object,
    cross_axes: Sequence[Sequence[str]] | None,
) -> tuple[PanopticSummary, object, int, int]:
    """Run the panoptic partitioned eval through the C3 FFI.

    Returns a ``(overall_summary, slices_record_batch_capsule,
    overall_n_images, overall_n_detections)`` tuple. The caller wraps
    these into the paradigm-local :class:`EvalResult` dataclass.

    Per ADR-0046's C3 axiom, the matching + attribution pass runs
    exactly once regardless of slice count; per-slice rows are
    produced by folding the retained per-image deltas under each
    slice's image-id filter and summarizing the result.
    """
    cross = [list(t) for t in cross_axes] if cross_axes is not None else None
    report = evaluate_panoptic_partitioned(
        gt,
        dt,
        parity_mode,
        things_stuff_split,
        boundary,
        dilation_ratio,
        manifest,
        cross_axes=cross,
    )
    return (
        report.overall,
        report.slices_capsule(),
        int(report.overall_n_images),
        int(report.overall_n_detections),
    )
