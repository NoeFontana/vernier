"""Semantic ADR-0046 partitioned-eval glue.

The semantic substrate accumulates per-image confusion matrices that
sum (u64-additive) into the global matrix used by
:func:`summarize_with_options`. The
:func:`vernier._core.evaluate_semantic_partitioned` FFI runs the
per-image fold **exactly once** to populate a
``Vec<(image_id, ConfusionMatrix)>`` and then aggregates +
summarizes those matrices under (a) no filter for ``overall`` and
(b) each slice's image-id set for the per-slice rows — the C3 axiom
of ADR-0046.

The ``overall`` summary is bit-identical to a non-partitioned
:meth:`Evaluator.evaluate` over the same mappings — ADR-0046's
load-bearing parity claim — because the un-filtered sum reproduces
the canonical accumulation step verbatim. Confusion-matrix sums are
u64-additive, so the parity contract is unconditional (no f64
non-associativity to worry about, unlike the panoptic path).

The semantic schema carries an ``n_detections`` column for cross-
paradigm shape parity with the panoptic slices table, but
semantic-segmentation has no notion of a detection — the column is
populated with ``0`` per row. The ``n_images`` column carries the
manifest-assigned image count, which *is* meaningful.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from vernier._core import (
    SemanticSummary,
    evaluate_semantic_partitioned,
)

if TYPE_CHECKING:  # pragma: no cover — type-checker only
    from vernier._types import ParityMode


def evaluate_partitioned(
    gt_label_maps: Mapping[int, np.ndarray],
    dt_label_maps: Mapping[int, np.ndarray],
    *,
    n_classes: int,
    parity_mode: ParityMode,
    ignore_label: int | None,
    label_remap: dict[int, int] | None,
    class_filter: list[int] | None,
    class_grouping: list[tuple[str, list[int]]] | None,
    manifest: object,
    cross_axes: Sequence[Sequence[str]] | None,
    num_threads: int | None = None,
) -> tuple[SemanticSummary, object, int, int]:
    """Run the semantic partitioned eval through the C3 FFI.

    Returns a ``(overall_summary, slices_record_batch_capsule,
    overall_n_images, overall_n_detections)`` tuple. The caller wraps
    these into the paradigm-local :class:`EvalResult` dataclass.
    ``overall_n_detections`` is ``0`` — semantic has no detection
    notion; the column is shape-parity with panoptic / instance.

    Per ADR-0046's C3 axiom, the per-image confusion-matrix fold
    runs exactly once regardless of slice count; per-slice rows are
    produced by summing the retained per-image matrices under each
    slice's image-id filter and summarizing the result.
    """
    cross = [list(t) for t in cross_axes] if cross_axes is not None else None
    report = evaluate_semantic_partitioned(
        dict(gt_label_maps),
        dict(dt_label_maps),
        n_classes,
        parity_mode,
        manifest,
        ignore_label=ignore_label,
        label_remap=dict(label_remap) if label_remap is not None else None,
        class_filter=class_filter,
        class_grouping=class_grouping,
        cross_axes=cross,
        num_threads=num_threads,
    )
    return (
        report.overall,
        report.slices_capsule(),
        int(report.overall_n_images),
        int(report.overall_n_detections),
    )
