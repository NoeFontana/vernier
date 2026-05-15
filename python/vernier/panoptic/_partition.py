"""Panoptic ADR-0046 partitioned-eval glue.

The panoptic substrate's :class:`PanopticDataset` / :class:`PanopticPredictions`
are immutable handles built from per-image label-map dicts;
:class:`PanopticSummary` is computed from per-image accumulations
rather than from an AP-shaped accumulator tensor that the
``vernier_core::partition::evaluate_partitioned`` C3 orchestrator
could fan out over. As a phase-1 fallback (per ADR-0046
§"Performance"), this module drives a per-slice Python loop that
calls :func:`vernier._core.evaluate_panoptic` once per slice over a
filtered ``PanopticDataset`` / ``PanopticPredictions`` pair (via the
Rust-side :meth:`subset_by_image_ids` accessors).

The ``overall`` summary is computed by a single unchanged call to
:func:`evaluate_panoptic` over the full input — bit-identical to a
non-partitioned :meth:`Evaluator.evaluate` over the same handles,
which is ADR-0046's load-bearing parity claim. The per-slice loop is
order-O(slices) extra matching work over the un-partitioned path;
LVIS-scale panoptic users who need the C3 path (one matching pass,
N cheap summarize passes) should file an issue. For COCO-panoptic
scale crossed with a handful of slices this is fine.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vernier._core import (
    PanopticDataset,
    PanopticPredictions,
    PanopticSummary,
    evaluate_panoptic,
    slices_batch_panoptic,
)
from vernier._partition_spec import PartitionSpec, build_spec

if TYPE_CHECKING:  # pragma: no cover — type-checker only
    from vernier._types import ParityMode


#: Per-slice f64 columns reported when a slice carries no images.
#: Empty slices are legal in the partition spec (they arise naturally
#: for ``__unassigned__`` buckets when the manifest covers every
#: image) but the panoptic kernel rejects an empty category filter
#: (quirk **W6**); rather than re-routing through ``parity_mode=
#: "corrected"`` we short-circuit empty slices in the orchestrator
#: and report a zero-valued row. Matches the Rust spec builder's
#: "empty slices are legal" comment.
_EMPTY_PQ: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _build_slices_batch(
    spec: PartitionSpec,
    *,
    dt_segment_counts: dict[str, int],
    summaries: dict[str, PanopticSummary | None],
) -> object:
    """Pack the per-slice ``(axis, value, n_images, n_detections, pq,
    sq, rq)`` rows into the canonical panoptic slices Arrow
    RecordBatch via :func:`vernier._core.slices_batch_panoptic`.

    ``dt_segment_counts`` and ``summaries`` are both keyed by the same
    ``(axis, value)``-joined cell key so the Python wrapper only walks
    the slice list once. A ``None`` summary signals an empty slice
    (no images assigned); the metric columns are zero-filled.
    """
    rows: list[tuple[str, str, int, int, float, float, float]] = []
    for sl in spec.slices:
        key = f"{sl.axis}\x00{sl.value}"
        summary = summaries[key]
        if summary is None:
            pq_v, sq_v, rq_v = _EMPTY_PQ
        else:
            pq_v, sq_v, rq_v = summary.pq, summary.sq, summary.rq
        rows.append(
            (
                sl.axis,
                sl.value,
                len(sl.image_ids),
                dt_segment_counts[key],
                pq_v,
                sq_v,
                rq_v,
            )
        )
    return slices_batch_panoptic(rows)


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
    """Run the panoptic partitioned eval as one ``evaluate_panoptic``
    call per slice (plus one for ``overall``).

    Returns a ``(overall_summary, slices_record_batch_capsule,
    overall_n_images, overall_n_detections)`` tuple. The caller wraps
    these into the paradigm-local :class:`EvalResult` dataclass.

    The ``overall`` summary is computed by calling
    :func:`evaluate_panoptic` once over the full handles — i.e. the
    same code path :meth:`Evaluator.evaluate` would take without
    ``manifest=`` — so the bit-identical-overall parity contract is
    preserved by construction.
    """
    all_image_ids = frozenset(int(i) for i in gt.image_ids())
    spec = build_spec(manifest, all_image_ids=all_image_ids, cross_axes=cross_axes)

    # Overall — un-partitioned eval over the full handles.
    overall = evaluate_panoptic(
        gt,
        dt,
        parity_mode,
        things_stuff_split,
        boundary=boundary,
        dilation_ratio=dilation_ratio,
    )
    overall_n_images = int(gt.num_images)
    overall_n_detections = int(dt.num_segments)

    # Per-slice loop. Each slice rebuilds a filtered (Dataset,
    # Predictions) pair via the Rust-side subset accessor (cheap clone
    # of the per-image entries) and re-runs the kernel + summarize.
    # Empty slices short-circuit to a zero-valued row (see _EMPTY_PQ)
    # — the panoptic kernel rejects empty inputs (quirk W6) and
    # re-routing through `parity_mode="corrected"` per-slice would
    # diverge from the user-selected parity contract.
    summaries: dict[str, PanopticSummary | None] = {}
    dt_segment_counts: dict[str, int] = {}
    for sl in spec.slices:
        ids = sorted(sl.image_ids)
        key = f"{sl.axis}\x00{sl.value}"
        if not ids:
            summaries[key] = None
            dt_segment_counts[key] = 0
            continue
        sub_gt = gt.subset_by_image_ids(ids)
        sub_dt = dt.subset_by_image_ids(ids)
        summaries[key] = evaluate_panoptic(
            sub_gt,
            sub_dt,
            parity_mode,
            things_stuff_split,
            boundary=boundary,
            dilation_ratio=dilation_ratio,
        )
        dt_segment_counts[key] = int(dt.num_segments_for(ids))

    slices_batch = _build_slices_batch(
        spec, dt_segment_counts=dt_segment_counts, summaries=summaries
    )
    return overall, slices_batch, overall_n_images, overall_n_detections
