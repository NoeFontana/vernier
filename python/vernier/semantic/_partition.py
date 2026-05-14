"""Semantic ADR-0046 partitioned-eval glue.

Same shape as :mod:`vernier.panoptic._partition`: the semantic
substrate's :class:`vernier.semantic.Dataset` carries a per-image
``Mapping[int, ndarray]`` of class-id label maps that is conceptually
filterable; the local :class:`vernier.semantic.EvalResult` does not
yet carry a ``slices`` field, and the per-class table emission is
class-id-keyed (not image-id-keyed), so the partitioned path needs a
Python-level loop that rebuilds the filtered dataset / predictions
mapping per slice and reruns :func:`evaluate_semantic_from_arrays`.

The ``overall`` summary is computed by a single unchanged
:func:`evaluate_semantic_from_arrays` call over the full mappings —
bit-identical to a non-partitioned :meth:`Evaluator.evaluate`, which
is ADR-0046's load-bearing parity claim. Per-slice work is one
mapping filter + one kernel + summarize pass; for COCO-scale
semantic this is fine. Heavier workloads should file an issue (the
C3 path would route through a Rust-side image-id filter on the
confusion-matrix accumulator).

The semantic schema carries an ``n_detections`` column for
cross-paradigm shape parity with the panoptic slices table, but
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
    evaluate_semantic_from_arrays,
    slices_batch_semantic,
)
from vernier._partition_spec import PartitionSpec, build_spec

if TYPE_CHECKING:  # pragma: no cover — type-checker only
    from vernier._types import ParityMode


#: Per-slice f64 columns reported when a slice carries no images.
#: Empty slices are legal in the partition spec but the semantic
#: kernel rejects an empty dataset (no images to evaluate), so we
#: short-circuit them in the orchestrator and report a zero-valued
#: row — same convention as the panoptic partition lane.
_EMPTY_SEMANTIC: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def _build_slices_batch(
    spec: PartitionSpec,
    *,
    summaries: dict[str, SemanticSummary | None],
) -> object:
    """Pack the per-slice ``(axis, value, n_images, n_detections, miou,
    fwiou, pixel_accuracy, mean_accuracy)`` rows into the canonical
    semantic slices Arrow RecordBatch via
    :func:`vernier._core.slices_batch_semantic`. ``n_detections`` is
    always ``0`` — see the module docstring. A ``None`` summary
    signals an empty slice (no images assigned); the metric columns
    are zero-filled (see :data:`_EMPTY_SEMANTIC`)."""
    rows: list[tuple[str, str, int, int, float, float, float, float]] = []
    for sl in spec.slices:
        key = f"{sl.axis}\x00{sl.value}"
        summary = summaries[key]
        if summary is None:
            miou, fwiou, pa, ma = _EMPTY_SEMANTIC
        else:
            miou, fwiou, pa, ma = (
                summary.miou,
                summary.fwiou,
                summary.pixel_accuracy,
                summary.mean_accuracy,
            )
        rows.append(
            (
                sl.axis,
                sl.value,
                len(sl.image_ids),
                0,  # semantic has no detections; shape parity column.
                miou,
                fwiou,
                pa,
                ma,
            )
        )
    return slices_batch_semantic(rows)


def _filter_label_maps(
    label_maps: Mapping[int, np.ndarray],
    image_ids: frozenset[int],
) -> dict[int, np.ndarray]:
    """Return a fresh dict carrying only the entries whose key is in
    ``image_ids``. Image ids the manifest assigns to a slice but that
    are absent from the input mapping are silently skipped — the spec
    builder has already filtered against the live dataset's image-id
    set, so this should only ever shrink to the GT side of the
    dataset / predictions intersection."""
    return {k: v for k, v in label_maps.items() if k in image_ids}


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
) -> tuple[SemanticSummary, object, int, int]:
    """Run the semantic partitioned eval as one
    :func:`evaluate_semantic_from_arrays` call per slice (plus one for
    ``overall``).

    Returns a ``(overall_summary, slices_record_batch_capsule,
    overall_n_images, overall_n_detections)`` tuple. The caller wraps
    these into the paradigm-local :class:`EvalResult` dataclass.
    ``overall_n_detections`` is ``0`` — see the module docstring on
    the ``n_detections`` column being a shape-parity placeholder for
    semantic.
    """
    all_image_ids = frozenset(int(i) for i in gt_label_maps)
    spec = build_spec(manifest, all_image_ids=all_image_ids, cross_axes=cross_axes)

    overall = evaluate_semantic_from_arrays(
        dict(gt_label_maps),
        dict(dt_label_maps),
        n_classes=n_classes,
        parity_mode=parity_mode,
        ignore_label=ignore_label,
        label_remap=dict(label_remap) if label_remap is not None else None,
        class_filter=class_filter,
        class_grouping=class_grouping,
    )
    overall_n_images = len(all_image_ids)
    overall_n_detections = 0

    # Empty slices short-circuit to a zero-valued row — the semantic
    # kernel rejects an empty image set, mirroring panoptic W6.
    summaries: dict[str, SemanticSummary | None] = {}
    for sl in spec.slices:
        key = f"{sl.axis}\x00{sl.value}"
        if not sl.image_ids:
            summaries[key] = None
            continue
        gt_sub = _filter_label_maps(gt_label_maps, sl.image_ids)
        dt_sub = _filter_label_maps(dt_label_maps, sl.image_ids)
        summaries[key] = evaluate_semantic_from_arrays(
            gt_sub,
            dt_sub,
            n_classes=n_classes,
            parity_mode=parity_mode,
            ignore_label=ignore_label,
            label_remap=dict(label_remap) if label_remap is not None else None,
            class_filter=class_filter,
            class_grouping=class_grouping,
        )

    slices_batch = _build_slices_batch(spec, summaries=summaries)
    return overall, slices_batch, overall_n_images, overall_n_detections
