"""Real-image parity test against val2017-derived semantic data
(ADR-0036, S3-B re-grade).

Loads the COCO val2017 panoptic-derived semantic workload (133
contiguous train-ids, ignore=255) from the bench cache, runs both the
vendored mmsegmentation ``IoUMetric`` (oracle) and vernier's
:class:`vernier.semantic.Evaluator` in ``parity_mode="strict"``
(candidate), and asserts bit-equality on the per-class u64 totals.

Skipped (with an actionable hint) when the panoptic cache isn't
provisioned; runs ~30s on the perfect-DT cell at val2017 scale and
is marked ``slow`` so the default ``just test`` loop is unaffected.

Streamed pair-by-pair via :func:`harness.run_streaming_pair`: peak
RAM is bounded by one decoded label-map per side plus the bg
worker's ``queue_capacity`` (default 8). Materializing the full
val2017 dataset as int64 (the natural mmseg dtype) would peak at
~21 GB — re-promoting to uint32 for vernier's input adds another
~10 GB and OOMs a 32 GB box.

The strict-mode parity claim is the integer confusion-matrix totals.
Float scalars (aAcc, mIoU, per-class IoU/Acc) follow trivially from
the same inputs, but are only loosely asserted here — the oracle does
its float arithmetic in torch (float32 promotion in the histc path)
while vernier does it in numpy float64. A separate per-quirk fixture
suite covers the float-edge rows (AL2 NaN handling, etc.); here we
care that the underlying counts agree.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import vernier.semantic as vsem

from .harness import run_streaming_pair


def _load_workload() -> tuple[Path, Path, int, int]:
    """Resolve the val2017-derived semantic workload from the panoptic cache.

    Returns ``(gt_dir, dt_dir, n_classes, ignore_label)``. Raises
    :class:`pytest.skip` when the panoptic cache (the source) isn't
    provisioned.
    """
    from panoptic_val_cache import (
        SEMANTIC_IGNORE_LABEL,
        ensure_semantic_gt,
        ensure_semantic_perfect_dt,
    )

    try:
        gt_dir, n_classes, _ = ensure_semantic_gt()
        dt_dir = ensure_semantic_perfect_dt()
    except FileNotFoundError as e:
        pytest.skip(str(e))

    return gt_dir, dt_dir, n_classes, SEMANTIC_IGNORE_LABEL


def _stream_pairs(
    image_ids: list[int],
    gt_pngs: dict[int, Path],
    dt_pngs: dict[int, Path],
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    for iid in image_ids:
        yield iid, vsem.decode_label_map_png(gt_pngs[iid]), vsem.decode_label_map_png(dt_pngs[iid])


@pytest.mark.parity_semantic
@pytest.mark.parity_semantic_val
@pytest.mark.slow
def test_strict_mode_parity_on_val2017_perfect_dt() -> None:
    from panoptic_val_cache import scan_label_map_pngs

    gt_dir, dt_dir, n_classes, ignore_label = _load_workload()

    gt_pngs = scan_label_map_pngs(gt_dir)
    dt_pngs = scan_label_map_pngs(dt_dir)
    assert gt_pngs.keys() == dt_pngs.keys(), (
        f"GT/DT image-id sets differ: GT-only={sorted(gt_pngs.keys() - dt_pngs.keys())[:5]}, "
        f"DT-only={sorted(dt_pngs.keys() - gt_pngs.keys())[:5]}"
    )

    image_ids = sorted(gt_pngs)
    oracle, candidate = run_streaming_pair(
        _stream_pairs(image_ids, gt_pngs, dt_pngs),
        num_classes=n_classes,
        ignore_index=ignore_label,
    )

    np.testing.assert_array_equal(
        candidate.intersect, oracle.intersect, err_msg="per-class TP totals diverge"
    )
    np.testing.assert_array_equal(
        candidate.label, oracle.label, err_msg="per-class GT-row totals diverge"
    )
    np.testing.assert_array_equal(
        candidate.pred, oracle.pred, err_msg="per-class DT-column totals diverge"
    )
    np.testing.assert_array_equal(
        candidate.union, oracle.union, err_msg="per-class union totals diverge"
    )

    # Derived from the same u64 totals via the same arithmetic in
    # :func:`harness._project_totals` — equal to the last bit by
    # construction, including NaN cells (AL2 zero-support classes).
    np.testing.assert_array_equal(candidate.iou, oracle.iou, err_msg="per-class IoU diverges")
    np.testing.assert_array_equal(candidate.acc, oracle.acc, err_msg="per-class Acc diverges")
    np.testing.assert_array_equal(candidate.aacc, oracle.aacc, err_msg="aAcc diverges")

    present = ~np.isnan(candidate.iou)
    assert present.any(), "no classes present in val2017-derived GT — workload is degenerate"
    np.testing.assert_array_equal(candidate.iou[present], np.ones(present.sum(), dtype=np.float64))
