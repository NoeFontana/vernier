"""Real-prediction parity smoke for Mask2Former Swin-T ADE-semantic vs mmsegmentation.

Sibling to ``test_mask2former_panoptic_real_models.py`` but on
ADE20K val (semantic segmentation). The model is
``facebook/mask2former-swin-tiny-ade-semantic``; predictions land
under ``mask2former_ade_cache_dir()`` as one single-channel label-map
PNG per image (train-id ``0..149``, no JSON sidecar — semantic is
just label maps).

What this suite gates:

- **Strict bit-equality** on the per-class u64 confusion-matrix
  totals (``intersect`` / ``union`` / ``pred`` / ``label``). Both
  mmsegmentation's ``IoUMetric`` and vernier-semantic produce
  identical totals from the same per-pixel (gt, dt) arrays; any
  drift implies a real divergence in the accumulator. Derived float
  scalars (mIoU, aAcc, per-class IoU/Acc) follow trivially from the
  same u64 inputs.

Streaming evaluation: ADE20K val is 2000 images at ~512x512, each
image's gt+dt label map is ~512KB as uint32, ~256KB as uint8 —
materializing the full dataset is bounded (~1GB), but we stream
pair-by-pair via :func:`harness.run_streaming_pair` for consistency
with the COCO-derived semantic val smoke. Peak RAM stays at one
decoded label-map per side.

Skips cleanly when:

- ``real-models`` extra is missing.
- ``MASK2FORMER_ADE_REVISION`` is the ``_UNPINNED_REVISION`` sentinel.
- ADE20K val cache is not provisioned (i.e. user hasn't run
  ``python -m ade20k_val_cache``).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import vernier.semantic as vsem

from ....parity_semantic.harness import run_streaming_pair

pytestmark = [pytest.mark.real_models, pytest.mark.slow]


def _stream_pairs(
    image_ids: list[int],
    gt_pngs: dict[int, Path],
    dt_pngs: dict[int, Path],
) -> Iterator[tuple[int, NDArray[np.integer], NDArray[np.integer]]]:
    """Yield ``(image_id, gt_arr, dt_arr)`` triples decoded from PNGs.

    Mirrors the COCO-derived semantic val smoke's stream pattern.
    Both arrays are uint32 after :func:`vsem.decode_label_map_png` —
    the mmseg oracle's :func:`intersect_and_union` promotes to int64,
    vernier-semantic consumes uint32 natively, so the streaming
    fold cost stays one decoded label-map per side.
    """
    for iid in image_ids:
        yield iid, vsem.decode_label_map_png(gt_pngs[iid]), vsem.decode_label_map_png(dt_pngs[iid])


def test_mask2former_ade_parity_vs_mmsegmentation(
    ade20k_val_gt_dir: Path,
    mask2former_ade_cache_paths: Path,
) -> None:
    """Strict-tier parity vs mmsegmentation on real Mask2Former predictions.

    Stream-folds the (gt, dt) pairs through the vendored ``IoUMetric``
    accumulator and vernier's ``BackgroundEvaluator`` in lockstep;
    asserts the per-class u64 totals match exactly. Failure surfaces
    as a class-level diff before the float projection runs, isolating
    the divergence to a specific (intersect | union | pred | label)
    bucket.
    """
    from ade20k_val_cache import ADE20K_IGNORE_LABEL, ADE20K_NUM_CLASSES, scan_label_map_pngs

    gt_pngs = scan_label_map_pngs(ade20k_val_gt_dir)
    dt_pngs = scan_label_map_pngs(mask2former_ade_cache_paths)
    assert gt_pngs.keys() == dt_pngs.keys(), (
        f"GT/DT image-id sets differ: "
        f"GT-only={sorted(gt_pngs.keys() - dt_pngs.keys())[:5]}, "
        f"DT-only={sorted(dt_pngs.keys() - gt_pngs.keys())[:5]}. "
        f"This implies the populator stopped partway — re-running it "
        f"is idempotent at the per-image level."
    )

    image_ids = sorted(gt_pngs)
    oracle, candidate = run_streaming_pair(
        _stream_pairs(image_ids, gt_pngs, dt_pngs),
        num_classes=ADE20K_NUM_CLASSES,
        ignore_index=ADE20K_IGNORE_LABEL,
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

    # Sanity guard: the model emits non-trivial predictions across multiple
    # classes. A degenerate cache (all-zeros PNGs) would still pass the
    # totals assertions above by accident; this check catches it.
    present_classes = (candidate.label > 0).sum()
    assert present_classes >= 50, (
        f"only {present_classes} classes present in the ADE20K val GT — "
        f"likely a degenerate cache. ADE20K val should cover most of the "
        f"150 classes."
    )
