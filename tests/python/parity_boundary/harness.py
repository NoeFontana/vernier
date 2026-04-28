"""Dual-oracle parity harness for boundary IoU (ADR-0010 §"Oracle (E2 + E3)").

Compares the vendored upstream oracle (``bowenc0221/boundary-iou-api``,
the cv2-based strict-mode reference) against the local NumPy sidecar
(``numpy_reference``, the spec oracle). Vernier itself is not exercised
here: the Python boundary-IoU surface lands in a follow-up PR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Import from the vendored tree. ``conftest.py`` puts
# ``oracle/boundary_iou_api/`` on ``sys.path`` so this resolves to the
# pinned upstream copy rather than any pip-installed package.
from boundary_iou.utils.boundary_utils import (  # type: ignore[import-not-found]
    mask_to_boundary as _upstream_mask_to_boundary,
)

from . import numpy_reference


@dataclass(frozen=True)
class BoundaryParityResult:
    dilation_ratio: float
    upstream_boundary: np.ndarray
    sidecar_boundary: np.ndarray
    upstream_iou: float
    sidecar_iou: float
    band_xor_count: int
    iou_diff: float


def compare_boundary(
    gt_mask: np.ndarray,
    dt_mask: np.ndarray,
    dilation_ratio: float = 0.02,
) -> BoundaryParityResult:
    """Run both oracles and return their outputs side-by-side."""
    if gt_mask.shape != dt_mask.shape:
        raise ValueError(f"shape mismatch: gt {gt_mask.shape} vs dt {dt_mask.shape}")

    gt_u8 = (gt_mask != 0).astype(np.uint8)
    dt_u8 = (dt_mask != 0).astype(np.uint8)

    upstream_gt = _upstream_mask_to_boundary(gt_u8, dilation_ratio)
    upstream_dt = _upstream_mask_to_boundary(dt_u8, dilation_ratio)
    sidecar_gt = numpy_reference.mask_to_boundary(gt_u8, dilation_ratio)
    sidecar_dt = numpy_reference.mask_to_boundary(dt_u8, dilation_ratio)

    upstream_inter = int(np.logical_and(upstream_gt, upstream_dt).sum())
    upstream_union = int(np.logical_or(upstream_gt, upstream_dt).sum())
    upstream_iou = (upstream_inter / upstream_union) if upstream_union else 0.0

    sidecar_iou = numpy_reference.boundary_iou(gt_u8, dt_u8, dilation_ratio)

    # Diff is on the GT band (the operand most often shared between
    # comparisons); a single scalar is enough for "do the oracles agree
    # on this fixture?". Per-pixel debugging uses the boundary arrays.
    band_xor = int(np.logical_xor(upstream_gt, sidecar_gt).sum()) + int(
        np.logical_xor(upstream_dt, sidecar_dt).sum()
    )

    return BoundaryParityResult(
        dilation_ratio=dilation_ratio,
        upstream_boundary=upstream_gt,
        sidecar_boundary=sidecar_gt,
        upstream_iou=upstream_iou,
        sidecar_iou=sidecar_iou,
        band_xor_count=band_xor,
        iou_diff=abs(upstream_iou - sidecar_iou),
    )
