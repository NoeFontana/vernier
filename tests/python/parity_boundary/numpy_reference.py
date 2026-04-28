"""NumPy sidecar oracle for boundary IoU (ADR-0010 §"Oracle (E3 sidecar)").

Implements the spec from ADR-0010 §"Algorithm specification (A2)" in pure
NumPy. No ``cv2`` import: the whole point of the sidecar is to let the
parity harness distinguish "vernier diverges from upstream" from "vernier
and upstream both diverge from the spec".

The vendored ``bowenc0221/boundary-iou-api`` is the strict-mode oracle
(E2). This file is the spec oracle (E3). They are independent.
"""

from __future__ import annotations

import math

import numpy as np


def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """Boundary band of a binary mask per ADR-0010 §A2.

    Returns a uint8 array same shape as `mask` with 1s on the boundary
    band and 0s elsewhere. Input may be bool or any integer dtype;
    binarized as `mask != 0`.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    h, w = mask.shape
    binary = (mask != 0).astype(np.uint8)

    # ADR-0010 §A2 step 1: half-to-even rounding. Python's builtin
    # `round` is banker's rounding (matches np.rint); we use the builtin
    # to mirror the reference's `int(round(...))` exactly.
    diag = math.sqrt(h * h + w * w)
    d = round(dilation_ratio * diag)
    if d < 1:
        d = 1

    # Step 2: 1-pixel zero pad on all sides.
    padded = np.pad(binary, 1, mode="constant", constant_values=0)

    # Step 3: erode by a (2d+1)x(2d+1) all-ones structuring element.
    # On binary input, iterating a 3x3 min-filter `d` times is bit-equal
    # to a single Chebyshev-ball erosion of radius `d` (ADR-0010 §A2,
    # also the Cons section "small correctness argument"). Reference-
    # quality only — no SIMD, no van Herk; production path lives in
    # vernier-mask.
    eroded = padded
    for _ in range(d):
        eroded = _min_filter_3x3(eroded)

    # Step 4: strip the pad.
    eroded = eroded[1 : h + 1, 1 : w + 1]

    # Step 5: B(M) = M AND NOT M_d. Since erosion is monotone, M_d <= M,
    # so subtraction matches the reference's `mask - mask_erode`.
    return (binary & ~eroded).astype(np.uint8)


def boundary_iou(
    gt_mask: np.ndarray,
    dt_mask: np.ndarray,
    dilation_ratio: float = 0.02,
) -> float:
    """Boundary IoU = ``|inter(B(gt), B(dt))| / |union(B(gt), B(dt))|``.

    Returns 0.0 when the union is 0. Both masks must share shape;
    raises ValueError otherwise.
    """
    if gt_mask.shape != dt_mask.shape:
        raise ValueError(f"shape mismatch: gt {gt_mask.shape} vs dt {dt_mask.shape}")

    gt_b = mask_to_boundary(gt_mask, dilation_ratio).astype(bool)
    dt_b = mask_to_boundary(dt_mask, dilation_ratio).astype(bool)

    inter = int(np.logical_and(gt_b, dt_b).sum())
    union = int(np.logical_or(gt_b, dt_b).sum())
    if union == 0:
        return 0.0
    return inter / union


def _min_filter_3x3(a: np.ndarray) -> np.ndarray:
    """3x3 minimum filter with implicit `1` border (matches the spec's
    BORDER_CONSTANT-with-fill-1 semantics: only the explicit zero ring
    contributes zeros to the min). Used as the inner loop of iterative
    Chebyshev erosion on binary input.
    """
    out = a.copy()
    out[1:, :] = np.minimum(out[1:, :], a[:-1, :])
    out[:-1, :] = np.minimum(out[:-1, :], a[1:, :])
    tmp = out.copy()
    out[:, 1:] = np.minimum(out[:, 1:], tmp[:, :-1])
    out[:, :-1] = np.minimum(out[:, :-1], tmp[:, 1:])
    return out
