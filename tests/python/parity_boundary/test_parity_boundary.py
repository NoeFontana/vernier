"""Boundary-IoU parity tests (ADR-0010).

Asserts that the vendored upstream oracle and the NumPy sidecar produce
identical boundary bands and identical boundary-IoU values on a small
fixture corpus. Vernier itself is not yet wired in; that lands in a
follow-up PR (the harness API is stable in advance).
"""

from __future__ import annotations

import numpy as np
import pytest

from .harness import compare_boundary
from .numpy_reference import boundary_iou, mask_to_boundary

# Mirrors `BOUNDARY_PARITY_EPS` in
# `crates/vernier-core/src/boundary_parity.rs`. Hardcoded rather than
# imported because vernier-core has no Python surface yet.
BOUNDARY_PARITY_EPS = 1e-9

pytestmark = pytest.mark.parity_boundary


def _square(h: int, w: int, y0: int, x0: int, side: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0 : y0 + side, x0 : x0 + side] = 1
    return m


def _l_shape(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[5:25, 5:12] = 1
    m[18:25, 5:25] = 1
    return m


def _circle(h: int, w: int, cy: int, cx: int, r: int) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).astype(np.uint8)


def _random_mask(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((h, w)) > 0.5).astype(np.uint8)


SHAPE_FIXTURES: list[tuple[str, np.ndarray]] = [
    ("square_5x5_in_20x20", _square(20, 20, 7, 7, 5)),
    ("l_shape_30x30", _l_shape(30, 30)),
    ("circle_r10_in_40x40", _circle(40, 40, 20, 20, 10)),
    ("empty_25x25", np.zeros((25, 25), dtype=np.uint8)),
    ("full_25x25", np.ones((25, 25), dtype=np.uint8)),
    ("random_32x32_seed_0", _random_mask(32, 32, 0)),
    # Degenerate sizes: `d` is clamped to 1, so the boundary is the
    # whole foreground. Guards against off-by-one in the pad/erode loop.
    ("single_pixel_on", np.ones((1, 1), dtype=np.uint8)),
    ("single_pixel_off", np.zeros((1, 1), dtype=np.uint8)),
]


@pytest.mark.parametrize(("name", "mask"), SHAPE_FIXTURES, ids=[n for n, _ in SHAPE_FIXTURES])
def test_boundary_band_matches_upstream(name: str, mask: np.ndarray) -> None:
    """Vendored oracle and NumPy sidecar produce bit-identical bands."""
    del name  # used only for the parametrize id
    result = compare_boundary(mask, mask)
    assert result.band_xor_count == 0


def test_identical_nonempty_masks_iou_is_one() -> None:
    mask = _circle(40, 40, 20, 20, 10)
    assert boundary_iou(mask, mask) == pytest.approx(1.0, abs=BOUNDARY_PARITY_EPS)


def test_identical_empty_masks_iou_is_zero() -> None:
    empty = np.zeros((25, 25), dtype=np.uint8)
    # Explicit "no foreground" defense: empty bands -> union 0 -> 0.0.
    assert boundary_iou(empty, empty) == 0.0


def test_disjoint_masks_iou_is_zero() -> None:
    gt = _square(40, 40, 2, 2, 8)
    dt = _square(40, 40, 28, 28, 8)
    assert boundary_iou(gt, dt) == 0.0


def test_partial_overlap_iou_matches_oracles() -> None:
    # Two overlapping rectangles; a hand-traceable case where neither
    # boundary nor IoU is degenerate.
    gt = _square(30, 30, 5, 5, 14)
    dt = _square(30, 30, 9, 9, 14)
    result = compare_boundary(gt, dt)
    assert result.band_xor_count == 0
    assert result.iou_diff < BOUNDARY_PARITY_EPS
    assert 0.0 < result.sidecar_iou < 1.0


def test_lvis_dilation_ratio_matches_oracles() -> None:
    """LVIS variant `dilation_ratio = 0.008` (ADR-0010 §A2)."""
    gt = _circle(80, 80, 40, 40, 25)
    dt = _circle(80, 80, 42, 38, 25)
    result = compare_boundary(gt, dt, dilation_ratio=0.008)
    assert result.band_xor_count == 0
    assert result.iou_diff < BOUNDARY_PARITY_EPS


@pytest.mark.parametrize(("name", "mask"), SHAPE_FIXTURES, ids=[n for n, _ in SHAPE_FIXTURES])
def test_iou_agreement_against_self(name: str, mask: np.ndarray) -> None:
    """Boundary IoU agrees within `BOUNDARY_PARITY_EPS` for self-vs-self."""
    del name
    result = compare_boundary(mask, mask)
    assert result.iou_diff < BOUNDARY_PARITY_EPS


def test_mask_to_boundary_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2D"):
        mask_to_boundary(np.zeros((3, 3, 3), dtype=np.uint8))


def test_boundary_iou_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        boundary_iou(np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 11), dtype=np.uint8))
