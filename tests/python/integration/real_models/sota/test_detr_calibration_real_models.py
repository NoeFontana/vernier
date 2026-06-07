"""Real-prediction parity for DETR-R50 calibration vs the numpy oracle.

Calibration analogue of ``test_detr_real_models.py``: same SHA-pinned
DETR-R50 prediction cache, same COCO val2017 GT, same ``0.05`` score
floor — but the gate is now ``vernier-calibration`` (ADR-0018) versus
the clean-room numpy reference at
``tests/python/parity_calibration/numpy_oracle.py``. Both implementations
consume the **same** per-image cells (lifted out of vernier's grid
``eval_imgs``), so the test isolates calibration-kernel parity from
matching-kernel parity (which the existing detection smoke already gates
bit-equal — the dtScores ULP drift documented there does NOT surface
here because the calibration kernel reads the bool ``dt_matched`` /
``dt_ignore`` surfaces, not the raw score float).

What this suite gates:

- **Strict tier on the per-bin u64 surface** — ``reliability.count`` is
  bit-equal across all bins for every (iou_index, default params)
  combination. ``count`` is a pure histogram reduction; drift here is
  an outright accumulator bug because integers cannot shift under
  reduction-order changes.
- **Aligned tier, 8 ULP relative + absolute, on the float surface** —
  the scalar ``ece`` / ``mce`` reductions and every float column of
  the reliability table (``mean_score`` / ``accuracy`` / ``gap`` /
  ``ci_lo`` / ``ci_hi``). The kernel uses Faer's pulp-driven
  per-bin reductions; numpy uses ``np.bincount``. On a quantised score
  stream of ~150k detections both sides agree to within a couple of
  ULP. 8 ULP keeps any genuine kernel divergence (e.g. a wrong Wilson
  arithmetic) well above the gate; ``atol`` mirrors the panoptic
  rationale so a per-bin metric that collapses to exact ``0.0`` on one
  side and a sub-ULP non-zero on the other still passes.

Why three IoU thresholds: ADR-0018 §"DETR-aware defaults" claims
quantile binning + ``min_score=0.05`` + Wilson CIs hold across the
COCO T-axis, not just at ``iou=0.5``. Gating at ``0.5``, ``0.75``, and
``0.95`` exercises both the high-IoU tail (where ``dt_matched`` is
sparse — many detections fail to clear the stricter IoU bar and the
oracle's histogram concentrates around the no-object cluster) and the
permissive ``0.5`` regime. ADR-0018 specifies a single IoU per
``calibrate(...)`` call; the "0.5:0.95" mean-over-thresholds shape
familiar from AP does not apply at the calibration surface, so this
test covers the two endpoints + 0.75 instead of a meta-aggregate.

Skips cleanly when the ``real-models`` extra is missing
(``transformers`` import fails in conftest), or ``VERNIER_COCO_CACHE``
doesn't point at a populated val2017 layout (GT JSON + images
directory), or the DETR-R50 prediction cache isn't on disk — the same
skip semantics as ``test_detr_real_models.py``. Inference cost is
identical because both tests share the
:func:`detr_predictions_path` session fixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vernier._core as _vernier_core

from ....parity_calibration.harness import (
    snapshot_oracle,
    snapshot_vernier,
)
from ....parity_calibration.numpy_oracle import CalibrationParams, PerImageCell

pytestmark = [pytest.mark.real_models, pytest.mark.slow]

#: Permissive score floor — matches the populator's
#: ``_SCORE_THRESHOLD`` in ``_detr_predict.py`` and the ADR-0018
#: "DETR-aware default" ``min_score=0.05``. Pinned here as a module
#: constant so a future float-typo split between populator and oracle
#: would fail loud rather than silently shifting bin masses.
_MIN_SCORE: float = 0.05

#: ADR-0018 DETR-aware defaults: 15 quantile bins + Wilson CIs.
_N_BINS: int = 15

#: Canonical COCO 10-point IoU ladder (0.5:0.05:0.95). The matching
#: kernel's default; pinned here so we can map the human-readable
#: ``iou=0.5/0.75/0.95`` test parameters to integer ``iou_index``
#: slots without round-tripping through ``cells.iou_to_index`` on the
#: opaque handle (which is a separate parity surface itself).
_COCO_IOU_LADDER: tuple[float, ...] = tuple(
    float(x) for x in np.linspace(0.5, 0.95, 10, dtype=np.float64)
)

#: 8 ULP of float64 — used as BOTH ``rtol`` and ``atol`` on the
#: aligned-tier scalar / per-bin float assertions. Mirrors the panoptic
#: gate's symmetric band: a bin whose oracle gap rounds to exact
#: ``0.0`` while vernier yields a sub-ULP non-zero (or vice versa)
#: still passes; an ``rtol``-only check would fail at ``rtol * 0 = 0``
#: on that case. Same 8-ULP justification as the Mask2Former-panoptic
#: cell — keeps reduction-order drift inside the gate but any genuine
#: kernel divergence outside it.
_CAL_PARITY_RTOL: float = 8.0 * float(np.finfo(np.float64).eps)
_CAL_PARITY_ATOL: float = _CAL_PARITY_RTOL

# Pycocotools' detection defaults (`cocoeval.Params.setDetParams`). The
# grid needs the largest cap to seed the per-image cells; the
# calibration kernel does NOT re-apply max_dets — it folds whatever
# detections survive the matching pass plus the ``min_score`` cutoff.
# Matches the parity smoke's ``_DEFAULT_MAX_DETS``.
_DEFAULT_MAX_DETS: tuple[int, ...] = (1, 10, 100)


def _iou_index(iou: float) -> int:
    """Resolve ``iou`` to its slot on :data:`_COCO_IOU_LADDER`.

    Uses ``np.isclose`` with the default tolerance — equivalent to the
    kernel's ``iou_to_index`` lookup under PARITY_EPS, but stays
    inside numpy so the lookup itself never depends on a vernier-side
    method we'd otherwise need to call on an opaque handle. Loud-fails
    on a value that doesn't pin to a slot; the test parameters are
    fixed constants, so a miss here is a typo in the test, not a
    runtime fluke.
    """
    matches = np.flatnonzero(np.isclose(_COCO_IOU_LADDER, iou, atol=1e-9))
    if matches.size != 1:
        raise ValueError(
            f"iou={iou} does not pin to exactly one slot on the COCO ladder "
            f"{list(_COCO_IOU_LADDER)} (matches={matches.tolist()})"
        )
    return int(matches[0])


def _extract_cells_by_category(
    eval_imgs: list[dict[str, Any] | None],
    *,
    n_area_ranges: int,
    n_images: int,
) -> dict[int, list[PerImageCell]]:
    """Slice the ``(K, A, I)`` flat ``eval_imgs`` into the per-class
    cell store the calibration oracle consumes.

    The calibration kernel reads only the **``a = 0``** (``all`` area)
    bucket — the per-class oracle harness (``_build_cells_dict``)
    documents this and the kernel asserts ``n_area_ranges == 1`` at
    construction time. Walks ``eval_imgs[k * A * I + 0 * I + i]`` for
    every ``(k, i)`` and reshapes each cell's pycocotools-shaped
    ``dtMatches`` / ``dtIgnore`` (float ids / u8 bool) into the
    ``PerImageCell`` shape (bool / bool). ``dt_matched[t, d] =
    dtMatches[t, d] > 0`` mirrors pycocotools' convention (a zero in
    ``dtMatches`` means "unmatched at this T").
    """
    cells_by_category: dict[int, list[PerImageCell]] = {}
    stride = n_area_ranges * n_images
    # ``eval_imgs`` is the K*A*I flat list. We only want the a=0 cells,
    # but the K stride depends on n_area_ranges so we can't pre-filter
    # via a single slice — index explicitly to keep the indexing
    # convention visible at the call site.
    n_categories = len(eval_imgs) // stride
    # Guard against silent truncation: integer division would mask a
    # trailing partial class block if the grid's flat layout ever
    # diverges from the documented (K, A, I) shape. Loud-fail rather
    # than silently dropping the last category's cells.
    assert len(eval_imgs) == n_categories * stride, (
        f"eval_imgs shape mismatch: {len(eval_imgs)} not divisible by stride={stride}"
    )
    for k in range(n_categories):
        base = k * stride  # start of class k's A*I block.
        category_id: int | None = None
        per_class: list[PerImageCell] = []
        for i in range(n_images):
            cell = eval_imgs[base + 0 * n_images + i]
            if cell is None:
                continue
            scores = np.asarray(cell["dtScores"], dtype=np.float64)
            if scores.size == 0:
                continue
            # Keep ``dtMatches`` as int64: the entries are GT IDs (or 0
            # for unmatched), and casting through float64 would lose
            # precision on IDs above 2**53. COCO val2017 stays well
            # under that bound but the same builder is reused for LVIS,
            # where IDs can collide with the f64 mantissa. ``> 0`` on
            # int64 is also one less cast than ``> 0.0`` on float64.
            dt_matches_arr = np.asarray(cell["dtMatches"], dtype=np.int64)
            dt_ignore_arr = np.asarray(cell["dtIgnore"], dtype=np.uint8)
            # pycocotools-shaped (T, D). The matching kernel emits one
            # row per IoU threshold; the calibration oracle reads a
            # single iou_index but the full (T, D) shape must travel
            # so the kernel's per-cell shape check passes.
            dt_matched = dt_matches_arr > 0
            dt_ignore = dt_ignore_arr.astype(bool, copy=False)
            per_class.append(
                PerImageCell(
                    dt_scores=scores,
                    dt_matched=dt_matched,
                    dt_ignore=dt_ignore,
                )
            )
            if category_id is None:
                category_id = int(cell["category_id"])
        if not per_class or category_id is None:
            # No non-empty cell for this class — skip rather than
            # emit a (category_id=None, []) row. The oracle handles
            # missing classes silently; matching that semantics keeps
            # the per-class table row-for-row comparable.
            continue
        cells_by_category[category_id] = per_class
    return cells_by_category


def _build_cells_by_k(
    cells_by_category: dict[int, list[PerImageCell]],
) -> dict[int, list[PerImageCell]]:
    """Remap ``category_id`` → dense ``k`` (the kernel's slot index).

    Same remap the synthetic ``parity_calibration.harness`` does (see
    ``load_fixture_cells``); duplicated here because the SOTA cell
    builds cells from ``eval_imgs`` rather than ``cells.json`` and the
    harness helper expects a fixture path. Sort by ``category_id`` so
    the per-class table row order is deterministic and matches what
    the kernel's ``0..n_categories`` iteration emits.
    """
    return {
        k: cells_by_category[cat_id] for k, cat_id in enumerate(sorted(cells_by_category.keys()))
    }


def _build_grid_and_cells(
    gt_bytes: bytes,
    dt_bytes: bytes,
) -> tuple[dict[int, list[PerImageCell]], int]:
    """Drive ``evaluate_bbox_grid`` once + lift the per-class cells out.

    Returns ``(cells_by_k, n_iou_thresholds)``. The grid runs at the
    canonical COCO defaults (10-point IoU ladder, 4 area ranges,
    maxDet=100); the caller picks an ``iou_index`` against the same
    ladder in :data:`_COCO_IOU_LADDER`. Matching cost is the dominant
    term; we only ever call this once per test invocation, then
    re-fold across IoU thresholds in-process — calibration is cheap
    relative to matching by orders of magnitude.
    """
    max_dets_per_image = max(_DEFAULT_MAX_DETS)
    grid = _vernier_core.evaluate_bbox_grid(
        gt_bytes,
        dt_bytes,
        "strict",  # ADR-0002: matching parity is strict at the cell level.
        max_dets_per_image,
        use_cats=True,
    )
    eval_imgs = grid.eval_imgs()
    cells_by_category = _extract_cells_by_category(
        eval_imgs,
        n_area_ranges=grid.n_area_ranges,
        n_images=grid.n_images,
    )
    cells_by_k = _build_cells_by_k(cells_by_category)
    return cells_by_k, len(_COCO_IOU_LADDER)


@pytest.fixture(scope="module")
def detr_calibration_cells(
    coco_gt_path: Path,
    detr_predictions_path: Path,
) -> tuple[dict[int, list[PerImageCell]], int]:
    """Module-scoped grid + per-class cells for the calibration tests.

    Reading the GT (~22 MiB) and DT (~9 MiB) JSON blobs and re-running
    ``evaluate_bbox_grid`` (the dominant cost — bbox matching across
    80 classes x ~5000 images x 10 IoU thresholds) is IoU-independent:
    the calibration kernel re-folds the same ``(T, D)`` cell shape
    against a chosen ``iou_index``. Hoisting to module scope amortises
    those three costs across all parametrisations of
    :func:`test_detr_r50_calibration_parity_vs_numpy_oracle`; without
    this the fixture would run 3x and the test would dominate the
    real-models suite wall clock.
    """
    gt_bytes = coco_gt_path.read_bytes()
    dt_bytes = detr_predictions_path.read_bytes()
    return _build_grid_and_cells(gt_bytes, dt_bytes)


@pytest.mark.parametrize("iou", [0.5, 0.75, 0.95])
def test_detr_r50_calibration_parity_vs_numpy_oracle(
    detr_calibration_cells: tuple[dict[int, list[PerImageCell]], int],
    iou: float,
) -> None:
    """Two-tier calibration parity on real DETR-R50 predictions.

    Strict tier: per-bin u64 ``count`` bit-equal (integer reductions
    cannot drift under reduction-order changes — see module docstring).
    Aligned tier: ``ece`` / ``mce`` scalars + every float column of
    the reliability table at 8 ULP rtol + atol. Single-IoU calibrate()
    call per parametrisation; ADR-0018's surface picks one T-slot at
    a time, so 0.5 / 0.75 / 0.95 cover the COCO T-axis endpoints + a
    midpoint without a meta-aggregate concept the calibration kernel
    doesn't expose.
    """
    cells_by_k, n_iou_thresholds = detr_calibration_cells

    params = CalibrationParams(
        iou_index=_iou_index(iou),
        n_bins=_N_BINS,
        binning="quantile",
        min_score=_MIN_SCORE,
        confidence="wilson",
        per_class=False,
        per_class_aggregation="macro",
    )

    oracle = snapshot_oracle(cells_by_k, params)
    candidate = snapshot_vernier(cells_by_k, params, n_iou_thresholds)

    # Sanity gate before the per-bin diff — a divergent ``n_detections``
    # would explain ECE/MCE drift via a different denominator, not a
    # reduction-order one, and the module docstring would be wrong
    # about what this test catches.
    assert oracle.n_detections == candidate.n_detections, (
        f"n_detections diverge at iou={iou}: "
        f"oracle={oracle.n_detections} vs vernier={candidate.n_detections}"
    )
    assert oracle.effective_n_bins == candidate.effective_n_bins, (
        f"effective_n_bins diverge at iou={iou}: "
        f"oracle={oracle.effective_n_bins} vs vernier={candidate.effective_n_bins}"
    )

    # Strict tier — u64 ``count`` per bin. The synthetic-fixture
    # harness's ``assert_snapshots_match`` already enforces bit-
    # equality on integer columns regardless of rtol, but we re-check
    # here so a future harness refactor that loosens the integer
    # check doesn't silently weaken this PR's load-bearing claim, and
    # so the failure surface points specifically at the calibration
    # u64 reduction (not at any of the float reductions, which the
    # aligned-tier helper covers).
    oracle_count = oracle.reliability["count"]
    candidate_count = candidate.reliability["count"]
    assert np.array_equal(oracle_count, candidate_count), (
        f"per-bin u64 count diverges at iou={iou}:\n"
        f"oracle ={oracle_count}\nvernier={candidate_count}"
    )

    # Aligned tier — ECE/MCE scalars + float columns of the
    # reliability table at 8 ULP rtol + atol. The synthetic-fixture
    # harness pins ``aligned``-mode at ``4 * eps``; we widen to 8 ULP
    # here because the real-prediction reduction over ~150k
    # detections accumulates more drift than the 10-60 detection
    # fixtures the harness's 4-ULP gate was calibrated on.
    _assert_calibration_aligned(oracle, candidate, iou=iou)


def _assert_calibration_aligned(
    oracle: Any,
    candidate: Any,
    *,
    iou: float,
) -> None:
    """8-ULP rtol+atol comparison of ECE/MCE + reliability float cols.

    Mirrors ``parity_calibration.harness.assert_snapshots_match``'s
    aligned mode but with the wider 8-ULP gate justified above and a
    per-IoU error tag so a multi-parametrisation failure points at the
    exact threshold. NaN positions must still match bit-exact (zero-
    count bins emit NaN on both sides per the R2 convention).
    """
    # Scalars.
    np.testing.assert_allclose(
        oracle.ece,
        candidate.ece,
        rtol=_CAL_PARITY_RTOL,
        atol=_CAL_PARITY_ATOL,
        err_msg=f"ece at iou={iou}",
    )
    np.testing.assert_allclose(
        oracle.mce,
        candidate.mce,
        rtol=_CAL_PARITY_RTOL,
        atol=_CAL_PARITY_ATOL,
        err_msg=f"mce at iou={iou}",
    )

    # Bin edges are STRICT-tier per ADR-0018 §P1: quantile bin edges
    # are computed by a pure ``np.quantile`` reduction over the same
    # score stream on both sides, with no per-bin float accumulator in
    # the way. Any drift here is a real divergence in the edge
    # construction (sort order, interpolation choice, or quantile
    # weighting), not a reduction-order artefact, so the 8-ULP aligned
    # band is the wrong gate. Bit-equal check FIRST so a strict-tier
    # regression surfaces as the strict-tier failure it is, not as a
    # loosened-tolerance pass.
    for edge_col in ("score_lo", "score_hi"):
        oc_edge = np.asarray(oracle.reliability[edge_col], dtype=np.float64)
        cc_edge = np.asarray(candidate.reliability[edge_col], dtype=np.float64)
        np.testing.assert_array_equal(
            oc_edge,
            cc_edge,
            err_msg=f"reliability.{edge_col} (strict per ADR-0018 §P1) at iou={iou}",
        )

    # Float columns of the reliability table (aligned tier — per-bin
    # float reductions over ~150k detections, where reduction-order
    # drift is expected at the few-ULP level).
    float_cols = ("mean_score", "accuracy", "gap", "ci_lo", "ci_hi")
    for col in float_cols:
        oc = np.asarray(oracle.reliability[col], dtype=np.float64)
        cc = np.asarray(candidate.reliability[col], dtype=np.float64)
        nan_o = np.isnan(oc)
        nan_c = np.isnan(cc)
        assert np.array_equal(nan_o, nan_c), (
            f"reliability.{col} NaN positions diverge at iou={iou}:\n"
            f"oracle ={nan_o}\nvernier={nan_c}"
        )
        np.testing.assert_allclose(
            oc[~nan_o],
            cc[~nan_c],
            rtol=_CAL_PARITY_RTOL,
            atol=_CAL_PARITY_ATOL,
            err_msg=f"reliability.{col} at iou={iou}",
        )
