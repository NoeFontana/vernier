"""Real-prediction parity smoke for rfdetr-segnano boundary IoU vs ``bowenc0221/boundary-iou-api``.

Sibling to the three Mask2Former / DETR SOTA cells: real DT (the
rfdetr-segnano cache the TIDE harness already populates), real GT
(COCO val2017), strict + aligned two-tier parity claim. Unlike the
other SOTA cells, this one does NOT run inference — the rfdetr cache
is keyed on the pip-pinned ``RFDETR_VERSION`` and is the only one the
TIDE side already populates as part of its own validation run. We
reuse the bytes-on-disk; boundary IoU is just a different metric over
the same RLE masks.

What this suite gates:

- **Strict bit-equality on the integer-ish detection-matching surface
  AND the AP / AR summary stats** — per-category ``tp / fp / fn``
  derived from ``eval_imgs[*].dtMatches / gtMatches`` at IoU=0.5 /
  aRng="all" / maxDet=100, plus the dense ``precision`` / ``recall``
  / ``counts`` aggregates, plus the 12-stat AP / AR summary vector.
  ``counts`` is the ``[T, R, K, A, M]`` shape vector and must be
  bit-identical; ``precision`` / ``recall`` are integer ratios over
  identical match sets and must also be bit-equal. ``stats`` is a
  pure reduction over ``precision`` / ``recall``, so the same
  bit-equality holds — the parser-drift band documented below
  affects only the score-threshold projection (``scores`` tensor),
  not the AP integral itself. Mirrors the DETR-R50 cell exactly.
- **Aligned tier, ``rtol = 2 * eps``** — the dense ``scores`` tensor
  (per-recall-grid score-threshold projection of dtScores) at the
  parser-drift band documented on the DETR-R50 cell (see
  ``test_detr_real_models.py`` module docstring). rfdetr-segnano
  emits per-detection scores via the same JSON path that produces
  the near-tie f64 rounding drift between ``serde_json`` and
  ``strtod``; the ``scores`` tensor inherits the same drift through
  the projection but does NOT propagate into match decisions
  (matches are made on boundary IoU, not score values) or into the
  AP integrals (AP depends on detection order, not exact score
  value). ``rtol`` (not ``atol``) so the band tracks score magnitude
  across the full ``[0.05, 1.0]`` range.

Skips cleanly when:

- the ``real-models`` extra is missing (conftest's ``importorskip``),
- the vendored ``boundary_iou_api`` checkout is missing,
- the rfdetr-segnano predictions cache is not populated (no inference
  is triggered here — the TIDE harness owns that step),
- the COCO val2017 GT is not provisioned.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ....parity_boundary.e2e_harness import BoundaryEvalSnapshot, snapshot

pytestmark = [pytest.mark.real_models, pytest.mark.slow]

#: 2 ULP of float64 expressed as a RELATIVE tolerance. Mirrors the
#: DETR-R50 cell's ``_DTSCORES_RTOL`` exactly — same documented
#: ``serde_json`` ↔ ``strtod`` near-tie f64 rounding drift on
#: JSON-encoded per-detection scores. The drift propagates through
#: the score-threshold projection (``scores`` tensor) but not into
#: match decisions (matches are made on boundary IoU, not score
#: values) or into the AP integrals (AP depends on detection ORDER,
#: not exact score value), so both the integer-ish surface and the
#: ``stats`` AP / AR summary stay strictly bit-equal in the strict
#: tier above. ``rtol``, not ``atol``: at the 0.05 score floor
#: ``atol = 2 * eps`` would be ~40 ULP and silently absorb genuine
#: kernel divergence; ``rtol`` keeps the band ≤ 4 ULP wide across
#: the full score range.
_SCORES_RTOL = 2.0 * float(np.finfo(np.float64).eps)


def _per_class_tp_fp_fn(
    eval_imgs: list[dict[str, object] | None],
    *,
    iou_thr_idx: int = 0,
    area_rng: tuple[float, float] = (0.0, 1e10),
    max_det: int = 100,
) -> dict[int, tuple[int, int, int]]:
    """Derive per-category ``(tp, fp, fn)`` from a boundary eval_imgs list.

    The COCOeval state machine carries the matching decisions in
    ``eval_imgs[*].{dtMatches, gtMatches, dtIgnore, gtIgnore}``;
    summarising to per-category counts gives the integer surface a
    drift in match decisions would shift. Both oracles share the same
    eval_imgs shape per :func:`tests.python.parity_boundary.e2e_harness`,
    so the same derivation applies to both ``ref`` and ``cand``.

    - ``iou_thr_idx=0`` → IoU=0.50 row of the (T=10) threshold sweep;
      a single index is enough for the integer surface — any drift
      in match decisions shifts at least one IoU row.
    - ``area_rng=(0, 1e10)`` → the "all" entry, the first of the four
      area ranges (all / small / medium / large).
    - ``max_det=100`` → the largest of ``_DEFAULT_MAX_DETS=(1, 10, 100)``
      and the only one the AP-driving stats use.
    """
    out: dict[int, tuple[int, int, int]] = {}
    # Local name matches the COCO field's lowercase shape (the
    # original COCOeval param is ``aRng`` — camelCase; this is the
    # snake-case rename to satisfy ruff N806 without renaming the
    # comparison target).
    arng_target = [float(area_rng[0]), float(area_rng[1])]
    for entry in eval_imgs:
        if entry is None:
            continue
        if entry["maxDet"] != max_det:
            continue
        if entry["aRng"] != arng_target:
            continue
        cat_id = int(entry["category_id"])  # type: ignore[arg-type]
        dt_matches = np.asarray(entry["dtMatches"])
        gt_matches = np.asarray(entry["gtMatches"])
        dt_ignore = np.asarray(entry["dtIgnore"])
        gt_ignore = np.asarray(entry["gtIgnore"])

        # Row T=iou_thr_idx of the (T, D) / (T, G) matrices. Boolean
        # tp / fp masks subtract ignores so a low-conf det matched
        # only to an ignored GT doesn't double-count.
        dt_match_row = dt_matches[iou_thr_idx]
        dt_ignore_row = dt_ignore[iou_thr_idx]
        gt_match_row = gt_matches[iou_thr_idx]

        tp = int(np.logical_and(dt_match_row > 0, dt_ignore_row == 0).sum())
        fp = int(np.logical_and(dt_match_row == 0, dt_ignore_row == 0).sum())
        fn = int(np.logical_and(gt_match_row == 0, gt_ignore == 0).sum())

        cur = out.get(cat_id, (0, 0, 0))
        out[cat_id] = (cur[0] + tp, cur[1] + fp, cur[2] + fn)
    return out


def _assert_strict_surface(
    ref: BoundaryEvalSnapshot,
    cand: BoundaryEvalSnapshot,
) -> None:
    """Strict bit-equality on the AP-driving integer-ish surface and stats.

    Per-class ``tp / fp / fn`` (derived from ``eval_imgs``), plus the
    dense ``precision`` / ``recall`` arrays, the ``counts`` shape, AND
    the 12-stat AP / AR summary. ``stats`` is a pure reduction over
    ``precision`` / ``recall`` (themselves integer ratios over
    identical match sets), so the parser-drift band the aligned tier
    absorbs cannot reach the summary numerics. Mirrors the DETR-R50
    cell's strict surface exactly. Drift in any of these is a real
    boundary-kernel divergence, not the documented score-parser drift
    the aligned tier absorbs.
    """
    assert ref.counts == cand.counts, f"counts differ: {ref.counts} vs {cand.counts}"
    np.testing.assert_array_equal(ref.precision, cand.precision, err_msg="precision")
    np.testing.assert_array_equal(ref.recall, cand.recall, err_msg="recall")
    np.testing.assert_array_equal(ref.stats, cand.stats, err_msg="stats")

    ref_per_class = _per_class_tp_fp_fn(ref.eval_imgs)
    cand_per_class = _per_class_tp_fp_fn(cand.eval_imgs)
    ref_cats = set(ref_per_class)
    cand_cats = set(cand_per_class)
    only_ref = ref_cats - cand_cats
    only_cand = cand_cats - ref_cats
    assert not only_ref, f"oracle has categories vernier dropped: {sorted(only_ref)[:10]}"
    assert not only_cand, f"vernier has categories oracle dropped: {sorted(only_cand)[:10]}"
    for cat_id in sorted(ref_cats):
        rtp, rfp, rfn = ref_per_class[cat_id]
        ctp, cfp, cfn = cand_per_class[cat_id]
        assert (rtp, rfp, rfn) == (ctp, cfp, cfn), (
            f"cat {cat_id} (tp,fp,fn) diverges: oracle={(rtp, rfp, rfn)} "
            f"vs vernier={(ctp, cfp, cfn)}"
        )


def test_rfdetr_segnano_boundary_parity_vs_bowenc0221(
    coco_gt_path: Path,
    rfdetr_segnano_predictions_path: Path,
) -> None:
    """Two-tier boundary-IoU parity vs ``boundary_iou_api`` on the
    rfdetr-segnano cache.

    Strict tier: per-class ``tp / fp / fn`` + dense ``precision`` /
    ``recall`` / ``counts`` + 12-stat AP / AR summary.
    Aligned tier: dense ``scores`` tensor at ``rtol = 2 * eps`` (parser
    drift band, same constant as DETR-R50 cell).
    """
    ref = snapshot("oracle", coco_gt_path, rfdetr_segnano_predictions_path)
    cand = snapshot("vernier", coco_gt_path, rfdetr_segnano_predictions_path)

    # Strict tier — integer-ish detection-matching surface plus the
    # AP / AR summary. Boundary IoU drift propagates through the
    # match-decision integer here; the score-parser drift the aligned
    # tier absorbs reaches only the per-recall-grid ``scores``
    # projection, not match decisions and not the AP integrals.
    _assert_strict_surface(ref, cand)

    # Aligned tier — the dense ``scores`` tensor at 2 ULP relative.
    # Scoped narrowly to ``scores`` so the parser-drift band doesn't
    # accidentally re-relax ``precision`` / ``recall`` / ``stats``
    # (already strict above). ``scores`` is the per-recall-grid
    # score-threshold projection of dtScores and inherits the
    # ``serde_json`` ↔ ``strtod`` near-tie rounding drift one-for-one
    # — same shape as the DETR-R50 cell's ``_DTSCORES_RTOL`` gate.
    np.testing.assert_allclose(
        ref.scores, cand.scores, rtol=_SCORES_RTOL, atol=0.0, err_msg="scores"
    )
