"""LRP / oLRP reference oracle (numpy).

This module is the spec the Rust ``vernier.instance.optimal_lrp``
implementation will be validated against. It is pure numpy / Python
with no vernier imports. Correctness is pinned by hand-computed
assertions on small synthetic fixtures (see ``test_oracle.py``).

Algorithm summary (canonical, per Oksuz et al. TPAMI 2021, eqs. 6-10)
====================================================================

Per category ``k``, given a fixed TP-IoU threshold ``tau_tp`` (default
``0.5``) and a confidence threshold ``s``:

  - **Greedy matching.** Detections of category ``k`` are sorted by
    score-descending and, per ``(image, k)`` cell, greedily matched to
    the highest-similarity unmatched non-crowd GT whose IoU >=
    ``tau_tp``.
  - **At a candidate ``s``** the active set is detections with
    ``score >= s``:
      * ``NTP(s)`` = active matched detections.
      * ``NFP(s)`` = active unmatched detections.
      * ``NFN(s)`` = non-crowd GTs with no active detection matched to
        them.

  - **LRP(s)** is then (paper eq. 9):

      ``LRP(s) = ( (1/(1 - tau_tp)) * sum_TP (1 - IoU_i)``
                ``+ NFP(s) + NFN(s) )``
              ``/ (NTP(s) + NFP(s) + NFN(s))``

    with the convention ``LRP = 0`` if the denominator is zero.

  - **oLRP** is the minimum of ``LRP(s)`` over a tau-grid (default
    ``np.arange(0.0, 1.0 + 1e-9, 0.01)``). The argmin's
    ``s*`` is the "optimal" confidence threshold; ties on the grid are
    broken by selecting the **larger ``s``** (more conservative — fewer
    detections survive).

  - **At ``s*`` the components decompose oLRP additively** (paper eq. 10):

      ``oLRP_Loc = (sum_TP* (1 - IoU_i) / NTP*) / (1 - tau_tp)``
      ``oLRP_FP  = NFP* / (NTP* + NFP*)``
      ``oLRP_FN  = NFN* / (NTP* + NFN*)``

  - **Headline metrics** are the means across categories that have at
    least one non-crowd GT. Per-class entries with no positive GTs
    contribute NaN to ``tau_per_class`` / ``olrp_per_class`` and are
    excluded from the headline mean.

Crowd discipline
================

A ``iscrowd=1`` GT is treated as ignore: it cannot create FNs (so an
unmatched crowd GT does not contribute to ``NFN``), and a detection
matched to a crowd GT does not count as TP or FP (it is ignored, as if
removed). This matches COCO eval semantics used throughout vernier.

Implementation notes
====================

This oracle implements the algorithm faithfully and without
optimisation. The Rust implementation will run a vectorised matching
loop; this one runs a Python ``for`` loop and is correct by
construction.

No vernier imports. No SciPy. Numpy only.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np

# Pairwise similarity kernel: ``(gts, dts) -> (D, G)`` IoU matrix.
SimilarityFn: TypeAlias = Callable[[list[dict[str, Any]], list[dict[str, Any]]], np.ndarray]

DEFAULT_TAU_GRID: np.ndarray = np.arange(0.0, 1.0 + 1e-9, 0.01)
"""101-point tau grid (0.00, 0.01, ..., 1.00). Matches the kemaloksuz
reference implementation's default and is dense enough for the
hand-computed fixtures here."""


def bbox_iou(gts: list[dict[str, Any]], dts: list[dict[str, Any]]) -> np.ndarray:
    """Pairwise IoU between COCO-style ``[x, y, w, h]`` bboxes.

    Args:
        gts: ground-truth annotation dicts, each with a ``bbox`` key.
        dts: detection dicts, each with a ``bbox`` key.

    Returns:
        ``(D, G)`` IoU matrix in row-major order (one row per detection,
        one column per GT). Returns the zero matrix when either list is
        empty.
    """
    if not gts or not dts:
        return np.zeros((len(dts), len(gts)), dtype=np.float64)
    gt_arr = np.array([g["bbox"] for g in gts], dtype=np.float64).reshape(-1, 4)
    dt_arr = np.array([d["bbox"] for d in dts], dtype=np.float64).reshape(-1, 4)
    gt_x1 = gt_arr[:, 0]
    gt_y1 = gt_arr[:, 1]
    gt_x2 = gt_arr[:, 0] + gt_arr[:, 2]
    gt_y2 = gt_arr[:, 1] + gt_arr[:, 3]
    gt_area = gt_arr[:, 2] * gt_arr[:, 3]
    dt_x1 = dt_arr[:, 0]
    dt_y1 = dt_arr[:, 1]
    dt_x2 = dt_arr[:, 0] + dt_arr[:, 2]
    dt_y2 = dt_arr[:, 1] + dt_arr[:, 3]
    dt_area = dt_arr[:, 2] * dt_arr[:, 3]
    inter_x1 = np.maximum(dt_x1[:, None], gt_x1[None, :])
    inter_y1 = np.maximum(dt_y1[:, None], gt_y1[None, :])
    inter_x2 = np.minimum(dt_x2[:, None], gt_x2[None, :])
    inter_y2 = np.minimum(dt_y2[:, None], gt_y2[None, :])
    inter_w = np.clip(inter_x2 - inter_x1, 0.0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0.0, None)
    inter = inter_w * inter_h
    union = dt_area[:, None] + gt_area[None, :] - inter
    safe = np.where(union > 0.0, union, 1.0)
    return np.where(union > 0.0, inter / safe, 0.0).astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Per-class greedy matching.
# ---------------------------------------------------------------------------


def _match_per_class(
    gts: list[dict[str, Any]],
    dts: list[dict[str, Any]],
    similarity_fn: SimilarityFn,
    tp_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Run score-descending greedy matching per ``(image, category)`` cell.

    Args:
        gts: all ground-truth dicts in this class.
        dts: all detection dicts in this class.
        similarity_fn: ``(gts, dts) -> (D, G)`` IoU matrix.
        tp_threshold: IoU floor for a (dt, gt) pair to be a TP match.

    Returns:
        Tuple ``(dt_score, dt_matched, dt_ignore, dt_iou, n_pos_gt)``:

        - ``dt_score``: shape ``(N,)`` of detection scores in the input order.
        - ``dt_matched``: shape ``(N,)`` bool — True if matched to a
          non-crowd GT above the threshold.
        - ``dt_ignore``: shape ``(N,)`` bool — True if matched only to a
          crowd GT (and so does not count as TP or FP).
        - ``dt_iou``: shape ``(N,)`` IoU of each TP match (``0.0`` for
          unmatched / ignored detections).
        - ``n_pos_gt``: number of non-crowd GTs.

    Matching is per-image, per-category. The ``(N,)`` arrays are aligned
    to the input ``dts`` order (not score order) so the caller can apply
    confidence thresholds without an extra permutation.
    """
    n_dt = len(dts)
    dt_score = np.array([float(d["score"]) for d in dts], dtype=np.float64)
    dt_matched = np.zeros(n_dt, dtype=bool)
    dt_ignore = np.zeros(n_dt, dtype=bool)
    dt_iou = np.zeros(n_dt, dtype=np.float64)

    # Group by image_id (matching is independent across images).
    images = sorted({int(g["image_id"]) for g in gts} | {int(d["image_id"]) for d in dts})

    n_pos_gt = sum(1 for g in gts if not bool(g.get("iscrowd", 0)))

    for img_id in images:
        gt_local_idx = [i for i, g in enumerate(gts) if int(g["image_id"]) == img_id]
        dt_local_idx = [i for i, d in enumerate(dts) if int(d["image_id"]) == img_id]
        if not dt_local_idx:
            continue
        gts_img = [gts[i] for i in gt_local_idx]
        dts_img = [dts[i] for i in dt_local_idx]
        iou = similarity_fn(gts_img, dts_img)
        is_crowd = np.array([bool(g.get("iscrowd", 0)) for g in gts_img], dtype=bool)
        scores = np.array([float(d["score"]) for d in dts_img], dtype=np.float64)

        # Score-descending traversal with stable tie-breaking. Mirrors
        # the deterministic "first detection wins on tie" rule the
        # vernier matching engine uses (see ADR-0005 / quirk A1).
        perm = np.argsort(-scores, kind="mergesort")
        gt_taken = np.zeros(len(gts_img), dtype=bool)

        for d_local in perm:
            # Prefer non-crowd GTs first; if none qualify, fall back to
            # crowd GTs (matched -> ignore).
            best_iou = tp_threshold
            best_g = -1
            for g_local in range(len(gts_img)):
                if is_crowd[g_local]:
                    continue
                if gt_taken[g_local]:
                    continue
                v = float(iou[d_local, g_local])
                if v >= best_iou:
                    best_iou = v
                    best_g = g_local
            if best_g >= 0:
                gt_taken[best_g] = True
                global_d = dt_local_idx[d_local]
                dt_matched[global_d] = True
                dt_iou[global_d] = best_iou
                continue
            # Fall back to crowd GTs. Crowd matches do not consume the
            # GT; subsequent detections can match the same crowd region.
            best_iou = tp_threshold
            best_g = -1
            for g_local in range(len(gts_img)):
                if not is_crowd[g_local]:
                    continue
                v = float(iou[d_local, g_local])
                if v >= best_iou:
                    best_iou = v
                    best_g = g_local
            if best_g >= 0:
                global_d = dt_local_idx[d_local]
                dt_ignore[global_d] = True

    return dt_score, dt_matched, dt_ignore, dt_iou, n_pos_gt


# ---------------------------------------------------------------------------
# Per-class LRP scan + minimisation.
# ---------------------------------------------------------------------------


def _lrp_per_class(
    dt_score: np.ndarray,
    dt_matched: np.ndarray,
    dt_ignore: np.ndarray,
    dt_iou: np.ndarray,
    n_pos_gt: int,
    *,
    tp_threshold: float,
    tau_grid: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Find the optimal LRP and its three components for one class.

    Returns:
        Tuple ``(olrp, loc, fp_rate, fn_rate, tau_star)``.

        - ``olrp``: ``min_s LRP(s)``. Equal to ``1.0`` for an all-FN
          class. ``NaN`` if the class has no positive GTs (caller
          excludes from the headline mean).
        - ``loc, fp_rate, fn_rate``: components at ``s = tau_star``.
          ``NaN`` when ``NTP* == 0`` (loc / fp_rate denominators are
          zero); ``fn_rate`` reads ``NFN* / (NTP* + NFN*)`` so it is
          well-defined whenever there is at least one positive GT.
        - ``tau_star``: confidence threshold where the minimum is
          achieved. ``NaN`` if every grid value gives identical LRP and
          there are no active detections at any of them — concretely,
          the all-FN case (no detections; or no detection ever surfaces
          a TP at any tau).

    Tie-breaking: the maximisation runs over the full grid, and on equal
    LRP values the **larger** ``tau`` wins (``np.argmin`` of reversed
    arrays). Reasoning: the larger tau prunes more detections, which is
    the more conservative point on the operating curve.
    """
    if n_pos_gt == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    # Active set: detections whose score >= tau and which are not ignore.
    # We pre-mask ignores out: ignored detections never count as TP or FP.
    active_dt_score = dt_score[~dt_ignore]
    active_dt_matched = dt_matched[~dt_ignore]
    active_dt_iou = dt_iou[~dt_ignore]

    n_tau = len(tau_grid)
    lrp_vals = np.empty(n_tau, dtype=np.float64)
    n_tp_arr = np.empty(n_tau, dtype=np.int64)
    n_fp_arr = np.empty(n_tau, dtype=np.int64)
    n_fn_arr = np.empty(n_tau, dtype=np.int64)
    sum_loc_arr = np.empty(n_tau, dtype=np.float64)

    one_minus_tau_tp = 1.0 - float(tp_threshold)
    # Guard against tau_tp == 1.0 which would make the loc-error weight
    # undefined. The Rust caller will never pass 1.0; tests should pin
    # the default 0.5. Use 1.0 as a benign denominator (loc-error
    # collapses to a sum-of-zeros if any TP existed with IoU == 1.0).
    if one_minus_tau_tp <= 0.0:
        one_minus_tau_tp = 1.0

    for i, s in enumerate(tau_grid):
        active = active_dt_score >= float(s)
        n_tp = int(active_dt_matched[active].sum())
        n_fp = int(active.sum() - n_tp)
        # n_fn = positive GTs not matched by any active TP. Among the
        # masked detections, the matched ones correspond 1:1 to GTs
        # (greedy matching consumed one GT per matched detection), so
        # n_fn = n_pos_gt - n_tp.
        n_fn = n_pos_gt - n_tp
        sum_loc = float((1.0 - active_dt_iou[active & active_dt_matched]).sum())

        denom = n_tp + n_fp + n_fn
        if denom == 0:
            lrp_vals[i] = 0.0
        else:
            lrp_vals[i] = (sum_loc / one_minus_tau_tp + n_fp + n_fn) / denom
        n_tp_arr[i] = n_tp
        n_fp_arr[i] = n_fp
        n_fn_arr[i] = n_fn
        sum_loc_arr[i] = sum_loc

    # Argmin with "largest tau wins on ties". Equivalent to
    # ``len(tau_grid) - 1 - np.argmin(lrp_vals[::-1])``.
    rev_min_idx = int(np.argmin(lrp_vals[::-1]))
    star = n_tau - 1 - rev_min_idx

    olrp = float(lrp_vals[star])
    n_tp_star = int(n_tp_arr[star])
    n_fp_star = int(n_fp_arr[star])
    n_fn_star = int(n_fn_arr[star])
    sum_loc_star = float(sum_loc_arr[star])

    # Components at the optimal tau (paper eq. 10).
    if n_tp_star > 0:
        loc = (sum_loc_star / n_tp_star) / one_minus_tau_tp
        fp_rate = n_fp_star / (n_tp_star + n_fp_star) if (n_tp_star + n_fp_star) > 0 else 0.0
        fn_rate = n_fn_star / (n_tp_star + n_fn_star) if (n_tp_star + n_fn_star) > 0 else 0.0
        tau_star = float(tau_grid[star])
    else:
        # Degenerate: no TPs surfaced at any tau. ``loc`` and ``fp_rate``
        # are undefined; ``fn_rate`` is 1 (all positives missed). The
        # paper does not define ``tau_star`` in this case, so we return
        # NaN. Test fixtures pin this convention.
        loc = float("nan")
        fp_rate = float("nan") if n_fp_star > 0 else 0.0
        fn_rate = 1.0
        tau_star = float("nan")

    return olrp, loc, fp_rate, fn_rate, tau_star


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def optimal_lrp(
    gt: list[dict[str, Any]],
    dt: list[dict[str, Any]],
    *,
    similarity_fn: SimilarityFn = bbox_iou,
    tp_threshold: float = 0.5,
    tau_grid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute optimal LRP and its three additive components.

    Args:
        gt: list of COCO ground-truth annotation dicts. Each dict needs
            at least ``image_id``, ``category_id``, and ``bbox`` (or
            whatever the ``similarity_fn`` reads). ``iscrowd`` defaults
            to 0 if absent.
        dt: list of COCO detection dicts. Each dict needs ``image_id``,
            ``category_id``, ``score``, and ``bbox``.
        similarity_fn: ``(gts, dts) -> (D, G)`` IoU matrix. Defaults to
            ``bbox_iou``. Pass a custom callable for segm, boundary, or
            keypoint (OKS) kernels.
        tp_threshold: IoU floor for a (dt, gt) pair to count as a TP.
            Default ``0.5`` (paper convention).
        tau_grid: confidence-threshold grid scanned for the argmin.
            Default ``DEFAULT_TAU_GRID`` (101 points in ``[0, 1]``).

    Returns:
        Dict with keys:

        - ``olrp``: mean of ``olrp_per_class`` across classes that have
          at least one non-crowd GT. ``0.0`` if no such class exists.
        - ``loc``, ``fp``, ``fn``: means of the corresponding
          per-class components across classes with at least one TP at
          their optimal tau. Empty mean -> ``0.0``.
        - ``tau_per_class``: ``{class_id: tau_star}``. ``NaN`` if the
          class has no positive GTs *or* if the class has no TPs at
          any tau.
        - ``olrp_per_class``: ``{class_id: olrp}``. ``NaN`` if the class
          has no positive GTs; ``1.0`` for an all-FN class with TPs
          impossible at every tau (the worst-case ``LRP`` lower bound).
    """
    grid = DEFAULT_TAU_GRID if tau_grid is None else np.asarray(tau_grid, dtype=np.float64)

    # Group annotations by category_id. Classes with no GT but with
    # detections (pure FPs) still get an entry — their olrp_per_class is
    # 1.0 by paper convention (all FP, no positives).
    class_ids = sorted({int(g["category_id"]) for g in gt} | {int(d["category_id"]) for d in dt})

    olrp_per_class: dict[int, float] = {}
    tau_per_class: dict[int, float] = {}
    loc_per_class: dict[int, float] = {}
    fp_per_class: dict[int, float] = {}
    fn_per_class: dict[int, float] = {}

    for k in class_ids:
        gts_k = [g for g in gt if int(g["category_id"]) == k]
        dts_k = [d for d in dt if int(d["category_id"]) == k]

        if not gts_k or all(bool(g.get("iscrowd", 0)) for g in gts_k):
            # No positive GTs. The class is entirely FPs (if dts_k is
            # non-empty) or empty. Per paper, oLRP for a class with no
            # positives is conventionally undefined / excluded from the
            # mean. Mirror that by emitting NaN.
            olrp_per_class[k] = float("nan")
            tau_per_class[k] = float("nan")
            loc_per_class[k] = float("nan")
            fp_per_class[k] = float("nan")
            fn_per_class[k] = float("nan")
            continue

        dt_score, dt_matched, dt_ignore, dt_iou, n_pos_gt = _match_per_class(
            gts_k, dts_k, similarity_fn, tp_threshold
        )
        olrp_k, loc_k, fp_k, fn_k, tau_star_k = _lrp_per_class(
            dt_score,
            dt_matched,
            dt_ignore,
            dt_iou,
            n_pos_gt,
            tp_threshold=tp_threshold,
            tau_grid=grid,
        )
        olrp_per_class[k] = olrp_k
        tau_per_class[k] = tau_star_k
        loc_per_class[k] = loc_k
        fp_per_class[k] = fp_k
        fn_per_class[k] = fn_k

    def _nanmean(values: list[float]) -> float:
        finite = [v for v in values if not math.isnan(v)]
        if not finite:
            return 0.0
        return float(np.mean(finite))

    headline = _nanmean(list(olrp_per_class.values()))
    loc = _nanmean(list(loc_per_class.values()))
    fp = _nanmean(list(fp_per_class.values()))
    fn = _nanmean(list(fn_per_class.values()))

    return {
        "olrp": headline,
        "loc": loc,
        "fp": fp,
        "fn": fn,
        "tau_per_class": tau_per_class,
        "olrp_per_class": olrp_per_class,
    }
