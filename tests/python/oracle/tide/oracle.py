"""TIDE error-decomposition reference oracle (numpy).

Per ADR-0021, this module is the spec the Rust `vernier.error_decomposition`
implementation is validated against. It is pure numpy / Python with no
vernier imports. Correctness is pinned by hand-computed assertions on
small synthetic fixtures (see `test_oracle.py`).

Scope (Week 1):
    - bbox IoU only
    - mode="single": one ``t_f`` for bin assignment

Algorithm summary (canonical, per ADR-0021):

    For each detection (per image), at threshold ``t_f``:
      - If matched to a same-class GT: TP.
      - Else (FP):
          iou_same  = best IoU vs same-class same-image GTs
          iou_cross = best IoU vs different-class same-image GTs
          - Bkg  if max(iou_same, iou_cross) < t_b
          - Dupe if iou_same >= t_f  (but a higher-scoring DT already took it)
          - Cls  if iou_cross >= t_f  (and iou_same < t_f)
          - Both if t_b <= iou_cross < t_f  and iou_same < t_f
          - Loc  if t_b <= iou_same  < t_f

    For each unmatched non-ignore GT (after Cls/Loc/Both/Dupe attribution):
      - Missed.

    Per-bin delta:
      1. Rewrite the cells to "fix" only this bin's errors:
         - Cls    -> reattribute the DT to the wrong-class GT (relabel).
         - Loc    -> treat the DT as a TP (snap IoU to 1.0 with its best
                     same-class GT so it matches at every threshold).
         - Both   -> remove the DT.
         - Dupe   -> remove the DT.
         - Bkg    -> remove the DT.
         - Missed -> mark the unmatched GT as ignore.
      2. Re-run AP accumulation across the full 10-IoU-threshold ladder.
      3. delta = mAP_fixed - mAP_baseline  (positive when the fix helps).

Sanity bin: ``delta_all_fp_removed`` is the exact ΔmAP from removing
every FP detection (every detection not classified as TP at ``t_f``)
in one pass. The naive expectation that
``delta_all_fp_removed ≈ delta_cls + delta_loc + delta_both +
delta_dupe + delta_bkg`` only holds when fixing one bin does not
induce a different bin's degeneracy. The synthetic single-bin
fixtures in this directory deliberately violate that — fixing the
sole FP bin leaves zero detections and AP collapses — so the assertion
on ``delta_all_fp_removed`` for those fixtures pins the *exact* value
(not an inequality bound).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np

SimilarityFn: TypeAlias = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""Pairwise IoU kernel: ``(gts, dts) -> (D, G)`` IoU matrix."""

# Canonical 10-point COCO IoU ladder, built via numpy linspace to match
# pycocotools (and vernier-core) bit-for-bit.
IOU_THRESHOLDS: np.ndarray = np.linspace(0.5, 0.95, 10)
RECALL_THRESHOLDS: np.ndarray = np.linspace(0.0, 1.0, 101)

# Pycocotools' B1 quirk: a DT whose IoU equals the threshold exactly is
# a match. Implemented as a tiny shim on the threshold seed so the `<`
# comparison in the matching loop accepts the boundary value.
# See `crates/vernier-core/src/parity.rs::IOU_BOUNDARY_EPS`.
_IOU_BOUNDARY_EPS: float = 1e-10


# ---------------------------------------------------------------------------
# Similarity kernels (oracle-side, numpy).
# ---------------------------------------------------------------------------


def bbox_iou(gts: np.ndarray, dts: np.ndarray) -> np.ndarray:
    """Pairwise IoU between GT and DT bbox arrays in COCO ``[x, y, w, h]``.

    Args:
        gts: shape ``(G, 4)``.
        dts: shape ``(D, 4)``.

    Returns:
        shape ``(D, G)`` IoU matrix in row-major order
        (one row per detection, one column per GT) — the orientation
        the bin-assignment loop expects.
    """
    if gts.size == 0 or dts.size == 0:
        return np.zeros((dts.shape[0], gts.shape[0]), dtype=np.float64)
    gt_x1 = gts[:, 0]
    gt_y1 = gts[:, 1]
    gt_x2 = gts[:, 0] + gts[:, 2]
    gt_y2 = gts[:, 1] + gts[:, 3]
    gt_area = gts[:, 2] * gts[:, 3]
    dt_x1 = dts[:, 0]
    dt_y1 = dts[:, 1]
    dt_x2 = dts[:, 0] + dts[:, 2]
    dt_y2 = dts[:, 1] + dts[:, 3]
    dt_area = dts[:, 2] * dts[:, 3]
    # Broadcast (D, 1) vs (1, G).
    inter_x1 = np.maximum(dt_x1[:, None], gt_x1[None, :])
    inter_y1 = np.maximum(dt_y1[:, None], gt_y1[None, :])
    inter_x2 = np.minimum(dt_x2[:, None], gt_x2[None, :])
    inter_y2 = np.minimum(dt_y2[:, None], gt_y2[None, :])
    inter_w = np.clip(inter_x2 - inter_x1, 0.0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0.0, None)
    inter = inter_w * inter_h
    union = dt_area[:, None] + gt_area[None, :] - inter
    out = np.where(union > 0.0, inter / np.where(union == 0.0, 1.0, union), 0.0)
    return out.astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Internal cell representation.
# ---------------------------------------------------------------------------


@dataclass
class _GT:
    """Internal ground-truth annotation, normalised from the COCO dict."""

    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    iscrowd: bool  # COCO crowd flag → ignore in matching
    ignore: bool  # explicit `ignore` flag on the COCO annotation
    ann_id: int  # original COCO annotation id (for stable identity)


@dataclass
class _DT:
    """Internal detection, normalised from the COCO list dict."""

    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    score: float
    dt_idx: int  # stable per-call index across the input list


@dataclass
class _Image:
    """Per-image grouping of GTs and DTs."""

    image_id: int
    gts: list[_GT] = field(default_factory=list)
    dts: list[_DT] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AP / mAP — pure numpy reimplementation (independent of vernier-core).
# ---------------------------------------------------------------------------


def _stable_argsort_desc(scores: np.ndarray) -> np.ndarray:
    """Score-descending argsort with stable tie-breaking (mergesort).

    Mirrors the per-cell merged-stream sort vernier-core uses (quirk A1)
    so the oracle exposes the same tie behaviour as the reference for any
    fixtures that happen to have score ties.
    """
    return np.argsort(-scores, kind="mergesort")


def _ap_single_category_threshold(
    dt_records: list[tuple[float, bool, bool]],
    n_pos_gt: int,
) -> float:
    """101-point AP for one (category, IoU-threshold) cell.

    Args:
        dt_records: list of ``(score, matched, ignore)`` per detection,
            in the order they were emitted by matching. The function
            re-sorts by score descending (stable).
        n_pos_gt: number of non-ignore GTs in this category.

    Returns:
        AP in [0, 1], or -1.0 if the cell is empty (no DTs and no
        positive GTs — pycocotools' sentinel, filtered by the mean).
    """
    if n_pos_gt == 0:
        return -1.0
    if not dt_records:
        # Recall is 0; precision lane is 0 at every recall threshold.
        return 0.0

    scores = np.array([r[0] for r in dt_records], dtype=np.float64)
    matched = np.array([r[1] for r in dt_records], dtype=bool)
    ignore = np.array([r[2] for r in dt_records], dtype=bool)

    perm = _stable_argsort_desc(scores)
    matched = matched[perm]
    ignore = ignore[perm]

    tp = np.zeros(len(perm), dtype=np.float64)
    fp = np.zeros(len(perm), dtype=np.float64)
    cum_tp = 0.0
    cum_fp = 0.0
    for i in range(len(perm)):
        if ignore[i]:
            tp[i] = cum_tp
            fp[i] = cum_fp
            continue
        if matched[i]:
            cum_tp += 1.0
        else:
            cum_fp += 1.0
        tp[i] = cum_tp
        fp[i] = cum_fp

    recall = tp / float(n_pos_gt)
    # np.spacing(1) == f64::EPSILON, matches vernier-core's PARITY_EPS.
    precision = tp / (tp + fp + np.spacing(1))

    # Right-to-left running max (precision envelope).
    for j in range(len(precision) - 1, 0, -1):
        if precision[j] > precision[j - 1]:
            precision[j - 1] = precision[j]

    # 101-point recall sampling: searchsorted-left.
    inds = np.searchsorted(recall, RECALL_THRESHOLDS, side="left")
    sampled = np.zeros(len(RECALL_THRESHOLDS), dtype=np.float64)
    for ri, pi in enumerate(inds):
        if pi < len(precision):
            sampled[ri] = precision[pi]
        # else: leave sampled at 0.0 (past-the-curve, quirk C3).

    return float(sampled.mean())


def _greedy_match(
    iou: np.ndarray,
    dt_scores: np.ndarray,
    gt_ignore: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy DT→GT matching at a single IoU threshold.

    Mirrors `pycocotools.cocoeval.COCOeval.evaluateImg` for one image-and-
    category cell. DTs are walked in score-descending order; each takes
    the highest-IoU GT (above threshold) that is not yet taken. If the
    chosen GT is an ignore-GT, the DT is matched but flagged ignore so
    the AP accumulator skips it (quirks B6/C7).

    Returns:
        ``(dt_matched, dt_ignore, gt_matched_by)`` — first two are bool
        arrays of length ``D`` in the input order; ``gt_matched_by`` is
        an int array of length ``G`` carrying the matched DT's index in
        the input order, or ``-1`` if unmatched.
    """
    n_d = iou.shape[0]
    n_g = iou.shape[1]
    dt_matched = np.zeros(n_d, dtype=bool)
    dt_ignore = np.zeros(n_d, dtype=bool)
    gt_matched_by = -np.ones(n_g, dtype=np.int64)

    if n_d == 0:
        return dt_matched, dt_ignore, gt_matched_by

    perm = _stable_argsort_desc(dt_scores)
    # Pycocotools matching prefers non-ignore GTs over ignore-GTs:
    # the search picks the best-IoU non-ignore GT first; if none is
    # above threshold, falls back to the best-IoU ignore GT and flags
    # the DT as ignore. This matches the canonical eval semantics
    # (see cocoeval.py:271-298).
    for d in perm:
        # Best non-ignore GT first.
        best_iou = threshold - _IOU_BOUNDARY_EPS
        best_g = -1
        for g in range(n_g):
            if gt_ignore[g]:
                continue
            if gt_matched_by[g] >= 0:
                continue
            if iou[d, g] < best_iou:
                continue
            best_iou = iou[d, g]
            best_g = g
        if best_g >= 0:
            dt_matched[d] = True
            gt_matched_by[best_g] = int(d)
            continue
        # Fall back to ignore GTs.
        best_iou = threshold - _IOU_BOUNDARY_EPS
        best_g = -1
        for g in range(n_g):
            if not gt_ignore[g]:
                continue
            if iou[d, g] < best_iou:
                continue
            best_iou = iou[d, g]
            best_g = g
        if best_g >= 0:
            dt_matched[d] = True
            dt_ignore[d] = True
            # Ignore GTs can be matched by multiple DTs in pycocotools
            # (they don't get "consumed"). For oracle accounting we
            # still record the last DT that matched, but we don't mark
            # it as taken — let subsequent DTs match too.
            # (gt_matched_by stays -1 to allow further matches.)

    return dt_matched, dt_ignore, gt_matched_by


def _compute_map(
    images: list[_Image],
    category_ids: list[int],
    similarity_fn: SimilarityFn,
    *,
    use_cats: bool,
    max_dets_per_image: int,
) -> float:
    """Compute mAP across the canonical 10-IoU-threshold ladder.

    Per-category, per-threshold AP is computed via `_ap_single_category_threshold`.
    The mean is taken across the (T, K) grid for cells that have any
    non-ignore GT (cells with ``n_pos_gt == 0`` return -1 and are
    filtered, matching pycocotools' `s[s>-1]` semantics).
    """
    cats = list(category_ids) if use_cats else [-1]

    ap_values: list[float] = []
    for t_iou in IOU_THRESHOLDS:
        for k in cats:
            dt_records: list[tuple[float, bool, bool]] = []
            n_pos_gt = 0
            for img in images:
                if use_cats:
                    gts = [g for g in img.gts if g.category_id == k]
                    dts_unsorted = [d for d in img.dts if d.category_id == k]
                else:
                    gts = list(img.gts)
                    dts_unsorted = list(img.dts)
                # max-dets cap (per image, score-desc).
                dts_sorted = sorted(dts_unsorted, key=lambda d: -d.score)[:max_dets_per_image]
                gt_ignore = np.array([g.iscrowd or g.ignore for g in gts], dtype=bool)
                n_pos_gt += int((~gt_ignore).sum()) if gt_ignore.size else 0
                if not dts_sorted:
                    continue
                gt_boxes = np.array([g.bbox for g in gts], dtype=np.float64).reshape(-1, 4)
                dt_boxes = np.array([d.bbox for d in dts_sorted], dtype=np.float64).reshape(-1, 4)
                dt_scores = np.array([d.score for d in dts_sorted], dtype=np.float64)
                iou = similarity_fn(gt_boxes, dt_boxes)
                dt_matched, dt_ignore, _ = _greedy_match(iou, dt_scores, gt_ignore, float(t_iou))
                for d_idx, d in enumerate(dts_sorted):
                    dt_records.append((d.score, bool(dt_matched[d_idx]), bool(dt_ignore[d_idx])))
            ap = _ap_single_category_threshold(dt_records, n_pos_gt)
            if ap >= 0.0:
                ap_values.append(ap)
    if not ap_values:
        return 0.0
    return float(np.mean(ap_values))


# ---------------------------------------------------------------------------
# Bin attribution.
# ---------------------------------------------------------------------------


@dataclass
class _BinAttribution:
    """Per-detection bin label at threshold ``t_f``.

    Each FP detection is labelled with exactly one of the five FP bins;
    TP detections are labelled "tp"; ignore-matched DTs are "ignore"
    (they don't count as FPs and aren't binned).

    The two ``*_local_idx`` fields point at the GT (within the same
    image as the DT) whose category/bbox the fix path needs:

    - ``cross_gt_local_idx`` (Cls only): wrong-class GT to relabel to.
    - ``same_gt_local_idx`` (Loc only): same-class GT to snap the bbox
      onto so IoU becomes 1.0.
    """

    bin: str  # one of: tp, ignore, cls, loc, both, dupe, bkg
    cross_gt_local_idx: int = -1
    same_gt_local_idx: int = -1


def _attribute_bins(
    images: list[_Image],
    similarity_fn: SimilarityFn,
    *,
    t_f: float,
    t_b: float,
    max_dets_per_image: int,
    use_cats: bool,
) -> tuple[dict[int, _BinAttribution], set[tuple[int, int]]]:
    """Assign each detection to a bin and find unmatched GTs (Missed).

    Returns:
        - mapping ``dt_idx -> _BinAttribution`` covering every detection
          that survives the per-image max-dets cap.
        - set of ``(image_id, gt_ann_id)`` for unmatched non-ignore GTs.
    """
    attribution: dict[int, _BinAttribution] = {}
    missed: set[tuple[int, int]] = set()

    for img in images:
        # Apply per-image max-dets cap on DTs (score-desc), as the AP
        # accumulator does. DTs evicted by the cap are NOT binned —
        # they're invisible to AP and to TIDE.
        dts_sorted = sorted(img.dts, key=lambda d: -d.score)[:max_dets_per_image]

        # Per-class same-class greedy match at t_f. Track which GTs
        # were taken (per-class) for Dupe attribution.
        gt_ignore_all = np.array([g.iscrowd or g.ignore for g in img.gts], dtype=bool)
        # gt_taken_by[g_local] = dt_idx that matched GT g_local at t_f.
        # Absence means unmatched (used for Missed attribution).
        gt_taken_by: dict[int, int] = {}
        # Track per-DT: was matched at t_f.
        per_dt_matched_at_tf: dict[int, bool] = {}
        per_dt_ignore_at_tf: dict[int, bool] = {}

        if use_cats:
            cats_in_image = {g.category_id for g in img.gts} | {d.category_id for d in dts_sorted}
        else:
            cats_in_image = {-1}

        for k in cats_in_image:
            if use_cats:
                gt_local_idx = [i for i, g in enumerate(img.gts) if g.category_id == k]
                dt_subset = [d for d in dts_sorted if d.category_id == k]
            else:
                gt_local_idx = list(range(len(img.gts)))
                dt_subset = list(dts_sorted)

            if not dt_subset:
                continue
            gts_k = [img.gts[i] for i in gt_local_idx]
            gt_ignore_k = np.array([g.iscrowd or g.ignore for g in gts_k], dtype=bool)
            gt_boxes = np.array([g.bbox for g in gts_k], dtype=np.float64).reshape(-1, 4)
            dt_boxes = np.array([d.bbox for d in dt_subset], dtype=np.float64).reshape(-1, 4)
            dt_scores = np.array([d.score for d in dt_subset], dtype=np.float64)
            iou_same = similarity_fn(gt_boxes, dt_boxes)
            dt_matched, dt_ignore, gt_matched_by_local = _greedy_match(
                iou_same, dt_scores, gt_ignore_k, t_f
            )
            for d_local, d in enumerate(dt_subset):
                per_dt_matched_at_tf[d.dt_idx] = bool(dt_matched[d_local])
                per_dt_ignore_at_tf[d.dt_idx] = bool(dt_ignore[d_local])
            for g_local, dt_idx_local in enumerate(gt_matched_by_local):
                if dt_idx_local >= 0:
                    g_global_local = gt_local_idx[g_local]
                    gt_taken_by[g_global_local] = dt_subset[dt_idx_local].dt_idx

        # Compute per-DT iou_same and iou_cross for FP bin attribution.
        # iou_same = best IoU vs same-class same-image GTs (ignore-aware:
        #   treat ignore-GTs as available — TIDE attribution treats Loc
        #   based on geometry, not on whether the GT was an ignore).
        #   But for Dupe, we use the matching above which already
        #   accounted for ignore.
        # iou_cross = best IoU vs different-class same-image GTs.
        # For use_cats=False, iou_cross is always 0 (no other classes).
        gt_boxes_all = (
            np.array([g.bbox for g in img.gts], dtype=np.float64).reshape(-1, 4)
            if img.gts
            else np.zeros((0, 4))
        )
        for d in dts_sorted:
            if per_dt_ignore_at_tf.get(d.dt_idx, False):
                attribution[d.dt_idx] = _BinAttribution(bin="ignore")
                continue
            if per_dt_matched_at_tf.get(d.dt_idx, False):
                attribution[d.dt_idx] = _BinAttribution(bin="tp")
                continue
            # FP: compute iou_same / iou_cross.
            dt_box = np.array([d.bbox], dtype=np.float64)
            ious = (
                similarity_fn(gt_boxes_all, dt_box)[0]  # (G,)
                if img.gts
                else np.zeros(0)
            )
            iou_same = 0.0
            best_same_local = -1
            iou_cross = 0.0
            best_cross_local = -1
            for g_local, g in enumerate(img.gts):
                if use_cats and g.category_id == d.category_id:
                    if ious[g_local] > iou_same:
                        iou_same = float(ious[g_local])
                        best_same_local = g_local
                elif use_cats and g.category_id != d.category_id:
                    if ious[g_local] > iou_cross:
                        iou_cross = float(ious[g_local])
                        best_cross_local = g_local
                elif not use_cats and ious[g_local] > iou_same:
                    iou_same = float(ious[g_local])
                    best_same_local = g_local

            # Bin priority follows Bolya 2020 §3: the max-IoU GT's class
            # determines whether an "almost matched" detection is Loc
            # (closest GT is same-class) or Both (closest GT is wrong-
            # class). Concretely: if iou_same >= iou_cross, the DT was
            # closer to a same-class GT and should report as Loc. Ties
            # go to Loc (same-class wins) — consistent with the
            # paper's view that Loc is the "less compound" error.
            if iou_same >= t_f:
                # Dupe — same-class GT exists at >= t_f, but a higher-
                # scoring DT must have taken it (otherwise this DT
                # would be matched).
                attribution[d.dt_idx] = _BinAttribution(
                    bin="dupe",
                    same_gt_local_idx=best_same_local,
                )
                continue
            if iou_cross >= t_f:
                attribution[d.dt_idx] = _BinAttribution(
                    bin="cls",
                    cross_gt_local_idx=best_cross_local,
                )
                continue
            if iou_same >= t_b and iou_same >= iou_cross:
                attribution[d.dt_idx] = _BinAttribution(
                    bin="loc",
                    same_gt_local_idx=best_same_local,
                )
                continue
            if iou_cross >= t_b:
                attribution[d.dt_idx] = _BinAttribution(
                    bin="both",
                    cross_gt_local_idx=best_cross_local,
                )
                continue
            # Otherwise Bkg.
            attribution[d.dt_idx] = _BinAttribution(bin="bkg")

        # Missed: unmatched non-ignore GTs (after Cls/Loc/Both/Dupe
        # attribution). For now we use the same-class greedy match's
        # gt_matched_by to determine matched GTs. Note that Cls
        # attribution does NOT mark the cross-class GT as matched —
        # the cross-class GT in question may itself be unmatched and
        # therefore "missed", but a Missed GT is one with no DT of
        # its own class matching it, which is the same-class match
        # we already performed. So the "Missed" set is exactly the
        # set of non-ignore GTs unmatched by their own class.
        for g_local, g in enumerate(img.gts):
            if gt_ignore_all[g_local]:
                continue
            if g_local in gt_taken_by:
                continue
            missed.add((img.image_id, g.ann_id))

    return attribution, missed


# ---------------------------------------------------------------------------
# Bin-fix application.
# ---------------------------------------------------------------------------


def _apply_fix(
    images: list[_Image],
    attribution: dict[int, _BinAttribution],
    missed: set[tuple[int, int]],
    bin_to_fix: str,
) -> list[_Image]:
    """Return a deep-ish copy of ``images`` with one bin's errors fixed.

    Fixes per bin (per the algorithm in this module's docstring):
      - cls   : relabel each Cls DT to the wrong-class GT's category.
      - loc   : add a synthetic "perfect-IoU" annotation alongside the
                Loc DT so it matches at every IoU threshold. Implementation:
                we simply replace the DT's bbox with the matched same-class
                GT's bbox (IoU = 1.0 with that GT).
      - both, dupe, bkg : remove the DT.
      - missed: mark the Missed GTs as ignore (iscrowd-equivalent).
      - all_fp: remove every FP detection (any non-tp non-ignore bin).
    """
    out: list[_Image] = []
    for img in images:
        new_gts: list[_GT] = []
        for g in img.gts:
            if bin_to_fix == "missed" and (img.image_id, g.ann_id) in missed:
                new_gts.append(
                    _GT(
                        image_id=g.image_id,
                        category_id=g.category_id,
                        bbox=g.bbox,
                        iscrowd=g.iscrowd,
                        ignore=True,
                        ann_id=g.ann_id,
                    )
                )
            else:
                new_gts.append(g)
        new_dts: list[_DT] = []
        for d in img.dts:
            attr = attribution.get(d.dt_idx)
            if attr is None:
                # DT was evicted by max-dets cap; pass through unchanged.
                new_dts.append(d)
                continue
            if bin_to_fix == "all_fp":
                if attr.bin in ("tp", "ignore"):
                    new_dts.append(d)
                # else: drop (any FP bin).
                continue
            if attr.bin != bin_to_fix:
                new_dts.append(d)
                continue
            # bin matches the fix.
            if bin_to_fix == "cls":
                # Relabel to the wrong-class GT's category.
                tgt_g = img.gts[attr.cross_gt_local_idx]
                new_dts.append(
                    _DT(
                        image_id=d.image_id,
                        category_id=tgt_g.category_id,
                        bbox=d.bbox,
                        score=d.score,
                        dt_idx=d.dt_idx,
                    )
                )
            elif bin_to_fix == "loc":
                # Snap DT bbox to the same-class GT bbox. IoU becomes 1.0
                # at every threshold — the DT now matches.
                tgt_g = img.gts[attr.same_gt_local_idx]
                new_dts.append(
                    _DT(
                        image_id=d.image_id,
                        category_id=d.category_id,
                        bbox=tgt_g.bbox,
                        score=d.score,
                        dt_idx=d.dt_idx,
                    )
                )
            else:
                # bin_to_fix in ("both", "dupe", "bkg") → drop the DT.
                # Caller is internal; no other bin names reach this branch.
                pass
        out.append(_Image(image_id=img.image_id, gts=new_gts, dts=new_dts))
    return out


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def _normalise(gt: dict[str, Any], dt: list[dict[str, Any]]) -> tuple[list[_Image], list[int]]:
    """Convert COCO-format inputs into per-image grouped internal records."""
    cat_ids = sorted({c["id"] for c in gt.get("categories", [])})
    by_image: dict[int, _Image] = {
        img["id"]: _Image(image_id=img["id"]) for img in gt.get("images", [])
    }
    for ann in gt.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in by_image:
            continue
        bbox = tuple(ann["bbox"])  # type: ignore[assignment]
        if len(bbox) != 4:
            raise ValueError(f"GT ann {ann.get('id')} has bbox of len {len(bbox)} != 4")
        by_image[img_id].gts.append(
            _GT(
                image_id=img_id,
                category_id=ann["category_id"],
                bbox=bbox,  # type: ignore[arg-type]
                iscrowd=bool(ann.get("iscrowd", 0)),
                ignore=bool(ann.get("ignore", 0)),
                ann_id=ann.get("id", -1),
            )
        )
    for d_idx, det in enumerate(dt):
        img_id = det["image_id"]
        if img_id not in by_image:
            continue
        bbox = tuple(det["bbox"])  # type: ignore[assignment]
        if len(bbox) != 4:
            raise ValueError(f"DT #{d_idx} has bbox of len {len(bbox)} != 4")
        by_image[img_id].dts.append(
            _DT(
                image_id=img_id,
                category_id=det["category_id"],
                bbox=bbox,  # type: ignore[arg-type]
                score=float(det["score"]),
                dt_idx=d_idx,
            )
        )
    images = [by_image[img_id] for img_id in sorted(by_image.keys())]
    return images, cat_ids


def error_decomposition(
    gt: dict[str, Any],
    dt: list[dict[str, Any]],
    similarity_fn: SimilarityFn = bbox_iou,
    *,
    t_f: float = 0.5,
    t_b: float = 0.1,
    max_dets_per_image: int = 100,
    use_cats: bool = True,
) -> dict[str, Any]:
    """TIDE error decomposition (single-mode, bbox kernel only).

    Args:
        gt: COCO-format ground truth dict with keys ``images``,
            ``annotations``, ``categories``.
        dt: COCO-format detection list (each entry has ``image_id``,
            ``category_id``, ``bbox`` in xywh, ``score``).
        similarity_fn: pairwise IoU kernel ``(gts, dts) -> (D, G)`` IoU
            matrix. Defaults to `bbox_iou`. The Week-3 segm/boundary
            lift will accept the corresponding mask kernels here.
        t_f: foreground / match threshold (≥ ⇒ TP).
        t_b: background threshold (< ⇒ Bkg).
        max_dets_per_image: per-image DT cap (score-desc), matching
            pycocotools' max-dets semantics.
        use_cats: if False, all categories are merged into one bucket
            (mirrors pycocotools' ``useCats=False``); cross-class
            attribution is degenerate (no other classes exist) and the
            Cls / Both bins are always empty.

    Returns:
        ``{"baseline_map": ..., "delta": {...}, "delta_all_fp_removed": ...,
           "config": {"t_f": ..., "t_b": ..., "kernel": "bbox"}}``.
    """
    images, cat_ids = _normalise(gt, dt)
    baseline = _compute_map(
        images,
        cat_ids,
        similarity_fn,
        use_cats=use_cats,
        max_dets_per_image=max_dets_per_image,
    )
    attribution, missed = _attribute_bins(
        images,
        similarity_fn,
        t_f=t_f,
        t_b=t_b,
        max_dets_per_image=max_dets_per_image,
        use_cats=use_cats,
    )

    deltas: dict[str, float] = {}
    for bin_name in ("cls", "loc", "both", "dupe", "bkg", "missed"):
        fixed_images = _apply_fix(images, attribution, missed, bin_name)
        fixed_map = _compute_map(
            fixed_images,
            cat_ids,
            similarity_fn,
            use_cats=use_cats,
            max_dets_per_image=max_dets_per_image,
        )
        deltas[bin_name] = fixed_map - baseline

    fixed_all_fp = _apply_fix(images, attribution, missed, "all_fp")
    fixed_all_fp_map = _compute_map(
        fixed_all_fp,
        cat_ids,
        similarity_fn,
        use_cats=use_cats,
        max_dets_per_image=max_dets_per_image,
    )

    return {
        "baseline_map": baseline,
        "delta": deltas,
        "delta_all_fp_removed": fixed_all_fp_map - baseline,
        "config": {"t_f": t_f, "t_b": t_b, "kernel": "bbox"},
    }
