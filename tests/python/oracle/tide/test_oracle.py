"""Pin the TIDE numpy oracle against hand-computed fixture values.

Each fixture isolates one bin so reviewers can read the test and see
"this fixture exists to verify <bin> binning works." The hand-computed
math is documented inline alongside each assertion block so the values
can be re-derived without running the oracle.

Per ADR-0021 these assertions ARE the spec for vernier's TIDE
implementation. The Rust implementation (Week 2) will be validated
against the oracle's outputs on the same fixtures within `1e-9`
ΔmAP per bin.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import pytest

from .oracle import bbox_iou, boundary_iou, error_decomposition, segm_iou

FIXTURES = Path(__file__).parent / "fixtures"
EXPECTED = Path(__file__).parent / "expected"

# Per ADR-0022 the boundary `t_b` default is `0.05` at
# `dilation_ratio=0.02` (a tentative default with the empirical
# ratification deferred to a 0.5.x follow-up; see ADR-0022's "Decision
# gate (boundary default)" section). The boundary fixtures are pinned
# at this value so the row in the ADR and the assertion in this file
# move together.
_BOUNDARY_T_B = 0.05
_BOUNDARY_DILATION_RATIO = 0.02
# Every boundary fixture below uses a 200x200 canvas: the diag-derived
# erosion radius is ``round(0.02 * sqrt(80000)) = 6``, which is large
# enough that the band is a non-trivial 6-pixel frame on a 50x50 mask
# (band area = 50^2 - 38^2 = 1056) — comfortably past the d=1 clamp.
_BOUNDARY_IMAGE_HW = (200, 200)


def _boundary_sim() -> Any:
    """Build the boundary similarity_fn baked at the fixture canvas."""
    return functools.partial(
        boundary_iou,
        dilation_ratio=_BOUNDARY_DILATION_RATIO,
        image_hw=_BOUNDARY_IMAGE_HW,
    )


# Hand-computed assertions are authored to ~10 decimal places of float
# precision. The 1e-6 tolerance is the ADR-0021 oracle-stability budget
# (Rust-vs-oracle parity has its own tighter `1e-9` budget — that's a
# different test, deferred to Week 2).
TOL = 1e-6


@functools.cache
def _load(name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fix_dir = FIXTURES / name
    gt = json.loads((fix_dir / "gt.json").read_text())
    dt = json.loads((fix_dir / "dt.json").read_text())
    return gt, dt


@functools.cache
def _load_expected(name: str) -> dict[str, Any]:
    """Load the per-fixture expected report from `expected/<name>.json`.

    The same JSON is consumed by ``crates/vernier-core/tests/tide_oracle_parity.rs``
    — Rust and Python read identical bytes, which is what keeps the two
    implementations from drifting. The *why* behind each pinned value
    lives in the test docstrings below; re-derive from there rather
    than from the JSON.
    """
    return json.loads((EXPECTED / f"{name}.json").read_text())


def _assert_close(actual: float, expected: float, label: str) -> None:
    assert abs(actual - expected) < TOL, (
        f"{label}: expected {expected!r}, got {actual!r}, "
        f"diff {actual - expected!r} exceeds tolerance {TOL!r}"
    )


def _assert_report_matches(out: dict[str, Any], expected: dict[str, Any]) -> None:
    """Assert ``out`` matches every pinned value in ``expected``: baseline,
    every per-bin Δ, and the all-FPs-removed sanity delta. Bins absent
    from ``expected['deltas']`` default to zero, so the JSON only needs
    to enumerate the bins a fixture actually pins.
    """
    _assert_close(out["baseline_map"], expected["baseline_map"], "baseline_map")
    for bin_name in ("cls", "loc", "both", "dupe", "bkg", "missed"):
        want = expected["deltas"].get(bin_name, 0.0)
        _assert_close(out["delta"][bin_name], want, f"delta[{bin_name}]")
    _assert_close(
        out["delta_all_fp_removed"],
        expected["delta_all_fp_removed"],
        "delta_all_fp_removed",
    )


def test_all_perfect_baseline_one_and_no_deltas() -> None:
    """All detections are correct; nothing for TIDE to attribute.

    Math:
        - Two GTs (cat 1 at [10,10,40,40], cat 2 at [100,100,40,40]).
        - Two DTs identical to each GT, score 0.9. Each matches IoU=1.0.
        - Per (cat, IoU-threshold): 1 TP, 0 FP. Recall reaches 1 at the
          first DT. The 101-point precision lane is uniformly 1.0 →
          AP = 1.0. Mean across 10 thresholds x 2 cats = 1.0.
        - Every bin is empty. All deltas = 0.
    """
    gt, dt = _load("all_perfect")
    out = error_decomposition(gt, dt)
    _assert_report_matches(out, _load_expected("all_perfect"))


def test_all_bkg_isolates_bkg_bin() -> None:
    """Every FP detection is pure background; only delta_bkg is non-zero.

    Math (per category, identical for cat 1 and cat 2):
        - 1 GT (1 positive). DTs sorted by score:
            DT_bkg (s=0.9, FP, IoU=0 with everything),
            DT_tp  (s=0.5, TP, IoU=1.0 with the cat's GT).
        - Cumulative TP/FP after each DT:
            after DT_bkg: tp=0, fp=1 → recall=0,    precision=0
            after DT_tp:  tp=1, fp=1 → recall=1.0,  precision=0.5
        - Right-to-left envelope: precision = [0.5, 0.5].
        - 101-point lane (recall = [0, 1]):
            target=0     → searchsorted-left = 0 → pr[0] = 0.5
            target=0.01..1.0 → searchsorted-left = 1 → pr[1] = 0.5
          All 101 samples = 0.5. AP = 0.5.
        - Mean over 2 cats x 10 thresholds = 0.5. baseline_map = 0.5.

    After Bkg fix (removes both bkg-FP DTs):
        - Per category: 1 DT (TP at score 0.5). Recall=1, precision=1.
          AP = 1.0. mAP = 1.0. delta_bkg = 0.5.

    Other bins are empty. delta_all_fp_removed equals delta_bkg
    (the only FPs were Bkg).
    """
    gt, dt = _load("all_bkg")
    out = error_decomposition(gt, dt)
    _assert_report_matches(out, _load_expected("all_bkg"))


def test_all_cls_isolates_cls_bin() -> None:
    """Every FP detection has the wrong class but right geometry.

    Math:
        - 1 image, 2 GTs: GT1=cat 1 at [10,10,40,40]; GT2=cat 2 at
          [100,100,40,40].
        - 2 DTs:
            DT1: cat 2 at [10,10,40,40], score 0.9
                 → IoU=1.0 with GT1 (wrong class), IoU=0 with GT2.
            DT2: cat 1 at [100,100,40,40], score 0.9
                 → IoU=1.0 with GT2 (wrong class), IoU=0 with GT1.
          Each DT: iou_same=0, iou_cross=1.0 ≥ t_f → Cls.
        - Baseline:
            cat 1: GT1 + DT2. DT2 IoU=0 with GT1 → unmatched.
                   tp=0, fp=1, recall=0, AP = 0.
            cat 2: GT2 + DT1. Symmetric. AP = 0.
            mAP = 0. baseline_map = 0.
        - Cls fix relabels DT1→cat 1, DT2→cat 2.
            cat 1: 2 anns (GT1) + DT1@[10,10,40,40] cat 1 (TP), no DT2.
                   AP = 1.0.
            cat 2: similar. AP = 1.0.
            mAP = 1.0. delta_cls = 1.0.
        - All other bins empty. delta_all_fp_removed = 0 (removing the
          two FPs leaves 0 detections, AP = 0 — sanity bound is loose
          when correcting Cls also fixes the corresponding Missed).
    """
    gt, dt = _load("all_cls")
    out = error_decomposition(gt, dt)
    _assert_report_matches(out, _load_expected("all_cls"))


def test_all_loc_isolates_loc_bin() -> None:
    """Every FP detection has the right class but IoU in [t_b, t_f).

    Math:
        - 1 GT cat 1 at [0,0,100,100] (area 10000).
        - 1 DT cat 1 at [0,0,80,50] (area 4000), score 0.9.
            intersection = min(80,100) * min(50,100) = 80 * 50 = 4000
            union        = 10000 + 4000 - 4000 = 10000
            IoU = 4000 / 10000 = 0.4
          0.1 ≤ 0.4 < 0.5 → Loc.
        - Baseline: at every IoU threshold ≥ 0.5, IoU=0.4 < threshold.
          Unmatched. tp=0, fp=1, AP = 0. mAP = 0.
        - Loc fix snaps the DT bbox to GT bbox → IoU=1.0 → TP at every
          threshold. AP = 1.0 across the ladder. mAP = 1.0.
          delta_loc = 1.0.
        - Other bins empty.
        - delta_all_fp_removed: removes the lone DT → no detections
          → AP = 0 → mAP = 0 → delta = 0. Sanity bound is loose for the
          same reason as in all_cls.
    """
    gt, dt = _load("all_loc")
    out = error_decomposition(gt, dt)
    _assert_report_matches(out, _load_expected("all_loc"))


def test_all_dupe_isolates_dupe_bin() -> None:
    """Multiple high-scoring same-class DTs land on each GT.

    Math:
        - 2 GTs cat 1: GT1 at [10,10,40,40], GT2 at [100,100,40,40].
        - 4 DTs cat 1, sorted by score:
            DT1: [10,10,40,40], 0.9 → matches GT1 (TP).
            DT2: [10,10,40,40], 0.8 → IoU=1 with GT1 (taken)
                                    → unmatched at t_f → Dupe.
            DT3: [10,10,40,40], 0.7 → same → Dupe.
            DT4: [100,100,40,40], 0.6 → matches GT2 (TP).
        - n_pos_gt = 2. Cumulative after each DT:
            DT1: tp=1, fp=0, recall=0.5, precision=1.0
            DT2: tp=1, fp=1, recall=0.5, precision=0.5
            DT3: tp=1, fp=2, recall=0.5, precision≈0.333
            DT4: tp=2, fp=2, recall=1.0, precision=0.5
        - Right-to-left envelope:
            pr=[1.0, 0.5, 0.333, 0.5] →
            pr[3]=0.5; pr[2]=max(0.333, 0.5)=0.5; pr[1]=0.5; pr[0]=1.0
            envelope = [1.0, 0.5, 0.5, 0.5]
        - 101-point sampling on recall=[0.5, 0.5, 0.5, 1.0]:
            targets in [0, 0.50] (51 samples) → searchsorted-left = 0 →
                pr[0] = 1.0
            targets in (0.50, 1.00] (50 samples) → searchsorted-left = 3 →
                pr[3] = 0.5
            AP = (51*1.0 + 50*0.5) / 101 = 76 / 101 ≈ 0.7524752475...
            mAP = 76/101 (one cat, all 10 thresholds identical).
        - After Dupe fix (remove DT2, DT3): two TPs, no FPs. recall=[0.5, 1].
          Precision lane uniformly 1.0. AP = 1.0.
          delta_dupe = 1.0 - 76/101 = 25/101 ≈ 0.2475247525...
        - Other bins empty. delta_all_fp_removed equals delta_dupe.
    """
    gt, dt = _load("all_dupe")
    out = error_decomposition(gt, dt)
    _assert_report_matches(out, _load_expected("all_dupe"))


def test_with_ignore_does_not_bin_crowd_matched_dts() -> None:
    """Crowd GTs are ignore; DTs matching them must not be FP-binned.

    Math:
        - Image 1: GT1 cat 1 at [10,10,40,40], iscrowd=1 (ignore).
        - Image 2: GT2 cat 1 at [10,10,40,40], iscrowd=0.
        - 3 DTs (all cat 1):
            DT1 (image 1, [10,10,40,40], 0.6): IoU=1 with the crowd GT1
                → matched-to-ignore in greedy match → dt_ignore=True
                → attribution "ignore" (not binned).
            DT2 (image 1, [500,500,30,30], 0.9): IoU=0 with everything
                → Bkg.
            DT3 (image 2, [10,10,40,40], 0.4): IoU=1 with GT2 → TP.
        - n_pos_gt = 1 (only GT2 is non-ignore). Sorted by score:
            DT2 (0.9, FP), DT1 (0.6, ignore), DT3 (0.4, TP).
            DT2: tp=0, fp=1, recall=0, precision≈0
            DT1: ignore → cum unchanged. recall=0, precision≈0.
            DT3: tp=1, fp=1, recall=1.0, precision=0.5
        - Envelope right-to-left: pr=[0, 0, 0.5] →
            pr[2]=0.5; pr[1]=0.5; pr[0]=0.5 → envelope=[0.5, 0.5, 0.5].
        - 101-point sampling on recall=[0, 0, 1]:
            target=0 → idx=0 (rc[0]=0 ≥ 0) → pr[0]=0.5.
            target=0.01..1.0 → idx=2 → pr[2]=0.5.
            AP = 0.5. mAP = 0.5.
        - Bkg fix (remove DT2): two DTs left.
            DT1 (ignore, then DT3 (TP). recall=[0, 1] (DT1 cum stays at 0).
            Envelope makes precision=[1.0, 1.0]. AP = 1.0. mAP = 1.0.
            delta_bkg = 0.5.
        - GT1 is crowd → not in Missed set. GT2 is matched. So no Missed.
        - DT1 is ignore-bin (not Cls/Loc/Both/Dupe/Bkg). Verifies the
          ignore semantics: DTs matched to crowd GTs are not binned.
        - delta_all_fp_removed equals delta_bkg (DT2 is the lone FP).
    """
    gt, dt = _load("with_ignore")
    out = error_decomposition(gt, dt)
    _assert_report_matches(out, _load_expected("with_ignore"))


def test_loc_vs_both_priority_loc_wins_when_same_class_gt_is_closer() -> None:
    """Same- and cross-class GTs both in [t_b, t_f); same-class wins → Loc.

    Per Bolya 2020 §3, the max-IoU GT's class determines the bin: if
    the closest GT is same-class the error is Loc, if it's wrong-class
    the error is Both. This fixture pins that priority — without the
    rule, a `t_b ≤ iou_cross < t_f` check would unconditionally bin
    as Both even when there's a closer same-class GT.

    Math:
        - 1 image, 2 GTs:
            GT1 cat 1 at [0, 0, 100, 100], area 10000.
            GT2 cat 2 at [10, 10, 50, 50], area 2500.
        - 1 DT cat 1 at [30, 30, 50, 50], score 0.9 — DT spans
          (30, 30) → (80, 80).
            iou_same vs GT1 (cat 1, spans (0,0)→(100,100)):
              inter = 50*50 = 2500. union = 10000+2500-2500 = 10000.
              IoU = 0.25.
            iou_cross vs GT2 (cat 2, spans (10,10)→(60,60)):
              inter x = max(30,10)..min(80,60) = 30 wide.
              inter y = same. inter = 900.
              union = 2500+2500-900 = 4100. IoU ≈ 0.21951.
          Both in [0.1, 0.5); iou_same > iou_cross → Loc.
        - Baseline: DT IoU 0.25 < 0.5, unmatched at every threshold.
            cat 1: tp=0, fp=1, AP = 0.
            cat 2: 0 DTs, 1 positive GT, AP = 0.
            mAP = 0.
        - Loc fix snaps DT bbox to GT1 → IoU=1.0 → TP everywhere.
            cat 1: 1 TP, AP = 1. cat 2: unchanged, AP = 0.
            mAP = 0.5. delta_loc = 0.5.
        - delta_both = 0 (no Both-binned DTs). With the priority bug,
          this DT would have been binned as Both, yielding
          delta_loc=0 and delta_both=0 — the delta_loc=0.5 assertion
          is what discriminates the fix.
        - delta_missed = 0 (marking GT1 and GT2 as ignore zeroes
          every cell; n_pos_gt=0 returns -1 sentinel, all filtered,
          mAP collapses to 0).
        - delta_all_fp_removed = 0 (removing the lone FP leaves 0
          detections; same loose-bound shape as `all_loc`).
    """
    gt, dt = _load("loc_vs_both_priority")
    out = error_decomposition(gt, dt)
    _assert_report_matches(out, _load_expected("loc_vs_both_priority"))


def test_segm_all_perfect_baseline_one_and_no_deltas() -> None:
    """Segm kernel: every DT mask matches a GT mask; baseline=1, all deltas=0.

    Math:
        - Two GTs (cat 1 polygon [10,10,50,50], cat 2 polygon
          [100,100,140,140]); each polygon is the axis-aligned rectangle
          covering the same area as its bbox.
        - Two DTs identical polygons / categories. The oracle's
          :func:`segm_iou` rasterizes each polygon onto a 200x200 grid and
          gets a 40x40 = 1600-pixel mask per annotation. Each DT
          intersects 1600 pixels with its same-class GT and 0 with the
          other → IoU = 1600 / 1600 = 1.0 against the same-class GT and
          0.0 against the cross-class GT.
        - mAP arithmetic mirrors `test_all_perfect_baseline_one_and_no_deltas`
          exactly because the IoU values feeding the matching engine are
          identical to the bbox case. baseline=1.0, every delta=0.
    """
    gt, dt = _load("segm_all_perfect")
    out = error_decomposition(gt, dt, segm_iou)
    _assert_report_matches(out, _load_expected("segm_all_perfect"))


def test_segm_all_loc_isolates_loc_bin() -> None:
    """Segm kernel: every DT mask has same-class IoU in [t_b, t_f) → Loc.

    Math:
        - 1 GT cat 1 polygon [0,0,100,100] → mask is the [0:100, 0:100]
          slice of the 200x200 grid; area = 10000 pixels.
        - 1 DT cat 1 polygon [0,0,80,50] → mask is [0:50, 0:80] slice;
          area = 4000 pixels.
        - Intersection: the [0:50, 0:80] slice (DT is fully contained in
          GT) = 4000 pixels.
            union = 10000 + 4000 - 4000 = 10000.
            IoU = 0.4.
          In [0.1, 0.5) → Loc.
        - Baseline mirrors `test_all_loc_isolates_loc_bin`: at every
          IoU threshold ≥ 0.5 the DT is unmatched → AP = 0 → mAP = 0.
        - Loc fix snaps DT polygon to the GT polygon (the oracle's
          `_apply_fix` for "loc" replaces both bbox and segmentation),
          so segm IoU becomes 1.0 → TP everywhere → AP = 1, mAP = 1.
          delta_loc = 1.0.
        - delta_all_fp_removed = 0 (removing the lone FP leaves 0
          detections; same loose-bound shape as `all_loc`).
    """
    gt, dt = _load("segm_all_loc")
    out = error_decomposition(gt, dt, segm_iou)
    _assert_report_matches(out, _load_expected("segm_all_loc"))


def test_segm_all_cls_isolates_cls_bin() -> None:
    """Segm kernel: every DT mask has wrong class but right geometry → Cls.

    Math:
        - 1 image, 2 GTs:
            GT1 cat 1 polygon [10,10,50,50] (40x40 mask).
            GT2 cat 2 polygon [100,100,140,140] (40x40 mask).
        - 2 DTs:
            DT1 cat 2 polygon [10,10,50,50] → IoU=1 with GT1 (wrong
                class), 0 with GT2.
            DT2 cat 1 polygon [100,100,140,140] → IoU=1 with GT2 (wrong
                class), 0 with GT1.
          Each DT: iou_same=0, iou_cross=1.0 ≥ t_f → Cls.
        - Baseline / cls-fix arithmetic mirrors
          `test_all_cls_isolates_cls_bin` exactly (same per-class IoU
          values, same shape). baseline_map=0, delta_cls=1.0, others=0,
          delta_all_fp_removed=0 (loose-bound shape, removing FPs leaves
          zero detections).
    """
    gt, dt = _load("segm_all_cls")
    out = error_decomposition(gt, dt, segm_iou)
    _assert_report_matches(out, _load_expected("segm_all_cls"))


def test_boundary_all_perfect_baseline_one_and_no_deltas() -> None:
    """Identical GT / DT masks under the boundary kernel; nothing to bin.

    Math:
        - 200x200 image; ``d = round(0.02 * sqrt(80000)) = 6``.
        - GT cat 1 at ``[50, 50, 50, 50]`` and GT cat 2 at
          ``[120, 120, 50, 50]``; DTs identical to each GT, score 0.9.
        - For each pair: mask_iou = 1 (identical 50x50 rectangles),
          band_iou = 1 (identical 6-pixel frames). ``min = 1`` ≥ t_f →
          TP at every threshold. Per-cat AP = 1.0 across the 10-IoU
          ladder; mAP = 1.0.
        - Every bin is empty. All deltas = 0.
    """
    gt, dt = _load("boundary_all_perfect")
    out = error_decomposition(
        gt,
        dt,
        _boundary_sim(),
        t_f=0.5,
        t_b=_BOUNDARY_T_B,
        kernel_name="boundary",
    )
    _assert_report_matches(out, _load_expected("boundary_all_perfect"))


def test_boundary_all_loc_isolates_loc_bin() -> None:
    """Same-class boundary IoU lands in ``[t_b, t_f) = [0.05, 0.5)`` → Loc.

    Math (200x200 canvas, dilation_ratio=0.02 → d=6):
        - 1 GT cat 1 at ``[50, 50, 50, 50]``; rasterized mask covers
          ``x ∈ [50, 100), y ∈ [50, 100)`` (area 2500). Eroded interior
          covers ``x ∈ [56, 94), y ∈ [56, 94)`` (area 38*38 = 1444).
          Band area = 2500 - 1444 = 1056.
        - 1 DT cat 1 at ``[75, 50, 50, 50]`` (shifted +25 in x). Mask
          covers ``x ∈ [75, 125), y ∈ [50, 100)`` (area 2500). Eroded
          interior covers ``x ∈ [81, 119), y ∈ [56, 94)`` (area 1444).
          Band area = 1056.
        - **Mask IoU.** Mask intersection = ``x ∈ [75, 100) * y ∈ [50, 100) = 25 * 50 = 1250``.
          Mask union = 2500 + 2500 - 1250 = 3750. Mask IoU = 1250/3750 = 1/3.
        - **Band intersection.** Inclusion-exclusion on the mask
          intersection:
            - GT interior pixels inside the mask intersection:
              ``x ∈ [75, 94) * y ∈ [56, 94) = 19 * 38 = 722``.
            - DT interior pixels inside the mask intersection:
              ``x ∈ [81, 100) * y ∈ [56, 94) = 19 * 38 = 722``.
            - Both interiors at once (subtracted twice, add back):
              ``x ∈ [81, 94) * y ∈ [56, 94) = 13 * 38 = 494``.
            - Band intersection = 1250 - 722 - 722 + 494 = 300.
          Band union = 1056 + 1056 - 300 = 1812. Band IoU = 300/1812
          ≈ 0.16556.
        - **Boundary IoU = min(mask, band) ≈ 0.16556** ∈ [0.05, 0.5)
          → Loc bin. ``iou_same`` is the only IoU (no other classes),
          so ``iou_same >= t_b and iou_same >= iou_cross`` succeeds and
          attribution is Loc.
        - Baseline: at every IoU threshold ≥ 0.5, the DT's IoU 0.166 is
          below; unmatched. tp=0, fp=1, AP = 0. mAP = 0.
        - Loc fix snaps DT segmentation to GT's → identical masks →
          IoU = 1.0 → TP at every threshold → AP = 1.0 → mAP = 1.0.
          delta_loc = 1.0.
        - Other bins empty. delta_all_fp_removed = 0 (removing the lone
          FP leaves 0 detections; same loose-bound shape as the bbox
          ``all_loc`` fixture).
    """
    gt, dt = _load("boundary_all_loc")
    out = error_decomposition(
        gt,
        dt,
        _boundary_sim(),
        t_f=0.5,
        t_b=_BOUNDARY_T_B,
        kernel_name="boundary",
    )
    _assert_report_matches(out, _load_expected("boundary_all_loc"))


def test_boundary_all_cls_isolates_cls_bin() -> None:
    """DT geometry matches a wrong-class GT exactly → Cls.

    Math (200x200, d=6):
        - 2 GTs: GT1 cat 1 at ``[50, 50, 50, 50]``; GT2 cat 2 at
          ``[120, 120, 50, 50]``. Both rectangles disjoint (no mask
          overlap, no band overlap).
        - 2 DTs:
            DT1: cat 2 at ``[50, 50, 50, 50]`` (GT1 geometry, wrong
                 class). Boundary IoU vs GT1 = 1.0 (identical masks),
                 vs GT2 = 0 (disjoint). iou_same (cat 2) = 0,
                 iou_cross (cat 1) = 1.0 ≥ t_f → Cls.
            DT2: cat 1 at ``[120, 120, 50, 50]`` — symmetric. Cls.
        - Baseline:
            cat 1: GT1 + DT2. DT2 IoU=0 with GT1 → unmatched. AP = 0.
            cat 2: symmetric. AP = 0.
            mAP = 0.
        - Cls fix relabels each DT to its cross-class GT's category:
            cat 1 has 1 TP (DT1 relabelled), AP = 1.0.
            cat 2 has 1 TP (DT2 relabelled), AP = 1.0.
            mAP = 1.0. delta_cls = 1.0.
        - Other bins empty. delta_all_fp_removed = 0 (removing the two
          FPs leaves 0 detections; same loose-bound shape as the bbox
          ``all_cls`` fixture).
    """
    gt, dt = _load("boundary_all_cls")
    out = error_decomposition(
        gt,
        dt,
        _boundary_sim(),
        t_f=0.5,
        t_b=_BOUNDARY_T_B,
        kernel_name="boundary",
    )
    _assert_report_matches(out, _load_expected("boundary_all_cls"))


_ALL_FIXTURES = [
    "all_perfect",
    "all_bkg",
    "all_cls",
    "all_loc",
    "all_dupe",
    "with_ignore",
    "loc_vs_both_priority",
]

_SEGM_FIXTURES = [
    "segm_all_perfect",
    "segm_all_loc",
    "segm_all_cls",
]

_BOUNDARY_FIXTURES = [
    "boundary_all_perfect",
    "boundary_all_loc",
    "boundary_all_cls",
]


@pytest.mark.parametrize("name", _ALL_FIXTURES)
def test_report_carries_resolved_config(name: str) -> None:
    """The report's `config` block carries the resolved thresholds.

    ADR-0022 requires the report to record `(t_f, t_b, kernel)` so a
    screenshot of a number can be re-derived from the report alone.
    """
    gt, dt = _load(name)
    out = error_decomposition(gt, dt, t_f=0.5, t_b=0.1)
    assert out["config"] == {"t_f": 0.5, "t_b": 0.1, "kernel": "bbox"}


@pytest.mark.parametrize("name", _SEGM_FIXTURES)
def test_segm_report_carries_resolved_config(name: str) -> None:
    """Segm kernel: `config.kernel` is `"segm"` when caller pins it via kwarg."""
    gt, dt = _load(name)
    out = error_decomposition(gt, dt, segm_iou, t_f=0.5, t_b=0.1, kernel_name="segm")
    assert out["config"] == {"t_f": 0.5, "t_b": 0.1, "kernel": "segm"}


@pytest.mark.parametrize("name", _ALL_FIXTURES + _SEGM_FIXTURES)
def test_report_shape(name: str) -> None:
    """Sanity-check the dict shape so downstream Rust matchers can rely on it."""
    gt, dt = _load(name)
    kernel = segm_iou if name.startswith("segm_") else bbox_iou
    out = error_decomposition(gt, dt, kernel)
    assert set(out.keys()) == {
        "baseline_map",
        "delta",
        "delta_all_fp_removed",
        "config",
    }
    assert set(out["delta"].keys()) == {
        "cls",
        "loc",
        "both",
        "dupe",
        "bkg",
        "missed",
    }
    assert isinstance(out["baseline_map"], float)
    for v in out["delta"].values():
        assert isinstance(v, float)
    assert isinstance(out["delta_all_fp_removed"], float)


@pytest.mark.parametrize("name", _BOUNDARY_FIXTURES)
def test_boundary_report_carries_resolved_config(name: str) -> None:
    """The boundary kernel report carries ``kernel = "boundary"`` and the
    ADR-0022 boundary defaults (``t_f=0.5``, ``t_b=0.05``).
    """
    gt, dt = _load(name)
    out = error_decomposition(
        gt,
        dt,
        _boundary_sim(),
        t_f=0.5,
        t_b=_BOUNDARY_T_B,
        kernel_name="boundary",
    )
    assert out["config"] == {"t_f": 0.5, "t_b": _BOUNDARY_T_B, "kernel": "boundary"}
