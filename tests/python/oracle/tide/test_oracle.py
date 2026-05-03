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

import json
from pathlib import Path
from typing import Any

import pytest

from .oracle import error_decomposition

FIXTURES = Path(__file__).parent / "fixtures"

# Hand-computed assertions are authored to ~10 decimal places of float
# precision. The 1e-6 tolerance is the ADR-0021 oracle-stability budget
# (Rust-vs-oracle parity has its own tighter `1e-9` budget — that's a
# different test, deferred to Week 2).
TOL = 1e-6


def _load(name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fix_dir = FIXTURES / name
    gt = json.loads((fix_dir / "gt.json").read_text())
    dt = json.loads((fix_dir / "dt.json").read_text())
    return gt, dt


def _assert_close(actual: float, expected: float, label: str) -> None:
    assert abs(actual - expected) < TOL, (
        f"{label}: expected {expected!r}, got {actual!r}, "
        f"diff {actual - expected!r} exceeds tolerance {TOL!r}"
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
    _assert_close(out["baseline_map"], 1.0, "baseline_map")
    for bin_name in ("cls", "loc", "both", "dupe", "bkg", "missed"):
        _assert_close(out["delta"][bin_name], 0.0, f"delta[{bin_name}]")
    _assert_close(out["delta_all_fp_removed"], 0.0, "delta_all_fp_removed")


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
    _assert_close(out["baseline_map"], 0.5, "baseline_map")
    _assert_close(out["delta"]["bkg"], 0.5, "delta[bkg]")
    for bin_name in ("cls", "loc", "both", "dupe", "missed"):
        _assert_close(out["delta"][bin_name], 0.0, f"delta[{bin_name}]")
    _assert_close(out["delta_all_fp_removed"], 0.5, "delta_all_fp_removed")


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
    _assert_close(out["baseline_map"], 0.0, "baseline_map")
    _assert_close(out["delta"]["cls"], 1.0, "delta[cls]")
    for bin_name in ("loc", "both", "dupe", "bkg", "missed"):
        _assert_close(out["delta"][bin_name], 0.0, f"delta[{bin_name}]")
    _assert_close(out["delta_all_fp_removed"], 0.0, "delta_all_fp_removed")


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
    _assert_close(out["baseline_map"], 0.0, "baseline_map")
    _assert_close(out["delta"]["loc"], 1.0, "delta[loc]")
    for bin_name in ("cls", "both", "dupe", "bkg", "missed"):
        _assert_close(out["delta"][bin_name], 0.0, f"delta[{bin_name}]")
    _assert_close(out["delta_all_fp_removed"], 0.0, "delta_all_fp_removed")


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
    expected_baseline = 76.0 / 101.0
    expected_dupe_delta = 25.0 / 101.0
    _assert_close(out["baseline_map"], expected_baseline, "baseline_map")
    _assert_close(out["delta"]["dupe"], expected_dupe_delta, "delta[dupe]")
    for bin_name in ("cls", "loc", "both", "bkg", "missed"):
        _assert_close(out["delta"][bin_name], 0.0, f"delta[{bin_name}]")
    _assert_close(out["delta_all_fp_removed"], expected_dupe_delta, "delta_all_fp_removed")


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
    _assert_close(out["baseline_map"], 0.5, "baseline_map")
    _assert_close(out["delta"]["bkg"], 0.5, "delta[bkg]")
    for bin_name in ("cls", "loc", "both", "dupe", "missed"):
        _assert_close(out["delta"][bin_name], 0.0, f"delta[{bin_name}]")
    _assert_close(out["delta_all_fp_removed"], 0.5, "delta_all_fp_removed")


@pytest.mark.parametrize(
    "name",
    ["all_perfect", "all_bkg", "all_cls", "all_loc", "all_dupe", "with_ignore"],
)
def test_report_carries_resolved_config(name: str) -> None:
    """The report's `config` block carries the resolved thresholds.

    ADR-0022 requires the report to record `(t_f, t_b, kernel)` so a
    screenshot of a number can be re-derived from the report alone.
    """
    gt, dt = _load(name)
    out = error_decomposition(gt, dt, t_f=0.5, t_b=0.1)
    assert out["config"] == {"t_f": 0.5, "t_b": 0.1, "kernel": "bbox"}


@pytest.mark.parametrize(
    "name",
    ["all_perfect", "all_bkg", "all_cls", "all_loc", "all_dupe", "with_ignore"],
)
def test_report_shape(name: str) -> None:
    """Sanity-check the dict shape so downstream Rust matchers can rely on it."""
    gt, dt = _load(name)
    out = error_decomposition(gt, dt)
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
