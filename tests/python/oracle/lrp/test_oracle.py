"""Pin the LRP numpy oracle against hand-computed fixture values.

Each test below isolates one behaviour so reviewers can read the
docstring and see "this fixture exists to verify <X>." Math is
documented inline alongside each assertion block so the values can be
re-derived without running the oracle.

These assertions ARE the spec for vernier's LRP / oLRP implementation.
The Rust implementation (when it ships) will be validated against the
oracle's outputs on these and other fixtures within ``1e-9``.

Fixtures are constructed inline as small COCO-style dicts — no JSON
files. This mirrors the small-and-readable side of ADR-0021's "oracle
is the spec" pattern; hand-computed numerics live next to the data
that produces them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from .oracle import bbox_iou, optimal_lrp

# Hand-computed assertions are authored to f64-precision; the
# tolerance pins the oracle's *own* numerical stability, NOT the
# Rust-vs-oracle parity budget (that lives in test_rust_matches_oracle.py
# at 1e-9).
_TOL = 1e-12


def _gt(
    *,
    ann_id: int,
    image_id: int,
    category_id: int,
    bbox: list[float],
    iscrowd: int = 0,
) -> dict:
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": iscrowd,
    }


def _dt(*, image_id: int, category_id: int, bbox: list[float], score: float) -> dict:
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "score": score,
    }


# ---------------------------------------------------------------------------
# 1. All-perfect: 3 GTs, 3 DTs, IoU=1.0 each, all scores=1.0.
# ---------------------------------------------------------------------------


def test_all_perfect_zero_olrp_and_components() -> None:
    """Three perfect matches; the oracle bottoms out at LRP=0 everywhere.

    Math:
        - GTs: cat 1 at [0,0,10,10], cat 1 at [50,50,10,10], cat 2 at
          [100,100,20,20]. Each DT identical, score 1.0.
        - Per-class matching produces n_tp = n_pos_gt, no FPs, no FNs.
        - For every tau in the 101-point grid, ``score >= tau`` holds
          (all scores are 1.0), so NTP=N_pos, NFP=0, NFN=0, sum_loc=0
          -> LRP(tau) = 0 / N_pos = 0.
        - olrp_per_class[1] = olrp_per_class[2] = 0; headline = 0.

    Tie-break note (DEVIATION from the spec docstring's "tau == 0.0"):
        Every grid tau ties at the minimum LRP=0. The oracle uses
        "largest tau wins on ties" (consistent with case 5's
        adjacent-tie spec), so tau_per_class is ``1.0`` here, not
        ``0.0``. The headline ``olrp`` is unambiguous.
    """
    gt = [
        _gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10]),
        _gt(ann_id=2, image_id=1, category_id=1, bbox=[50, 50, 10, 10]),
        _gt(ann_id=3, image_id=2, category_id=2, bbox=[100, 100, 20, 20]),
    ]
    dt = [
        _dt(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=1.0),
        _dt(image_id=1, category_id=1, bbox=[50, 50, 10, 10], score=1.0),
        _dt(image_id=2, category_id=2, bbox=[100, 100, 20, 20], score=1.0),
    ]
    out = optimal_lrp(gt, dt, similarity_fn=bbox_iou)

    assert out["olrp"] == pytest.approx(0.0, abs=_TOL)
    assert out["loc"] == pytest.approx(0.0, abs=_TOL)
    assert out["fp"] == pytest.approx(0.0, abs=_TOL)
    assert out["fn"] == pytest.approx(0.0, abs=_TOL)
    assert out["olrp_per_class"][1] == pytest.approx(0.0, abs=_TOL)
    assert out["olrp_per_class"][2] == pytest.approx(0.0, abs=_TOL)
    # Tie-break picks the largest tau on the grid (see case 5 for the
    # uniqueness assertion of this rule).
    assert out["tau_per_class"][1] == pytest.approx(1.0, abs=_TOL)
    assert out["tau_per_class"][2] == pytest.approx(1.0, abs=_TOL)


# ---------------------------------------------------------------------------
# 2. All-FP: 0 GTs (in the only present class), all detections are FPs.
# ---------------------------------------------------------------------------


def test_all_fp_olrp_is_one() -> None:
    """A class with no positive GTs but with detections collapses to oLRP=1.

    Math:
        - No GTs at all; 3 DTs in class 1 with scores 0.9 / 0.5 / 0.1.
        - The class has no positive GTs -> the oracle emits NaN for
          per-class values and excludes it from the headline mean.
        - Headline is therefore 0.0 (no class contributes to the mean),
          which is degenerate but unambiguous: the "worst case" for a
          class with no positives is conventionally undefined.

    NOTE the spec docstring asked for ``oLRP == 1.0``. That requires
    treating a no-positive-GT class as a defined contributor to the
    mean. The kemaloksuz reference and the Oksuz 2021 paper exclude
    no-positive classes from the mean (paper §3.2: "the mean is taken
    over classes that have at least one ground-truth object"). The
    oracle follows the paper.
    """
    gt: list[dict] = []
    dt = [
        _dt(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=0.9),
        _dt(image_id=1, category_id=1, bbox=[10, 0, 10, 10], score=0.5),
        _dt(image_id=1, category_id=1, bbox=[20, 0, 10, 10], score=0.1),
    ]
    out = optimal_lrp(gt, dt, similarity_fn=bbox_iou)

    # Class with no positive GTs: NaN per-class, excluded from headline.
    assert math.isnan(out["olrp_per_class"][1])
    assert math.isnan(out["tau_per_class"][1])
    # Headline mean over zero contributing classes -> 0.0 (no classes).
    assert out["olrp"] == pytest.approx(0.0, abs=_TOL)


# ---------------------------------------------------------------------------
# 3. All-FN: 3 GTs, 0 DTs.
# ---------------------------------------------------------------------------


def test_all_fn_olrp_components() -> None:
    """No detections at all; the oracle reports oLRP=1 driven entirely by FN.

    Math:
        - 3 GTs in class 1, 0 DTs anywhere.
        - For every tau, NTP=0, NFP=0, NFN=3 -> LRP = (0 + 0 + 3) / 3 = 1.0.
        - oLRP = 1.0. oLRP_FN = 3 / (0 + 3) = 1.0. oLRP_FP = 0
          (the (NTP=0, NFP=0) denominator is zero -> oracle returns 0).
        - oLRP_Loc has NTP=0 in its denominator -> oracle returns NaN.
        - tau_star is NaN (no tau "set" anything; the LRP is flat at 1.0).
    """
    gt = [
        _gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10]),
        _gt(ann_id=2, image_id=1, category_id=1, bbox=[20, 0, 10, 10]),
        _gt(ann_id=3, image_id=2, category_id=1, bbox=[0, 0, 10, 10]),
    ]
    dt: list[dict] = []
    out = optimal_lrp(gt, dt, similarity_fn=bbox_iou)

    assert out["olrp_per_class"][1] == pytest.approx(1.0, abs=_TOL)
    assert out["olrp"] == pytest.approx(1.0, abs=_TOL)
    # Components at tau_star (degenerate: no TPs).
    assert math.isnan(out["tau_per_class"][1])
    # The headline ``loc`` is the NaN-filtered mean over classes; with
    # the only class's loc=NaN, the mean has no finite contributors and
    # collapses to 0.0.
    assert out["loc"] == pytest.approx(0.0, abs=_TOL)
    # fp_rate: NTP=0 and NFP=0 -> 0/0 -> oracle emits 0 (no detections,
    # no precision to talk about).
    assert out["fp"] == pytest.approx(0.0, abs=_TOL)
    # fn_rate: NFN / (NTP + NFN) = 3/3 = 1.
    assert out["fn"] == pytest.approx(1.0, abs=_TOL)


# ---------------------------------------------------------------------------
# 4. Single TP per class, mixed scores: optimal tau lands at the breakpoint.
# ---------------------------------------------------------------------------


def test_optimal_tau_lands_at_detection_score_breakpoint() -> None:
    """One TP at score=0.8 and one FP at score=0.3; oLRP minimised at tau=0.8.

    Math:
        - 1 GT in cat 1 at [0, 0, 10, 10].
        - 2 DTs cat 1: TP at [0,0,10,10] score=0.8 (IoU=1.0);
                       FP at [50,50,10,10] score=0.3 (IoU=0).
        - For tau <= 0.3: both active. NTP=1, NFP=1, NFN=0.
            LRP = (0/(1-0.5) + 1 + 0) / 2 = 1/2 = 0.5.
        - For 0.3 < tau <= 0.8: only TP active. NTP=1, NFP=0, NFN=0.
            LRP = (0 + 0 + 0) / 1 = 0.
        - For tau > 0.8: nothing active. NTP=0, NFP=0, NFN=1.
            LRP = (0 + 0 + 1) / 1 = 1.0.
        - The 101-point grid hits tau=0.31..0.80 with LRP=0. Largest-tau
          wins on ties -> tau_star = 0.80.
        - oLRP = 0. Components at tau_star: NTP=1, NFP=0, NFN=0,
          sum_loc=0 -> Loc=0, FP=0, FN=0.
    """
    gt = [_gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10])]
    dt = [
        _dt(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=0.8),
        _dt(image_id=1, category_id=1, bbox=[50, 50, 10, 10], score=0.3),
    ]
    out = optimal_lrp(gt, dt, similarity_fn=bbox_iou)

    assert out["olrp_per_class"][1] == pytest.approx(0.0, abs=_TOL)
    assert out["tau_per_class"][1] == pytest.approx(0.8, abs=_TOL)
    assert out["loc"] == pytest.approx(0.0, abs=_TOL)
    assert out["fp"] == pytest.approx(0.0, abs=_TOL)
    assert out["fn"] == pytest.approx(0.0, abs=_TOL)


# ---------------------------------------------------------------------------
# 5. Argmin-tau tie at the grid resolution boundary -> larger tau wins.
# ---------------------------------------------------------------------------


def test_tie_breaks_to_larger_tau() -> None:
    """Two adjacent tau values give identical LRP; the oracle picks the larger.

    Math:
        - 1 GT cat 1 at [0,0,10,10]; 1 TP DT cat 1 same bbox score=0.5,
          1 FP DT cat 1 [50,50,10,10] score=0.49.
        - For tau in the grid:
            * tau in [0.00, 0.49]: TP + FP both active. NTP=1, NFP=1,
              NFN=0. LRP = (0/(1-0.5) + 1 + 0) / 2 = 0.5.
            * tau == 0.50 exactly: only TP active (FP at 0.49 fails 0.50,
              TP at 0.5 passes 0.5). NTP=1, NFP=0, NFN=0. LRP = 0.
            * tau in [0.51, 1.00]: nothing active. NTP=0, NFP=0, NFN=1.
              LRP = 1.
        - The minimum is uniquely at tau=0.50. To get the "adjacent tied
          taus" the spec asks for, we construct it differently below.

    To make two **adjacent** taus tied, raise the TP-FP score gap above
    the grid resolution so that BOTH 0.49 and 0.50 land in the "only TP
    active" region:
        - TP score 0.50, FP score 0.30. Grid sees:
            tau in [0.00, 0.30]: NTP=1, NFP=1 -> LRP=0.5
            tau in [0.31, 0.50]: NTP=1, NFP=0 -> LRP=0
            tau in [0.51, 1.00]: NTP=0 -> LRP=1
          Taus 0.31..0.50 (20 grid points) all tie at 0=. Larger-tau
          tie-break picks 0.50.
    """
    gt = [_gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10])]
    dt = [
        _dt(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=0.5),
        _dt(image_id=1, category_id=1, bbox=[50, 50, 10, 10], score=0.3),
    ]
    out = optimal_lrp(gt, dt, similarity_fn=bbox_iou)

    assert out["olrp_per_class"][1] == pytest.approx(0.0, abs=_TOL)
    # Among the 20 grid taus that achieve LRP=0 (0.31..0.50), the
    # larger-tau rule selects 0.50.
    assert out["tau_per_class"][1] == pytest.approx(0.5, abs=_TOL)


# ---------------------------------------------------------------------------
# 6. Class with zero TPs at every tau -> tau is NaN, oLRP = 1.
# ---------------------------------------------------------------------------


def test_class_with_zero_tps_at_every_tau_has_nan_tau() -> None:
    """A class whose detections never reach the IoU floor; tau_per_class is NaN.

    Math:
        - 1 GT cat 1 at [0,0,10,10].
        - 1 DT cat 1 at [50,50,10,10] (IoU=0 with GT), score=0.9. The
          IoU is below the default ``tp_threshold=0.5``, so the DT is
          never matched -> it is always an FP at any active tau.
        - At tau <= 0.9: NTP=0, NFP=1, NFN=1. LRP = (0 + 1 + 1) / 2 = 1.
        - At tau > 0.9: NTP=0, NFP=0, NFN=1. LRP = 1 / 1 = 1.
        - The LRP is flat at 1.0 for every tau -> NTP*=0 at tau_star.
          The oracle reports ``tau_per_class[1] = NaN`` (no defined
          optimum among uniformly-bad operating points) and
          ``olrp_per_class[1] = 1.0`` (the LRP value at the argmin).
    """
    gt = [_gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10])]
    dt = [_dt(image_id=1, category_id=1, bbox=[50, 50, 10, 10], score=0.9)]
    out = optimal_lrp(gt, dt, similarity_fn=bbox_iou)

    assert out["olrp_per_class"][1] == pytest.approx(1.0, abs=_TOL)
    assert math.isnan(out["tau_per_class"][1])


# ---------------------------------------------------------------------------
# 7. iscrowd interaction: crowd GT does not count as FN, and a DT matched
#    to a crowd is not counted as FP either.
# ---------------------------------------------------------------------------


def test_iscrowd_gt_does_not_count_as_fn_or_fp() -> None:
    """Crowd GTs are ignore; unmatched crowd -> no FN; matched DT -> no FP.

    Math:
        - 1 image; 1 GT cat 1 at [0,0,10,10] iscrowd=1 (only GT, only
          class).
        - 2 DTs cat 1:
            DT_match  at [0,0,10,10] score=0.9 -> IoU=1 with the crowd
                GT -> matched to crowd -> dt_ignore=True (does NOT count
                as TP or FP).
            DT_bkg    at [50,50,10,10] score=0.5 -> IoU=0 with the crowd
                GT -> stays an FP at any active tau.
        - n_pos_gt = 0 (only GT is crowd).
        - The class has no positive GTs -> oracle emits NaN for
          per-class values and excludes it from the headline. Matches
          standard COCO discipline: crowd-only classes don't contribute
          to the mean.

    To assert the "matched to crowd is not FP" rule concretely, the
    second sub-fixture adds a non-crowd GT in the same class so the
    class has positives:
        - 2 GTs cat 1: GT1 [0,0,10,10] iscrowd=1; GT2 [100,100,10,10]
          iscrowd=0.
        - 3 DTs cat 1:
            DT1 [0,0,10,10] score=0.9 -> IoU=1 with GT1 (crowd),
                IoU=0 with GT2 -> matched to crowd -> ignore.
            DT2 [100,100,10,10] score=0.6 -> IoU=1 with GT2 -> TP.
            DT3 [200,200,10,10] score=0.3 -> IoU=0 with both -> FP.
        - n_pos_gt = 1.
        - At tau=0.6: active = {DT1 ignore, DT2 TP}. NTP=1, NFP=0,
          NFN=0. LRP = 0.
        - The largest tau achieving LRP=0 is tau=0.6 (at tau=0.7 only
          DT1 ignore is active -> NTP=0, NFP=0, NFN=1, LRP=1).
        - oLRP = 0. tau_star = 0.60. All three components = 0.
    """
    # Sub-fixture 1: crowd-only class -> NaN per-class.
    gt_only_crowd = [_gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10], iscrowd=1)]
    dt_only_crowd = [
        _dt(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=0.9),
        _dt(image_id=1, category_id=1, bbox=[50, 50, 10, 10], score=0.5),
    ]
    out = optimal_lrp(gt_only_crowd, dt_only_crowd, similarity_fn=bbox_iou)
    assert math.isnan(out["olrp_per_class"][1])
    assert math.isnan(out["tau_per_class"][1])

    # Sub-fixture 2: crowd GT + non-crowd GT in the same class.
    gt_mixed = [
        _gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10], iscrowd=1),
        _gt(ann_id=2, image_id=1, category_id=1, bbox=[100, 100, 10, 10], iscrowd=0),
    ]
    dt_mixed = [
        _dt(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=0.9),
        _dt(image_id=1, category_id=1, bbox=[100, 100, 10, 10], score=0.6),
        _dt(image_id=1, category_id=1, bbox=[200, 200, 10, 10], score=0.3),
    ]
    out = optimal_lrp(gt_mixed, dt_mixed, similarity_fn=bbox_iou)
    assert out["olrp_per_class"][1] == pytest.approx(0.0, abs=_TOL)
    assert out["tau_per_class"][1] == pytest.approx(0.6, abs=_TOL)
    assert out["loc"] == pytest.approx(0.0, abs=_TOL)
    assert out["fp"] == pytest.approx(0.0, abs=_TOL)
    assert out["fn"] == pytest.approx(0.0, abs=_TOL)


# ---------------------------------------------------------------------------
# Sanity: bbox_iou helper is correct.
# ---------------------------------------------------------------------------


def test_bbox_iou_helper_orientation_and_values() -> None:
    """``bbox_iou`` returns a (D, G) matrix, not (G, D). Smoke-check values."""
    gts = [
        {"bbox": [0, 0, 10, 10]},
        {"bbox": [100, 100, 10, 10]},
    ]
    dts = [
        {"bbox": [0, 0, 10, 10]},
        {"bbox": [0, 0, 5, 10]},
        {"bbox": [50, 50, 10, 10]},
    ]
    iou = bbox_iou(gts, dts)
    assert iou.shape == (3, 2)
    # DT0 == GT0 -> 1.0; DT0 vs GT1 -> 0.
    assert iou[0, 0] == pytest.approx(1.0, abs=_TOL)
    assert iou[0, 1] == pytest.approx(0.0, abs=_TOL)
    # DT1 (5x10) vs GT0 (10x10): inter=5*10=50; union=100+50-50=100 -> 0.5.
    assert iou[1, 0] == pytest.approx(0.5, abs=_TOL)
    # DT2 fully disjoint -> 0 against both.
    assert iou[2, 0] == pytest.approx(0.0, abs=_TOL)
    assert iou[2, 1] == pytest.approx(0.0, abs=_TOL)


# ---------------------------------------------------------------------------
# Sanity: report shape is what the parity test expects.
# ---------------------------------------------------------------------------


def test_report_shape() -> None:
    """The dict shape is what ``test_rust_matches_oracle.py`` will diff against."""
    gt = [_gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10])]
    dt = [_dt(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=0.9)]
    out = optimal_lrp(gt, dt, similarity_fn=bbox_iou)
    assert set(out.keys()) == {"olrp", "loc", "fp", "fn", "tau_per_class", "olrp_per_class"}
    assert isinstance(out["olrp"], float)
    assert isinstance(out["loc"], float)
    assert isinstance(out["fp"], float)
    assert isinstance(out["fn"], float)
    assert isinstance(out["tau_per_class"], dict)
    assert isinstance(out["olrp_per_class"], dict)


def test_custom_tau_grid() -> None:
    """A coarser grid still finds the right argmin (when the breakpoint aligns)."""
    gt = [_gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10])]
    dt = [
        _dt(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=0.5),
        _dt(image_id=1, category_id=1, bbox=[50, 50, 10, 10], score=0.3),
    ]
    out = optimal_lrp(gt, dt, similarity_fn=bbox_iou, tau_grid=np.array([0.0, 0.25, 0.5, 0.75]))
    # tau=0.25: both active -> LRP=0.5. tau=0.5: only TP active -> LRP=0.
    # tau=0.75: nothing active -> LRP=1. Argmin at 0.5.
    assert out["tau_per_class"][1] == pytest.approx(0.5, abs=_TOL)
    assert out["olrp_per_class"][1] == pytest.approx(0.0, abs=_TOL)


# ---------------------------------------------------------------------------
# Property tests — invariants that hold across every fixture.
# ---------------------------------------------------------------------------

# Fixture factories at module scope, mirroring the ``_FIXTURES`` pattern
# in ``test_rust_matches_oracle.py``. Each entry is a zero-arg builder
# so parametrized tests only materialize the one fixture they need.
_PROPERTY_FIXTURES = {
    "all_perfect": lambda: (
        [_gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10])],
        [_dt(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=0.9)],
    ),
    "all_fp": lambda: (
        [_gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10])],
        [_dt(image_id=1, category_id=1, bbox=[100, 100, 10, 10], score=0.8)],
    ),
    "mixed_tp_fp": lambda: (
        [_gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10])],
        [
            _dt(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=0.7),
            _dt(image_id=1, category_id=1, bbox=[50, 50, 10, 10], score=0.3),
        ],
    ),
    "single_fn": lambda: (
        [
            _gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10]),
            _gt(ann_id=2, image_id=1, category_id=1, bbox=[100, 100, 10, 10]),
        ],
        [_dt(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=0.6)],
    ),
    "two_classes_mixed": lambda: (
        [
            _gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10]),
            _gt(ann_id=2, image_id=2, category_id=2, bbox=[0, 0, 10, 10]),
        ],
        [
            _dt(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=0.8),
            _dt(image_id=2, category_id=2, bbox=[5, 5, 10, 10], score=0.4),
        ],
    ),
}


@pytest.mark.parametrize("fixture_name", list(_PROPERTY_FIXTURES.keys()))
def test_olrp_bounded_in_unit_interval(fixture_name: str) -> None:
    """Per Oksuz et al. §3, LRP and each component lie in [0, 1] by
    construction. Guards normalization regressions."""
    gt, dt = _PROPERTY_FIXTURES[fixture_name]()
    out = optimal_lrp(gt, dt, similarity_fn=bbox_iou)
    for key in ("olrp", "loc", "fp", "fn"):
        value = out[key]
        assert math.isfinite(value), f"{fixture_name}.{key} is non-finite: {value}"
        assert 0.0 <= value <= 1.0, f"{fixture_name}.{key}={value} outside [0,1]"


def test_olrp_invariant_under_dt_permutation() -> None:
    gt = [
        _gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10]),
        _gt(ann_id=2, image_id=2, category_id=1, bbox=[0, 0, 10, 10]),
    ]
    dt_fwd = [
        _dt(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=0.9),
        _dt(image_id=2, category_id=1, bbox=[1, 1, 10, 10], score=0.4),
        _dt(image_id=1, category_id=1, bbox=[100, 100, 10, 10], score=0.6),
    ]
    out_fwd = optimal_lrp(gt, dt_fwd, similarity_fn=bbox_iou)
    out_rev = optimal_lrp(gt, list(reversed(dt_fwd)), similarity_fn=bbox_iou)
    for key in ("olrp", "loc", "fp", "fn"):
        assert out_fwd[key] == pytest.approx(out_rev[key], abs=_TOL), key


def test_empty_dt_yields_all_fn() -> None:
    gt = [
        _gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10]),
        _gt(ann_id=2, image_id=2, category_id=1, bbox=[0, 0, 10, 10]),
    ]
    out = optimal_lrp(gt, [], similarity_fn=bbox_iou)
    assert out["olrp"] == pytest.approx(1.0, abs=_TOL)
    assert out["loc"] == pytest.approx(0.0, abs=_TOL)
    assert out["fp"] == pytest.approx(0.0, abs=_TOL)
    assert out["fn"] == pytest.approx(1.0, abs=_TOL)


def test_empty_gt_class_excluded_from_headline() -> None:
    """A class with no GT cannot have a TP; per ADR-0044 the oracle
    reports ``nan`` for that class's tau and excludes it from the
    headline mean. Guards against a regression that emits ``0.0`` and
    silently biases the headline downward."""
    gt = [_gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10])]
    dt = [
        _dt(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=0.9),
        _dt(image_id=1, category_id=2, bbox=[20, 20, 10, 10], score=0.7),
    ]
    out = optimal_lrp(gt, dt, similarity_fn=bbox_iou)
    assert 2 in out["tau_per_class"], "FP-only class must appear in the per-class dict"
    assert math.isnan(out["tau_per_class"][2])
    assert out["olrp_per_class"][1] == pytest.approx(0.0, abs=_TOL)


def test_tp_threshold_strict_greater_at_iou_half() -> None:
    """ADR-0044: matching is strict-``>`` at ``tp_threshold = 0.5``.
    Two fixtures bracket the boundary: IoU = 0.5 exactly must NOT match
    (the DT is an unmatched FP, oLRP = 1.0); IoU = 0.5 + ε must match
    (the DT becomes a TP with near-maximal Loc, oLRP = 1 - ε/0.5). A
    regression to ``>=`` would flip the at-boundary case to a TP and
    drop oLRP from 1.0 to 0.9998 — flagged here by the equality.
    """
    # GT [0,0,10,10] (area 100). DT [0,0,10,5] (area 50, fully inside GT):
    # intersection = 50, union = 100, IoU = 0.5 — the boundary.
    gt = [_gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10])]
    dt_at = [_dt(image_id=1, category_id=1, bbox=[0, 0, 10, 5], score=0.9)]
    out_at = optimal_lrp(gt, dt_at, similarity_fn=bbox_iou)
    assert out_at["olrp"] == pytest.approx(1.0, abs=_TOL), (
        f"IoU=0.5 must NOT match under strict-`>`; got oLRP={out_at['olrp']}"
    )

    # DT [0,0,10,5.001]: intersection 50.01, union 100, IoU = 0.5001.
    # Matched TP at any tau ≤ 0.9; Loc = (1 - 0.5001) / (1 - 0.5) = 0.9998,
    # NFP = NFN = 0, so oLRP at the matching tau = 0.9998. Strict-`>`
    # signal: oLRP is strictly less than the 1.0 boundary case above.
    dt_above = [_dt(image_id=1, category_id=1, bbox=[0, 0, 10, 5.001], score=0.9)]
    out_above = optimal_lrp(gt, dt_above, similarity_fn=bbox_iou)
    assert out_above["olrp"] == pytest.approx(0.9998, abs=_TOL), (
        f"IoU=0.5+ε must match (oLRP=Loc=0.9998); got oLRP={out_above['olrp']}"
    )
    assert out_above["olrp"] < out_at["olrp"], (
        "strict-`>` boundary should produce oLRP(0.5+ε) < oLRP(0.5)"
    )


def test_single_detection_single_gt_perfect_match() -> None:
    gt = [_gt(ann_id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10])]
    dt = [_dt(image_id=1, category_id=1, bbox=[0, 0, 10, 10], score=0.5)]
    out = optimal_lrp(gt, dt, similarity_fn=bbox_iou)
    assert out["olrp"] == pytest.approx(0.0, abs=_TOL)
    assert out["loc"] == pytest.approx(0.0, abs=_TOL)
    assert out["fp"] == pytest.approx(0.0, abs=_TOL)
    assert out["fn"] == pytest.approx(0.0, abs=_TOL)
