"""Strict-mode bit-equality parity tests against the vendored
panopticapi oracle (ADR-0025).

Each fixture closes one open question from the ADR appendix:
- **Q2** U7 strict-greater at exactly 0.5: oracle and vernier both
  reject (no TP).
- **Q3** U9 iteration-order independence: shuffled fixtures produce
  the same TP set on the vernier side (the panopticapi side is
  PNG-driven and order-deterministic by construction; this is a
  vernier property test).
- **Q4** V3 multi same-category crowd: vernier `Strict` reproduces
  panopticapi last-wins; vernier `Corrected` (default) sums overlaps
  and reports a different FP count.
- **Q5** W7 long-tailed dataset: pin global SQ ≠ pooled
  `total_iou / total_TP`. The oracle's pq_average computes
  `mean(SQ_c)`; vernier matches it bit-equally and the test asserts
  divergence from the pooled formula to catch a future "innocent"
  refactor.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image as PILImage

import vernier
import vernier.panoptic
from vernier._impl import StreamingPanopticEvaluator

from .harness import (
    _oracle_snapshot,
    _vernier_snapshot,
    assert_snapshots_equal,
    id2rgb,
    summary_to_snapshot,
)

pytestmark = pytest.mark.parity_panoptic


def test_perfect_match_strict_bit_equal() -> None:
    """Sanity: a 1x10 fixture with two perfectly-matching segments
    bit-equals the oracle on every output field. This is the
    end-to-end parity smoke against the panopticapi oracle."""
    gt = {1: np.array([[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]], dtype=np.uint32)}
    dt = {1: np.array([[10, 10, 10, 10, 10, 11, 11, 11, 11, 11]], dtype=np.uint32)}
    gt_segs = {
        1: [
            {"id": 1, "category_id": 100, "iscrowd": False, "area": 5},
            {"id": 2, "category_id": 200, "iscrowd": False, "area": 5},
        ]
    }
    dt_segs = {
        1: [
            {"id": 10, "category_id": 100, "iscrowd": False, "area": 5},
            {"id": 11, "category_id": 200, "iscrowd": False, "area": 5},
        ]
    }
    cats = [
        {"id": 100, "isthing": True},
        {"id": 200, "isthing": False},
    ]

    oracle = _oracle_snapshot(gt, gt_segs, dt, dt_segs, cats)
    vsnap = _vernier_snapshot(gt, gt_segs, dt, dt_segs, cats, "corrected")
    assert_snapshots_equal(oracle, vsnap)
    # Sanity: this is a perfect match on a single thing + single stuff.
    assert oracle.pq == 1.0
    assert oracle.pq_things == 1.0
    assert oracle.pq_stuff == 1.0


def test_q2_iou_at_exactly_half_rejected_u7() -> None:
    """Quirk **U7**. Construct a candidate pair with IoU exactly 1/2
    in f64 and confirm both oracle and vernier reject the match.

    Layout: 1x2 image. GT id=1 (cat=100) at pixel 0; GT id=2 (cat=999)
    at pixel 1 (categories disagree with DT, so this pair is U5-skipped
    out of the candidate pool — keeps the test focused on (gt=1, dt=10)
    at IoU=0.5). DT id=10 (cat=100) at both pixels.
    (gt=1, dt=10) intersection=1, gt_area=1, dt_area=2, void_overlap=0.
    union = 1 + 2 - 1 - 0 = 2; IoU = 1/2. U7 strict-greater rejects.
    """
    gt = {1: np.array([[1, 2]], dtype=np.uint32)}
    dt = {1: np.array([[10, 10]], dtype=np.uint32)}
    gt_segs = {
        1: [
            {"id": 1, "category_id": 100, "iscrowd": False, "area": 1},
            {"id": 2, "category_id": 999, "iscrowd": False, "area": 1},
        ]
    }
    dt_segs = {1: [{"id": 10, "category_id": 100, "iscrowd": False, "area": 2}]}
    cats = [
        {"id": 100, "isthing": True},
        {"id": 999, "isthing": False},  # stuff so the bucket is non-empty
    ]

    oracle = _oracle_snapshot(gt, gt_segs, dt, dt_segs, cats)
    vsnap = _vernier_snapshot(gt, gt_segs, dt, dt_segs, cats, "corrected")
    assert_snapshots_equal(oracle, vsnap)
    # U7 strict-greater: no TP for cat=100.
    assert oracle.per_class[100]["pq"] == 0.0
    # GT id=1 unmatched non-crowd → FN at cat=100.
    # DT id=10 unmatched, void_overlap=0, no crowd → FP at cat=100.
    # PQ_100 = 0 / (0 + 0.5 + 0.5) = 0.


def test_q3_iter_order_independence_property() -> None:
    """Quirk **U9**. Build a fixture where one DT overlaps two GTs;
    only one match clears U7; assert the TP set is identical when
    we shuffle the segments_info ordering and the pixel ordering.
    Vernier-only property test (the panopticapi side is PNG-driven
    and order-deterministic by construction)."""
    # 1x4 image. GT id=1 covers [0..3] (3 px), GT id=2 covers pixel 3
    # (1 px). DT id=10 covers all 4 pixels. (gt=1, dt=10) iou = 0.75
    # → matches; (gt=2, dt=10) iou = 0.25 → no match.
    layouts = [
        # Original
        (
            np.array([[1, 1, 1, 2]], dtype=np.uint32),
            [
                {"id": 1, "category_id": 100, "iscrowd": False, "area": 3},
                {"id": 2, "category_id": 100, "iscrowd": False, "area": 1},
            ],
        ),
        # Pixel-shuffled
        (
            np.array([[2, 1, 1, 1]], dtype=np.uint32),
            [
                {"id": 1, "category_id": 100, "iscrowd": False, "area": 3},
                {"id": 2, "category_id": 100, "iscrowd": False, "area": 1},
            ],
        ),
        # Segments_info reordering
        (
            np.array([[1, 1, 1, 2]], dtype=np.uint32),
            [
                {"id": 2, "category_id": 100, "iscrowd": False, "area": 1},
                {"id": 1, "category_id": 100, "iscrowd": False, "area": 3},
            ],
        ),
    ]
    dt = {1: np.array([[10, 10, 10, 10]], dtype=np.uint32)}
    dt_segs = {1: [{"id": 10, "category_id": 100, "iscrowd": False, "area": 4}]}
    cats = [{"id": 100, "isthing": True}]

    snaps = []
    for label_map, segs in layouts:
        gt = {1: label_map}
        gt_segs = {1: segs}
        snaps.append(_vernier_snapshot(gt, gt_segs, dt, dt_segs, cats, "corrected"))

    # All snapshots must agree on the per-class PQ.
    pqs = [s.per_class[100]["pq"] for s in snaps]
    assert pqs[0] == pqs[1] == pqs[2], f"iter-order independence violated: {pqs}"


def test_q4_v3_multi_same_category_crowd_strict_vs_corrected() -> None:
    """Quirk **V3**. Two same-category crowds on one image; corrected
    sums their overlaps with an unmatched DT (V4-excluding it),
    strict only consults one (last-wins) and reports a FP.

    Fixture sized so the strict result is FP=1 regardless of which
    crowd "wins" (max crowd overlap alone is below the V4 threshold;
    only the sum exceeds it). See attribute.rs unit test for the
    arithmetic derivation.
    """
    gt = {1: np.array([[1, 1, 2, 2, 2, 2, 3, 3, 3, 3]], dtype=np.uint32)}
    dt = {1: np.array([[10] * 10], dtype=np.uint32)}
    gt_segs = {
        1: [
            {"id": 1, "category_id": 100, "iscrowd": True, "area": 2},
            {"id": 2, "category_id": 100, "iscrowd": True, "area": 4},
            {"id": 3, "category_id": 999, "iscrowd": False, "area": 4},
        ]
    }
    dt_segs = {1: [{"id": 10, "category_id": 100, "iscrowd": False, "area": 10}]}
    cats = [
        {"id": 100, "isthing": True},
        {"id": 999, "isthing": False},
    ]

    # Vernier strict reproduces the oracle exactly.
    oracle = _oracle_snapshot(gt, gt_segs, dt, dt_segs, cats)
    vsnap_strict = _vernier_snapshot(gt, gt_segs, dt, dt_segs, cats, "strict")
    assert_snapshots_equal(oracle, vsnap_strict)

    # Vernier corrected V3 sums all same-category crowd overlaps,
    # which V4-excludes the otherwise-FP DT 10. Observable difference:
    # under strict, cat 100 has FP=1 → contributes to n. Under
    # corrected, cat 100 has all-zero counts → no row, n drops by 1.
    # (PQ_100 itself is 0 in both modes since cat 100 has no TP.)
    vsnap_corrected = _vernier_snapshot(gt, gt_segs, dt, dt_segs, cats, "corrected")
    assert vsnap_strict.n_things != vsnap_corrected.n_things, (
        f"strict n_things={vsnap_strict.n_things} vs corrected "
        f"n_things={vsnap_corrected.n_things}; the V3 disposition flip "
        f"must shift cat 100's contribution to the things bucket"
    )


def test_q5_w7_long_tailed_global_sq_bit_equal() -> None:
    """Quirk **W7**. Long-tailed fixture: cat 100 has multiple TPs
    contributing high SQ; cat 200 has one TP at lower SQ. Global SQ
    is the unweighted **mean** of SQ_c, not pooled
    `total_iou / total_TP`. Pin the bit-equal value against the oracle.
    """
    # Image 1: GT id=1 (cat=100, area=10) covers all 10 px.
    #          DT id=10 (cat=100, area=10) covers all 10 px.
    #          intersection=10, void=0, union=10, iou=1.0.
    # Image 2: GT id=1 (cat=100, area=10) covers all 10 px.
    #          DT id=10 (cat=100, area=10) covers 7 px; pixels
    #          [7..10]=0 (VOID). intersection=7, void_overlap_dt10=
    #          (gt=1 ∩ dt=0) — wait, dt=0 means "no DT pixel" which
    #          on the DT side means VOID, but VOID on DT side is the
    #          same pixel-value 0. Need to be careful.
    #
    # Actually if pixel=0 on DT, the (VOID, dt) intersection isn't
    # what's needed — that's (0=gt-VOID, 10=dt). The U6 union
    # subtracts hist[(VOID, dt_10)] which means GT pixels that are
    # VOID where DT covers segment 10. In image 2, GT pixels [7..10]
    # are=1 (not VOID); DT covers [0..7] as 10. So (1, 10) = 7,
    # (1, 0) = 3 (wait, dt_10 doesn't cover these so it's (gt, dt=0)
    # — but dt=0 means VOID on the DT side which is pixel value 0.
    #
    # Hmm DT side cannot have VOID-as-segment-id (every non-zero
    # pixel must be in segments_info per S1). DT pixels with value
    # 0 represent "not covered by any pred segment" — which is
    # VOID on the DT side too. The U6 formula uses hist[(VOID, dt)],
    # i.e. the pair where GT=VOID and DT=segment. For image 2,
    # GT has no VOID pixels (cover entire image), so void_overlap_with_dt_10 = 0.
    # union_image2 = gt_area + dt_area - intersection - void_overlap
    #              = 10 + 7 - 7 - 0 = 10. iou = 7/10 = 0.7.
    # Wait DT id=10 area is 7 (only 7 pixels covered), not 10.
    #
    # So image 2 (gt=1, dt=10): intersection=7, gt_area=10, dt_area=7,
    #     void=0, union=10, iou=7/10 = 0.7.
    # And there's also a (gt=1, dt=0) row with intersection=3 — but
    # dt=0 isn't a segment, so the row is skipped by U3.

    image1_gt = np.array([[1] * 10], dtype=np.uint32)
    image1_dt = np.array([[10] * 10], dtype=np.uint32)
    image2_gt = np.array([[1] * 10], dtype=np.uint32)
    image2_dt = np.array([[10] * 7 + [0] * 3], dtype=np.uint32)

    # Image 3: GT id=2 (cat=200, area=10) covers all 10 px.
    #          DT id=11 (cat=200, area=6) covers [0..6]; rest VOID.
    #          intersection=6, gt_area=10, dt_area=6, void=0,
    #          union=10, iou=0.6.
    image3_gt = np.array([[2] * 10], dtype=np.uint32)
    image3_dt = np.array([[11] * 6 + [0] * 4], dtype=np.uint32)

    gt = {1: image1_gt, 2: image2_gt, 3: image3_gt}
    dt = {1: image1_dt, 2: image2_dt, 3: image3_dt}
    gt_segs = {
        1: [{"id": 1, "category_id": 100, "iscrowd": False, "area": 10}],
        2: [{"id": 1, "category_id": 100, "iscrowd": False, "area": 10}],
        3: [{"id": 2, "category_id": 200, "iscrowd": False, "area": 10}],
    }
    dt_segs = {
        1: [{"id": 10, "category_id": 100, "iscrowd": False, "area": 10}],
        2: [{"id": 10, "category_id": 100, "iscrowd": False, "area": 7}],
        3: [{"id": 11, "category_id": 200, "iscrowd": False, "area": 6}],
    }
    cats = [
        {"id": 100, "isthing": True},
        {"id": 200, "isthing": False},
    ]

    oracle = _oracle_snapshot(gt, gt_segs, dt, dt_segs, cats)
    vsnap = _vernier_snapshot(gt, gt_segs, dt, dt_segs, cats, "corrected")
    assert_snapshots_equal(oracle, vsnap)

    # Sanity: cat 100 has 2 TPs (iou 1.0 + 0.7), cat 200 has 1 TP
    # (iou 0.6). SQ_100 = 1.7/2 = 0.85; SQ_200 = 0.6.
    # mean(SQ) = (0.85 + 0.6) / 2 = 0.725.
    # pooled = (1.7 + 0.6) / (2 + 1) = 2.3 / 3 ≈ 0.7667.
    # The two differ; the oracle reports 0.725 (mean), confirmed by
    # bit-equality.
    assert abs(oracle.sq - 0.725) < 1e-12, oracle.sq
    pooled = (1.7 + 0.6) / 3.0
    assert abs(oracle.sq - pooled) > 0.01, "Q5 fixture must show mean ≠ pooled"


def _label_map_to_png_bytes(label_map: np.ndarray) -> bytes:
    """Encode a `(H, W) uint32` panoptic label map as RGB PNG bytes
    via the rgb2id convention. Round-trippable through both
    `vernier.panoptic.decode_label_map_png` (Pillow path) and
    `submit_png` / `update_png` (Rust path)."""
    rgb = id2rgb(label_map)
    buf = io.BytesIO()
    PILImage.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_submit_png_matches_array_path_perfect_match() -> None:
    """The new ``submit_png`` / ``update_png`` decode-in-Rust path
    must produce a byte-identical :class:`PanopticSummary` to the
    array path for the same fixture. Same fold, same kernel, same
    parity mode — diff is wall-time, not correctness.
    """
    gt_lm = np.array([[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]], dtype=np.uint32)
    dt_lm = np.array([[10, 10, 10, 10, 10, 11, 11, 11, 11, 11]], dtype=np.uint32)
    gt_segs = [
        {"id": 1, "category_id": 100, "iscrowd": False, "area": 5},
        {"id": 2, "category_id": 200, "iscrowd": False, "area": 5},
    ]
    dt_segs = [
        {"id": 10, "category_id": 100, "iscrowd": False, "area": 5},
        {"id": 11, "category_id": 200, "iscrowd": False, "area": 5},
    ]
    cats = [
        {"id": 100, "isthing": True},
        {"id": 200, "isthing": False},
    ]
    cats_bytes = json.dumps(cats).encode()
    gt_segs_bytes = json.dumps(gt_segs).encode()
    dt_segs_bytes = json.dumps(dt_segs).encode()
    gt_png = _label_map_to_png_bytes(gt_lm)
    dt_png = _label_map_to_png_bytes(dt_lm)

    # Array path: pre-decoded uint32 arrays via update.
    ev_array = StreamingPanopticEvaluator(cats_bytes, "strict")
    ev_array.update(1, gt_lm, gt_segs_bytes, dt_lm, dt_segs_bytes)
    summary_array = ev_array.finalize()

    # PNG path: raw bytes via update_png; Rust does decode + RGB→id +
    # S3 area fold + S1/S11 validation in one pass.
    ev_png = StreamingPanopticEvaluator(cats_bytes, "strict")
    ev_png.update_png(1, gt_png, gt_segs_bytes, dt_png, dt_segs_bytes)
    summary_png = ev_png.finalize()

    snap_array = summary_to_snapshot(summary_array)
    snap_png = summary_to_snapshot(summary_png)
    assert_snapshots_equal(snap_array, snap_png)


def test_submit_png_matches_array_path_with_void_and_jitter() -> None:
    """Same equivalence under non-trivial coverage: VOID pixels (S6
    union subtraction site), DT-side area marginals (S3), and a
    mismatched DT segment that drives a TP via the U6 panoptic union
    rather than a literal IoU.
    """
    # 1x4: GT covers pixel 0 as id=1; pixels 1..3 are VOID. DT covers
    # all 4 as id=10. U6 union = 1 + 4 - 1 - 3 = 1; IoU = 1.0 → TP.
    gt_lm = np.array([[1, 0, 0, 0]], dtype=np.uint32)
    dt_lm = np.array([[10, 10, 10, 10]], dtype=np.uint32)
    gt_segs = [{"id": 1, "category_id": 100, "iscrowd": False, "area": 1}]
    # Lying JSON DT area; S3 should overwrite to 4 on both paths.
    dt_segs = [{"id": 10, "category_id": 100, "iscrowd": False, "area": 99_999}]
    cats = [{"id": 100, "isthing": True}]
    cats_bytes = json.dumps(cats).encode()
    gt_segs_bytes = json.dumps(gt_segs).encode()
    dt_segs_bytes = json.dumps(dt_segs).encode()
    gt_png = _label_map_to_png_bytes(gt_lm)
    dt_png = _label_map_to_png_bytes(dt_lm)

    # Single-thing fixture: switch to corrected mode to skip the W6
    # strict-mode raise on the empty stuff bucket. Equivalence between
    # array and PNG paths is the property under test, not strict-W6.
    ev_array = StreamingPanopticEvaluator(cats_bytes, "corrected")
    ev_array.update(1, gt_lm, gt_segs_bytes, dt_lm, dt_segs_bytes)
    summary_array = ev_array.finalize()

    ev_png = StreamingPanopticEvaluator(cats_bytes, "corrected")
    ev_png.update_png(1, gt_png, gt_segs_bytes, dt_png, dt_segs_bytes)
    summary_png = ev_png.finalize()

    snap_array = summary_to_snapshot(summary_array)
    snap_png = summary_to_snapshot(summary_png)
    assert_snapshots_equal(snap_array, snap_png)
    # Sanity: the U6 + S3 fixture lands a perfect TP in both paths.
    assert snap_png.pq == 1.0
    assert snap_array.pq == 1.0


def test_submit_png_matches_oracle_strict() -> None:
    """End-to-end strict-tier: ``submit_png`` reproduces panopticapi's
    ``pq_compute_single_core`` bit-exactly on the perfect-match
    fixture from :func:`test_perfect_match_strict_bit_equal`. Pins
    the new entry point against the oracle directly, not just
    against the array path."""
    gt = {1: np.array([[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]], dtype=np.uint32)}
    dt = {1: np.array([[10, 10, 10, 10, 10, 11, 11, 11, 11, 11]], dtype=np.uint32)}
    gt_segs = {
        1: [
            {"id": 1, "category_id": 100, "iscrowd": False, "area": 5},
            {"id": 2, "category_id": 200, "iscrowd": False, "area": 5},
        ]
    }
    dt_segs = {
        1: [
            {"id": 10, "category_id": 100, "iscrowd": False, "area": 5},
            {"id": 11, "category_id": 200, "iscrowd": False, "area": 5},
        ]
    }
    cats = [
        {"id": 100, "isthing": True},
        {"id": 200, "isthing": False},
    ]
    oracle = _oracle_snapshot(gt, gt_segs, dt, dt_segs, cats)

    cats_bytes = json.dumps(cats).encode()
    ev = StreamingPanopticEvaluator(cats_bytes, "strict")
    for image_id, gt_lm in gt.items():
        dt_lm = dt[image_id]
        gt_png = _label_map_to_png_bytes(gt_lm)
        dt_png = _label_map_to_png_bytes(dt_lm)
        ev.update_png(
            int(image_id),
            gt_png,
            json.dumps(gt_segs[image_id]).encode(),
            dt_png,
            json.dumps(dt_segs[image_id]).encode(),
        )
    snap = summary_to_snapshot(ev.finalize())
    assert_snapshots_equal(oracle, snap)
    assert snap.pq == 1.0
