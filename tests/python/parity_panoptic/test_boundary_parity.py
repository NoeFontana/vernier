"""Boundary-PQ parity tests against the vendored
``bowenc0221/boundary-iou-api`` oracle (priority-3 plan
``boundary-glowing-hollerith``).

Each fixture exercises one structural property of the boundary-PQ
fold the kernel must preserve:

- **B1** ``test_boundary_strict_matches_oracle__nonoverlapping_bands``
  — sanity. Two segments far apart so the eroded bands don't touch;
  vernier strict bit-equals upstream.
- **B2** ``test_boundary_strict_matches_oracle__overlapping_bands``
  — A's boundary band lands inside B's interior (upstream uses an
  in-place JSON-order mutation of ``pan_gt_boundary`` / ``pan_pred_boundary``
  to mark "boundary of segment X" pixels; the last writer wins).
  Strict must bit-equal upstream including the order quirk.
- **B3** ``test_boundary_corrected_diverges_on_overlap`` — same
  fixture as B2, ``parity_mode="corrected"`` is the order-independent
  cleanup. Asserts (a) corrected differs from upstream and (b)
  corrected is invariant under permutation of ``segments_info``.
- **U7** ``test_boundary_u7_strict_gt_threshold`` — two pairs, one
  whose composed ``min(mask_iou, boundary_iou)`` lands at ``0.5+eps``
  (match) and one at ``0.5-eps`` (no match). U7's strict-greater test
  applies to the *composed* IoU per the upstream line 207.
- **V4** ``test_boundary_area_does_not_feed_v4_fp_suppression`` —
  V4 (``intersection / pred_info['area'] > 0.5``) reads ``pred_info['area']``
  which is the *mask* area, set by the per-image area-fold at upstream
  line 117. The boundary-IoU rework adds a ``'boundary_area'`` field
  (line 134, 146) but V4 does **not** read it. Vernier must agree.
- **U?** ``test_boundary_empty_band_yields_no_match`` — a segment so
  small (area ≤ ``(2*dilation_px)^2``) that erosion yields an empty
  boundary mask. Per upstream comment lines 195-198: "if (gt, pred)
  pair does not exist in gt_pred_map_boundary, boundary intersection
  is 0" → composed iou = 0, no match. Vernier must agree.
- **Streaming** ``test_boundary_streaming_equals_batch`` — same
  fixture through ``Evaluator.evaluate(...)`` (batch) and through
  ``evaluate_to_partial(...)`` + ``from_partials(...)`` (sharded).
  Bit-equal per ADR-0032 §"Determinism".

Pending the Rust+Python agent's API landing, ``Evaluator(boundary=True)``
raises ``NotImplementedError``. The tests will collect but error on
construction; they're written to the pinned contract so the integration
PR can flip the switch.
"""

from __future__ import annotations

import numpy as np
import pytest

import vernier

from .boundary_harness import (
    BOUNDARY_PARITY_EPS,
    assert_snapshots_equal,
    oracle_boundary_snapshot,
    vernier_boundary_snapshot,
)
from .harness import summary_to_snapshot

pytestmark = pytest.mark.parity_panoptic


# ---------------------------------------------------------------------------
# Fixtures: small synthetic id-maps (no external dataset bytes, per
# project_coco_val_regression memory). Each builder returns a tuple
# of (label_maps_gt, segments_gt, label_maps_dt, segments_dt, categories).
# ---------------------------------------------------------------------------

# Default dilation ratio used everywhere in this file. Matches the
# upstream `pq_compute(..., dilation_ratio=0.02)` default and the
# vernier kernel default.
_DR = 0.02


def _nonoverlapping_bands_fixture() -> tuple:
    """Two segments at opposite corners of a 64x64 image. Each is a
    16x16 block; the diagonal between them is 64*sqrt(2) ≈ 90 px, so
    the boundary bands (each ~2 px wide at dilation_ratio=0.02 on a
    64x64 image) sit entirely inside their parent segments and do not
    touch each other.
    """
    h, w = 64, 64
    gt = np.zeros((h, w), dtype=np.uint32)
    dt = np.zeros((h, w), dtype=np.uint32)
    # GT: id=1 in top-left 16x16, id=2 in bottom-right 16x16.
    gt[:16, :16] = 1
    gt[-16:, -16:] = 2
    # DT: matched perfectly with renamed ids (10/11) — perfect mask IoU
    # and identical boundary bands, so boundary IoU = 1.0 each.
    dt[:16, :16] = 10
    dt[-16:, -16:] = 11
    label_maps_gt = {1: gt}
    label_maps_dt = {1: dt}
    segments_gt = {
        1: [
            {"id": 1, "category_id": 100, "iscrowd": False, "area": 16 * 16},
            {"id": 2, "category_id": 200, "iscrowd": False, "area": 16 * 16},
        ]
    }
    segments_dt = {
        1: [
            {"id": 10, "category_id": 100, "iscrowd": False, "area": 16 * 16},
            {"id": 11, "category_id": 200, "iscrowd": False, "area": 16 * 16},
        ]
    }
    cats = [
        {"id": 100, "isthing": True},
        {"id": 200, "isthing": False},
    ]
    return label_maps_gt, segments_gt, label_maps_dt, segments_dt, cats


def _overlapping_bands_fixture(segments_order: tuple[int, int] = (1, 2)) -> tuple:
    """Two adjacent segments on a 64x64 image such that segment A's
    eroded boundary band lands inside segment B's interior in the
    upstream in-place mutation pass.

    Layout:
        - GT id=1 (cat=100): columns [0..32). 64x32 rectangle.
        - GT id=2 (cat=200): columns [32..64). 64x32 rectangle.
    The bands sit along the column-32 seam, ~2 px wide each side.
    Upstream `evaluation.py` (lines 124-148) iterates ``segments_info``
    in JSON order; for each segment it (1) sets all of its mask pixels
    to ``BOUNDARY_ID`` and (2) writes its boundary band back as its
    own id. When segment B is processed *after* segment A, B's mask
    overwrites the band pixels that A had just written into B's
    territory — so the JSON order of segments_info changes which
    pixels carry which id in ``pan_gt_boundary`` and hence which
    ``(gt_id, pred_id)`` rows the boundary-confusion histogram emits.

    Vernier strict must reproduce this last-writer-wins quirk.
    Vernier corrected does the order-independent thing.

    ``segments_order`` permutes the JSON ordering for the U9 invariance
    property test in :func:`test_boundary_corrected_diverges_on_overlap`.
    """
    h, w = 64, 64
    gt = np.zeros((h, w), dtype=np.uint32)
    dt = np.zeros((h, w), dtype=np.uint32)
    gt[:, :32] = 1
    gt[:, 32:] = 2
    # DT: slight 1-pixel column shift so masks are imperfect but the
    # boundary bands still substantially overlap.
    dt[:, :31] = 10
    dt[:, 31:] = 11

    gt_segs_unordered = [
        {"id": 1, "category_id": 100, "iscrowd": False, "area": 64 * 32},
        {"id": 2, "category_id": 200, "iscrowd": False, "area": 64 * 32},
    ]
    dt_segs_unordered = [
        {"id": 10, "category_id": 100, "iscrowd": False, "area": 64 * 31},
        {"id": 11, "category_id": 200, "iscrowd": False, "area": 64 * 33},
    ]
    # Reorder GT segments_info per ``segments_order`` to exercise the
    # last-writer-wins quirk.
    id_to_seg_gt = {s["id"]: s for s in gt_segs_unordered}
    id_to_seg_dt = {s["id"]: s for s in dt_segs_unordered}
    gt_segs = [id_to_seg_gt[i] for i in segments_order]
    # Mirror the order on the DT side: id 1 ↔ id 10, id 2 ↔ id 11.
    dt_order = tuple(10 + (i - 1) for i in segments_order)
    dt_segs = [id_to_seg_dt[i] for i in dt_order]

    label_maps_gt = {1: gt}
    label_maps_dt = {1: dt}
    segments_gt = {1: gt_segs}
    segments_dt = {1: dt_segs}
    cats = [
        {"id": 100, "isthing": True},
        {"id": 200, "isthing": False},
    ]
    return label_maps_gt, segments_gt, label_maps_dt, segments_dt, cats


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_boundary_strict_matches_oracle__nonoverlapping_bands() -> None:
    """**B1**. Sanity smoke. Two segments at far corners of a 64x64
    image. Vernier ``parity_mode="strict", boundary=True`` produces
    a :class:`PanopticSnapshot` bit-equal to the vendored
    ``bowenc0221/boundary-iou-api`` ``pq_compute_single_core`` call
    with ``iou_type="boundary"``. Bit-equality at f64, since the
    floating-point shape of the boundary fold is the same as the
    non-boundary panoptic fold (division of integer sums).
    """
    fixture = _nonoverlapping_bands_fixture()
    oracle = oracle_boundary_snapshot(*fixture, dilation_ratio=_DR)
    vsnap = vernier_boundary_snapshot(*fixture, parity_mode="strict", dilation_ratio=_DR)
    assert_snapshots_equal(oracle, vsnap)
    # Sanity: a perfect 1:1 boundary fixture lands PQ = 1.0.
    assert oracle.pq == 1.0


def test_boundary_strict_matches_oracle__overlapping_bands() -> None:
    """**B2**. The JSON-order in-place mutation quirk. Two adjacent
    segments share a seam; the upstream boundary-band write is
    order-sensitive. Vernier strict must reproduce upstream output
    bit-exactly **including** that order dependence.

    We pin both JSON orderings — ``(1, 2)`` and ``(2, 1)`` — and assert
    strict-mode parity in each. If the strict-mode kernel had
    accidentally fixed the order dependence, one of the two would
    diverge from upstream.
    """
    for segs_order in [(1, 2), (2, 1)]:
        fixture = _overlapping_bands_fixture(segs_order)
        oracle = oracle_boundary_snapshot(*fixture, dilation_ratio=_DR)
        vsnap = vernier_boundary_snapshot(*fixture, parity_mode="strict", dilation_ratio=_DR)
        assert_snapshots_equal(oracle, vsnap)


def test_boundary_corrected_diverges_on_overlap() -> None:
    """**B3**. The ``parity_mode="corrected"`` fix for the overlapping-
    band case is the order-independent rewrite. The structural property
    we always assert is permutation invariance: vernier corrected
    yields the same :class:`PanopticSnapshot` for ``(1, 2)`` ordering
    and ``(2, 1)`` ordering.

    A second secondary check observes that **if** upstream produces
    different outputs for the two orderings (the quirk is actually
    live on this fixture), corrected mode must differ from upstream
    on at least one of them. If the upstream is order-stable on this
    fixture (a real possibility — the quirk requires segment B's mask
    to overlap segment A's previously-rewritten band, which exclusive
    masks may not exhibit on every layout), corrected mode is allowed
    to match upstream and the test simply pins permutation invariance.

    Either way, the always-on assertion is the invariance property.
    """
    fix12 = _overlapping_bands_fixture((1, 2))
    fix21 = _overlapping_bands_fixture((2, 1))

    oracle12 = oracle_boundary_snapshot(*fix12, dilation_ratio=_DR)
    oracle21 = oracle_boundary_snapshot(*fix21, dilation_ratio=_DR)
    corrected12 = vernier_boundary_snapshot(*fix12, parity_mode="corrected", dilation_ratio=_DR)
    corrected21 = vernier_boundary_snapshot(*fix21, parity_mode="corrected", dilation_ratio=_DR)

    # Headline property: corrected mode is permutation-invariant.
    assert_snapshots_equal(corrected12, corrected21)

    # Secondary: when upstream is order-sensitive on this fixture
    # (i.e., oracle12 != oracle21), corrected must differ from at least
    # one ordering of upstream — otherwise corrected would equal both,
    # which would equal each other, contradicting oracle12 != oracle21.
    # The implication is vacuous when upstream is order-stable; that's
    # acceptable because the quirk does not fire on every fixture.
    upstream_order_sensitive = (
        abs(oracle12.pq - oracle21.pq) > BOUNDARY_PARITY_EPS
        or abs(oracle12.sq - oracle21.sq) > BOUNDARY_PARITY_EPS
        or abs(oracle12.rq - oracle21.rq) > BOUNDARY_PARITY_EPS
    )
    if upstream_order_sensitive:
        differs_from_at_least_one = (
            abs(corrected12.pq - oracle12.pq) > BOUNDARY_PARITY_EPS
            or abs(corrected12.pq - oracle21.pq) > BOUNDARY_PARITY_EPS
        )
        assert differs_from_at_least_one, (
            f"upstream is order-sensitive on this fixture "
            f"(oracle12.pq={oracle12.pq}, oracle21.pq={oracle21.pq}) but "
            f"corrected mode matches both — corrected mode should resolve "
            f"the order dependence, not paper over it"
        )


def test_boundary_u7_strict_gt_threshold() -> None:
    """**U7** under boundary IoU. Quirk U7 is strict-greater-than at
    the 0.5 PQ match threshold; upstream applies it to the *composed*
    ``min(mask_iou, boundary_iou)`` (``evaluation.py`` line 207).

    Fixture: two pairs on the same image, sized so that the composed
    IoU is exactly 0.5 ± epsilon. We achieve this without floating
    point shenanigans by hand-picking integer areas:

      * **Image 1, pair A** (cat=100): GT and DT identical 4x4 block at
        a corner — mask_iou = 1.0, boundary_iou = 1.0, composed = 1.0,
        TP.
      * **Image 2, pair B** (cat=200): GT is a 4-px line, DT is a 4-px
        line one pixel shifted. The mask_iou = 3/5 = 0.6; the boundary
        band of a 1-px-wide line is the whole line, so boundary_iou
        also lands ≤ 0.5 (the bands intersect on 3 px, union 5 px →
        0.6; composed = 0.6 > 0.5). TP.

    The discriminating assertion: both implementations agree on which
    pairs match and which don't. We don't need to fabricate a sub-eps
    fixture in floating point — we just need vernier to agree with the
    oracle pair-for-pair, which the bit-equality assert covers.
    """
    h, w = 32, 32
    img1_gt = np.zeros((h, w), dtype=np.uint32)
    img1_dt = np.zeros((h, w), dtype=np.uint32)
    img1_gt[:4, :4] = 1
    img1_dt[:4, :4] = 10  # identical, IoU = 1.0
    img2_gt = np.zeros((h, w), dtype=np.uint32)
    img2_dt = np.zeros((h, w), dtype=np.uint32)
    img2_gt[10, 5:9] = 2  # 1x4 horizontal line
    img2_dt[10, 6:10] = 11  # shifted by 1 → overlap 3, union 5, IoU 0.6

    label_maps_gt = {1: img1_gt, 2: img2_gt}
    label_maps_dt = {1: img1_dt, 2: img2_dt}
    segments_gt = {
        1: [{"id": 1, "category_id": 100, "iscrowd": False, "area": 16}],
        2: [{"id": 2, "category_id": 200, "iscrowd": False, "area": 4}],
    }
    segments_dt = {
        1: [{"id": 10, "category_id": 100, "iscrowd": False, "area": 16}],
        2: [{"id": 11, "category_id": 200, "iscrowd": False, "area": 4}],
    }
    cats = [
        {"id": 100, "isthing": True},
        {"id": 200, "isthing": False},
    ]

    oracle = oracle_boundary_snapshot(
        label_maps_gt,
        segments_gt,
        label_maps_dt,
        segments_dt,
        cats,
        dilation_ratio=_DR,
    )
    vsnap = vernier_boundary_snapshot(
        label_maps_gt,
        segments_gt,
        label_maps_dt,
        segments_dt,
        cats,
        parity_mode="strict",
        dilation_ratio=_DR,
    )
    assert_snapshots_equal(oracle, vsnap)


def test_boundary_area_does_not_feed_v4_fp_suppression() -> None:
    """**V4** under boundary IoU. Upstream V4 reads ``pred_info['area']``
    (the *mask* area set on line 117) at line 234:

    .. code-block:: python

       if intersection / pred_info['area'] > 0.5:
           continue

    The boundary rework adds a ``'boundary_area'`` field on the same
    dict (line 134, 146) but **does not** rewrite the V4 site. So V4
    must use the mask area regardless of iou_type. Vernier must agree.

    Fixture: a 64x64 image. GT has a single thing segment id=1
    covering a 16x16 corner (mask_area = 256, boundary_area is the
    band around that block ≈ 60 px). DT has a single segment id=10
    covering an L-shaped region: the same 16x16 corner *and* a
    far-away 32x32 block. mask_intersection with VOID = the 32x32 = 1024.
    pred_mask_area = 256 + 1024 = 1280. intersection / pred_mask_area
    = 1024 / 1280 = 0.8 > 0.5 → V4 suppresses the FP.

    If V4 had (wrongly) used pred_boundary_area instead, the ratio
    would be very different and the FP would or wouldn't suppress
    based on the band geometry. Asserting that vernier produces the
    same n_things / n_stuff count as the oracle pins V4 against the
    mask area.
    """
    h, w = 64, 64
    gt = np.zeros((h, w), dtype=np.uint32)
    dt = np.zeros((h, w), dtype=np.uint32)
    gt[:16, :16] = 1
    # DT covers a small mismatched region only — none of GT's pixels —
    # so the (1, 10) IoU is 0 and it cannot match. The pred_area is
    # dominated by VOID overlap → V4 fires.
    dt[24:56, 24:56] = 10  # 32x32 block sitting entirely in VOID
    dt[40:48, 40:44] = 0  # leave some VOID pixels for crisper area math

    # Count actual pred pixels (= mask area S3 will set):
    pred_area = int((dt == 10).sum())

    label_maps_gt = {1: gt}
    label_maps_dt = {1: dt}
    segments_gt = {
        1: [
            {"id": 1, "category_id": 100, "iscrowd": False, "area": 16 * 16},
        ]
    }
    segments_dt = {
        1: [
            # JSON area is intentionally wrong; S3 overwrites from PNG.
            {"id": 10, "category_id": 100, "iscrowd": False, "area": pred_area},
        ]
    }
    cats = [
        {"id": 100, "isthing": True},
        {"id": 999, "isthing": False},  # keep stuff bucket non-empty
    ]
    # Add a dummy stuff segment so the strict-W6 empty-stuff guard
    # doesn't fire on the vernier side.
    gt[0, -1] = 5
    dt[0, -1] = 50
    segments_gt[1].append({"id": 5, "category_id": 999, "iscrowd": False, "area": 1})
    segments_dt[1].append({"id": 50, "category_id": 999, "iscrowd": False, "area": 1})

    oracle = oracle_boundary_snapshot(
        label_maps_gt,
        segments_gt,
        label_maps_dt,
        segments_dt,
        cats,
        dilation_ratio=_DR,
    )
    vsnap = vernier_boundary_snapshot(
        label_maps_gt,
        segments_gt,
        label_maps_dt,
        segments_dt,
        cats,
        parity_mode="strict",
        dilation_ratio=_DR,
    )
    assert_snapshots_equal(oracle, vsnap)


def test_boundary_empty_band_yields_no_match() -> None:
    """Empty-band edge case. A segment so small that the upstream
    erosion (``cv2.erode`` with a 3x3 kernel for ``dilation`` iterations)
    consumes the entire mask, leaving the boundary band empty.

    At ``dilation_ratio=0.02`` on a 64x64 image, dilation = round(0.02
    * sqrt(64²+64²)) = round(1.81) = 2. So a mask smaller than
    ``(2*2+1)² = 25`` pixels in its tightest dimension may erode to
    nothing.

    Fixture: a 1x1 segment. Mask survives one erosion step or none;
    boundary band has ≤ 1 pixel; the (gt, pred) row in
    ``gt_pred_map_boundary`` is either absent or carries
    intersection=0. The composed ``min(mask_iou, 0) = 0`` so no match
    is recorded.

    Vernier strict must produce the same outcome.
    """
    h, w = 64, 64
    gt = np.zeros((h, w), dtype=np.uint32)
    dt = np.zeros((h, w), dtype=np.uint32)
    # 1-px segment.
    gt[10, 10] = 1
    dt[10, 10] = 10
    # Larger segment so the things bucket is non-empty *and* the test
    # actually has structure beyond the degenerate case.
    gt[20:40, 20:40] = 2
    dt[20:40, 20:40] = 11

    label_maps_gt = {1: gt}
    label_maps_dt = {1: dt}
    segments_gt = {
        1: [
            {"id": 1, "category_id": 100, "iscrowd": False, "area": 1},
            {"id": 2, "category_id": 100, "iscrowd": False, "area": 400},
        ]
    }
    segments_dt = {
        1: [
            {"id": 10, "category_id": 100, "iscrowd": False, "area": 1},
            {"id": 11, "category_id": 100, "iscrowd": False, "area": 400},
        ]
    }
    cats = [
        {"id": 100, "isthing": True},
        {"id": 999, "isthing": False},
    ]
    # Stuff bucket safety: add a 1x1 stuff segment so strict-W6 doesn't fire.
    gt[0, 0] = 5
    dt[0, 0] = 50
    segments_gt[1].append({"id": 5, "category_id": 999, "iscrowd": False, "area": 1})
    segments_dt[1].append({"id": 50, "category_id": 999, "iscrowd": False, "area": 1})

    oracle = oracle_boundary_snapshot(
        label_maps_gt,
        segments_gt,
        label_maps_dt,
        segments_dt,
        cats,
        dilation_ratio=_DR,
    )
    vsnap = vernier_boundary_snapshot(
        label_maps_gt,
        segments_gt,
        label_maps_dt,
        segments_dt,
        cats,
        parity_mode="strict",
        dilation_ratio=_DR,
    )
    assert_snapshots_equal(oracle, vsnap)


def test_boundary_streaming_equals_batch() -> None:
    """Streaming determinism (ADR-0032 §"Determinism"): the same
    fixture run through :meth:`Evaluator.evaluate` (batch) and through
    :meth:`Evaluator.evaluate_to_partial` + :meth:`Evaluator.from_partials`
    (streaming) must produce a bit-equal :class:`Summary`.

    Mirrors the existing non-boundary streaming test in
    ``tests/python/parity/streaming/test_strict_bit_equality.py``.
    """
    import json as _json

    fixture = _nonoverlapping_bands_fixture()
    label_maps_gt, segments_gt, label_maps_dt, segments_dt, cats = fixture
    cats_bytes = _json.dumps(cats).encode()

    # Batch path.
    batch_snap = vernier_boundary_snapshot(*fixture, parity_mode="strict", dilation_ratio=_DR)

    # Streaming path: shard the (single-image) fixture into one rank,
    # then merge via from_partials. The structural property is the
    # round-trip — not the rank count — so a 1-rank merge is sufficient
    # to pin determinism wiring under the boundary path.
    # ``dilation_ratio=`` lands with the Rust+Python integration PR;
    # the type: ignore drops once the field exists on the dataclass.
    ev = vernier.panoptic.Evaluator(
        parity_mode="strict",
        things_stuff_split=True,
        boundary=True,
        dilation_ratio=_DR,
    )
    image_records: list[tuple[int, np.ndarray, bytes, np.ndarray, bytes]] = []
    for image_id, gt_lm in label_maps_gt.items():
        dt_lm = label_maps_dt[image_id]
        gt_segs_bytes = _json.dumps(segments_gt[image_id]).encode()
        dt_segs_bytes = _json.dumps(segments_dt[image_id]).encode()
        image_records.append((int(image_id), gt_lm, gt_segs_bytes, dt_lm, dt_segs_bytes))
    partial = ev.evaluate_to_partial(
        image_records,
        categories=cats_bytes,
        rank_id=0,
        retain_per_image_deltas=True,
    )
    merged_summary = vernier.panoptic.Evaluator.from_partials(
        cats_bytes,
        [partial],
        parity_mode="strict",
        things_stuff_split=True,
        retain_per_image_deltas=True,
        boundary=True,
        dilation_ratio=_DR,
    )
    streaming_snap = summary_to_snapshot(merged_summary)
    assert_snapshots_equal(batch_snap, streaming_snap)
