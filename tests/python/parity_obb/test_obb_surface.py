"""Public-surface tests for oriented-box evaluation (ADR-0063)."""

from __future__ import annotations

import math

import pytest

from vernier._types import ParityMode
from vernier.instance import (
    DetectionsInput,
    Evaluator,
    IouKind,
    Quad,
    ResultAnnotation,
    RotatedBox,
    obb,
)

from . import fixtures as fx

PARITY_MODES: tuple[ParityMode, ...] = ("strict", "corrected")


def ap(
    iou: IouKind,
    dts: DetectionsInput,
    *,
    gt: bytes | None = None,
    parity_mode: ParityMode = "corrected",
) -> float:
    summary = Evaluator(iou=iou, parity_mode=parity_mode).evaluate(
        gt if gt is not None else fx.ground_truth(), dts
    )
    return summary.stats[0]


def _without(records: list[ResultAnnotation], key: str) -> list[ResultAnnotation]:
    """Drop one key from every record, keeping the TypedDict shape.

    `dict` comprehensions widen to `dict[str, object]`, which is not a
    `ResultAnnotation` as far as a strict checker is concerned — and the
    tests below are precisely about the surface being strict.
    """
    out: list[ResultAnnotation] = []
    for rec in records:
        copy = dict(rec)
        copy.pop(key, None)
        out.append(copy)  # pyright: ignore[reportArgumentType]
    return out


# --------------------------------------------------------------------------
# the kernels evaluate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("parity_mode", PARITY_MODES)
def test_perfect_rotated_box_detections_score_one(parity_mode):
    got = ap(
        RotatedBox(unit=fx.UNIT, rotation=fx.ROTATION),
        fx.detections(),
        parity_mode=parity_mode,
    )
    assert got == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("parity_mode", PARITY_MODES)
def test_perfect_quad_detections_score_one(parity_mode):
    got = ap(Quad(), fx.detections(geometry="quad"), parity_mode=parity_mode)
    assert got == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("parity_mode", PARITY_MODES)
def test_the_two_kernels_agree_on_the_same_geometry(parity_mode):
    """A rotated box and its own corners are the same shape, so the two
    kernels must agree — through entirely separate code: four
    axis-aligned clip stages against general half-planes, and two
    different strict oracles underneath."""
    jittered = fx.jitter()
    rb = ap(
        RotatedBox(unit=fx.UNIT, rotation=fx.ROTATION),
        fx.detections(jittered),
        parity_mode=parity_mode,
    )
    q = ap(Quad(), fx.detections(jittered, geometry="quad"), parity_mode=parity_mode)
    assert rb == pytest.approx(q, abs=1e-6)
    assert 0.0 < rb < 1.0, "the jittered scene should be a partial match"


def test_strict_and_corrected_differ_but_only_slightly():
    """detectron2 computes in f32 and vernier's own kernel in f64, so
    they cannot agree to the bit — and must agree to f32 resolution."""
    jittered = fx.jitter()
    dts = fx.detections(jittered)
    iou = RotatedBox(unit=fx.UNIT, rotation=fx.ROTATION)
    strict = ap(iou, dts, parity_mode="strict")
    corrected = ap(iou, dts, parity_mode="corrected")
    assert strict == pytest.approx(corrected, abs=1e-5)


def test_a_ninety_degree_orientation_error_destroys_ap():
    """The failure IoU alone cannot see, seen.

    On `ELONGATED` rather than `SCENE`, and the difference is the point:
    `SCENE` contains a square, and a square rotated a quarter turn is
    the same square. It matches perfectly and drags the AP back up. The
    square stays in the default scene precisely because it is the shape
    an orientation bug hides behind.
    """
    turned = [[cx, cy, w, h, t + 90.0] for cx, cy, w, h, t in fx.ELONGATED]
    got = ap(
        RotatedBox(unit=fx.UNIT, rotation=fx.ROTATION),
        fx.detections(turned),
        gt=fx.ground_truth(fx.ELONGATED),
    )
    assert got == 0.0


# --------------------------------------------------------------------------
# the convention is required, and wrong ones are refused
# --------------------------------------------------------------------------


def test_rotated_box_requires_both_convention_fields():
    with pytest.raises(TypeError):
        RotatedBox()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        RotatedBox(unit="deg")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("unit", "rotation", "expected"),
    [
        ("turns", "screen_ccw", "unit"),
        ("deg", "widdershins", "rotation"),
    ],
)
def test_bad_convention_values_are_named(unit: str, rotation: str, expected: str) -> None:
    kernel = RotatedBox(unit=unit, rotation=rotation)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match=expected):
        ap(kernel, fx.detections())


def test_both_rotation_conventions_describe_the_same_scene():
    """A box at +30 under screen_cw is the same shape as one at -30
    under screen_ccw. The sign flip is exact, so the AP must match to
    the bit."""
    flipped = [[cx, cy, w, h, -t] for cx, cy, w, h, t in fx.SCENE]
    ccw = ap(
        RotatedBox(unit="deg", rotation="screen_ccw"),
        fx.detections(fx.jitter()),
    )
    cw = ap(
        RotatedBox(unit="deg", rotation="screen_cw"),
        fx.detections(fx.jitter(flipped), rotation="screen_cw"),
        gt=fx.ground_truth(flipped, rotation="screen_cw"),
    )
    assert ccw == cw


def test_radians_and_degrees_agree():
    rad_scene = [[cx, cy, w, h, math.radians(t)] for cx, cy, w, h, t in fx.SCENE]
    deg = ap(RotatedBox(unit="deg", rotation="screen_ccw"), fx.detections(fx.jitter()))
    rad = ap(
        RotatedBox(unit="rad", rotation="screen_ccw"),
        fx.detections(
            [
                [cx + 1.5, cy - 1.0, w * 0.97, h / 0.97, t + math.radians(2.0)]
                for cx, cy, w, h, t in rad_scene
            ]
        ),
        gt=fx.ground_truth(rad_scene),
    )
    assert deg == pytest.approx(rad, abs=1e-12)


# --------------------------------------------------------------------------
# the len-5 bbox trap
# --------------------------------------------------------------------------


def test_a_length_five_bbox_is_rejected_not_truncated():
    """The release's worst possible failure: detectron2 stores a rotated
    box in `bbox` as five numbers, and every axis-aligned reader takes
    the first four and evaluates a box at the wrong place, silently."""
    bad: list[ResultAnnotation] = []
    for rec, rb in zip(fx.detections(), fx.SCENE, strict=True):
        widened = dict(rec)
        widened.pop("rbox", None)
        widened["bbox"] = [*fx.envelope(rb), 30.0]
        bad.append(widened)  # type: ignore[arg-type]
    with pytest.raises((ValueError, TypeError)):
        ap(RotatedBox(unit=fx.UNIT, rotation=fx.ROTATION), bad)


def test_a_missing_rbox_names_the_field_and_the_trap():
    bare = _without(fx.detections(), "rbox")
    with pytest.raises(ValueError, match="rbox") as excinfo:
        ap(RotatedBox(unit=fx.UNIT, rotation=fx.ROTATION), bare)
    assert "length-5" in str(excinfo.value), "the message must warn about the len-5 bbox trap"


def test_a_missing_quad_names_the_field():
    bare = _without(fx.detections(geometry="quad"), "quad")
    with pytest.raises(ValueError, match="quad"):
        ap(Quad(), bare)


# --------------------------------------------------------------------------
# degenerate geometry
# --------------------------------------------------------------------------


def test_a_zero_area_quad_is_a_typed_error_in_both_modes():
    """DOTA_devkit evaluates 0/0 here. A NaN in the similarity matrix
    corrupts every match in the cell, so both modes refuse."""
    collinear = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    dts = fx.detections(geometry="quad")
    dts[0]["quad"] = collinear
    for mode in PARITY_MODES:
        with pytest.raises(ValueError, match=r"quad|area"):
            ap(Quad(), dts, parity_mode=mode)


def test_a_self_intersecting_quad_is_mode_dependent():
    """`corrected` refuses a bowtie; `strict` accepts it, because
    reproducing the oracle is what `strict` is for."""
    bowtie = [0.0, 0.0, 4.0, 4.0, 4.0, 0.0, 0.0, 1.0]
    dts = fx.detections(geometry="quad")
    dts[0]["quad"] = bowtie
    with pytest.raises(ValueError, match="self-intersecting"):
        ap(Quad(), dts, parity_mode="corrected")
    # No exception: the DK replica consumes it as submitted.
    ap(Quad(), dts, parity_mode="strict")


# --------------------------------------------------------------------------
# paths that deliberately do not support oriented kernels
# --------------------------------------------------------------------------


def test_streaming_says_which_paths_do_work():
    ev = Evaluator(iou=RotatedBox(unit=fx.UNIT, rotation=fx.ROTATION))
    with pytest.raises(NotImplementedError, match="evaluate") as excinfo:
        ev.background(fx.ground_truth())
    assert "manifest" in str(excinfo.value), "the error must name the paths that do work"


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------


def test_to_quad_round_trips_through_the_quad_kernel():
    """The conversion is the corrected kernel's own corner bits, so a
    converted scene must evaluate identically under either kernel."""
    for rb in fx.SCENE:
        assert obb.to_quad(rb, unit=fx.UNIT, rotation=fx.ROTATION) == pytest.approx(
            fx.corners(rb), abs=1e-12
        )


def test_min_area_rect_recovers_a_rectangle_and_refuses_a_line():
    rect = obb.min_area_rect(
        fx.corners([10.0, 20.0, 8.0, 3.0, 25.0]), unit=fx.UNIT, rotation=fx.ROTATION
    )
    assert rect is not None
    assert rect[0] == pytest.approx(10.0, abs=1e-9)
    assert rect[1] == pytest.approx(20.0, abs=1e-9)
    assert rect[2] * rect[3] == pytest.approx(24.0, abs=1e-9)
    assert (
        obb.min_area_rect(
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            unit=fx.UNIT,
            rotation=fx.ROTATION,
        )
        is None
    )


def test_label_ceiling_is_one_for_rectangles_and_below_one_otherwise():
    rects = [fx.corners(rb) for rb in fx.SCENE]
    ceiling = obb.label_ceiling(rects, [1] * len(rects))
    assert ceiling[1].n_scored == len(rects)
    assert ceiling[1].mean_iou == pytest.approx(1.0, abs=1e-9)

    # A triangle written as a degenerate quad fills half its own
    # minimum-area rectangle.
    triangle = [0.0, 0.0, 4.0, 0.0, 2.0, 3.0, 2.0, 3.0]
    mixed = obb.label_ceiling([rects[0], triangle], [1, 2])
    assert mixed[1].mean_iou == pytest.approx(1.0, abs=1e-9)
    assert mixed[2].mean_iou == pytest.approx(0.5, abs=1e-9)


def test_angle_error_sees_what_iou_cannot():
    gt = [0.0, 0.0, 10.0, 2.0, 0.0]
    turned = [0.0, 0.0, 10.0, 2.0, 30.0]
    assert obb.angle_error_deg(gt, turned, unit=fx.UNIT, rotation=fx.ROTATION) == pytest.approx(
        30.0, abs=1e-9
    )
    # Invariant to the parameterization: same box, written two ways.
    reparameterized = [0.0, 0.0, 2.0, 10.0, 90.0]
    assert obb.angle_error_deg(
        gt, reparameterized, unit=fx.UNIT, rotation=fx.ROTATION
    ) == pytest.approx(0.0, abs=1e-9)


def test_convention_check_finds_a_mislabeled_unit():
    """The whole point: a mislabeled convention is silent, and this is
    what makes it loud.

    The data is in degrees and the caller declares radians — the most
    common form of the bug, and the one with the loudest signal, since
    a degree read as a radian is a nonsense angle.
    """
    gt = fx.ground_truth(fx.ELONGATED)
    # Detections offset enough that the IoU sits in the sensitive range;
    # a near-perfect detection scores 1.0 under every hypothesis and
    # tells you nothing.
    dts = fx.detections([[cx + 9.0, cy, w, h, t] for cx, cy, w, h, t in fx.ELONGATED])

    with pytest.warns(UserWarning, match="convention"):
        report = obb.convention_check(gt, dts, unit="rad", rotation="screen_ccw")

    assert not report.declared_is_best
    assert report.best.unit == "deg"
    assert len(report.ranked) == 4
    assert report.ranked[0].ap50 > report.declared.ap50


def test_convention_check_reports_the_rotation_axis_even_when_it_is_blind():
    """The rotation sign is not always observable, and the report says
    so by the numbers rather than by pretending otherwise.

    Reinterpreting `sigma` mirrors *both* the ground truth and the
    detection about their own centers. When the configuration is itself
    mirror-symmetric — which a translation along a single axis is — the
    two hypotheses score identically and there is nothing to detect.
    That is a fact about the geometry, not a gap in the check.
    """
    gt = fx.ground_truth(fx.ELONGATED)
    dts = fx.detections([[cx + 9.0, cy, w, h, t] for cx, cy, w, h, t in fx.ELONGATED])
    report = obb.convention_check(gt, dts, unit="deg", rotation="screen_ccw", warn=False)
    by_rotation = {h.rotation: h.ap50 for h in report.ranked if h.unit == "deg"}
    assert by_rotation["screen_cw"] == by_rotation["screen_ccw"]


def test_convention_check_is_quiet_when_the_declaration_is_right():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = obb.convention_check(
            fx.ground_truth(), fx.detections(), unit="deg", rotation="screen_ccw"
        )
    assert report.declared_is_best


def test_convention_check_rejects_an_unknown_convention_before_evaluating():
    """Four evaluations is four times the work, and the answer to a typo
    should not cost any of it — nor arrive as a bare ``StopIteration``
    from the lookup that follows them.
    """
    with pytest.raises(ValueError, match="unknown convention"):
        obb.convention_check(
            fx.ground_truth(),
            fx.detections(),
            unit="degrees",  # pyright: ignore[reportArgumentType]
            rotation="screen_ccw",
            warn=False,
        )
    with pytest.raises(ValueError, match="unknown convention"):
        obb.convention_check(
            fx.ground_truth(),
            fx.detections(),
            unit="deg",
            rotation="ccw",  # pyright: ignore[reportArgumentType]
            warn=False,
        )


def test_min_area_rect_refuses_non_finite_vertices():
    """A ``NaN`` vertex would make the hull's comparator intransitive.
    It is an input error, not the ``None`` that means "degenerate".
    """
    with pytest.raises(ValueError, match="quad"):
        obb.min_area_rect(
            [0.0, 0.0, float("nan"), 1.0, 2.0, 2.0, 3.0, 3.0],
            unit="deg",
            rotation="screen_ccw",
        )
    with pytest.raises(ValueError, match="quads\\[1\\]"):
        obb.label_ceiling(
            [fx.corners(fx.SCENE[0]), [0.0, 0.0, 1.0, float("inf"), 2.0, 2.0, 3.0, 3.0]],
            [1, 1],
            unit="deg",
            rotation="screen_ccw",
        )


def test_columnar_ground_truth_refuses_oriented_columns_rather_than_dropping_them():
    """``Dataset.from_arrays`` has no ``rbox`` / ``quad`` column, so a
    caller who passes one would otherwise get "GT id=1 has no `rbox`
    field" much later — blamed for a field they did supply.
    """
    from typing import cast

    import numpy as np

    from vernier._array_types import GtAnnotations, GtCategory, GtImages
    from vernier.instance import CocoDataset

    images: GtImages = {
        "id": np.array([1], dtype=np.int64),
        "width": np.array([1024], dtype=np.int64),
        "height": np.array([1024], dtype=np.int64),
    }
    rb = fx.SCENE[0]
    annotations: GtAnnotations = {
        "id": np.array([1], dtype=np.int64),
        "image_id": np.array([1], dtype=np.int64),
        "category_id": np.array([1], dtype=np.int64),
        "bbox": np.array([fx.envelope(rb)], dtype=np.float64),
        "area": np.array([rb[2] * rb[3]], dtype=np.float64),
        "iscrowd": np.array([0], dtype=np.int64),
    }
    categories: list[GtCategory] = [{"id": 1, "name": "ship"}]

    # The cast is the point of the test: `rbox` is deliberately *not* in
    # `GtAnnotations`, so a typed caller cannot write this by accident.
    # An untyped one can, and the runtime has to say so.
    with_rbox = cast(
        "GtAnnotations",
        {**annotations, "rbox": np.array([rb], dtype=np.float64)},
    )
    with pytest.raises(ValueError, match="does not carry oriented geometry"):
        CocoDataset.from_arrays(images, with_rbox, categories)

    with_quad = cast(
        "GtAnnotations",
        {**annotations, "quad": np.array([fx.corners(rb)], dtype=np.float64)},
    )
    with pytest.raises(ValueError, match="does not carry oriented geometry"):
        CocoDataset.from_arrays(images, with_quad, categories)

    # Without the column it still builds, so the guard is about the
    # dropped field and not about the route.
    assert CocoDataset.from_arrays(images, annotations, categories) is not None


def test_the_area_bucket_tracks_the_oriented_geometry_not_the_envelope():
    """Quirk **OB13**. A 60x15 box at 45 degrees has oriented area 900 —
    *small* — while its axis-aligned envelope is 2812, which is
    *medium*. A spurious detection of that shape must therefore count
    against ``AP_small``; bucketing it by the envelope would file it
    under ``AP_medium`` and leave ``AP_small`` perfect.

    The spurious detection is scored *above* the real one on purpose. A
    false positive ranked below every true positive costs nothing under
    COCO's max-precision interpolation, so ordering it last would make
    the test pass whichever rule is in force.
    """
    small = [100.0, 100.0, 30.0, 30.0, 0.0]  # area 900, axis-aligned
    spurious = [800.0, 800.0, 60.0, 15.0, 45.0]  # oriented 900, envelope 2812
    env = fx.envelope(spurious)
    assert spurious[2] * spurious[3] < 1024.0 < env[2] * env[3]

    gt = fx.ground_truth([small])
    dts = fx.detections([spurious, small])
    summary = Evaluator(iou=RotatedBox(unit="deg", rotation="screen_ccw")).evaluate(gt, dts)
    # Stat 3 of the canonical 12 is AP@[.5:.95] for small objects: one
    # TP behind one FP, so 0.5. Bucketing by the envelope would file the
    # FP under medium and leave this at 1.0.
    assert summary.stats[3] == pytest.approx(0.5), (
        f"the spurious detection must land in the small bucket, got {summary.stats[3]}"
    )
