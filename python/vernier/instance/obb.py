"""Oriented-box diagnostics (ADR-0063).

None of this is evaluation. These answer the questions that come up
*around* an oriented-box evaluation, and they exist because the answers
are otherwise expensive to get wrong:

- :func:`convention_check` — "is my angle convention right?" It is the
  most common real-world OBB bug, and the failure is silent: a wrong
  unit or rotation direction produces plausible numbers rather than an
  error.
- :func:`label_ceiling` — "what does my label format cost me?" Quad
  ground truth against a rectangle-predicting detector has a ceiling
  that belongs to the dataset, not the model.
- :func:`to_quad` / :func:`min_area_rect` — format conversion.
- :func:`angle_error_deg` — "was this detection misplaced, or just
  misoriented?" IoU cannot tell you.

Everything here routes through the same ``vernier-geom`` kernels the
evaluator uses, so a diagnostic cannot drift from the metric it is
diagnosing.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple

from vernier._core import (
    obb_angle_error_deg as _angle_error_deg,
)
from vernier._core import (
    obb_label_ceiling as _label_ceiling,
)
from vernier._core import (
    obb_min_area_rect as _min_area_rect,
)
from vernier._core import (
    obb_rbox_to_quad as _rbox_to_quad,
)

if TYPE_CHECKING:
    from vernier.instance import DetectionsInput

__all__ = [
    "ClassCeiling",
    "ConventionHypothesis",
    "ConventionReport",
    "angle_error_deg",
    "convention_check",
    "label_ceiling",
    "min_area_rect",
    "to_quad",
]

AngleUnit = Literal["deg", "rad"]
Rotation = Literal["screen_cw", "screen_ccw"]

#: Every `(unit, rotation)` pair, in the order `convention_check`
#: reports them. Four, not three: the unit is as easy to get wrong as
#: the direction, and a radians-as-degrees mix-up is spectacular rather
#: than subtle, which makes it easy to miss when you are looking for
#: something subtle.
_HYPOTHESES: tuple[tuple[AngleUnit, Rotation], ...] = (
    ("deg", "screen_ccw"),
    ("deg", "screen_cw"),
    ("rad", "screen_ccw"),
    ("rad", "screen_cw"),
)


def to_quad(
    rbox: Sequence[float],
    *,
    unit: AngleUnit,
    rotation: Rotation,
) -> list[float]:
    """``[cx, cy, w, h, theta]`` -> ``[x0, y0, x1, y1, x2, y2, x3, y3]``.

    These are the **corrected** kernel's corner bits. That matters for
    exactly one use: a DK-strict parity claim must be made against the
    quads you would actually submit to DOTA_devkit, not against quads
    vernier synthesized on your behalf. For conversion, visualization,
    and comparing the two formats, this is the function.
    """
    return list(_rbox_to_quad(_five(rbox, "rbox"), unit, rotation))


def min_area_rect(
    quad: Sequence[float],
    *,
    unit: AngleUnit,
    rotation: Rotation,
) -> list[float] | None:
    """Minimum-area enclosing rectangle of a quad, as ``[cx, cy, w, h, theta]``.

    ``None`` for a degenerate quad — fewer than three distinct,
    non-collinear vertices — rather than a zero-area rectangle that would
    read as a real answer.

    Deliberately **not** bit-equal to ``cv2.minAreaRect``: it is a
    different implementation with its own conventions, and vernier makes
    parity claims only against the two oracles in ADR-0063.
    """
    out = _min_area_rect(_eight(quad, "quad"), unit, rotation)
    return None if out is None else list(out)


def angle_error_deg(
    gt: Sequence[float],
    dt: Sequence[float],
    *,
    unit: AngleUnit,
    rotation: Rotation,
    tau: float = 0.05,
) -> float:
    """Orientation error between two rotated boxes, in degrees, in ``[0, 90]``.

    IoU is blind to *how* a detection is wrong: a well-localized box that
    is 90 degrees out can score the same as a roughly-placed one that is
    correctly oriented. This separates them.

    The error is reduced modulo the box's own symmetry, so it is
    invariant to the parameterization — ``theta + 180`` and the
    ``w``/``h`` swap describe the same box and report the same error.
    ``tau`` is the near-square tolerance: below
    ``|w - h| / max(w, h) < tau`` the box has no distinguishable long
    axis and the modulus drops from 180 to 90 degrees.
    """
    return float(_angle_error_deg(_five(gt, "gt"), _five(dt, "dt"), unit, rotation, tau))


class ClassCeiling(NamedTuple):
    """One class's label ceiling. See :func:`label_ceiling`."""

    #: Mean ``IoU(quad, minAreaRect(quad))`` over the class's scored
    #: annotations.
    mean_iou: float
    #: How many annotations were scored. A gap against the class's
    #: annotation count means degenerate quads were skipped, which is
    #: itself a finding.
    #:
    #: Named ``n_scored`` rather than ``count`` because a
    #: :class:`~typing.NamedTuple` field called ``count`` shadows
    #: :meth:`tuple.count`.
    n_scored: int


def label_ceiling(
    gt_quads: Sequence[Sequence[float]],
    category_ids: Sequence[int],
    *,
    unit: AngleUnit = "deg",
    rotation: Rotation = "screen_ccw",
) -> dict[int, ClassCeiling]:
    """What a rectangle-predicting model gives up on quad ground truth.

    For each annotation, ``IoU(quad, minAreaRect(quad))``: the best any
    rotated-*rectangle* detector could score on it, since it cannot
    represent a non-rectangular label at all. The result is a property of
    the dataset, computed before any detector runs.

    A class at ``0.93`` is saying that a *perfect* rotated-box detector
    caps out near ``0.93`` IoU on it — under the ``0.95`` rung of the
    COCO ladder, so its contribution to headline AP is bounded by the
    annotation format rather than by the model. That is worth knowing
    before spending a month on the model.

    ``unit`` and ``rotation`` only affect the *reported* angle of the
    fitted rectangle, never the IoU, so they are defaulted here — unlike
    everywhere else in this ADR, where the convention describes input
    that would otherwise be misread.
    """
    quads = [_eight(q, f"gt_quads[{i}]") for i, q in enumerate(gt_quads)]
    cats = [int(c) for c in category_ids]
    raw: Mapping[int, tuple[float, int]] = _label_ceiling(quads, cats, unit, rotation)
    return {cat: ClassCeiling(mean, n) for cat, (mean, n) in raw.items()}


class ConventionHypothesis(NamedTuple):
    """One ``(unit, rotation)`` hypothesis and the AP it produced."""

    unit: AngleUnit
    rotation: Rotation
    #: ``AP @ IoU=0.50``, the stat most sensitive to a convention error
    #: while still tolerant of ordinary localization noise.
    ap50: float


class ConventionReport(NamedTuple):
    """Result of :func:`convention_check`."""

    #: The convention the caller declared.
    declared: ConventionHypothesis
    #: Every hypothesis, best first.
    ranked: tuple[ConventionHypothesis, ...]

    @property
    def best(self) -> ConventionHypothesis:
        """The highest-scoring hypothesis."""
        return self.ranked[0]

    @property
    def declared_is_best(self) -> bool:
        """Whether the declared convention scored highest."""
        b = self.best
        return (self.declared.unit, self.declared.rotation) == (b.unit, b.rotation)


def convention_check(
    gt: bytes,
    dt: DetectionsInput,
    *,
    unit: AngleUnit,
    rotation: Rotation,
    margin: float = 0.05,
    warn: bool = True,
) -> ConventionReport:
    """Score a sample under all four ``(unit, rotation)`` hypotheses.

    Angle unit and rotation direction are the most common real-world OBB
    bug, and the failure is silent — nothing raises, the numbers are just
    wrong, usually low enough to look like a bad model. Running the same
    data under every hypothesis makes the mistake loud: if an undeclared
    convention beats yours by more than ``margin`` AP@0.50, something is
    mislabeled.

    It **warns and never switches**. An automatic switch would be a
    convenience that silently changes what a number means, and a sample
    where the wrong convention happens to win — a rotationally symmetric
    class, a near-square one, too few annotations — would then propagate
    into every downstream comparison. Reporting leaves the decision with
    the person who knows the data.

    Run it on a *sample*: four evaluations is four times the work, and
    the signal is large enough that a few hundred images settle it.

    **Two things it cannot see.** Reinterpreting the convention changes
    how *both* the ground truth and the detections are read, so a
    near-perfect detection scores 1.0 under every hypothesis and tells
    you nothing — run this on real detector output, not on ground truth
    copied into the detection slot. And reinterpreting ``rotation``
    mirrors both boxes about their own centers, so a configuration that
    is itself mirror-symmetric (detections offset along a single axis,
    concentric pairs) scores identically under both signs. The unit axis
    has no such blind spot: a degree read as a radian is a nonsense
    angle and shows up immediately. Read the whole ``ranked`` table
    rather than only ``best``; ties are information.
    """
    from vernier.instance import Evaluator, RotatedBox

    # Before four full evaluations, not after: the loop below iterates
    # `_HYPOTHESES` rather than the caller's arguments, so a typo in
    # either one would otherwise surface as a bare `StopIteration` from
    # the lookup at the end -- no message, and reading like an iterator
    # bug rather than a bad argument.
    if (unit, rotation) not in _HYPOTHESES:
        raise ValueError(
            f"unknown convention unit={unit!r} rotation={rotation!r}; "
            f"unit is 'deg' or 'rad' and rotation is 'screen_cw' or 'screen_ccw'"
        )

    scored: list[ConventionHypothesis] = []
    for u, r in _HYPOTHESES:
        summary = Evaluator(iou=RotatedBox(unit=u, rotation=r), parity_mode="corrected").evaluate(
            gt, dt
        )
        # Stat 1 of the canonical 12: AP @ IoU=0.50.
        scored.append(ConventionHypothesis(u, r, float(summary.stats[1])))

    # Guaranteed to hit: `(unit, rotation)` was checked against
    # `_HYPOTHESES` above, and `scored` covers all four.
    declared = next(h for h in scored if (h.unit, h.rotation) == (unit, rotation))
    ranked = tuple(sorted(scored, key=lambda h: h.ap50, reverse=True))
    report = ConventionReport(declared=declared, ranked=ranked)

    if warn and not report.declared_is_best:
        best = report.best
        gap = best.ap50 - declared.ap50
        if gap > margin:
            warnings.warn(
                f"declared convention unit={unit!r} rotation={rotation!r} scores "
                f"AP@0.50={declared.ap50:.4f}, but unit={best.unit!r} "
                f"rotation={best.rotation!r} scores {best.ap50:.4f} "
                f"(+{gap:.4f}). That gap is what a mislabeled angle convention "
                f"looks like. vernier does not switch for you: check the "
                f"producer of your annotations and your detections, then declare "
                f"the convention you actually have.",
                UserWarning,
                stacklevel=2,
            )
    return report


def _five(v: Sequence[float], name: str) -> list[float]:
    out = [float(x) for x in v]
    if len(out) != 5:
        raise ValueError(f"{name}: expected 5 values [cx, cy, w, h, theta], got {len(out)}")
    return out


def _eight(v: Sequence[float], name: str) -> list[float]:
    out = [float(x) for x in v]
    if len(out) != 8:
        raise ValueError(
            f"{name}: expected 8 values [x0, y0, x1, y1, x2, y2, x3, y3], got {len(out)}"
        )
    return out
