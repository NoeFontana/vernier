"""Shared oriented-box fixture builders (ADR-0063).

Kept out of the test modules because three of them need the same COCO
records, and a second transcription of the envelope formula is exactly
the kind of thing that drifts.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

from vernier.instance import ResultAnnotation

#: detectron2's convention, used everywhere in this package unless a
#: test is specifically about a different one.
UNIT = "deg"
ROTATION = "screen_ccw"


def _algebraic(theta_deg: float, rotation: str) -> float:
    """`sigma * kappa * theta`, in radians."""
    sign = 1.0 if rotation == "screen_cw" else -1.0
    return math.radians(theta_deg) * sign


def envelope(rbox: Sequence[float], rotation: str = ROTATION) -> list[float]:
    """Tight axis-aligned `[x, y, w, h]` around a rotated box."""
    cx, cy, w, h, t = rbox
    a = _algebraic(t, rotation)
    ex = abs(math.cos(a)) * w / 2 + abs(math.sin(a)) * h / 2
    ey = abs(math.sin(a)) * w / 2 + abs(math.cos(a)) * h / 2
    return [cx - ex, cy - ey, 2 * ex, 2 * ey]


def corners(rbox: Sequence[float], rotation: str = ROTATION) -> list[float]:
    """The four corners, in the canonical kernel's order."""
    cx, cy, w, h, t = rbox
    a = _algebraic(t, rotation)
    c, s = math.cos(a), math.sin(a)
    ux, uy = c * w / 2, s * w / 2
    vx, vy = -s * h / 2, c * h / 2
    return [
        cx + ux + vx,
        cy + uy + vy,
        cx - ux + vx,
        cy - uy + vy,
        cx - ux - vx,
        cy - uy - vy,
        cx + ux - vx,
        cy + uy - vy,
    ]


#: A small aerial-ish scene: two elongated objects at different angles
#: and one square. The square is there on purpose — it is the case where
#: the angle is least observable, and therefore the one a convention bug
#: hides behind.
SCENE: tuple[list[float], ...] = (
    [100.0, 100.0, 60.0, 20.0, 30.0],
    [400.0, 300.0, 80.0, 25.0, -15.0],
    [700.0, 700.0, 40.0, 40.0, 0.0],
)

#: The same scene without the square. Needed wherever the *angle* has
#: to be observable: a square is invariant under a quarter turn, so it
#: scores a perfect match against its own 90-degree rotation and would
#: mask exactly the error under test.
ELONGATED: tuple[list[float], ...] = (
    [100.0, 100.0, 60.0, 20.0, 30.0],
    [400.0, 300.0, 80.0, 25.0, -40.0],
    [700.0, 700.0, 70.0, 18.0, 62.0],
    [200.0, 600.0, 90.0, 22.0, -70.0],
)


def ground_truth(
    rboxes: Sequence[Sequence[float]] = SCENE,
    *,
    rotation: str = ROTATION,
) -> bytes:
    """COCO ground-truth JSON carrying both `rbox` and `quad`."""
    doc: dict[str, Any] = {
        "images": [{"id": 1, "width": 1024, "height": 1024}],
        "categories": [{"id": 1, "name": "ship"}],
        "annotations": [
            {
                "id": i + 1,
                "image_id": 1,
                "category_id": 1,
                "iscrowd": 0,
                "area": rb[2] * rb[3],
                "bbox": envelope(rb, rotation),
                "rbox": list(rb),
                "quad": corners(rb, rotation),
            }
            for i, rb in enumerate(rboxes)
        ],
    }
    return json.dumps(doc).encode()


def detections(
    rboxes: Sequence[Sequence[float]] = SCENE,
    *,
    rotation: str = ROTATION,
    geometry: str = "rbox",
) -> list[ResultAnnotation]:
    """Result dicts, scored descending, carrying `rbox` or `quad`.

    `area` is spelled out to document the expected value, not to supply
    it: quirk **OB13** has the oriented kernels *derive* the detection
    area from the geometry, matching `loadRes`'s `bb[2] * bb[3]` on a
    length-5 `bbox`. The evaluator overwrites whatever is here with the
    same number, exactly as `loadRes` overwrites `area` on every result.
    """
    out: list[ResultAnnotation] = []
    for i, rb in enumerate(rboxes):
        rec: ResultAnnotation = {
            "image_id": 1,
            "category_id": 1,
            "score": 0.9 - 0.05 * i,
            "bbox": envelope(rb, rotation),
            "area": rb[2] * rb[3],
        }
        if geometry == "rbox":
            rec["rbox"] = list(rb)
        else:
            rec["quad"] = corners(rb, rotation)
        out.append(rec)
    return out


def jitter(
    rboxes: Sequence[Sequence[float]] = SCENE,
    *,
    dx: float = 1.5,
    dy: float = -1.0,
    dtheta: float = 2.0,
    scale: float = 0.97,
) -> list[list[float]]:
    """Perturbed boxes, so AP lands strictly between 0 and 1."""
    return [[rb[0] + dx, rb[1] + dy, rb[2] * scale, rb[3] / scale, rb[4] + dtheta] for rb in rboxes]
