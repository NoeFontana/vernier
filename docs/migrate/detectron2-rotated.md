# Migrating from detectron2's `RotatedCOCOeval`

vernier reproduces detectron2's rotated-box IoU **bit for bit** under
`parity_mode="strict"`. Not "to within a tolerance", not "close enough
for a leaderboard" — the same `float32` bits, including the parts that
are arguably bugs.

## The short version

```python
from vernier.instance import Evaluator, RotatedBox

evaluator = Evaluator(
    iou=RotatedBox(unit="deg", rotation="screen_ccw"),
    parity_mode="strict",
)
summary = evaluator.evaluate(gt_json_bytes, detections)
print(summary.pretty_lines())
```

`unit="deg"`, `rotation="screen_ccw"` **is** detectron2's convention. If
your boxes come from detectron2, those are the two values you want.

## The one thing you have to change: `bbox` becomes `rbox`

detectron2 stores a rotated box in the `bbox` field, as five numbers:

```json
{"bbox": [cx, cy, w, h, angle], "category_id": 1, ...}
```

vernier refuses that, on purpose, and it is the single most important
sentence on this page:

> A length-5 `bbox` read as `[x, y, w, h]` is silent and catastrophic.

Every axis-aligned reader in the ecosystem — including `pycocotools`
itself — will take those first four numbers and evaluate a box at
`(cx, cy)` with size `(w, h)`. That is not your box; it is a box at the
wrong place, and the resulting AP is *plausible*. Nothing raises.

So vernier requires a separate key:

```json
{"bbox": [x, y, w, h], "rbox": [cx, cy, w, h, angle], "category_id": 1, ...}
```

and hard-fails on a length-5 `bbox` on every native path. The `bbox` you
supply alongside is the axis-aligned envelope, used for area bucketing
and for nothing else.

Converting a detectron2-shaped file is four lines:

```python
import json, math

def to_vernier(record):
    cx, cy, w, h, a = record["bbox"]
    t = math.radians(-a)                      # screen_ccw -> algebraic
    ex = abs(math.cos(t)) * w / 2 + abs(math.sin(t)) * h / 2
    ey = abs(math.sin(t)) * w / 2 + abs(math.cos(t)) * h / 2
    return {**record, "bbox": [cx - ex, cy - ey, 2 * ex, 2 * ey],
            "rbox": [cx, cy, w, h, a], "area": w * h}
```

Note the `area`: detectron2's `loadRes` sets it to `bb[2] * bb[3]`,
which on a length-5 `bbox` is the *oriented* `w * h`, not the envelope's.
Carrying it explicitly keeps the area buckets in step with the kernel
(quirk **OB13**).

## What "bit for bit" covers

`parity_mode="strict"` reproduces, deliberately:

- **f32 geometry.** detectron2 instantiates its kernel at `float`,
  because `RotatedCOCOeval` builds tensors with `torch.FloatTensor`.
- **The truncated constant.** Its degrees-to-radians factor is
  `0.01745329251`, not `pi/180` — short by about `1e-11`. vernier uses
  the same literal.
- **The pair-midpoint center shift**, the epsilon-relaxed vertex
  collection, and the hand-written hull sort, op for op.
- **IoU outside `[0, 1]`.** detectron2#350 is real and reachable;
  `strict` preserves those values rather than clamping them.
- **The threshold comparison dtype.** This one is subtle enough to
  deserve its own section.

The evidence is `crates/vernier-geom/tests/d2_bridge.rs`: twenty million
pairs across five adversarial regimes — near-coincident, grazing, exact
quadrant angles, square aspect ratios, DOTA-scale coordinates — compared
against the pinned C++ header compiled with release-equivalent flags.
Zero divergences.

## The threshold comparison (why your AP may move by ~0.0001)

`RotatedCOCOeval.computeIoU` returns a **torch tensor**, and
`pycocotools`' `evaluateImg` then compares it against an `np.float64`
threshold. PyTorch treats that scalar as weakly typed, so the comparison
runs in `float32` — against `f32(0.7)`, which is `0.699999988...`, not
`0.7`.

A detection whose IoU is exactly that value therefore **matches** under
detectron2 and would not under a faithful f64 comparison. vernier
reproduces the behavior by projecting the threshold ladder onto the f32
lattice for this kernel under `strict`, so the same pairs match.

If you have been comparing vernier's `corrected` mode against
detectron2 and wondering about a difference in the fourth decimal, this
is where a piece of it lives.

## What `corrected` fixes

`parity_mode="corrected"` runs vernier's own f64 kernel instead:

| | `strict` (detectron2) | `corrected` |
| --- | --- | --- |
| Geometry | f32 | f64, clipped in the ground truth's own frame |
| `IoU(a, a)` | ~1.0 | **exactly** 1.0 |
| Range | can exceed `[0, 1]` | clamped |
| Disjoint pairs | exactly `+0.0` | exactly `+0.0` |
| Error at DOTA scale | ~`1e-3` | ~`1e-13`, independent of image size |

Use `strict` to reproduce a published number. Use `corrected` when the
number is the point.

## What is not covered

`RotatedCOCOeval` under `patch_pycocotools()` is not yet routed to the
Rust kernel: if you subclass `COCOeval` and override `computeIoU`, the
shim does not yet call your override. That is ADR-0063 M4 PR-4.5 and it
is not in this release. Use the native `Evaluator` surface above.

Streaming and background evaluation do not support oriented kernels
yet; batch and partitioned evaluation do.
