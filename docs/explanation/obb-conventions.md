# Oriented boxes: the two numbers that decide everything

An oriented box is five numbers, `[cx, cy, w, h, theta]`. Four of them
mean the same thing everywhere. The fifth does not, and that is the
whole story of this page.

`theta` has no meaning at all until you say two things:

- **the unit** — is it degrees or radians?
- **the rotation direction** — which way does a positive `theta` turn?

Get either wrong and nothing raises. You get numbers. They are usually
low enough to look like a mediocre model and high enough not to look
like a bug, which is exactly the range in which people spend a month on
the model.

That is why vernier has no default for either:

```python
from vernier.instance import Evaluator, RotatedBox

Evaluator(iou=RotatedBox(unit="deg", rotation="screen_ccw"))
```

There is no `RotatedBox()`. It will not compile in your head and it will
not construct in your interpreter.

## Why "screen_cw" and not "positive is counter-clockwise"

The pixel frame has `x` going right and `y` going **down**. That single
flip is where most of the confusion comes from: the algebraic sign of a
rotation and the direction a viewer sees are opposites of each other,
and almost every explanation of a rotation convention quietly assumes
`y` goes up.

So vernier names the variants after what a person looking at the image
sees:

| Variant | `sigma` | A positive `theta` turns... |
| --- | --- | --- |
| `"screen_cw"` | `+1` | `+x` toward `+y`, i.e. **clockwise on screen** |
| `"screen_ccw"` | `-1` | `+x` toward `-y`, i.e. **counter-clockwise on screen** |

detectron2 is `"screen_ccw"`. So is `mmrotate`, which inherits the same
kernel with radians instead of degrees.

You can check this yourself with detectron2's own documented example: a
box at `(5, 3, 4, 2, 90)` has corners `(4, 5)`, `(4, 1)`, `(6, 1)`,
`(6, 5)`. vernier reproduces it — the assertion is
`convention::tests::d2_sigma_probe_5_3_4_2_90`.

## What is *not* a convention: `le90`, `le135`, `oc`

These names come up constantly in the oriented-detection literature, and
they are not a parameter here. They cannot be, because they do not
change the box.

A rotated box is a set of points, and that set is unchanged by:

- `theta -> theta + 180`, and
- swapping `w` and `h` while adding 90 to `theta`.

`le90`, `le135` and `oc` are three ways of choosing one representative
from each equivalence class of that relation — three ways of *writing
down* the same box. An IoU is a function of the point sets, so it cannot
depend on which representative a producer happened to pick.

vernier does not ask, and does not need to know. Feed it `le135`
annotations and `oc` detections; it will not notice, and the answer will
be the same as if you had normalized both.

(The *bits* are a slightly different matter. detectron2 derives its
corners from `(w, h, theta)`, so writing the same box the other way
moves its last bits. `parity_mode="strict"` passes your numbers through
untouched so that its bit-equality claim is about your input; the
corrected kernel is invariant to within its error bound. See quirk
**OB3**.)

## Checking your convention on real data

If you are not sure — and after the third dataset, nobody is sure —
score a sample under all four hypotheses:

```python
from vernier.instance import obb

report = obb.convention_check(
    gt_bytes, detections, unit="deg", rotation="screen_ccw"
)
for h in report.ranked:
    print(f"{h.unit:>4} {h.rotation:<11} AP@0.50 = {h.ap50:.4f}")
```

A convention error is not subtle in this table. The right hypothesis
usually beats the wrong ones by tens of AP points, because a wrongly
signed angle turns every elongated object across its own axis.

`convention_check` **warns and never switches**. An automatic switch
would be a convenience that silently changes what a number means — and
on a sample where the wrong convention happens to win (a rotationally
symmetric class, a near-square one, too few annotations) it would
propagate into every comparison downstream. The decision stays with the
person who knows the data.

Two blind spots, stated because a check you trust blindly is worse than
no check:

- Reinterpreting the convention changes how **both** sides are read, so
  a near-perfect detection scores 1.0 under every hypothesis. Run this
  on real detector output, not on ground truth copied into the
  detection slot.
- Reinterpreting `rotation` mirrors both boxes about their own centers.
  A configuration that is *itself* mirror-symmetric — detections offset
  along a single axis, or concentric with their ground truth — scores
  identically under both signs. The `unit` axis has no equivalent blind
  spot.

Read the whole `ranked` table, not just `best`. A tie is information.

## Rotated boxes or quads?

vernier has two oriented kernels, and which one you want is a question
about your *annotations*, not your model.

- **`RotatedBox`** — five numbers, a genuine rectangle. This is what
  aerial-detection models predict and what detectron2 evaluates.
- **`Quad`** — four vertices, any shape. This is what DOTA ground truth
  actually is: the annotations are free quadrilaterals, and many of them
  are not rectangles.

If your ground truth is quads and your model predicts rectangles, there
is a ceiling on your score that has nothing to do with your model:

```python
ceiling = obb.label_ceiling(gt_quads, category_ids)
for cat, c in sorted(ceiling.items()):
    print(f"class {cat}: best possible IoU {c.mean_iou:.3f} over {c.count} anns")
```

A class at `0.93` is telling you that a *perfect* rotated-box detector
tops out near `0.93` IoU on it — under the `0.95` rung of the COCO
ladder. That is a property of the annotation format, and it is worth
knowing before you go looking for the missing points in your model.

## Angle error is not IoU

IoU cannot tell you *how* a detection is wrong. A well-placed box that
is 90 degrees out can score the same as a roughly-placed one that is
correctly oriented, and for anything downstream of the detector —
grasping, tracking, heading estimation — those are not the same failure.

```python
obb.angle_error_deg(gt_rbox, dt_rbox, unit="deg", rotation="screen_ccw")
```

The error is reduced modulo the box's own symmetry, so it is invariant
to the parameterization for the same reason the IoU is. For a near-square
box the long axis is not meaningful, so the modulus drops from 180 to 90
degrees; the `tau` parameter controls where that changeover happens.

## Further reading

- [ADR-0063](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0063-oriented-box-evaluation.md)
  — the decision record, including the error analysis.
- [Migrating from detectron2's RotatedCOCOeval](../migrate/detectron2-rotated.md)
- [Migrating from DOTA_devkit](../migrate/dota-devkit.md)
