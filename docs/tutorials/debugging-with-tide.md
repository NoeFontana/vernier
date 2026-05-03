# Debugging detection failures with TIDE

When a model's mAP drops five points after retraining, the headline
number tells you something is wrong but not what. TIDE — Toolbox for
Identifying Detection Errors, [Bolya et al. 2020](https://arxiv.org/abs/2008.08115)
— answers the second question. It splits the gap between the model's
measured mAP and the perfect-mAP upper bound into six interpretable
bins and reports how many points each bin is costing. A regression
that shows up as `Bkg=+3.0, Loc=+0.5` is a different bug than one that
shows up as `Loc=+3.5`, and the fix lives in a different layer of the
pipeline. TIDE is what tells you which.

This tutorial walks through `vernier.error_decomposition` end-to-end:
the call shape, what each bin means, how to choose thresholds, and
where TIDE stops being the right tool.

## Quick start

The `tests/python/oracle/tide/fixtures/all_loc/` fixture is a
synthetic single-image case engineered so every detection is a
localization error: one ground truth at `[0, 0, 100, 100]`, one
detection at `[0, 0, 80, 50]` — same class, IoU low enough to miss
the match threshold but high enough to clear the background cutoff.

```python
from pathlib import Path

from vernier import Bbox, error_decomposition

fixture = Path("tests/python/oracle/tide/fixtures/all_loc")
gt_bytes = (fixture / "gt.json").read_bytes()
dt_bytes = (fixture / "dt.json").read_bytes()

report = error_decomposition(gt_bytes, dt_bytes, iou=Bbox())
print(report.baseline_map)         # 0.0
print(report.delta)                # {'cls': 0.0, 'loc': 0.999..., 'both': 0.0,
                                   #  'dupe': 0.0, 'bkg': 0.0, 'missed': 0.0}
print(report.delta_all_fp_removed) # 0.0
print(report.config)               # TideConfig(t_f=0.5, t_b=0.1, kernel='bbox')
```

The model's baseline mAP is `0.0` (the lone detection misses the
match threshold, so it scores nothing). `delta['loc']` rounds to a
full point — exactly the right answer for a single-image fixture
where the only error is a Loc error. The other five bins are zero
because the rewrite layer finds no detections to assign there.

## Reading the report

Each delta is the mAP increase the model would achieve if every
detection assigned to that bin were corrected. The bins are mutually
exclusive: a single detection lands in exactly one (matched first to
its highest-scoring eligible GT, then bin-assigned by the IoU it
achieved). See [ADR-0021](../adr/0021-tide-oracle.md) for the
algorithmic spec.

- **`cls` (Classification)** — The detection matched a GT of the
  *wrong* class with IoU at or above `t_f`. The localization is
  correct; the label is not. Common when a category taxonomy has
  visually similar siblings (e.g. `dog` vs `wolf`).
- **`loc` (Localization)** — Right class, IoU in `[t_b, t_f)`. The
  detection landed near a real GT but not tightly enough. Common with
  underfit regressors or aggressive box-pruning.
- **`both`** — Both classification and localization wrong: matched a
  GT of a different class with IoU in `[t_b, t_f)`. Rarer than `cls`
  or `loc` alone; usually flags a region-proposal stage that's noisy
  in both dimensions.
- **`dupe` (Duplicate)** — A higher-scoring detection of the same
  class already matched this GT. NMS at inference time should have
  caught this; if `dupe` is large the IoU threshold of NMS is too
  loose, or NMS is being skipped entirely.
- **`bkg` (Background)** — IoU `< t_b` against every GT (any class).
  The detector is firing on background. Common with aggressive
  recall-tuning or a confidence threshold set too low.
- **`missed`** — A GT survived the eight passes with no matching
  detection. The detector never fired on this region. Usually the
  flip-side of a too-strict confidence threshold or a recall ceiling
  in the proposal stage.

The two summary numbers carry their own information.
`baseline_map` is the headline mAP before any bin-specific
correction is applied — the same number `Evaluator.evaluate` returns.
`delta_all_fp_removed` is the paper's "perfect rejection" upper bound:
what mAP would be if every false positive were dropped. The per-bin
deltas should sum to at most `delta_all_fp_removed`; the slack is the
overlap between Missed-GT recovery and the per-bin FP corrections.

## Choosing thresholds

`error_decomposition` takes two threshold knobs:

- **`t_f`** (foreground / match) — the IoU at-or-above which a
  detection is a TP. Defaults to `0.5` everywhere. This matches both
  the TIDE paper and AP@0.5; making the bin-assignment threshold the
  same as the user's intuitive "good match" cutoff means there's no
  double-cutoff to reason about.
- **`t_b`** (background) — IoU `< t_b` is "background"; IoU
  `∈ [t_b, t_f)` is "almost matched". This carves the boundary
  between Bkg and Loc/Both.

The defaults for `t_b` are per-kernel per
[ADR-0022](../adr/0022-tide-thresholds.md), because IoU distributions
are not the same shape across kernels:

| Kernel    | `t_b` default | Anchored on                      |
|-----------|---------------|----------------------------------|
| `Bbox`    | `0.1`         | TIDE paper, COCO bbox            |
| `Segm`    | `0.1`         | Bbox-row extrapolation; tentative |
| `Boundary`| `0.05`        | Band-area compression argument; tentative |

Boundary IoU concentrates at lower values than bbox IoU on the same
overlap geometry — the kernel's "almost-matched" regime measures
roughly half the value the bbox kernel reports. Reusing `t_b = 0.1`
for boundary would push the Bkg/Loc cutoff into a part of the
distribution that no longer corresponds to "almost matched", silently
moving Loc errors into the Bkg bin. This is the only kernel where
the per-kernel default is doing real work; for bbox and segm it is
the paper's number unchanged.

Override per call when the model's distribution warrants it:

```python
report = error_decomposition(
    gt_bytes, dt_bytes,
    iou=Bbox(),
    t_f=0.5,
    t_b=0.2,   # tighter Bkg cutoff for a model that fires near-misses everywhere
)
```

The resolved values land on `report.config`, so a screenshot of a
report is self-describing: the reader does not need to know which
defaults were in force.

## What TIDE does NOT do

The 0.5.0 surface intentionally stops short of three things people
sometimes expect from TIDE-shaped tools:

- **Keypoints (OKS).** `Keypoints(...)` raises `NotImplementedError`
  per [ADR-0024](../adr/0024-tide-keypoints-deferred.md). COCO
  keypoints is single-class, which makes the Cls and Both bins
  structurally empty; OKS is not IoU, so the `(t_b, t_f)` phase
  diagram does not carve the same error geometry. The right TIDE-shaped
  tool for keypoints is per-keypoint OKS contribution analysis,
  planned as a separate capability in a future minor.
- **Per-class drill-down.** The current `TideReport` aggregates across
  classes. A `report.per_class` polars view (which classes contribute
  most to each bin) is planned for 0.5.x. Today, the workaround is to
  call `error_decomposition` once per category subset of the GT/DT
  payload — wasteful for large category counts but mechanically correct.
- **Per-threshold mode.** The opt-in `mode="per_threshold"` variant
  the [TIDE paper](https://arxiv.org/abs/2008.08115) describes
  (computing the decomposition at every IoU threshold in the AP grid
  and averaging) is deferred to a 0.5.x follow-up. The single-`t_f`
  form vernier ships today is the paper-faithful default; the
  per-threshold form is opt-in even in `tidecv`.

## Performance note

`error_decomposition` runs eight evaluation passes per call: one
baseline plus one per bin (Cls / Loc / Both / Dupe / Bkg / Missed)
plus the all-FPs-removed sanity total. Expect roughly 6× the cost of
a single `Evaluator.evaluate` call on the same dataset. This is by
design — the cell-rewrite layer makes each pass cheap relative to a
full re-match, but it cannot make eight passes free.

TIDE is offline error analysis, not a hot-path metric. Run it once
per model checkpoint, not once per training step.

## Sibling capability

For the related "which classes get confused with which" question, see
`vernier.confusion_matrix(...)`. Where TIDE answers "what kind of
error", a confusion matrix answers "between which two labels". The two
are complementary; a model with high `cls` ΔmAP in TIDE is a candidate
for confusion-matrix inspection.

## See also

- [ADR-0021](../adr/0021-tide-oracle.md) — the numpy oracle that pins
  vernier's TIDE semantics. Contains the full algorithmic spec.
- [ADR-0022](../adr/0022-tide-thresholds.md) — the per-kernel default
  thresholds (`t_f`, `t_b`) and their defenses.
- [ADR-0024](../adr/0024-tide-keypoints-deferred.md) — why TIDE on
  keypoints (OKS) is not in 0.5.0.
- Bolya et al., ["TIDE: A General Toolbox for Identifying Object
  Detection Errors"](https://arxiv.org/abs/2008.08115), ECCV 2020 —
  the paper this implementation is faithful to.
