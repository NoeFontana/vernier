# Calibration: what it answers, and what it doesn't

vernier ships a detection-family calibration summarizer (ADR-0018)
via `Evaluator.evaluate(..., calibration=True)` and the lazy
`result.calibration(...)` fold. Three things come out: a scalar
**ECE** (Expected Calibration Error), a scalar **MCE** (Maximum
Calibration Error), and a **reliability table** of per-bin gaps.
The how-to ([`calibration.md`](../how-to/calibration.md)) walks the
*how*; this page is the *what for, and what against*.

## What calibration answers

**Are my model's confidence scores trustworthy as probabilities?**
A perfectly calibrated detector that emits score `0.7` for a set of
predictions has those predictions correct 70% of the time. The
reliability table is the per-bin breakdown; ECE is a single-number
summary of how far the model deviates from that diagonal.

**At which scores does my model lie most?** MCE — the worst-bin gap
— answers "what is the largest amount of confidence I am ever wrong
by, in a populated bin?" When you tune a downstream decision rule
("only accept detections above 0.9"), MCE is the right alarm to
read; ECE averages a true overconfidence pocket in with calibrated
mid-range bins it would otherwise drown.

**Which classes are uncalibrated?** With `per_class=True`, the
per-class table surfaces ECE per category. `result.calibration(...).worst(5)`
returns the top-5 most miscalibrated classes — the headline read for
a regression triage pass.

## How calibration differs from AP and oLRP

AP and oLRP measure different things than calibration. The
three are orthogonal, and a model can be excellent on one while
catastrophic on another:

- **AP** integrates the precision-recall curve over every threshold.
  A miscalibrated model can have perfect AP if its score *ordering*
  is correct — AP is invariant to monotone score transformations.
  Calibration is not: rescale every score by `score ** 2` and AP
  stays the same; ECE moves.
- **oLRP** ([`lrp-and-its-limits.md`](lrp-and-its-limits.md)) finds
  a single operating point and reports the error structure there.
  oLRP cares about the threshold; calibration cares about *every*
  score. A model whose oLRP `tau` is `0.42` can still be wildly
  miscalibrated below `0.42` (the threshold filtered those scores
  out before oLRP looked).

Read together: AP says "is the ranking right?"; oLRP says "at the
best threshold, what's costing me?"; calibration says "do my scores
mean what they claim?" Three different questions, three different
diagnoses.

## Why DETR-era defaults

Modern transformer detectors (DETR-family, Deformable-DETR,
DINO-DETR, RT-DETR) produce degenerate score distributions that
break the obvious choices for ECE:

- **Quantile binning, not equal-width.** DETR-family models emit a
  pile of low-score padding queries (typically `~270` of `300`
  queries score below `0.05`). Equal-width binning lands ~9 of 15
  bins inside that pile, leaving the high-score region — the one
  practitioners actually care about — under-resolved. Quantile
  bins (parity-pinned to `numpy.quantile(method='linear')`, quirk
  P1) put the histogram resolution where the detections live. On
  degenerate score distributions, duplicate edges get merged
  (quirk P5), so `effective_n_bins` may be `< n_bins`; arithmetic
  over bins should reference `effective_n_bins`.
- **`min_score = 0.05`.** Below this threshold the predictions are
  padding-token artifacts of the architecture, not the model's
  actual epistemic uncertainty. The default cutoff matches the
  filter every DETR evaluation script applies before doing anything
  else. Override with `min_score=0.0` to include them; you will see
  spuriously high "accuracy" in the low-score bins (the model is
  right that those aren't objects, but it's right in a degenerate
  way).
- **Wilson confidence intervals, not Clopper-Pearson.** Per-bin
  accuracies have small `n` in the tails. Wilson intervals are
  closed-form, well-defined down to `n = 1`, and contract slightly
  faster than Clopper-Pearson for the bin sizes that typically
  result from quantile binning at `n_bins = 15` on a COCO-scale
  evaluation. The Clopper-Pearson alternative is wired in
  (`confidence="clopper_pearson"`) but documented Phase-2 — the
  kernel-level tests pin only the Wilson path today.
- **Macro per-class aggregation.** With `per_class=True`, the
  marginal ECE/MCE are the unweighted mean across classes
  (quirk P6) — the same scale invariance the COCO AP convention
  uses. Override with `per_class_aggregation="micro"` to pool
  detections globally; the per-class table is unchanged either way.

The full quirk survey (P1–P10) and three-tier dispositions live in
[`calibration-quirks.md`](../engineering/calibration-quirks.md).

## Why detection-family only today

ADR-0018's per-paradigm shape map sorts every kernel into one of
three shapes:

- **Shape 1 (instance / bbox / segm / boundary / keypoints).** Each
  detection has a scalar score; the histogram fold is unambiguous.
  This ships.
- **Shape 2 (panoptic).** Predictions are dense label maps. Per-
  segment scores exist in the dataset format but are not part of the
  matching contract — the PQ kernel ignores them. Calibrating
  panoptic outputs needs a per-segment confidence convention the
  paradigm does not yet pin; gated on that data-model decision.
- **Shape 3 (semantic).** Per-pixel softmax outputs are scores, but
  there's no notion of a "prediction" to bin against. Calibration
  here is pixel-wise (per-class ECE over pixels), which is a
  different kernel than the detection fold. Gated on that work.

These deferrals are explicit in the ADR. Today
`Evaluator.evaluate(..., calibration=True)` only appears on
`vernier.instance.Evaluator`.

## Limits

### ECE is a single number summarizing many gaps

By design, ECE collapses the reliability table to one scalar. Two
detectors with very different error structures can have identical
ECE — a uniform 5%-off bias across every bin, versus a perfectly-
calibrated mid-range with two pathologically broken bins at the
extremes. ECE alone cannot distinguish them.

The reliability table is what differentiates. If you read only ECE,
you read only the integral; the per-bin breakdown is where the
actionable information lives.

### Small per-class supports → wide CIs

Per-class accuracy estimates ride on per-class detection counts.
COCO has 80 categories but the long tail of those categories has
hundreds of detections at best; LVIS makes this dramatically worse
(1203 categories, many with `<10` detections in val). The Wilson CI
on a 7-out-of-10 bin is `[0.40, 0.89]` — useful, but you cannot read
"73% accuracy" off it. The `n` column in the per-class table is the
disclaimer.

### Calibration is post-hoc; it does not fix anything

Like AP and oLRP, calibration is a measurement. It tells you the
calibration is bad; it does not improve calibration. The
deployment-side fix is a post-hoc calibrator (temperature scaling,
isotonic regression) trained on a held-out set. vernier's job ends
at the measurement; the calibrator is downstream tooling.

### Strict mode is bit-equal to an oracle that doesn't exist publicly

vernier's calibration kernel has no canonical reference implementation
the way pycocotools is the COCO AP oracle. The parity contract is a
clean-room NumPy oracle (`tests/python/parity_calibration/numpy_oracle.py`,
hand-derived from the ADR), bit-equal to the Rust kernel at strict
mode. Three-tier dispositions live in
[`calibration-quirks.md`](../engineering/calibration-quirks.md);
where the literature disagrees on a definition (binning convention,
CI flavor), the disposition makes the choice explicit. There is no
upstream library to cite as "we match X" — the calibrator survey is
the receipt.

## See also

- [ADR-0018](../adr/0018-calibration.md) — correctness model,
  per-paradigm shape map, deferral table.
- [`docs/how-to/calibration.md`](../how-to/calibration.md) —
  recipe-style guide.
- [`docs/engineering/calibration-quirks.md`](../engineering/calibration-quirks.md)
  — P1–P10 disposition survey.
- [`lrp-and-its-limits.md`](lrp-and-its-limits.md) — the operating-
  point sibling diagnostic.
- [`tide-and-its-limits.md`](tide-and-its-limits.md) — the error-
  structure sibling diagnostic. Calibration is orthogonal to both.
