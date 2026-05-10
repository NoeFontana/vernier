# TIDE: what it answers, and what it doesn't

vernier ships a TIDE error decomposition (Bolya et al. 2020) via
`vernier.instance.error_decomposition`. Six bins — Cls, Loc, Both, Dupe, Bkg,
Missed — together account for the gap between a model's headline mAP
and the perfect-mAP upper bound. The tutorial
([`docs/tutorials/debugging-with-tide.md`](../tutorials/debugging-with-tide.md))
walks the *how*; this page is the *what for, and what against*. Long
enough that an issue starting with "the bins don't add up" can be
closed by a link.

## What TIDE answers well

**Which kind of error is costing the most points?** That's the question
the paper was designed for, and the question vernier's implementation
answers correctly per [ADR-0021](../adr/0021-tide-oracle.md) (numpy
oracle, 1e-9 parity on synthetic fixtures). If a model's mAP drops by
0.04 and the Loc bin accounts for 0.03, the right place to look is
localization — head architecture, regression target, IoU-based loss
choice. If Bkg dominates, score calibration or NMS aggressiveness.
TIDE turns "the number went down" into a hypothesis.

**Comparing two models on the same data.** A model that trades 0.01
points of Cls error for 0.02 points of Loc error is making a different
tradeoff than a model that does the reverse, even if their mAPs match.
The decomposition surfaces those tradeoffs cleanly.

**Validating a fix.** If a change was meant to reduce duplicate
detections, the Dupe bin should shrink. If it doesn't, the fix isn't
doing what was claimed. TIDE is a sharper signal than mAP for
mechanism-targeted changes.

## What TIDE does *not* answer

### "The six bins should add up to the mAP gap"

They don't. Not in the paper, not in vernier, not in `tidecv`. The
six corrections are computed *independently*, each by removing one
bin's errors and recomputing mAP. The corrections overlap: turning a
Cls error into a TP via the rewrite layer can incidentally elevate
nearby detections in the score-sorted list, raising AP at recall
points that other corrections also hit. So:

```
sum(report.delta.values()) ≠ report.delta_all_fp_removed
```

is not a bug. The structurally enforced invariant is per-bin: every
delta is non-negative, and every delta is bounded above by
`delta_all_fp_removed` (the perfect-rejection sanity total).
`delta_all_fp_removed` is the upper bound because removing every FP
is at least as good as removing any single bin's errors. Reviewers
wanting an "everything fixed" gap should compare baseline to
`baseline_map + delta_all_fp_removed`.

### "Move t_f / t_b to make the numbers look better"

Threshold tuning is moving error mass between bins, not making errors
disappear. Lower `t_b` shifts mass from Bkg to Loc; higher `t_f`
shifts mass from Both/Cls into Loc/Bkg by demanding more localization
precision before counting a match as the right class. The defaults
([ADR-0022](../adr/0022-tide-thresholds.md)) are deliberately chosen
to match the paper (`t_f=0.5`) and to assign the geometrically natural
floor for "could plausibly be a match" (`t_b=0.1` for area-IoU kernels,
`t_b=0.05` for boundary IoU at `dilation_ratio=0.02` because the band
geometry compresses the IoU range).

The report records the resolved `(t_f, t_b)` on every call's
`config`, so a screenshot of the output is self-describing — but if
two runs use different thresholds, their bin masses are *not*
comparable. Compare bin masses across models with the same thresholds,
or compare thresholds within the same model with the same bin masses.
Cross-pairs are uninterpretable.

### "Bin masses on a small dataset are meaningful"

The decomposition is computed at dataset scope. On a 50-image subset,
bin masses can be dominated by a handful of images and are not stable
across resamples. Same disease as per-image AP (see
[`docs/explanation/why-no-per-image-ap.md`](why-no-per-image-ap.md)):
AP integrates a curve, and curves built from few points are
high-variance. TIDE on COCO val2017 (5000 images) is the canonical
operating regime; subsets below ~500 images should be read as
qualitative, not measured.

### "TIDE replaces mAP"

It doesn't. mAP is the headline number; TIDE is a debugging aid.
A model's TIDE bins can all be small and its mAP can still be
unsatisfactory because the *baseline* is low — meaning the model is
not finding many true positives at all (Missed bin large), or the
score calibration is so bad that the PR curve never gets above the
threshold. The Missed and Bkg bins surface those failures, but the
right action is sometimes "retrain the model from scratch", not "tune
NMS." TIDE doesn't tell you when you're past diminishing returns.

### "I can read the bins as percentages of error"

Nominally yes — each bin is a fraction of mAP. But "fraction of
mAP" rolls in two different things: the *frequency* of the error
type, and the *cost* per error (high-confidence FPs cost more than
low-confidence ones because they elevate first in the PR curve). A
1.0% Cls bin can mean "many low-cost classification errors" or "few
high-cost ones." The decomposition is silent on which, and that
matters when picking the right intervention.

If the distinction matters, the per-class TIDE drill-down is the
right shape (mentioned in the tutorial as a deferred follow-up), or
read the confusion matrix output of
`vernier.instance.confusion_matrix`. Confusion gives
you raw counts of `(true_class, predicted_class)` pairs, which is the
frequency view; TIDE's Cls bin is the cost view. They answer different
questions and ideally you read both.

## Where the kernel-specific defaults come from

[ADR-0022](../adr/0022-tide-thresholds.md) is canonical; in summary:

- **bbox** — `t_f=0.5, t_b=0.1`, both anchored on the TIDE paper's
  COCO bbox numbers. The paper's main table is reproducible against
  these defaults on canonical models.
- **segm** — same `t_f=0.5, t_b=0.1`. *Tentative*: argued from the
  bound that segmentation IoU is at most bbox IoU on the same
  instance, and tracks within ~10% in practice on standard models.
  The validation harness
  ([`tests/python/integration/real_models/tide/`](../../tests/python/integration/real_models/tide/))
  exercises the segm kernel against rf-detr `RFDETRSegNano` on COCO
  val2017; this is empirical anchoring, not theoretical proof.
- **boundary** — `t_f=0.5, t_b=0.05` at `dilation_ratio=0.02`, also
  tentative, anchored geometrically: boundary IoU at small dilation
  ratios compresses the IoU distribution, so a tighter floor is
  needed before "plausibly a match" carries the same meaning as on
  area IoU.

The "segm and boundary defaults are tentative" warning in ADR-0022
is honest; if your workload's models cluster differently, override
per call. The report's `config.t_b` makes the resolution audit-trail
available.

## The keypoints gap

[ADR-0024](../adr/0024-tide-keypoints-deferred.md) deferred TIDE on
keypoints (OKS) entirely. The structural reasons:

- COCO keypoints is single-class. The Cls and Both bins are
  empty by construction — there's no "wrong class" to flip into.
  Two of six bins doing nothing is not a useful decomposition.
- OKS is not an IoU. The geometric phase diagram TIDE relies on
  (`[t_b, t_f)` carves Loc; `>= t_f` carves Cls/match) doesn't carve
  the right error space for a per-keypoint similarity score.
- No published convention exists for TIDE-on-OKS. Picking defaults
  would be inventing a contract we couldn't validate against
  literature.

The path forward is a per-keypoint OKS drill-down (a different
metric, not TIDE-shaped), tracked as a follow-up to ADR-0024.
`vernier.instance.error_decomposition(..., iou=Keypoints(...))`
raises `NotImplementedError` with a pointer to ADR-0024 today.

## Where the validation comes from

The Rust core is gated against a numpy oracle on 13 synthetic
fixtures (one per dispositioned bin per kernel; see
`tests/python/oracle/tide/`). Synthetic fixtures pin behavior the
oracle and the implementation must agree on at 1e-9; that's how we
know the rewrite layer doesn't have an off-by-one in the Cls
cell-move.

Real-model behavior is validated via the rf-detr harness in
[`tests/python/integration/real_models/tide/`](../../tests/python/integration/real_models/tide/).
That harness asserts structural coherence (per-bin non-negativity,
upper bound) and determinism (two runs, byte-identical reports) on
COCO val2017. It is not a regression gate on bin masses — those
are model-dependent — but it is a regression gate on "the
implementation runs correctly on real data and produces sensible
numbers."

## See also

- [`docs/tutorials/debugging-with-tide.md`](../tutorials/debugging-with-tide.md)
  — worked example of using TIDE on a model, end-to-end.
- [ADR-0021](../adr/0021-tide-oracle.md) — correctness model.
- [ADR-0022](../adr/0022-tide-thresholds.md) — threshold defaults.
- [ADR-0023](../adr/0023-tide-cross-class-strategy.md) —
  cross-class IoU side pass (and the basis for `confusion_matrix`).
- [ADR-0024](../adr/0024-tide-keypoints-deferred.md) — keypoints
  deferral.
- [`docs/explanation/why-no-per-image-ap.md`](why-no-per-image-ap.md)
  — same statistical reasoning for "scale matters" applied to a
  different metric.
