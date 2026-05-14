# ADR-0045: LRP for keypoints — shipped, not deferred

- **Status:** proposed
- **Date:** 2026-05-14
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

vernier 0.5.x ships LRP / oLRP (ADR-0043, ADR-0044) for the bbox /
segm / boundary kernels under `vernier.instance.optimal_lrp` and
`vernier.panoptic.optimal_lrp`. The question this ADR answers is
whether LRP ships for the *keypoints* (OKS) kernel in 0.5.x.

The default expectation, set by ADR-0024, is "no" — that ADR deferred
TIDE on keypoints. The reasoning was structural: TIDE's six-bin
decomposition has two cross-class bins (Cls / Both) that are
algorithmically undefined on COCO keypoints (single-class), and the
`(t_b, t_f)` phase diagram that carves the remaining four bins does
not carve the OKS error geometry. ADR-0024 was right for TIDE.

The question this ADR answers is whether ADR-0024's reasoning transfers
to LRP. We argue it does not — the structural objections to
TIDE-on-OKS do not apply to LRP-on-OKS, and LRP-on-OKS answers a
question keypoints users have today.

This is a **positive ship decision**, not a deferral. The whole point
of the ADR is to make explicit *why* TIDE-0024's deferral reasoning
does NOT transfer to LRP — so a future contributor reading the two
ADRs side-by-side does not assume symmetry where none exists.

## Decision drivers

The structural argument is the load-bearing driver. ADR-0024
deferred TIDE-on-OKS for two reasons:

- **(a) Cls / Both bins are structurally zero on COCO keypoints.**
  Single-class workload, no other class to confuse with. Two of six
  bins doing nothing is not a useful decomposition.
- **(b) The `(t_b, t_f)` phase diagram does not carve OKS error
  geometry.** OKS is a per-keypoint Gaussian similarity weighted by
  per-category sigmas; the geometric reading that TIDE relies on for
  "almost matched vs background" does not transfer.

For LRP, both objections vanish:

- **LRP has no Cls / Both bins.** The decomposition is `oLRP_Loc /
  oLRP_FP / oLRP_FN`. All three are well-defined on a single-class
  workload. There is no cross-class machinery to be undefined.
  Objection (a) does not apply.
- **`oLRP_Loc = 1 − mean(OKS on TPs)` is well-defined.** LRP's
  localization term is the average similarity score across TPs,
  inverted. For OKS this reads as "the average OKS gap on the
  detections we counted as TPs" — a directly interpretable number.
  No invented threshold is needed for the Loc term itself.
  Objection (b) does not apply at the component level; the Loc
  computation is a continuous-similarity average, not a binned phase
  diagram.

The OKS TP threshold for the FP / FN counting (the cutoff above which
a matched pair is a TP at all, before the tau search) is a defaults
question, handled in ADR-0044. **We are not inventing a threshold for
LRP-on-OKS.** LRP's tau search is over the *confidence* threshold,
not the *similarity* threshold; the similarity threshold is the
matching-engine TP gate, and the same gate exists for bbox / segm /
boundary. ADR-0044 commits to `0.5` as the operating-point anchor
across kernels, including OKS, with the same defensibility argument.

The third driver is the user's question. The Phase-3 per-keypoint
OKS contribution analysis that ADR-0024 named as the alternative
diagnostic answers "which limb is the limiting factor?" That is a
useful question. It is not a *substitute* for "what is the
deployable confidence threshold for the model, and how much
localization error remains at that threshold?" The two diagnostics
answer different questions and both can ship.

- **Structural fit.** All three LRP components are defined on OKS
  workloads. The metric's shape and the kernel's shape are
  compatible.
- **Paper anchoring exists.** The Oksuz TPAMI 2021 paper uses LRP
  on OKS in the keypoints sections. There is a published convention
  to anchor against; ADR-0024's "no paper anchor and no community
  precedent" argument for TIDE-on-OKS does not apply to LRP-on-OKS.
- **The keypoints user has a deployable-threshold question too.**
  Keypoints models ship with confidence thresholds the same way bbox
  models do. The "what tau should I deploy at?" question is
  paradigm-independent.
- **Complementarity with the Phase-3 alternative.** Per-keypoint OKS
  contribution analysis (ADR-0024's planned alternative) answers a
  different question. Both diagnostics fit; LRP-on-OKS is not
  blocking the per-keypoint analysis and vice versa.
- **Symmetry on the instance surface.** Shipping LRP for three of
  four kernels and deferring the fourth invites the same wrong-shape
  reading ADR-0024 was careful to avoid for TIDE. With LRP the
  symmetry is *real* — the metric works on OKS — so withholding
  would itself be the misleading move.

## Considered options

1. **Defer LRP-on-OKS, matching ADR-0024.** `vernier.instance.optimal_lrp(...,
   iou=Keypoints(...))` raises `NotImplementedError` pointing at this ADR.
   The per-keypoint OKS contribution analysis (Phase-3, per ADR-0024) is
   the only diagnostic shipped for keypoints.
2. **Ship LRP-on-OKS in 0.5.x.** `vernier.instance.optimal_lrp(...,
   iou=Keypoints(...))` returns the three-component table for the
   `person` class (and any future multi-category keypoints dataset).

## Decision outcome

Chosen option: **(2) ship.**

0.5.x extends `vernier.instance.optimal_lrp` to accept
`iou=Keypoints(...)`. The output is the same three-component table
plus per-class `tau` the other kernels return. On COCO the table has
one class (`person`); the API is identical on multi-category
keypoints datasets when they appear.

The per-keypoint OKS contribution analysis from ADR-0024's planned
follow-up remains on the Phase-3 keypoints track. The two are
**complementary, not substitutes**:

- LRP-on-OKS answers "what's the deployable threshold and how
  much localization error remains at it?"
- Per-keypoint OKS contribution answers "which keypoint (limb /
  torso / face) is the limiting factor at the operating point?"

A user diagnosing a keypoint model reads both: LRP tells you *where
the model lives* on the precision-recall-localization surface, the
per-keypoint analysis tells you *what to fix to move it*. Shipping
LRP-on-OKS does not preempt or compete with the per-keypoint
diagnostic; it carries different information.

### Consequences

- **Positive.** Keypoints users get the same diagnostic surface
  as bbox / segm / boundary users on 0.5.x. The `vernier.instance.
  optimal_lrp` entry point is uniform across the four supported
  kernels. No `NotImplementedError` branch to maintain. The Phase-3
  per-keypoint contribution analysis has a clean scope (it is the
  *complement* to LRP-on-OKS, not the substitute) and does not
  carry the implicit pressure of being "the only keypoints
  diagnostic."
- **Negative.** ADR-0024 and ADR-0045 read as opposed positions on
  the same question if a reader skims the titles. The full reasoning
  (TIDE's structural objections do not transfer to LRP) is laid out
  here, but the ADR-pair carries an explanatory cost: every future
  contributor working on a new instance-level metric for keypoints
  has to read both to understand which precedent applies.
- **Neutral.** The keypoints LRP code path adds one branch in
  `lrp::defaults_for(kernel)` (per ADR-0044) and one branch in the
  Python dispatch. The matching engine already supports OKS
  (ADR-0012). No new kernel work.

## What this ADR explicitly does not decide

- **Per-keypoint OKS contribution analysis specifics.** Belongs in
  the Phase-3 keypoints track, with its own ADR. ADR-0045 only
  commits to "LRP-on-OKS ships in 0.5.x"; the per-keypoint
  diagnostic shape is unchanged from ADR-0024's deferral.
- **LRP on multi-category keypoints datasets.** Shipping the
  `vernier.instance.optimal_lrp(..., iou=Keypoints(...))` path
  makes multi-category datasets work transparently — the
  per-class table just has more rows. No special-casing is needed
  in this ADR; the multi-category support follows from the
  per-class shape ADR-0043 already commits to.
- **OKS-specific tau-search optimizations.** If the tau-search loop
  is a bottleneck on real OKS workloads, optimize in a follow-up.
  This ADR commits to "ship correctly first," matching ADR-0043's
  oracle-first posture.
- **TIDE-on-OKS reconsideration.** ADR-0024 stays in force. The
  structural objection to TIDE-on-OKS does not transfer to LRP-on-OKS;
  it has not changed for TIDE itself.
- **Cross-paradigm panoptic-keypoints LRP.** Panoptic LRP (ADR-0043)
  is defined for stuff-and-things on segmentation kernels. Panoptic
  with keypoints is not a paradigm vernier supports; out of scope.

## Pros and cons of the options

### (1) Defer LRP-on-OKS

- 👍 Mirrors ADR-0024's deferral; surface symmetry across the two
  keypoint-diagnostic ADRs.
- 👎 The structural reasoning that justified ADR-0024 does not
  apply to LRP. Deferring on symmetric appearance would defer a
  metric that *does* fit OKS workloads.
- 👎 Keypoints users get no deployable-threshold answer at all in
  0.5.x. The Phase-3 per-keypoint analysis answers a different
  question; deferring LRP leaves a real gap.
- 👎 Invites the wrong-shape reading ADR-0024 was careful to
  avoid: users would see "vernier supports keypoints for AP but
  not for LRP" and conclude there is a structural problem with
  LRP-on-OKS, when in fact LRP is the metric whose shape fits OKS
  best of any in the project.

### (2) Ship LRP-on-OKS (chosen)

- 👍 Diagnostic surface uniform across all four supported kernels.
- 👍 Honest about the structural distinction: TIDE's bin-geometry
  problem on OKS is real; LRP's continuous-similarity geometry on
  OKS works. Documenting the distinction is cheaper than carrying
  a "deferred for symmetry" gap.
- 👍 Complements the Phase-3 per-keypoint analysis cleanly; the two
  answer different questions, both can ship.
- 👎 The ADR-0024 / ADR-0045 pair requires explanation. We accept
  the documentation cost.
- 👎 Boundary / keypoints `tp_threshold` defaults (ADR-0044) ship
  as "tentative" on the same 0.5.x follow-up gate. Shipping the
  keypoints kernel locks the tentative default into the published
  surface; revising is a 0.5.x patch. Acceptable per
  `project_release_pace.md`.

## Links and references

- ADR-0001 — Record architecture decisions.
- ADR-0012 — OKS keypoints surface (the kernel this ADR ships
  LRP on).
- ADR-0024 — TIDE on keypoints — deferred (the precedent this
  ADR explicitly distinguishes from).
- ADR-0029 — Per-paradigm submodule namespace (places the entry
  point at `vernier.instance.optimal_lrp(..., iou=Keypoints(...))`).
- ADR-0043 — LRP oracle and namespace (the parent decision; this
  ADR is the kernel-coverage extension).
- ADR-0044 — LRP thresholds and tau grid (where the keypoints
  `tp_threshold` default is committed).
- Oksuz, Cam, Akbas, Kalkan, "One Metric to Measure them All:
  Localisation Recall Precision (LRP) for Evaluating Visual
  Detection Tasks" (TPAMI 2021) — the paper that uses LRP on OKS
  in its keypoints sections.
- Phase-3 keypoints track (memory: `project_keypoints_track.md`) —
  where the per-keypoint OKS contribution analysis ADR-0024 named
  will be specified.
