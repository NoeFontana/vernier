# ADR-0024: TIDE on keypoints (OKS) — deferred

- **Status:** proposed
- **Date:** 2026-05-02
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

vernier 0.5.0 ships TIDE error decomposition (ADR-0021/0022/0023) for
three kernels: bbox, segm, boundary. Keypoints (OKS, ADR-0012) is the
fourth supported kernel; the question this ADR answers is whether TIDE
ships for it in 0.5.0.

TIDE was designed around bbox detection on multi-class benchmarks
(ADR-0021): all six bins are defined in terms of IoU between detections
and GTs. Two of them — Cls and Both — are *cross-class* (ADR-0023),
requiring IoU against GTs of *other* classes.

Keypoints on COCO is single-class: only the `person` category carries
keypoint annotations. There is no other class for a `person` detection
to be confused with, so the Cls and Both bins are structurally empty —
not empirically rare, but algorithmically undefined for the only
dataset keypoints users actually have.

The remaining four bins (Loc / Dupe / Bkg / Miss) are well-defined for
single-class workloads, but sit on top of OKS, which is not IoU. OKS
is a per-keypoint Gaussian similarity weighted by per-category sigmas;
the `(t_b, t_f)` phase diagram that TIDE's bin assignment relies on
does not carve the same error geometry on OKS as on IoU. There is no
published TIDE-on-OKS convention; we would be inventing thresholds
with no reference to defend.

The question keypoints users actually ask in production is not "is
this a Loc or Bkg error" — it is "are limbs (low-sigma keypoints)
the limiting factor, or torso (high-sigma keypoints)?" That is a
per-keypoint OKS contribution analysis, a different shape of diagnostic
that TIDE does not produce.

## Decision drivers

- **Structural fit, not effort.** Cls and Both being structurally zero
  means a published TIDE-on-keypoints number is half-decoration. Users
  who read the report would interpret the four nonzero bins through a
  "TIDE-trained" intuition built on bbox; the headline ΔmAP per bin
  would invite the wrong conclusion.
- **No reference to defend defaults against.** ADR-0022 anchors `t_f`
  on the TIDE paper for bbox and on empirical measurement for boundary.
  For keypoints there is no paper anchor and no community precedent;
  any number we ship is invented.
- **The right diagnostic for keypoints is not TIDE.** Per-keypoint OKS
  contribution analysis answers the question keypoints users have. It
  is a separate capability with its own shape (per-keypoint score
  decomposition), not a parameter on TIDE.
- **Project pace.** Per memory `project_release_pace.md`, vernier is
  on 0.0.x patches; defer-with-rationale is a cheaper artifact than
  a 600-line implementation that ships wrong-shape numbers.
- **Single-class workloads are growing (synthetic, robotics) but
  remain niche on the keypoints surface.** Phase-3 keypoints work
  (per memory `project_keypoints_track.md`) is gated on ADR-0012
  follow-up PRs; the right time to revisit keypoints diagnostics is
  alongside that track, not bolted into the 0.5.0 TIDE release.

## Considered options

1. **Ship TIDE-on-OKS anyway in 0.5.0.** Implement Loc/Dupe/Bkg/Miss
   over OKS with invented thresholds; hard-zero the Cls and Both
   bins; document the limitations in the explanation page.
2. **Defer with no follow-up plan.** TIDE for keypoints simply not
   on the roadmap.
3. **Defer with an explicit alternative track in Phase 3.** TIDE
   for keypoints is not implemented; per-keypoint OKS contribution
   analysis is the planned follow-up, scoped separately under the
   ADR-0012 keypoints track.

## Decision outcome

Chosen option: **(3) defer with an explicit alternative track.**

The 0.5.0 release ships `vernier.error_decomposition(dataset, dt, *,
iou=Bbox() | Segm() | Boundary(...))`. Passing `iou=Keypoints(...)`
raises `NotImplementedError` with a message pointing at this ADR
and at the planned Phase-3 alternative.

The alternative — per-keypoint OKS contribution analysis — is added
to the Phase-3 keypoints track as a non-blocking follow-up. Its
shape, scope, and ADR live there, not here. This ADR commits to
*not shipping TIDE-on-OKS in 0.5.0* and to documenting the
alternative diagnostic as the answer for keypoints users; the
question is revisited when Phase-3 keypoints work decides on the
per-keypoint contribution diagnostic.

The 0.5.0 changelog calls this out explicitly: "TIDE supports bbox,
segm, boundary; keypoints is structurally a different question and
gets its own diagnostic in Phase 3."

### Consequences

- **Positive.** No invented thresholds. No half-zero bin reports
  inviting wrong-shape interpretation. Phase-3 keypoints work has
  room to design the right diagnostic for the question keypoints
  users ask. Documentation effort (this ADR + a paragraph in the
  explanation page) is far less than implementation + maintenance
  of a TIDE-on-OKS path that ships ambiguous numbers.
- **Negative.** Keypoints users who specifically want a TIDE-shaped
  report for parity with bbox workflows will have to wait. We accept
  this; the alternative is shipping a misleading capability.
- **Neutral.** `NotImplementedError` is one branch in the
  top-level `vernier.error_decomposition` dispatch. Trivial code
  cost, explicit user surface.

## What this ADR explicitly does not decide

- **Per-keypoint OKS contribution analysis specifics.** Belongs in
  the Phase-3 keypoints track, with its own ADR.
- **TIDE on multi-category keypoints datasets** (custom datasets
  that annotate keypoints on multiple categories). Out of scope
  until such a dataset is the target workload; the structural
  argument for Cls/Both being zero is COCO-specific, not OKS-specific.
- **Whether OKS-without-TIDE is a complete diagnostic story for
  keypoints users.** The Phase-3 follow-up answers this; this ADR
  only commits to "TIDE is not the answer."
- **Future TIDE generalizations for non-IoU similarities.** If the
  TIDE methodology grows a multi-class OKS variant in the literature,
  vernier revisits this ADR. Until then, no speculative
  implementation.

## Pros and cons of the options

### (1) Ship TIDE-on-OKS anyway

- 👍 Surface symmetry across all four kernels in 0.5.0.
- 👎 Cls and Both are structurally zero on COCO; users read the
  report through a bbox-trained lens and reach wrong conclusions.
- 👎 Loc/Dupe/Bkg/Miss thresholds on OKS have no paper or community
  anchor; the defaults are invented and undefendable per the ADR-0022
  decision-driver list.
- 👎 ~600 lines of implementation effort for a diagnostic the target
  user does not actually want.

### (2) Defer with no follow-up plan

- 👍 Smallest commitment.
- 👎 Leaves keypoints users with no diagnostic story at all.
  vernier becomes "good for bbox / segm, silent on keypoints."
- 👎 No signal to Phase-3 work that keypoints diagnostics is on the
  radar.

### (3) Defer with explicit alternative track (chosen)

- 👍 Honest about the structural mismatch.
- 👍 Names the right diagnostic (per-keypoint OKS contribution) so
  Phase-3 has a concrete target rather than a void.
- 👍 0.5.0 ships the capabilities it can defend; later releases
  ship the keypoints diagnostic that fits the question.
- 👎 Two diagnostics for two domains (TIDE for bbox/segm/boundary,
  per-keypoint contribution for keypoints) instead of one. Accepted
  as the correct shape — different questions deserve different tools.

## Links and references

- ADR-0001 — Record architecture decisions.
- ADR-0012 — OKS keypoints surface (the kernel this ADR declines to
  decompose with TIDE).
- ADR-0021 — TIDE numpy oracle and correctness model (the methodology
  this ADR scopes).
- ADR-0022 — TIDE thresholds and per-kernel defaults (the defaults
  framework this ADR refuses to invent values for in the OKS case).
- ADR-0023 — Cross-class IoU strategy (the cross-class machinery
  Cls/Both rely on, which is structurally undefined for single-class
  COCO keypoints).
- Bolya et al., "TIDE: A General Toolbox for Identifying Object
  Detection Errors" (ECCV 2020) — the methodology designed around
  multi-class IoU, not OKS.
- Phase-3 keypoints track (memory: `project_keypoints_track.md`) —
  where per-keypoint OKS contribution analysis will be specified.
