# ADR-0044: LRP — tau grid resolution and per-kernel TP thresholds

- **Status:** proposed
- **Date:** 2026-05-14
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

ADR-0043 commits to LRP / oLRP as a 0.5.x deliverable under the
`vernier.instance.optimal_lrp` and `vernier.panoptic.optimal_lrp`
entry points. The oracle and namespace are settled. What remains is
the *defaults question*: what numbers does the user get when they
call `optimal_lrp(dataset, dt)` with no kwargs?

Two parameters drive the answer:

- **The tau grid.** The search over confidence thresholds is discrete:
  we evaluate LRP at every `tau` in a grid spanning `[0.0, 1.0]` and
  pick the argmin. The grid resolution trades off precision of the
  reported `tau` against runtime cost. Too coarse and the reported
  threshold is wrong by a deployment-relevant fraction; too fine and
  the panoptic side (which already pays a 2× matching cost per
  ADR-0043) compounds the bill.
- **The per-kernel TP threshold.** Before the tau sweep enters, each
  matched (DT, GT) pair carries an IoU (or boundary-IoU, or OKS) score
  from the matching engine. The TP threshold is the cutoff above which
  a matched pair counts as a TP at all — the kernel-level "this is a
  match" line that the tau search then operates over. The Oksuz TPAMI
  2021 paper picks `0.5` for IoU on COCO bbox and stops there. Boundary
  IoU and OKS have different distribution shapes and the question is
  whether `0.5` transfers.

The user-experience contract is the same as ADR-0022: the defaults the
user gets must be defensible per-kernel, reproducible from the report,
and overridable per call without boilerplate.

## Decision drivers

- **Paper-faithfulness for bbox.** Users who run oLRP on bbox-COCO and
  compare to numbers from the TPAMI 2021 paper or from
  `kemaloksuz/LRP-Error` (per the ADR-0043 tripwire) must get the
  paper's defaults. Anything else makes the headline numbers look
  wrong on contact.
- **Per-kernel honesty.** A default that mis-thresholds boundary or
  OKS is worse than no default. Reported numbers carry vernier's
  authority; a wrong default is a wrong-by-our-fault number. Same
  driver as ADR-0022.
- **Grid resolution must be deployment-relevant.** The reported `tau`
  is what a practitioner would set on the model. A grid step of `0.05`
  reports a threshold accurate to ±0.025, which is large in production
  terms — confidence cutoffs of `0.45` vs `0.50` materially shift
  detection counts. A step of `0.01` is the natural deployment
  granularity (operators tune thresholds in 1% increments) and is the
  smallest step the search costs do not punish.
- **Reproducibility in the report.** The resolved `(tp_threshold,
  tau_grid)` lives in the result's config field, like ADR-0022's
  `(t_f, t_b)`. Screenshots of an oLRP table are re-derivable from
  the table alone.
- **Override ergonomics.** The ratio of "user accepts default" to
  "user overrides" will be high; default + per-call kwarg is enough,
  no global config knob. Same as ADR-0022.
- **Defaults must be defended.** Each default carries a one-line
  rationale in this ADR — not a magic constant in code.

## Considered options

1. **Single `(tp_threshold, tau_step)` across all kernels.** Bolya-style
   uniform default. Simple. Wrong-shape for OKS (different scale) and
   tentative for boundary.
2. **Per-kernel `(tp_threshold, tau_step)` with paper-anchored bbox/segm
   and defensibly-anchored boundary/keypoints.** Four numbers in a
   table; each defended.
3. **No default for `tp_threshold`; require it.** Forces the user to
   think. Loses to copy-paste from README defaults the user does not
   read.
4. **Per-kernel × per-parameter table** (e.g., boundary's
   `tp_threshold` as a function of `dilation_ratio`). More accurate;
   more surface; revisit later.

## Decision outcome

Chosen option: **(2) per-kernel `(tp_threshold, tau_step)` with
paper-anchored defaults for bbox/segm and defensibly-anchored
defaults for boundary/keypoints.**

### Tau grid

The tau grid is `0.01` step over `[0.0, 1.0]` by default — 101 points
inclusive of both endpoints. Step `0.01` matches the deployment
granularity practitioners tune confidence cutoffs at and is the
smallest step the panoptic-side matching cost does not punish. The
search dominates the per-class component computation in big-O but is
linear in the grid size; doubling grid resolution doubles the search
loop, not the matching pass.

Per-kernel overrides go through a new function `lrp::defaults_for(kernel:
&EvalIouType) -> LrpDefaults` mirroring `tide::defaults_for(kernel)`
from ADR-0022 line 104-106. The defaults are not Python-side constants;
they live with the algorithm.

### Per-kernel TP threshold

The IoU/OKS cutoff above which a matched pair counts as a TP, before
tau enters the search:

| Kernel                              | `tp_threshold` default | Anchored on                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|-------------------------------------|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Bbox`                              | `0.5`                  | Oksuz TPAMI 2021, COCO bbox — the paper's recommended operating point and the same `0.5` reading users have intuition for from AP@0.5.                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `Segm`                              | `0.5` (tentative)      | Anchored on the bbox row by extrapolation, **not** by direct measurement on COCO val2017 — vernier does not ship the COCO val dataset in CI per the licensing policy in `project_coco_val_regression.md`, so the empirical histograms the original plan called for are not derivable from the in-tree fixtures. Segm IoU is bounded above by bbox IoU on the same instance and tracks within ~10% on standard detection models, so reusing `0.5` is the least-surprising default. Revisit in a 0.5.x follow-up once whole-dataset parity infrastructure (planned post-Week 5) is wired.                      |
| `Boundary(dilation_ratio=0.02)`     | `0.5` (tentative)      | Anchored on the kernel's *reported* anchor, not on a recalibrated boundary scale. The `0.5` threshold is the recommended LRP operating point per Oksuz TPAMI 2021; we apply it to boundary IoU directly rather than rescaling. ADR-0022 carved a tighter `t_b` floor for boundary because TIDE's bin geometry uses the threshold differently (it separates "almost matched" from "background"); LRP's `tp_threshold` is the "this is a TP" line, not a phase-diagram cutoff, and the paper's `0.5` is the right anchor for it. Empirical anchoring on real models is deferred to a 0.5.x follow-up.          |
| `Keypoints(...)`                    | `0.5` (tentative)      | Anchored on OKS = `0.5` as the reported operating point — consistent with `0.5` being LRP's recommended operating point per Oksuz TPAMI 2021 and with the OKS scale's "this is a meaningful match" reading. The ADR-0045 argument for shipping LRP-on-OKS rests on `oLRP_Loc = 1 − mean(OKS on TPs)` being well-defined for OKS; the TP threshold for FP/FN counting is this defaults question, handled here. Empirical anchoring on real keypoint models is deferred to the same 0.5.x follow-up as boundary/segm.                                                                                          |

This ADR remains `proposed`: the segm (`0.5`), boundary (`0.5`), and
keypoints (`0.5`) rows are defended-by-extrapolation defaults, not
measurement-anchored ratifications. Promoting to `accepted` is gated
on the deferred 0.5.x follow-up that runs the empirical sweeps on
real models against COCO val2017 once the whole-dataset parity
infrastructure is wired (planned post-Week 5 — see
`project_coco_val_regression.md`).

The `Boundary` default is keyed to `dilation_ratio=0.02` (the COCO
default). For other `dilation_ratio` values (LVIS uses `0.008`), the
same `tp_threshold=0.5` default applies but the user is expected to
override based on their kernel's distribution. The tutorial
(`docs/tutorials/debugging-with-lrp.md`) shows how to choose.

### Decision gate (boundary and keypoints defaults)

The empirical-anchoring step ADR-0022 deferred for boundary `t_b`
applies the same way here for boundary and keypoints `tp_threshold`:
the measurement is infeasible from inside CI — vernier never commits
COCO val data (`project_coco_val_regression.md`: license restriction)
and there is no out-of-CI infrastructure available at this point in
the release line. The defaults shipped here are defensible-on-paper-
anchoring; the 0.5.x follow-up ratifies them against real-model
sweeps. Per `project_release_pace.md` the user is on 0.0.x patch
releases; revising a default in 0.5.x is cheap.

### Defaults plumbing

The defaults live in a single Rust function `lrp::defaults_for(kernel:
&EvalIouType) -> LrpDefaults` in the `lrp` module, surfaced through
FFI as the `None` resolver in `vernier.instance.optimal_lrp(...,
tp_threshold=None, tau_grid=None)`. `LrpDefaults` carries
`tp_threshold: f64` and `tau_grid: TauGrid` where `TauGrid` is either
the `0.01`-step default or a user-supplied `Vec<f64>` of explicit
thresholds.

The result's `config` field always carries the resolved
`(tp_threshold, tau_grid)` and the kernel — no "the default may have
been ..." ambiguity in any recorded report. Same convention as
ADR-0022.

### Consequences

- **Positive.** Bbox oLRP numbers from `vernier.instance.optimal_lrp`
  reproduce the paper directly. Segm / boundary / keypoints defaults
  are defended in this ADR rather than copy-pasted. The report carries
  reproducibility for free. The tau grid step matches deployment
  granularity, so the reported `tau` is directly actionable.
- **Negative.** Four numbers to keep in the head (one is the same as
  the others, but each is defended on its own row). Any kernel added
  later needs a defaults entry and a defense. The boundary and
  keypoints rows age with model architecture if score distributions
  shift; revisit on the 0.5.x follow-up.
- **Neutral.** Users who care override per call; the override syntax
  is `tp_threshold=0.6, tau_grid=np.linspace(0, 1, 51)` and that is
  the entire ergonomic cost. The defaults are starting points, not
  commitments.

## What this ADR explicitly does not decide

- **Whether tau is user-overridable vs. always-optimal.** Default is
  always-optimal (the "o" in oLRP). Making it configurable is a
  separate decision after real-data sweeps exist — once users have
  shipped models against the metric, the question of "I want LRP at
  *my* deployed threshold, not the optimal one" gets a real answer.
  Until then we punt.
- **Oracle and namespace.** ADR-0043.
- **LRP for keypoints — ship or defer.** ADR-0045 (ship).
- **LaECE integration.** ADR-0018 / out of scope for 0.5.x.
- **Defaults for non-COCO `dilation_ratio` boundary.** Tutorial
  guidance, not an ADR commitment. Same as ADR-0022.
- **Empirical anchoring on three reference models.** Gated on the
  whole-dataset parity infrastructure; 0.5.x follow-up.
- **Multi-class tau aggregation.** Tau is per-class per ADR-0043.
  Whether the result table also reports a dataset-level tau (e.g.,
  median over classes, or per-frequency-bucket) is a result-tables
  question, not a thresholds question.

## Pros and cons of the options

### (1) Single `(tp_threshold, tau_step)` across all kernels

- 👍 One number to remember. Simplest possible API.
- 👎 Boundary IoU and OKS have different distribution shapes from
  bbox IoU. The `0.5` reading translates well enough at the paper-
  recommended operating point, but the *next* threshold the user
  asks about (`0.7`, `0.9`) does not transfer; not surfacing that
  per-kernel hides the asymmetry.

### (2) Per-kernel defaults (chosen)

- 👍 Honest. Each default defended on the kernel's own scale. Bbox
  matches paper.
- 👍 Override surface is one kwarg; defaults table is small.
- 👍 Decision gate documents when to revise (empirical follow-up
  on the 0.5.x line).
- 👎 Four numbers, three of which (segm / boundary / keypoints) age
  with model architecture if score distributions shift. Configurable
  per call is the answer; we accept the defaults as starting points.
- 👎 The boundary default depends on `dilation_ratio`; we pin to the
  COCO default and document the rest. Imperfect but bounded.

### (3) No default for `tp_threshold`; require it

- 👍 Forces the user to choose. No silent wrong-thresholding.
- 👎 The default value the user picks will be whatever the README
  says. We have not removed the defaults problem; we have externalized
  it to the README. Worse, the report cannot carry "the default"
  because there is not one — every screenshot needs the caller to
  remember which number they passed.

### (4) Per-kernel × per-parameter table

- 👍 Most accurate. `tp_threshold` as a function of `dilation_ratio`
  for boundary, of OKS sigmas for keypoints, is the principled
  answer.
- 👎 Surface explosion. The defaults-for-this-call computation
  becomes its own subroutine with its own tests. Diminishing returns
  past four kernels and one parameter; revisit if a real workload
  surfaces a case where this matters.

## Links and references

- ADR-0001 — Record architecture decisions.
- ADR-0010 — Boundary IoU isolated subsystem (the kernel whose
  `dilation_ratio` parameter scopes the boundary default).
- ADR-0012 — OKS keypoints surface (the kernel the keypoints row
  applies to).
- ADR-0022 — TIDE thresholds and per-kernel defaults (the structural
  template this ADR mirrors; the substantive difference is that LRP's
  TP threshold is a "this is a TP" line, not a phase-diagram cutoff,
  so the boundary row anchors differently).
- ADR-0043 — LRP oracle and namespace (the harness that pins
  threshold semantics regardless of value).
- ADR-0045 — LRP on keypoints — shipped (the decision that makes
  the keypoints row in this defaults table load-bearing).
- Oksuz, Cam, Akbas, Kalkan, "One Metric to Measure them All:
  Localisation Recall Precision (LRP) for Evaluating Visual
  Detection Tasks" (TPAMI 2021) — the paper whose `0.5` operating
  point this ADR adopts.
- Companion data file `docs/explanation/lrp-and-its-limits.md` — the
  user-facing prose on what these defaults imply.
