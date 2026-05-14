# ADR-0043: LRP / oLRP — numpy oracle, kemaloksuz tripwire, and cross-paradigm namespace

- **Status:** proposed
- **Date:** 2026-05-14
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

vernier 0.5.x introduces LRP / oLRP (Oksuz et al., ECCV 2018; TPAMI 2021):
a single-number detection metric that decomposes into three orthogonal
components — `oLRP_Loc` (localization error on TPs), `oLRP_FP` (false-
positive rate at the optimal threshold), `oLRP_FN` (false-negative rate
at the optimal threshold) — and ships a *per-class deployable confidence
threshold* `tau` as a first-class output. The "optimal" in oLRP is the
infimum over a sweep of confidence thresholds: for each class we search
`tau ∈ [0.0, 1.0]` and report the threshold that minimizes LRP, alongside
the components evaluated at that threshold. The headline deliverable is
not the number alone — it is the *(number, threshold)* pair, because
`tau` is what a practitioner would set on the model to get the reported
behavior.

LRP is structurally a *different shape of metric* from AP and from TIDE:
it is a continuous IoU score (`1 − mean(IoU)` over TPs, not a binned
"close enough at this threshold") combined with FP/FN rates at a swept
operating point. That continuous-IoU posture means the same problems
ADR-0021 flagged for TIDE apply — there is no canonical numerical
reference, and unit-testing a Rust implementation against itself is
circular. But it also opens one new question TIDE did not face: a
first-party reference implementation exists. `kemaloksuz/LRP-Error`
(Anaconda Oksuz, same author as the paper) is the published code that
accompanies the TPAMI 2021 paper. It is not pycocotools-shaped — there
is no version-pinned community standard — but it is the closest thing
to a "canonical executable" the metric has.

We need a correctness model that (a) is debuggable on a whiteboard like
ADR-0021's TIDE oracle, (b) acknowledges the first-party reference
exists without inheriting its bugs as our contract, (c) places the
public symbols consistent with ADR-0029's per-paradigm namespace, and
(d) decides whether panoptic LRP reuses PQ's match list or runs its
own pass.

## Decision drivers

- **Correctness must be diff-able.** Same as ADR-0021: reviewers need
  to point at a known-good number and say "the Rust differs here." The
  LRP component computations have at least three numerical traps — the
  `1 − mean(IoU)` averaging is undefined when `TP = 0`, the tau search
  ties non-trivially, and the FN counting at non-zero tau requires
  retaining the matched-but-below-tau detections separately from the
  unmatched ones. These compose silently.
- **First-party reference, research-code quality.** Oksuz wrote both
  the paper and `kemaloksuz/LRP-Error`. The standing question that
  ADR-0021 did not have to answer is: does first-party origin earn
  bitwise-parity status? We say no. `LRP-Error` is research code; it
  has documented quirks (single-class behavior at low recall, the
  argmin-tau tie-handling is unspecified in the code), and its repo
  is not versioned with the discipline pycocotools is. But it is too
  useful to *ignore* — it catches whole-class regressions a synthetic
  oracle would not.
- **The three-tier disposition model (ADR-0002) does not apply.** Same
  reasoning as ADR-0021 line 46-48. `strict / corrected` is keyed to
  "reproduce pycocotools or knowingly diverge." LRP has no
  `pycocotools` analogue. Inheriting the three-tier surface for one
  metric would dilute the meaning of "strict" for AP and invite users
  to think `LRP-Error` is to LRP what pycocotools is to AP, which it
  is not.
- **Namespace consistency.** ADR-0029 places paradigm-specific symbols
  under `vernier.instance` / `vernier.panoptic` / `vernier.semantic`.
  LRP is well-defined for instance and panoptic; semantic LRP is not
  a concept the literature speaks about. Each paradigm gets its own
  entry point under the namespace established for it.
- **Panoptic matching is not free.** PQ's match list is built with a
  hard IoU > 0.5 cutoff (the panopticapi convention). LRP's tau
  search requires the *full* matched-IoU list, not the > 0.5 subset —
  the moment you collapse below 0.5 into "unmatched", the tau sweep
  becomes degenerate above 0.5 and the per-class deployable threshold
  (the metric's headline deliverable) disappears. Reusing the PQ
  match list would ship a wrong-shape number.
- **The user-facing API does not surface the IoU-retention flag.**
  Per ADR-0019, `EvaluateParams::retain_iou` is an engine-internal
  switch; the LRP entry point sets it on the caller's behalf.

## Considered options

1. **Bitwise parity with `kemaloksuz/LRP-Error`.** Treat the published
   repo as a pinned reference; build an LRP-quirks survey paralleling
   pycocotools-quirks; ship strict / corrected dispositions per
   divergence.
2. **Algorithmic parity with the TPAMI 2021 paper.** Implement what the
   paper says; assert against intuition and small examples; no
   executable reference.
3. **Numpy oracle as the correctness contract + kemaloksuz tripwire
   as CI sanity gate.** A pure-numpy reference implementation at
   `tests/python/oracle/lrp/oracle.py` is the spec; the Rust
   implementation is correct iff it agrees with the oracle within
   `1e-9` per component per fixture. `kemaloksuz/LRP-Error` is vendored
   commit-pinned as a test-only artifact and ONE CI fixture asserts
   `|oracle − kemaloksuz| < 1e-6` on a synthetic case where both ought
   to agree. The tripwire is a sanity gate, not a parity contract.
4. **No oracle; trust unit tests.** Test component computations in
   isolation; trust composition.

## Decision outcome

Chosen option: **(3) numpy oracle as the correctness contract, with a
commit-pinned `kemaloksuz/LRP-Error` tripwire on a single CI fixture.**

### Oracle policy

The oracle lives at `tests/python/oracle/lrp/oracle.py` (pure numpy,
no vernier imports). It takes a `(gt, dt, similarity_fn, tp_threshold,
tau_grid)` tuple and returns `(oLRP, oLRP_Loc, oLRP_FP, oLRP_FN,
tau)` per class plus the dataset-level reduction. It is correct by
construction: a literal transcription of the TPAMI 2021 equations,
no SIMD, no perf optimizations, the tau sweep done as a Python `for`
loop over the grid. Mirrors ADR-0021's pattern.

Oracle correctness is pinned by hand-computed assertions on small
fixtures at `tests/python/oracle/lrp/test_oracle.py`: an all-perfect
case (`oLRP = 0, tau = min(scores)`), an all-FP case
(`oLRP = 1.0, tau = NaN`), a single-TP-with-localization-error case
(`oLRP_Loc = 1 − IoU` exactly), a tied-tau case (verifies the
argmin tie-break — see below), and a single-class-no-TPs case
(`oLRP = 1.0, tau = NaN`, flagged in the result table).

The Rust implementation's correctness contract is
`tests/python/oracle/lrp/test_rust_matches_oracle.py`: it runs the
Rust `optimal_lrp` entry point and the oracle on the same fixture
and asserts `|delta_rust − delta_oracle| < 1e-9` per component per
class per fixture across the supported kernels.

### kemaloksuz tripwire

`kemaloksuz/LRP-Error` is vendored commit-pinned at
`tests/python/oracle/lrp/vendor/lrp_error/`. The vendoring is
test-only: the directory is excluded from the wheel build and from
the Rust workspace. ONE CI fixture cross-checks
`|oracle − kemaloksuz| < 1e-6` on a synthetic case where both ought
to agree (typically the all-perfect or single-class-no-FPs case).

**This is a sanity gate, not a parity contract.** Where the two
diverge on real data — and they will, because `LRP-Error` is research
code with unspecified tie-handling and known low-recall edge cases —
the numpy oracle is authoritative. The tripwire exists to catch the
class of failure where someone unintentionally reorders the tau
sweep or transposes a component formula and the oracle silently
agrees with itself; it is *not* a contract to reproduce `LRP-Error`'s
quirks.

This is the substantive departure from ADR-0021. ADR-0021's
spot-check-during-Week-1 (line 128-130) is a one-shot;
this tripwire is permanent CI. We accept it because the reference is
first-party (Oksuz wrote both the paper and the repo) and the
single-fixture cost is bounded. We explicitly reject pycocotools-style
pinning of `LRP-Error` because:

- `LRP-Error` is research code; pinning encodes its bugs as our
  "correct" behavior, then forces every fix to ship as a "corrected"
  disposition.
- The three-tier disposition model (ADR-0002) breaks for one metric.
  Telling users a `strict` LRP fixture is calibrated against
  `kemaloksuz` while a `strict` AP fixture is calibrated against
  `pycocotools` would invent two parallel definitions of "strict."
- `LRP-Error` has no community calibration story — no downstream
  consumer expects bitwise reproduction of its outputs.

### Three-tier disposition (ADR-0002) does not apply

LRP has no `pycocotools` analogue. The `kemaloksuz` tripwire is a
sanity gate, not a calibration target. The disposition machinery
remains AP-specific.

### Namespace

LRP entry points live under the per-paradigm submodules per ADR-0029:

- `vernier.instance.optimal_lrp(dataset, dt, *, iou=Bbox() | Segm()
  | Boundary(...) | Keypoints(...))` — returns the per-class component
  table and dataset-level reduction.
- `vernier.panoptic.optimal_lrp(dataset, predictions)` — same shape,
  defined over the panoptic match space. The entry point ships now
  as a typed `NotImplementedError` stub (see "Panoptic LRP ships as
  a stub" below) — the namespace placement is pinned today; the
  computation lands once panoptic predictions carry per-segment
  scores. This is the first TIDE-shape decomposition to be wired
  on the panoptic paradigm at all (TIDE itself is instance-only per
  ADR-0021 / ADR-0024).

### Panoptic LRP ships as a stub; full implementation gated on scored predictions

The algorithmic decision is unchanged: panoptic LRP must run its
own matching pass and **not** reuse PQ's match list. PQ's matching
applies a hard IoU > 0.5 cutoff before counting a pair as matched;
the LRP tau search needs the full matched-IoU list at every threshold
the sweep visits, including pairs that PQ would have dropped.
Collapsing below 0.5 into "unmatched" makes the tau sweep degenerate
above 0.5 — every threshold yields the same FP/FN counts — and
removes the per-class deployable threshold, which IS the metric's
headline deliverable.

What ships today: `vernier.panoptic.optimal_lrp` is registered in
the panoptic namespace and raises `NotImplementedError` with a
remediation pointer back to this ADR. Reason: the current panoptic
prediction wire format (`vernier_panoptic::dataset::SegmentInfo`)
carries `id`, `category_id`, `iscrowd`, `area` — **no per-segment
score**. Without scores, the tau sweep has nothing to scan, so the
parallel matching pass cannot produce the per-class deployable
threshold that defines the metric.

What lands the real implementation: a follow-up ADR that extends
the panoptic prediction format with per-segment scores. When that
lands, this entry point fills in. The 2× matching cost noted earlier
applies at that point — users who only call
`vernier.panoptic.Evaluator.evaluate()` continue to pay nothing.

Why ship the stub now rather than wait: pinning the namespace
`vernier.panoptic.optimal_lrp` today keeps the cross-paradigm
symmetry decided in ADR-0029 visible in the surface, gives users a
clear error message with an ADR pointer rather than an
`AttributeError` mystery, and means the prediction-format ADR
doesn't also have to negotiate naming.

### Tau semantics

Defaults (tau grid resolution, per-kernel TP thresholds) are not
this ADR's decision; ADR-0044 handles them. This ADR fixes only the
semantics:

- **Per-class tau, not global.** The optimal threshold is a per-class
  output. Aggregating across classes is a downstream choice the user
  makes from the table.
- **argmin-tau tie resolution: larger tau wins.** When two thresholds
  produce the same LRP, we report the larger one. The deployment
  reading is "fewer FPs at the operating point"; that is the more
  conservative choice and matches what a practitioner who saw "either
  threshold gives the same score" would default to.
- **Empty-TP classes.** Classes with zero TPs at every tau in the grid
  report `tau = NaN, oLRP = 1.0` and are flagged in the result table.
  The component-level fields (`oLRP_Loc`, `oLRP_FP`, `oLRP_FN`) are
  reported as `NaN` for those classes; the dataset-level reduction
  excludes them from the mean and reports the empty-TP class count
  separately.

Tau is reported in the per-class section of the result table.

### Consequences

- **Positive.** Reviewers diff against a numpy reference they can
  read in an afternoon. The kemaloksuz tripwire catches whole-class
  regressions a synthetic oracle would not (e.g., someone refactors
  the tau sweep and the oracle adapts in agreement). Panoptic gains
  its first TIDE-shape decomposition; the panoptic paradigm becomes
  a real peer on the diagnostic surface, not just on the headline-
  number surface.
- **Negative.** Two implementations of LRP in the repo (numpy + Rust)
  plus a vendored third for the tripwire. Drift risk: a behavior
  change to one but not the others ships either a parity miss
  (caught by the oracle test) or a tripwire failure (caught by the
  vendored test). The tripwire failure case requires investigation
  on every spurious agree-to-disagree because the resolution is
  *always* "oracle wins" but the cost of confirming is non-zero.
  Panoptic LRP doubles the matching cost when requested. Vendoring
  research code carries the ambient cost of keeping the pin alive
  through Python and dependency upgrades.
- **Neutral.** vernier's oLRP numbers will not match `LRP-Error`'s
  on real models. This is a design output, not a defect. Documented
  in `docs/explanation/lrp-and-its-limits.md`.
- **Neutral.** The LRP entry point sets `retain_iou=true` internally
  on `EvaluateParams` (`crates/vernier-core/src/evaluate.rs:255`).
  Not a user-facing flag.

## What this ADR explicitly does not decide

- **Tau grid resolution and per-kernel TP-threshold defaults.**
  ADR-0044.
- **LRP for keypoints (OKS kernel).** ADR-0045. The structural
  argument against TIDE-on-OKS (ADR-0024) does not transfer.
- **LaECE integration** (`fiveai/detection_calibration` style).
  Calibration is ADR-0018; integrating LaECE into the LRP result
  table is out of scope for 0.5.x.
- **Real-data smoke against `kemaloksuz` on full COCO val2017.** Tied
  to deferred `project_coco_val_regression.md` infrastructure, same
  out as ADR-0022 line 124-130: vernier does not commit COCO val
  data in CI per the licensing policy, and there is no out-of-CI
  infrastructure available at this point in the release line. The
  whole-dataset real-model cross-check is a 0.5.x follow-up.
- **Whether tau is user-overridable.** Default is always-optimal
  (the "o" in oLRP); making it configurable is a separate decision
  after real-data sweeps exist. ADR-0044 punts the same question.
- **LRP for semantic segmentation.** Not a concept the literature
  defines; out of scope unless and until it is.
- **The panoptic prediction format extension.** Adding per-segment
  scores to `vernier_panoptic::dataset::SegmentInfo` and the
  corresponding JSON wire format is its own ADR. Until it lands,
  `vernier.panoptic.optimal_lrp` raises `NotImplementedError` per
  the "Panoptic LRP ships as a stub" section above.

## Pros and cons of the options

### (1) Bitwise parity with `kemaloksuz/LRP-Error`

- 👍 First-party reference; the same author wrote both the paper and
  the code.
- 👎 `LRP-Error` is research code, not a calibration target.
  Pinning encodes its bugs as vernier's "correct" behavior; future
  fixes ship as "corrected" dispositions users have to opt into to
  get a mathematically-right answer.
- 👎 No community downstream calibrates against `LRP-Error`. The
  three-tier-disposition machinery would exist for one metric, with
  no other consumer, and would dilute the meaning of "strict" for AP.

### (2) Algorithmic parity with the TPAMI 2021 paper

- 👍 No extra code. The paper is the spec.
- 👎 The paper is English. Reviewers cannot diff Rust output against
  a paragraph. Every PR re-derives whether the number is right from
  first principles. Same trap that drives the project to use parity
  harnesses for AP and oracles for TIDE.

### (3) Numpy oracle + kemaloksuz tripwire (chosen)

- 👍 Executable spec. Diff-able. Debuggable on a whiteboard via the
  hand-computed fixtures, like ADR-0021's TIDE oracle.
- 👍 The tripwire catches whole-class regressions the synthetic
  oracle would miss — refactoring the tau sweep, reordering the
  component computation — at the cost of one fixture in CI.
- 👍 Decouples vernier from `LRP-Error`'s research-code quirks while
  acknowledging the reference exists. Where they differ, ours is
  right by construction and `LRP-Error` is the comparator, not the
  spec.
- 👎 Two implementations of LRP in the repo plus a vendored third.
  Drift risk on three artifacts instead of two. Mitigated by both
  the oracle parity test and the tripwire running on every CI build.
- 👎 The tripwire's "sanity gate, not parity contract" semantics
  require documentation discipline — if a future contributor reads
  the tripwire as a parity contract, they will burn time chasing
  divergences the ADR says are acceptable.

### (4) No oracle; trust unit tests

- 👍 Least code.
- 👎 LRP's failure modes (tau-sweep tie-handling, component coupling
  under non-trivial recall, empty-TP class flagging) are exactly the
  bugs unit tests miss because they only manifest on realistic score
  distributions. Composition bugs silently ship.

## Links and references

- ADR-0001 — Record architecture decisions (the discipline this ADR
  follows).
- ADR-0002 — Three-tier parity model (the model this ADR explicitly
  declines to extend, same as ADR-0021).
- ADR-0005 — Similarity trait and matching engine API (the matching
  engine the oracle and Rust both consume).
- ADR-0013 — Streaming evaluator (the cells the LRP component
  computation reads from).
- ADR-0020 — Parsed-once `Dataset` handle (the entry point
  `vernier.instance.optimal_lrp(dataset, ...)` builds on).
- ADR-0021 — TIDE numpy oracle (the structural template this ADR
  mirrors; the substantive departure is the kemaloksuz tripwire as
  permanent CI rather than a one-shot Week-1 spot-check).
- ADR-0025 — Panoptic API (the paradigm the second entry point ships
  on, gaining its first TIDE-shape decomposition).
- ADR-0029 — Per-paradigm submodule namespace (the convention that
  places `optimal_lrp` under `vernier.instance` and
  `vernier.panoptic`).
- Oksuz, Cam, Akbas, Kalkan, "Localization Recall Precision (LRP): A
  New Performance Metric for Object Detection" (ECCV 2018) — the
  original LRP paper.
- Oksuz, Cam, Akbas, Kalkan, "One Metric to Measure them All:
  Localisation Recall Precision (LRP) for Evaluating Visual
  Detection Tasks" (TPAMI 2021) — the algorithmic spec this ADR
  makes executable.
- `kemaloksuz/LRP-Error` — the reference implementation we vendor
  commit-pinned as a CI sanity gate, explicitly *not* as a parity
  contract.
- `fiveai/detection_calibration` — referenced for LaECE; integration
  scoped out of 0.5.x.
