# ADR-0018: Calibration metrics for modern detection architectures

- **Status:** proposed
- **Date:** 2026-05-01
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors
- **Target landing:** Phase 5, Week 4 (per `docs/explanation/possible-extensions.md`),
  feeding into the 0.5.0 release
- **Maps to:** §4 of `Possible_Extensions`, expanded to be load-bearing for
  transformer-based detectors and prompt-conditioned foundation models
- **Will spawn ADRs:** see end of doc

## Summary

Ship calibration metrics (ECE, MCE, reliability table, per-class breakdowns)
as a first-class extended-API capability in vernier. Scope is set by what
actually breaks on transformer-based detectors trained with focal-loss
set-prediction objectives, not by what historical CNN detectors needed.

The capability is a new summarizer over the existing per-image evaluation
grid; it does not touch `matching.rs` or `accumulate.rs` (ADR-0005 invariant
holds). The parity story is a clean-room numpy oracle, not pycocotools —
calibration is not a pycocotools concept, and the analogue of ADR-0010's
isolated boundary-IoU subsystem applies. Total envelope: ~1500 lines core
+ Python surface + oracle + validation, four weeks of one engineer's time.

The headline output is a polars DataFrame and three scalars, with defaults
chosen so that DETR-family score distributions don't silently produce
misleading numbers.

## Problem statement

mAP is a ranking metric: it is invariant under any monotonic transform of
detection scores. Calibration is a probability metric: it asks whether a
detection emitted with score 0.9 is actually correct ~90% of the time.
These are independent, and shipping perception cares about both.

The robotics and medical-imaging users who would adopt vernier deploy
detectors at fixed confidence thresholds, fuse detections probabilistically
in trackers and Kalman filters, and write safety arguments that quote
miss-rate bounds at chosen operating points. None of these workflows are
served by mAP. All of them are served by calibration metrics.

The current state of the art in eval tooling is that every team builds this
capability themselves, in pandas, with off-by-one bin edges and inconsistent
choices about what counts as a "correct" detection. Vernier can ship the
shared baseline.

The framing of "calibration metrics" in
`docs/explanation/possible-extensions.md` was correct in spirit but assumed a
CNN-detector world (Faster R-CNN, YOLO, RetinaNet). For DETR-family models,
focal-loss training, and prompt-conditioned VLM detectors, three additional
constraints apply:

1. Score distributions are bimodal (set prediction). Equal-width binning is
   useless; quantile binning is the default that works.
2. Models are systematically overconfident; the high-confidence tail is
   where deployment thresholds live and where statistical noise is highest.
   Confidence intervals and per-bin counts must be first-class.
3. Calibration depends on prompt for text-conditioned detectors. The API
   needs to surface the assumption (one prompt per evaluation pass) and
   make it cheap to re-evaluate after prompt change.

This plan reflects those constraints in the defaults, the API, and the
validation set.

## Decision drivers

- ADR-0005 invariant: no edits to `matching.rs` or `accumulate.rs`.
- ADR-0013 composition: calibration must work on `StreamingEvaluator`
  snapshots without storing additional state.
- ADR-0002 parity model: calibration is not a pycocotools concept; new
  three-tier table is needed, with a numpy oracle, mirroring ADR-0010.
- ADR-0004 numerical policy: histograms in `f64` end-to-end; no `f32`
  truncation in the binning. ECE differences smaller than ~1e-4 are noise
  to users but bigger than the parity tolerance, so we want the math tight.
- The 0.5.0 release timeline is one week. Week 4 of Phase 5 has to land
  the headline capability or be cut from the release; nothing in this
  plan is a "nice to have" inside that week.
- Calibration is a diagnostic, not a fix. Post-hoc calibrators (temperature
  scaling, isotonic regression) are training-side tooling and live outside
  vernier.

## Architecture

### Where the code lives

```
crates/vernier-core/src/
  calibration.rs         # new: summarizer kernels, oracle parity helpers
  summarize.rs           # extended to register CalibrationSummary as a
                         # variant of the existing summarizer interface
crates/vernier-ffi/src/
  calibration.rs         # new: thin PyO3 conversion only
python/vernier/
  calibration.py         # new: user-facing API; produces polars DataFrames
  _polars_compat.py      # extended if necessary
docs/
  adr/0019-calibration-summarizer.md           # to be drafted
  adr/0020-calibration-oracle-and-quirks.md    # to be drafted
  adr/0021-calibration-defaults.md             # to be drafted
  engineering/calibration-quirks.md            # disposition table
  reference/calibration.md                     # API reference
  tutorials/calibrating-a-detr.md              # walkthrough
```

The clean parallel to ADR-0010 (boundary-IoU isolation) is intentional. A
new metric family with no pycocotools precedent should not be wedged into
the pycocotools quirks table; it gets its own quirks survey, its own oracle,
and its own three-tier disposition list. Reusing the existing parity
machinery is not free — it ties calibration's evolution to pycocotools'
governance even though they share no semantics.

### FFI surface

A single PyO3 entry point per evaluator. The function consumes the
already-computed per-image evaluation cells (the same ones used for `mAP`
in `accumulate.rs`) and emits a flat struct: `(ece, mce, reliability_rows)`.
Python wraps `reliability_rows` into a polars DataFrame. The FFI does no
DataFrame construction in Rust because polars-Rust as a vernier-ffi
dependency is a top-level dependency change requiring its own ADR-0001
trigger.

If someone later wants `polars-rs` end-to-end (for the per-image and
per-class DataFrames in §3 of `Possible_Extensions`), that can be a
follow-up ADR. Calibration does not justify pulling it in alone.

### Parity: numpy oracle, three-tier disposition

The parity strategy mirrors ADR-0010 boundary-IoU.

**Oracle:** a vendored, pinned numpy reference implementation at
`tests/python/parity_calibration/oracle/numpy_calibration.py`. Single file,
~150 lines, deliberately readable, no performance optimization. This is
the spec-of-record for what "correct" means.

**Three-tier dispositions** for every choice point in the algorithm:

- *Strict.* The choices that pin behavior (binomial denominator,
  count-of-detections weighting, IoU threshold default 0.5) are strict
  against the oracle. Bit-equality required.
- *Aligned.* Numpy's histogram routines vs. a hand-rolled Rust one — same
  outputs to ~4 ULP, faster. Quirks-table entry, no user-visible change.
- *Corrected.* Specifically for choices the oracle gets demonstrably wrong
  (none expected at v0.5.0; if a corrected entry shows up during
  implementation, it surfaces as an ADR).

A new survey lands at `docs/engineering/calibration-quirks.md` with a
two-column row per choice: choice, disposition, link to oracle line. Initial
rows are bin-edge convention, ignore-region handling, area-filter
interaction, no-detection-image handling, per-class aggregation order. Five
to ten rows, enough to be a spec, not so many they become unmaintainable.

### Numerical layout

`f64` end-to-end. Calibration math is small (constant memory, linear
sweeps), and the SIMD argument from ADR-0004 doesn't apply. No mixed-
precision boundary, no pulp dispatch, no platform variation. ADR-0008's
precedent (bbox IoU `f64` end-to-end after parity bugs) is the cleanest
path; calibration takes the same lesson upfront rather than after.

The one constant that does need pinning: bin-edge construction for
quantile binning. Numpy's `np.quantile(scores, q, method='linear')` is
specified; the Rust equivalent is documented and unit-tested against a
fixture. This goes in `parity.rs` next to `linspace`, with a quirk-survey
row.

## Public API

### Python surface

```python
from vernier import Evaluator
from vernier.calibration import Calibration  # explicit import; not auto-bound

ev = Evaluator(gt, dt, iou_type="bbox")
result = ev.finalize()

# Default: marginal calibration at IoU=0.5, 15 quantile bins, all detections
cal = result.calibration()
cal.ece                     # float, e.g. 0.097
cal.mce                     # float, e.g. 0.140
cal.n_detections            # int, the denominator
cal.reliability             # polars.DataFrame
                            # cols: bin_id, score_lo, score_hi, mean_score,
                            #       accuracy, count, gap, ci_lo, ci_hi
```

The reliability DataFrame ships per-bin Wilson confidence intervals on
`accuracy` by default. This is non-negotiable for the deployment use case:
without CIs, users will read accuracy in a 12-detection bin as a fact and
make threshold decisions on noise. Wilson is closed-form, cheap, and well-
behaved at the boundaries; Clopper-Pearson is offered as an option for the
audit case where conservatism matters.

Per-class is opt-in but easy:

```python
cal_pc = result.calibration(per_class=True)
cal_pc.per_class            # DataFrame indexed by class_id; ECE, MCE, n
cal_pc.worst(k=5)           # convenience: 5 classes with highest ECE
cal_pc.class_reliability(class_id=3)  # full per-bin DataFrame for one class
```

Override knobs with sensible defaults for transformer detectors:

```python
cal = result.calibration(
    iou=0.5,                # the matching threshold defining "correct"
    n_bins=15,              # bin count
    binning="quantile",     # default; "equal_width" available
    min_score=0.05,         # ignore detections below this — important for
                            # set-prediction models with no-object queries
    confidence="wilson",    # CI flavor
)
```

### Defaults aimed at transformer detectors

Two non-obvious defaults, both load-bearing, both worth defending in an ADR
rather than burying in code:

1. **Quantile binning, not equal-width.** Equal-width binning produces
   garbage ECE on bimodal score distributions (DETR, RT-DETR, OWLv2, etc.).
   Quantile binning gives every bin equal statistical weight by
   construction. The historical default for image classification ECE was
   equal-width because softmax distributions weren't bimodal; that default
   does not transfer.
2. **`min_score=0.05` rather than 0.0.** Including the no-object tail of a
   DETR-family score distribution in calibration is mathematically valid
   but semantically meaningless: the user will never threshold against
   those detections in deployment. Document the choice; make it overridable.
   Users who want pycocotools-compatible "all detections" semantics pass
   `min_score=0.0`.

These defaults differ from what every Python-implemented calibration
library on PyPI does. We will be explicit about why in the docstring and
the tutorial, and we will own the decision rather than punting to the user.

### Streaming integration

`StreamingEvaluator.snapshot().calibration(...)` works as a fold over the
per-image cell store from ADR-0013. Same API, same numbers if the snapshot
is `deterministic=True`, modulo the running-mode caveat ADR-0013 already
documents.

For training-loop telemetry:

```python
snap = streaming_ev.snapshot(deterministic=True)
log.scalar("eval/map_running", snap.summary["AP"])
log.scalar("eval/ece_running", snap.calibration().ece)
```

The monitoring use case is the killer feature here. ECE drift mid-training
is a much earlier signal of overfitting than mAP, especially for fine-tuned
foundation detectors where mAP plateaus while calibration silently
degrades.

## Implementation milestones

A four-week slice. Each week is a working increment; the deliverable can
ship at the end of any week and the unfinished weeks become follow-ups.

**Week 1 — kernel and oracle.** Land the numpy oracle, the Rust kernel,
and the FFI entry point. Marginal (single-class-aggregated) calibration
only. ECE, MCE, reliability rows. No per-class, no streaming, no
DataFrames. Strict-mode parity test against the oracle on three fixtures
(synthetic perfect, synthetic overconfident, real COCO sample). Quirk
survey lands with three rows.

**Week 2 — Python surface and per-class.** The polars DataFrame
construction. Per-class calibration. Wilson CIs. `min_score` and binning
options. API reference doc. Three more parity-survey rows. The user-
facing Python tests live here, pinned to the oracle so that future
refactors don't drift.

**Week 3 — streaming integration and tutorials.** The
`StreamingEvaluator.snapshot().calibration()` path. Tutorial:
"Calibrating a DETR." Tutorial: "Reading a reliability diagram for safety
arguments." Documented streaming caveats. Performance baseline: <500ms
on COCO val (5k images, ~36k detections).

**Week 4 — real-model validation and the 0.5.0 release cut.** Run vernier
calibration on four representative checkpoints: Faster R-CNN R50 (COCO),
YOLOv8m (COCO), DINO Swin-L (COCO), Grounding DINO (COCO). Document the
patterns each architecture exhibits. Cross-check against an independent
calibration library (sklearn `calibration_curve`) for sanity. Land the
ADRs (0016, 0017). Tag 0.5.0.

The validation work in week 4 is not optional. It's how we discover that
our defaults don't actually work on the architectures the docs claim to
support. Without it, week 1's parity-against-oracle is a tautology — the
oracle and the kernel agree, but neither has been seen by a real model
checkpoint. Put it in the schedule, defend it in review.

## Testing strategy

**Unit tests in Rust.** Every calibration kernel function has a property
test: `ECE` of perfect predictions is 0; `ECE` of always-wrong-but-
confident predictions is large; reliability table sums to total detection
count; per-class ECE averaged with the right weights equals marginal ECE
when classes are balanced. Aliased synthetic fixtures, deterministic seeds.

**Parity against numpy oracle.** Same harness pattern as ADR-0010's
boundary-IoU isolation. Strict mode: bit-equal. Aligned mode: 4 ULP
relative. Run on the three week-1 fixtures plus the four week-4 real-model
checkpoints. Failures are blockers.

**Real-model validation.** Run calibration on the four checkpoints (week 4)
and produce a reference table — ECE, MCE, top-5-worst-classes. Pin it as
a regression fixture in `tests/python/regression/calibration/`. Future
PRs that change kernel internals must reproduce these numbers within 1e-6.
This catches the case where an "innocent refactor" silently changes which
detections fall into which bin.

**Performance baseline.** `pytest-benchmark` on the COCO-val fixture.
Budget: ECE+MCE+reliability for marginal under 500ms; per-class under 2s.
Regressions over 10% block the PR.

**Cross-library sanity.** Spot-check ECE values against
`sklearn.calibration.calibration_curve` on a synthetic dataset with known
ground truth. Not a parity test (semantics differ around bin edges); a
sanity test that catches order-of-magnitude bugs.

## Risk register

**R1 — Quantile binning produces non-monotonic bin edges on small
samples.** A streaming snapshot at step 100 might have 200 detections, and
the bins produced from `np.quantile(scores, [.., .., ..])` can have
duplicate edges if scores cluster. Mitigation: detect duplicates, merge
the offending bins, surface the resulting bin count to the user. Document
the rule. Test fixture lives in week 1.

**R2 — Wilson CI on zero-count bins.** Edge case: a bin with zero
detections in some per-class breakdowns. Wilson is undefined. Mitigation:
emit `NaN` for `accuracy`, `ci_lo`, `ci_hi` in that row; surface the count
column so downstreams can filter. Test fixture in week 2.

**R3 — Crowd / ignore region semantics.** `pycocotools` matches detections
into crowd regions without a positive flag; vernier's matcher already
handles this. Question: is a detection that matched a crowd region
"correct" for calibration purposes? Defensible answer: ignore (don't count
it as TP or FP, drop from histogram entirely). This needs a quirks-survey
row, ratification, and a fixture. Risk is that we silently pick the wrong
default and quietly bias every reported ECE. Mitigation: the survey row.

**R4 — Per-class aggregation: macro vs. micro.** "Class-wise ECE" can
mean either average-of-per-class-ECEs (macro) or per-class numerator and
denominator pooled (micro). Both have legitimate uses. Mitigation: ship
both; pick macro as the default because it weights rare classes equally
and is the version that shows up in safety-case documents.

**R5 — Validation set is too small for tight CIs.** The 4-checkpoint
week-4 validation depends on having enough validation predictions to make
ECE estimates meaningful. COCO val (5k images) is borderline for per-class
on the long-tail classes. Mitigation: document the n threshold below which
per-class numbers are noisy; emit a warning if any class has n < 50.

**R6 — Performance on the per-class path.** Per-class calibration is K
independent binnings. Naive implementation is O(K · D) where D is detection
count; for COCO K=80, D=36k, that's still only ~3M operations and trivially
fast. For LVIS (K=1203), it's ~43M, still fine but worth measuring. The
real risk is the polars DataFrame construction overhead on the per-class
return; profile and possibly fold it into a single column-major op.

## Success criteria for 0.5.0

1. Marginal and per-class calibration available on `Evaluator` and
   `StreamingEvaluator`.
2. Defaults that work for transformer-based models without intervention,
   documented as such.
3. Parity-against-oracle test passes in CI on every supported platform.
4. Performance baseline met (<500ms marginal on COCO val).
5. One tutorial — "Calibrating a DETR" — that walks a user from a
   prediction file to a reliability diagram to a deployment threshold
   choice.
6. A regression-pinned reference table for four real models, published in
   the docs as a sanity reference for future contributors.

We are not committing to a "vernier is the authoritative calibration
library" claim until at least one external user has run it on their
production model and reported back. That's a 0.6.x conversation.

## ADRs that spin out of this plan

Three:

- **ADR-0019: Calibration as a per-image-cells summarizer.** Locks the
  architectural choice (no edits to matching/accumulate, fold over the
  ADR-0013 cell store, summarizer pattern). Should be ratified before
  Week 1 lands.
- **ADR-0020: Calibration parity model — numpy oracle, isolated
  disposition table.** Same shape as ADR-0010. Locks the oracle, the
  parity tiers, and the quirks-survey location.
- **ADR-0021: Calibration defaults for modern detectors — quantile
  binning, `min_score=0.05`, Wilson CIs.** This one is the most
  interesting from a design-review perspective; it's where we have to
  defend that we are knowingly diverging from sklearn-style and
  TensorFlow-style ECE conventions because they don't fit our deployed-
  model audience. Worth the explicit ADR.

A possible fourth ADR — calibration-extended outputs (Brier score, NLL,
per-detection rows) — is deferred. None of those is load-bearing for v0.5;
they accumulate to a useful set in v0.6 once the foundation is stable.

## What this plan explicitly does NOT do

- **Post-hoc calibration fitting.** Temperature scaling, Platt scaling,
  and isotonic regression are training-side tooling. Vernier produces the
  diagnostic; the user wires it into whatever calibrator their training
  framework already has. Building the calibrator in vernier conflates
  evaluation with training and complicates the API in ways that pay no
  rent.
- **Domain-shift detection.** "My ECE went from 0.04 to 0.12 between
  training and deployment" is a useful workflow, but it's a workflow on
  top of vernier outputs, not a vernier feature. A standalone tool or
  user notebook.
- **Visualization / reliability-diagram plotting.** Vernier returns a
  DataFrame; the user plots it with matplotlib, plotly, holoviews, or
  whatever. Shipping plotting in a Rust+Python eval library is range
  creep.
- **Probabilistic IoU thresholding.** Some literature treats "is this
  detection correct" as a soft label (mean IoU across thresholds) rather
  than a hard one. Defensible, but not the safety-case audience's
  convention; out of v0.5.0 scope. If demand materializes, it becomes a
  parameter, not a default.
- **LVIS / open-vocabulary calibration.** Not architecturally different,
  but the long-tail makes per-class calibration a different conversation
  (most classes have n < 10 in val). Out of scope for v0.5.0; revisit
  once the LVIS evaluation surface itself is more mature.
- **Multi-prompt aggregation for VLM detectors.** Calibration as a
  function of `(image, prompt)` rather than `(image,)` is conceptually
  similar but operationally a different surface (detection sets vary by
  prompt). Document the assumption (one prompt per evaluation pass).
  Worth its own design pass in v0.6+.

## Links and references

- ADR-0002 — three-tier parity model (the pattern this plan applies
  to a new oracle).
- ADR-0005 — locked matching/accumulate API (the invariant this plan
  preserves).
- ADR-0010 — boundary-IoU isolation pattern (the template for ADR-0020).
- ADR-0013 — streaming evaluator (the cell store this plan folds over).
- `docs/explanation/possible-extensions.md` §4 — the design note this
  plan expands on.
- Guo et al. 2017, "On Calibration of Modern Neural Networks" — the
  original observation about deep-network overconfidence.
- Carion et al. 2020, "End-to-End Object Detection with Transformers" —
  DETR set-prediction, the architecture this plan's defaults are aimed at.
- Kirchheim et al. 2024, "On the Calibration of Object Detectors" — the
  detection-specific calibration literature this plan is consistent with.
