# ADR-0018: Calibration metrics — detection-family summarizer and the per-paradigm shape map

- **Status:** proposed (revised)
- **Original date:** 2026-05-01
- **Revised:** 2026-05-14
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors
- **Related:** ADR-0004, ADR-0005, ADR-0010, ADR-0013, ADR-0019, ADR-0025,
  ADR-0026, ADR-0028

## Revision note

This ADR was first drafted 2026-05-01, **before** the panoptic
(ADR-0025), semantic (ADR-0028), and mature LVIS (ADR-0026) surfaces
landed, and **before** ADR-0019 settled the Arrow-`RecordBatch`-via-PyCapsule
output posture. The original was a detection-only plan carrying a
release-timeline structure (week-by-week milestones, "success criteria
for 0.5.0"). It never reached `accepted`, and the three follow-up ADRs
it planned to spawn were never written — their numbers were taken by
other work.

This revision:

1. **Retrofits the output mechanism** to the ADR-0019
   Arrow-`RecordBatch`-via-PyCapsule pattern, replacing the bespoke
   polars-construction path (`_polars_compat.py`).
2. **Makes `iou_type`-genericity explicit** — bbox / segm / boundary /
   keypoints calibration is one summarizer, a tested scope item, not an
   implicit hope behind a `iou_type="bbox"` example.
3. **Adds the per-paradigm shape map** — calibration is three
   structurally distinct things across vernier's paradigms; panoptic
   and semantic calibration are *separate* efforts, each gated on a
   data-model prerequisite, not capabilities that fall out of this ADR.
4. **Reframes LVIS** from "deferred" to "the detection-family kernel
   with documented small-`n` discipline."
5. **Drops** the stale spawned-ADR numbering and the week-by-week
   milestone planning — that is not architecture-decision content.

The sound core is **preserved**: a summarizer over the existing
per-image grid (no `matching.rs` / `accumulate.rs` edits, ADR-0005
invariant holds), a clean-room numpy oracle with an isolated quirks
survey (the ADR-0010 pattern), `f64` end-to-end (ADR-0004), and the
DETR-aware defaults (quantile binning, `min_score=0.05`, Wilson CIs).

## Context and problem statement

mAP is a ranking metric — invariant under any monotonic transform of
detection scores. Calibration is a probability metric: it asks whether
a detection emitted at score 0.9 is correct ~90% of the time. The two
are independent, and shipping perception cares about both. Robotics and
medical-imaging users deploy detectors at fixed confidence thresholds,
fuse detections probabilistically in trackers and Kalman filters, and
write safety arguments quoting miss-rate bounds at chosen operating
points — none of which mAP serves. Today every team rebuilds this
capability in pandas, with off-by-one bin edges and inconsistent
definitions of what counts as a "correct" detection. vernier can ship
the shared baseline.

The defaults must be set by what actually breaks on transformer-based
detectors trained with focal-loss set-prediction objectives, not by
what historical CNN detectors needed. For DETR-family models,
focal-loss training, and prompt-conditioned VLM detectors: score
distributions are bimodal (set prediction) so equal-width binning is
useless; models are systematically overconfident and the
high-confidence tail — where deployment thresholds live — is where
statistical noise is highest, so per-bin counts and confidence
intervals must be first-class.

### What changed since the original draft

The original ADR assumed a detection-only vernier. Since then:

- **ADR-0025** landed panoptic evaluation as a sibling crate with its
  own kernel and no shared AP fold.
- **ADR-0028** landed semantic segmentation as a per-pixel
  confusion-matrix evaluator that ingests argmax'd label maps.
- **ADR-0026** matured the LVIS federated-evaluation surface.
- **ADR-0019** established the Arrow-`RecordBatch`-via-PyCapsule
  pattern as *the* Rust→Python tabular-output mechanism, explicitly
  superseding the bespoke polars-construction approach this ADR's
  original draft assumed.

A calibration design that does not account for these is incomplete —
not because calibration must cover every paradigm, but because the
honest scope ("which paradigms, and what blocks the others") has to be
written down.

### Constraints

- **ADR-0005** locks `matching.rs` / `accumulate.rs`. Calibration is a
  summarizer over the per-image grid; it invokes the spine, never
  edits it.
- **ADR-0010** is the template: a new metric family with no
  pycocotools precedent gets its own isolated quirks survey and oracle,
  not a wedge into the pycocotools quirks table.
- **ADR-0013** is the substrate: detection-family calibration folds
  over the per-image cell store, so it composes with
  `StreamingEvaluator` snapshots without storing extra state.
- **ADR-0019** is the output posture: tabular results are Arrow
  `RecordBatch`es via the PyCapsule interface; `arrow-rs` is a
  workspace dep, `polars-rs` is not in the FFI.
- **ADR-0004** is the numerical policy: histograms `f64` end-to-end, no
  `f32` truncation in binning.

## Decision drivers

- **Calibration is a diagnostic, not a fix.** vernier produces the
  reliability data; it does not fit calibrators. Post-hoc calibration
  (temperature scaling, Platt, isotonic) is training-side tooling —
  building it here conflates evaluation with training and crosses the
  same boundary ADR-0015 draws with "not a prediction runner."
- **One tabular-output mechanism across vernier.** The reliability and
  per-class tables must be ADR-0019-shaped Arrow tables, not a second
  bespoke path.
- **Honest, enumerated scope.** "Detection-only and silent on the
  rest" is undefined scope, not small scope. The per-paradigm shape
  map must be explicit so panoptic / semantic calibration are not
  mistakenly assumed to fall out of this ADR.
- **Defaults that work on transformer detectors without intervention**,
  owned and documented rather than punted to the user.
- **Parity discipline.** A clean-room numpy oracle with an isolated
  three-tier disposition table — the ADR-0010 pattern.

## The per-paradigm shape map

Calibration is **not one thing** across vernier's paradigms. It is
three structurally distinct shapes, and being explicit about this is
the central contribution of this revision.

### Shape 1 — Detection-family calibration (bbox / segm / boundary / keypoints). **In scope; this ADR's deliverable.**

*"Is a **detection** emitted at score `s` correct ~`s` of the time?"*

This is a summarizer over the per-image grid (the ADR-0013 cell
store). It is **`iou_type`-generic by construction**: the grid shape is
identical regardless of which IoU / OKS kernel produced the matches —
calibration reads the per-detection match outcome at a chosen T-index.
bbox, segm, and boundary are *the same code*. Keypoints is the same
code with two footnotes: the histogram denominator shifts under the
keypoint-canonical `max_dets=[20]`, and the no-`small`-area-bucket
quirk (ADR-0012 D5) is irrelevant because calibration does not bucket
by area. iou_type-genericity is a **tested scope item** — the parity
harness carries a segm fixture and a keypoints fixture, not only bbox.

### Shape 2 — Panoptic calibration. **Separate effort; data-model prerequisite.**

*"Is a **segment** emitted at score `s` matched ~`s` of the time?"*

Two structural breaks. First, `vernier-panoptic` (ADR-0025) is a
sibling crate with no ADR-0013 cell store — panoptic calibration would
be a *separate* summarizer over panoptic's own per-image matching
results, not this ADR's kernel. Second, and decisively: **the COCO
panoptic format carries no per-segment confidence score** — a panoptic
prediction is a PNG plus `segments_info` with `category_id` / `area` /
`iscrowd`, and PQ is score-free. vernier's panoptic `Predictions` model
therefore has no score field for calibration to read. Panoptic
calibration requires the panoptic input data model to gain an optional
per-segment score *first*, and is only meaningful for the subset of
models (Mask2Former-style) that emit segment confidence. Deferred — the
rationale is a data-model prerequisite, not effort. When taken up, it
is its own ADR against `vernier-panoptic`.

### Shape 3 — Semantic calibration. **Separate effort; input-type prerequisite; low-value caveat.**

*"Is a **pixel** predicted as class `c` at confidence `s` actually
class `c` ~`s` of the time?"* — per-pixel ECE.

vernier's semantic surface (ADR-0028) ingests **argmax'd label maps**,
not per-pixel **probability maps**; the kernel walks `(gt_label,
dt_label)` pairs and retains no score. Per-pixel ECE needs the
probability maps — a new ingestion path (`Predictions.from_probability_maps`
or similar) that does not exist. Additionally, the calibration
literature (Küppers et al.) finds semantic pixel calibration error is
naturally low because pixel softmaxes self-regularize — so it is
plausibly low-value as well as high-cost. Semantic also has its own
streaming substrate (`StreamingSemanticEvaluator` folding a confusion
matrix), so this ADR's streaming integration does not transfer either.
Deferred — revisit only if a probability-map ingestion path is built
for another reason.

### LVIS — detection-family kernel, small-`n` discipline. **In scope.**

LVIS calibration is Shape 1's kernel — there is no structural
difference. The only wrinkle is statistical: long-tail classes have
`n < 10` in val, making per-class calibration noise. This is **not a
deferral**: LVIS calibration is in scope, with the macro-vs-micro
aggregation choice made explicit and a warning emitted below a
per-class `n` threshold (see R5). The original ADR's "LVIS deferred"
line is replaced by this discipline.

## Considered options

### Axis A — output mechanism

- **A1.** Bespoke polars-`DataFrame` construction in
  `python/vernier/calibration.py` + `_polars_compat.py` — the original
  draft's choice.
- **A2.** Arrow `RecordBatch` via the PyCapsule interface — the
  ADR-0019 pattern, identical to the shipped `per_class` / `per_image`
  / `per_pair` tables.

### Axis B — scope shape

- **B1.** Detection-only, silent on the other paradigms — the original
  draft (it predated them).
- **B2.** Detection-family explicit and in scope; panoptic and semantic
  mapped as separate prerequisite-gated efforts; LVIS in scope with
  small-`n` discipline.

## Decision outcome

Chosen: **A2 + B2.**

### B2 — detection-family summarizer, in `vernier-core`

`crates/vernier-core/src/calibration.rs` holds the summarizer kernels;
`summarize.rs` registers `CalibrationSummary` as a variant of the
existing summarizer interface. It folds over the ADR-0013 per-image
cell store. No edits to `matching.rs` / `accumulate.rs` — the ADR-0005
invariant holds. Calibration lives in `vernier-core`, **not** a sibling
crate: it is a summarizer over the detection grid, not a separate
evaluation paradigm. (Panoptic and semantic calibration, when taken up,
*do* land in their respective sibling crates — because those paradigms
*are* separate, per the shape map.)

### A2 — Arrow output, no `polars-rs` in the FFI

The `reliability` and `per_class` tables are Arrow `RecordBatch`es
exposed through the PyCapsule interface — the ADR-0019 mechanism,
byte-for-byte the same plumbing as the existing result tables.
`crates/vernier-ffi/src/calibration.rs` is thin PyO3 conversion only;
there is no `_polars_compat.py`, no `polars-rs` dependency in the FFI.
Table schemas are golden-pinned under `tests/python/tables/schemas/`
and carry `vernier.schema_version` in the Arrow schema metadata. The
three headline scalars (`ece`, `mce`, `n_detections`) stay plain Python
`float` / `int`. A1 is rejected: it predates ADR-0019, would give
vernier two tabular-output mechanisms, and would re-open the
`polars-rs`-in-FFI question this ADR's original draft itself had
closed.

### `iou_type`-genericity is explicit and tested

The summarizer runs unchanged for bbox / segm / boundary / keypoints.
The parity harness includes a segm fixture and a keypoints fixture
alongside the bbox fixtures; "calibration works for segm" is a CI
assertion, not an assumption. Keypoints carries the two footnotes from
Shape 1 (`max_dets=[20]` denominator; no-`small`-bucket irrelevance).

### DETR-aware defaults — preserved verbatim

Unchanged from the original draft; these are load-bearing and
well-defended:

- **Quantile binning, not equal-width** — equal-width produces garbage
  ECE on bimodal set-prediction score distributions.
- **`min_score=0.05`** rather than `0.0` — the no-object tail of a
  DETR-family distribution is mathematically valid but semantically
  meaningless for a deployment threshold. Overridable; `min_score=0.0`
  recovers "all detections" semantics.
- **Wilson confidence intervals** on per-bin accuracy, by default —
  without CIs a user reads a 12-detection bin as fact. Clopper-Pearson
  offered for the conservative audit case.
- **`iou=0.5`** single-threshold definition of "correct" — calibration
  asks a different question than AP; the soft-correctness alternative
  (precision×IoU, mean-IoU-across-thresholds) is deferred as a future
  parameter, not a default.
- **Macro** per-class aggregation as the default — it weights rare
  classes equally and is the form that appears in safety-case
  documents; micro is also shipped.

These defaults differ from every Python calibration library on PyPI;
the docstring and tutorial own the divergence explicitly.

### Parity model

A clean-room **numpy oracle** with an **isolated** calibration quirks
survey (`docs/engineering/calibration-quirks.md`) and a three-tier
disposition table — the ADR-0010 pattern. Strict mode: bit-equal.
Aligned mode: 4-ULP relative. `sklearn.calibration.calibration_curve`
is a **sanity cross-check** (semantics differ around bin edges), *not*
the oracle. The one constant needing pinning is quantile bin-edge
construction (`np.quantile(..., method='linear')`); it goes in
`parity.rs` next to `linspace` with a quirks-survey row.

### Streaming

`StreamingEvaluator.snapshot().calibration(...)` folds over the
ADR-0013 cell store — **detection-family only**. The monitoring use
case (ECE drift mid-training as an earlier overfitting signal than
mAP) is the headline. Panoptic and semantic have *different* streaming
substrates; their calibration streaming is part of their separate
efforts, not this ADR.

## Public API

```python
from vernier.instance import Evaluator

# iou_type-generic: bbox / segm / boundary / keypoints all run the
# same summarizer.
result = Evaluator(iou_type="segm").evaluate(gt, dt)

cal = result.calibration()        # marginal, IoU=0.5, 15 quantile bins
cal.ece                           # float
cal.mce                           # float
cal.n_detections                  # int
cal.reliability                   # Arrow RecordBatch (PyCapsule) —
                                  # bin_id, score_lo, score_hi,
                                  # mean_score, accuracy, count, gap,
                                  # ci_lo, ci_hi
# zero-copy into the user's DataFrame library:
#   pl.from_arrow(cal.reliability)   /   cal.reliability via duckdb / pandas

cal_pc = result.calibration(per_class=True)
cal_pc.per_class                  # Arrow RecordBatch: class_id, ece, mce, n
cal_pc.worst(k=5)                 # convenience

cal = result.calibration(
    iou=0.5, n_bins=15, binning="quantile",
    min_score=0.05, confidence="wilson",
    per_class_aggregation="macro",   # or "micro"
)
```

The shape mirrors ADR-0019's result tables: scalars are plain Python
numbers, tabular outputs are Arrow `RecordBatch`es the consumer reads
zero-copy with whatever DataFrame library they already have.

## Consequences

### Positive

- Calibration becomes a first-class extended-API capability with
  defaults that work on transformer detectors out of the box.
- The output mechanism is consistent with every other vernier table
  (ADR-0019) — one tabular mechanism, no new dependency (`arrow-rs` is
  already in the workspace), zero-copy to the user's DataFrame.
- The scope is *enumerated*: detection-family is in; LVIS is in with
  small-`n` discipline; panoptic and semantic are explicitly separate
  efforts with named prerequisites — no reader mistakes silence for
  coverage.
- The ADR-0005 spine is untouched; calibration composes with the
  streaming evaluator for free (detection-family).

### Negative

- Two paradigms (panoptic, semantic) are explicitly *not* covered, and
  each carries a data-model prerequisite before it can be — a real
  capability gap, now documented rather than hidden.
- A third schema surface to golden-pin (the calibration Arrow
  schemas), on top of the calibration quirks survey.
- The DETR-aware defaults diverge from every PyPI calibration library;
  this is a documentation and user-education cost, accepted
  deliberately.

### Neutral

- Post-hoc calibration fitting is out of scope by principle, not
  omission — vernier produces the diagnostic, the user's training
  framework owns the calibrator.
- The original draft's release-timeline framing is dropped; landing
  order is a planning concern, not an ADR concern.

## What this ADR explicitly does *not* decide

- **Post-hoc calibration fitting.** Temperature scaling, Platt scaling,
  isotonic regression are training-side tooling. vernier ships the
  diagnostic; the user wires it into whatever calibrator their training
  framework already has. Building it here conflates evaluation with
  training — the same boundary ADR-0015 draws with "not a prediction
  runner."
- **Panoptic calibration.** Shape 2. Requires the panoptic
  `Predictions` data model to gain an optional per-segment score field;
  its own ADR against `vernier-panoptic` when taken up.
- **Semantic calibration.** Shape 3. Requires a per-pixel
  probability-map ingestion path that does not exist; plausibly
  low-value (pixel softmaxes self-regularize). Revisit only if the
  probability-map path is built for another reason.
- **Domain-shift detection.** "ECE went from 0.04 to 0.12 between
  training and deployment" is a workflow *on top of* vernier outputs,
  not a vernier feature.
- **Reliability-diagram plotting / `vernier-viz`.** vernier returns the
  data; the user plots it. Inherited from ADR-0019's plotting deferral.
- **Multi-prompt aggregation for VLM detectors.** Calibration as a
  function of `(image, prompt)` rather than `(image,)` is a different
  surface (detection sets vary by prompt). One prompt per evaluation
  pass is the documented assumption; multi-prompt is a v0.6+ design
  pass.
- **Extended outputs — Brier score, NLL, per-detection rows.** Not
  load-bearing for the first ship; they accumulate into a useful set
  once the foundation is stable.
- **Probabilistic / soft "correct" definitions.** Mean-IoU-across-thresholds
  or precision×IoU correctness (the LaECE-style soft label) is
  defensible but not the safety-case audience's convention. If demand
  materializes it becomes a `correctness=` parameter, not a default.

## Risk register

- **R1 — Quantile bin-edge degeneracy on small samples.** Clustered
  scores produce duplicate `np.quantile` edges. Mitigation: detect
  duplicates, merge the offending bins, surface the resulting bin
  count.
- **R2 — Wilson CI on zero-count bins.** Undefined. Mitigation: emit
  `NaN` for `accuracy` / `ci_lo` / `ci_hi`; the `count` column lets
  downstreams filter.
- **R3 — Crowd / ignore-region semantics.** Is a detection matched into
  a crowd region "correct" for calibration? Defensible answer: ignore
  it — neither TP nor FP, drop from the histogram entirely. Needs a
  quirks-survey row and a fixture; the risk is silently picking the
  wrong default and biasing every reported ECE.
- **R4 — Macro vs micro per-class aggregation.** Both have legitimate
  uses. Ship both; macro is the default (weights rare classes equally;
  the safety-case form).
- **R5 — Small-`n` per-class calibration (LVIS, long-tail).** Per-class
  numbers are noise below a threshold. Mitigation: document the
  threshold; emit a warning for any class with `n` below it.
- **R6 — Per-class path performance.** K independent binnings; trivial
  at COCO scale (K=80), ~43M ops at LVIS scale (K=1203) — fine, but the
  Arrow `RecordBatch` construction on the per-class return should be a
  single column-major op, not a per-class loop.

## Links and references

- ADR-0004 — numerical policy; calibration histograms are `f64`
  end-to-end.
- ADR-0005 — locked matching / accumulate API; calibration is a
  summarizer that invokes the spine, never edits it.
- ADR-0010 — boundary-IoU isolation; the template for the calibration
  oracle and isolated quirks survey.
- ADR-0013 — streaming evaluator; the per-image cell store
  detection-family calibration folds over.
- ADR-0019 — result tables; the Arrow-`RecordBatch`-via-PyCapsule
  output posture this revision retrofits, and the "no `polars-rs` in
  FFI" position.
- ADR-0025 — panoptic evaluation; the sibling-crate, score-free
  structure that makes panoptic calibration a separate effort (Shape 2).
- ADR-0026 — LVIS federated evaluation; the long-tail surface that sets
  the small-`n` discipline.
- ADR-0028 — semantic segmentation; the label-map (not probability-map)
  ingestion that makes semantic calibration a separate effort (Shape 3).
- Guo et al. 2017, "On Calibration of Modern Neural Networks."
- Carion et al. 2020, "End-to-End Object Detection with Transformers" —
  the set-prediction architecture the defaults target.
- Kirchheim et al. 2024, "On the Calibration of Object Detectors."
- `docs/engineering/calibration-quirks.md` — the isolated three-tier
  disposition table (to be drafted with the implementation).
