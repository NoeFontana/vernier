# ADR-0022: TIDE thresholds and per-kernel defaults

- **Status:** proposed
- **Date:** 2026-05-02
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

TIDE bin assignment is parameterized by two thresholds:

- **`t_f`** (foreground / match threshold) — at-or-above is a match;
  detections matched at `t_f` are TPs; unmatched detections are
  candidates for one of the FP bins.
- **`t_b`** (background threshold) — IoU `< t_b` against every GT
  (any class) means "background"; IoU `∈ [t_b, t_f)` means
  "almost-matched", which carves the Loc / Both bins.

The Bolya et al. paper picks `t_f = 0.5`, `t_b = 0.1` for bbox on COCO
and stops there. The numbers users see when they call
`vernier.error_decomposition(...)` with no kwargs are entirely
determined by these two. Picking them wrong silently mis-bins
detections — a Loc error becomes a Bkg error or vice versa, the
headline ΔmAP per bin shifts, and the user reads the wrong story.

The problem is per-kernel. Bbox IoU and segm IoU have similar
distribution shapes on COCO; bbox `t_b = 0.1` transfers to segm
without controversy. Boundary IoU (ADR-0010) does not. At
`dilation_ratio = 0.02` (the COCO default), boundary IoU concentrates
at lower values — a detection a human would call "almost right" scores
~0.3 boundary-IoU where the bbox version would score ~0.7. Reusing
`t_b = 0.1` for boundary moves the Bkg/Loc cutoff to a place that no
longer corresponds to "almost-matched" in the kernel's own scale.

We need a defaults policy that (a) matches the paper for bbox so
published TIDE numbers reproduce, (b) extends defensibly to segm and
boundary, and (c) leaves the user free to override per call without
boilerplate.

## Decision drivers

- **Paper-faithfulness for bbox.** Users who run TIDE on bbox-COCO
  and compare to published TIDE numbers must get the paper's
  defaults. Anything else makes the headline numbers look wrong on
  contact.
- **Per-kernel honesty.** A default that mis-bins for boundary-segm
  is worse than no default. The reported numbers carry vernier's
  authority; a wrong default is a wrong-by-our-fault number.
- **Reproducibility in the report.** Whatever defaults the call used
  must surface in `TideReport.config` so a screenshot of a number
  can be re-derived from the report alone.
- **Override ergonomics.** The ratio of "user accepts default" to
  "user overrides" will be high; default + per-call kwarg is enough,
  no global config knob.
- **Defaults must be defended.** Per memory `feedback_naming.md`'s
  spirit (don't borrow numbers without naming the mechanism), each
  default carries a one-line rationale in this ADR — not a magic
  constant in code.

## Considered options

1. **Single `(t_f, t_b)` across all kernels.** Bolya's `(0.5, 0.1)`
   for everything. Simple. Wrong for boundary.
2. **Per-kernel `(t_f, t_b)` with paper-anchored bbox/segm and
   empirical-anchored boundary.** Three numbers in a table; each
   defended in the ADR.
3. **No default for `t_b`; require it.** Force the user to think.
   Loses to copy-paste from README defaults the user doesn't read.
4. **Per-kernel × per-parameter table** (e.g., boundary's `t_b` as a
   function of `dilation_ratio`). More accurate; more surface.

## Decision outcome

Chosen option: **(2) per-kernel `(t_f, t_b)` with paper-anchored
defaults for bbox/segm and empirical-anchored defaults for boundary.**

`t_f` defaults to `0.5` everywhere — matches the TIDE paper, matches
the AP@0.5 reading users already have intuition for, and makes the
bin-assignment threshold the same as the matching threshold (no
double-cutoff to reason about).

`t_b` defaults are per-kernel:

| Kernel | `t_b` default | Anchored on |
|--------|---------------|-------------|
| `Bbox` | `0.1` | TIDE paper, COCO bbox |
| `Segm` | **TBD** (Week-2 measurement) | Pending verification that segm-IoU FP distribution on COCO val2017 matches bbox closely enough to reuse `0.1`. If yes, ratify `0.1`; if not, anchor empirically as for boundary. |
| `Boundary(dilation_ratio=0.02)` | **TBD** (Week-3 measurement) | Pending empirical anchoring on three reference models on COCO val2017 (Mask R-CNN, Cascade Mask R-CNN, ViT-Det). Anchor target: the value at which the "binned as Bkg" fraction stabilizes across the three models. Histograms in the ADR companion data file. |

This ADR is `proposed` until the TBD rows are filled. The Week-2 PR
ratifies the segm row; the Week-3 PR ratifies the boundary row (or
triggers the decision gate below). Both are amendments to this ADR,
not new ADRs.

The `Boundary` default is keyed to `dilation_ratio=0.02` (the COCO
default). For other `dilation_ratio` values (LVIS uses `0.008`),
the same `t_b=0.05` default applies but the user is expected to
override based on their kernel's distribution. The tutorial
(`docs/tutorials/debugging-with-tide.md`) shows how to choose.

The defaults live in a single Rust function
`tide::defaults_for(kernel: &EvalIouType) -> (f64, f64)` in the
`tide` module, surfaced through FFI as the `None` resolver in
`vernier.error_decomposition(t_f=None, t_b=None, ...)`. Defaults are
not Python-side constants; they live with the algorithm.

`TideReport.config` always carries the resolved `(t_f, t_b)` and the
kernel — no "the default may have been ..." ambiguity in any
recorded report.

### Decision gate (boundary default)

The Week-3 empirical anchoring is a hard gate. If the histogram
analysis on the three reference models does not produce a `t_b`
default with a coherent defense (specifically: a value at which the
"bin-as-Bkg" fraction stabilizes within ±10% across the three
models), we cut boundary-segm from 0.5.0 and ship it in 0.5.1 with
a follow-up to this ADR. Better to ship two kernels right than three
with a dubious threshold.

### Consequences

- **Positive.** Bbox TIDE numbers from `vernier.error_decomposition`
  reproduce the paper directly. Segm and boundary defaults are
  defended in this ADR rather than copy-pasted. The report carries
  reproducibility for free.
- **Negative.** Three numbers to keep in the head. Any kernel added
  later (Phase-3 keypoints, future formats) needs a defaults entry
  and a defense, even if the answer is "same as bbox."
- **Neutral.** Users who care override per call; the override syntax
  is `t_b=0.07` and that is the entire ergonomic cost. The defaults
  are starting points, not commitments.

## What this ADR explicitly does not decide

- **Numpy oracle and correctness.** ADR-0021.
- **Cross-class IoU computation strategy.** ADR-0023.
- **TIDE-on-OKS / keypoints.** ADR-0024 (deferred — no thresholds
  to argue about).
- **Bin semantics at boundaries.** "IoU exactly equal to `t_f`" is a
  match (paper convention); "IoU exactly equal to `t_b`" is the
  boundary of Loc/Bkg and goes to Loc (the higher bin), matching the
  paper. These follow from the paper's `>=`/`<` convention; the
  oracle implements them; the Rust matches the oracle. No ADR fight.
- **Mode-specific defaults.** `mode="per_threshold"` (the opt-in 10×
  variant) uses the same `(t_f, t_b)` defaults; the per-threshold
  axis is the IoU threshold the bin assignment is computed at, not
  a different `t_f`. ADR-0021 already covers this.
- **Defaults for non-COCO `dilation_ratio` boundary.** Tutorial
  guidance, not an ADR commitment.

## Pros and cons of the options

### (1) Single `(t_f, t_b)` across all kernels

- 👍 One number to remember. Simplest possible API.
- 👎 Boundary IoU's distribution makes `t_b=0.1` mis-bin. Reported
  numbers under-count Loc and over-count Bkg by a factor that depends
  on the model. We'd be shipping wrong defaults to look simple.

### (2) Per-kernel defaults (chosen)

- 👍 Honest. Each default defended on the kernel's own scale. Bbox
  matches paper.
- 👍 Override surface is one kwarg; defaults table is small.
- 👍 Decision gate documents when to back out (cut boundary if
  defense fails).
- 👎 Three numbers, one of which (boundary) ages with model
  architecture if FP-IoU distributions shift. Configurable per call
  is the answer; we accept the default as a starting point.
- 👎 The boundary default depends on `dilation_ratio`; we pin to the
  COCO default and document the rest. Imperfect but bounded.

### (3) No default for `t_b`; require it

- 👍 Forces the user to choose. No silent wrong-binning.
- 👎 The default value the user picks will be whatever the README
  says. We have not removed the defaults problem; we have
  externalized it to the README. Worse, the report can't carry "the
  default" because there isn't one — every screenshot needs the
  caller to remember which number they passed.

### (4) Per-kernel × per-parameter table

- 👍 Most accurate. `t_b` as a function of `dilation_ratio` for
  boundary is the principled answer.
- 👎 Surface explosion. The defaults-for-this-call computation
  becomes its own subroutine with its own tests. Diminishing returns
  past three kernels and one parameter; revisit if Phase-3 surfaces a
  case where this matters.

## Links and references

- ADR-0001 — Record architecture decisions.
- ADR-0010 — Boundary IoU isolated subsystem (the kernel whose
  distribution shape forces the per-kernel default).
- ADR-0021 — TIDE numpy oracle and correctness model (the harness
  that pins the bin-boundary semantics regardless of threshold value).
- ADR-0023 — Cross-class IoU computation strategy (the data the
  thresholds slice through).
- Bolya et al., "TIDE: A General Toolbox for Identifying Object
  Detection Errors" (ECCV 2020) — the paper whose bbox defaults this
  ADR adopts.
- Companion data file `docs/explanation/tide-and-its-limits.md`
  (Week-3 deliverable) — the histograms that defend the boundary
  default.
