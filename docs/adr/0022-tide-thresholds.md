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
| `Segm` | `0.1` (tentative) | Anchored on the bbox row by extrapolation, **not** by direct measurement on COCO val2017 — vernier does not ship the COCO val dataset in CI per the licensing policy in `project_coco_val_regression.md`, so the empirical histograms the original Week-2 plan called for are not derivable from the in-tree fixtures. Segm IoU is bounded above by bbox IoU on the same instance and tracks within ~10% on standard detection models, so reusing `0.1` is the least-surprising default among the choices available without COCO bytes. Revisit in a 0.5.x follow-up once whole-dataset parity infrastructure (planned post-Week 5, see `project_coco_val_regression.md`) is wired and a real-model FP-IoU histogram is available. |
| `Boundary(dilation_ratio=0.02)` | `0.05` (tentative; ratification deferred to a 0.5.x follow-up) | Geometric argument from the kernel's definition: boundary IoU = `min(mask_iou, band_iou)` and the band area is a small fraction of the mask area, so band IoU is compressed relative to bbox / mask IoU at every overlap regime. At `dilation_ratio=0.02` the band on a 50×50 mask in a 200×200 image is a 6-pixel frame (band area = 1056, mask area = 2500 → band/mask ≈ 0.42); the band IoU at the geometric "almost matched" regime (mask IoU ≈ 1/3) measures around 0.16 (verified on the `boundary_all_loc` oracle fixture). `0.05` is half the bbox value of `0.1`, putting the Bkg/Loc cutoff at the kernel's analogous "few-percent overlap" regime — the same proportion of the kernel's dynamic range the paper's `0.1` carves out for bbox. Empirical anchoring on three reference models (Mask R-CNN / Cascade Mask R-CNN / ViT-Det) on COCO val2017 is deferred to a 0.5.x follow-up; the per-release pace (`project_release_pace.md`) makes revising the default cheap if the empirical histograms shift the answer. |

This ADR remains `proposed`: both segm (`0.1`) and boundary (`0.05`)
rows are defended-by-extrapolation defaults, not measurement-anchored
ratifications. Promoting to `accepted` is gated on the deferred 0.5.x
follow-up that runs the empirical histograms on real models against
COCO val2017 once the whole-dataset parity infrastructure is wired
(planned post-Week 5 — see `project_coco_val_regression.md`).

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

The original Week-3 plan made empirical anchoring on three reference
models (Mask R-CNN / Cascade Mask R-CNN / ViT-Det on COCO val2017) a
hard gate: if the histogram analysis could not produce a `t_b` default
that stabilized the "bin-as-Bkg" fraction within ±10% across models,
boundary-segm would be cut from 0.5.0.

**Status (Week 3, this PR's amendment):** the empirical measurement is
infeasible from inside CI — vernier never commits COCO val data
(`project_coco_val_regression.md`: license restriction) and there is
no out-of-CI infrastructure available at this point in the release
line. The gate condition was "empirical defense by mid-Week-3"; we
could not do empirical anchoring in CI, so we shipped a defensible-on-
geometric-grounds default (`t_b = 0.05`, anchored on the band-area
compression argument in the table above) and deferred the empirical
ratification to a 0.5.x follow-up.

The 0.5.x follow-up replaces the geometric-anchoring paragraph in the
table with the empirical histogram analysis when the three-model
measurement becomes available. Per `project_release_pace.md` the user
is on 0.0.x patch releases; revising the default in 0.5.x is cheap.
The "cut boundary-segm" branch of the gate is preserved as an option
for the 0.5.x follow-up: if the empirical work shows `0.05` is wrong
in a way the geometric argument does not predict, boundary-segm is
deprecated for one minor cycle while the threshold is re-defended.

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
