# ADR-0021: TIDE error decomposition — numpy oracle as the correctness model

- **Status:** proposed
- **Date:** 2026-05-02
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

vernier 0.5.0 introduces TIDE (Bolya et al., 2020) error decomposition:
the headline ΔmAP is broken into six bins (Cls / Loc / Both / Dupe / Bkg /
Miss) by running corrected accumulations per bin and subtracting from
baseline. Each bin's correction rewrites the per-image cells the matching
engine produced — moving DTs across `(category, image)` cells, flipping
`gt_ignore`, synthesizing matches — and re-runs `accumulate` + `summarize`.
Eight accumulate passes per call.

The cell-rewrite layer is the most semantically delicate code vernier has
shipped. The Cls correction in particular *moves* a detection across
cells, which interacts with `accumulate_cell`'s `max_det` truncation
(`accumulate.rs:280`) and with score-sort tie-breaking (quirk A1). A
silent bug in any one bin only surfaces under specific score
distributions; unit-testing the Rust implementation against itself is
circular.

The pycocotools-style discipline this project uses for AP — a pinned
reference oracle, the parity harness, the three-tier disposition model
(ADR-0002) — does not transfer. The published reference implementation
`tidecv` diverges from the TIDE paper in known ways (single-class
degeneracy; crowd-region interaction; inconsistent area-range and
max-dets handling across bins) and is not itself the spec. The TIDE
paper is the algorithmic spec, but it is English; reviewers cannot diff
against an English specification. We need an executable artifact that
*is* the spec for vernier.

## Decision drivers

- **Correctness must be diff-able.** Reviewers and contributors need to
  point at a known-good output and say "the Rust differs here." Without a
  reference, every TIDE PR re-litigates whether the new number is right.
- **No canonical numerical reference exists.** `tidecv` has
  documented divergences from the paper, and the paper is text. We
  cannot bind correctness to either.
- **The three-tier model (ADR-0002) does not apply.** `strict / aligned /
  corrected` is keyed to "reproduce pycocotools or knowingly diverge."
  TIDE has no `pycocotools` analogue. Inheriting the three-tier surface
  would invent a fake reference and dilute the meaning of "strict" for AP.
- **Hand-computed assertions must be possible.** The oracle must be
  debuggable on a whiteboard via small fixtures before it can debug the
  Rust implementation.
- **Quirk surface should not bleed into pycocotools-quirks.md.** The
  pycocotools quirks survey is about reproducing one specific upstream
  in version-pinned detail. TIDE's design choices belong in TIDE's own
  ADRs, not appended to a survey of a different library.

## Considered options

1. **Bitwise parity with `tidecv`.** Treat `tidecv` as a pinned reference;
   build a TIDE quirks survey paralleling the pycocotools one; ship a
   strict / aligned / corrected disposition for each divergence.
2. **Algorithmic parity with the TIDE paper.** Implement what the paper
   says; assert against intuition and small examples; no executable
   reference.
3. **Numpy oracle as the correctness contract.** A pure-numpy reference
   implementation in `tests/python/oracle/tide/oracle.py`. The Rust
   implementation is correct iff it agrees with the oracle within `1e-9`
   ΔmAP per bin per fixture. The oracle's own correctness is pinned by
   hand-computed assertions on small fixtures.
4. **No oracle; trust unit tests.** Test cell-rewrite mutations in
   isolation; trust composition.

## Decision outcome

Chosen option: **(3) numpy oracle as the correctness contract.**

The oracle lives at `tests/python/oracle/tide/oracle.py` (~400 lines,
pure numpy, no vernier imports). It takes a `(gt, dt, similarity_fn,
t_f, t_b, mode)` tuple and returns the six ΔmAP values plus the
all-FPs-removed sanity total. It is correct by construction:
readable, no SIMD, no perf optimizations, single-pass per bin without
the cross-cell DT-move surgery the Rust implementation needs.

The oracle's own correctness is pinned by `test_oracle.py`: ~6 synthetic
fixtures small enough to hand-compute (all-perfect, all-Bkg, all-Cls,
crowd interaction, ignore interaction, deliberately-constructed Dupes),
each with hand-computed ΔmAP values asserted on the oracle's output.

The Rust implementation's correctness contract is then:
`test_rust_matches_oracle.py` runs the Rust `error_decomposition` and
the oracle on the same fixture, asserts `|delta_rust − delta_oracle| <
1e-9` per bin per fixture across all three kernels (bbox, segm, boundary).

Where vernier diverges from `tidecv` on real-data fixtures (it will),
the divergence is recorded in `docs/explanation/tide-and-its-limits.md`
(per the plan) with the oracle's value as the authoritative one. We do
not ship `tidecv` parity, but we do ship "we know where we differ and
why."

### Consequences

- **Positive.** Reviewers diff against a Python reference they can read
  in an afternoon. New contributors debug TIDE bugs by stepping through
  the oracle, not by spelunking SIMD-dispatched cell-rewrite code.
  Future quirks (max-dets interactions, crowd handling refinements)
  land as oracle changes first, Rust-implementation changes second —
  with the oracle commit being the spec.
- **Negative.** ~400 lines of test infrastructure to maintain. The
  oracle is slow (single-pass numpy, no batching) — full COCO val
  fixtures are pytest-`slow`-marked. Any TIDE behavior change requires
  changing the oracle and the Rust together; a change to one without
  the other is automatically caught as a parity miss.
- **Neutral.** vernier's TIDE numbers will not match `tidecv`'s on
  real models. This is a design output, not a defect. Documented in
  the explanation page.

## What this ADR explicitly does not decide

- **Threshold defaults (`t_f`, `t_b`, per-kernel choices).** ADR-0022.
- **Cross-class IoU computation strategy.** ADR-0023.
- **TIDE-on-OKS / keypoints.** ADR-0024 (deferred).
- **Per-threshold mode mathematics.** The oracle implements the
  paper-faithful single-`t_f` semantics by default, with a
  `mode="per_threshold"` parameter for the opt-in 10× variant. The
  variant is correct-by-construction in numpy (the same algorithm
  applied 10×); no separate ADR needed.
- **Oracle vs `tidecv` reconciliation on every divergence.** Spot-check
  during Week 1 implementation, document divergences in the explanation
  page, but do not commit to a `tidecv`-parity test. The oracle is the
  reference.

## Pros and cons of the options

### (1) Bitwise parity with `tidecv`

- 👍 Mirrors the discipline that worked for pycocotools/AP.
- 👎 `tidecv` is not the spec. Pinning to it encodes its bugs as
  vernier's "correct" behavior, then forces every fix to ship as a
  "corrected" disposition the user has to opt into to get a
  mathematically-right answer.
- 👎 Three-tier dispositions for a library nobody calibrates downstream
  tooling against is overhead without payoff. Users compare against the
  TIDE paper, not against `tidecv`.

### (2) Algorithmic parity with the TIDE paper

- 👍 No extra code. The paper is the spec.
- 👎 The paper is English. Reviewers cannot diff Rust output against a
  paragraph. Every PR review re-derives whether the number is right
  from first principles, the same trap that drives the project to use
  parity harnesses for AP.

### (3) Numpy oracle (chosen)

- 👍 Executable spec. Diff-able. Debuggable on a whiteboard via the
  hand-computed fixtures. The Rust implementation has one job: match
  the oracle within `1e-9`. No re-derivation per PR.
- 👍 Decouples vernier from `tidecv`'s defects without inheriting them
  as quirks. Where they differ, ours is right by construction (because
  the oracle says so) and `tidecv` is documented as the comparator,
  not the spec.
- 👎 ~400 lines of code to write and maintain. Slow on real-data
  fixtures (pytest `slow` marker hygiene applies — fixtures that
  actually run in seconds get marked, faster ones don't).
- 👎 Two implementations of TIDE in the repo (numpy + Rust). Drift
  risk: a behavior change to one but not the other ships a silent
  semantic divergence. Mitigated by the parity test running on every
  CI build; any drift fails CI.

### (4) No oracle; trust unit tests

- 👍 Least code.
- 👎 The cell-rewrite layer's failure modes (cross-cell DT move under
  `max_det` truncation, score-sort tie-breaking interaction) are
  exactly the bugs unit tests miss because they only manifest on
  realistic score distributions across many cells. Composition bugs
  silently ship.

## Links and references

- ADR-0001 — Record architecture decisions (the discipline this ADR
  follows).
- ADR-0002 — Three-tier parity model (the model this ADR explicitly
  declines to extend).
- ADR-0005 — Similarity trait and matching engine API (the matching
  engine the oracle and Rust both consume).
- ADR-0013 — Streaming evaluator (the cells store the cell-rewrite
  layer mutates).
- ADR-0020 — Parsed-once `Dataset` handle (the entry point
  `vernier.error_decomposition(dataset, ...)` builds on).
- Bolya et al., "TIDE: A General Toolbox for Identifying Object
  Detection Errors" (ECCV 2020). The English specification this ADR
  makes executable.
- `tidecv` — the reference implementation we are explicitly *not*
  binding to.
