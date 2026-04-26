# ADR-0002: Adopt a three-tier parity model against pycocotools

- **Status:** proposed
- **Date:** 2026-04-26
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

vernier is a Rust+PyO3 reimplementation of `pycocotools`. The reference oracle
is pinned exactly to `pycocotools==2.0.11` in `pyproject.toml`, and the parity
harness at `tests/python/parity/harness.py` double-runs reference and
candidate to compare every intermediate of the COCOeval state machine.

Reading `pycocotools/cocoeval.py`, `coco.py`, `_mask.pyx`, and `maskApi.c`
line by line surfaces a long tail of numerical and structural quirks. The
working note `docs/engineering/pycocotools-quirks.md` enumerates 61 of them
(rows A1 through L8) across sorting, matching, recall integration, area
ranges, crowd semantics, OKS, RLE encoding, RLE decoding, IoU, `loadRes`,
type discrimination, and API hygiene.

Each quirk forces a binary question and so a third option in front of every
implementation choice: should vernier reproduce the quirk bit-exactly, match
its semantics with a cleaner implementation, or correct it as an opinionated
fix? Without a single rule applied uniformly, every PR re-litigates the
question from scratch — and the answer drifts, because the reviewer who
recalls one decision rarely recalls another.

The disposition column in the quirks survey is a draft proposal by the
survey's author. It is not load-bearing until ratified in an ADR.

## Decision drivers

- Users have downstream tooling (mmdetection, ultralytics, detectron2, lvis-api
  forks) calibrated to pycocotools' exact numerical behavior, including its
  bugs. A drop-in replacement that silently changes scores would be a
  migration hazard, not a migration path.
- Some pycocotools behaviors are clearly defects (silent zero-row outputs on
  dimension mismatch; bare `except: pass` swallowing `IndexError`; dead-code
  reads that overwrite a user's `ignore` field). Reproducing them faithfully
  in 2026 would be cargo-culting.
- Some behaviors are semantic conventions (101-point recall, terminal-recall
  AR, asymmetric crowd IoU) that user-facing scores are calibrated against.
  Diverging would invalidate every published COCO benchmark.
- Reviewers need a default disposition so that PR review doesn't degenerate
  into per-quirk negotiation.
- The disposition for each quirk must be discoverable: future maintainers
  should be able to read why the code does what it does without reverse-
  engineering it.

## Considered options

1. **Strict-only.** vernier reproduces every pycocotools behavior bit-exactly,
   including obvious bugs. Migration is trivial; users can never opt out of
   defects.
2. **Corrected-only.** vernier ships an opinionated, cleaned-up evaluator
   from day one. Numerically diverges from pycocotools on day one. Forces
   every downstream user to re-baseline.
3. **Two-tier (strict / corrected).** Each quirk is either reproduced or
   fixed; users select one mode globally. No room for semantic-equivalent
   cleanups that produce identical outputs faster or more safely.
4. **Three-tier (strict / aligned / corrected).** Every quirk is dispositioned
   into one of three buckets:
   - **strict** — reproduce bit-exactly. This is the default for behaviors
     that are observable in published scores or in user-visible artifacts.
   - **aligned** — match the semantics; outputs are bit-identical (or within
     a documented tolerance) but the implementation is cleaner, faster, or
     safer. The user cannot tell the difference numerically; the code is
     better.
   - **corrected** — opinionated fix. Default behavior diverges from
     pycocotools and the divergence is documented. User opts in to strict
     mode if exact reproduction is required.

## Decision outcome

Chosen option: **three-tier (strict / aligned / corrected)**.

Strict is the default disposition. Aligned is reserved for changes that are
observably indistinguishable from pycocotools — same outputs, better
implementation. Corrected is reserved for behaviors the project is willing
to defend as bugs, with a strict-mode opt-out so calibrated downstreams can
still reproduce the original numbers.

The disposition column in `docs/engineering/pycocotools-quirks.md` is hereby
ratified for every row. Future quirks discovered after this ADR is accepted
get a disposition assigned in the same survey by the PR that introduces
their handling; the disposition is reviewed but does not require its own
ADR unless it changes the default-mode behavior of an already-shipped path.

The user-facing surface of the three tiers:

- A single parity-mode setting (`strict` | `aligned` | `corrected`), exposed
  on the public Python `Evaluator` and on the Rust core's eval entry point.
- `aligned` is not a separate user-facing mode in the API: aligned changes
  ship in every mode, because by definition they are output-equivalent to
  strict.
- `strict` reproduces every quirk on the survey marked **strict**, plus the
  opt-in switches required to reproduce quirks marked **corrected** that
  have a strict alternative.
- `corrected` (the default for net-new users) applies every disposition
  marked **corrected** by default, and reproduces every disposition marked
  **strict** as before. (Strict-marked quirks are inherent to the algorithm,
  not opt-in fixes.)

The CI parity harness runs in strict mode. The corrected-mode results are
captured separately and asserted to differ from strict only at the rows
flagged as corrected — a regression test for the disposition table itself.

### Consequences

- **Positive.** Every behavior change has a single canonical disposition.
  Reviewers point at a row in the survey and a column in this ADR rather
  than re-deriving the answer. Users get a documented migration path:
  drop-in strict mode today, opt into corrected as their downstreams catch
  up. The survey becomes the project's most valuable artifact: each row
  encodes a quirk found by reading source line-by-line, and the disposition
  is now load-bearing rather than aspirational.
- **Negative.** Three modes are more surface than two. Tests, fixtures, and
  documentation have to cover each. Some quirks (B5, C8, D3) sit in the
  aligned tier where the line between "same outputs" and "indistinguishable
  outputs" is a tolerance argument that has to be defended.
- **Neutral.** The line between strict and corrected for any specific quirk
  is a judgment call the team will revisit as user feedback arrives. Moving
  a quirk from strict to corrected (or vice versa) is itself an ADR.

## Pros and cons of the options

### Option 1 — strict-only

- 👍 Minimum API surface; one mode; trivially explainable.
- 👍 Migration is "drop in and run".
- 👎 Reproduces obvious bugs forever (H2 silent merge, I2 magic `-1`, I6 1-D
  reshape landmine, K1 2-point polygon collision, L5/L6/L7 stdout-only API).
- 👎 Cleaner implementations of equivalent semantics (C6 vector vs Python
  loop, G3 sign extension, B5 id storage) become harder to justify against
  a "bit-exact only" rule.

### Option 2 — corrected-only

- 👍 Smallest, cleanest codebase.
- 👎 Day-one numerical divergence from every published COCO score.
  Adoption requires every downstream to re-baseline before they can trust
  vernier's numbers. This is the path that killed every previous
  pycocotools rewrite.

### Option 3 — two-tier (strict / corrected)

- 👍 Fewer modes than three.
- 👎 No bucket for "same outputs, better implementation". Either every
  implementation cleanup is forbidden under strict, or strict's invariant
  becomes "behaviorally indistinguishable" — which is just aligned without
  the name. Compresses two distinct ideas into one mode and confuses both.

### Option 4 (chosen) — three-tier (strict / aligned / corrected)

- 👍 Every quirk gets one disposition. The survey is now the spec.
- 👍 Aligned tier lets us write clean code without changing user-visible
  outputs, captured in one column rather than buried in PR reviews.
- 👍 Corrected mode is opt-in (or out, depending on default), giving users
  agency without making the API more complex than two settings.
- 👎 Three buckets means three things to remember; survey rows that sit on
  the boundary will get re-classified at least once.
- 👎 The aligned tier is the most fragile — what counts as "indistinguishable
  outputs" depends on a tolerance argument we have to defend per row.

## Disposition assignments

The disposition column of `docs/engineering/pycocotools-quirks.md` is the
authoritative table. As of this ADR's date, the proposed dispositions are
ratified as follows. Dispositions cited by quirk ID; full text in the
survey.

**Strict (most rows; the default).** A1 (default — opt-in `corrected`
tiebreaker available), A4, B1, B2, B3, B4, B6, B7, C1, C2, C4, C5, C7, D2,
D4, D5, D6, D7, E1, E2, E3, F3, F4, G1, G2, G4, G5, G6, H1, H3, H4, H5, H6,
I4, J2, J3, J4, J5, K2, L1, L2.

**Aligned (semantic match, cleaner implementation).** A2, B5, C6, C8, D3,
F2, F5, G3, I3, I5, J1, K3, K4, L4, L8.

**Corrected (opinionated fix; strict mode reproduces original).** A3, C3,
D1, F1, H2, H7, I2, I6, J6, K1, L3, L5, L6, L7.

The five **open questions** at the bottom of the survey (B1+integer IoU,
D6 vs D7, H3 polygon direction, K1 2-point polygon, G3 sign extension)
remain open. Each gets its own follow-up commit with a fixture; their
disposition column entries above are best-effort and will be revisited
once the fixtures land.

## Links and references

- ADR-0001 — Record architecture decisions (the discipline this ADR follows).
- `docs/engineering/pycocotools-quirks.md` — the disposition table this ADR
  ratifies.
- `tests/python/parity/harness.py` — the harness that asserts strict-mode
  bit-equality.
- `tests/python/parity/test_parity.py` — `ALL_FIXTURES` corpus.
- `pyproject.toml` — the `pycocotools==2.0.11` pin that defines the oracle.
