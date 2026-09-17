# ADR-0052: Derive every `accumulate` sort from one permutation per category

- **Status:** proposed
- **Date:** 2026-09-17
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

## Context and problem statement

`accumulate` walks `K · A · M` cells. Every cell concatenates the
per-image detection scores of its `(category, areaRange)` slice,
truncates each image's block to `maxDet`, and sorts the result
score-descending before the cumulative TP/FP sweep. That sort is
`argsort_score_desc` at `accumulate.rs`, called once per
`(category, areaRange, maxDet)` — **12 sorts per category** on the COCO
defaults (`A = 4`, `M = {1, 10, 100}`).

The 12 streams are not 12 independent problems. Pycocotools has the same
structure (`cocoeval.py:378`, `np.argsort(-dtScores, kind='mergesort')`
inside the `(k, a, m)` triple loop) and vernier mirrors it faithfully,
including the redundancy. Mirroring the *semantics* is the parity
contract; mirroring the *work* is not.

How much that redundancy costs is a property of the dataset's *shape*,
not its size, and the answer is not the one the redundancy suggests.
`accumulate` pays two unrelated costs: the per-category sorting above,
whose stream length grows with `DT / K`, and the dense grid traversal
that gathers each area range's cells by walking `I` slots of the
`K · A · I` grid regardless of occupancy. On a val2017-shaped grid
(~6.8k detections per category) sorting dominates; on an LVIS-shaped
one (1203 categories, ~300 detections each) the streams are short
enough that the grid walk dominates and the 12 sorts are close to free.

ADR-0050 measured `accumulate` at 19 % of a COCO val2017 evaluation and
22 % of Objects365 val, and noted its parallel scaling is sub-linear
because that per-category gather is memory-bandwidth bound. Removing
the redundant sorts attacks the other term, and composes with ADR-0050
rather than competing with it: fewer sorts per category, still one
category per worker. It does not address the grid walk, which stays the
larger lever on long-tail datasets.

## Decision drivers

- Strict mode is bit-exact. Any reordering argument must be a *proof*
  that the emitted permutation is unchanged, not a tolerance claim.
- `accumulate` is `pub` in `vernier-core` and does not control who
  builds its `eval_imgs` grid. A derivation that depends on a producer
  invariant must verify that invariant at runtime, not assume it.
- ADR-0050's sequential and parallel walks share `accumulate_category`.
  Whatever lands here must land inside that shared function so the two
  walks cannot drift.

## Considered options

1. **Keep 12 sorts.** Status quo.
2. **Derive the `maxDet` streams from the largest one** (12 → 4 sorts).
3. **Derive across both the `maxDet` and `areaRange` axes** (12 → 1),
   guarded by a runtime equality check on the per-area score streams.
4. **Replace the comparison sort with a k-way merge** of the per-image
   runs, exploiting that each cell's `dt_scores` is already descending.

## Decision outcome

Chosen option: **Option 3, with option 4 measured and rejected.**

### Why the derivation is exact

Let `M` be the largest `maxDet` in the ladder (`max_dets` is sorted
ascending — quirk **A2**), and let the *cap stream* be the concatenation,
in image order, of `cell.dt_scores[..min(len, M)]`.

- **Claim 1 — the stream is invariant across `areaRange`.** `evaluate_cell`
  is called once per area range over one `CellBuffers`, whose `dt_scores`
  field is built before the area loop and never re-derived inside it; the
  matcher runs on the identity permutation because `dt_top_indices_for_cell_into`
  already sorted the cell once. So `PerImageEval.dt_scores` is the same
  vector for all four area ranges of a `(category, image)` pair. Only
  `gt_ignore` and `dt_ignore` vary with `a`, and neither feeds the sort.
- **Claim 2 — the smaller `maxDet` streams are induced subsequences.**
  Each cell contributes a *prefix* of its own `dt_scores`, so
  `take_m[c] = min(take_M[c], m)`: the `m` stream is the cap stream with
  a suffix dropped from each image block, and the map from cap-stream
  position to `m`-stream position is strictly increasing. A stable
  descending sort orders by `(-score, position)`; restricting a stable
  order to a subset, under a strictly increasing position map, is the
  stable order of the subset. Filtering the cap permutation therefore
  yields exactly the permutation the `m` stream would have produced.

Claim 1 is a property of *this repository's* producer, not of the type.
`accumulate` is public and its grid can be built by hand (the tests do
exactly that), so the implementation **verifies** Claim 1 rather than
trusting it: a cached plan is reused only when the next area range
presents the same cell count and bitwise-equal score blocks. A mismatch
— including a `NaN` score, which fails `==` — rebuilds the plan. The
fallback is the status quo, so a producer that violates the invariant
gets the old behaviour, not a wrong answer.

Claim 2 needs no runtime guard: it follows from `takes` being prefix
lengths, which the plan computes itself.

### Why not the k-way merge

Option 4 is correct — each `dt_scores` block is descending, so the cap
stream is a concatenation of sorted runs and a loser-tree merge that
breaks ties toward the lower run index reproduces the stable order in
`O(N log I)`. It is rejected on measurement, not on correctness:

- The premise "runs are long" does not hold for COCO-family data. A
  val2017 category sees `D ≈ 4–8` detections per image spread over
  thousands of images, so `I` is within a factor of two of `N` and
  `log I` is not meaningfully below `log N`. The asymptotic win is
  ~5 %, against a 5 000-leaf loser tree with a random-access memory
  pattern.
- Rust's `slice::sort_by` (driftsort) already detects and merges
  pre-sorted runs, so the existing call *is* a run-aware merge — just
  one that also handles the case where the producer did not sort.

Keeping the comparison sort preserves the "`accumulate` does not assume
its input is sorted" property, which Claim 1's runtime guard otherwise
would have had to extend to.

### Consequences

- **Positive:** 12 sorts per category become 1, replaced by `O(N)`
  filter passes, with bit-equal tensors. Measured on
  `cargo bench -p vernier-core --bench accumulate_shapes` (median of 5):

  | arm | before | after | delta |
  | --- | ---: | ---: | ---: |
  | `coco_like` (80 cats × 5000 imgs, ~6.8k DT/cat) | 522.7 ms | 434.5 ms | **−16.9 %** |
  | `long_tail` (1203 cats, ~300 DT/cat) | 280.3 ms | 276.9 ms | −1.2 % |

  Real LVIS v1 val (19 809 images, 1203 categories, `perfect_dt`) moves
  565 ms → 561 ms, matching the `long_tail` arm. So the win is real
  where streams are long and near-zero where they are short — roughly
  3 % of a COCO-shaped end-to-end evaluation, nothing on LVIS. No API
  change, no new knob, no parity tier change.
- **Positive:** the `m == M` cell stops re-gathering its own score
  stream — the plan already holds it.
- **Negative:** `accumulate_category` now carries a cached plan with a
  validity check, which is more state than a stateless loop. The check
  is `O(A · N)` bitwise comparisons against `O(A · N log N)` saved.
- **Neutral:** peak memory grows by one `Vec<f64>` and one `Vec<usize>`
  of cap-stream length per *worker* (not per cell), bounded by the
  largest category's detection count.
- **Neutral:** this does not make `accumulate` fast on long-tail grids.
  The measurement above says the remaining cost there is the dense
  `K · A · I` walk and the per-threshold `dtm` / `dtg` gather, which run
  `T · A · M` times over the stream. Those are the next levers and are
  out of scope here; ADR-0051 made *evaluate* sparse but `accumulate`
  still consumes the dense grid.

## Pros and cons of the options

### Option 1 (keep 12 sorts)

- 👍 Nothing to prove.
- 👎 Pays 12× for one permutation, in the block ADR-0050 identified as
  the evaluation's largest serial cost.

### Option 2 (derive the `maxDet` axis only)

- 👍 Needs no runtime guard: Claim 2 is structural.
- 👎 Leaves 4× on the table for the axis where the invariant is easiest
  to verify cheaply.

### Option 3 (derive both axes, chosen)

- 👍 Full 12 → 1; exact by the two claims; degrades to the status quo
  when the guard fails.
- 👎 Carries a cache and a validity check.

### Option 4 (k-way merge)

- 👍 Better asymptotics on data with long per-image runs.
- 👎 The target data has short runs; driftsort already exploits them.

## Links and references

- Composes with [ADR-0050](0050-parallel-accumulate.md) (category
  fan-out) and [ADR-0051](0051-occupied-cell-visiting.md) (sparse cell
  enumeration).
- Quirks **A1** (stable mergesort tie-break) and **A2** (ascending
  `maxDets` ladder) in `docs/engineering/pycocotools-quirks.md` are the
  invariants the derivation leans on.
- Reference: `pycocotools/cocoeval.py:378` at the pinned 2.0.11.
- Measurements: `crates/vernier-core/benches/accumulate_shapes.rs`, and
  `crates/vernier-core/examples/o365_phase_profile.rs` pointed at the
  LVIS v1 val cache.
