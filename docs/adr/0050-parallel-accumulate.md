# ADR-0050: Parallelize `accumulate` across the category axis

- **Status:** proposed
- **Date:** 2026-09-16
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

## Context and problem statement

[ADR-0047](0047-threading-model.md) parallelized per-image matching and
explicitly deferred `accumulate()`:

> `accumulate()` is *not* parallelized in this ADR (option A3 rejected).
> The merged-stream score-descending sort is global and not naturally
> parallel; parallelizing the `(K, A, M)` cell fan-out would buy a small
> constant and risks introducing a parallel f64 reduction that breaks
> strict-mode equality. […] its own ADR with its own parity story
> (likely corrected-tier only).

Two of those three premises hold. The third — "buys a small constant" —
was measured at COCO scale, where `accumulate` is ~70 ms of a ~360 ms
evaluation. The constant is not small once the category axis is long,
because `accumulate` walks `K · A · M` cells and gathers `I` images for
each:

| workload | categories | `accumulate` | share of total |
| --- | ---: | ---: | ---: |
| COCO val2017 bbox | 80 | 70 ms | 19 % |
| Objects365 val bbox | 365 | 2 110 ms | 22 % |

At `num_threads=8` on Objects365 the rest of the evaluation drops but
`accumulate` does not, so it becomes the largest single serial block and
caps the achievable speedup (Amdahl) well below the thread count.

## Decision drivers

- Strict mode is bit-exact. A parallel AP fold that only matches within
  a tolerance would have to ship as corrected-tier, which is a much
  worse trade than staying serial.
- ADR-0047's `num_threads=None` contract: the sequential path must stay
  byte-identical and must not enter rayon at all.
- The fix should not require callers to learn a new knob.

## Considered options

1. **Stay serial.** Accept the Amdahl ceiling on long-tail datasets.
2. **Parallelize the merged-stream sort** inside a cell.
3. **Parallelize the cell fan-out across the category axis**, keeping
   each cell's arithmetic exactly as it is.

## Decision outcome

Chosen option: **Option 3, and the "parallel f64 reduction" risk does
not arise.**

The concern ADR-0047 raised is real for option 2 and for any design that
splits a *cell*: the score-descending sort and the cumulative TP/FP
sweep are order-dependent, so distributing them changes results. Option
3 splits at the category boundary instead, where:

- each category owns a disjoint slice of all three output tensors
  (`precision` / `scores` are `(T, R, K, A, M)`, `recall` is
  `(T, K, A, M)` — `K` is an axis of every output);
- no accumulator, running total, or scratch buffer is shared between
  categories;
- every cell runs the same `accumulate_cell` call, in the same order,
  with the same inputs as the sequential walk.

So the result is bit-identical by construction rather than within a
tolerance, and no float reduction crosses a thread boundary. Both walks
call one shared `accumulate_category`; the only difference is which
thread holds the category. `crates/vernier-core/src/accumulate.rs`'s
`parallel_accumulate_is_bit_identical_to_sequential` asserts tensor
equality (`==`, not `abs_diff <= eps`) over a ragged multi-category grid
that includes empty cells, all-ignore cells, and score ties across
images.

Implementation notes:

- `accumulate_parallel` is a sibling of `accumulate`, matching the
  `evaluate_*` / `evaluate_*_parallel` shape ADR-0047 established.
- The FFI grid records the `ThreadPolicy` it was evaluated under, and
  `EvalGrid.accumulate()` reuses it. `num_threads=None` grids call the
  sequential walk and never enter rayon; no public signature changes.
- A pool that fails to build falls back to the sequential walk — the
  tensors are the same, only slower.

### Consequences

- **Positive:** Objects365 val `accumulate` drops 2.14 s → 1.23 s at 8
  threads with bit-equal output. Long-tail datasets (LVIS's 1203
  categories) benefit most. No new knob, no parity tier change.
- **Negative:** `accumulate` now has two walks to keep in step. The
  shared `accumulate_category` limits that to the fan-out itself, and
  the bit-identity test fails loudly if they diverge.
- **Neutral:** Speedup is sub-linear (~1.7× at 8 threads) because the
  per-category gather streams the dense `K · A · I` cell grid and is
  memory-bandwidth bound. Making that grid sparse is a separate change;
  this ADR does not depend on it.

## Pros and cons of the options

### Option 1 (stay serial)

- 👍 Nothing to review; strict parity trivially preserved.
- 👎 Leaves the largest serial block in place exactly where the category
  axis makes it matter.

### Option 2 (parallelize the in-cell sort)

- 👍 Would help a single enormous category.
- 👎 The sweep is order-dependent: this is the design ADR-0047 correctly
  rejected, and it could only ship corrected-tier.

### Option 3 (category fan-out, chosen)

- 👍 Bit-identical by construction; disjoint outputs; no shared state.
- 👎 Sub-linear scaling; two walks to keep in step.

## Links and references

- Supersedes the deferral in [ADR-0047](0047-threading-model.md)
  §"Parallel axis: per-image matching, serial accumulate" and the
  "rayon AP fold" item deferred in
  [ADR-0033](0033-multi-paradigm-bench.md).
- Measurements: `docs/engineering/benchmarking/2026-09-hotcoco-fce-1.8-and-scale.md`
  (phase split) and `crates/vernier-core/examples/o365_phase_profile.rs`.
