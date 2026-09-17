# ADR-0053: Skip the GT scan for detections that cannot match

- **Status:** proposed
- **Date:** 2026-09-17
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

## Context and problem statement

`match_image_with_perm` runs a `T · D · G` triple loop. For each IoU
threshold it walks detections in score order and, for each, scans every
GT looking for the best unclaimed overlap. The scan is not a
first-match: `best` seeds at `min(t, 1 - 1e-10)` and only rises, so the
loop reads all `G` entries unless quirk **B3**'s ignore-ordering break
fires.

A detection that overlaps nothing still pays that full scan, `T` times.
On COCO-shaped data this is invisible — the median non-empty cell holds
`G · D = 1` (`benches/evaluate_bbox.rs`). It stops being invisible in
the dense regime vernier also targets: surveillance and
autonomous-driving workloads run 250 GTs and 250 DTs in one image, and
at `T = 10` that is 625 k scan steps per cell, most of them for
detections whose best overlap is nowhere near the lowest rung.

## Decision drivers

- Strict mode is bit-exact; a skip has to be provably a no-op, not a
  heuristic.
- The COCO-shaped path is the common case and must not regress to buy
  the dense case. A tuning constant that helps one regime by taxing the
  other is not worth having.
- The matching loop is the parity-critical core (quirks **B1**–**B6**).
  Whatever lands must leave the loop body readable as the reference.

## Considered options

1. **Nothing.** The dense regime pays the full scan.
2. **Per-DT column maxima**, computed once per cell, skipping any DT
   whose best overlap is below the threshold seed.
3. **Spatial index** (grid or BVH) over GT boxes per cell.

## Decision outcome

Chosen option: **Option 2**, gated on cell size and monomorphized so
the sub-gate path is unchanged.

`dt_best[d] = max_g iou[(g, d)]` is computed in one row-major pass.
Inside the ladder, a DT with `dt_best[d] < min(t, 1 - 1e-10)` is
skipped: `best` starts at that seed and only rises, so every `iou < best`
test in its scan would `continue`, `m` would stay `-1`, and the `m < 0`
guard would make the iteration a no-op. The skip is therefore
output-identical by construction — it removes work, not a branch of the
decision.

Two details carry the correctness:

- **`NaN` maps to `+inf`, not away.** `f64::max` ignores `NaN`, but the
  match loop does not: `iou < best` is false for `NaN`, so a `NaN`
  overlap *matches* (and, once `best` is `NaN`, so does every later GT
  in that row). If the column maxima dropped it, the prefilter would
  skip a DT the reference matches. A `NaN` bbox is reachable through
  the ADR-0030 array-ingest path, which does not go via JSON, so this
  is not hypothetical.
- **The gate is a pure performance switch.** `PREFILTER_MIN_CELL = 256`
  (`G · D`) sits well above the COCO median cell and well below the
  dense regime. `match_image_with_perm_gated` takes it as a parameter
  so tests can force the prefilter on and off over the same cell and
  assert the two results agree field for field.

The ladder is monomorphized on a `const PREFILTER: bool` rather than
branching on an `Option`. That is not cosmetic: the residual
per-`(threshold, DT)` branch on an inactive `Option` cost ~4 % on
`framework_coco_like`, which would have made this a bad trade.

### Consequences

- **Positive:** dense cells collapse. Measured on
  `cargo bench -p vernier-core --bench evaluate_bbox` (median of 100):

  | arm | before | after | delta |
  | --- | ---: | ---: | ---: |
  | `framework_dense_mle_1cat` | 8.16 ms | 1.29 ms | **−84 %** |
  | `framework_dense_mle_5cat` | 4.08 ms | 0.81 ms | **−80 %** |
  | `framework_coco_like` | 1.82 ms | 1.82 ms | neutral |

- **Negative:** the ladder moved out of `match_image_with_perm` into a
  generic `run_ladder`, so the parity-critical loop is one call away
  from its validation. The quirk comments moved with it.
- **Neutral:** one `Vec<f64>` of length `D` per cell above the gate.
  Below the gate nothing is allocated and nothing is checked.

## Pros and cons of the options

### Option 1 (nothing)

- 👍 No new constant, no new code path.
- 👎 Leaves a 6× on the table for a regime the project explicitly
  benchmarks.

### Option 2 (column maxima, chosen)

- 👍 Exact by construction; one pass; helps exactly where the scan is
  long; costs nothing below the gate.
- 👎 Introduces a tuning constant and a second monomorphization.

### Option 3 (spatial index)

- 👍 Would also cut the scan for DTs that *do* match.
- 👎 Build cost per cell dwarfs the scan at COCO shapes, and the
  traversal order would have to reproduce the `k_g`-ascending
  tie-break exactly (**B2**) — a much larger parity surface for a
  regime option 2 already handles.

## Follow-up (not in this change)

`column_maxima` is recomputed once per area range over the *same*
matrix. `evaluate_cell` is called four times per `(category, image)`
cell — once per area range — against one `buffers.iou`
(`evaluate.rs`, `evaluate_parallel.rs`); only `gt_ignore` varies
between those calls, and `dt_best` does not depend on it. So the
prefilter pays for four passes and four `D`-long allocations where one
would do.

Hoisting `dt_best` into `CellScratch` as a reusable scratch `Vec` — in
the style of the existing `gt_ignore_buf` — would recover most of
that, and it lands in exactly the dense regime this ADR targets.
Deliberately left out of scope here: it is **unmeasured**, and the
decision above rests on measurements. Take it as its own change, with
its own numbers, on an uncontended box.

## Links and references

- Quirks **B1** (threshold seed), **B2** (non-strict comparison),
  **B3** (ignore-ordering break), **B4** (crowd re-match) in
  `docs/engineering/pycocotools-quirks.md` — all preserved verbatim in
  `run_ladder`.
- The `NaN` reachability argument depends on [ADR-0030](0030-buffer-protocol.md).
- Measurements: `crates/vernier-core/benches/evaluate_bbox.rs`.
