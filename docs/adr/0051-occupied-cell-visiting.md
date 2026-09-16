# ADR-0051: Visit only occupied cells; fill the grid in parallel

- **Status:** proposed
- **Date:** 2026-09-16
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

## Context and problem statement

Both evaluate paths walk the full `K x I` cell grid and decide a cell is
a no-op *inside* the per-cell body, after two index lookups
(`gt_indices_for_cell` + `raw_dt_indices_for_cell`). The parallel path
additionally filled an image-major `(i, k, a)` buffer and permuted it in
place to the canonical `(k, a, i)` layout.

Both costs scale with `K · A · I` rather than with how much of the grid
holds anything. On COCO (80 categories, 5000 images) that is invisible.
On the long-tail datasets vernier also targets it is not:

| dataset | categories | cells | occupied | occupancy |
| --- | ---: | ---: | ---: | ---: |
| COCO val2017 | 80 | 1.6 M | ~347 k | ~22 % |
| Objects365 val | 365 | 29.2 M | 626 k | **2.1 %** |
| LVIS v1 val | 1203 | 23.8 M | — | lower still |

Measured on Objects365 val at 8 CPUs before this change: the parallel
matching stage spent 2 032 ms in the `par_iter` region and **2 238 ms**
in the serial transpose, which moved 2.5 M live entries by permuting
116.8 M slots and allocating a 117 MB `visited` bitmap. The transpose
alone was the largest serial block in the whole evaluation.

## Decision drivers

- Strict-mode output must not change — not "within a tolerance", not at
  all.
- The fix should help the sequential path too, not just `num_threads>1`.
- It should not require reshaping `EvalGrid`'s storage, which is
  consumed by `accumulate`, `tables`, `partition`, `stream`,
  `calibration` and the rkyv partial wire format (ADR-0031).

## Considered options

1. **Keep the exhaustive walk.** Accept the `K · A · I` floor.
2. **Visit only cells that can hold something**, keeping the dense
   output storage.
3. **Make `EvalGrid` storage sparse end-to-end**, so nothing downstream
   is sized by `K · A · I` either.

## Decision outcome

Chosen option: **Option 2**, with option 3 recorded as anticipated
follow-up work rather than attempted here.

A cell is a candidate when its image has at least one GT annotation or
one detection in that category. That is exactly the negation of the
`gt_indices.is_empty() && raw_dt_indices.is_empty()` early-return the
per-cell body already performs, so skipping a non-candidate is
**output-identical**: the cell would have produced `None`, and `None` is
what the unvisited slot already holds. `cell_occupancy` builds that
candidate list once, in annotation order, as CSR — by image for the
parallel fan-out, by category for the sequential walk.

The parallel path also stops permuting. Workers emit only the cells they
filled; the fill is then done one category per worker, over disjoint
contiguous `A · I` ranges of the output.

Two details worth pinning:

- **Repeated category ids.** `CocoDataset::from_parts` validates that an
  annotation's category is *known* but does not reject a `categories`
  array that repeats an id, so two buckets can carry the same
  `CategoryId`. The exhaustive loop evaluated both; a naive
  `category_id -> k` map would keep one and silently blank the other's
  precision row. The map therefore keeps the earliest bucket and carries
  repeats on a side list, empty for every well-formed dataset.
- **Self-checking is weaker now.** The occupancy index is the only thing
  deciding which cells *either* path visits, so "parallel matches
  sequential" can no longer catch a bug in it — both walks would skip
  the same cell. It is pinned instead against a brute-force enumeration
  of the per-cell predicate over a federated dataset with crowd,
  zero-area and DT-only cells, plus the pycocotools parity suites.

### Consequences

- **Positive.** Objects365 val at 8 CPUs: matching 4 271 → 1 991 ms
  (par region 2 032 → 1 270, post-pass 2 238 → 715). Sequential
  matching 5 696 → 4 538 ms, so single-threaded callers benefit too.
  COCO val2017 bbox is unchanged (355.1 ms vs 356.3 ms release median).
  Output is bit-equal to pycocotools at `num_threads` None, 2 and 8.
- **Negative.** One more index to build (~50 ms on O365, from a single
  pass over annotations) and one more invariant to hold: the candidate
  set must stay a superset of the cells the per-cell predicate accepts.
  If a future quirk makes an empty cell meaningful — a federated rule
  that must observe a cell with neither GT nor DT — the index must learn
  it too.
- **Neutral.** The dense `K · A · I` output buffer survives. On O365 it
  is 1.78 GiB of pointers holding 2.1 % live entries, and first-touching
  it is what the remaining 715 ms post-pass is. That is option 3's
  territory.

## Anticipated follow-up: sparse grid storage (option 3)

Not attempted here because it reshapes the contract every consumer of
`EvalGrid.eval_imgs` reads: `accumulate` (per-`(k,a)` gather),
`tables`, `partition`, `stream`, `calibration`, the FFI, and the rkyv
partial archive whose layout is ADR-0031 wire format. It would remove
the 1.78 GiB buffer, make `accumulate`'s gather proportional to
occupancy rather than to `K · A · I`, and cut peak RSS on O365 and LVIS
by roughly the buffer's size. It should land as its own ADR with a
migration for the partial format, not as a rider on this one.

## Links and references

- [ADR-0047](0047-threading-model.md) — threading model; the
  `num_threads=None` path stays sequential and rayon-free here too.
- [ADR-0050](0050-parallel-accumulate.md) — the sibling serial block.
- [ADR-0031](0031-dist-eval.md) — partial wire format, the main obstacle
  to option 3.
- Measurements: `crates/vernier-core/examples/o365_phase_profile.rs`.
