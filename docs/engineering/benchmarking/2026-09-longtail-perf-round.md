# 2026-09 long-tail perf round — stop paying for empty cells

Before/after for the four changes that landed between `b012b46c9087` and
`e361050ef582`: [#285](https://github.com/NoeFontana/vernier/pull/285),
[#288](https://github.com/NoeFontana/vernier/pull/288),
[#286](https://github.com/NoeFontana/vernier/pull/286),
[#287](https://github.com/NoeFontana/vernier/pull/287) (+ #289). Same
host (`59aab88b17f4`), same harness mode, same CPU budget, so the two
snapshots are directly comparable.

Design decisions: [ADR-0050](../../adr/0050-parallel-accumulate.md),
[ADR-0051](../../adr/0051-occupied-cell-visiting.md).

## What was wrong

The round that added Objects365 ended with vernier **losing** to hotcoco
at `nt=8` (8.76 s vs 7.78 s) while winning on a single CPU. Profiling the
cell (`cargo run --release --features bench-timings --example
o365_phase_profile -p vernier-core`) showed why — 5.9 s of the 9.0 s was
serial:

| phase | before | parallel? |
| --- | ---: | --- |
| parse GT + DT | 1 651 ms | no |
| match — `par_iter` region | 2 032 ms | yes, ~1.7× on 8 CPUs |
| match — serial transpose | 2 238 ms | **no** |
| accumulate | 2 110 ms | **no** |

The common root cause is the cell grid being **dense and 97.9 % empty**.
Objects365 val is 365 categories × 80 000 images = 29.2 M cells of which
626 k hold any annotation or detection; the output buffer is 116.8 M
slots (1.78 GiB of pointers). Both evaluate paths discovered emptiness
*inside* the per-cell body after two index lookups, the parallel path
permuted the whole buffer to reach canonical layout, and `accumulate`
walked `K · A · M · I` serially. None of that is visible at COCO's 80
categories, which is why it survived this long.

## Changes

1. **Visit only occupied cells** (#285). `cell_occupancy` builds the
   candidate list once, as CSR. Skipping a non-candidate is
   output-identical — it is the negation of the emptiness check the
   per-cell body already performs.
2. **Scatter instead of transposing** (#285), then **fill in parallel,
   one category per worker** (#288). Categories own disjoint contiguous
   ranges, so the fill parallelizes and the page faults on that 1.78 GiB
   buffer spread across workers.
3. **Parallel `accumulate` across categories** (#286). `K` is an axis of
   all three output tensors, so the fan-out is bit-identical by
   construction — no float reduction crosses a thread boundary.
4. **Overlap GT and DT parsing** (#287). Independent payloads that were
   parsed in sequence.

## Objects365 val — `objects365_val_jittered_seed0`, bbox

80 000 images · 1 240 587 GT boxes · 365 categories · ~1.06 M detections.
Harness mode `dev` (N=1); pycocotools needs ~6 min/rep.

| impl | 1 CPU before | 1 CPU after | 8 threads before | 8 threads after |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 9.997 s | **8.771 s** | 8.762 s | **4.702 s** |
| hotcoco | 14.784 s | 15.041 s | 7.777 s | 8.021 s |
| pycocotools | 357.4 s | 367.4 s | — | — |
| faster-coco-eval | DNF (OOM ~30 GiB) | DNF | DNF | DNF |

vernier's CPU/wall at `nt=8` went 2.59 → **4.27**: the serial blocks that
capped it are gone. The ranking at `nt=8` inverts back — **1.71× ahead of
hotcoco** where it was 0.89× behind. Peak RSS is unchanged (4.78 GiB),
because the dense buffer is still allocated; that is ADR-0051's deferred
option 3.

Phase split after, at 8 CPUs: parse 1 636 ms (overlapped to ~950 ms
wall), match par region 1 270 ms, post-pass 715 ms, accumulate 1 233 ms.

## LVIS v1 val — the other long-tail dataset

1203 categories, so it benefits from the same change without being
touched directly:

| impl | before | after |
| --- | ---: | ---: |
| **vernier** | 3.420 s | **2.640 s** (−23 %) |
| hotcoco | 3.508 s (1.03×) | 3.573 s (1.35×) |
| lvis-api | 187.6 s (54.9×) | 193.0 s (**73.1×**) |

Strict-tier parity holds: `vernier_lvis` is bit-equal to the `lvis-api`
oracle (identical tensor SHA-256). The cell's parity report still fails
on the *aligned* tier — that is hotcoco's own LVIS divergence (6060 cells,
worst 9.6e-4), unrelated to this round.

## COCO val2017 — unchanged, as expected

80 categories means the dense tax is small, and these numbers confirm the
change is neutral there rather than a trade:

| iou | before | after |
| --- | ---: | ---: |
| bbox | 356.3 ms | 354.3 ms |
| segm | 975.8 ms | 967.9 ms |
| boundary | 3.168 s | 3.191 s |
| keypoints | 134.5 ms | 136.0 ms |

Panoptic (10.520 → 10.564 s) and semantic (2.856 → 2.857 s) are likewise
flat; neither paradigm shares the instance cell grid.

## Thread scaling moved the most

`coco_val2017_jittered_seed0`, median total, before → after:

| iou | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| bbox | 358 → 354 ms | 350 → 267 ms | 316 → 229 ms | 314 → **226 ms** |
| segm | 980 → 983 ms | 652 → 569 ms | 468 → 375 ms | 408 → **319 ms** |
| boundary | 3.155 → 3.198 s | 1.769 → 1.700 s | 1.022 → 0.937 s | 867 → **790 ms** |
| keypoints | 133 → 136 ms | 118 → 118 ms | 106 → 109 ms | 99 → **104 ms** |

bbox was the parse-bound cell that barely scaled (1.14× from 1 to 8
threads); it now reaches 1.56×, because the parse overlap and the
parallel fill removed two of its serial blocks. Keypoints is unchanged —
its grid is one category, so none of this applies.

## What is still on the table

- **The dense buffer itself.** 1.78 GiB of pointers at 2.1 % occupancy;
  the 715 ms post-pass is first-touching it. Removing it means sparse
  `EvalGrid` storage end-to-end, which reshapes the rkyv partial wire
  format (ADR-0031) — ADR-0051's deferred option 3.
- **Serial parse.** 1.1 s of serde plus 0.5 s of index building on O365,
  now overlapped but not itself parallel. Chunked parallel parsing of the
  annotation array keeps `serde_json`'s number parsing (so it stays
  bit-identical) and would cut the remaining floor.
- **DNF as a result state.** faster-coco-eval's OOM aborts a shared cell,
  so Objects365 has to be run per-impl.
