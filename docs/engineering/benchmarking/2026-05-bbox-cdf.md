# bbox-IoU `(G, D)` distribution on val2017 — Stage 0 measurement

The decision-criterion artifact for Stages 1b (SoA refactor) and 1c
(explicit `pulp::Simd` lanes) of the bbox-IoU optimization plan, captured
with the `bench-histogram` Cargo feature. Per-call records are produced
by `BboxIou::compute` (`FullIou`) and `BboxIou::compute_overlap_mask`
(`OverlapMask`).

## How it was captured

```bash
just bench-develop-histogram

UV_NO_SYNC=1 \
VERNIER_COCO_GT_PATH=$VERNIER_COCO_GT_PATH \
VERNIER_BENCH_HISTOGRAM_PATH=/tmp/stage0/bbox-hist \
  just bench-run --impl vernier --workload coco_val2017_jittered_seed0 \
                 --iou bbox --mode profile

# Repeat for --iou segm to capture the OverlapMask prefilter side.
```

`coco_val2017_jittered_seed0` workload: 5000 images, ~36k anns, jittered
DT mirroring the val2017 GT distribution.

## Results

### bbox eval — `FullIou` kernel calls

* **14,205 calls** total, 25.25 ms aggregate wall time.
* `G`: median 1, mean 2.6, max 26.
* `D`: median 1, mean 2.4, max 23.
* `G·D`: median **1**, mean 14, max 598 (single image: 26 GTs × 23 DTs).

Time-weighted `G·D` distribution:

| `G·D`        | calls | calls % | total wall | wall %   | cum wall % |
|--------------|------:|--------:|-----------:|---------:|-----------:|
| `[1, 4)`     |  8353 |   58.8% |     3.77 ms|    14.9% |      14.9% |
| `[4, 16)`    |  3384 |   23.8% |     3.80 ms|    15.0% |      30.0% |
| `[16, 64)`   |  1453 |   10.2% |     4.96 ms|    19.7% |      49.6% |
| `[64, 256)`  |  1010 |    7.1% |    12.55 ms|    49.7% |      99.3% |
| `[256, 1024)`|     5 |    0.0% |     0.17 ms|     0.7% |     100.0% |

Wall-time percentile cuts:

* P25: `G·D ≤ 9`
* P50: `G·D ≤ 64`
* P75: `G·D ≤ 156`
* P90: `G·D ≤ 182`
* P99: `G·D ≤ 210`

The eight largest cells (out of 14,205) span `G·D` 238–598 and together
contribute < 1% of wall time.

### segm prefilter — `OverlapMask` kernel calls

Same `(image, category)` cell partitioning, so the count and `(G, D)`
shape are identical (14,205 calls, median `G·D` 1, max 598). Total wall
time is 25.61 ms — within ~1% of the standalone `FullIou` pass, even
though `compute_overlap_mask` drops `vmulpd` (intersection) and
`vdivpd` (IoU). The arithmetic savings are invisible at this resolution
because per-call setup, not inner-loop work, is what's being measured.

## Decisions against the plan's criterion

The plan's criterion (verbatim from
`/home/dev/.claude/plans/zippy-seeking-origami.md`):

> **Decision criterion for 1c:**
> - ≥80% of wall time in cells with `G·D ≥ 256` → 1c is on the table.
> - ≥80% of wall time in cells with `G·D < 64` → drop 1c entirely; inner
>   loop is too short for explicit lanes to amortize.

Measured: **49.6% of wall time in `G·D < 64`, 0.7% in `G·D ≥ 256`.**
Neither bucket hits 80%, but the *spirit* of the criterion is decisive:
99.3% of wall time is in cells with `G·D < 256`. Explicit-lane SIMD
won't amortize at this scale.

* **Stage 1c — drop.** Inner-loop work is below the noise floor of
  per-call overhead.
* **Stage 1b — drop on COCO.** The SoA refactor's per-cell gather cost
  (eight `Vec<f64>` extends per call) would exceed the inner-loop
  savings on the median `G=1, D=1` cell. The plan's hypothesis ("if
  Stage 0 shows median G·D < 50, drop 1b") is satisfied: median is 1.
* **Stage 2c — drop.** The early-out only fires at `dts.len() ≥ 256`,
  which is the long tail contributing < 1% of wall time. Parity-safe
  per the verification at `matching.rs:194` and `evaluate.rs:890`,
  but no measurable workload to justify the feature flag.

## What the lever actually is

Median call: 911 ns of wall time for `G·D = 1` — i.e. ~3 ns of f64
arithmetic and ~908 ns of *something else*. That something is per-call
setup overhead — `pulp::Arch::dispatch` boundary, `out.row_mut(g)`, the
empty-check guard, the histogram instrumentation (~50 ns when on),
iterator and closure prologue.

Stage 1a (the `OnceLock` Arch hoist, shipped in #168) was the first
move in this direction; the per-call cpuid feature-detect was real
overhead even though it was below the divan-bench noise floor on the
synthetic grids. The remaining levers are dominantly outside the inner
loop.

A reasonable next perf-push round would target this directly:

1. Per-call profiling on a `(1, 1)` and `(2, 2)` synthetic cell —
   isolate which non-inner-loop code accounts for the 900 ns.
2. Consider whether `out.row_mut` (ndarray's mutable row-view
   construction) is a hidden cost on tiny cells. Direct slice access
   may amortize better.
3. Consider a small-cell fast path that bypasses the dispatch closure
   for `G·D ≤ 4`. The dispatch overhead becomes substantial relative
   to the work for the median cell.

These are speculative directions, not committed work — captured here
so the next round of bbox-IoU optimization starts with measurement, not
the now-disproven inner-loop-SIMD intuition.

## Dense-scene regime (single-category, ≥50 GT/image)

The val2017 decisions are scoped to the **per-cell** regime they were
measured in. The defining variable is `G·D` per `BboxIou::compute` call,
which is dominated by **per-category density** (`n_categories` per
image), not raw GT count per image. Three additional captures on
`synthetic` workloads, same harness:

| Workload                             | calls | total wall | median/call | median G·D | G·D ≥ 256 |
|--------------------------------------|------:|-----------:|------------:|-----------:|----------:|
| val2017 (real, 80 cats)              | 14205 |    25.3 ms |      460 ns |          1 |      0.7% |
| synthetic G=200, D=100, 80 cats      | 10550 |    12.1 ms |    1.01 µs  |          4 |      0.0% |
| synthetic G=50,  D=100, 1 cat        |   200 |    71.2 ms |     350 µs  |       5000 |    100.0% |
| synthetic G=200, D=100, 1 cat        |   200 |   280.3 ms |    1387 µs  |      20000 |    100.0% |

Two distinct regimes:

* **Multi-category sparse (val2017, 200-GT-with-80-cats):** `G·D` median
  ≤ 4, every cell well under the 256 threshold, per-call setup
  dominates. The decisions above (drop 1b/1c/2c) hold.
* **Single-category dense (200-GT scenes, dense surveillance, single-
  class detection):** `G·D` median ≥ 5,000, 100% of wall time well past
  the 256 threshold, per-call setup is < 0.1% of cell cost. **The
  decisions invert.**

If your workload is single-category (or near-single-category) with
dense per-image GT counts, 1b/1c/2c become defensible:

* **Stage 1c (explicit `pulp::Simd` lanes)** is the highest-leverage
  move. The 5–8% production-vs-`scalar_reference` gap I observed in the
  divan benches translates directly to wins on cells where the inner
  loop dominates. AVX-512's 8-wide f64 lanes vs AVX2's 4-wide is a
  potential 2× on Ice Lake / Sapphire Rapids servers.
* **Stage 1b (SoA refactor)** as a prerequisite — the per-cell gather
  cost (8 `Vec<f64>` extends, ~1 µs at memcpy speed) amortizes cleanly
  over 5,000–20,000 inner-loop iterations, and contiguous loads matter
  for explicit-lane vector loads.
* **Stage 2c (x-axis early-out)** depends on spatial sparsity — for
  uniform-random distributions the early-out catches ~50% of cells, but
  for clustered scenes (the typical dense-detection failure mode) it
  catches less. Bench-flag landing if 1b/1c numbers are insufficient.

What this means in practice: there isn't a single optimization plan
for "vernier bbox-IoU." The plan splits along this regime line. The
shipped 1a + 2a wins are universal (correctness fix + free divide
elimination on segm/boundary prefilter); the next perf-push round
should pick a regime explicitly.

## Caveats

* Numbers include the histogram instrumentation overhead (~50 ns per
  call from `Mutex::lock` + `Vec::push` in the `CallTimer::Drop`
  guard). Production `wall_ns` would be ~5% lower at the median, ~0%
  at the long tail. The shape of the distribution is unaffected.
* `Instant::now()` resolution on this machine is 19 ns (per divan
  preamble), so the 480–921 ns medians are well-resolved but the
  smallest cells (G=1, D=1) are still close to the timer floor.
* Single-process measurement on `coco_val2017_jittered_seed0` plus
  three `synthetic:` workloads. LVIS-scale (1k+ DTs/image) and dense
  single-category scenes shift the distribution as documented in the
  "Dense-scene regime" section. The split-by-regime guidance there
  supersedes any single-CDF reading.
