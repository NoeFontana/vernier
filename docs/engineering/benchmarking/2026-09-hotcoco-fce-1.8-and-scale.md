# 2026-09 hotcoco enters the matrix, faster-coco-eval 1.8, and an Objects365 scale cell

Engineer-facing snapshot of the round that (a) added
[hotcoco](https://github.com/derekallman/hotcoco) as a baseline, (b) moved
faster-coco-eval from 1.7.2 to 1.8.0, (c) made the thread budget of a cell
enforceable rather than advisory, (d) started measuring memory exactly, and
(e) added an Objects365 workload so the matrix has a cell where COCO-scale
conclusions stop applying.

The methodology changes are ADR-0049. Read that first if you only
want the "why"; this doc is the measurement.

## Shared configuration

- **Harness mode**: release (N=10 + 2 warmup, randomised impl order, 5%
  relative-IQR gate per impl) for every COCO cell. The Objects365 cells are
  **dev mode (N=1)** — pycocotools alone needs ~6 minutes per rep there.
- **Git SHA**: `b012b46c9087`
- **Machine fingerprint**: `59aab88b17f4` — KVM VPS, AMD EPYC-Milan, 4 physical
  cores x SMT-2 = 8 logical CPUs, 30 GiB RAM, **no swap**. A different
  fingerprint from the 2026-05 snapshot (`37652a58e939`), so absolute numbers
  are not comparable across the two; ratios within this snapshot are.
- **Build profile**: cargo release defaults (`opt-level=3`, `lto=thin`,
  `codegen-units=1`, no `target-cpu`) — the PyPI wheel's profile.
- **Baselines**: `hotcoco==1.0.1`, `faster-coco-eval==1.8.0`,
  `pycocotools==2.0.11`, `boundary-iou-api @ 37d2558`, `lvis-api @ 031ac21`.
  hotcoco and faster-coco-eval are installed **wheel-only**
  (`tool.uv.no-build-package`) so we measure what `pip install` delivers; the
  1.7.2 env had been compiling faster-coco-eval from sdist on this host.

### What changed in how a cell is measured

1. **The CPU budget is enforced.** faster-coco-eval 1.8 parallelises RLE IoU
   (`rle_iou_max_workers`, default 8) plus image evaluation and accumulation in
   C++ pools sized from `hardware_concurrency()` with no knob; hotcoco uses
   rayon's global pool; mmsegmentation runs on torch, whose intra-op pool took
   ~4 CPUs. Every batch runner — instance, LVIS, panoptic, semantic — is now
   pinned to `cpu_budget(num_threads)` logical CPUs before `exec` — 1 CPU for
   the headline, N for an `nt=N` cell — one CPU per physical core before any
   SMT sibling. The budget is also forwarded to each library's own knob
   (`RAYON_NUM_THREADS`, `rle_iou_max_workers`, `boundary_cpu_count`,
   `torch.set_num_threads`).
2. **Evidence rides along.** Each stage records process CPU time and the `total`
   stage records the affinity mask the runner observed, so a reader can check
   the budget held instead of trusting the harness. The rendered tables carry a
   CPU/wall column.
3. **Memory is exact, and split in two.** Each stage resets the kernel's RSS
   high-water mark (`/proc/self/clear_refs`) and reads `VmHWM` afterwards, both
   outside the timed span. "Peak RSS" is the absolute peak over the timed
   stages; "eval Δ RSS" subtracts the RSS captured just before the first stage,
   which isolates the evaluation from interpreter and import cost.
   **Caveat**: resetting `VmHWM` also resets the accounting behind
   `getrusage(...).ru_maxrss`, so that field is no longer a process-lifetime
   peak for these runners and is not reported.

Timed span, unchanged and identical for every impl: annotation files on disk →
summary stats, JSON parsing included, interpreter start-up and imports
excluded. Per-stage splits stay non-comparable across impls — vernier parses
GT/DT inside `evaluate`, the pycocotools-shaped libraries parse in `load` — so
only the total is compared.

## Instance — `coco_val2017_jittered_seed0`, 1 CPU

5000 val2017 images, deterministic jittered DT (seed 0).

### bbox

| impl             |   median | IQR              | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| ---------------- | -------: | ---------------: | -------: | -------: | ---------: | ---------: |
| **vernier**      | 356.3 ms | 3.6 ms (1.02%)   |     1.00 |  256 MiB |    179 MiB |  **1.00x** |
| hotcoco          | 575.7 ms | 40.5 ms (7.04%)* |     1.00 |  274 MiB |    226 MiB |      1.62x |
| faster-coco-eval |  1.631 s | 23.2 ms (1.43%)  |     1.00 |  692 MiB |    642 MiB |      4.58x |
| pycocotools      |  5.541 s | 99.6 ms (1.80%)  |     1.00 |  577 MiB |    529 MiB |     15.55x |

### segm

| impl             |   median | IQR             | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| ---------------- | -------: | --------------: | -------: | -------: | ---------: | ---------: |
| **vernier**      | 975.8 ms | 9.5 ms (0.97%)  |     1.00 |  256 MiB |    180 MiB |  **1.00x** |
| hotcoco          |  1.338 s | 19.0 ms (1.42%) |     1.00 |  361 MiB |    313 MiB |      1.37x |
| faster-coco-eval |  3.357 s | 18.4 ms (0.55%) |     1.00 |  744 MiB |    694 MiB |      3.44x |
| pycocotools      |  6.443 s | 144 ms (2.24%)  |     1.00 |  569 MiB |    521 MiB |      6.60x |

### boundary

hotcoco has no boundary-IoU surface, so the impl set is unchanged.

| impl             |  median | IQR             | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| ---------------- | ------: | --------------: | -------: | -------: | ---------: | ---------: |
| **vernier**      | 3.168 s | 21.0 ms (0.66%) |     1.00 |  258 MiB |    181 MiB |  **1.00x** |
| faster-coco-eval | 52.81 s | 98.6 ms (0.19%) |     1.00 |  813 MiB |    764 MiB |     16.67x |
| boundary-iou-api | 61.57 s | 46.6 ms (0.08%) |     1.00 |  667 MiB |    596 MiB |     19.43x |

The faster-coco-eval boundary number moved from 17.6 s (2026-05) to 52.8 s, and
**that is a methodology correction, not a regression**: `boundary_cpu_count`
defaults to `min(cpu_count, 4)`, so the old "single-threaded" headline was
vernier on 1 thread against faster-coco-eval on 4. At `nt=4` below it lands at
18.6 s, in line with the old figure.

### keypoints — `coco_val2017_keypoints_jittered_seed0`

| impl             |   median | IQR             | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| ---------------- | -------: | --------------: | -------: | -------: | ---------: | ---------: |
| **vernier**      | 134.5 ms | 941 μs (0.70%)  |     1.00 |  128 MiB |     53 MiB |  **1.00x** |
| hotcoco          | 210.3 ms | 2.7 ms (1.27%)  |     1.00 |  123 MiB |     75 MiB |      1.56x |
| faster-coco-eval | 771.9 ms | 8.8 ms (1.14%)  |     1.00 |  165 MiB |    115 MiB |      5.74x |
| pycocotools      |  2.298 s | 36.4 ms (1.59%) |     1.00 |  163 MiB |    115 MiB |     17.09x |

faster-coco-eval 1.8's vectorised `computeOks` is the biggest single-version
win in this round: 12.3x slower than vernier at 1.7.2, 5.7x now.

`*` The hotcoco bbox cell missed the 5% IQR gate (7.04%). Its `evaluate` stage
is bimodal in-harness (~152 ms / ~195 ms) while a standalone loop on an idle
core is not, and a pinning-vs-sibling probe (8 runs each) showed 1-CPU pinning
is if anything *faster* for hotcoco than letting its rayon worker have a second
hyperthread (556 ms vs 578 ms median total). Treat it as host noise; the median
is unaffected across three separate matrix runs.

## Thread scaling

Same workloads, `num_threads ∈ {1,2,4,8}`, each cell pinned to that many CPUs.
pycocotools and boundary-iou-api are single-threaded and omitted. On this
8-vCPU host the `nt=8` column is effectively each library's out-of-the-box
configuration.

### bbox — median total (ratio vs vernier at the same nt)

| impl             |          nt=1 |          nt=2 |          nt=4 |          nt=8 |
| ---------------- | ------------: | ------------: | ------------: | ------------: |
| **vernier**      |      358.4 ms |      350.2 ms |      316.4 ms |      313.6 ms |
| hotcoco          | 587.5 (1.64x) | 431.2 (1.23x) | 364.3 (1.15x) | 351.7 (1.12x) |
| faster-coco-eval | 1.632 (4.55x) | 1.503 (4.29x) | 1.449 (4.58x) | 1.436 (4.58x) |

### segm — median total

| impl             |          nt=1 |          nt=2 |          nt=4 |          nt=8 |
| ---------------- | ------------: | ------------: | ------------: | ------------: |
| **vernier**      |      980.2 ms |      651.8 ms |      468.1 ms |      407.7 ms |
| hotcoco          | 1.327 (1.35x) | 834.5 (1.28x) | 583.1 (1.25x) | 529.8 (1.30x) |
| faster-coco-eval | 3.368 (3.44x) | 3.436 (5.27x) | 3.398 (7.26x) | 3.398 (8.33x) |

### boundary — median total

| impl             |          nt=1 |           nt=2 |           nt=4 |           nt=8 |
| ---------------- | ------------: | -------------: | -------------: | -------------: |
| **vernier**      |       3.155 s |        1.769 s |        1.022 s |       866.8 ms |
| faster-coco-eval | 52.73 (16.7x) | 29.90 (16.90x) | 18.58 (18.18x) | 16.70 (19.27x) |

### keypoints — median total

| impl             |          nt=1 |          nt=2 |          nt=4 |          nt=8 |
| ---------------- | ------------: | ------------: | ------------: | ------------: |
| **vernier**      |      133.3 ms |      117.8 ms |      105.5 ms |       99.3 ms |
| hotcoco          | 209.1 (1.57x) | 177.2 (1.50x) | 160.3 (1.52x) | 156.0 (1.57x) |
| faster-coco-eval | 760.8 (5.71x) | 749.1 (6.36x) | 743.0 (7.04x) | 746.1 (7.51x) |

### Read against the table

- **vernier's bbox cell is parse-bound**, as it was in the 2026-05 round: 358 →
  314 ms across 8x the CPUs. segm (2.4x at nt=8) and boundary (3.6x) scale.
- **hotcoco scales bbox best in relative terms** (587 → 352 ms, 1.67x) and stays
  1.12–1.64x behind vernier at every thread count.
- **faster-coco-eval 1.8's parallelism is real only on boundary** (3.2x from
  nt=1 to nt=8). segm and keypoints are flat within noise, and bbox gains ~12%.
  Since vernier does scale on those kernels, the gap *widens* with thread count:
  segm goes from 3.44x at nt=1 to 8.33x at nt=8.
- **Memory is flat in the thread count** for every impl (vernier segm 180 → 184
  MiB across nt=1..8), so the thread budget is not paid for in RSS.

## Objects365 — the scale cell

`objects365_val_jittered_seed0`, dev mode (N=1), bbox only: 80,000 images,
1,240,587 GT boxes, 365 categories, ~1.06M jittered detections. Annotations are
CC BY 4.0 from the Objects365 Consortium; images are never downloaded.

| impl             |    total | CPU/wall |  peak RSS | eval Δ RSS |     vs vernier |
| ---------------- | -------: | -------: | --------: | ---------: | -------------: |
| **vernier**      |  9.997 s |     1.00 |  4.78 GiB |   4.71 GiB |      **1.00x** |
| hotcoco          | 14.784 s |     1.00 |  6.49 GiB |   6.45 GiB |          1.48x |
| pycocotools      | 357.45 s |     1.00 | 21.34 GiB |  21.30 GiB |         35.76x |
| faster-coco-eval |  **DNF** |        — |   ~30 GiB |          — | OOM-killed (1) |

(1) `Out of memory: Killed process … (python3) total-vm:31557940kB,
anon-rss:30912672kB` — the runner exceeded the host's 30 GiB with no swap, at
both `nt=1` and `nt=8`. hotcoco's own Objects365 benchmark reports the same
failure mode (~30 GiB committed for faster-coco-eval on a 16 GiB host). The
harness aborts a whole cell when one runner dies, so this cell was re-run
per-impl; a DNF is not yet a first-class result state.

At `nt=8`: hotcoco 7.78 s (4.94 GiB), vernier 8.76 s (5.14 GiB). **hotcoco is
faster than vernier here** — the one cell in this round where that happens.
Both only reach ~2.4–2.6 CPU/wall, so both are dominated by serial JSON parsing
at this size; hotcoco spends 5.7 s of its total in `load` against vernier's
0.3 s (vernier parses inside `evaluate`, so the stages are not comparable, but
the totals are).

Scale changes the ranking, which is the point of having the cell:
vernier's lead over pycocotools grows from 15.6x (COCO bbox) to 35.8x, its lead
over hotcoco shrinks from 1.62x to 1.48x and inverts at nt=8, and
faster-coco-eval stops finishing at all.

## Panoptic and semantic

Re-measured on this host only so the whole page shares one SHA and one
fingerprint; neither baseline changed this round. Both now run under the same
1-CPU budget as everything else, which is what moved the semantic ratio.

| paradigm · workload                       |     vernier |         oracle | vs vernier |
| ----------------------------------------- | ----------: | -------------: | ---------: |
| panoptic · `coco_panoptic_val2017_perfect` |    10.520 s | 34.512 s (panopticapi) |     3.28x |
| semantic · `coco_val2017_semantic_perfect` |     2.856 s | 39.906 s (mmsegmentation) |    13.97x |
| semantic · `synthetic_semantic_n200_c19_s0` |   65.8 ms | 534.2 ms (mmsegmentation) |     8.12x |

- **mmsegmentation was never single-threaded.** Unpinned it ran at 3.99
  CPU/wall (torch intra-op) for 16.679 s on val2017; pinned to one CPU it takes
  39.906 s. The published 2026-05 ratio of 7.40x was vernier on one thread
  against mmseg on ~4 — per equal CPU it is 13.97x. The CPU/wall column exists
  so this cannot hide again.
- **vernier's panoptic cell pays for the budget.** Unpinned it ran at 1.18
  CPU/wall (the `BackgroundPanopticEvaluator`'s producer/consumer overlap with
  its worker thread) for 8.865 s; pinned it is 10.520 s. The 1-CPU number is
  the honest one for a single-CPU comparison, and 10.5 s also happens to match
  the 2026-05 figure on the other host.
- The panoptic cell's chronic IQR-gate failure did not reproduce: 1.31%.

## Parity

Every COCO cell passed its tiers. Measured divergences, all at 1 ULP
(`2.22e-16`) of the precision tensor:

| pair                           |               bbox |               segm |     keypoints |            O365 bbox |
| ------------------------------ | -----------------: | -----------------: | ------------: | -------------------: |
| vernier vs pycocotools         |          bit-equal |          bit-equal |     bit-equal |            bit-equal |
| hotcoco vs vernier             |  7589 cells @ 1ULP |  8753 cells @ 1ULP | 6 cells @1ULP | 8862 cells @ 1ULP    |
| faster-coco-eval vs vernier    |  7589 cells @ 1ULP |  8753 cells @ 1ULP | 6 cells @1ULP |                    — |

- **vernier remains bit-exact against pycocotools** at every size tested,
  including 1.24M boxes across 365 categories.
- hotcoco and faster-coco-eval diverge from pycocotools in the *same* cells and
  by the same magnitude — a shared float-order choice in the accumulation, not
  independent bugs. Both sit inside the aligned tier (4 ULP).
- Summary stats: faster-coco-eval reproduces pycocotools' 12 stats exactly;
  hotcoco's drift up to 3.2e-14, consistent with the ≤3.7e-14 its own docs
  claim.

## LVIS — federated AP

`lvis_v1_val_perfect` is release mode; the jittered cell is dev mode (N=1) and
exists to answer one question, below.

| impl        | perfect-DT median |  peak RSS | jittered median (dev) |
| ----------- | ----------------: | --------: | --------------------: |
| **vernier** |           3.420 s |  1.44 GiB |               3.741 s |
| hotcoco     |    3.508 s (1.03x) |  1.46 GiB |        3.892 s (1.04x) |
| lvis-api    |  187.58 s (54.85x) | 15.01 GiB |       176.49 s (47.2x) |

Two results worth recording.

**vernier is now bit-equal to lvis-api** on both cells — identical tensor
SHA-256 (`e5612b8fe6ad` on perfect-DT). The 2026-06 open follow-up recorded
2730 divergent cells concentrated on K=168 / K=817 with vernier reading 1.0
where lvis-api read 0.987–0.99904. That is closed: the bench cell now feeds the
**bbox-shaped** perfect-DT, and lvis-api assigns area buckets from the DT's
`area` field, which the segm-shaped file derives from the rasterised mask.

**hotcoco diverges from the lvis-api reference by more than float noise.** On
perfect-DT: 6060 cells, worst 9.6e-4, AP 0.99965 vs 0.99826. Perfect-DT gives
every detection `score=1.0`, so score-tie ordering was the obvious suspect —
hence the jittered cell, whose scores are drawn from a beta and are distinct.
It diverges there too: 7069 cells, worst 8.0e-3 (first at `[T=0, R=0, K=60,
A=1]`: vernier/lvis-api 0.65116, hotcoco 0.65919), AP 0.320582 vs 0.320533.
So it is not tie-breaking. Not root-caused here; the candidates are the
federated pieces (`not_exhaustive_category_ids` / `neg_category_ids` handling,
frequency-group assignment). Recorded as a divergence against the reference,
not as a hotcoco bug — hotcoco publishes no LVIS parity claim against
lvis-api.

### LVIS cache pin bug

The LVIS cell could not run at first: `lvis_val_cache.ensure_gt` compared the
**extracted JSON** against a SHA-256 that is actually the **zip's**
(`5cae9a3c…`), so it raised on every clean cache. The sibling
`lvis_v1_val_cache` pins the same value under the correct name
(`GT_ZIP_SHA256`) and verifies the archive, and its comment already described
that as "the convention `lvis_val_cache` uses" — which it wasn't.

A fresh download hashes to `5cae9a3c…`, so the upstream artefact has **not**
drifted and ADR-0026's "bumping the pin is ADR-level" rule is not engaged: the
pinned bytes are unchanged, only the file they were being compared against was
wrong. Fixed by verifying the zip before extraction. `just test-parity-lvis-val`
was broken the same way from a clean cache.

## How to refresh

```bash
just bench-sync
# COCO headline (release, all impls, one CPU each)
for iou in bbox segm keypoints boundary; do
  just bench-run --impl all --workload coco_val2017_jittered_seed0 --iou $iou
done
# Thread scaling
just bench-run --impl hotcoco --workload coco_val2017_jittered_seed0 \
    --iou segm --num-threads 1,2,4,8 --no-parity
# Objects365 (per-impl: one OOM must not abort the others)
just bench-run --impl vernier --workload objects365_val_jittered_seed0 \
    --iou bbox --mode dev --no-parity
python tools/render_benchmarks.py
```

## Follow-ups

- **DNF as a result state.** A runner that OOMs aborts its whole cell. Recording
  "did not finish, reason" per impl would let a scale cell run all impls in one
  pass and render an honest blank.
- **hotcoco's `load` stage at scale.** 5.7 s of its 7.8 s `nt=8` Objects365 total
  is serial parsing; vernier's parse is inside `evaluate` and threaded. Worth a
  per-stage comparison the current stage split can't express.
- **hotcoco beats vernier at `nt=8` on Objects365.** One data point, dev mode.
  Worth a release-mode confirmation before drawing any conclusion.
