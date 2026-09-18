# Benchmarks

Comparison of vernier against the third-party libraries it targets parity
against, on a single machine and a single git revision. The numbers below
are the median total-stage wall time over the non-warmup reps recorded by
the local bench harness ([ADR-0017](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0017-local-bench-harness.md),
extended cross-paradigm in
[ADR-0033](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0033-multi-paradigm-bench.md)).
The IQR column reports the spread (Q3 - Q1) across the 10 measurement
reps and the same value as a percentage of the median; release mode
gates each cell at 5% relative IQR.

**Provenance** — git SHA `e361050ef582` · machine fingerprint `59aab88b17f4` · CPU AMD EPYC-Milan Processor (x86_64) · harness
mode `release` · build profile = cargo release defaults
(`opt-level=3`, `lto=thin`, `codegen-units=1`, no `target-cpu`). The
release wheel on PyPI is built with the same profile — no
benchmarking-only flags.

**Baselines pinned for these numbers** — [`hotcoco==1.0.1`](https://pypi.org/project/hotcoco/1.0.1/) · [`faster-coco-eval==1.8.0`](https://pypi.org/project/faster-coco-eval/1.8.0/) · [`pycocotools==2.0.11`](https://pypi.org/project/pycocotools/2.0.11/) · [`boundary-iou-api` @ `37d2558`](https://github.com/bowenc0221/boundary-iou-api/commit/37d25586a677) · [`panopticapi` @ `7bb4655`](https://github.com/cocodataset/panopticapi/commit/7bb4655548f9) · [`mmsegmentation` @ `c685fe6`](https://github.com/open-mmlab/mmsegmentation/commit/c685fe6767c4cadf6b051983ca6208f1b9d1ccb8) · [`lvis-api` @ `031ac21`](https://github.com/lvis-dataset/lvis-api/commit/031ac21f939bcb5f1ca8de2ab8704082e101ff9b). Each baseline is locked in its own uv-managed venv per ADR-0017.

For the full per-cell deep-dive (per-stage breakdown, RSS evolution,
parity gating, narrative on what moved each round), see
[`docs/engineering/benchmarking/`](https://github.com/NoeFontana/vernier/tree/main/docs/engineering/benchmarking).

This page is regenerated from the harness result tree by
`tools/render_benchmarks.py`. To refresh after a new bench run, see the
[release runbook](https://github.com/NoeFontana/vernier/blob/main/docs/engineering/release-runbook.md)
§0.

## Instance — bbox / segm / boundary / keypoints (AP)

### Workload: `coco_val2017_jittered_seed0`

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 354.3 ms | 4.3 ms (1.22%) | 1.00 | 255 MiB | 179 MiB | **1.00×** |
| hotcoco | 559.5 ms | 9.1 ms (1.63%) | 1.00 | 274 MiB | 226 MiB | 1.58× |
| faster-coco-eval | 1.647 s | 23.8 ms (1.45%) | 1.00 | 691 MiB | 642 MiB | 4.65× |
| pycocotools | 5.667 s | 31.8 ms (0.56%) | 1.00 | 576 MiB | 529 MiB | 15.99× |

**`segm`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 967.9 ms | 3.8 ms (0.39%) | 1.00 | 255 MiB | 180 MiB | **1.00×** |
| hotcoco | 1.351 s | 21.2 ms (1.57%) | 1.00 | 361 MiB | 313 MiB | 1.40× |
| faster-coco-eval | 3.401 s | 21.8 ms (0.64%) | 1.00 | 744 MiB | 694 MiB | 3.51× |
| pycocotools | 6.518 s | 101.5 ms (1.56%) | 1.00 | 569 MiB | 521 MiB | 6.73× |

**`boundary`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 3.191 s | 14.0 ms (0.44%) | 1.00 | 257 MiB | 182 MiB | **1.00×** |
| faster-coco-eval | 53.129 s | 473.5 ms (0.89%) | 1.00 | 813 MiB | 764 MiB | 16.65× |
| boundary-iou-api | 62.066 s | 643.3 ms (1.04%) | 1.00 | 666 MiB | 596 MiB | 19.45× |

### Workload: `coco_val2017_keypoints_jittered_seed0`

**`keypoints`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 136.0 ms | 1.9 ms (1.37%) | 1.00 | 128 MiB | 53 MiB | **1.00×** |
| hotcoco | 211.0 ms | 5.0 ms (2.37%) | 1.00 | 123 MiB | 75 MiB | 1.55× |
| faster-coco-eval | 776.7 ms | 8.1 ms (1.05%) | 1.00 | 164 MiB | 115 MiB | 5.71× |
| pycocotools | 2.302 s | 17.0 ms (0.74%) | 1.00 | 163 MiB | 115 MiB | 16.92× |

### Workload: `objects365_val_jittered_seed0`

*Scale workload: Objects365 v2 val, 80,000 images · 1,240,587 GT boxes · 365 categories, with ~1.06 M jittered detections (bbox only). Annotations © [Objects365 Consortium](https://www.objects365.org/), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); images are never downloaded. faster-coco-eval is absent because it does not finish: it is OOM-killed at ~30 GiB anon RSS on this 30 GiB host, at 1 and at 8 threads.*

*Recorded in harness mode `dev` (not `release`): one measurement rep per impl, no IQR gate.*

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 8.771 s | 0 ns | 1.00 | 4.78 GiB | 4.71 GiB | **1.00×** |
| hotcoco | 15.041 s | 0 ns | 1.00 | 6.49 GiB | 6.45 GiB | 1.71× |
| pycocotools | 367.433 s | 0 ns | 1.00 | 21.34 GiB | 21.30 GiB | 41.89× |


## Panoptic — PQ

### Workload: `coco_panoptic_val2017_perfect`

**`pq`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 10.564 s | 125.1 ms (1.18%) | 1.00 | 143 MiB | 67 MiB | **1.00×** |
| panopticapi | 34.971 s | 409.3 ms (1.17%) | 1.00 | 148 MiB | 92 MiB | 3.31× |


## Semantic — mIoU

### Workload: `coco_val2017_semantic_perfect`

**`miou`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 2.857 s | 16.1 ms (0.56%) | 1.00 | 97 MiB | 21 MiB | **1.00×** |
| mmsegmentation | 40.099 s | 231.0 ms (0.58%) | 1.00 | 544 MiB | 41 MiB | 14.04× |

### Workload: `synthetic_semantic_n200_c19_s0`

**`miou`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 66.0 ms | 209.8 μs (0.32%) | 1.00 | 78 MiB | 2 MiB | **1.00×** |
| mmsegmentation | 524.8 ms | 8.4 ms (1.61%) | 1.00 | 509 MiB | 11 MiB | 7.95× |


## Instance — LVIS federated AP

### Workload: `lvis_v1_val_perfect`

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 2.640 s | 32.6 ms (1.23%) | 1.00 | 1.45 GiB | 1.37 GiB | **1.00×** |
| hotcoco | 3.573 s | 35.7 ms (1.00%) | 1.00 | 1.46 GiB | 1.41 GiB | 1.35× |
| lvis-api | 193.001 s | 2.725 s (1.41%) | 1.00 | 15.01 GiB | 14.91 GiB | 73.12× |


## Thread scaling

**`coco_val2017_jittered_seed0` · `bbox`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 353.7 ms | 266.7 ms | 228.9 ms | 226.3 ms |
| hotcoco | 566.7 ms (1.60×) | 437.4 ms (1.64×) | 376.4 ms (1.64×) | 366.0 ms (1.62×) |
| faster-coco-eval | 1.674 s (4.73×) | 1.541 s (5.78×) | 1.496 s (6.54×) | 1.467 s (6.48×) |

**`coco_val2017_jittered_seed0` · `bbox`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 179 MiB | 182 MiB | 184 MiB | 185 MiB |
| hotcoco | 226 MiB | 227 MiB | 228 MiB | 229 MiB |
| faster-coco-eval | 642 MiB | 642 MiB | 642 MiB | 641 MiB |

**`coco_val2017_jittered_seed0` · `boundary`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 3.198 s | 1.700 s | 937.3 ms | 790.3 ms |
| faster-coco-eval | 53.208 s (16.64×) | 30.093 s (17.70×) | 18.731 s (19.98×) | 16.987 s (21.49×) |

**`coco_val2017_jittered_seed0` · `boundary`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 182 MiB | 189 MiB | 193 MiB | 201 MiB |
| faster-coco-eval | 764 MiB | 771 MiB | 771 MiB | 767 MiB |

**`coco_val2017_jittered_seed0` · `segm`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 982.6 ms | 569.1 ms | 375.3 ms | 319.1 ms |
| hotcoco | 1.355 s (1.38×) | 850.2 ms (1.49×) | 601.9 ms (1.60×) | 546.0 ms (1.71×) |
| faster-coco-eval | 3.444 s (3.50×) | 3.484 s (6.12×) | 3.428 s (9.13×) | 3.500 s (10.97×) |

**`coco_val2017_jittered_seed0` · `segm`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 180 MiB | 185 MiB | 186 MiB | 188 MiB |
| hotcoco | 313 MiB | 314 MiB | 316 MiB | 318 MiB |
| faster-coco-eval | 694 MiB | 694 MiB | 694 MiB | 694 MiB |

**`coco_val2017_keypoints_jittered_seed0` · `keypoints`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 135.8 ms | 117.6 ms | 109.2 ms | 103.6 ms |
| hotcoco | 211.3 ms (1.56×) | 180.9 ms (1.54×) | 162.7 ms (1.49×) | 160.0 ms (1.54×) |
| faster-coco-eval | 779.7 ms (5.74×) | 772.8 ms (6.57×) | 768.5 ms (7.04×) | 760.5 ms (7.34×) |

**`coco_val2017_keypoints_jittered_seed0` · `keypoints`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 53 MiB | 55 MiB | 55 MiB | 55 MiB |
| hotcoco | 75 MiB | 75 MiB | 76 MiB | 77 MiB |
| faster-coco-eval | 115 MiB | 115 MiB | 115 MiB | 115 MiB |

**`objects365_val_jittered_seed0` · `bbox`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=8` |
| --- | ---: |
| **vernier** | 4.705 s |
| hotcoco | 8.124 s (1.73×) |

**`objects365_val_jittered_seed0` · `bbox`** — eval Δ RSS

| impl | `nt=8` |
| --- | ---: |
| **vernier** | 4.84 GiB |
| hotcoco | 6.47 GiB |


## Methodology in one paragraph

Every cell runs in its own subprocess with its own uv-managed venv (one
per impl), so a single Python process never has competing
pycocotools-flavored packages on its `sys.path`. The harness records
`(load, evaluate, accumulate, summarize, total)` wall_ns per stage,
discards the warmup reps, and reports the median total plus the
inter-quartile range (IQR = Q3 - Q1, with the relative spread shown as
a percentage of the median). The timed span is the same for every impl:
annotation files on disk → summary stats, including JSON parsing and
index building, excluding interpreter start-up and imports. Per-stage
splits are *not* comparable across impls (vernier parses JSON inside
`evaluate`; the pycocotools-shaped libraries parse in `load`), so only
the total is reported. Instance and LVIS cells run under an enforced
CPU budget: every runner process is pinned (CPU affinity, one logical
CPU per physical core before SMT siblings) to 1 CPU for the headline
tables and `N` CPUs for `nt=N` cells, and the budget is also passed to
each library's own thread knob (vernier `num_threads`, hotcoco
`RAYON_NUM_THREADS`, faster-coco-eval `rle_iou_max_workers` /
`boundary_cpu_count`). The CPU/wall column is process CPU time over
wall time — ~1.00 means the impl used one core; anything well below
the budget means it spent wall time waiting rather than computing.
Memory is reported two ways. Peak RSS is the exact resident-memory
high-water mark over the timed stages (the kernel's `VmHWM`, reset
through `/proc/self/clear_refs` at every stage start, max across
stages and reps); it includes the interpreter and the library's
imports. eval Δ RSS is the median across reps of that peak minus RSS
just before the first stage: the memory the evaluation itself needed,
input parsing included.
Release mode (N=10 + 2 warmup) gates each impl on relative IQR ≤ 5%;
cells where the gate failed are marked with
` *` next to their IQR value — the median is still the best estimator,
just with a wider confidence band than the gate accepts. Parity is a
side effect of every timing run — the bit-equal tier (vs pycocotools)
and the float-tolerance tier (vs faster-coco-eval and hotcoco) where
applicable; a failed tier writes a divergence report next to the cell.
