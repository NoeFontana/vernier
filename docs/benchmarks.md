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

**Provenance** — git SHA `f0e39a6a8a98` · machine fingerprint `59aab88b17f4` · CPU AMD EPYC-Milan Processor (x86_64) · harness
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
| **vernier** | 304.6 ms | 3.3 ms (1.10%) | 1.00 | 200 MiB | 124 MiB | **1.00×** |
| hotcoco | 571.1 ms | 43.9 ms (7.69%) * | 1.00 | 274 MiB | 226 MiB | 1.87× |
| faster-coco-eval | 1.694 s | 16.7 ms (0.99%) | 1.00 | 691 MiB | 642 MiB | 5.56× |
| pycocotools | 5.710 s | 82.8 ms (1.45%) | 1.00 | 576 MiB | 528 MiB | 18.74× |

**`segm`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 931.1 ms | 9.9 ms (1.07%) | 1.00 | 200 MiB | 125 MiB | **1.00×** |
| hotcoco | 1.356 s | 43.5 ms (3.21%) | 1.00 | 361 MiB | 313 MiB | 1.46× |
| faster-coco-eval | 3.438 s | 11.7 ms (0.34%) | 1.00 | 744 MiB | 694 MiB | 3.69× |
| pycocotools | 6.671 s | 95.9 ms (1.44%) | 1.00 | 569 MiB | 521 MiB | 7.16× |

**`boundary`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 3.146 s | 158.0 ms (5.02%) * | 1.00 | 202 MiB | 127 MiB | **1.00×** |
| faster-coco-eval | 53.101 s | 1.397 s (2.63%) | 1.00 | 813 MiB | 764 MiB | 16.88× |
| boundary-iou-api | 62.128 s | 1.999 s (3.22%) | 1.00 | 666 MiB | 596 MiB | 19.75× |

### Workload: `coco_val2017_keypoints_jittered_seed0`

**`keypoints`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 137.0 ms | 1.2 ms (0.85%) | 1.00 | 120 MiB | 45 MiB | **1.00×** |
| hotcoco | 212.9 ms | 6.7 ms (3.13%) | 1.00 | 123 MiB | 75 MiB | 1.55× |
| faster-coco-eval | 776.3 ms | 5.5 ms (0.71%) | 1.00 | 164 MiB | 115 MiB | 5.67× |
| pycocotools | 2.300 s | 22.5 ms (0.98%) | 1.00 | 163 MiB | 115 MiB | 16.79× |

### Workload: `objects365_val_jittered_seed0`

*Scale workload: Objects365 v2 val, 80,000 images · 1,240,587 GT boxes · 365 categories, with ~1.06 M jittered detections (bbox only). Annotations © [Objects365 Consortium](https://www.objects365.org/), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); images are never downloaded. faster-coco-eval is absent because it does not finish: it is OOM-killed at ~30 GiB anon RSS on this 30 GiB host, at 1 and at 8 threads.*

*Recorded in harness mode `dev` (not `release`): one measurement rep per impl, no IQR gate.*

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 6.604 s | 0 ns | 1.00 | 2.54 GiB | 2.47 GiB | **1.00×** |
| hotcoco | 15.129 s | 0 ns | 1.00 | 6.49 GiB | 6.45 GiB | 2.29× |
| pycocotools | 373.241 s | 0 ns | 1.00 | 21.34 GiB | 21.30 GiB | 56.52× |


## Panoptic — PQ

### Workload: `coco_panoptic_val2017_perfect`

**`pq`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 10.461 s | 169.8 ms (1.62%) | 1.00 | 146 MiB | 68 MiB | **1.00×** |
| panopticapi | 34.444 s | 820.7 ms (2.38%) | 1.00 | 145 MiB | 95 MiB | 3.29× |


## Semantic — mIoU

### Workload: `coco_val2017_semantic_perfect`

**`miou`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 2.869 s | 12.1 ms (0.42%) | 1.00 | 97 MiB | 21 MiB | **1.00×** |
| mmsegmentation | 40.455 s | 719.6 ms (1.78%) | 1.00 | 545 MiB | 41 MiB | 14.10× |

### Workload: `synthetic_semantic_n200_c19_s0`

**`miou`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 64.8 ms | 262.7 μs (0.41%) | 1.00 | 78 MiB | 2 MiB | **1.00×** |
| mmsegmentation | 515.9 ms | 9.5 ms (1.85%) | 1.00 | 509 MiB | 10 MiB | 7.96× |


## Instance — LVIS federated AP

### Workload: `lvis_v1_val_jittered_seed0`

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 2.498 s | 29.3 ms (1.17%) | 1.00 | 1.11 GiB | 1.04 GiB | **1.00×** |
| hotcoco | 4.050 s | 61.7 ms (1.52%) | 1.00 | 1.47 GiB | 1.43 GiB | 1.62× |
| lvis-api | 202.298 s | 13.334 s (6.59%) * | 1.00 | 15.08 GiB | 14.98 GiB | 80.98× |

### Workload: `lvis_v1_val_perfect`

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 2.277 s | 41.3 ms (1.81%) | 1.00 | 981 MiB | 906 MiB | **1.00×** |
| hotcoco | 3.666 s | 35.9 ms (0.98%) | 1.00 | 1.46 GiB | 1.41 GiB | 1.61× |
| lvis-api | 206.394 s | 10.208 s (4.95%) | 1.00 | 15.01 GiB | 14.91 GiB | 90.63× |


## Thread scaling

**`coco_val2017_jittered_seed0` · `bbox`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 306.3 ms | 220.5 ms | 145.9 ms | 131.1 ms |
| hotcoco | 583.6 ms (1.91×) | 444.2 ms (2.02×) | 376.3 ms (2.58×) | 362.1 ms (2.76×) |
| faster-coco-eval | 1.667 s (5.44×) | 1.552 s (7.04×) | 1.503 s (10.30×) | 1.496 s (11.41×) |

**`coco_val2017_jittered_seed0` · `bbox`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 124 MiB | 129 MiB | 130 MiB | 131 MiB |
| hotcoco | 226 MiB | 227 MiB | 228 MiB | 229 MiB |
| faster-coco-eval | 642 MiB | 642 MiB | 642 MiB | 642 MiB |

**`coco_val2017_jittered_seed0` · `boundary`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 3.183 s | 1.680 s | 865.6 ms | 711.2 ms |
| faster-coco-eval | 52.965 s (16.64×) | 31.777 s (18.92×) | 18.835 s (21.76×) | 16.887 s (23.75×) |

**`coco_val2017_jittered_seed0` · `boundary`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 127 MiB | 134 MiB | 141 MiB | 148 MiB |
| faster-coco-eval | 764 MiB | 771 MiB | 771 MiB | 767 MiB |

**`coco_val2017_jittered_seed0` · `segm`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 926.1 ms | 536.3 ms | 303.9 ms | 239.3 ms |
| hotcoco | 1.350 s (1.46×) | 838.4 ms (1.56×) | 595.2 ms (1.96×) | 545.8 ms (2.28×) |
| faster-coco-eval | 3.469 s (3.75×) | 3.652 s (6.81×) | 3.578 s (11.77×) | 3.488 s (14.58×) |

**`coco_val2017_jittered_seed0` · `segm`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 125 MiB | 130 MiB | 131 MiB | 134 MiB |
| hotcoco | 313 MiB | 314 MiB | 316 MiB | 318 MiB |
| faster-coco-eval | 694 MiB | 694 MiB | 694 MiB | 694 MiB |

**`coco_val2017_keypoints_jittered_seed0` · `keypoints`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 139.0 ms | 89.8 ms | 59.7 ms | 52.1 ms |
| hotcoco | 210.9 ms (1.52×) | 179.0 ms (1.99×) | 164.8 ms (2.76×) | 160.8 ms (3.09×) |
| faster-coco-eval | 805.2 ms (5.79×) | 774.1 ms (8.62×) | 772.5 ms (12.95×) | 764.3 ms (14.67×) |
| pycocotools | 2.332 s (16.78×) | 2.313 s (25.76×) | 2.321 s (38.90×) | 2.295 s (44.05×) |

**`coco_val2017_keypoints_jittered_seed0` · `keypoints`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 45 MiB | 46 MiB | 47 MiB | 47 MiB |
| hotcoco | 75 MiB | 75 MiB | 76 MiB | 76 MiB |
| faster-coco-eval | 115 MiB | 115 MiB | 115 MiB | 115 MiB |
| pycocotools | 115 MiB | 115 MiB | 115 MiB | 115 MiB |

**`objects365_val_jittered_seed0` · `bbox`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=8` |
| --- | ---: |
| **vernier** | 3.200 s |
| hotcoco | 8.173 s (2.55×) |

**`objects365_val_jittered_seed0` · `bbox`** — eval Δ RSS

| impl | `nt=8` |
| --- | ---: |
| **vernier** | 2.59 GiB |
| hotcoco | 6.48 GiB |


*Cells marked ` *` next to their IQR exceeded the release-mode 5% relative-IQR gate. Median still reported; treat the gap to the next impl as the load-bearing signal rather than the precise ratio.*

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
