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

**Provenance** — git SHA `b012b46c9087` · machine fingerprint `59aab88b17f4` · CPU AMD EPYC-Milan Processor (x86_64) · harness
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
| **vernier** | 356.3 ms | 3.6 ms (1.02%) | 1.00 | 256 MiB | 179 MiB | **1.00×** |
| hotcoco | 575.7 ms | 40.5 ms (7.04%) * | 1.00 | 274 MiB | 226 MiB | 1.62× |
| faster-coco-eval | 1.631 s | 23.2 ms (1.43%) | 1.00 | 692 MiB | 642 MiB | 4.58× |
| pycocotools | 5.541 s | 99.6 ms (1.80%) | 1.00 | 577 MiB | 529 MiB | 15.55× |

**`segm`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 975.8 ms | 9.5 ms (0.97%) | 1.00 | 256 MiB | 180 MiB | **1.00×** |
| hotcoco | 1.338 s | 19.0 ms (1.42%) | 1.00 | 361 MiB | 313 MiB | 1.37× |
| faster-coco-eval | 3.357 s | 18.4 ms (0.55%) | 1.00 | 744 MiB | 694 MiB | 3.44× |
| pycocotools | 6.443 s | 144.1 ms (2.24%) | 1.00 | 569 MiB | 521 MiB | 6.60× |

**`boundary`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 3.168 s | 21.0 ms (0.66%) | 1.00 | 258 MiB | 181 MiB | **1.00×** |
| faster-coco-eval | 52.812 s | 98.6 ms (0.19%) | 1.00 | 813 MiB | 764 MiB | 16.67× |
| boundary-iou-api | 61.566 s | 46.6 ms (0.08%) | 1.00 | 667 MiB | 596 MiB | 19.43× |

### Workload: `coco_val2017_keypoints_jittered_seed0`

**`keypoints`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 134.5 ms | 941.2 μs (0.70%) | 1.00 | 129 MiB | 53 MiB | **1.00×** |
| hotcoco | 210.3 ms | 2.7 ms (1.27%) | 1.00 | 123 MiB | 75 MiB | 1.56× |
| faster-coco-eval | 771.9 ms | 8.8 ms (1.14%) | 1.00 | 165 MiB | 115 MiB | 5.74× |
| pycocotools | 2.298 s | 36.4 ms (1.59%) | 1.00 | 163 MiB | 115 MiB | 17.09× |

### Workload: `objects365_val_jittered_seed0`

*Scale workload: Objects365 v2 val, 80,000 images · 1,240,587 GT boxes · 365 categories, with ~1.06 M jittered detections (bbox only). Annotations © [Objects365 Consortium](https://www.objects365.org/), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); images are never downloaded.*

*Recorded in harness mode `dev` (not `release`): one measurement rep per impl, no IQR gate.*

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 9.997 s | 0 ns | 1.00 | 4.78 GiB | 4.71 GiB | **1.00×** |
| hotcoco | 14.784 s | 0 ns | 1.00 | 6.49 GiB | 6.45 GiB | 1.48× |
| pycocotools | 357.446 s | 0 ns | 1.00 | 21.34 GiB | 21.30 GiB | 35.76× |


## Panoptic — PQ

### Workload: `coco_panoptic_val2017_perfect`

**`pq`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 10.520 s | 137.4 ms (1.31%) | 1.00 | 144 MiB | 68 MiB | **1.00×** |
| panopticapi | 34.512 s | 335.1 ms (0.97%) | 1.00 | 145 MiB | 95 MiB | 3.28× |


## Semantic — mIoU

### Workload: `coco_val2017_semantic_perfect`

**`miou`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 2.856 s | 4.3 ms (0.15%) | 1.00 | 97 MiB | 21 MiB | **1.00×** |
| mmsegmentation | 39.906 s | 406.5 ms (1.02%) | 1.00 | 546 MiB | 43 MiB | 13.97× |

### Workload: `synthetic_semantic_n200_c19_s0`

**`miou`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 65.8 ms | 1.5 ms (2.24%) | 1.00 | 78 MiB | 2 MiB | **1.00×** |
| mmsegmentation | 534.2 ms | 11.0 ms (2.06%) | 1.00 | 509 MiB | 11 MiB | 8.12× |


## Instance — LVIS federated AP

### Workload: `lvis_v1_val_jittered_seed0`

*Recorded in harness mode `dev` (not `release`): one measurement rep per impl, no IQR gate.*

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 3.741 s | 0 ns | 1.00 | 1.59 GiB | 1.52 GiB | **1.00×** |
| hotcoco | 3.892 s | 0 ns | 1.00 | 1.47 GiB | 1.43 GiB | 1.04× |
| lvis-api | 176.491 s | 0 ns | 1.00 | 15.08 GiB | 14.98 GiB | 47.18× |

### Workload: `lvis_v1_val_perfect`

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 3.420 s | 68.5 ms (2.00%) | 1.00 | 1.44 GiB | 1.37 GiB | **1.00×** |
| hotcoco | 3.508 s | 27.5 ms (0.79%) | 1.00 | 1.46 GiB | 1.41 GiB | 1.03× |
| lvis-api | 187.584 s | 1.833 s (0.98%) | 1.00 | 15.01 GiB | 14.91 GiB | 54.85× |


## Thread scaling

**`coco_val2017_jittered_seed0` · `bbox`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 358.4 ms | 350.2 ms | 316.4 ms | 313.6 ms |
| hotcoco | 587.5 ms (1.64×) | 431.2 ms (1.23×) | 364.3 ms (1.15×) | 351.7 ms (1.12×) |
| faster-coco-eval | 1.632 s (4.55×) | 1.503 s (4.29×) | 1.449 s (4.58×) | 1.436 s (4.58×) |

**`coco_val2017_jittered_seed0` · `bbox`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 179 MiB | 180 MiB | 181 MiB | 181 MiB |
| hotcoco | 226 MiB | 227 MiB | 228 MiB | 229 MiB |
| faster-coco-eval | 642 MiB | 642 MiB | 642 MiB | 641 MiB |

**`coco_val2017_jittered_seed0` · `boundary`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 3.155 s | 1.769 s | 1.022 s | 866.8 ms |
| faster-coco-eval | 52.731 s (16.71×) | 29.897 s (16.90×) | 18.579 s (18.18×) | 16.701 s (19.27×) |

**`coco_val2017_jittered_seed0` · `boundary`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 181 MiB | 184 MiB | 190 MiB | 198 MiB |
| faster-coco-eval | 764 MiB | 771 MiB | 771 MiB | 764 MiB |

**`coco_val2017_jittered_seed0` · `segm`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 980.2 ms | 651.8 ms | 468.1 ms | 407.7 ms |
| hotcoco | 1.327 s (1.35×) | 834.5 ms (1.28×) | 583.1 ms (1.25×) | 529.8 ms (1.30×) |
| faster-coco-eval | 3.368 s (3.44×) | 3.436 s (5.27×) | 3.398 s (7.26×) | 3.398 s (8.33×) |

**`coco_val2017_jittered_seed0` · `segm`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 180 MiB | 181 MiB | 182 MiB | 184 MiB |
| hotcoco | 313 MiB | 314 MiB | 316 MiB | 318 MiB |
| faster-coco-eval | 694 MiB | 694 MiB | 694 MiB | 694 MiB |

**`coco_val2017_keypoints_jittered_seed0` · `keypoints`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 133.3 ms | 117.8 ms | 105.5 ms | 99.3 ms |
| hotcoco | 209.1 ms (1.57×) | 177.2 ms (1.50×) | 160.3 ms (1.52×) | 156.0 ms (1.57×) |
| faster-coco-eval | 760.8 ms (5.71×) | 749.1 ms (6.36×) | 743.0 ms (7.04×) | 746.1 ms (7.51×) |

**`coco_val2017_keypoints_jittered_seed0` · `keypoints`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 53 MiB | 53 MiB | 54 MiB | 54 MiB |
| hotcoco | 75 MiB | 75 MiB | 76 MiB | 77 MiB |
| faster-coco-eval | 115 MiB | 115 MiB | 115 MiB | 115 MiB |

**`objects365_val_jittered_seed0` · `bbox`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=8` |
| --- | ---: |
| **vernier** | 8.762 s |
| hotcoco | 7.777 s (0.89×) |

**`objects365_val_jittered_seed0` · `bbox`** — eval Δ RSS

| impl | `nt=8` |
| --- | ---: |
| **vernier** | 4.94 GiB |
| hotcoco | 6.47 GiB |


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
side effect of every timing run — strict-tier (vs pycocotools) and
aligned-tier (vs faster-coco-eval and hotcoco) where applicable;
a failed tier writes a divergence report next to the cell.
