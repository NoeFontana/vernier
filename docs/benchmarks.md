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

**Provenance** — git SHA `c30db82867de` · machine fingerprint `a6ada888c850` · CPU AMD EPYC-Milan Processor (x86_64) · harness
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
| **vernier** | 299.6 ms | 3.3 ms (1.09%) | 1.00 | 199 MiB | 122 MiB | **1.00×** |
| hotcoco | 590.4 ms | 30.1 ms (5.10%) * | 1.00 | 274 MiB | 226 MiB | 1.97× |
| faster-coco-eval | 1.747 s | 31.6 ms (1.81%) | 1.00 | 691 MiB | 642 MiB | 5.83× |
| pycocotools | 6.712 s | 75.3 ms (1.12%) | 1.00 | 576 MiB | 528 MiB | 22.41× |

**`segm`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 910.1 ms | 13.6 ms (1.49%) | 1.00 | 199 MiB | 123 MiB | **1.00×** |
| hotcoco | 1.401 s | 41.9 ms (2.99%) | 1.00 | 361 MiB | 313 MiB | 1.54× |
| faster-coco-eval | 3.546 s | 91.3 ms (2.58%) | 1.00 | 744 MiB | 694 MiB | 3.90× |
| pycocotools | 7.541 s | 893.6 ms (11.85%) * | 1.00 | 569 MiB | 521 MiB | 8.29× |

**`boundary`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 3.115 s | 36.5 ms (1.17%) | 1.00 | 201 MiB | 125 MiB | **1.00×** |
| faster-coco-eval | 53.204 s | 256.7 ms (0.48%) | 1.00 | 813 MiB | 764 MiB | 17.08× |
| boundary-iou-api | 59.458 s | 1.670 s (2.81%) | 1.00 | 670 MiB | 597 MiB | 19.09× |

### Workload: `coco_val2017_keypoints_jittered_seed0`

**`keypoints`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 135.3 ms | 1.5 ms (1.12%) | 1.00 | 121 MiB | 44 MiB | **1.00×** |
| hotcoco | 213.1 ms | 8.8 ms (4.15%) | 1.00 | 123 MiB | 75 MiB | 1.58× |
| faster-coco-eval | 798.5 ms | 9.8 ms (1.23%) | 1.00 | 164 MiB | 115 MiB | 5.90× |
| pycocotools | 2.363 s | 12.9 ms (0.55%) | 1.00 | 163 MiB | 115 MiB | 17.47× |

### Workload: `coco_val2017_rfdetr_nano_v1.6.5.post0`

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 954.6 ms | 7.1 ms (0.74%) | 1.00 | 390 MiB | 313 MiB | **1.00×** |
| hotcoco | 2.291 s | 37.5 ms (1.64%) | 1.00 | 997 MiB | 949 MiB | 2.40× |
| faster-coco-eval | 5.438 s | 65.4 ms (1.20%) | 1.00 | 1.35 GiB | 1.30 GiB | 5.70× |
| pycocotools | 22.765 s | 339.5 ms (1.49%) | 1.00 | 1.40 GiB | 1.35 GiB | 23.85× |

### Workload: `coco_val2017_rfdetr_segnano_v1.6.5.post0`

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 1.288 s | 22.3 ms (1.73%) | 1.00 | 585 MiB | 508 MiB | **1.00×** |
| hotcoco | 2.590 s | 25.1 ms (0.97%) | 1.00 | 961 MiB | 913 MiB | 2.01× |
| faster-coco-eval | 5.436 s | 50.8 ms (0.94%) | 1.00 | 1.45 GiB | 1.40 GiB | 4.22× |
| pycocotools | 21.506 s | 544.7 ms (2.53%) | 1.00 | 1.47 GiB | 1.42 GiB | 16.70× |

**`segm`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 2.343 s | 13.9 ms (0.60%) | 1.00 | 586 MiB | 509 MiB | **1.00×** |
| hotcoco | 4.863 s | 68.8 ms (1.41%) | 1.00 | 1.44 GiB | 1.40 GiB | 2.08× |
| faster-coco-eval | 9.967 s | 64.8 ms (0.65%) | 1.00 | 1.71 GiB | 1.66 GiB | 4.25× |
| pycocotools | 22.271 s | 407.4 ms (1.83%) | 1.00 | 1.45 GiB | 1.40 GiB | 9.50× |

### Workload: `objects365_val_jittered_seed0`

*Scale workload: Objects365 v2 val, 80,000 images · 1,240,587 GT boxes · 365 categories, with ~1.06 M jittered detections (bbox only). Annotations © [Objects365 Consortium](https://www.objects365.org/), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); images are never downloaded. faster-coco-eval is absent because it does not finish: it is OOM-killed at ~30 GiB anon RSS on this 30 GiB host, at 1 and at 8 threads.*

*Recorded in harness mode `dev` (not `release`): one measurement rep per impl, no IQR gate.*

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 6.131 s | 0 ns | 1.00 | 2.48 GiB | 2.40 GiB | **1.00×** |
| hotcoco | 15.179 s | 0 ns | 1.00 | 6.49 GiB | 6.45 GiB | 2.48× |
| pycocotools | 367.966 s | 0 ns | 1.00 | 21.34 GiB | 21.30 GiB | 60.02× |


## Panoptic — PQ

### Workload: `coco_panoptic_val2017_perfect`

**`pq`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 10.463 s | 45.3 ms (0.43%) | 1.00 | 147 MiB | 68 MiB | **1.00×** |
| panopticapi | 34.447 s | 126.9 ms (0.37%) | 1.00 | 145 MiB | 95 MiB | 3.29× |


## Semantic — mIoU

### Workload: `coco_val2017_semantic_perfect`

**`miou`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 2.858 s | 8.0 ms (0.28%) | 1.00 | 99 MiB | 21 MiB | **1.00×** |
| mmsegmentation | 39.854 s | 311.4 ms (0.78%) | 1.00 | 533 MiB | 41 MiB | 13.94× |


## Instance — LVIS federated AP

### Workload: `lvis_v1_val_jittered_seed0`

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 2.318 s | 14.5 ms (0.63%) | 1.00 | 1.02 GiB | 966 MiB | **1.00×** |
| hotcoco | 4.043 s | 53.0 ms (1.31%) | 1.00 | 1.47 GiB | 1.43 GiB | 1.74× |
| lvis-api | 195.437 s | 2.683 s (1.37%) | 1.00 | 15.08 GiB | 14.98 GiB | 84.31× |


## Thread scaling

**`coco_val2017_jittered_seed0` · `bbox`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 293.4 ms | 212.8 ms | 141.2 ms | 124.7 ms |
| hotcoco | 578.3 ms (1.97×) | 440.3 ms (2.07×) | 375.9 ms (2.66×) | 364.6 ms (2.92×) |
| faster-coco-eval | 1.671 s (5.70×) | 1.547 s (7.27×) | 1.493 s (10.57×) | 1.478 s (11.85×) |

**`coco_val2017_jittered_seed0` · `bbox`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 122 MiB | 127 MiB | 128 MiB | 129 MiB |
| hotcoco | 226 MiB | 227 MiB | 228 MiB | 229 MiB |
| faster-coco-eval | 642 MiB | 642 MiB | 642 MiB | 641 MiB |

**`coco_val2017_jittered_seed0` · `boundary`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 3.102 s | 1.638 s | 853.0 ms | 703.2 ms |
| faster-coco-eval | 53.055 s (17.10×) | 30.101 s (18.38×) | 18.746 s (21.98×) | 16.892 s (24.02×) |

**`coco_val2017_jittered_seed0` · `boundary`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 125 MiB | 132 MiB | 138 MiB | 146 MiB |
| faster-coco-eval | 764 MiB | 771 MiB | 763 MiB | 763 MiB |

**`coco_val2017_jittered_seed0` · `segm`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 886.1 ms | 513.9 ms | 291.8 ms | 233.6 ms |
| hotcoco | 1.346 s (1.52×) | 840.4 ms (1.64×) | 588.7 ms (2.02×) | 543.2 ms (2.32×) |
| faster-coco-eval | 3.435 s (3.88×) | 3.515 s (6.84×) | 3.448 s (11.82×) | 3.468 s (14.84×) |

**`coco_val2017_jittered_seed0` · `segm`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 123 MiB | 128 MiB | 130 MiB | 132 MiB |
| hotcoco | 313 MiB | 314 MiB | 316 MiB | 318 MiB |
| faster-coco-eval | 694 MiB | 694 MiB | 694 MiB | 694 MiB |

**`coco_val2017_keypoints_jittered_seed0` · `keypoints`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 134.0 ms | 86.9 ms | 59.2 ms | 50.0 ms |
| hotcoco | 213.9 ms (1.60×) | 178.8 ms (2.06×) | 163.5 ms (2.76×) | 159.4 ms (3.19×) |
| faster-coco-eval | 793.1 ms (5.92×) | 778.4 ms (8.96×) | 767.7 ms (12.97×) | 762.1 ms (15.25×) |

**`coco_val2017_keypoints_jittered_seed0` · `keypoints`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 44 MiB | 46 MiB | 47 MiB | 47 MiB |
| hotcoco | 75 MiB | 75 MiB | 76 MiB | 77 MiB |
| faster-coco-eval | 115 MiB | 115 MiB | 115 MiB | 115 MiB |

**`objects365_val_jittered_seed0` · `bbox`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=8` |
| --- | ---: |
| **vernier** | 2.858 s |
| hotcoco | 8.326 s (2.91×) |

**`objects365_val_jittered_seed0` · `bbox`** — eval Δ RSS

| impl | `nt=8` |
| --- | ---: |
| **vernier** | 2.53 GiB |
| hotcoco | 6.46 GiB |


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
