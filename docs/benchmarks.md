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

**Provenance** — git SHA `b7b160d56be0` · machine fingerprint `a2bed1e4a18e` · CPU AMD EPYC-Milan Processor (x86_64) · harness
mode `release` · build profile = cargo release defaults
(`opt-level=3`, `lto=thin`, `codegen-units=1`, no `target-cpu`). The
release wheel on PyPI is built with the same profile — no
benchmarking-only flags.

**Baselines pinned for these numbers** — [`hotcoco==1.5.0`](https://pypi.org/project/hotcoco/1.5.0/) · [`ultrafast-pycocotools==0.1.14`](https://pypi.org/project/ultrafast-pycocotools/0.1.14/) · [`faster-coco-eval==1.8.1`](https://pypi.org/project/faster-coco-eval/1.8.1/) · [`pycocotools==2.0.11`](https://pypi.org/project/pycocotools/2.0.11/) · [`boundary-iou-api` @ `37d2558`](https://github.com/bowenc0221/boundary-iou-api/commit/37d25586a677) · [`panopticapi` @ `7bb4655`](https://github.com/cocodataset/panopticapi/commit/7bb4655548f9) · [`mmsegmentation` @ `c685fe6`](https://github.com/open-mmlab/mmsegmentation/commit/c685fe6767c4cadf6b051983ca6208f1b9d1ccb8) · [`lvis-api` @ `031ac21`](https://github.com/lvis-dataset/lvis-api/commit/031ac21f939bcb5f1ca8de2ab8704082e101ff9b). Each baseline is locked in its own uv-managed venv per ADR-0017.

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
| **vernier** | 134.3 ms | 3.8 ms (2.84%) | 1.00 | 173 MiB | 94 MiB | **1.00×** |
| hotcoco | 210.9 ms | 2.6 ms (1.21%) | 1.00 | 131 MiB | 83 MiB | 1.57× |
| ufcoco | 154.7 ms | 1.5 ms (0.94%) | 1.00 | 118 MiB | 69 MiB | 1.15× |
| faster-coco-eval | 1.622 s | 42.8 ms (2.64%) | 1.00 | 692 MiB | 642 MiB | 12.07× |
| pycocotools | 5.938 s | 256.5 ms (4.32%) | 1.00 | 577 MiB | 529 MiB | 44.21× |

**`segm`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 360.3 ms | 3.1 ms (0.86%) | 1.00 | 193 MiB | 114 MiB | **1.00×** |
| hotcoco | 833.8 ms | 6.4 ms (0.77%) | 1.00 | 203 MiB | 155 MiB | 2.31× |
| ufcoco | 731.4 ms | 4.7 ms (0.64%) | 1.00 | 154 MiB | 105 MiB | 2.03× |
| faster-coco-eval | 3.375 s | 18.6 ms (0.55%) | 1.00 | 744 MiB | 694 MiB | 9.37× |
| pycocotools | 6.849 s | 333.2 ms (4.86%) | 1.00 | 569 MiB | 521 MiB | 19.01× |

**`boundary`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 2.678 s | 9.4 ms (0.35%) | 1.00 | 193 MiB | 113 MiB | **1.00×** |
| ufcoco | 210.484 s | 7.873 s (3.74%) | 1.00 | 288 MiB | 239 MiB | 78.61× |
| faster-coco-eval | 53.394 s | 111.8 ms (0.21%) | 1.00 | 813 MiB | 764 MiB | 19.94× |
| boundary-iou-api | 59.633 s | 532.2 ms (0.89%) | 1.00 | 670 MiB | 597 MiB | 22.27× |

### Workload: `coco_val2017_keypoints_jittered_seed0`

**`keypoints`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 97.8 ms | 1.1 ms (1.13%) | 1.00 | 129 MiB | 50 MiB | **1.00×** |
| hotcoco | 124.9 ms | 2.1 ms (1.66%) | 1.00 | 87 MiB | 38 MiB | 1.28× |
| ufcoco | 117.8 ms | 1.8 ms (1.51%) | 1.00 | 85 MiB | 36 MiB | 1.20× |
| faster-coco-eval | 802.0 ms | 19.5 ms (2.43%) | 1.00 | 164 MiB | 115 MiB | 8.20× |
| pycocotools | 2.367 s | 10.8 ms (0.46%) | 1.00 | 163 MiB | 115 MiB | 24.21× |

### Workload: `coco_val2017_rfdetr_nano_v1.6.5.post0`

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 434.0 ms | 9.6 ms (2.22%) | 1.00 | 276 MiB | 197 MiB | **1.00×** |
| hotcoco | 676.6 ms | 15.0 ms (2.21%) | 1.00 | 267 MiB | 218 MiB | 1.56× |
| ufcoco | 543.2 ms | 11.6 ms (2.14%) | 1.00 | 224 MiB | 176 MiB | 1.25× |
| faster-coco-eval | 4.925 s | 168.2 ms (3.41%) | 1.00 | 1.35 GiB | 1.30 GiB | 11.35× |
| pycocotools | 23.387 s | 1.187 s (5.07%) * | 1.00 | 1.40 GiB | 1.35 GiB | 53.88× |

### Workload: `coco_val2017_rfdetr_segnano_v1.6.5.post0`

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 529.6 ms | 8.1 ms (1.53%) | 1.00 | 372 MiB | 293 MiB | **1.00×** |
| hotcoco | 837.6 ms | 15.1 ms (1.80%) | 1.00 | 347 MiB | 298 MiB | 1.58× |
| ufcoco | 641.9 ms | 11.7 ms (1.83%) | 1.00 | 329 MiB | 280 MiB | 1.21× |
| faster-coco-eval | 5.036 s | 65.2 ms (1.29%) | 1.00 | 1.45 GiB | 1.40 GiB | 9.51× |
| pycocotools | 21.880 s | 514.7 ms (2.35%) | 1.00 | 1.47 GiB | 1.42 GiB | 41.31× |

**`segm`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 1.195 s | 17.1 ms (1.43%) | 1.00 | 502 MiB | 423 MiB | **1.00×** |
| hotcoco | 2.008 s | 28.1 ms (1.40%) | 1.00 | 648 MiB | 599 MiB | 1.68× |
| ufcoco | 1.816 s | 45.2 ms (2.49%) | 1.00 | 380 MiB | 331 MiB | 1.52× |
| faster-coco-eval | 9.854 s | 227.4 ms (2.31%) | 1.00 | 1.71 GiB | 1.66 GiB | 8.24× |
| pycocotools | 23.878 s | 510.2 ms (2.14%) | 1.00 | 1.45 GiB | 1.40 GiB | 19.97× |

### Workload: `objects365_val_jittered_seed0`

*Scale workload: Objects365 v2 val, 80,000 images · 1,240,587 GT boxes · 365 categories, with ~1.06 M jittered detections (bbox only). Annotations © [Objects365 Consortium](https://www.objects365.org/), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); images are never downloaded. faster-coco-eval is absent because it does not finish: it is OOM-killed at ~30 GiB anon RSS on this 30 GiB host, at 1 and at 8 threads.*

*Recorded in harness mode `dev` (not `release`): one measurement rep per impl, no IQR gate.*

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 2.530 s | 0 ns | 1.00 | 987 MiB | 909 MiB | **1.00×** |
| hotcoco | 4.460 s | 0 ns | 1.00 | 1.23 GiB | 1.19 GiB | 1.76× |
| ufcoco | 3.261 s | 0 ns | 1.00 | 855 MiB | 807 MiB | 1.29× |
| pycocotools | 397.987 s | 0 ns | 1.00 | 21.34 GiB | 21.30 GiB | 157.32× |


## Panoptic — PQ

### Workload: `coco_panoptic_val2017_perfect`

**`pq`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 3.679 s | 91.9 ms (2.50%) | 1.00 | 204 MiB | 125 MiB | **1.00×** |
| hotcoco | 7.066 s | 36.9 ms (0.52%) | 1.00 | 91 MiB | 42 MiB | 1.92× |
| panopticapi | 35.353 s | 427.9 ms (1.21%) | 1.00 | 144 MiB | 94 MiB | 9.61× |


## Semantic — mIoU

### Workload: `coco_val2017_semantic_perfect`

**`miou`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 1.556 s | 9.3 ms (0.60%) | 1.00 | 102 MiB | 23 MiB | **1.00×** |
| mmsegmentation | 44.501 s | 764.1 ms (1.72%) | 1.00 | 290 MiB | 42 MiB | 28.60× |


## Instance — LVIS federated AP

### Workload: `lvis_v1_val_jittered_seed0`

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 1.045 s | 19.8 ms (1.89%) | 1.00 | 787 MiB | 709 MiB | **1.00×** |
| hotcoco | 1.856 s | 38.2 ms (2.06%) | 1.00 | 590 MiB | 542 MiB | 1.78× |
| lvis-api | 226.740 s | 4.584 s (2.02%) | 1.00 | 15.08 GiB | 14.98 GiB | 217.00× |


## Thread scaling

**`coco_val2017_jittered_seed0` · `bbox`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 135.5 ms | 93.8 ms | 61.1 ms | 56.6 ms |
| hotcoco | 211.1 ms (1.56×) | 128.2 ms (1.37×) | 83.8 ms (1.37×) | 74.5 ms (1.32×) |
| ufcoco | 155.3 ms (1.15×) | 127.2 ms (1.36×) | 111.7 ms (1.83×) | 109.7 ms (1.94×) |
| faster-coco-eval | 1.648 s (12.16×) | 1.518 s (16.19×) | 1.451 s (23.77×) | 1.468 s (25.93×) |

**`coco_val2017_jittered_seed0` · `bbox`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 94 MiB | 114 MiB | 120 MiB | 129 MiB |
| hotcoco | 83 MiB | 85 MiB | 86 MiB | 90 MiB |
| ufcoco | 69 MiB | 70 MiB | 71 MiB | 73 MiB |
| faster-coco-eval | 642 MiB | 642 MiB | 642 MiB | 641 MiB |

**`coco_val2017_jittered_seed0` · `boundary`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 2.670 s | 1.363 s | 718.6 ms | 602.8 ms |
| ufcoco | 214.110 s (80.20×) | 110.351 s (80.95×) | 54.468 s (75.80×) | 50.451 s (83.69×) |
| faster-coco-eval | 53.466 s (20.03×) | 30.229 s (22.18×) | 18.817 s (26.19×) | 16.916 s (28.06×) |

**`coco_val2017_jittered_seed0` · `boundary`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 113 MiB | 119 MiB | 140 MiB | 163 MiB |
| ufcoco | 239 MiB | 241 MiB | 244 MiB | 248 MiB |
| faster-coco-eval | 764 MiB | 771 MiB | 771 MiB | 771 MiB |

**`coco_val2017_jittered_seed0` · `segm`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 364.1 ms | 208.1 ms | 121.4 ms | 101.3 ms |
| hotcoco | 833.4 ms (2.29×) | 446.2 ms (2.14×) | 248.7 ms (2.05×) | 202.9 ms (2.00×) |
| ufcoco | 729.9 ms (2.00×) | 424.5 ms (2.04×) | 264.5 ms (2.18×) | 224.1 ms (2.21×) |
| faster-coco-eval | 3.483 s (9.57×) | 3.539 s (17.01×) | 3.498 s (28.80×) | 3.495 s (34.49×) |

**`coco_val2017_jittered_seed0` · `segm`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 114 MiB | 120 MiB | 136 MiB | 155 MiB |
| hotcoco | 155 MiB | 156 MiB | 158 MiB | 161 MiB |
| ufcoco | 105 MiB | 107 MiB | 108 MiB | 110 MiB |
| faster-coco-eval | 694 MiB | 694 MiB | 694 MiB | 694 MiB |

**`coco_val2017_keypoints_jittered_seed0` · `keypoints`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 98.0 ms | 62.4 ms | 38.4 ms | 32.3 ms |
| hotcoco | 123.5 ms (1.26×) | 77.7 ms (1.25×) | 53.1 ms (1.38×) | 43.4 ms (1.34×) |
| ufcoco | 117.5 ms (1.20×) | 87.8 ms (1.41×) | 73.3 ms (1.91×) | 69.5 ms (2.15×) |
| faster-coco-eval | 806.1 ms (8.22×) | 796.2 ms (12.77×) | 787.1 ms (20.48×) | 797.7 ms (24.66×) |

**`coco_val2017_keypoints_jittered_seed0` · `keypoints`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 50 MiB | 58 MiB | 67 MiB | 84 MiB |
| hotcoco | 38 MiB | 39 MiB | 39 MiB | 39 MiB |
| ufcoco | 36 MiB | 36 MiB | 36 MiB | 36 MiB |
| faster-coco-eval | 115 MiB | 115 MiB | 115 MiB | 115 MiB |

**`objects365_val_jittered_seed0` · `bbox`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=8` |
| --- | ---: |
| **vernier** | 961.6 ms |
| hotcoco | 1.398 s (1.45×) |
| ufcoco | 1.908 s (1.98×) |

**`objects365_val_jittered_seed0` · `bbox`** — eval Δ RSS

| impl | `nt=8` |
| --- | ---: |
| **vernier** | 995 MiB |
| hotcoco | 1.22 GiB |
| ufcoco | 866 MiB |


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
each library's own thread knob (vernier `num_threads`, hotcoco and
ufcoco `RAYON_NUM_THREADS`, faster-coco-eval `rle_iou_max_workers` /
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
