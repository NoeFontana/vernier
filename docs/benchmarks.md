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

**Provenance** — git SHA `a79f67e6ba7e` · machine fingerprint `a6ada888c850` · CPU AMD EPYC-Milan Processor (x86_64) · harness
mode `release` · build profile = cargo release defaults
(`opt-level=3`, `lto=thin`, `codegen-units=1`, no `target-cpu`). The
release wheel on PyPI is built with the same profile — no
benchmarking-only flags.

**Baselines pinned for these numbers** — [`hotcoco==1.0.1`](https://pypi.org/project/hotcoco/1.0.1/) · [`ultrafast-pycocotools==0.1.14`](https://pypi.org/project/ultrafast-pycocotools/0.1.14/) · [`faster-coco-eval==1.8.0`](https://pypi.org/project/faster-coco-eval/1.8.0/) · [`pycocotools==2.0.11`](https://pypi.org/project/pycocotools/2.0.11/) · [`boundary-iou-api` @ `37d2558`](https://github.com/bowenc0221/boundary-iou-api/commit/37d25586a677) · [`panopticapi` @ `7bb4655`](https://github.com/cocodataset/panopticapi/commit/7bb4655548f9) · [`mmsegmentation` @ `c685fe6`](https://github.com/open-mmlab/mmsegmentation/commit/c685fe6767c4cadf6b051983ca6208f1b9d1ccb8) · [`lvis-api` @ `031ac21`](https://github.com/lvis-dataset/lvis-api/commit/031ac21f939bcb5f1ca8de2ab8704082e101ff9b). Each baseline is locked in its own uv-managed venv per ADR-0017.

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
| **vernier** | 164.9 ms | 2.6 ms (1.56%) | 1.00 | 164 MiB | 88 MiB | **1.00×** |
| hotcoco | 580.4 ms | 9.3 ms (1.60%) | 1.00 | 274 MiB | 226 MiB | 3.52× |
| ufcoco | 156.7 ms | 1.6 ms (1.02%) | 1.00 | 118 MiB | 69 MiB | 0.95× |
| faster-coco-eval | 1.735 s | 46.0 ms (2.65%) | 1.00 | 692 MiB | 642 MiB | 10.52× |
| pycocotools | 6.075 s | 180.4 ms (2.97%) | 1.00 | 576 MiB | 528 MiB | 36.84× |

**`segm`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 512.6 ms | 3.0 ms (0.59%) | 1.00 | 174 MiB | 98 MiB | **1.00×** |
| hotcoco | 1.399 s | 16.8 ms (1.20%) | 1.00 | 361 MiB | 313 MiB | 2.73× |
| ufcoco | 742.3 ms | 5.3 ms (0.71%) | 1.00 | 154 MiB | 105 MiB | 1.45× |
| faster-coco-eval | 3.580 s | 58.8 ms (1.64%) | 1.00 | 744 MiB | 694 MiB | 6.98× |
| pycocotools | 7.190 s | 602.4 ms (8.38%) * | 1.00 | 569 MiB | 521 MiB | 14.03× |

**`boundary`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 2.762 s | 11.1 ms (0.40%) | 1.00 | 175 MiB | 99 MiB | **1.00×** |
| ufcoco | 214.261 s | 1.278 s (0.60%) | 1.00 | 288 MiB | 239 MiB | 77.58× |
| faster-coco-eval | 53.419 s | 120.4 ms (0.23%) | 1.00 | 813 MiB | 764 MiB | 19.34× |
| boundary-iou-api | 59.660 s | 445.1 ms (0.75%) | 1.00 | 669 MiB | 597 MiB | 21.60× |

### Workload: `coco_val2017_keypoints_jittered_seed0`

**`keypoints`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 114.6 ms | 1.5 ms (1.31%) | 1.00 | 121 MiB | 44 MiB | **1.00×** |
| hotcoco | 215.4 ms | 4.2 ms (1.97%) | 1.00 | 123 MiB | 75 MiB | 1.88× |
| ufcoco | 118.5 ms | 669.3 μs (0.56%) | 1.00 | 85 MiB | 36 MiB | 1.03× |
| faster-coco-eval | 808.8 ms | 10.8 ms (1.34%) | 1.00 | 165 MiB | 115 MiB | 7.06× |
| pycocotools | 2.341 s | 27.3 ms (1.17%) | 1.00 | 163 MiB | 115 MiB | 20.43× |

### Workload: `coco_val2017_rfdetr_nano_v1.6.5.post0`

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 492.5 ms | 5.3 ms (1.07%) | 1.00 | 257 MiB | 180 MiB | **1.00×** |
| hotcoco | 2.333 s | 17.1 ms (0.73%) | 1.00 | 997 MiB | 949 MiB | 4.74× |
| ufcoco | 538.8 ms | 4.0 ms (0.74%) | 1.00 | 225 MiB | 176 MiB | 1.09× |
| faster-coco-eval | 5.539 s | 93.5 ms (1.69%) | 1.00 | 1.35 GiB | 1.30 GiB | 11.25× |
| pycocotools | 23.361 s | 302.9 ms (1.30%) | 1.00 | 1.40 GiB | 1.35 GiB | 47.44× |

### Workload: `coco_val2017_rfdetr_segnano_v1.6.5.post0`

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 576.3 ms | 9.0 ms (1.55%) | 1.00 | 359 MiB | 283 MiB | **1.00×** |
| hotcoco | 2.593 s | 41.3 ms (1.59%) | 1.00 | 961 MiB | 913 MiB | 4.50× |
| ufcoco | 635.3 ms | 9.7 ms (1.52%) | 1.00 | 329 MiB | 280 MiB | 1.10× |
| faster-coco-eval | 5.457 s | 79.6 ms (1.46%) | 1.00 | 1.45 GiB | 1.40 GiB | 9.47× |
| pycocotools | 21.456 s | 471.7 ms (2.20%) | 1.00 | 1.47 GiB | 1.42 GiB | 37.23× |

**`segm`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 1.431 s | 17.3 ms (1.21%) | 1.00 | 481 MiB | 404 MiB | **1.00×** |
| hotcoco | 4.914 s | 73.7 ms (1.50%) | 1.00 | 1.44 GiB | 1.40 GiB | 3.43× |
| ufcoco | 1.785 s | 22.6 ms (1.27%) | 1.00 | 380 MiB | 331 MiB | 1.25× |
| faster-coco-eval | 10.140 s | 228.5 ms (2.25%) | 1.00 | 1.71 GiB | 1.66 GiB | 7.09× |
| pycocotools | 23.009 s | 461.6 ms (2.01%) | 1.00 | 1.45 GiB | 1.40 GiB | 16.08× |

### Workload: `objects365_val_jittered_seed0`

*Scale workload: Objects365 v2 val, 80,000 images · 1,240,587 GT boxes · 365 categories, with ~1.06 M jittered detections (bbox only). Annotations © [Objects365 Consortium](https://www.objects365.org/), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); images are never downloaded. faster-coco-eval is absent because it does not finish: it is OOM-killed at ~30 GiB anon RSS on this 30 GiB host, at 1 and at 8 threads.*

*Recorded in harness mode `dev` (not `release`): one measurement rep per impl, no IQR gate.*

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 2.940 s | 0 ns | 1.00 | 885 MiB | 809 MiB | **1.00×** |
| hotcoco | 16.234 s | 0 ns | 1.00 | 6.49 GiB | 6.45 GiB | 5.52× |
| ufcoco | 3.549 s | 0 ns | 1.00 | 856 MiB | 807 MiB | 1.21× |
| pycocotools | 389.849 s | 0 ns | 1.00 | 21.34 GiB | 21.30 GiB | 132.62× |


## Panoptic — PQ

### Workload: `coco_panoptic_val2017_perfect`

**`pq`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 10.575 s | 52.2 ms (0.49%) | 1.00 | 145 MiB | 67 MiB | **1.00×** |
| panopticapi | 35.533 s | 316.7 ms (0.89%) | 1.00 | 148 MiB | 97 MiB | 3.36× |


## Semantic — mIoU

### Workload: `coco_val2017_semantic_perfect`

**`miou`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 2.887 s | 6.9 ms (0.24%) | 1.00 | 98 MiB | 21 MiB | **1.00×** |
| mmsegmentation | 41.388 s | 272.8 ms (0.66%) | 1.00 | 546 MiB | 41 MiB | 14.33× |


## Instance — LVIS federated AP

### Workload: `lvis_v1_val_jittered_seed0`

**`bbox`**

| impl | median | IQR | CPU/wall | peak RSS | eval Δ RSS | vs vernier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **vernier** | 1.283 s | 19.6 ms (1.53%) | 1.00 | 771 MiB | 694 MiB | **1.00×** |
| hotcoco | 4.119 s | 124.0 ms (3.01%) | 1.00 | 1.47 GiB | 1.43 GiB | 3.21× |
| lvis-api | 220.012 s | 20.733 s (9.42%) * | 1.00 | 15.08 GiB | 14.98 GiB | 171.54× |


## Thread scaling

**`coco_val2017_jittered_seed0` · `bbox`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 164.4 ms | 116.5 ms | 78.8 ms | 78.3 ms |
| hotcoco | 589.0 ms (3.58×) | 440.7 ms (3.78×) | 372.5 ms (4.72×) | 366.2 ms (4.68×) |
| ufcoco | 154.9 ms (0.94×) | 124.4 ms (1.07×) | 110.9 ms (1.41×) | 109.6 ms (1.40×) |
| faster-coco-eval | 1.747 s (10.62×) | 1.606 s (13.79×) | 1.516 s (19.23×) | 1.507 s (19.23×) |

**`coco_val2017_jittered_seed0` · `bbox`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 88 MiB | 91 MiB | 94 MiB | 94 MiB |
| hotcoco | 226 MiB | 227 MiB | 228 MiB | 229 MiB |
| ufcoco | 69 MiB | 70 MiB | 71 MiB | 73 MiB |
| faster-coco-eval | 642 MiB | 642 MiB | 642 MiB | 642 MiB |

**`coco_val2017_jittered_seed0` · `boundary`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 2.764 s | 1.412 s | 757.7 ms | 633.8 ms |
| ufcoco | 227.008 s (82.13×) | 109.759 s (77.75×) | 54.443 s (71.85×) | 50.470 s (79.63×) |
| faster-coco-eval | 53.595 s (19.39×) | 30.475 s (21.59×) | 19.048 s (25.14×) | 17.055 s (26.91×) |

**`coco_val2017_jittered_seed0` · `boundary`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 99 MiB | 104 MiB | 107 MiB | 111 MiB |
| ufcoco | 239 MiB | 241 MiB | 244 MiB | 247 MiB |
| faster-coco-eval | 764 MiB | 771 MiB | 771 MiB | 764 MiB |

**`coco_val2017_jittered_seed0` · `segm`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 507.1 ms | 293.0 ms | 173.8 ms | 140.6 ms |
| hotcoco | 1.383 s (2.73×) | 849.9 ms (2.90×) | 594.9 ms (3.42×) | 549.9 ms (3.91×) |
| ufcoco | 737.5 ms (1.45×) | 431.4 ms (1.47×) | 265.6 ms (1.53×) | 225.1 ms (1.60×) |
| faster-coco-eval | 3.547 s (6.99×) | 3.610 s (12.32×) | 3.527 s (20.29×) | 3.521 s (25.05×) |

**`coco_val2017_jittered_seed0` · `segm`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 98 MiB | 102 MiB | 106 MiB | 107 MiB |
| hotcoco | 313 MiB | 314 MiB | 316 MiB | 318 MiB |
| ufcoco | 105 MiB | 107 MiB | 108 MiB | 110 MiB |
| faster-coco-eval | 694 MiB | 694 MiB | 694 MiB | 694 MiB |

**`coco_val2017_keypoints_jittered_seed0` · `keypoints`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 113.1 ms | 74.9 ms | 46.2 ms | 38.7 ms |
| hotcoco | 216.7 ms (1.92×) | 180.1 ms (2.41×) | 164.2 ms (3.55×) | 159.7 ms (4.12×) |
| ufcoco | 119.6 ms (1.06×) | 88.4 ms (1.18×) | 73.9 ms (1.60×) | 69.7 ms (1.80×) |
| faster-coco-eval | 797.3 ms (7.05×) | 800.5 ms (10.69×) | 784.5 ms (16.99×) | 783.7 ms (20.24×) |

**`coco_val2017_keypoints_jittered_seed0` · `keypoints`** — eval Δ RSS

| impl | `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 44 MiB | 46 MiB | 47 MiB | 47 MiB |
| hotcoco | 75 MiB | 75 MiB | 76 MiB | 76 MiB |
| ufcoco | 36 MiB | 36 MiB | 36 MiB | 36 MiB |
| faster-coco-eval | 115 MiB | 115 MiB | 115 MiB | 115 MiB |

**`objects365_val_jittered_seed0` · `bbox`** — median total; ratio vs vernier at the same `nt`

| impl | `nt=8` |
| --- | ---: |
| **vernier** | 1.233 s |
| hotcoco | 8.749 s (7.09×) |
| ufcoco | 1.999 s (1.62×) |

**`objects365_val_jittered_seed0` · `bbox`** — eval Δ RSS

| impl | `nt=8` |
| --- | ---: |
| **vernier** | 940 MiB |
| hotcoco | 6.47 GiB |
| ufcoco | 874 MiB |


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
