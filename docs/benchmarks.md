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

**Provenance** — git SHA `1fd5720bf56c` · machine fingerprint `1655eb18a194` · CPU AMD EPYC-Milan Processor (x86_64) · harness
mode `release` · build profile = cargo release defaults
(`opt-level=3`, `lto=thin`, `codegen-units=1`, no `target-cpu`). The
release wheel on PyPI is built with the same profile — no
benchmarking-only flags.

**Baselines pinned for these numbers** — [`faster-coco-eval==1.7.2`](https://pypi.org/project/faster-coco-eval/1.7.2/) · [`pycocotools==2.0.11`](https://pypi.org/project/pycocotools/2.0.11/) · [`boundary-iou-api` @ `37d2558`](https://github.com/bowenc0221/boundary-iou-api/commit/37d25586a677) · [`panopticapi` @ `7bb4655`](https://github.com/cocodataset/panopticapi/commit/7bb4655548f9) · [`mmsegmentation` @ `c685fe6`](https://github.com/open-mmlab/mmsegmentation/commit/c685fe6767c4cadf6b051983ca6208f1b9d1ccb8) · [`lvis-api` @ `031ac21`](https://github.com/lvis-dataset/lvis-api/commit/031ac21f939b) (PyPI `lvis==0.5.3`). Each baseline is locked in its own uv-managed venv per ADR-0017.

The LVIS section below was measured at HEAD `e9d9c4d71303` after the
bench paradigm landed; every other section is at `1fd5720bf56c`. The
next full bench refresh will collapse the LVIS section into the same
SHA as the others.

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

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 360.0 ms | 3.8 ms (1.07%) | 236 MiB | **1.00×** |
| faster-coco-eval | 2.127 s | 21.8 ms (1.03%) | 661 MiB | 5.91× |
| pycocotools | 5.820 s | 65.8 ms (1.13%) | 576 MiB | 16.17× |

**`segm`**

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 967.7 ms | 13.4 ms (1.38%) | 236 MiB | **1.00×** |
| faster-coco-eval | 3.605 s | 63.7 ms (1.77%) | 721 MiB | 3.73× |
| pycocotools | 6.853 s | 71.8 ms (1.05%) | 569 MiB | 7.08× |

**`boundary`**

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 3.130 s | 21.7 ms (0.69%) | 238 MiB | **1.00×** |
| faster-coco-eval | 17.837 s | 48.8 ms (0.27%) | 794 MiB | 5.70× |
| boundary-iou-api | 62.233 s | 228.1 ms (0.37%) | 666 MiB | 19.88× |

### Workload: `coco_val2017_keypoints_jittered_seed0`

**`keypoints`**

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 135.7 ms | 2.4 ms (1.76%) | 102 MiB | **1.00×** |
| faster-coco-eval | 1.700 s | 20.1 ms (1.18%) | 154 MiB | 12.53× |
| pycocotools | 2.317 s | 13.3 ms (0.57%) | 163 MiB | 17.07× |


## Panoptic — PQ

### Workload: `coco_panoptic_val2017_perfect`

**`pq`**

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 11.615 s | 605.4 ms (5.21%) * | 118 MiB | **1.00×** |
| panopticapi | 35.327 s | 344.5 ms (0.98%) | 145 MiB | 3.04× |


## Semantic — mIoU

### Workload: `coco_val2017_semantic_perfect`

**`miou`**

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 5.070 s | 25.5 ms (0.50%) | 92 MiB | **1.00×** |
| mmsegmentation | 21.377 s | 237.4 ms (1.11%) | 648 MiB | 4.22× |

### Workload: `synthetic_semantic_n200_c19_s0`

**`miou`**

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 63.1 ms | 618.8 μs (0.98%) | 88 MiB | **1.00×** |
| mmsegmentation | 437.5 ms | 46.5 ms (10.64%) * | 631 MiB | 6.93× |


*Cells marked ` *` next to their IQR exceeded the release-mode 5% relative-IQR gate. Median still reported; treat the gap to the next impl as the load-bearing signal rather than the precise ratio.*

## Instance — LVIS federated AP

### Workload: `lvis_v1_val_perfect`

LVIS v1 val (19809 images, 1203 categories), GT-as-DT (perfect bbox-shape DT). Federated semantics layer over the shared AP-fold core per ADR-0026.

**`bbox`**

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 3.691 s | 46.7 ms (1.26%) | 1.49 GiB | **1.00×** |
| lvis-api | 210.086 s | 9.72 s (4.63%) | 15.01 GiB | 56.92× |

vernier reports AP=0.9983; lvis-api reports the same headline. Strict
bit-equality on the `(T, R, K, A)` precision tensor passes on every
category except K=168 and K=817 (2730 cells out of 4.86M, ~0.06%);
those two K-rows differ at recall thresholds where lvis-api drops a
fraction of a recall point on multi-GT-per-image cells under
score-tie ordering. Tracked in [ADR-0026 §"Known follow-up"](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0026-lvis-support.md).
The headline 56.9× speedup and 10× lower peak RSS are not affected
by the open divergence — the timing + memory measurements stand on
their own.

The 22 GB peak ADR-0026 called out at acceptance was the structural
upper bound on the dense `Vec<Option<PerImageEval>>` orchestrator grid
(95M slots × 232 B). PR #179 collapsed the slot type via the
`Box`-niche trick; the measured peak above is 1.49 GiB.

## Methodology in one paragraph

Every cell runs in its own subprocess with its own uv-managed venv (one
per impl), so a single Python process never has competing
pycocotools-flavored packages on its `sys.path`. The harness records
`(load, evaluate, accumulate, summarize, total)` wall_ns per stage,
discards the warmup reps, and reports the median total plus the
inter-quartile range (IQR = Q3 - Q1, with the relative spread shown as
a percentage of the median). Release mode (N=10 + 2 warmup) gates each
impl on relative IQR ≤ 5%; cells where the gate failed are marked with
` *` next to their IQR value — the median is still the best estimator,
just with a wider confidence band than the gate accepts. Parity is a
side effect of every timing run — strict-tier (vs pycocotools) and
aligned-tier (vs faster-coco-eval) where applicable; failed parity
fails the cell. Memory is `getrusage(RUSAGE_CHILDREN).ru_maxrss`,
high-water-marked across the rep set.
