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

**Provenance** — git SHA `0a39957821bf` · machine fingerprint `1655eb18a194` · CPU AMD EPYC-Milan Processor (x86_64) · harness
mode `release` · build profile = cargo release defaults
(`opt-level=3`, `lto=thin`, `codegen-units=1`, no `target-cpu`). The
release wheel on PyPI is built with the same profile — no
benchmarking-only flags.

**Baselines pinned for these numbers** — [`faster-coco-eval==1.7.2`](https://pypi.org/project/faster-coco-eval/1.7.2/) · [`pycocotools==2.0.11`](https://pypi.org/project/pycocotools/2.0.11/) · [`boundary-iou-api` @ `37d2558`](https://github.com/bowenc0221/boundary-iou-api/commit/37d25586a677) · [`panopticapi` @ `7bb4655`](https://github.com/cocodataset/panopticapi/commit/7bb4655548f9). Each baseline is locked in its own uv-managed venv per ADR-0017.

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
| **vernier** | 359.6 ms | 7.2 ms (2.01%) | 235 MiB | **1.00×** |
| faster-coco-eval | 2.121 s | 30.2 ms (1.42%) | 661 MiB | 5.90× |
| pycocotools | 5.833 s | 129.9 ms (2.23%) | 576 MiB | 16.22× |

**`segm`**

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 968.1 ms | 6.0 ms (0.62%) | 236 MiB | **1.00×** |
| faster-coco-eval | 3.553 s | 53.4 ms (1.50%) | 721 MiB | 3.67× |
| pycocotools | 6.690 s | 163.7 ms (2.45%) | 569 MiB | 6.91× |

**`boundary`**

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 3.121 s | 15.5 ms (0.50%) | 238 MiB | **1.00×** |
| faster-coco-eval | 17.772 s | 83.2 ms (0.47%) | 794 MiB | 5.70× |
| boundary-iou-api | 62.161 s | 526.6 ms (0.85%) | 666 MiB | 19.92× |

### Workload: `coco_val2017_keypoints_jittered_seed0`

**`keypoints`**

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 134.7 ms | 1.5 ms (1.08%) | 102 MiB | **1.00×** |
| faster-coco-eval | 1.707 s | 18.5 ms (1.09%) | 154 MiB | 12.67× |
| pycocotools | 2.308 s | 26.7 ms (1.16%) | 163 MiB | 17.13× |


## Panoptic — PQ

### Workload: `coco_panoptic_val2017_perfect`

**`pq`**

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 13.012 s | 2.588 s (19.89%) * | 117 MiB | **1.00×** |
| panopticapi | 34.640 s | 336.1 ms (0.97%) | 146 MiB | 2.66× |


## Semantic — mIoU

### Workload: `synthetic_semantic_n200_c19_s0`

**`miou`**

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 62.3 ms | 506.3 μs (0.81%) | 88 MiB | **1.00×** |


*Cells marked ` *` next to their IQR exceeded the release-mode 5% relative-IQR gate. Median still reported; treat the gap to the next impl as the load-bearing signal rather than the precise ratio.*

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
