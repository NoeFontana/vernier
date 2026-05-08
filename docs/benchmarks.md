# Benchmarks

Comparison of vernier against the third-party libraries it targets parity
against, on a single machine and a single git revision. The numbers below
are the median total-stage wall time over the non-warmup reps recorded by
the local bench harness ([ADR-0017](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0017-local-bench-harness.md),
extended cross-paradigm in
[ADR-0033](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0033-multi-paradigm-bench.md)).

**Provenance** — git SHA `a81a86db789b` · machine fingerprint `5658de0e29a3` · harness
mode `dev` · build profile = cargo release defaults
(`opt-level=3`, `lto=thin`, `codegen-units=1`, no `target-cpu`). The
release wheel on PyPI is built with the same profile — no
benchmarking-only flags.

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

| impl | median | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: |
| **vernier** | 652.4 ms | 784 MiB | **1.00×** |
| faster-coco-eval | 2.096 s | 661 MiB | 3.21× |
| pycocotools | 5.989 s | 576 MiB | 9.18× |

**`segm`**

| impl | median | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: |
| **vernier** | 1.283 s | 785 MiB | **1.00×** |
| faster-coco-eval | 3.532 s | 721 MiB | 2.75× |
| pycocotools | 6.814 s | 569 MiB | 5.31× |

**`boundary`**

| impl | median | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: |
| **vernier** | 39.918 s | 787 MiB | **1.00×** |
| boundary-iou-api | 61.982 s | 666 MiB | 1.55× |

### Workload: `coco_val2017_keypoints_jittered_seed0`

**`keypoints`**

| impl | median | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: |
| **vernier** | 144.4 ms | 113 MiB | **1.00×** |
| faster-coco-eval | 1.703 s | 154 MiB | 11.79× |
| pycocotools | 2.288 s | 163 MiB | 15.84× |

### Workload: `synthetic_n10000_c80_g10_d30_s0`

**`bbox`**

| impl | median | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: |
| **vernier** | 2.433 s | 1.67 GiB | **1.00×** |
| faster-coco-eval | 5.965 s | 1.36 GiB | 2.45× |
| pycocotools | 34.808 s | 2.63 GiB | 14.30× |

### Workload: `synthetic_n1000_c80_g10_d30_s0`

**`bbox`**

| impl | median | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: |
| **vernier** | 214.5 ms | 233 MiB | **1.00×** |
| faster-coco-eval | 538.1 ms | 210 MiB | 2.51× |
| pycocotools | 2.849 s | 330 MiB | 13.28× |

### Workload: `synthetic_n50000_c80_g10_d30_s0`

**`bbox`**

| impl | median | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: |
| **vernier** | 12.690 s | 8.17 GiB | **1.00×** |
| faster-coco-eval | 35.342 s | 6.45 GiB | 2.78× |
| pycocotools | 194.651 s | 12.78 GiB | 15.34× |


## Panoptic — PQ

### Workload: `coco_panoptic_val2017_perfect`

**`pq`**

| impl | median | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: |
| **vernier** | 86.117 s | 21.18 GiB | **1.00×** |
| panopticapi | 34.762 s | 146 MiB | 0.40× |


## Methodology in one paragraph

Every cell runs in its own subprocess with its own uv-managed venv (one
per impl), so a single Python process never has competing
pycocotools-flavored packages on its `sys.path`. The harness records
`(load, evaluate, accumulate, summarize, total)` wall_ns per stage,
discards the warmup reps, and reports the median total. Parity is a
side effect of every timing run — strict-tier (vs pycocotools) and
aligned-tier (vs faster-coco-eval) where applicable; failed parity
fails the cell. Memory is `getrusage(RUSAGE_CHILDREN).ru_maxrss`,
high-water-marked across the rep set.
