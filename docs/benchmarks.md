# Benchmarks

Comparison of vernier against the third-party libraries it targets parity
against, on a single machine and a single git revision. The numbers below
are the median total-stage wall time over the non-warmup reps recorded by
the local bench harness ([ADR-0017](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0017-local-bench-harness.md),
extended cross-paradigm in
[ADR-0033](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0033-multi-paradigm-bench.md)).

**Provenance** — git SHA `f58f1075985f` · machine fingerprint `5658de0e29a3` · harness
mode `dev` · build profile = cargo release defaults
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

| impl | median | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: |
| **vernier** | 361.6 ms | 235 MiB | **1.00×** |
| faster-coco-eval | 2.123 s | 661 MiB | 5.87× |
| pycocotools | 5.811 s | 576 MiB | 16.07× |

**`segm`**

| impl | median | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: |
| **vernier** | 960.9 ms | 236 MiB | **1.00×** |
| faster-coco-eval | 3.496 s | 721 MiB | 3.64× |
| pycocotools | 6.727 s | 569 MiB | 7.00× |

**`boundary`**

| impl | median | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: |
| **vernier** | 3.136 s | 238 MiB | **1.00×** |
| faster-coco-eval | 17.670 s | 794 MiB | 5.63× |
| boundary-iou-api | 61.838 s | 666 MiB | 19.72× |

### Workload: `coco_val2017_keypoints_jittered_seed0`

**`keypoints`**

| impl | median | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: |
| **vernier** | 130.2 ms | 101 MiB | **1.00×** |
| faster-coco-eval | 1.654 s | 154 MiB | 12.70× |
| pycocotools | 2.288 s | 163 MiB | 17.57× |


## Panoptic — PQ

### Workload: `coco_panoptic_val2017_perfect`

**`pq`**

| impl | median | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: |
| **vernier** | 32.004 s | 127 MiB | **1.00×** |
| panopticapi | 34.397 s | 145 MiB | 1.07× |


## Semantic — mIoU

### Workload: `synthetic_semantic_n200_c19_s0`

**`miou`**

| impl | median | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: |
| **vernier** | 193.0 ms | 88 MiB | **1.00×** |


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
