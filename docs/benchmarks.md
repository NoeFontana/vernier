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

**Provenance** — git SHA `885d385d63e1` · machine fingerprint `37652a58e939` · CPU AMD EPYC-Milan Processor (x86_64) · harness
mode `release` · build profile = cargo release defaults
(`opt-level=3`, `lto=thin`, `codegen-units=1`, no `target-cpu`). The
release wheel on PyPI is built with the same profile — no
benchmarking-only flags.

**Baselines pinned for these numbers** — [`faster-coco-eval==1.7.2`](https://pypi.org/project/faster-coco-eval/1.7.2/) · [`pycocotools==2.0.11`](https://pypi.org/project/pycocotools/2.0.11/) · [`boundary-iou-api` @ `37d2558`](https://github.com/bowenc0221/boundary-iou-api/commit/37d25586a677) · [`panopticapi` @ `7bb4655`](https://github.com/cocodataset/panopticapi/commit/7bb4655548f9) · [`mmsegmentation` @ `c685fe6`](https://github.com/open-mmlab/mmsegmentation/commit/c685fe6767c4cadf6b051983ca6208f1b9d1ccb8) · [`lvis-api` @ `031ac21`](https://github.com/lvis-dataset/lvis-api/commit/031ac21f939bcb5f1ca8de2ab8704082e101ff9b). Each baseline is locked in its own uv-managed venv per ADR-0017.

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
| **vernier** | 370.6 ms | 4.1 ms (1.10%) | 261 MiB | **1.00×** |
| faster-coco-eval | 2.060 s | 29.2 ms (1.42%) | 661 MiB | 5.56× |
| pycocotools | 5.753 s | 195.1 ms (3.39%) | 576 MiB | 15.52× |

**`segm`**

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 970.6 ms | 3.9 ms (0.40%) | 262 MiB | **1.00×** |
| faster-coco-eval | 3.498 s | 13.0 ms (0.37%) | 721 MiB | 3.60× |
| pycocotools | 6.635 s | 76.7 ms (1.16%) | 569 MiB | 6.84× |

**`boundary`**

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 3.143 s | 17.0 ms (0.54%) | 264 MiB | **1.00×** |
| faster-coco-eval | 17.616 s | 41.2 ms (0.23%) | 794 MiB | 5.61× |
| boundary-iou-api | 61.544 s | 225.2 ms (0.37%) | 666 MiB | 19.58× |

### Workload: `coco_val2017_keypoints_jittered_seed0`

**`keypoints`**

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 137.1 ms | 2.3 ms (1.69%) | 127 MiB | **1.00×** |
| faster-coco-eval | 1.661 s | 25.9 ms (1.56%) | 154 MiB | 12.11× |
| pycocotools | 2.261 s | 20.2 ms (0.89%) | 163 MiB | 16.49× |


## Panoptic — PQ

### Workload: `coco_panoptic_val2017_perfect`

**`pq`**

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 12.592 s | 2.673 s (21.22%) * | 143 MiB | **1.00×** |
| panopticapi | 34.440 s | 258.9 ms (0.75%) | 146 MiB | 2.73× |


## Semantic — mIoU

### Workload: `coco_val2017_semantic_perfect`

**`miou`**

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 5.004 s | 39.0 ms (0.78%) | 99 MiB | **1.00×** |
| mmsegmentation | 20.605 s | 172.5 ms (0.84%) | 647 MiB | 4.12× |

### Workload: `synthetic_semantic_n200_c19_s0`

**`miou`**

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 63.2 ms | 678.2 μs (1.07%) | 88 MiB | **1.00×** |
| mmsegmentation | 430.7 ms | 53.0 ms (12.32%) * | 631 MiB | 6.82× |


## Instance — LVIS federated AP

### Workload: `lvis_v1_val_perfect`

**`bbox`**

| impl | median | IQR | RSS (max) | vs vernier |
| --- | ---: | ---: | ---: | ---: |
| **vernier** | 3.727 s | 67.5 ms (1.81%) | 1.48 GiB | **1.00×** |
| lvis-api | 210.688 s | 6.600 s (3.13%) | 15.01 GiB | 56.53× |


*Cells marked ` *` next to their IQR exceeded the release-mode 5% relative-IQR gate. Median still reported; treat the gap to the next impl as the load-bearing signal rather than the precise ratio.*

## Threading scaling

[ADR-0047](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0047-threading-model.md)
adds an opt-in `num_threads` kwarg on every public evaluate surface
(batch, streaming, background) for all four paradigms — instance
(bbox / segm / boundary / keypoints), semantic, panoptic, and lvis.
The default `num_threads=None` is byte-for-byte the sequential
pre-0.0.5 path; no rayon symbol is entered.

Wall-clock numbers below are the `evaluate`-stage stage timer on COCO
val2017 perfect-DT (the jittered-seed-0 workload for the instance
paradigm; `coco_panoptic_val2017_perfect` for panoptic;
`coco_val2017_semantic_perfect` for semantic). Hardware: AMD EPYC
Milan, 4 physical cores + SMT-2 (8 vCPUs total). Dev-mode single rep
per cell; expect ±10–20% noise on small workloads.

### Instance (val2017 jittered-seed-0)

| iou_type | `nt=1` | `nt=2` | `nt=4` | `nt=8` | scaling at `nt=4` |
| --- | ---: | ---: | ---: | ---: | ---: |
| bbox | 280 ms | 296 ms | 285 ms | 291 ms | ~flat (parse-bound) |
| segm | 890 ms | 570 ms | 389 ms | 346 ms | **2.29×** |
| boundary | 3146 ms | 1700 ms | 969 ms | 864 ms | **3.25×** |

Bbox is parse-bound (the GT JSON parse takes ~110 ms of the 280 ms
total before any per-cell work runs); the parallel region itself is
too small to amortise per-call rayon overhead. Segm and boundary
scale near-linearly on 4 physical cores after the
`ThreadLocal<RefCell<Scratch>>` + image-major dispatch fixes.

### Semantic (val2017 perfect-DT, ~5000 images)

| `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| ---: | ---: | ---: | ---: |
| 5024 ms | 2699 ms | **1372 ms** | 936 ms |

3.66× at `nt=4` on 4 physical cores, 5.37× at `nt=8` with SMT. The
per-image confusion-matrix fold is u64-additive so strict-mode
bit-equality is unconditional regardless of reduction order.

### Panoptic (val2017 perfect-DT, `BackgroundPanopticEvaluator`)

| `nt=1` | `nt=2` | `nt=4` | `nt=8` |
| ---: | ---: | ---: | ---: |
| 12850 ms | 8156 ms | **5280 ms** | 4479 ms |

2.43× at `nt=4`. The threaded path runs PNG decode inside the
per-worker rayon pool (zero-copy via `PyBackedBytes`), so libpng
parallelises across submissions; the single-threaded path keeps
producer/consumer overlap against the worker thread and is
byte-for-byte unchanged from pre-0.0.5.

### vs faster-coco-eval (boundary at `nt={1,4,8}`)

faster-coco-eval exposes parallelism only on boundary IoU
(`boundary_cpu_count`); bbox / segm / keypoints stay single-threaded.

| impl | `nt=1` | `nt=4` | `nt=8` |
| --- | ---: | ---: | ---: |
| **vernier** | 3146 ms | **969 ms** | 864 ms |
| faster-coco-eval | 49500 ms | 17206 ms | 15654 ms |
| vernier advantage | 15.8× | 17.2× | 18.1× |

### Reproduce

```bash
# Instance paradigm sweep across thread counts on val2017
just bench-run --impl vernier --workload coco_val2017_jittered_seed0 \
    --iou boundary --num-threads "1,2,4,8" --no-parity

# Plumbing smoke (small synthetic fixture, no external data)
just bench-threads-smoke
```

Result JSONs land under
`bench/results/<sha>/<fp>/<paradigm>/<workload>_t<N>/<metric>/<impl>.json`.

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
