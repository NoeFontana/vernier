# 2026-05 vernier-bench results

First release-mode capture from `vernier-bench` (ADR-0017). Every cell
captured this month — bbox/segm/boundary on val2017, a synthetic
stress smoke at n=500, and the harness's per-IoU fan-out smoke on the
parity fixture — consolidated here.

## Shared configuration

- **Harness:** release mode, N=10 measurement reps with 2 warmup
  discarded, randomised impl order per rep, IQR-relative-to-median
  gate at 5%. Every cell came in well under the threshold.
- **Parity tiers** (ADR-0002): strict (`np.array_equal` vs
  pycocotools), aligned (`np.allclose` at 4×ULP vs faster-coco-eval),
  strict vs the vendored `bowenc0221/boundary-iou-api` oracle for
  boundary. **Every cell parity OK.**
- **Pinned baselines:** pycocotools `2.0.11` and faster-coco-eval as
  resolved by `bench/envs/*/uv.lock`; bumping is an ADR-level
  decision.
- **Machine:** AMD EPYC-Milan / 4 cores / 32 GiB / kernel
  `6.8.0-107-generic`, fingerprint `82013f18a44d`, git_sha
  `58f09cb9149b`. cpufreq unexposed on this VM; the IQR gate carries
  the noise budget per the ADR-0017 "quiet on machines without
  cpufreq" clause.

## Headline — all cells

`× vernier` is total wall median for the baseline divided by
vernier's; greater than 1.0 means vernier is faster.

| workload                                  | iou      | vernier (median) | pycocotools     | faster-coco-eval | boundary-iou-api |
| ----------------------------------------- | -------- | ---------------: | --------------: | ---------------: | ---------------: |
| `coco_val2017_jittered_seed42`            | bbox     |       644.156 ms |    9.12× slower |     3.20× slower |             —    |
| `synthetic n=500,c=80,g=10,d=30,seed=0`   | bbox     |       116.763 ms |   12.57× slower |     2.35× slower |             —    |
| `coco_val2017_perfect_segm`               | segm     |         1.742 s  |    4.66× slower |     2.61× slower |             —    |
| `coco_val2017_perfect_segm`               | boundary |        75.861 s  |             —   |              —   | **0.84× — vernier 1.19× slower** |
| `smoke_perfect_match_segm`                | segm     |         0.280 ms |   23.26× slower |    19.81× slower |             —    |
| `smoke_perfect_match_segm`                | boundary |         0.355 ms |             —   |              —   |    20.71× slower |

Takeaways:

- **bbox is the load-bearing headline.** vernier is 9.1× faster than
  pycocotools and 3.2× faster than faster-coco-eval on val2017.
- **segm wins are real.** 4.66× / 2.61× on val2017 perfect-match.
- **Synthetic widens the gap.** 12.6× / 2.35× at n=500 — pycocotools'
  quadratic-ish hot loops scale less harshly when per-image work is
  small but uniform.
- **Boundary on val2017 is the one cell where vernier loses to a
  baseline.** 1.19× slower than `boundary-iou-api`. Cause is
  algorithmic and known; tracked as a follow-up, not a release
  blocker. See §Boundary regression below.
- **Smoke fixture sub-millisecond ratios** are dominated by fixed
  Python/PyO3 conversion overhead at that scale; treat them as parity
  smoke, not a perf claim.

---

## bbox — `coco_val2017_jittered_seed42`

`instances_val2017.json` (sha256-pinned `e8c7f7…`) with deterministic
Gaussian-noise jittered DT (seed 42), generated on the fly per
`bench/bench/workloads/jittered_predictions.py`.

| impl              | total median |    IQR (rel) | RAM (max RSS) | vs vernier |
| ----------------- | -----------: | -----------: | ------------: | ---------: |
| **vernier**       |   644.156 ms |  4.4 ms (0.7%) |     750.7 MiB |       1.0× |
| faster-coco-eval  |     2.058 s  | 64.3 ms (3.1%) |     644.7 MiB |  3.20× slower |
| pycocotools       |     5.871 s  | 82.7 ms (1.4%) |     558.3 MiB |  9.12× slower |

Per-stage medians:

| stage      | vernier  | faster-coco-eval | pycocotools |
| ---------- | -------: | ---------------: | ----------: |
| load       |  15.6 ms |         425.5 ms |    422.1 ms |
| evaluate   | 549.9 ms |        1632.6 ms |   4707.4 ms |
| accumulate |  78.1 ms |       *fused\**  |    747.6 ms |
| summarize  |  0.62 ms |          2.20 ms |     1.33 ms |

\* faster-coco-eval folds accumulate into evaluate; the timer reads zero.

`evaluate` dominates — pairwise IoU + per-image matching across ~5000
images and 80 categories.

---

## bbox — `synthetic:n=500`

500 images × 80 categories × 10 GT and 30 DT per image, seed 0.
Workload ID derives from the parameter tuple; cached at
`~/.cache/vernier-bench/synthetic/synthetic_n500_c80_g10_d30_s0_{gt,dt}.json`.

| impl              | total median |    IQR (rel) | RAM (max RSS) | vs vernier |
| ----------------- | -----------: | -----------: | ------------: | ---------: |
| **vernier**       |   116.763 ms | 2.95 ms (2.5%) |     147.7 MiB |       1.0× |
| faster-coco-eval  |   274.720 ms | 4.01 ms (1.5%) |     143.1 MiB |  2.35× slower |
| pycocotools       |     1.468 s  | 25.31 ms (1.7%) |     201.3 MiB | 12.57× slower |

The vernier-vs-pycocotools gap is **9.1× on val2017** but **12.6× on
synthetic n=500**: synthetic has uniformly small images and regular
per-image GT/DT counts; val2017 has wide image-size and per-image
distributions, which pycocotools handles less harshly than its
worst-case loops would suggest.

---

## segm — `coco_val2017_perfect_segm`

`instances_val2017.json` paired with `perfect_dt_segm.json` (the
parity-test "GT-as-DT" predictions generated by
`tools/make-perfect-dt.py`; 5000 images, ~36k annotations).

| impl              | total median |    IQR (rel) | RAM (max RSS) | vs vernier |
| ----------------- | -----------: | -----------: | ------------: | ---------: |
| **vernier**       |     1.742 s  |  55.1 ms (3.2%) |     765.3 MiB |       1.0× |
| faster-coco-eval  |     4.543 s  |  92.7 ms (2.0%) |     795.4 MiB |  2.61× slower |
| pycocotools       |     8.114 s  | 148.9 ms (1.8%) |     616.8 MiB |  4.66× slower |

Per-stage medians (ms):

| stage      | vernier  | faster-coco-eval | pycocotools |
| ---------- | -------: | ---------------: | ----------: |
| load       |   23.1   |          550.3   |      529.8  |
| evaluate   | 1658.9   |         3977.4   |     6854.4  |
| accumulate |   59.8   |       *fused\**  |      725.6  |
| summarize  |    0.62  |            2.30  |        1.34 |

**Caveat — perfect-match DT under-stresses matching.** Every DT lines
up with a GT exactly (score=1.0). Real detector output has unmatched
predictions which add false-positive accumulation work. These numbers
are realistic for the eval/accumulate hot path on real-mass mask
data; they understate the matching cost a Mask R-CNN inference would
exercise. Polygon-jitter DT generator is out of scope for the v1
harness (ADR-0017 §"Out of scope").

---

## boundary — `coco_val2017_perfect_segm` (regression)

| impl                | total median |     IQR (rel) | RAM (max RSS) | vs vernier |
| ------------------- | -----------: | ------------: | ------------: | ---------: |
| **vernier**         |    75.861 s  | 418.9 ms (0.6%) |     767.1 MiB |       1.0× |
| boundary-iou-api    |    63.642 s  | 928.3 ms (1.5%) |     705.6 MiB |  **0.84× — vernier 1.19× slower** |

Per-stage medians (ms):

| stage      | vernier   | boundary-iou-api |
| ---------- | --------: | ---------------: |
| load       |     23.3  |        55614.7   |
| evaluate   |  75778.4  |         7338.7   |
| accumulate |     59.1  |          709.7   |
| summarize  |      0.58 |            1.35  |

### Stage labels mislead — read the totals

`boundary-iou-api` precomputes a dilated boundary mask per annotation
inside its `cocoeval` constructor (i.e., during what our harness times
as "load"); the subsequent `evaluate()` then operates on cached
boundary masks. vernier does the boundary derivation lazily on each
IoU computation. So:

- "load" for boundary-iou-api is **mask augmentation**, not file I/O.
- "evaluate" for vernier includes **per-pair boundary derivation**,
  not just IoU.

The honest comparison is the **total** column. On val2017
perfect-match, boundary-iou-api's "augment once, evaluate cheaply"
trade beats vernier's "derive on demand" — vernier is **1.19× slower**
end-to-end here.

### Likely causes (untested hypotheses)

1. **No boundary-mask cache.** ADR-0010 didn't pin a caching strategy,
   and the current implementation re-derives the boundary on every
   IoU pair. A per-annotation cache parallel to boundary-iou-api's
   approach is the obvious fix; we'd lose the cache benefit when an
   annotation is only consulted once but win on the dense
   val2017-style usage.
2. **Per-IoU dilation cost.** vernier's `evaluate` totals 75.8 s, of
   which mask dilation is most of the work. boundary-iou-api's totals
   show `evaluate` at 7.3 s once dilation is precomputed, suggesting
   the cache alone could close most of the gap.
3. **Memory pattern.** vernier's RAM is 767 MiB vs
   boundary-iou-api's 706 MiB. A boundary cache will make the gap
   larger, but at val2017 scale we're not RAM-bound.

Tracked as a follow-up; not a release blocker for v0.0.x.

---

## Smoke fan-out — `smoke_perfect_match_segm` (parity smoke, not a perf claim)

1-image / 1-annotation / 1-category parity case from
`tests/python/parity/fixtures/perfect_match_segm/`. Confirms the
harness fans out across IoU types and parity holds; sub-millisecond
ratios are dominated by fixed Python/PyO3 conversion overhead.

segm:

| impl              | total median |    IQR (rel) | RAM (max RSS) | vs vernier |
| ----------------- | -----------: | -----------: | ------------: | ---------: |
| **vernier**       |     0.280 ms |   10 μs (3.7%) |      73.5 MiB |       1.0× |
| faster-coco-eval  |     5.546 ms | 0.24 ms (4.3%) |      73.5 MiB | 19.81× slower |
| pycocotools       |     6.513 ms | 0.27 ms (4.2%) |      73.5 MiB | 23.26× slower |

boundary:

| impl              | total median |    IQR (rel) | RAM (max RSS) | vs vernier |
| ----------------- | -----------: | -----------: | ------------: | ---------: |
| **vernier**       |     0.355 ms |   13 μs (3.8%) |      73.4 MiB |       1.0× |
| boundary-iou-api  |     7.352 ms | 0.29 ms (4.0%) |      76.5 MiB | 20.71× slower |

---

## Cross-cutting caveats

- **Single machine.** Release-mode results aren't aggregated across
  hosts; this is one snapshot from the dev VM. The harness scopes
  every result by machine fingerprint per ADR-0017 §"Out of scope".
- **dev-VM thermals.** Absolute totals are upper-bound; the shape of
  each comparison (the speedup ratios) is the load-bearing claim.
  Bare-metal with `cpupower frequency-set -g performance` will likely
  come in cleaner.
- **bbox real-data uses jittered DT, not real detector output.** The
  matching distribution is bbox-realistic but synthetic. Real
  detector output (e.g., a Mask R-CNN dump) is out of scope for v1.

## Follow-ups

- **Boundary-mask cache** (closes the only cell where vernier loses
  to a baseline on real data).
- **Polygon-jitter DT generator** (unlocks realistic segm + boundary
  workloads beyond perfect-match).
- **Synthetic ladder runs** at n=2000 and n=5000 — harness already
  accepts the parameters.
- **Keypoints workload** — no IoU type lists `keypoints` in
  `supported_iou_types` today; lands when the keypoints track from
  ADR-0012 ships.

## Reproduction

```bash
cd bench

VERNIER_COCO_GT_PATH=/path/to/instances_val2017.json \
  uv run python -m bench run --impl all --workload coco_val2017_jittered_seed42 --iou bbox --mode release

uv run python -m bench run --impl all \
  --workload "synthetic:n_images=500,n_categories=80,gt_per_image=10,dt_per_image=30,seed=0" \
  --iou bbox --mode release

VERNIER_COCO_GT_PATH=/path/to/instances_val2017.json \
VERNIER_COCO_DT_SEGM_PATH=/path/to/perfect_dt_segm.json \
  uv run python -m bench run --impl all --workload coco_val2017_perfect_segm --iou segm     --mode release

VERNIER_COCO_GT_PATH=/path/to/instances_val2017.json \
VERNIER_COCO_DT_SEGM_PATH=/path/to/perfect_dt_segm.json \
  uv run python -m bench run --impl all --workload coco_val2017_perfect_segm --iou boundary --mode release

uv run python -m bench run --impl all --workload smoke --iou segm     --mode release
uv run python -m bench run --impl all --workload smoke --iou boundary --mode release

uv run python -m bench report --since 1h
```

`tools/fetch-coco-val.sh` (and its `tools/make-perfect-dt.py`
companion) populates the val2017 GT and `perfect_dt_segm.json` in
`<repo>/.cache/coco-val2017/`; the harness fall-back picks them up
there if the env vars aren't set.
