# Smoke fixture — segm + boundary IoU

**Not a perf claim.** This page exists because the v1 bench harness has
no real-data segm or boundary workload yet (`jittered_predictions` is
bbox-only; `synthetic` is bbox-only). The smoke fixture is a 1-image /
1-annotation / 1-category parity case from
`tests/python/parity/fixtures/perfect_match_segm/`. Numbers below confirm
that the harness fans out across IoU types and parity holds — they are
not workload-realistic and should not be cited as headline performance.

The bbox numbers in
[2026-05-coco-val2017-bbox.md](./2026-05-coco-val2017-bbox.md) are the
load-bearing claim; this doc covers the cells the harness is currently
capable of running.

Configuration as elsewhere: release mode, N=10 reps, 2 warmup discarded,
randomised impl order, IQR gate 5%, parity OK.

## segm — `smoke_perfect_match_segm` / segm

| impl              | total median |    IQR (rel) | RAM (max RSS) | vs vernier |
| ----------------- | -----------: | -----------: | ------------: | ---------: |
| **vernier**       |     0.280 ms | 10 μs (3.7%) |      73.5 MiB |       1.0× |
| faster-coco-eval  |     5.546 ms | 0.24 ms (4.3%) |      73.5 MiB |  19.81× slower |
| pycocotools       |     6.513 ms | 0.27 ms (4.2%) |      73.5 MiB |  23.26× slower |

Per-stage medians (ms):

| stage      | vernier | faster-coco-eval | pycocotools |
| ---------- | ------: | ---------------: | ----------: |
| load       |   0.059 |            0.418 |       0.301 |
| evaluate   |   0.108 |            4.844 |       4.443 |
| accumulate |   0.094 |            0.004 |       1.563 |
| summarize  |   0.013 |            0.283 |       0.213 |

Parity: **OK** (strict tier vs pycocotools, aligned tier vs faster-coco-eval).

## boundary — `smoke_perfect_match_segm` / boundary

| impl                | total median |    IQR (rel) | RAM (max RSS) | vs vernier |
| ------------------- | -----------: | -----------: | ------------: | ---------: |
| **vernier**         |     0.355 ms | 13 μs (3.8%) |      73.4 MiB |       1.0× |
| boundary-iou-api    |     7.352 ms | 0.29 ms (4.0%) |      76.5 MiB |  20.71× slower |

Per-stage medians (ms):

| stage      | vernier | boundary-iou-api |
| ---------- | ------: | ---------------: |
| load       |   0.049 |            0.843 |
| evaluate   |   0.191 |            4.533 |
| accumulate |   0.095 |            1.608 |
| summarize  |   0.013 |            0.205 |

Parity: **OK** (strict tier vs the vendored `bowenc0221/boundary-iou-api`
oracle per ADR-0010).

## What this tells you

- The harness's per-IoU fan-out works (segm + boundary fixtures parse,
  every runner imports cleanly, parity diffing matches its tier
  contract).
- vernier-vs-baselines sub-millisecond ratios are dominated by
  fixed Python/PyO3 conversion overhead at this scale; they exaggerate
  the real-data win shown in the val2017 bbox doc. Treat them as a
  parity smoke.

## What this does not tell you

- segm or boundary performance on real data — not yet possible.
  `jittered_predictions.py` jitters bboxes; mask jitter that preserves
  polygon validity is non-trivial and out of scope for v1 per
  ADR-0017 §"Out of scope". Same goes for keypoints, which the harness
  has no workload for at all (no IoU type lists `keypoints` in
  `supported_iou_types`).

## Run reproduction

```bash
cd bench
uv run python -m bench run --impl all --workload smoke --iou segm     --mode release
uv run python -m bench run --impl all --workload smoke --iou boundary --mode release
uv run python -m bench report --since 1h
```

## Machine

Same dev VM as the val2017 doc — AMD EPYC-Milan / 4 cores / 32 GiB / fingerprint `82013f18a44d`, git_sha `58f09cb9149b`. No cpufreq exposed; governor pre-flight short-circuits per the ADR-0017 "quiet on machines without cpufreq" clause.
