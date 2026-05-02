# COCO val2017 bbox — vernier vs baselines

First headline benchmark from `vernier-bench` (ADR-0017). Release mode,
N=10 measurement reps with 2 warmup discarded, randomised impl order per
rep, IQR-relative-to-median gate at 5%, parity check across the three-tier
contract from ADR-0002.

Companion synthetic-stress numbers in [2026-05-synthetic-n500.md](./2026-05-synthetic-n500.md).

## Headline

| impl              | total median |   IQR (rel) | RAM (max RSS) | vs vernier |
| ----------------- | -----------: | ----------: | ------------: | ---------: |
| **vernier**       |   644.156 ms | 4.4 ms (0.7%) |     750.7 MiB |       1.0× |
| faster-coco-eval  |    2.058 s   | 64.3 ms (3.1%) |     644.7 MiB |    3.20× slower |
| pycocotools       |    5.871 s   | 82.7 ms (1.4%) |     558.3 MiB |    9.12× slower |

Parity: **OK** (strict tier vs pycocotools, aligned tier vs faster-coco-eval).

## Per-stage breakdown (median wall ns)

The `evaluate` stage dominates — pairwise IoU + per-image matching across
~5000 images and 80 categories. `accumulate` does the per-(T,R,K,A,M) tally
that backs the precision/recall tensor.

| stage      | vernier  | faster-coco-eval | pycocotools |
| ---------- | -------: | ---------------: | ----------: |
| load       |  15.6 ms |         425.5 ms |    422.1 ms |
| evaluate   | 549.9 ms |        1632.6 ms |   4707.4 ms |
| accumulate |  78.1 ms |       *fused\**  |    747.6 ms |
| summarize  |   0.62 ms |          2.20 ms |     1.33 ms |
| **total**  | **644.2 ms** |    **2058.0 ms** | **5870.7 ms** |

\* faster-coco-eval folds the accumulate step into evaluate; the timer is
zero rather than missing.

## Workload

`coco_val2017_jittered_seed42` — pinned `instances_val2017.json`
(`sha256:e8c7f7…`), with deterministic Gaussian-noise jittered detections
(seed 42) generated on the fly per the
`bench/bench/workloads/jittered_predictions.py` parameters. The runner emits
a precision tensor per impl; the harness diffs them via `np.array_equal`
(strict, vs pycocotools) and `np.allclose` at 4×ULP (aligned, vs
faster-coco-eval).

## Run reproduction

```bash
cd bench
VERNIER_COCO_GT_PATH=/path/to/instances_val2017.json \
  uv run python -m bench run \
    --impl all \
    --workload coco_val2017_jittered_seed42 \
    --iou bbox \
    --mode release
uv run python -m bench report --since 1h
```

Re-running on the same machine reproduces median timings within the
documented 5% IQR gate; cross-machine comparisons are intentionally
disallowed by the harness — fingerprints scope every result.

## Machine

* CPU: AMD EPYC-Milan, 4 cores
* RAM: 32 GiB
* Kernel: 6.8.0-107-generic
* Machine fingerprint: `82013f18a44d`
* git_sha: `58f09cb9149b` (`feat/bench-ram-aggregation` squash-merge)

`/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` is unexposed on
this host (VM); the governor pre-flight short-circuits per the ADR-0017
"quiet on machines without cpufreq" clause. The IQR gate carries the
noise budget instead — every impl came in well under the 5% threshold.

## Caveats

* **Single-machine** — release-mode results aren't aggregated across
  hosts; this is one snapshot from the dev VM.
* **No baselines fork** — pycocotools `2.0.11` and faster-coco-eval as
  resolved by `bench/envs/*/uv.lock`; bumping either is an ADR-level
  decision, not a routine refresh.
* **bbox only** — the jittered workload doesn't synthesize masks, so
  segm and keypoints aren't covered here. They land when the harness
  grows mask jittering (out of scope for v1 per ADR-0017 §"Out of scope").
* **dev-VM thermals** — the 644 ms total is upper-bound: a bare-metal
  run with `cpupower frequency-set -g performance` will likely come in
  cleaner. The shape of the comparison (3.2× / 9.1×) is the load-bearing
  claim; absolute numbers should be re-taken on whatever machine the
  reader cares about.
