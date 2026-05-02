# Synthetic n=500 bbox — vernier vs baselines

Stress-test workload, parametric: 500 images × 80 categories × 10 GT and
30 DT per image, seed 0. Smaller than COCO val2017, picked as the
"is the harness doing the right thing" smoke before the real run in
[2026-05-coco-val2017-bbox.md](./2026-05-coco-val2017-bbox.md).

Same harness configuration as the val2017 doc (release mode, N=10 reps,
2 warmup discarded, IQR gate 5%, parity checked).

## Headline

| impl              | total median |    IQR (rel) | RAM (max RSS) | vs vernier |
| ----------------- | -----------: | -----------: | ------------: | ---------: |
| **vernier**       |   116.763 ms | 2.95 ms (2.5%) |     147.7 MiB |       1.0× |
| faster-coco-eval  |   274.720 ms | 4.01 ms (1.5%) |     143.1 MiB |    2.35× slower |
| pycocotools       |     1.468 s  | 25.31 ms (1.7%) |     201.3 MiB |   12.57× slower |

Parity: **OK** (strict tier vs pycocotools, aligned tier vs faster-coco-eval).

## Workload

`synthetic:n_images=500,n_categories=80,gt_per_image=10,dt_per_image=30,seed=0`

Workload ID derived from the parameter tuple, cached at
`~/.cache/vernier-bench/synthetic/synthetic_n500_c80_g10_d30_s0_{gt,dt}.json`.
Deterministic given the seed — re-running rebuilds byte-identical inputs
from the cache.

## Why the gap shrinks vs val2017

vernier-vs-pycocotools is **9.1× on val2017** but **12.6× on synthetic n=500**:
the synthetic load has uniformly small images and a regular GT/DT count;
val2017 has a wide image-size and per-image GT distribution. pycocotools'
quadratic-ish hot loops scale less harshly when the per-image work is
small but uniform.

## Run reproduction

```bash
cd bench
uv run python -m bench run \
  --impl all \
  --workload "synthetic:n_images=500,n_categories=80,gt_per_image=10,dt_per_image=30,seed=0" \
  --iou bbox \
  --mode release
uv run python -m bench report --since 1h
```

## Machine

Same dev VM as the val2017 doc — AMD EPYC-Milan / 4 cores / 32 GiB / fingerprint `82013f18a44d`, git_sha `58f09cb9149b`.

## Caveats

Single-machine, no governor pinning available on this VM, single workload
size. The synthetic ladder (n=2000, n=5000) is in scope for follow-up
runs — the harness already accepts those parameters; this doc covers
n=500 as the smoke that motivated the val2017 capture above.
