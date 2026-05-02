# 2026-05 vernier-bench snapshot

Consolidated view of every release-mode cell captured this month.
Per-cell docs (linked below) keep the full per-stage breakdowns,
caveats, and reproduction commands.

All runs share the same configuration:

- **Harness:** `vernier-bench` per ADR-0017, release mode, N=10
  measurement reps with 2 warmup discarded, randomised impl order
  per rep, IQR-relative-to-median gate at 5%.
- **Parity tiers** (ADR-0002): strict (`np.array_equal` vs
  pycocotools), aligned (`np.allclose` at 4×ULP vs faster-coco-eval),
  strict vs the vendored `bowenc0221/boundary-iou-api` oracle for
  boundary.
- **Machine:** AMD EPYC-Milan / 4 cores / 32 GiB / kernel
  `6.8.0-107-generic`, fingerprint `82013f18a44d`, git_sha
  `58f09cb9149b`. cpufreq unexposed on this VM; the IQR gate carries
  the noise budget (every cell came in well under 5%).
- **Parity:** OK on every cell.

## All cells at a glance

`× vernier` is total wall median for the baseline divided by vernier's;
greater than 1.0 means vernier is faster.

| workload                                  | iou      | vernier (median) | pycocotools     | faster-coco-eval | boundary-iou-api |
| ----------------------------------------- | -------- | ---------------: | --------------: | ---------------: | ---------------: |
| `coco_val2017_jittered_seed42`            | bbox     |       644.156 ms |    9.12× slower |     3.20× slower |             —    |
| `synthetic:n=500,c=80,g=10,d=30,seed=0`   | bbox     |       116.763 ms |   12.57× slower |     2.35× slower |             —    |
| `coco_val2017_perfect_segm`               | segm     |         1.742 s  |    4.66× slower |     2.61× slower |             —    |
| `coco_val2017_perfect_segm`               | boundary |        75.861 s  |             —   |              —   | **0.84× — vernier 1.19× slower** |
| `smoke_perfect_match_segm`                | segm     |         0.280 ms |   23.26× slower |    19.81× slower |             —    |
| `smoke_perfect_match_segm`                | boundary |         0.355 ms |             —   |              —   |    20.71× slower |

Source docs:

- bbox real data — [2026-05-coco-val2017-bbox.md](./2026-05-coco-val2017-bbox.md)
- bbox synthetic stress — [2026-05-synthetic-n500.md](./2026-05-synthetic-n500.md)
- segm + boundary real data — [2026-05-coco-val2017-segm-boundary.md](./2026-05-coco-val2017-segm-boundary.md)
- smoke fan-out (parity smoke, not a perf claim) — [2026-05-smoke-segm-boundary.md](./2026-05-smoke-segm-boundary.md)

## What ships in the headline

- **bbox** — vernier 9.1× faster than pycocotools and 3.2× faster
  than faster-coco-eval on val2017 (~5000 imgs / 80 cats / jittered DT).
- **segm** — vernier 4.66× faster than pycocotools and 2.61× faster
  than faster-coco-eval on val2017 perfect-match (~36k anns).
- **synthetic** — gap widens to 12.6× / 2.35× on small uniform
  workloads (pycocotools' quadratic hot paths scale less harshly when
  per-image work is small but uniform).

## Honest finding: boundary on val2017

vernier is **1.19× slower than `boundary-iou-api`** end-to-end on the
val2017 perfect-match boundary cell (75.861 s vs 63.642 s). Cause is
algorithmic, not a regression: boundary-iou-api precomputes a dilated
boundary mask per annotation inside its `cocoeval` constructor (timed
as "load") and operates on cached masks during evaluate; vernier
derives the boundary lazily on each IoU pair. Stage labels mislead
here — the totals are the honest comparison.

ADR-0010 didn't pin a boundary-mask cache strategy. A per-annotation
cache parallel to boundary-iou-api's approach is the obvious fix and
would close most of the gap based on their `evaluate` total (7.3 s vs
vernier's 75.8 s once dilation is precomputed). Tracked as a
follow-up — not a release blocker for v0.0.x.

## Caveats that apply to every cell here

- **Single machine** — no cross-host aggregation; ADR-0017 §"Out of
  scope" makes this a deliberate harness constraint, not a TODO.
- **Pinned baselines** — pycocotools `2.0.11` and faster-coco-eval as
  resolved by `bench/envs/*/uv.lock`; bumping is an ADR-level
  decision.
- **Perfect-match DT under-stresses matching** for the val2017 segm +
  boundary cells: every DT lines up with a GT exactly, so
  false-positive accumulation work is missing. Realistic for the
  eval/accumulate hot path on real-mass mask data, not for the
  matching cost a Mask R-CNN inference would exercise. Polygon-jitter
  generator is out of scope for the v1 harness.
- **dev-VM thermals** — absolute totals are upper-bound; the shape of
  each comparison (the speedup ratios) is the load-bearing claim.

## Follow-ups

- Boundary-mask cache (see above) — closes the only cell where vernier
  loses to a baseline on real data.
- Polygon-jitter DT generator — unlocks realistic segm + boundary
  workloads (currently bbox-only or perfect-match).
- Synthetic ladder runs at n=2000 and n=5000 — harness already
  accepts the parameters; doc captured n=500 as the motivation smoke.
- Keypoints workload — no IoU type lists `keypoints` in
  `supported_iou_types` today; lands when the keypoints track from
  ADR-0012 ships.

## Reproduction

Per-cell commands live in each linked doc. The full sweep:

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
