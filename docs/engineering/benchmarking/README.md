# Benchmarking results

Captured runs of `vernier-bench` (ADR-0017). Each file is a snapshot of one
machine on one day — cross-machine aggregation is intentionally absent (the
harness scopes everything by machine fingerprint, see ADR-0017 §"Out of
scope"). One file per snapshot — typically a month's worth of cells captured
together.

## Index

* [2026-09-release-0.4.0-round.md](./2026-09-release-0.4.0-round.md)
  — **current headline snapshot.** Full release-mode re-measurement for
  the 0.4.0 tag, same host as the two rounds below. Objects365 8.771 s →
  6.60 s at one CPU and 4.702 s → 3.20 s at eight, with both competitor
  arms flat within 2 % as controls; LVIS 80.98× over lvis-api and
  bit-equal to it; COCO bbox 354 → 305 ms. Also records the two
  measurement failures this round caught — a contaminated thread job
  that produced a fake segm regression, and a refresh recipe that had
  been silently emitting `dev`-mode data.
* [2026-09-longtail-perf-round.md](./2026-09-longtail-perf-round.md)
  — superseded by the round above. Before/after for the four changes that
  stopped the evaluate path paying for empty cells (ADR-0050, ADR-0051).
  Objects365 at 8 threads 8.76 s → 4.70 s (from losing to hotcoco to
  1.7× ahead), LVIS −23 %, COCO flat.
* [2026-09-hotcoco-fce-1.8-and-scale.md](./2026-09-hotcoco-fce-1.8-and-scale.md)
  — the round that added hotcoco as a baseline, moved
  faster-coco-eval to 1.8.0, enforces a per-cell CPU budget so every
  impl gets equal compute (ADR-0049), measures memory exactly
  (per-stage `VmHWM`), and adds an Objects365 scale cell where
  faster-coco-eval OOMs. Supersedes the 2026-05 numbers below: the
  budget changed what a "single-thread" cell means, and the host
  differs.
* [2026-05-vernier-vs-alternatives.md](./2026-05-vernier-vs-alternatives.md)
  — current cross-paradigm dev-mode snapshot. Instance (bbox / segm /
  boundary / keypoints), panoptic, and the vernier-only synthetic
  semantic baseline. Carries the perf-round timeline so a reader can
  see how each cell got to its current numbers.
* [2026-05-detr-r50-real-predictions.md](./2026-05-detr-r50-real-predictions.md)
  — release-mode snapshot of the bbox cell against real
  `facebook/detr-resnet-50` predictions on COCO val2017 (150,680
  detections). Companion to the jittered-DT cells; documents the
  real-distribution gap that synthetic workloads don't exercise.
* [2026-05-mask2former-real-predictions.md](./2026-05-mask2former-real-predictions.md)
  — release-mode panoptic + semantic cells against real
  `facebook/mask2former-swin-tiny-coco-panoptic` (COCO val2017) and
  `facebook/mask2former-swin-tiny-ade-semantic` (ADE20K val)
  predictions. Companion to the perfect-DT panoptic / synthetic
  semantic cells; closes the real-distribution gap on two more
  paradigms.
* [2026-05-bbox-cdf.md](./2026-05-bbox-cdf.md) — Stage 0 measurement
  for the bbox-IoU optimization plan. **Two regimes**: multi-category
  sparse (val2017: median `G·D = 1`, drop 1b/1c/2c, lever is per-call
  overhead) vs single-category dense (synthetic G=200/c=1: median
  `G·D = 20k`, 1b/1c/2c become positive ROI, lever is the inner loop).
  Drove the call to drop explicit `pulp::Simd` lanes for the val2017
  shape.

## Instrumentation guides

* [bbox-iou-stage0-instrumentation.md](./bbox-iou-stage0-instrumentation.md)
  — `bench-histogram` Cargo-feature workflow for capturing the
  per-call `(G, D, wall_ns)` distribution. The feature still ships in
  `vernier-core` / `vernier-ffi`; this is the live how-to.

## Reproducing a run

One cell, to check the harness resolves before committing to a round:

```bash
just bench-run --impl all --workload coco_val2017_jittered_seed0 \
    --iou bbox --mode release

just bench-run --help          # every flag
uv run --directory bench python -m bench report --since 1h
uv run --directory bench python -m bench compare --base <sha> --head <sha>
```

> **`--mode` defaults to `dev`, which is one rep with no warmup and no
> IQR gate.** A cell run without `--mode release` produces a single
> sample whose reported IQR is `0 ns` — it is a smoke test, not a
> measurement, and it must never reach a published table. This is not
> hypothetical: the refresh recipe carried in the 2026-09 snapshot
> omitted the flag under a comment that said "release", and the
> `0 ns` IQR is what gave it away.

The harness writes JSON + `.npy` per impl under
`bench/results/<git_sha>/<machine_fp>/<paradigm>/<workload>/<iou>/<impl>.json`.
The COCO GT is sha256-pinned (`tools/fetch-coco-val.sh` matches); set
`VERNIER_COCO_GT_PATH` to skip the harness's own download. See ADR-0017
and `bench/README.md` for the full surface.

## Refreshing the published numbers

**This is the canonical command list.** It lives here and nowhere else:
it used to be copied into each snapshot and into the release runbook,
and the copies drifted — one of them silently produced `dev`-mode data
for a table captioned "release". Link to this section rather than
pasting it.

```bash
just bench-sync                 # rebuild vernier + every competitor venv

# --- instance: the headline cells -------------------------------------
# `--impl all` selects the impls that support each metric
# (matrix.py: IMPL_PARADIGM_SUPPORT) — boundary has no pycocotools or
# hotcoco arm, so its row set is deliberately smaller.
for iou in bbox segm boundary; do
  just bench-run --impl all --workload coco_val2017_jittered_seed0 \
      --iou "$iou" --mode release
done

# Keypoints has its OWN workload — the bbox/segm/boundary one does not
# carry keypoint annotations and rejects `--iou keypoints`.
just bench-run --impl all --workload coco_val2017_keypoints_jittered_seed0 \
    --iou keypoints --mode release

# --- thread scaling ---------------------------------------------------
for iou in bbox segm boundary; do
  just bench-run --impl vernier --workload coco_val2017_jittered_seed0 \
      --iou "$iou" --num-threads 1,2,4,8 --mode release --no-parity
done
just bench-run --impl vernier --workload coco_val2017_keypoints_jittered_seed0 \
    --iou keypoints --num-threads 1,2,4,8 --mode release --no-parity

# --- LVIS v1 val ------------------------------------------------------
just bench-run --impl all --workload lvis_v1_val_jittered_seed0 \
    --iou bbox --mode release

# --- panoptic / semantic ---------------------------------------------
# No `--iou` here: `--paradigm` auto-derives from the workload, and each
# of these paradigms has exactly one metric (`pq`, `miou`), chosen by the
# paradigm rather than the flag. `--iou` only accepts the four *instance*
# metrics and rejects `pq` / `miou` outright.
just bench-run --impl all --workload coco_panoptic_val2017_perfect --mode release
just bench-run --impl all --workload coco_val2017_semantic_perfect --mode release

# --- Objects365 scale cell -------------------------------------------
# Per-impl, and `dev` on purpose: one rep takes minutes, and a runner
# that OOMs (faster-coco-eval, ~30 GiB) aborts its whole cell, so one
# OOM must not take the others with it.
for impl in vernier hotcoco pycocotools; do
  just bench-run --impl "$impl" --workload objects365_val_jittered_seed0 \
      --iou bbox --mode dev --no-parity
done

python tools/render_benchmarks.py
```

`render_benchmarks.py` regenerates **`docs/benchmarks.md` only**. These
files hand-mirror numbers from it and have to be updated in the same
commit, or they drift — which is how `docs/comparison.md` came to claim
`~57×` against lvis-api while `README.md` claimed `73.1×`:

| file | what it mirrors |
| --- | --- |
| `README.md` (tagline, headline table, thread table, host footer) | instance + panoptic + semantic + LVIS cells, version pins |
| `docs/comparison.md` ("At a glance", per-library sections) | per-library ratios and peak-RSS figures |
| `docs/index.md` | the thread-scaling headline |
| `docs/migrate/from-faster-coco-eval.md` | boundary and segm ratios at 1 and 8 CPUs |
| `docs/how-to/configure-evaluator.md`, `docs/how-to/cli-eval.md` | the `num_threads` scaling figures |

Then add a dated snapshot under this directory and re-point the index
above at it.
