# Benchmarking results

Captured runs of `vernier-bench` (ADR-0017). Each file is a snapshot of one
machine on one day — cross-machine aggregation is intentionally absent (the
harness scopes everything by machine fingerprint, see ADR-0017 §"Out of
scope"). One file per snapshot — typically a month's worth of cells captured
together.

## Index

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

```bash
cd bench

VERNIER_COCO_GT_PATH=/path/to/instances_val2017.json \
  uv run python -m bench run \
    --impl all \
    --workload coco_val2017_jittered_seed0 \
    --iou bbox \
    --mode release

uv run python -m bench report --since 1h
uv run python -m bench compare --base <sha> --head <sha>
```

The harness writes JSON + `.npy` per impl under
`bench/results/<git_sha>/<machine_fp>/<paradigm>/<workload>/<iou>/<impl>.json`.
The COCO GT is sha256-pinned (`tools/fetch-coco-val.sh` matches); set
`VERNIER_COCO_GT_PATH` to skip the harness's own download. See ADR-0017
and `bench/README.md` for the full surface.
