# Benchmarking results

Captured runs of `vernier-bench` (ADR-0017). Each file is a snapshot of one
machine on one day — cross-machine aggregation is intentionally absent (the
harness scopes everything by machine fingerprint, see ADR-0017 §"Out of
scope").

Convention for filenames: `YYYY-MM-<workload>-<iou>.md`. New runs go in
this directory; older runs aren't deleted (they're history).

## Index

* [2026-05-coco-val2017-bbox.md](./2026-05-coco-val2017-bbox.md) —
  release-mode N=10, vernier 9.1× faster than pycocotools, 3.2× faster
  than faster-coco-eval; parity OK across both tiers.
* [2026-05-coco-val2017-segm-boundary.md](./2026-05-coco-val2017-segm-boundary.md) —
  release-mode N=10 on val2017 perfect-match: vernier 4.66× / 2.61×
  faster on segm; **vernier 1.19× slower than boundary-iou-api** on
  boundary (no boundary-mask cache yet — tracked follow-up).
* [2026-05-synthetic-n500.md](./2026-05-synthetic-n500.md) —
  parametric n=500 stress run; vernier 12.6× / 2.35× faster.
* [2026-05-smoke-segm-boundary.md](./2026-05-smoke-segm-boundary.md) —
  **not a perf claim**; harness fan-out smoke for segm + boundary IoU
  on the 1-image parity fixture. Superseded by the val2017 doc above
  for any actual numbers — kept for the parity-fixture cell.

## Reproducing a run

```bash
cd bench

VERNIER_COCO_GT_PATH=/path/to/instances_val2017.json \
  uv run python -m bench run \
    --impl all \
    --workload coco_val2017_jittered_seed42 \
    --iou bbox \
    --mode release

uv run python -m bench report --since 1h
uv run python -m bench compare --base <sha> --head <sha>
```

The harness writes JSON + `.npy` per impl under
`bench/results/<git_sha>/<machine_fp>/<workload>/<iou>/<impl>.json`.
The COCO GT is sha256-pinned (`tools/fetch-coco-val.sh` matches); set
`VERNIER_COCO_GT_PATH` to skip the harness's own download. See the
ADR-0017 doc and `bench/README.md` for the full surface.
