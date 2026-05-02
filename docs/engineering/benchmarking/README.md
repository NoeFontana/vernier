# Benchmarking results

Captured runs of `vernier-bench` (ADR-0017). Each file is a snapshot of one
machine on one day — cross-machine aggregation is intentionally absent (the
harness scopes everything by machine fingerprint, see ADR-0017 §"Out of
scope").

One file per snapshot — typically a month's worth of cells captured
together. Older snapshots aren't deleted (they're history); new
captures land alongside them.

## Index

* [2026-05-snapshot.md](./2026-05-snapshot.md) — first release-mode
  capture: bbox 9.1× / 3.2×, segm 4.66× / 2.61×, synthetic n=500
  12.6× / 2.35× — and an honest **vernier 1.19× slower than
  boundary-iou-api** on val2017 boundary (no boundary-mask cache yet,
  tracked follow-up). Parity OK on every cell.

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
