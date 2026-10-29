# Synthetic twins: gate reports

A twin (ADR-0067) is generated from a statistics profile of a real workload,
committed under `bench/twin_profiles/`. A twin may stand in for its real
cell only once its gate report passes. This directory keeps one report per
`(workload, iou)` cell, with the commit and machine it was measured on.

## Producing a report

```bash
# 1. Profile the real workload (needs its data locally; writes aggregate stats only).
uv run --directory bench python -m bench twin profile coco_val2017_jittered_seed0 \
    --name coco_val2017_jittered --masks

# 2. Measure both workloads on every impl in release mode and gate them.
just bench-twin-check coco_val2017_jittered_seed0 twin:coco_val2017_jittered bbox
```

`just bench-twin-check` exits non-zero unless all of these hold:

- **Structure:** totals within 3% and histograms within TV 0.05.
- **Cost:** for every impl and stage, wall and CPU time within 10% and peak
  RSS within 10%.
- **Ranking:** each impl's speed ratio to vernier within 10%.

If the generator changes, bump `TWIN_VERSION` in `bench/bench/workloads/twin.py`
and re-run every report.

**Cells where an impl cannot run on this host.** faster-coco-eval OOMs on
Objects365 at about 30 GiB. Run such cells per impl
(`just bench-run --impl <name> ...`) rather than `--impl all`, then
`python -m bench twin compare`. The report lists every impl it compared.

## Status

| cell | report | verdict |
|---|---|---|
| `coco_val2017_jittered` bbox | [coco_val2017_jittered-bbox.md](coco_val2017_jittered-bbox.md) | fail, 84/85 checks pass. faster-coco-eval's evaluate stage is about 10% cheaper on the twin (stable across re-runs), so a cost driver for its matcher is still missing from the profile. |
| `objects365_val_jittered` bbox (faster-coco-eval excluded: OOM) | [objects365_val_jittered-bbox.md](objects365_val_jittered-bbox.md) | inconclusive. Every CPU-time and memory check passes, as do both rankings. The wall-time checks need a quiet host: two runs failed their IQR gate, and pycocotools ran a single rep. |
| `coco_val2017_jittered` segm / boundary | — | not yet: GT mask complexity is 7.6% low, beyond the 3% structural limit |
