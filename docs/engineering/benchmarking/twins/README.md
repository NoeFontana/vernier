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
| `coco_val2017_jittered` bbox | [coco_val2017_jittered-bbox.md](coco_val2017_jittered-bbox.md) | **fail, 82/84 checks pass.** In total CPU: pycocotools −1.8%, hotcoco −1.3%, vernier +2.4%. faster-coco-eval's evaluate stage is −10% in every run, so a cost driver of its matcher is still missing from the profile. |
| `objects365_val_jittered` bbox (faster-coco-eval excluded: OOM) | [objects365_val_jittered-bbox.md](objects365_val_jittered-bbox.md) | **fail, 66/68 checks pass.** vernier's accumulate stage is +10–14% in all three runs. Likely cause: cell sizes are drawn independently of category, so per-category detection totals (what accumulate sorts) differ. Next profile dimension: `(g, d)` conditioned on category rank. |

**Measure real and twin back to back.** On a shared host, speed drifts by about 10%
between runs a few minutes apart, which is as large as the tolerance. A sequential
`--impl all` pass, real then twin, has produced both spurious failures and failed
IQR gates. `just bench-twin-check` interleaves the two per impl, and the COCO report
above was measured that way: all eight IQR gates passed. The Objects365 report is
from a sequential run.
| `coco_val2017_jittered` segm / boundary | — | not yet: GT mask complexity is 7.6% low, beyond the 3% structural limit |
