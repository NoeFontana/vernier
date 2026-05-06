# vernier-cityscapes-val-cache

Single source of truth for the Cityscapes val dev cache contract (ADR-0028
and ADR-0033 §B2). Mirrors `vernier-coco-val-cache` /
`vernier-panoptic-val-cache` / `vernier-lvis-val-cache` shape so the
semantic-segmentation Cityscapes bench cell
(`vernier-bench run --paradigm semantic --workload cityscapes_val_perfect`)
has the same idempotent fetch+verify flow.

## Access is gated upstream

Unlike COCO / LVIS, the Cityscapes dataset is **gated** — downloads
require a registered account on https://www.cityscapesdataset.com/ and
agreement to the dataset terms of use. There is no public, unauthenticated
URL for `gtFine_trainvaltest.zip` (~250 MB). Consequently, this cache
**does not download** the dataset; it expects the user to pre-populate
the cache directory with a `gtFine_trainvaltest.zip` file (or to set the
`VERNIER_CITYSCAPES_VAL_GT_DIR` / `VERNIER_CITYSCAPES_VAL_DT_DIR`
environment variables to point at already-extracted PNG directories).

The `cityscapes_val_perfect` workload (a GT-as-DT smoke; both inputs
point at the same trainId PNGs) is gated behind the
`VERNIER_BENCH_CITYSCAPES=1` environment variable so the bench harness
default test loop does not touch external systems. Set the gate to `1`
once you have populated the cache locally.

| Field                | Value |
| -------------------- | ----- |
| Cache env var        | `VERNIER_CITYSCAPES_CACHE` (defaults to `<repo>/.cache/cityscapes-val`) |
| Expected GT zip name | `gtFine_trainvaltest.zip` |
| Inner GT PNG glob    | `gtFine/val/*/*_gtFine_labelTrainIds.png` |
| Gate env var         | `VERNIER_BENCH_CITYSCAPES=1` (workload registration) |

Cityscapes data is governed by the Cityscapes Terms of Use; we never
commit it.

## Usage

```
$ python -m cityscapes_val_cache
Cache directory: /path/to/.cache/cityscapes-val
[Status report — see module docstring for the populate flow.]
```
