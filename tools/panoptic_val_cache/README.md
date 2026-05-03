# vernier-panoptic-val-cache

Single source of truth for the COCO panoptic val2017 dev cache
contract (ADR-0025 PR-6). Mirrors `vernier-coco-val-cache` /
`vernier-lvis-val-cache` shape so the panoptic whole-dataset parity
smoke (`just test-parity-panoptic-val`) has the same idempotent
fetch+verify flow.

The pinned URL is the upstream `cocodataset.org` panoptic-annotations
zip, which contains both the GT JSON and the per-image PNG label
maps. SHA-256 below is the upstream zip artifact (verified at vendor
time on 2026-05-03).

| Field                | Value |
| -------------------- | ----- |
| GT URL               | `http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip` |
| Inner JSON           | `annotations/panoptic_val2017.json` |
| Inner PNG dir        | `annotations/panoptic_val2017/` |
| Cache env var        | `VERNIER_PANOPTIC_CACHE` (defaults to `<repo>/.cache/panoptic-val2017`) |

COCO data is governed by the COCO terms of use; we never commit it.

## Perfect-DT synthesis

Unlike the COCO detection / LVIS bbox flow (one synthetic JSON), a
panoptic "perfect DT" is a copy of the GT — same PNGs (in a sibling
directory) plus a JSON with the same `annotations`. This module
emits both as part of `ensure_perfect_dt`.

## Usage

```
$ python -m panoptic_val_cache
Cache directory: /path/to/.cache/panoptic-val2017
GT JSON ready: …/panoptic_val2017.json
GT PNG dir ready: …/panoptic_val2017/
Perfect DT JSON: …/perfect_dt.json
Perfect DT PNG dir: …/perfect_dt_pngs/

Export the env vars and run `just test-parity-panoptic-val`:
  export VERNIER_PANOPTIC_GT_PATH=…
  export VERNIER_PANOPTIC_GT_PNG_DIR=…
  export VERNIER_PANOPTIC_DT_PATH=…
  export VERNIER_PANOPTIC_DT_PNG_DIR=…
```
