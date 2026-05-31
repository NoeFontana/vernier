# vernier-ade20k-val-cache

Single source of truth for the ADE20K SceneParse150 validation dev
cache contract. Sibling to `vernier-panoptic-val-cache` /
`vernier-coco-val-cache` so the Mask2Former ADE-semantic real-prediction
parity smoke has the same idempotent fetch+verify flow.

The pinned URL is the canonical SceneParse150 challenge bundle
(`data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip`),
which contains both validation images and per-image semantic GT PNGs.
The bytes are governed by the SceneParse150 / ADE20K research-use
license — distinct from COCO's CC-BY 4.0. We never commit the bytes;
they land in the user's per-machine cache directory.

| Field          | Value |
| -------------- | ----- |
| GT URL         | `http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip` |
| Inner val GT   | `ADEChallengeData2016/annotations/validation/*.png` |
| Inner val imgs | `ADEChallengeData2016/images/validation/*.jpg` |
| Cache env var  | `VERNIER_ADE20K_CACHE` (defaults to `<repo>/.cache/ade20k-val`) |
| n_classes      | 150 (mmseg `reduce_zero_label=True` convention) |
| ignore_label   | 255 (raw GT label 0 → ignore) |

## Label-space conversion

The upstream PNGs encode GT as `0..150` (uint8), where 0 is the
SceneParse150 "background / unlabeled" sentinel and 1..150 are the
semantic classes. `mmsegmentation` consumes ADE20K via
`reduce_zero_label=True`, which shifts to a contiguous 0..149
train-id space and treats the upstream 0 as `ignore_label=255`.

`ensure_gt` materializes a converted GT directory under
`<cache>/val_gt_train_ids/<image_id>.png` (single-channel uint8,
0..149 + 255) so vernier's `semantic.decode_label_map_png` consumes a
uniform shape across COCO-derived and ADE-derived workloads.
`val_images/` holds the upstream JPEGs unmodified for the inference
side.

## SHA pinning

`GT_ZIP_SHA256` is pinned in `ade20k_val_cache/__init__.py` (verified
2026-05-30 against the canonical MIT CSAIL mirror). Subsequent runs
verify strictly; an upstream re-host triggers a clear mismatch error
and deletes the corrupted download so a retry can't silently accept
unverified bytes.

To re-pin after an upstream revision:

```
$ python -m ade20k_val_cache --compute-sha
# downloads + prints SHA-256 + deletes the zip
```

Then edit `GT_ZIP_SHA256` to the printed value.

## Usage

```
$ python -m ade20k_val_cache
Cache directory: /path/to/.cache/ade20k-val
Validation GT ready: …/val_gt_train_ids/  (2000 PNGs)
Validation images ready: …/val_images/    (2000 JPGs)
```
