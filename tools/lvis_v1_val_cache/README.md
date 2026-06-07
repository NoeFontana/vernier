# vernier-lvis-v1-val-cache

Single source of truth for the LVIS v1 val real-prediction dev cache
contract. Sibling to `vernier-coco-val-cache` /
`vernier-panoptic-val-cache` / `vernier-ade20k-val-cache`.

LVIS v1 reuses the COCO val2017 image set (image records reference
images by `coco_url` and `file_name`); the LVIS GT JSON is the only
new artefact. This module orchestrates a one-shot setup that yields
both the LVIS GT JSON and the val2017 image directory, so the real-
prediction populator and the parity smoke see a single uniform
"prepare LVIS val" entry point.

The GT URL + SHA256 are inherited from `vernier-lvis-val-cache` (the
synthetic-DT parity smoke); only the image side is added here.

| Field          | Value                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------- |
| GT URL         | `https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip` (~190 MB zip) |
| GT SHA256      | `2bf946b92c3037f53c172d80017f5b74ea035f00a21b20e0766b3b638b2363f9` (FAIR public-files mirror)      |
| Inner JSON     | `lvis_v1_val.json` (~890 MB uncompressed)                                                          |
| Images URL     | `http://images.cocodataset.org/zips/val2017.zip` (~778 MB; via `coco_val_cache`)                   |
| Images dir     | `val2017/` (5000 JPEGs)                                                                            |
| Cache env var  | `VERNIER_LVIS_V1_VAL_CACHE` (defaults to `<repo>/.cache/lvis-v1-val`)                              |
| n_categories   | 1203                                                                                               |
| n_images       | 19,809                                                                                             |

## Why a separate package from `lvis_val_cache`?

`vernier-lvis-val-cache` provisions the LVIS v1 GT JSON only, paired
with a synthesised perfect-DT for the federated-evaluation parity
smoke (ADR-0026 PR-6). It never touches images because the smoke
runs on a perfect / GT-derived detection set.

`vernier-lvis-v1-val-cache` adds the real images so the SOTA harness
can run inference on the val set. Keeping the GT-only cache narrow
preserves its zero-network-after-pin contract; widening it to drag
in 778 MB of images would penalise every synthetic-DT user.

Both packages share the same GT SHA pin (and verify against it
independently); a future upstream re-host triggers a clear mismatch
in both modules in lockstep.

## Usage

```
$ python -m lvis_v1_val_cache fetch
Cache directory: /path/to/.cache/lvis-v1-val
LVIS GT ready: …/lvis_v1_val.json  (sha256 verified)
COCO val2017 images ready: …/val2017/  (5000 JPGs)
```

## Symlink convention

The GT JSON lives at `<cache>/lvis_v1_val.json` for the parity
fixture's expectation, and the image directory is reached via a
symlink at `<cache>/val2017` pointing at the canonical
`coco_val_cache` images directory. This way a single `VERNIER_LVIS_V1_VAL_CACHE`
env var lets a downstream tool find both sides without separate
COCO/LVIS env-var juggling.
