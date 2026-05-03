# vernier-lvis-val-cache

Single source of truth for the LVIS v1 val dev cache contract
(ADR-0026 PR-6). Mirrors `vernier-coco-val-cache`'s shape so the
LVIS whole-dataset parity smoke (`just test-parity-lvis-val`) has
the same idempotent fetch+verify flow.

LVIS reuses COCO 2017 images by `coco_url` reference; only the LVIS
GT JSON and the perfect-DT synthesized from it are cache content
here. The pinned URL points at the canonical FAIR / Facebook public
files mirror; the SHA256 below is the unzipped JSON.

| Field                | Value |
| -------------------- | ----- |
| GT URL               | `https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip` |
| Inner filename       | `lvis_v1_val.json` |
| GT SHA-256           | `2bf946b92c3037f53c172d80017f5b74ea035f00a21b20e0766b3b638b2363f9` |
| Cache env var        | `VERNIER_LVIS_CACHE` (defaults to `<repo>/.cache/lvis-val`) |

LVIS data is governed by the LVIS terms of use; we never commit it.

## Usage

```
$ python -m lvis_val_cache
Cache directory: /path/to/.cache/lvis-val
GT ready: …/lvis_v1_val.json
Perfect bbox DT: …/perfect_dt.json
Perfect segm DT: …/perfect_dt_segm.json

Export the env vars and run `just test-parity-lvis-val`:
  export VERNIER_LVIS_GT_PATH=…
  export VERNIER_LVIS_DT_PATH=…
  export VERNIER_LVIS_DT_SEGM_PATH=…
```
