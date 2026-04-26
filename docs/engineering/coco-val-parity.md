# COCO val2017 whole-dataset parity smoke

Five hand-crafted fixtures in `tests/python/parity/test_parity.py` pin
the known-quirk corners. The whole-dataset run in
`tests/python/parity/test_coco_val.py` is the headline parity claim
("full bbox parity green on COCO val2017") — vernier vs pycocotools,
strict mode, every snapshot field bit-identical.

The COCO dataset is governed by its own [terms of use][coco-terms] and
must not be committed to this repository. The smoke test reads its
inputs from environment variables and **skips when they are unset or
their paths do not exist**, so `just test` and ordinary CI stay green
on a clean checkout.

[coco-terms]: https://cocodataset.org/#termsofuse

## Environment contract

| Variable                  | Required | What it points at                                |
| ------------------------- | -------- | ------------------------------------------------ |
| `VERNIER_COCO_GT_PATH`    | yes      | path to `instances_val2017.json`                 |
| `VERNIER_COCO_DT_PATH`    | yes      | path to a detector predictions JSON (COCO-style) |
| `VERNIER_COCO_CACHE`      | no       | override the default cache dir                   |

Default cache dir: `<repo>/.cache/coco-val2017/` (gitignored).

## Local setup

1. **Fetch the GT annotations** with the helper:

   ```bash
   ./tools/fetch-coco-val.sh
   ```

   The script downloads the `annotations_trainval2017.zip` from the
   canonical CDN, extracts only `instances_val2017.json` into the
   cache, and verifies its SHA256. The expected hash is pinned in the
   script (`4949ddca…7e9`) so a corrupted or substituted download
   fails loudly.

2. **Provide a detector predictions JSON.** The repo ships no
   defaults — the parity claim is independent of which detector you
   run. Pick any published baseline whose predictions JSON is COCO-
   format compatible (`[{"image_id", "category_id", "bbox",
   "score"}, …]`). Common choices:

   - **Detectron2 model zoo** — e.g. Faster R-CNN R50-FPN's
     `coco_instances_results.json` from the official release.
   - **MMDetection** — pickle-free `*.bbox.json` outputs from
     `tools/test.py … --out`.
   - **Ultralytics** — `yolo val ... save_json=True` produces
     `predictions.json`.

   Save it anywhere; the test only cares about the path.

3. **Export both env vars** (the helper prints the line for the GT
   one) and run:

   ```bash
   export VERNIER_COCO_GT_PATH=/abs/path/instances_val2017.json
   export VERNIER_COCO_DT_PATH=/abs/path/predictions.json
   just test-coco-val
   ```

   The test calls the parity harness with strict mode and asserts
   bit-equality on `eval_imgs`, `precision`, `recall`, `scores`,
   `counts`, and the 12-element `stats` vector.

## CI integration

Not wired into the default CI matrix today — the GT alone is ~245 MB
unzipped and the run takes minutes, so it would dominate per-PR
feedback. Options when the suite stabilises:

- **Nightly workflow.** Cache the GT on the runner, commit a known
  predictions JSON to a separate read-only repo (or release asset),
  point env vars at both, run `just test-coco-val`.
- **Manual dispatch.** A `workflow_dispatch` job that takes the DT
  URL as an input parameter, useful for sanity-checking specific
  detector outputs.

Pick whichever the headline parity claim needs first; both keep the
PR-time matrix lean.

## What this test cannot catch

- **Pycocotools internal divergences from itself.** The reference is
  pinned at `pycocotools==2.0.11` per `pyproject.toml`. Bumping the
  pin is an ADR-level decision; until then "parity" means parity with
  this exact release.
- **Modes other than strict.** ADR-0002's `corrected` disposition is
  intentionally divergent and exercised by the per-quirk fixtures, not
  by this smoke.
- **Non-bbox iou types.** Phase 1 ships `bbox`; segm and keypoints
  arrive in Phase 2/3, at which point this file gains marker variants.
