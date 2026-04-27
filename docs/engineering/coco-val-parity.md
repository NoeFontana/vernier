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
   script (`e8c7f790…b6f`) so a corrupted or substituted download
   fails loudly.

2. **Provide a detector predictions JSON.** Two paths:

   - **Synthetic 'perfect' DT (smoke).** `fetch-coco-val.sh` calls
     `tools/make-perfect-dt.py` to materialise
     `.cache/coco-val2017/perfect_dt.json` — one detection (score
     1.0) per non-crowd GT annotation. This exercises the full
     pipeline at real-world scale (5000 images, 80 categories,
     ~36k annotations), but AP is trivially 1.0 — it's a plumbing
     and scale check, not a numerical parity claim.
   - **Real detector predictions (headline parity).** Pick any
     published baseline whose predictions JSON is COCO-format
     compatible (`[{"image_id", "category_id", "bbox", "score"},
     …]`) and point `VERNIER_COCO_DT_PATH` at it. Common sources:

     - **Detectron2 model zoo** — e.g. Faster R-CNN R50-FPN's
       `coco_instances_results.json` from the official release.
     - **MMDetection** — pickle-free `*.bbox.json` outputs from
       `tools/test.py … --out`.
     - **Ultralytics** — `yolo val ... save_json=True` produces
       `predictions.json`.

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

## Expected status today (Phase 1 Week 5)

`test_coco_val2017_bbox_parity_perfect_dt` is currently marked
`xfail(strict=False)`. The perfect-DT smoke runs end-to-end on all
5000 val2017 images, but the candidate diverges from pycocotools on a
handful of images where overlapping GT bboxes force an arbitrary
tiebreak (quirk **A4**, `_ignore`-ascending GT order). The matching
engine that owns this decision is the Phase 1 Week 3 deliverable —
Week 5 only ships the FFI surface around it. The xfail flips to
`xpass` as Weeks 3-4 land their final passes; at that point the
marker comes off.

`test_coco_val2017_bbox_parity` (real detector predictions, env-var
gated) has no xfail. Runs that diverge on real data are real
regressions to file.

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
