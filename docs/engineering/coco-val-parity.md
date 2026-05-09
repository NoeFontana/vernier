# COCO val2017 whole-dataset parity smoke

Hand-crafted fixtures in `tests/python/parity/test_parity.py` and
`tests/python/parity_boundary/test_parity_boundary.py` pin the known-
quirk corners. The whole-dataset runs in
`tests/python/parity/test_coco_val.py` (bbox + segm vs pycocotools)
and `tests/python/parity_boundary/test_coco_val.py` (boundary vs the
vendored `bowenc0221/boundary-iou-api` oracle, per ADR-0010) are the
headline parity claim — strict mode, every snapshot field bit-identical
on all 5000 val2017 images.

The COCO dataset is governed by its own [terms of use][coco-terms] and
must not be committed to this repository. The smoke test reads its
inputs from environment variables and **skips when they are unset or
their paths do not exist**, so `just test` and ordinary CI stay green
on a clean checkout.

[coco-terms]: https://cocodataset.org/#termsofuse

## Environment contract

| Variable                          | Required                          | What it points at                                       |
| --------------------------------- | --------------------------------- | ------------------------------------------------------- |
| `VERNIER_COCO_GT_PATH`            | yes (bbox + segm + boundary)      | path to `instances_val2017.json`                        |
| `VERNIER_COCO_DT_PATH`            | yes (bbox parity)                 | path to a bbox detector predictions JSON (COCO-style)   |
| `VERNIER_COCO_DT_SEGM_PATH`       | yes (segm + boundary parity)      | path to a segm predictions JSON (with `segmentation`)   |
| `VERNIER_COCO_GT_KEYPOINTS_PATH`  | yes (keypoints parity)            | path to `person_keypoints_val2017.json`                 |
| `VERNIER_COCO_DT_KEYPOINTS_PATH`  | yes (keypoints parity)            | path to a keypoint detector predictions JSON           |
| `VERNIER_COCO_CACHE`              | no                                | override the default cache dir                          |

Boundary parity reuses `VERNIER_COCO_DT_SEGM_PATH` because boundary IoU
operates on the same `segmentation` payload — there is no separate
boundary predictions format. Keypoints uses its own GT (the kp-flavored
`person_keypoints_val2017.json`, distinct from the detection
`instances_val2017.json`) and its own predictions file (each entry
carries a 51-element `keypoints` triplet array, per ADR-0012).

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

   For inference harnesses that also need pixels (e.g. the
   real-model TIDE validation in
   `tests/python/integration/real_models/tide/`), pass
   `--with-images`:

   ```bash
   ./tools/fetch-coco-val.sh --with-images
   ```

   This adds `val2017.zip` to the same canonical-CDN download
   (~778 MB zipped, ~6.2 GB extracted) and unpacks it into
   `<cache>/val2017/`. Image bytes aren't SHA-pinned (their bytes
   don't affect parity claims; only the GT's do); the integrity
   check is the canonical 5000-image count after extraction.
   Skipping the flag keeps the parity smoke's footprint small.

2. **Provide a detector predictions JSON.** Two paths:

   - **Synthetic 'perfect' DT (smoke).** `fetch-coco-val.sh` calls
     `tools/make-perfect-dt.py` to materialise both
     `.cache/coco-val2017/perfect_dt.json` (bbox-only) and
     `.cache/coco-val2017/perfect_dt_segm.json` (with
     `segmentation` and `area` copied from each GT) — one detection
     (score ≈ 1.0) per non-crowd GT annotation. This exercises the
     full pipeline at real-world scale (5000 images, 80 categories,
     ~36k annotations), but AP is trivially 1.0 — it's a plumbing
     and scale check, not a numerical parity claim.
   - **Real detector predictions (headline parity).** Pick any
     published baseline whose predictions JSON is COCO-format
     compatible (`[{"image_id", "category_id", "bbox", "score"},
     …]`) and point `VERNIER_COCO_DT_PATH` at it. Common sources:

     - **Detectron2 model zoo** — e.g. Faster R-CNN R50-FPN's
       `coco_instances_results.json` (bbox) or Mask R-CNN's
       `coco_segmentation_results.json` (segm) from the official
       release.
     - **MMDetection** — pickle-free `*.bbox.json` /
       `*.segm.json` outputs from `tools/test.py … --out`.
     - **Ultralytics** — `yolo val ... save_json=True` produces
       `predictions.json` (bbox; segm requires a segmentation
       model).

   Bbox and segm predictions are typically separate files; point
   `VERNIER_COCO_DT_PATH` at the bbox JSON and
   `VERNIER_COCO_DT_SEGM_PATH` at the segm JSON. Each env var is
   only required for its own test — the suite skips the others.

   Keypoints predictions live in a separate file (each detection
   carries a 51-element `keypoints` triplet array). Common sources:
   Detectron2's Keypoint R-CNN model zoo, MMPose's COCO-format export,
   or any baseline targeting `person_keypoints_val2017.json`. Point
   `VERNIER_COCO_GT_KEYPOINTS_PATH` at the GT and
   `VERNIER_COCO_DT_KEYPOINTS_PATH` at the predictions JSON.

3. **Export the env vars** (the helper prints the lines for the
   cached GT and synthesised DTs) and run:

   ```bash
   export VERNIER_COCO_GT_PATH=/abs/path/instances_val2017.json
   export VERNIER_COCO_DT_PATH=/abs/path/bbox_predictions.json
   export VERNIER_COCO_DT_SEGM_PATH=/abs/path/segm_predictions.json
   just test-coco-val
   ```

   The tests call the parity harness with strict mode and assert
   bit-equality on `eval_imgs`, `precision`, `recall`, `scores`,
   `counts`, and the stats vector for each iou_type — 12 elements
   for bbox/segm/boundary, 10 for keypoints (ADR-0012).

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

## Expected status today

All seven tests pass without xfail:

- `test_coco_val2017_bbox_parity_perfect_dt` — synthetic bbox smoke.
- `test_coco_val2017_bbox_parity` — real bbox predictions, env-gated.
- `test_coco_val2017_segm_parity_perfect_dt` — synthetic segm smoke,
  every det carries the GT polygons via `make-perfect-dt.py --segm`.
- `test_coco_val2017_segm_parity` — real segm predictions, env-gated.
- `test_coco_val2017_boundary_parity_perfect_dt` — synthetic boundary
  smoke (reuses `perfect_dt_segm.json`), oracle is bowenc0221.
- `test_coco_val2017_boundary_parity` — real segm predictions vs the
  bowenc0221 oracle, env-gated on `VERNIER_COCO_DT_SEGM_PATH`.
- `test_coco_val2017_keypoints_parity` — real kp predictions vs
  pycocotools, env-gated on `VERNIER_COCO_GT_KEYPOINTS_PATH` and
  `VERNIER_COCO_DT_KEYPOINTS_PATH`. Asserts byte-identical eval_imgs,
  precision/recall/scores tensors, and the 10-stat summary (ADR-0012).

Bit-exact parity holds on all 5000 val2017 images for every iou_type
(bbox / segm / boundary / keypoints): every `evalImgs` cell, the
precision/recall/scores tensors, and the stats vector (12 elements for
det iou_types, 10 for keypoints per ADR-0012) match the respective
reference oracle.

The earlier perfect-DT divergence on overlapping crowd/non-crowd ties
(quirk **A4**) was rooted in `f32` IoU intermediates losing
bit-equivalence with pycocotools' `f64` `maskUtils.iou` kernel.
ADR-0008 supersedes the bbox clause of ADR-0004 and pins IoU at
`f64` end-to-end; runs that diverge on real data are real regressions
to file.

## What this test cannot catch

- **Pycocotools internal divergences from itself.** The reference is
  pinned at `pycocotools==2.0.11` per `pyproject.toml`. Bumping the
  pin is an ADR-level decision; until then "parity" means parity with
  this exact release.
- **Modes other than strict.** ADR-0002's `corrected` disposition is
  intentionally divergent and exercised by the per-quirk fixtures, not
  by this smoke.
- **Pycocotools/bowenc0221 internal divergences from themselves.** The
  boundary track pins the vendored oracle at the snapshot in
  `tests/python/parity_boundary/oracle/`. Bumping it follows the same
  ADR-level discipline as bumping the pycocotools pin.
