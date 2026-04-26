# Parity fixtures

Hand-curated minimal COCO inputs, each designed to pin one boundary case in the
matching / accumulation pipeline. Every fixture is a `(gt.json, dt.json)` pair
plus a `meta.md` explaining what the fixture exercises.

| Fixture | Pins which quirk |
|---|---|
| `perfect_match` | Baseline: 1 GT + 1 perfectly-matching DT. AP=1.0 across all thresholds. |
| `zero_overlap` | DT bbox disjoint from GT — AP=0.0 baseline; accumulator must produce all-zero precision/recall. |
| `crowd_region` | Crowd GT (E1) allows many-to-one matching; matched DTs become "ignore" (B6). |
| `missing_dt_image` | DT covers a subset of GT image ids — partial recall, accumulator handles None evalImg cells. |
| `iou_at_threshold` | DT IoU exactly 0.5 — exercises the `min(t, 1 - 1e-10)` boundary fudge (B1). |
| `score_ties` | Two DTs with identical scores — pins stable mergesort tiebreak on input order (A1). |

Adding a new fixture: create a directory with `gt.json`, `dt.json`, and a
`meta.md` describing what it tests, then add a parametrize entry in
`test_parity.py`. Keep fixtures small (single images, ≤3 annotations); large
COCO subsets belong in integration tests pulled from `fiftyone`, not here.

All fixtures are bbox-only by default. Segmentation and keypoints variants
can be added once the segm/keypoints code paths land in vernier.
