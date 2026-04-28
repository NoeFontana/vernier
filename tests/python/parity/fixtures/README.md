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
| `crowd_overlap_tiebreak` | Overlapping crowd + non-crowd GT at IoU≈1 — pins f64-end-to-end IoU so the "later equal wins" matcher (B2) reproduces pycocotools' last-bit tiebreak (ADR-0008). |
| `perfect_match_segm` | Segm baseline: 1 polygon GT + 1 identical polygon DT. Mask-IoU=1.0 across all thresholds. |
| `zero_overlap_segm` | Segm twin of `zero_overlap` — disjoint polygons, mask-IoU=0; pins the no-match accumulator path through the segm kernel. |
| `crowd_region_segm` | Segm twin of `crowd_region` — crowd polygon covers image; pins asymmetric crowd mask-IoU (E1) and B6 ignore-inheritance under segm. |
| `score_ties_segm` | Segm twin of `score_ties` — equal-score DTs with mask-IoU=1; pins A1 stable-sort under segm. |
| `missing_dt_image_segm` | Segm twin of `missing_dt_image` — empty-DT cell handling under segm. |
| `multi_polygon_gt_segm` | Single GT with two-polygon `segmentation`; pins K2 multi-polygon→single-RLE merge. |
| `polygon_at_image_edge_segm` | Polygon with vertices at (W, H); pins H4/H5 asymmetric x/y boundary clipping and rounding in rleFrPoly. |
| `self_intersecting_polygon_segm` | Bowtie polygon (self-crossing at centre); pins H3 — rasterized mask depends on point-order under the supersampled Bresenham boundary walk. |
| `crowd_rle_gt_segm` | Crowd GT shipped as `{"counts": [...], "size": [...]}` dict (real-world COCO format) instead of polygon; pins the uncompressed-RLE input path alongside E1 asymmetric crowd IoU. |

Adding a new fixture: create a directory with `gt.json`, `dt.json`, and a
`meta.md` describing what it tests, then add the fixture name to
`BBOX_FIXTURES` (bbox path) or `SEGM_FIXTURES` (segm path) in
`test_parity.py`. Keep fixtures small (single images, ≤3 annotations); large
COCO subsets belong in integration tests pulled from `fiftyone`, not here.

Segm fixtures must include a `segmentation` field on every GT and DT —
absent fields raise `EvalError::InvalidAnnotation` instead of being
silently treated as empty. Keypoints fixtures land alongside the Phase 3
keypoints code path.
