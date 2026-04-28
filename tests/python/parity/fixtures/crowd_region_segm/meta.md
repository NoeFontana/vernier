# crowd_region_segm

Segm twin of `crowd_region`. Two GT polygons (one real, one crowd
covering the whole image) and two DT polygons.

- GT 1: real, 50×50 polygon at (100, 100), iscrowd=0
- GT 2: crowd, 200×200 polygon (full image), iscrowd=1
- DT 1: identical polygon to GT 1, score=0.9 → mask-IoU=1.0 with GT 1, TP
- DT 2: 30×30 polygon at (20, 20), score=0.7 → no overlap with GT 1;
  fully inside the crowd → asymmetric crowd IoU = 1.0 → DT 2 inherits
  gtIg=1 (B6) and disappears from the PR curve

Tests:
- E1 / mc:109: crowd IoU semantic `area(intersect) / area(dt)` survives
  the polygon→RLE rasterization round-trip in vernier-mask.
- B6 / ce:294: matched-to-ignore DT is dropped from TP/FP under segm.
- D1 / ce:107-109: crowd GT gets `_ignore=1`, `npig=1` (the non-crowd
  GT only).

Expected: AP=1.0 across iouThrs (DT 1 perfect-matches the only
positive GT, DT 2 is invisible).
