# perfect_match_segm

Segm baseline. One image, one GT polygon, one DT polygon identical to the GT.

- Polygon: a 50x50 square at (10, 10) — rasterized via `frPyObjects` to an
  identical RLE on both sides.
- mask-IoU(GT, DT) = 1.0 → matches at every iouThr in [0.5, 0.95]
- AP = 1.0, AR = 1.0 across every (iouThr, areaRng, maxDet) cell

Failure here means either the polygon rasterizer (vernier-mask) or the
segm Similarity kernel disagrees with pycocotools on the trivial case.
Should always be the first segm test that runs.
