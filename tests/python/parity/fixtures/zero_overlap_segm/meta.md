# zero_overlap_segm

Segm twin of `zero_overlap`. One GT polygon, one disjoint DT polygon.

- GT polygon = 20×20 square at (10, 10), rasterizes to 400-pixel RLE
- DT polygon = 20×20 square at (60, 60), rasterizes to 400-pixel RLE
- mask-IoU = 0.0 → DT is FP at every iouThr

Same accumulator path as the bbox twin, but routed through the segm
`Similarity` impl. Catches regressions where the segm kernel returns
non-zero IoU for fully disjoint masks (a bbox-prefilter (I1) bypass
that forgets the prefilter, or a polygon rasterizer that bleeds into
adjacent rows).
