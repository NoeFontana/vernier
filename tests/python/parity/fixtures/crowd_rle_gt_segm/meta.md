# crowd_rle_gt_segm

Pins the **uncompressed-RLE input path** for crowd GTs alongside
quirk **E1** (asymmetric crowd IoU). Real-world COCO crowd
annotations ship as `{"counts": [...], "size": [h, w]}` dicts — not
polygons — so this exercises a distinct code path from
`crowd_region_segm` (which uses a polygon crowd that goes through
`rleFrPoly`).

- Image 100×100
- GT 1: non-crowd polygon at `[10, 10, 30, 30]` (area 900).
- GT 2: crowd, segmentation = `{"counts": [0, 10000], "size": [100, 100]}`
  — full-image foreground in column-major order. The leading 0-length
  background run is the encoding convention (G5).
- DT 1: polygon identical to GT 1 — perfect non-crowd match.

Even when the DT cleanly matches the non-crowd GT, the matching
loop still computes `iou(dt, crowd_gt)`, so the RLE input path is
on the hot path of every fixture run. A vernier port that
mis-decodes the column-major order, drops the leading zero run, or
silently returns `-1` on uncompressed-RLE input would surface here
as a diverging IoU cell or a crashed accumulate.
