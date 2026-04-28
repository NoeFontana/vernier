# multi_polygon_gt_segm

Pins quirk **K2** (`strict`): a single annotation whose `segmentation`
is a list of multiple polygons must merge into one RLE before IoU
computation. Real-world COCO has many of these (e.g. a person split
across body and arm regions, a chair occluded by a table).

- Image 100×100
- One GT, `segmentation` = two disjoint 20×20 squares at (10,10) and
  (60,60); `area` = 800 (sum of parts), `bbox` = [10, 10, 70, 70]
  (covers both parts)
- One DT with the same multi-polygon segmentation → identical merged
  RLE → mask-IoU=1.0

Catches a vernier-mask `Rle::merge` regression that, applied to the
two sub-RLEs in a different order or with the wrong union semantics,
produces a different counts string than pycocotools' merge. Also
catches a `Rle::from_polygons` port that silently dropped all but the
first polygon.
