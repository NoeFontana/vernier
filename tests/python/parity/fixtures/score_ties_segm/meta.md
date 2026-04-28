# score_ties_segm

Segm twin of `score_ties`. Pins quirk **A1** (`kind='mergesort'`)
along the segm path: when two detections share a score, pycocotools'
stable argsort preserves input order and the matching assignment is
deterministic.

Two GT polygons and two DT polygons, each DT a perfect mask twin of one
GT, all four with score 0.7. With stable sort:

- DT 0 (input position 0) processed first → claims GT 0 at mask-IoU=1.0.
- DT 1 (input position 1) processed second → claims GT 1 at mask-IoU=1.0.

If the segm-path argsort were unstable, ties could swap and `dtMatches`
row 0 vs row 1 would flip — `stats[0]` (mAP) is invariant either way,
but the parity harness diffs every per-image array and would catch it.
