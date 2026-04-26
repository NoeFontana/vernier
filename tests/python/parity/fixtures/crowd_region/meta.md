# crowd_region

Two GTs (one real, one crowd) and two DTs.

- GT 1: real, bbox=[100,100,50,50], iscrowd=0
- GT 2: crowd, bbox=[0,0,200,200], iscrowd=1 — covers everything
- DT 1: bbox=[100,100,50,50], score=0.9 → IoU=1.0 with GT 1, matches GT 1 (TP)
- DT 2: bbox=[20,20,30,30], score=0.7 → no overlap with GT 1; falls into crowd
  → DT 2 inherits gtIg=1, becomes neither TP nor FP

Tests:
- E1 / mc:109: crowd IoU semantic `area(intersect) / area(dt)` (DT 2 is fully
  inside the crowd, so its crowd IoU is 1.0).
- B6 / ce:294: matched-to-ignore DT inherits ignore → vanishes from PR curve.
- D1 / ce:107-109: crowd GT gets `_ignore=1`, npig counts only the non-ignore
  GT (so npig=1, not 2).

Expected: AP=1.0 at iouThr=0.5..0.95 (DT 1 perfect-matches the only positive
GT, DT 2 is invisible).
