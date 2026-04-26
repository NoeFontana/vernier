# perfect_match

Baseline. One image, one GT, one DT identical to the GT.

- IoU(GT, DT) = 1.0
- Matches at every iouThr in [0.5, 0.95]
- AP = 1.0, AR = 1.0 across every (iouThr, areaRng, maxDet) cell

Failure here means the harness or the candidate evaluator can't even handle
the trivial case. Should always be the first test that runs.
