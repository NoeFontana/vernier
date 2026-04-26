# missing_dt_image

Two images, two GTs, only one image has a DT.

- GT 1 on image 1: matched perfectly by the only DT (TP)
- GT 2 on image 2: no DT for image 2 — counts as a missed positive
- npig = 2, total TP = 1, total FP = 0 → recall = 0.5

Expected:
- recall = 0.5 across iouThrs (only one of two GTs ever recovered)
- precision = 1.0 up to recall 0.5, then 0 (interpolated to ~0.5 over 101 pts)

Tests:
- ce:247-248: `evaluateImg` returns `None` for the (img=2, cat=1) cell because
  its DT list is empty AND its GT list is non-empty? Actually no — it returns
  None only if BOTH gt and dt are empty. Image 2 has GT but no DT → returns a
  dict with `dtIds=[]`, `dtMatches=zeros((T,0))`, `gtIgnore=[0]`, `gtIds=[2]`.
  This row contributes to npig but produces no DT for the cumulative curve.
- accumulate's per-cell `evalImgs` filtering correctly handles `D=0` cells.
