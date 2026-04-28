# missing_dt_image_segm

Segm twin of `missing_dt_image`. Two images, two GT polygons, only
one DT.

- GT 1 on image 1: matched perfectly by the only DT (mask-IoU=1.0, TP)
- GT 2 on image 2: no DT for image 2 — counts as a missed positive
- npig = 2, total TP = 1, total FP = 0 → recall = 0.5

Same per-image cell shape that `evaluateImg` returns under bbox: image
2 has GT but no DT → cell with `dtIds=[]`, `dtMatches=zeros((T,0))`,
`gtIgnore=[0]`, `gtIds=[2]`. The accumulator's `D=0` filtering must
behave identically on the segm path.

Catches regressions where the segm pipeline drops the empty-DT cell
entirely (changing `npig`) or fabricates a phantom DT row.
