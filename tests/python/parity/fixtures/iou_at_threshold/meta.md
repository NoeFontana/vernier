# iou_at_threshold

One GT, one DT, IoU exactly 0.5.

- GT bbox = [0, 0, 100, 100], area = 10000
- DT bbox = [0, 0, 100, 50],  area = 5000, intersect = 5000, union = 10000
- IoU = 5000 / 10000 = 0.5 (bit-exact in float64)

Pins the `iou = min(t, 1 - 1e-10)` initialization in `evaluateImg` (ce:276).

- At iouThr=0.5 → initial best = 0.5 - 1e-10 → DT IoU 0.5 > 0.4999... → match. TP.
- At iouThr=0.55+ → initial best = 0.55 - 1e-10 → DT IoU 0.5 < 0.55 → no match. FP.

Expected: only the iouThr=0.5 slot has a match. Mean AP = 1/10 = 0.1 (10
thresholds, 1 hits).

This is the canonical fixture that breaks any candidate evaluator that uses
plain `>=` or plain `>` at the threshold boundary instead of pycocotools's
`min(t, 1 - 1e-10)` initialization.
