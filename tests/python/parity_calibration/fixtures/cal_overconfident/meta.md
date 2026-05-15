# cal_overconfident

DETR-style bimodal scores: two classes, 30 low-tail detections in
`[0.0, 0.1]` (all wrong) plus 30 high-confidence detections in
`[0.9, 1.0]` with ~50% accuracy.

- Exercises **P1** (quantile binning collapses cleanly on the bimodal
  distribution; equal-width would produce empty middle bins).
- Exercises **P3** (`min_score=0.05` drops roughly half the no-object
  tail before binning).
- Expected ECE is large (~0.4): the high-confidence cluster reports
  ~0.95 mean score with ~0.5 accuracy.
