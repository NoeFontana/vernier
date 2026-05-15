# cal_keypoints_smoketest

Identical detection cells to `cal_overconfident`, tagged
`iou_type="keypoints"` and `max_dets=20` in `meta.json`.

- Exercises the keypoints footnote in ADR-0018 §"Shape 1": calibration
  reads OKS-matched cells the same way it reads IoU-matched ones. The
  canonical `max_dets=[20]` cap is applied **upstream** by the streaming
  evaluator before cells reach the oracle — this fixture does not
  exercise the cap directly (the cell input already reflects it).
- No `small`-area-bucket interaction (calibration does not bucket by
  area). Expected ECE matches `cal_overconfident`'s.
