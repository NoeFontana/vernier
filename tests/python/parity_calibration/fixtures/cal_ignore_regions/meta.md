# cal_ignore_regions

Two classes, 20 detections each, every other detection flagged
`dt_ignore[t, d] = True` at every threshold.

- Exercises **P2 / R3**: ignored detections must drop from the
  histogram entirely (neither TP nor FP). The post-filter detection
  count is half the input count.
- Sanity assertion: `n_detections` reported by the oracle equals
  exactly 20 (10 per class survived the ignore filter).
