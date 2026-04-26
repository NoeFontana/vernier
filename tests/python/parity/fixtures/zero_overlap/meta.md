# zero_overlap

One GT, one DT, disjoint bboxes.

- IoU = 0.0
- DT is unmatched (FP) at every iouThr
- recall = 0.0; precision = 0.0 at every recall threshold

Tests the "no match anywhere" path through the accumulator: `tp_sum` is all
zeros, `fp_sum` ends at 1, the descending-precision sweep does nothing, and
the 101-point integration produces a row of zeros (not -1 sentinel — that's
reserved for absent-category cells).
