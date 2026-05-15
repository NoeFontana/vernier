# cal_perfect

Baseline. Three classes, 10 detections each on a clean monotone score
ramp (`np.linspace(0.05, 1.0, 10)`), all matched at every IoU threshold,
all `dt_ignore=False`.

- Expected ECE = `1.0 - mean(linspace(0.05, 1.0, 10))` (every detection
  correct, so the gap collapses to `1 - mean_score`).
- Exercises no quirks beyond the baseline path; if this fails the bin
  bookkeeping is broken.
