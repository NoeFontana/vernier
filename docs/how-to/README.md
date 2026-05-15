# How-to guides

Task-oriented recipes for users who already know what they want to
accomplish.

- [Submit detections as numpy arrays or DLPack tensors](array-ingest.md) —
  skip JSON for tight model→evaluator loops.
- [Evaluate on a background thread](background-evaluator.md) —
  `BackgroundEvaluator` for training loops where the kernel measurably
  stalls the main thread.
- [Evaluate with boundary IoU](boundary-iou.md) — the `Boundary()`
  kernel and its dilation knob.
- [Evaluate with `vernier eval`](cli-eval.md) — the static CLI binary
  for CI pipelines without a Python interpreter.
- [Configure the evaluator](configure-evaluator.md) — the entry-point
  survey of `iou` / `parity_mode` / `max_dets` / `use_cats` /
  `cast_inputs` and how they compose.
- [Custom evaluation grids](custom-evaluation-grids.md) — the
  ADR-0040 axes (`iou_thresholds`, `recall_thresholds`,
  `area_ranges`) and the `evaluate_tables` route.
- [Distributed evaluation across ranks](distributed-eval.md) —
  rank-local + gather across instance / semantic / panoptic.
- [Evaluate keypoints with OKS](keypoints-oks.md) — `Keypoints()` with
  per-category sigmas.
- [Per-class metrics broken down by manifest slice](per-class-by-slice.md) —
  the recipe for the `tables=` + `manifest=` cross product (deliberate
  non-feature; the kwargs reject in combination).
- [Use result tables](result-tables.md) — the per-image / per-class /
  per-detection / per-pair polars DataFrames.
