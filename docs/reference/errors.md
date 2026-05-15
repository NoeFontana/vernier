# Error reference

Vernier surfaces evaluation errors as typed Python exception classes
re-exported from each paradigm namespace (`vernier.instance`,
`vernier.panoptic`, `vernier.semantic`). Each is a subclass of either
`ValueError` (data / config errors the caller could have prevented) or
`RuntimeError` (state-machine or coordination errors that arise
mid-run). The stringified message is the canonical, human-readable
explanation produced by the Rust [`EvalError`](../../crates/vernier-core/src/error.rs)
`Display` impl; attributes carry structured data callers may branch on.

| Python class | Rust variant | Base | Trigger |
|---|---|---|---|
| `InvalidAnnotationError` | `InvalidAnnotation` | `ValueError` | GT annotation references unknown image_id / category_id; DT has malformed segmentation or keypoints |
| `NonFiniteError` | `NonFinite` | `ValueError` | NaN / inf in detection score or coordinate |
| `DimensionMismatchError` | `DimensionMismatch` | `ValueError` | RLE / mask geometry disagrees; partial table cross-axis shape mismatch |
| `InvalidConfigError` | `InvalidConfig` | `ValueError` | unrecognized field on `EvaluateParams`, bad manifest version, IoU threshold absent from ladder |
| `OutOfBudgetError` | `OutOfBudget` | `RuntimeError` | streaming evaluator's `memory_budget_bytes` exceeded |
| `QueueFullError` | (FFI-local) | `RuntimeError` | `BackgroundEvaluator.submit(timeout=0.0)` on a full channel |
| `PartialFormatMismatch` | `PartialFormatMismatch` | `RuntimeError` | partial blob magic / version / CRC / kernel kind rejected |
| `PartialDatasetMismatch` | `PartialDatasetMismatch` | `RuntimeError` | partial's `dataset_hash` doesn't match the live dataset |
| `PartialParamsMismatch` | `PartialParamsMismatch` | `RuntimeError` | partial's `params_hash` doesn't match this evaluator's |
| `PartialPartitionOverlap` | `PartialPartitionOverlap` | `RuntimeError` | two ranks evaluated the same image (sampler bug) |
| `PartialRankCollision` | `PartialRankCollision` | `RuntimeError` | two strict-mode partials declare the same `rank_id` |
| (generic `ValueError`) | `Json` | `ValueError` | input GT / DT bytes are not valid JSON |
| (generic `ValueError`) | `Mask` | `ValueError` | mask codec / polygon rasterization failed |
| (generic `ValueError`) | `LvisFederatedConflict` | `ValueError` | LVIS GT puts a category in both `not_exhaustive` and `neg` for one image |
| (generic `ValueError`) | `MissingFrequency` | `ValueError` | LVIS GT category records lack a `frequency` field |
| (generic `ValueError`) | `PerPairOverflow` | `ValueError` | `per_pair` table row count exceeded `per_pair_max_rows` |
| `NotImplementedError` | `NotImplemented` | (built-in) | feature wired but not yet implemented in v0 |
