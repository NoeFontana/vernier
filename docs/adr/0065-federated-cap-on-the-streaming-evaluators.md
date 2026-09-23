# ADR-0065: Apply the LVIS per-image cap per streaming batch

- **Status:** proposed
- **Date:** 2026-09-23
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

ADR-0026 AC2 caps an LVIS evaluation at `max_dets` detections per image,
across categories, before matching: lvis-api's `LVISResults` applies
`limit_dets_per_image` when it builds the detection set. vernier's batch
paths apply it through `CocoDetections::lvis_trim` whenever the ground
truth is federated (since ADR-0064, on every `Evaluator.evaluate`
branch).

The streaming evaluators did not. `StreamingEvaluator` — and
`BackgroundEvaluator`, which drives one on a worker thread — accept a
`CocoDataset.from_lvis_json` handle, apply the federated matching
branches (AA3/AA4, read off the dataset), and skip the cap. An image
carrying more than `max_dets` detections therefore scores differently
streamed than batched, and differently from lvis-api.

Streaming receives detections in batches, so the question is whether a
per-image cap can be applied before the whole set is known.

## Decision drivers

- **ADR-0013: streaming is bit-identical to batch.** `finalize()` must
  equal a batch run over the union of the batches.
- **ADR-0057: refuse, never repair — and never ship a plausible wrong
  number.** The status quo does the latter.
- **No new buffering.** The streaming evaluators exist to bound memory
  (ADR-0013's budget); holding detections back until `finalize()` would
  undo that.
- **One definition of the trim.** Batch and streaming must not each
  carry their own "is it federated, what is the cap" logic.

## Considered options

1. **Trim each batch on arrival.**
2. **Buffer detections and trim at `finalize()`.**
3. **Refuse federated ground truth** on the streaming evaluators.

## Decision outcome

Chosen option: **option 1**, because an existing invariant makes it
exact.

`StreamingEvaluator::update` already rejects an `image_id` seen in a
prior batch ("submit all detections for an image in a single batch"),
which is what lets it file each cell once. So every image's detections
arrive together, and the cap — a function of one image's detections
alone — computed per batch is the whole-dataset cap. Distributed ranks
inherit the property: ADR-0031 D1 requires disjoint image sets across
partials.

The trim lives in one core method, `CocoDetections::trim_for(gt,
max_dets)`: `lvis_trim` when `gt` is federated, identity otherwise. Its
callers are whoever assembles a detection set, as `LVISResults` is for
lvis-api — the FFI's batch handle paths, and `StreamingEvaluator`'s
per-update admission step, which covers `BackgroundEvaluator`, the
parallel update path and distributed partials alike. The core batch
functions (`evaluate_bbox`, …) keep taking detections as given, as
`LVISEval` takes an already-built `LVISResults`.

Image ids are recorded before the trim, so an image capped to nothing
still counts as submitted.

### Consequences

- **Positive:** a federated handle scores the same streamed, batched and
  under lvis-api. `finalize()` is again bit-identical to the batch path.
- **Positive:** no memory, API or latency cost; the trim is a no-op on
  flat ground truth.
- **Negative:** a streamed LVIS evaluation's numbers move when an image
  exceeds the cap — as a fix, toward the oracle.
- **Neutral:** `UpdateReport.n_detections_accepted` and
  `detections_seen` count post-trim detections, which is what was
  evaluated.

## Pros and cons of the options

### Option 1 — trim each batch (chosen)

- 👍 Exact, given the one-batch-per-image rule the evaluator already
  enforces.
- 👍 No buffering; the memory budget keeps its meaning.
- 👎 Correctness leans on that rule. Relaxing it (merging an image across
  batches) would need this decision revisited — the trim then needs the
  image's full set.

### Option 2 — buffer until `finalize()`

- 👍 Would not depend on the one-batch-per-image rule.
- 👎 Defers all matching to `finalize()`, removing streaming's reason to
  exist and breaking `snapshot()`.
- 👎 Solves a problem the rule already rules out.

### Option 3 — refuse federated ground truth

- 👍 Smallest change.
- 👎 Removes a working LVIS training-loop path to avoid a fix that costs
  one method call.

## Links and references

- Related ADRs: [0013](0013-streaming-evaluator.md) (streaming, the
  one-batch rule), [0014](0014-background-evaluator.md),
  [0026](0026-lvis-support.md) (AC2), [0031](0031-dist-eval.md)
  (disjoint partials), [0057](0057-python-detection-ingest-routes.md),
  [0064](0064-dataset-handle-on-diagnostic-surfaces.md) (the batch paths).
- External references: `lvis/results.py` `LVISResults.limit_dets_per_image`.
