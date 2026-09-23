# ADR-0064: The parsed pair on every instance surface

- **Status:** proposed
- **Date:** 2026-09-22
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

ADR-0063 chose to return `(CocoDataset, DetectionsInput)` rather than a
metric, and rested that choice on composition: "One conversion, every
downstream surface." It then listed, honestly, the five places where
that was not yet true —

> TIDE, LRP, the confusion matrix, the FP-IoU histogram and the
> `tables=` / `manifest=` paths each refuse a `CocoDataset` handle today
> and ask for GT JSON bytes. That limitation predates this route and is
> theirs to lift.

— which is the gap this ADR closes. It is a bounded, mechanical gap, but
leaving it open costs more than it looks: a trainer that adopts the
ingest route gets AP and then has to serialize a COCO file anyway the
first time it wants to know *why* the AP moved. That is the one moment
the diagnostics exist for, and it is the one moment the route does not
reach.

The same five surfaces also take `dt` as `bytes` alone. The ingest route
produces an `(N, 7)` matrix or columnar `Detections` — never bytes — so
widening only the ground-truth side would leave the pair still
unusable. Both halves are in scope.

### What the boundary actually is

The refusals read like a design constraint. They are not. Every one of
them is a Python-level `raise NotImplementedError` in front of Rust that
was already shaped to take the parsed forms:

- **TIDE, the FP-IoU histogram, LRP, the confusion matrix and
  partitioned LRP.** Each `#[pyfunction]` takes `&Bound<PyBytes>`
  twice, calls `parse_gt` / `parse_dt` inside `py.detach`, and hands
  `&CocoDataset` + `&CocoDetections` to `vernier_core`. Those are the
  same two types `Evaluator.evaluate` produces. The FFI is bytes-only
  in its *signature* and parsed-pair everywhere below it; there is no
  bytes-shaped assumption to unpick.
- **`tables=`.** Vestigial outright. `_evaluate_with_tables` dispatches
  to `evaluate_*_grid`, which has accepted `bytes | CocoDataset` since
  ADR-0061, and then reads `grid.dataset()` — the handle the grid
  retains either way — to feed `per_class_to_arrow_pycapsule` and
  `per_image_to_arrow_pycapsule`, both of which take a `CocoDataset`
  and nothing else. The guard sits above code that has been
  handle-native since ADR-0020. It outlived the limitation it
  described.
- **`manifest=`.** One call site. `evaluate_instance_partitioned_impl`
  invokes `evaluate_grid_impl` — the bytes-only inner function — where
  its non-partitioned sibling invokes `evaluate_grid_any_gt`, the
  dispatcher that ADR-0061 added for exactly this. The partitioning
  itself only needs the grid's `image_id -> index` map, which it reads
  off the returned grid's dataset snapshot.

So the mechanism is present, reachable and already carries the parity
weight: `prepare_dt_payload` / `build_update_payload` / `realize_dt`
resolve the `dt` union for the streaming, background and foreground
evaluators, and `PyDataset::dataset_ref` hands out the parsed ground
truth. Nothing here needs inventing. What is missing is the wiring, and
a decision about the one case where the two spellings are *not*
interchangeable (below).

This ADR triggers ADR-0001 §"Affect the public API" and
§"Cross the FFI boundary".

### Out of scope

- **Any change to an evaluated number**, with one exception: the
  plain summary path now applies the ADR-0026 AC2 trim to a federated
  handle, as the grid path always has (§"Federated ground truth"). The
  parity contract, the disposition table and every quirk stay put.
- **`num_threads=`** on the four standalone diagnostics. They have never
  taken it; ADR-0047's policy is resolved per entry point and adding it
  is an independent change.
- **`evaluate_to_partial` / `from_partials`** (ADR-0031), which keep
  `gt: bytes`. The partial wire format is built by the streaming
  constructor, which owns its own ingest; it is not one of the five.
- **Keypoints on TIDE and on the confusion matrix**, still refused per
  ADR-0024. Widening `gt` does not revisit which kernels a surface
  supports.

## Decision drivers

- **ADR-0030 §"Extend, do not fork".** "Two ingest paths producing
  identical results is the maintenance ceiling. Three would not be."
  Whatever lands here must reduce to the two representations the
  evaluator already consumes, before any core code runs.
- **ADR-0061.** The shape of this widening is already decided: one `gt`
  parameter accepting `bytes | CocoDataset`, not a parallel
  `_with_dataset` function family. ADR-0061 retired that family
  precisely because "its gaps were invisible until a caller needed one
  — twice in two releases". Re-introducing it for the diagnostics would
  be the same mistake with a different prefix.
- **ADR-0063's claim must be true or withdrawn.** Three published
  documents currently state the limitation as fact. Either the
  limitation goes or the documents do; a third state, where the code
  works and the docs still refuse, is worse than both.
- **ADR-0057 §"Validation must refuse, never repair".** One input
  combination becomes *reachable* for the first time through this
  change (a federated LVIS handle on a diagnostic), and it must refuse
  rather than produce a plausible number.

## Considered options

1. **Status quo** — keep the five guards; tell trainers to write a COCO
   file when they want a diagnostic.
2. **Widen the existing entry points** to take the same `gt` and `dt`
   unions `Evaluator.evaluate` takes, reusing its resolvers.
3. **Add `*_with_dataset` siblings** to each diagnostic FFI function,
   as the evaluators had before ADR-0061.
4. **Serialize Python-side** — have the Python wrapper turn a
   `CocoDataset` back into JSON bytes when one is passed.

## Decision outcome

Chosen option: **option 2 — widen the existing entry points.**

Concretely:

- `gt` becomes `bytes | CocoDataset` on `error_decomposition`,
  `fp_iou_histogram`, `optimal_lrp` (both the plain and the `manifest=`
  form), `confusion_matrix`, and `Evaluator.evaluate(tables=...)` /
  `(manifest=...)`.
- `dt` becomes `DetectionsInput` on the same five — the identical union
  `Evaluator.evaluate` and `evaluate_*_grid` accept: `bytes`, columnar
  `Detections` (ADR-0030), result dicts or an `(N, 7)` matrix
  (ADR-0057).
- The five Python `NotImplementedError` guards are deleted.

One new piece of FFI plumbing, and it *replaces* code rather than
sitting beside it:

```rust
pub(crate) enum GtPayload {
    Bytes(PyBackedBytes),
    Parsed(DatasetSnapshot),
}
```

`GtPayload::extract` classifies the Python argument under the GIL;
`GtPayload::realize` produces a `DatasetSnapshot` inside `py.detach`,
parsing in the `Bytes` arm and moving the snapshot in the `Parsed` one.
It is the ground-truth mirror of `UpdatePayload`, which has done the
same job for `dt` since ADR-0030.

**It is the only classifier of the `gt=` union.** `evaluate_grid_any_gt`
and `evaluate_summary_any_gt` carried the same `cast::<PyBytes>()` →
`cast::<PyDataset>()` → `PyTypeError` ladder inline; they now `match` on
`GtPayload` and hand each arm's payload to their per-spelling
implementation, which had been re-deriving exactly that payload on entry
(`PyBackedBytes::from(gt_json.clone())` in one, `gt.snapshot()` in the
other). `BackgroundEvaluator`'s constructor, which carried a third
ladder and parsed bytes under the GIL, uses it too. So the accepted-type
list and its error message exist once, the implementations get shorter,
and `evaluate_summary_impl` — the
plain `Evaluator.evaluate` path, not one of the five — loses the
`gt_json.as_bytes().to_vec()` it was still paying, the same ~20 MB per
val2017 call the diagnostics lose.

`Parsed` carries a `DatasetSnapshot` — the dataset *and* its per-kernel
caches — because every consumer uses both halves (§"The handle's GT
caches").

The `dt` side adds no new machinery: `prepare_dt_payload` is widened to
`iou_type: impl Into<ArrayIouType>`, so the evaluators keep passing an
`EvalIouType` and the diagnostics pass the `"bbox"` / `"segm"` /
`"boundary"` / `"keypoints"` marker their per-kernel entry point already
names — one function, both spellings.

The two halves meet in `DiagnosticInputs`, which is what the four
standalone diagnostics actually call:

```rust
impl DiagnosticInputs {
    fn extract(py, gt, dt, kernel, cast_inputs, surface) -> PyResult<Self>;
    fn realize(self) -> PyResult<(DatasetSnapshot, CocoDetections)>;
}
```

The federated refusal below is folded into `extract`, so a sixth
diagnostic cannot obtain its inputs without being checked.
`realize` derives detection areas from the box
(`DetectionArea::FromBbox`), the quirk **J3** default on every non-grid
route, `Evaluator.evaluate` included; `dt_area="mask"` stays a grid knob.

`tables=` and `manifest=` need no FFI work beyond a call-site swap:
deleting the guard, and pointing `evaluate_instance_partitioned_impl`
at `evaluate_grid_any_gt` instead of `evaluate_grid_impl`.

### `cast_inputs` comes along

The four standalone diagnostics gain `cast_inputs: bool = False`,
matching `Evaluator`'s field and its default. Without it the union would
be nominally accepted and practically not: a caller holding an `f32`
detection matrix could evaluate it but could not decompose it, and the
error would arrive from a layer they did not call. ADR-0063's route is
unaffected either way — `coco_inputs` casts at build time, so what it
returns is already `f64`.

The default stays `False` here. ADR-0063's `cast_inputs=True` was
justified by *that* function's population — callers handing over tensors
straight from a forward pass — and was recorded as "a deliberate,
recorded divergence, confined to one function". It stays confined.

One cadence difference is worth naming: `Evaluator` holds `cast_inputs`
as a dataclass field, so the one-shot promotion `UserWarning` fires once
per evaluator. These are standalone functions, so the latch is built per
call and the warning is effectively per call. Same mechanism, noisier
rhythm — which is an argument for passing `f64`, not for suppressing it.

### Federated ground truth is refused, loudly

This is the one place where widening a type changes what is reachable
rather than only how it is spelled, and it is the reason this is an ADR
and not a chore.

`CocoDataset.from_json` discards LVIS federated metadata; `from_lvis_json`
retains it. That asymmetry is documented on `evaluate_grid_any_gt`: for
LVIS the handle is "not merely the faster spelling — it is the only
correct one". The evaluate paths honour it fully, including the ADR-0026
AC2 per-image detection trim.

That was true of the grid path only. The plain summary path —
`Evaluator.evaluate` with neither `tables=` nor `manifest=` — matched a
federated handle without trimming, so lifting the `tables=` / `manifest=`
guards would have let one handle score differently depending on which
keyword was passed. Both handle paths now apply the trim, capping at
the ladder's largest `max_dets` as the grid does (the streaming
evaluators follow in ADR-0065). The
fix moves a number only when an image carries more detections than that
cap, and moves it toward lvis-api.

The diagnostics have no trim, and no oracle. `vernier_core`'s
matching pass would apply the AA3/AA4 federated branches (it reads
`gt.federated()` directly), while the LVIS per-image detection cap would
not be applied — half of the LVIS semantics, silently. There is no
published TIDE-on-LVIS or LRP-on-LVIS convention to check the other half
against, and ADR-0026's acceptance criteria are written for AP.

So a federated handle on `error_decomposition`, `fp_iou_histogram`,
`optimal_lrp` or `confusion_matrix` raises `NotImplementedError` naming
ADR-0026. This is strictly more conservative than today: before this
change the input was unreachable, so nothing regresses, and the
alternative — accepting it — would ship exactly the "plausible, wrong
number" ADR-0057 rules out. `tables=` and `manifest=` route through the
grid and are therefore *not* restricted; every `Evaluator.evaluate`
branch applies the same federated semantics.

Lifting this needs its own ADR, with an oracle, which is the point.

### The handle's GT caches

ADR-0020's handle saves two costs: the JSON parse, and the per-annotation
segm / boundary derivations memoised on `SegmGtCache` /
`BoundaryGtCache`. The diagnostics get both, with no `vernier-core` API
change: every diagnostic already has a kernel-generic entry point
(`error_decomposition_with`, `compute_fp_iou_histogram_with`,
`optimal_lrp_with[_partitioned]`, `compute_confusion_matrix`), and the
cached kernels are public (`SegmIouCached::with_arc_cache`,
`BoundaryIouCached::with_arc_cache`). `DatasetSnapshot::segm_kernel` /
`boundary_kernel` build them over the snapshot's caches.

- **Bit-exact.** The cached and uncached kernels call one compute
  function; the cache only skips re-deriving a GT's RLE geometry.
- **Valid across TIDE's passes.** Each fix pass rebuilds the dataset,
  but `apply_fix` keeps every surviving GT's id and geometry (it only
  relabels or drops detections, or drops missed GTs), so an id-keyed
  entry stays correct.
- **Bytes too.** `DatasetSnapshot::from_parsed` gives the bytes path
  fresh caches, so TIDE's eight passes share one derivation per GT.

The core per-kernel wrappers (`error_decomposition_segm`,
`optimal_lrp_boundary`, …) get the same treatment for Rust callers: the
scratch-reusing kernels, and a call-local cache for TIDE.

**The `gt` union gets no type alias.** `dt` has one —
`DetectionsInput` in `python/vernier/_array_types.py` — so widening it
was a one-token edit per signature, while `bytes | CocoDataset` is
spelled out across the wrappers, the overloads and the stub. A
`GroundTruthInput` alias would restore the symmetry, and the stub's
conformance test checks shape rather than types so it could not break
one; but it is a new name on the public surface, which ADR-0001 makes
an ADR-level decision rather than a drive-by inside this one. Recorded
as the obvious follow-up.

**No surface gains a kernel it did not have.** Keypoints stays refused
on TIDE and on the confusion matrix (ADR-0024); `use_cats=False` stays
refused on the confusion matrix. Widening `gt` and `dt` is orthogonal to
which kernels a diagnostic supports, and conflating the two would hide a
capability change inside a plumbing change.

### Consequences

- **Positive.** ADR-0063's central claim becomes true without
  qualification: one conversion, every instance surface. A training loop
  can go from "AP dropped" to a TIDE decomposition of *why* without
  materializing a COCO file, which is the whole reason the route
  returns inputs.
- **Positive.** The `dt` union on these paths is the same union the
  evaluator takes, so a caller who already builds detections as arrays
  for `evaluate` does not switch representations to call
  `error_decomposition` on the same run.
- **Positive.** A handle's mask derivations are shared by the evaluator
  and all four diagnostics, and TIDE reuses them across its eight passes
  on either spelling.
- **Positive.** The `to_vec()` per call on the GT bytes path disappears
  — `PyBackedBytes` borrows instead. On a val2017-shaped payload that is
  ~20 MB of copy per call. Unifying the classifier extends that to
  `evaluate_summary_impl`, which is the plain `Evaluator.evaluate` path
  and the most-used surface in the library; its in-line justification
  for the memcpy ("buys a wider GIL-drop window") had been stale since
  `evaluate_grid_impl` switched to `PyBackedBytes`, which gives the same
  window without the copy.
- **Negative.** Five surfaces gain a `cast_inputs` keyword, which is
  five more places the dtype question can be asked. The answer is
  uniform (`False`, as everywhere but ADR-0063's route), which is the
  mitigation.
- **Negative.** A federated handle now fails at a *later* point than a
  reader might expect — at the diagnostic rather than at
  `from_lvis_json` — with a refusal that reads as a gap. It is one, and
  the error says so.
- **Neutral.** The Python guards' disappearance changes an exception
  type for anyone who was catching `NotImplementedError` to fall back to
  bytes. Nobody should be, and the fallback still works.

## Pros and cons of the options

### Option 1 — status quo

- 👍 Zero change; zero risk to the parity contract.
- 👎 Leaves ADR-0063's decision resting on a claim that is two-thirds
  true, and leaves three shipped documents describing a limitation
  whose cause is a `raise` statement.
- 👎 The workaround — serialize a COCO file — is precisely the ~250
  lines of conversion ADR-0063 exists to delete, reintroduced at the
  first interesting question.

### Option 2 — widen the existing entry points (chosen)

- 👍 Reuses the resolvers the evaluator uses, so a diagnostic cannot
  disagree with `evaluate` about what an input means. ADR-0030
  §"Extend, do not fork" is satisfied by construction rather than by
  review.
- 👍 One `gt` parameter, per ADR-0061 — no second function family whose
  gaps only surface when someone needs one.
- 👍 Cannot move a diagnostic's number: the resolvers yield the types
  core took before, and the cached kernels are bit-exact.
- 👎 Five `#[pyfunction]` signatures widen from `PyBytes` to `PyAny`,
  which moves a type error from compile time in Rust to runtime in
  Python. Mitigated by `GtPayload::extract` being the single classifier
  — one accepted-type list and one message for every `gt=` on the
  instance surface — and by `_core.pyi` carrying the precise union.

### Option 3 — `*_with_dataset` siblings

- 👍 No existing signature changes; the bytes paths keep a `PyBytes`
  argument and its compile-time guarantee.
- 👎 ADR-0061 retired exactly this pattern for the evaluators, and gave
  the reason: the family's gaps are invisible until a caller hits one.
  Four kernels x four diagnostics is sixteen places to forget.
- 👎 Doubles the surface `_core.pyi` must mirror and the conformance
  test must check, for no user-visible capability the union does not
  give.

### Option 4 — re-serialize Python-side

- 👍 Smallest diff imaginable; no Rust change at all.
- 👎 Reintroduces the JSON round-trip the handle exists to avoid, and
  makes the "parsed once" surface quietly parse twice.
- 👎 `CocoDataset` has no `to_json`, and adding one to enable a
  workaround would be a worse public surface than the fix.
- 👎 Would silently *lose* LVIS federated metadata on the way out,
  turning the refusal this ADR makes explicit into a wrong number.

## Links and references

- Related ADRs: [0020](0020-parsed-once-dataset-handle.md) (the handle
  and its caches), [0021](0021-tide-oracle.md),
  [0023](0023-tide-cross-class-strategy.md) (the confusion matrix),
  [0026](0026-lvis-support.md) (the federated semantics refused here),
  [0030](0030-buffer-protocol.md) ("extend, do not fork"; the `dt`
  union), [0043](0043-lrp-oracle-and-namespace.md),
  [0046](0046-slice-and-aggregate.md) (the `manifest=` path),
  [0057](0057-python-detection-ingest-routes.md) (refuse, never
  repair), [0061](0061-one-ground-truth-parameter-keyword-only-options.md)
  (one `gt` parameter; no `_with_dataset` family),
  [0063](0063-per-sample-ingest-route.md) (the claim this makes true).
