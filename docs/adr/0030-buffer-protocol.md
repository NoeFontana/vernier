# ADR-0030: Accept detection arrays alongside JSON bytes in streaming update

- **Status:** accepted
- **Date:** 2026-05-04
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

`StreamingEvaluator.update(bytes)` (ADR-0013) and `BackgroundEvaluator.submit(bytes)` (ADR-0014) accept detections as `loadRes`-shaped JSON bytes. This was the right call for ADR-0013: it kept the streaming surface bit-identical to `Evaluator.evaluate`, gave us a single parser, and matched how downstream tools (`pycocotools`, evaluation servers, dataset tooling) hand detections around.

But the training-loop persona that motivated ADR-0014 does not have JSON in hand. They have a model output: numpy arrays, or torch CPU tensors, already laid out as `boxes (N,4) f64`, `scores (N,) f64`, `labels (N,) i64`, plus a per-image-id integer. To submit those today they must:

1. Convert each detection into a Python dict.
2. `json.dumps` the resulting list.
3. `.encode()` the string to bytes.
4. Hand the bytes to `update`, which immediately parses them back into structs.

For a per-step submit cadence on a 1k–5k-detection batch this is between 10 and 100 ms of pure overhead per call. Profiles on the perception team's training rigs show the encode/parse round-trip dominating `update` wall-clock — the matching kernel itself is in the noise. The streaming surface was supposed to take eval off the critical path; the JSON tax is putting it back.

This ADR adds a parallel ingest path on the existing classes that accepts numpy arrays and any DLPack-compatible CPU tensor (torch CPU, jax CPU, cupy host buffers) directly, with no JSON intermediate. It does **not** introduce new evaluator types; the streaming and background classes (and their snapshot/finalize/checkpoint contracts) are unchanged. The change lands at the existing internal `ParsedDetections<K>` seam.

This ADR triggers ADR-0001 §"Affect the public API" (the `update` and `submit` signatures gain an overload) and §"Cross the FFI boundary" (a new ingest path).

### Out of scope

- GPU-resident inputs (DLPack tensors with `device_type != kCPU`). Rejected with a typed error in this ADR; tracked as future work below.
- Cross-process/cross-rank aggregation. Independent concern.
- Changes to `Evaluator.evaluate` (the frozen batch type). Confined to streaming surfaces.
- Extended metrics surface (per-class, custom thresholds). Owned by ADR-0016 / ADR-0019.

## Decision drivers

- **Skip JSON, not threading or parity.** The win is avoiding encode + parse. The threading model (ADR-0006: GIL drop on entry), the parity contract (ADR-0002, ADR-0004), and the streaming algebra (ADR-0013) all stay exactly as they are.
- **Extend, do not fork.** Two ingest paths producing identical results is the maintenance ceiling. Three would not be. The array path must reduce to the existing `ParsedDetections<K>` representation before any downstream code runs.
- **No silent dtype coercion.** ADR-0004 and ADR-0008 pin `f64` at the boundary for parity reasons. Accepting `f32` and silently promoting it would surface as parity drift in downstream ADR-0002 strict-mode tests, which is exactly the failure mode we want to avoid.
- **Single-writer and budget contracts unchanged.** The owner-thread rule (ADR-0013), the queue-full back-pressure (ADR-0014), and `OutOfBudgetError` semantics carry through verbatim. The array path is a different way to hand bytes-equivalent data to the same machinery.
- **Match existing protocol coverage.** numpy arrays via `rust-numpy` for the read-only-aware ergonomic path; DLPack for everything else (torch ≥1.13, jax, cupy host buffers). The dispatch is one `FromPyObject` enum per logical input, ~30 lines.

## Considered options

1. **Status quo — JSON bytes only.** Users encode arrays to JSON before calling `update`. Maintenance burden zero, but leaves the training-loop tax on the table.
2. **Replace the JSON path with arrays.** Forces every existing user (CLI, evaluation servers, anything reading `loadRes` files from disk) to adopt array encoding. Breaks the `pycocotools` migration story (ADR-0007). Non-starter.
3. **Add an array path alongside the JSON path on the existing classes.** Both forms land at `ParsedDetections<K>`. Single matching code path downstream. Public surface gains one overload per method; no new types, no new lifecycle.
4. **Add a sibling `ArrayStreamingEvaluator` / `ArrayBackgroundEvaluator`.** Cleanly separates the two ingest forms but doubles the type surface, the test matrix, and the documentation burden, for an input-format difference that's invisible after the first 50 lines of the call.

## Decision outcome

Chosen option: **option 3 — add an array path alongside the JSON path on the existing classes.**

Justification: the matching kernel, accumulate logic, snapshot/finalize lifecycle, single-writer rule, memory budget, and parity contract are all input-format-independent. A new class would re-export those properties unchanged. The honest description of the change is "another way to construct `ParsedDetections<K>`," which is a method overload, not a type.

### Surface

`StreamingEvaluator.update` and `BackgroundEvaluator.submit` accept either bytes (existing) or a `Detections` payload (new). Both produce identical `UpdateReport` / queue semantics.

```python
class Detections(TypedDict, total=False):
    image_id: int                       # required
    boxes: NDArray[np.float64]          # (N, 4), xywh, contiguous — required for bbox / segm
    scores: NDArray[np.float64]         # (N,) contiguous — required
    labels: NDArray[np.int64]           # (N,) contiguous — required
    # iou_type-specific:
    rles: Sequence[RLE]                 # for iou_type in {"segm", "boundary"}
    keypoints: NDArray[np.float64]      # (N, K, 3) for iou_type == "keypoints"

class RLE(TypedDict):
    counts: NDArray[np.uint32]          # contiguous
    size: tuple[int, int]               # (height, width)
```

Updated method signatures:

```python
class StreamingEvaluator:
    def update(self, detections: bytes | Detections | Sequence[Detections]) -> UpdateReport: ...

class BackgroundEvaluator:
    def submit(
        self,
        detections: bytes | Detections | Sequence[Detections],
        *,
        timeout: float | None = None,
    ) -> None: ...
```

The `Sequence[Detections]` form covers multi-image batches in a single call, matching the existing JSON `loadRes` shape (which is a flat list across images). A single `Detections` is the per-image case and is the form that drops out naturally from a model forward.

### Internal landing point

The new path constructs `ParsedDetections<K>` directly from validated array views, bypassing `CocoDetections::from_json_bytes`:

```rust
impl<K: EvalKernel> ParsedDetections<K> {
    pub fn from_arrays(views: DetectionArrayViews<'_>) -> Result<Self, EvalError> { ... }
}
```

`StreamingEvaluator::update` (Rust) gains a sibling `update_arrays` that calls `ParsedDetections::from_arrays` then `update_parsed` — the same final method the existing JSON path uses (ADR-0013, ADR-0014). The Python FFI layer dispatches on the input type and routes to the right Rust entry point. Every code path past `ParsedDetections<K>` is shared and unchanged.

### Validation rules at the boundary

| Field | Required dtype | Layout | On mismatch |
|---|---|---|---|
| `boxes` | `float64` | `(N, 4)` C-contiguous | `TypeError` naming `np.ascontiguousarray` and `astype(np.float64)` as the fix |
| `scores` | `float64` | `(N,)` contiguous | same |
| `labels` | `int64` | `(N,)` contiguous | same |
| `rles[i].counts` | `uint32` | contiguous | same |
| `keypoints` | `float64` | `(N, K, 3)` C-contiguous | same |

`f32` is rejected, not silently promoted, in line with ADR-0004 and ADR-0008. A documented opt-in `cast_inputs=True` constructor flag promotes-and-copies with a one-shot `UserWarning`, for users who genuinely want the convenience. Default off.

DLPack tensors with `device_type != kCPU` raise `TypeError("vernier-0030 does not accept GPU-resident detections; move to CPU with .cpu() or .to('cpu')")`. The error is explicit so the message is greppable.

### Mask handling

`iou_type="bbox"` ignores all mask fields. `iou_type="segm"` and `"boundary"` require `rles`. `iou_type="keypoints"` requires `keypoints`.

We do **not** accept polygons or HxW bitmasks on the detection-array path. Reasoning:

- **GT polygons are not detection-side.** `CocoDataset` parses GT once at construction (gt_bytes), and any polygon-in-GT is rasterized to RLE there (ADR-0020 territory). Detections never carry GT polygons.
- **DT polygons are not a real workflow.** No production detection or instance-segmentation model emits polygons; they emit per-pixel masks that downstream code RLE-encodes. The COCO `loadRes` JSON shape itself only specifies RLE for detection masks.
- **DT bitmasks are bulky and ambiguous at the boundary.** A `(H, W) bool` array per detection would dominate the per-image submit cost (a single 640×480 mask is 300 KB raw), and the right thing to do with it is RLE-encode it — which the user can do once at the dataloader boundary with `pycocotools.mask.encode` or `vernier.mask.encode`, in parallel with everything else dataloader workers do. Pulling that into the eval ingest path adds work to the critical path.

If a user has bitmasks and wants to skip the encode, the right place to add that affordance is `vernier.mask.encode_batch` returning a `Sequence[RLE]` shape this surface accepts — an additive helper, not an evaluator overload. Out of scope here.

### Memory ownership

Identical to the existing JSON path:

- **Sync `update`.** Validate views under the GIL, copy into a `ParsedDetections<K>` (owned `Vec<f64>` etc.), drop GIL, run the match. The user owns their arrays after `update` returns; we have our own copy. Same memcpy size as the JSON path's `bytes.to_vec()`, ~10 KB typical, dwarfed by the FFI dispatch cost.
- **Async `submit`.** Validate + copy into `ParsedDetections<K>` under the GIL, push across the channel. Worker thread runs match on its own copy. Identical lifetime story to the JSON `bytes.to_vec()` path on the same call site.

This was the misframing in the prior draft of this ADR: "zero-copy" overstates the win. The win is skipping JSON encode + JSON parse. The single ingest memcpy is unchanged from today and is fixed by ADR-0006, not negotiable here.

### Parity

Per-image matching, accumulate, snapshot/finalize, and the single-writer rule are unchanged. The `ParsedDetections<K>` produced by `from_arrays` is byte-identical to what `from_json_bytes` produces for the same logical detections (same field types, same ordering rules). CI gains a parity test that runs the COCO val2017 corpus through both ingest paths and asserts byte-equal `Summary.stats` arrays in both `parity_mode="strict"` and `parity_mode="corrected"`. This sits next to the existing streaming-vs-batch parity tests in `tests/python/parity/streaming/`.

### Consequences

- **Positive.** Eliminates JSON encode + parse from the streaming hot path, which was the dominant cost in training-loop profiles. No change to the streaming algebra, parity contract, single-writer rule, or memory budget. One new public overload per method; existing callers untouched.
- **Negative.** The public surface gains a typed-dict shape (`Detections`) that has to be documented, type-stub-tested, and kept in lockstep with kernel additions (a future iou_type ships with both a JSON shape and an array shape). Reviewers must enforce dtype strictness on the array path, where the JSON path was permissive about source dtypes (it parsed all numbers as f64 anyway).
- **Neutral.** The array path is opt-in. Existing users get exactly the surface they had; new users get one more option. Documentation cost is bounded — a single how-to under `docs/how-to/array-ingest.md` cross-linked from the streaming and background guides.

## Future work

- **GPU-resident detections (separate ADR).** DLPack already carries device info; the dispatch site can route GPU tensors to a future CUDA matching kernel without changing this Python surface. The current ADR rejects GPU inputs explicitly so the future extension does not break callers.
- **`vernier.mask.encode_batch` helper for users with bitmask outputs.** Additive, kernel-internal, no surface impact on the evaluators.
- **`from_arrays` parity oracle test against `from_json_bytes`.** Lands as part of this ADR's CI; called out here so it doesn't get lost in implementation.

## Links and references

- ADR-0001 — record architecture decisions (ADR significance criteria).
- ADR-0002 — three-tier parity model.
- ADR-0004 — numerical layout policy (f64 boundary).
- ADR-0006 — threading model (GIL drop on entry).
- ADR-0008 — bbox IoU computes in f64 end-to-end.
- ADR-0011 — discriminated kernel config.
- ADR-0013 — streaming evaluator (the `update` / `snapshot` / `finalize` lifecycle this ADR extends).
- ADR-0014 — `BackgroundEvaluator` (the `submit` lifecycle this ADR extends).
- ADR-0020 (proposed) — parsed-once `Dataset` handle (where GT polygon → RLE rasterization lives).
- DLPack specification: <https://dmlc.github.io/dlpack/latest/>
