# ADR-0031: Distributed evaluation — `from_partials` and the partial wire format

- **Status:** proposed
- **Date:** 2026-05-04
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

Every published `tutorials/training-loop.md` example is single-rank. The page admits it: *"Multi-process training. Running one StreamingEvaluator per rank and reducing summaries at log time is workable but bypasses kernel-level synchronization. The intended pattern is rank-0 evaluation; the multi-rank story is a roadmap item."* For DDP and multi-node training — the dominant deployment shape on the perception team's training rigs — rank-0 evaluation either bottlenecks on a single rank's CPU while N−1 GPUs idle, or forces the user to gather predictions back to rank 0 before submitting, which is exactly the JSON tax ADR-0030 just paid down. Neither is acceptable for the v1.0 narrative.

The shape of the right answer was already telegraphed by ADR-0013 ("users shard manually across processes and combine via `StreamingEvaluator.merge(other)` if they need it") and ADR-0014 ("`merge` API across instances. Out of scope for this ADR; same follow-up as ADR-0013's merge note"). This ADR is that follow-up. It also closes a parallel deferral: `checkpoint()` / `restore()` ship `EvalError::NotImplemented` today, and the obvious wire format for them is the obvious wire format for cross-process merge. One format, one schema, one test surface.

This ADR triggers ADR-0001 §"Affect the public API", §"Cross the FFI boundary", §"Set a project-wide convention" (the wire format is a forward-compat commitment), and §"Add or remove a top-level dependency". The dep trigger merits its own line: an earlier draft of this ADR claimed `rkyv` was already in `vernier-core` (for the RLE codec) and `blake3` was already transitive (via dataset-hash plumbing reserved in ADR-0013). Both claims were wrong on inspection — the mask codec at `crates/vernier-mask/src/codec.rs` is hand-rolled, and the dataset-hash slot in ADR-0013 was never plumbed. So this ADR adds three new top-level workspace dependencies: **`rkyv 0.8`** (with default `bytecheck` features for archived-type validation), **`blake3 1`** (dataset and params fingerprints), and **`crc32fast 1`** (wire-format footer). All three are MIT/Apache, mature, and pass `cargo deny check`. The threading model is untouched: compute stays single-threaded, merge is in-process aggregation of pre-computed cells.

### What "distributed eval" means here

A precise framing matters because two unrelated problems share the name in the literature:

1. **Cross-process aggregation of detection metrics over disjoint image partitions.** Each rank evaluates its own slice of the validation set; the union is what the headline AP describes. This is what every DDP training loop wants. It is what this ADR ships.
2. **Distributed matching across ranks for a single image.** Tile a high-resolution image, match against GTs across the tile boundary. Different problem, different math, different audience. Out of scope. (Notable for closing the door explicitly: vernier's matching engine is per-image and has no cross-image dependencies, so this would be a different system, not a knob on this one.)

### What the user actually does

The DDP integration pattern is the point of the design. From a robotics-team training script:

```python
from vernier.instance import StreamingEvaluator
import torch.distributed as dist

ev = StreamingEvaluator(gt_bytes, iou_type="bbox", rank_id=dist.get_rank())
for batch in val_loader_for_this_rank:
    detections = model(batch["images"])
    ev.update(detections)  # ADR-0030 array path

partial = ev.to_partial()  # bytes; rank-local final state

gathered: list[bytes | None] = [None] * dist.get_world_size()
dist.all_gather_object(gathered, partial)

if dist.get_rank() == 0:
    merged = StreamingEvaluator.from_partials(gt_bytes, gathered, iou_type="bbox")
    summary = merged.finalize()
    log_metrics(summary)
```

Three things to notice. First, vernier ships a bytes interface; `torch.distributed` is the user's problem. There is no `import torch` inside vernier and no torch-version pin to chase. Second, the `rank_id` is the user's responsibility — vernier doesn't try to discover it. Third, `from_partials()` returns a `StreamingEvaluator`, not a `Summary`, so the same instance can be checkpointed, queried for memory state, or further extended; merge is a constructor, not a terminal operation.

### Out of scope

- GPU all-reduce inside vernier. The bytes interface intentionally does not assume the user has a GPU communicator.
- Automatic `torch` / `jax` / MPI integration. Bytes interface only; the user owns the gather call.
- Distributed matching for a single image (cross-rank tile boundaries). Different problem, separate ADR if ever.
- Cross-version partial compatibility. Refused on `format_version` mismatch; documented as private format.

A more comprehensive non-goals list lives in *What this ADR explicitly does not decide* below.

## Decision drivers

- **Robustness — the partition guarantee is load-bearing.** Image-id disjointness across partials is what makes merge mathematically equal to a batch run over the union. Any design that silently tolerates partial overlap (last-writer-wins, first-writer-wins, "merge anyway") gives the user numbers they cannot reproduce with `Evaluator.evaluate(union)`. We refuse overlap with a typed error and document the partition rule prominently. The single-writer rule from ADR-0013 already enforces the same property within a process; this ADR extends it across processes.
- **Robustness — every wire-format invariant is checked, not assumed.** Dataset hash, params hash, parity mode, kernel kind, retain_iou flag, grid dimensions, format version: each gets a structured error on mismatch with the offending field named. The cost of this validation is tens of microseconds; the cost of debugging "my numbers are off by 0.003 mAP and I don't know why" without it is days.
- **Efficiency — N-way merge, not N pairwise.** The user's natural input is a list of N partials from N ranks, not a fold-from-the-left binary tree. An N-way merge validates once, allocates the final cells store at known size, and drains each partial as it consumes it. Pairwise merge with `merge(self, other)` would re-validate at every step, allocate a second store on every fold, and present a worse memory peak (`2 × sum` instead of `sum + max(partial)`).
- **Efficiency — share the format with `checkpoint()` / `restore()`.** Two formats means two rkyv schemas, two version bytes to evolve, two hash domains, two test matrices. One format is the right number. The currently-`NotImplemented` checkpoint and restore land via this ADR by composition: `checkpoint()` is `to_partial()`; `restore()` is `from_partials([bytes])` over a single-element slice.
- **Maintainability — ADR-0005 invariant.** No edits to `matching.rs` or `accumulate.rs`. Merge is orchestration above the spine, the same way `StreamingEvaluator` itself is. The cells store, `meta_cells`, `retained_ious`, and `dets_seen` are the seams; the matching kernel doesn't know merge exists.
- **Maintainability — one new public method, not a new evaluator type.** No `DistributedStreamingEvaluator`. No `MergedStreamingEvaluator`. `StreamingEvaluator` gains two methods (`to_partial`, classmethod `from_partials`) and one constructor parameter (`rank_id`); everything else composes through. `BackgroundEvaluator` inherits the surface for free — it already wraps a `StreamingEvaluator`, so `bg.to_partial()` drains and delegates.
- **Maintainability — no torch dependency, ever.** A bytes interface is the right interface. Users already on `torch.distributed` use `dist.gather_object` / `dist.all_gather_object`; users on `jax` use `jax.experimental.multihost_utils.process_allgather`; users on raw MPI use `comm.gather`. Vernier serves all three by knowing about none of them.
- **Determinism — strict-mode rank-order invariant.** ADR-0013 introduced "stream-order sensitivity" as a property that does not equal the three-tier ADR-0002 model. This ADR adds a sibling property: "rank-order invariant in strict mode given user-supplied `rank_id`". The mechanism is the same `(score, stream_position)` tiebreak ADR-0013 reserved on `next_dt_id`, lifted to `(rank_id, local_position)` for global ordering. Without rank_id, strict-mode merge has the same ULP wobble at score ties that mid-stream `snapshot()` already has — documented and bounded by the ADR-0004 4-ULP envelope, not pretended away.

## Considered options

The decision space splits across five orthogonal axes:

1. **Axis A — API shape.** Pairwise consuming `merge(self, other)` vs pairwise mutating `merge_in_place(&mut self, other)` vs N-way constructor `from_partials(&[bytes])`.
2. **Axis B — Wire format substrate.** Hand-rolled binary vs rkyv vs Arrow IPC vs JSON/CBOR.
3. **Axis C — Strict-mode rank determinism.** Accept ULP wobble vs require user-supplied `rank_id` vs auto-derive from process metadata.
4. **Axis D — Partial-overlap handling.** Disjoint partitions required vs first-wins vs last-wins vs configurable.
5. **Axis E — Validation strictness.** Best-effort with warnings vs strict typed errors vs configurable.

### Axis A — API shape

- **A1** `merge(self, other) -> Self` — pairwise consuming
- **A2** `merge_in_place(&mut self, other)` — pairwise mutating
- **A3 (chosen)** `StreamingEvaluator::from_partials(gt, &[bytes]) -> Self` — N-way constructor, plus instance method `to_partial(self) -> Vec<u8>`

A3 wins on every dimension that matters here. It validates the full partial set once, allocates the final cells store at known capacity, and treats merge as "build a fresh evaluator from a vector of states" — which is what it is. The pairwise variants encourage left-folds the user has to write and force the implementation to allocate intermediate evaluators that are immediately consumed. They also create a footgun: under A1, `a.merge(b).merge(c)` and `a.merge(b.merge(c))` allocate differently and have different transient memory peaks even though the result is the same.

### Axis B — Wire format substrate

- **B1** Hand-rolled binary
- **B2 (chosen)** rkyv (new top-level dep; see §"Context" for the dep-cost note)
- **B3** Arrow IPC
- **B4** JSON / CBOR

rkyv is the right format even paying the new-top-level-dep cost. ADR-0013 explicitly named it as the format for `cells` serialization in the deferred checkpoint plan. Zero-copy archive read on the deserialize side means a partial's `cells` HashMap doesn't have to be reconstructed entry-by-entry — we walk the archive directly during the merge fold. Arrow IPC would also be a new top-level dep at the `vernier-core` level (ADR-0019 already pulled `arrow-rs` for tables, but `vernier-core` doesn't transitively depend on it; using Arrow here would bind the cells-store wire format to the tables wire format, which conflates two evolution stories). JSON / CBOR pay parsing cost on every partial and don't give us the structural-validation property rkyv gives via its archived-type checks. The other two new deps (`blake3`, `crc32fast`) are small and orthogonal: `blake3` only fingerprints; `crc32fast` only frames. Total wheel-weight delta is ~150 KB.

### Axis C — Strict-mode rank determinism

- **C1** Accept strict-mode ULP wobble across rank orderings (same caveat as `snapshot()`)
- **C2 (chosen)** Require user-supplied `rank_id`; sort partials by `rank_id` during merge; reserve `(rank_id, local_position)` for the future strict-mode tiebreak
- **C3** Auto-derive rank_id from process metadata (PID, hostname, etc.)

C2 is the right answer because it's the only one that gives strict-mode users the bit-equals-batch guarantee they migrated from pycocotools to get. C1 leaks ULP wobble into strict-mode results, which contradicts the strict tier's whole point. C3 silently makes results irreproducible across runs (PID changes; hostname changes in containers); the user must own the rank identity. The cost of C2 is one constructor parameter (`rank_id: int | None = None`), defaulting to `None` for the common single-rank case where the parameter doesn't matter.

The wire format carries `rank_id` and `n_detections` (which becomes `local_position`'s upper bound). The future ADR that consumes the `(score, stream_position)` tiebreak in `matching.rs` lifts to `(score, rank_id, stream_position)` lexicographic order; the field is reserved here so the format doesn't have to rev when that ADR lands.

### Axis D — Partial-overlap handling

- **D1 (chosen)** Disjoint image-id partitions required; overlap is a typed error
- **D2** First-partial-wins (drop later overlapping cells)
- **D3** Last-partial-wins (overwrite)
- **D4** User-configurable policy

D1 because anything else lies. If two ranks evaluate the same image (which can happen if `DistributedSampler` is misconfigured, a common bug), the user wants to know — silently keeping one and dropping the other produces numbers that don't match either rank's local view or any batch reference. The error names which image_id collided across which rank pair so the user can fix their sampler.

D4 looks tempting until you realize "sometimes I want last-wins" is not a real use case for COCO-eval; it's a request to silently work around a sampler bug.

### Axis E — Validation strictness

- **E1** Best-effort: log warnings on mismatches, merge anyway
- **E2 (chosen)** Strict: typed error on any mismatch with the offending field named
- **E3** Configurable: user opts into strict or lenient

E2 because metric correctness is the headline property. The user calling `from_partials` is asking for a number their CI gate will read; producing a number whose provenance is "we noticed your params didn't match but we merged anyway" is worse than failing loudly. The cost of E2 is a few hash compares and an enum-tag check per partial — sub-millisecond at any realistic scale.

## Decision outcome

The combination is **A3 + B2 + C2 + D1 + E2**: an N-way `from_partials` constructor, rkyv wire format shared with `checkpoint()` / `restore()`, user-supplied `rank_id` for strict-mode global ordering, disjoint-partition requirement, and structured per-field validation.

### Rust core surface

A new module `crates/vernier-core/src/distributed.rs` (sibling of `stream.rs`, importing it):

```rust
/// Identifier for one rank in a multi-process eval. Strict-mode merge
/// uses `(rank_id, local_position)` as the global stream-order tiebreak;
/// corrected mode ignores it.
pub type RankId = u32;

/// Wire-format magic. ASCII "VRPS" = "vernier partial state".
const MAGIC: [u8; 4] = *b"VRPS";

/// Wire-format version. Increment on any breaking change to the
/// archived layout. Old versions are refused with a typed error.
const FORMAT_VERSION: u8 = 1;

impl<K: EvalKernel + 'static> StreamingEvaluator<K> {
    /// Set the rank identifier used for strict-mode global ordering.
    /// Calling this after the first `update()` is a programming error.
    /// Required (non-`None`) for strict-mode merge across ranks.
    pub fn with_rank(mut self, rank_id: RankId) -> Result<Self, EvalError>;

    /// Serialize the current evaluator state to an opaque byte blob.
    /// `from_partials([this_blob], ...)` reconstructs an equivalent
    /// evaluator on the receiving side. Consuming form; for a
    /// non-consuming snapshot use [`Self::snapshot_to_partial`].
    pub fn finalize_to_partial(self) -> Result<Vec<u8>, EvalError>;

    /// Non-consuming variant. Mid-stream snapshots that need to round-
    /// trip through a wire format use this; the evaluator stays
    /// usable for further `update()` calls.
    pub fn snapshot_to_partial(&mut self) -> Result<Vec<u8>, EvalError>;

    /// Construct an evaluator equivalent to a batch run over the union
    /// of all partials' submitted detections.
    ///
    /// All partials must share `dataset_hash`, `params_hash`,
    /// `parity_mode`, `kernel_kind`, `retain_iou`, and grid dimensions.
    /// In strict mode every partial must declare a distinct `rank_id`.
    /// Image-id sets across partials must be disjoint.
    ///
    /// # Errors
    ///
    /// - [`EvalError::PartialFormatMismatch`] on magic/version mismatch.
    /// - [`EvalError::PartialDatasetMismatch`] if `dataset_hash` differs
    ///   from the live dataset's hash.
    /// - [`EvalError::PartialParamsMismatch`] if any partial's params
    ///   diverge.
    /// - [`EvalError::PartialPartitionOverlap`] if two partials cover
    ///   the same `image_id`. The error names both rank_ids and the
    ///   colliding image_id.
    /// - [`EvalError::PartialRankCollision`] if two strict-mode partials
    ///   share a `rank_id`.
    /// - [`EvalError::OutOfBudget`] if the merged cells store would
    ///   exceed `budget`. Checked **before** allocation.
    pub fn from_partials(
        dataset: CocoDataset,
        kernel: K,
        params: OwnedEvaluateParams,
        parity_mode: ParityMode,
        budget: MemoryBudget,
        partials: &[&[u8]],
    ) -> Result<Self, EvalError>;
}
```

Two design choices worth pulling out:

- **`from_partials` takes `&[&[u8]]`, not `Vec<Vec<u8>>`.** The caller owns the partial buffers; we don't want to require them to be heap-allocated `Vec`s when in many DDP setups they arrive as borrowed slices into a gather buffer. The rkyv archived view is also borrow-friendly — we don't materialize the cells HashMap until we're ready to insert into the merged store.
- **`with_rank` is consuming-builder shape, not a `set_rank(&mut self)`.** Rank identity is a construction-time property, not a mid-run mutable parameter. The builder shape catches "I forgot to set rank before calling update" at the type level: a `StreamingEvaluator` that's been `update()`ed has the rank baked in.

`checkpoint()` and `restore()` are now expressible without changing their signatures — they were already in the public API per ADR-0013, just `NotImplemented`. They become thin wrappers:

```rust
pub fn checkpoint(&self) -> Result<Vec<u8>, EvalError> {
    self.snapshot_to_partial()
}

pub fn restore(
    dataset: CocoDataset, kernel: K, params: OwnedEvaluateParams,
    parity_mode: ParityMode, budget: MemoryBudget, bytes: &[u8],
) -> Result<Self, EvalError> {
    Self::from_partials(dataset, kernel, params, parity_mode, budget, &[bytes])
}
```

### Python surface

```python
class StreamingEvaluator:
    def __init__(
        self,
        ground_truth: bytes,
        *,
        iou_type: IouType = "bbox",
        parity_mode: ParityMode = "corrected",
        max_dets: tuple[int, ...] = (1, 10, 100),
        use_cats: bool = True,
        memory_budget_bytes: int | None = None,
        retain_iou: bool = False,
        rank_id: int | None = None,        # NEW
    ) -> None: ...

    def to_partial(self) -> bytes: ...                    # NEW; non-consuming
    def finalize_to_partial(self) -> bytes: ...           # NEW; consuming

    @classmethod
    def from_partials(
        cls,
        ground_truth: bytes,
        partials: Sequence[bytes],
        *,
        iou_type: IouType = "bbox",
        parity_mode: ParityMode = "corrected",
        max_dets: tuple[int, ...] = (1, 10, 100),
        use_cats: bool = True,
        memory_budget_bytes: int | None = None,
        retain_iou: bool = False,
    ) -> StreamingEvaluator: ...                          # NEW
```

Five new exception types ship in `vernier`:

```python
class PartialFormatMismatch(RuntimeError): ...
class PartialDatasetMismatch(RuntimeError): ...
class PartialParamsMismatch(RuntimeError): ...
class PartialPartitionOverlap(RuntimeError):
    rank_a: int
    rank_b: int
    image_id: int

class PartialRankCollision(RuntimeError):
    rank_id: int
```

`OutOfBudgetError` (ADR-0013) is reused unchanged for the merged-budget pre-check.

### Wire format (v1)

The on-the-wire layout is rkyv-archived and length-prefixed, but the conceptual schema is:

```
header {
    magic:           [u8; 4] = "VRPS"
    format_version:  u8 = 1
    parity_mode:     u8         // 0=Strict, 1=Corrected
    kernel_kind:     u8         // 0=Bbox, 1=Segm, 2=Boundary, 3=Keypoints
    retain_iou:      u8         // 0/1
    rank_id:         Option<u32>
    n_categories:    u32
    n_area_ranges:   u32
    n_images:        u32
    dataset_hash:    [u8; 32]   // blake3 of canonical dataset bytes
    params_hash:     [u8; 32]   // blake3 of OwnedEvaluateParams archived
}
body {
    n_detections:    u64
    next_dt_id:      i64
    seen_images:     ArchivedHashSet<i64>
    cells:           ArchivedHashMap<(u32, u32, u32), PerImageEval>
    meta_cells:      Optional<ArchivedHashMap<(u32, u32, u32), EvalImageMeta>>
    retained_ious:   Optional<ArchivedRetainedIous>
    dets_seen:       Optional<Vec<CocoDetection>>
}
footer {
    crc32:           u32         // over header + body
}
```

The hashes deserve specific attention:

- **`dataset_hash`** is `blake3(canonical_form(dataset))` where canonical form is a stable serialization of `(images sorted by id, categories sorted by id, annotations sorted by id)`. The annotation list is the longest part; we fingerprint per-annotation `(id, image_id, category_id, area, is_crowd, ignore, bbox, segmentation, keypoints)` in a fixed field order. ADR-0013 named this hash and reserved a slot for it; this ADR is where it lands. Independent ADR property: changing canonical-form requires bumping `format_version`.
- **`params_hash`** is `blake3(rkyv::to_bytes(&OwnedEvaluateParams))`. The archived form is the ground truth for "are these params equal?" — much safer than a hand-rolled equality check that has to be kept in sync as fields are added.

Backward compatibility commitment is the same as `checkpoint()` / `restore()`'s in ADR-0013: **the format is private, we make no cross-minor-version compatibility promises until the streaming surface stabilizes.** A user who needs partial-portability across vernier upgrades pins their vernier version. The CRC catches truncation; the magic + version refuses old binaries; that's the compatibility story.

### Validation order

Cheapest-first, so a malformed partial fails before we touch the body:

1. Length check: `bytes.len() >= MIN_HEADER_BYTES`
2. Magic match: `bytes[0..4] == "VRPS"`
3. Version match: `bytes[4] == FORMAT_VERSION`
4. CRC over `bytes[..bytes.len()-4]` matches `bytes[bytes.len()-4..]`
5. Header field validation: parity_mode / kernel_kind / retain_iou / grid dims match the live evaluator's
6. `dataset_hash` matches live dataset's hash
7. `params_hash` matches live params' hash
8. (Across all partials) `rank_id` distinct in strict mode
9. (Across all partials) `seen_images` pairwise disjoint
10. (Across all partials) `Σ |cells|` × cell-cost-estimate fits in `budget`

Step 10 is the load-bearing pre-check: we estimate the merged cells-store size from the partials' header counts, compare against `budget`, and refuse with `OutOfBudgetError` *before* allocating. This is what makes the merge predictable at scale — a 100k-image LVIS run across 8 ranks won't OOM mid-merge because the budget machinery already enforced the cap on each rank during its own streaming pass.

### Determinism contract

This ADR adds two rows and one column to the table introduced in ADR-0013:

| Surface | Stream-order invariant? | Rank-order invariant? | Bit-equals batch? |
|---|:---:|:---:|:---|
| `finalize()` (single rank) | yes | n/a | yes |
| `snapshot()` default | no (boundary ULPs) | n/a | yes-on-subset |
| `snapshot(running=True)` | no | n/a | no — running PR |
| `from_partials([..]).finalize()` (strict, all rank_ids set) | yes | **yes** | **yes** (over union) |
| `from_partials([..]).finalize()` (corrected) | no (boundary ULPs) | no (boundary ULPs) | yes-on-union |
| `from_partials([..]).snapshot()` | no | no | yes-on-union-subset |

The new "rank-order invariant" column documents the property that strict-mode users get when they pass `rank_id` and corrected-mode users do not. The 4-ULP envelope from ADR-0004 bounds the corrected-mode wobble; nothing about merge widens it.

### Test plan

The harness extends `tests/python/parity/streaming/` rather than forking. The fixture set is the same `BBOX_FIXTURES + SEGM_FIXTURES` the existing finalize-equals-batch test uses; we add a new file `test_distributed_merge.py` covering:

1. **Shard-and-merge equals batch** (the headline). For every fixture and every `n_ranks ∈ {2, 4, 16}`, partition DT bytes by image_id (`shard_dt_bytes` already exists in `conftest.py`), run one `StreamingEvaluator` per shard with `rank_id = i`, collect partials via `to_partial()`, reconstruct via `from_partials()`, finalize, assert bit-equality of `summary.stats` against `Evaluator.evaluate(union)`. Strict and corrected modes both.
2. **Roundtrip equals self.** `to_partial → from_partials([single]) → finalize` equals `finalize` on the original. Pins the checkpoint/restore property as a special case.
3. **N-way equals pairwise reduction.** `from_partials([a, b, c])` produces stats bit-equal to `from_partials([from_partials([a, b]).snapshot_to_partial(), c])`. Property test on associativity.
4. **Disjoint-partition rejected.** Construct two evaluators that both consume `image_id 7`; assert `from_partials` raises `PartialPartitionOverlap` naming both rank_ids and the colliding image_id.
5. **Dataset-hash mismatch rejected.** Construct partial against GT-A; merge against GT-B (one annotation moved by one pixel); assert `PartialDatasetMismatch` naming the mismatched hash.
6. **Params mismatch rejected.** Different `max_dets` across partials; `PartialParamsMismatch`.
7. **Kernel mismatch rejected.** One bbox partial, one segm partial; `PartialFormatMismatch`.
8. **Strict-mode rank collision rejected.** Two partials with the same `rank_id`; `PartialRankCollision` in strict mode. (Corrected mode permits this — `rank_id` is informational only in corrected.)
9. **Format-version refused.** Hand-craft a `format_version=99` header; `PartialFormatMismatch` with both versions named.
10. **CRC failure detected.** Flip one byte in a partial; `PartialFormatMismatch{kind: Crc, ..}`.
11. **Memory budget enforced pre-allocation.** Construct partials whose summed estimated size exceeds budget; `OutOfBudgetError` returns before any large allocation. Assert via `memory_used_bytes` on a fresh evaluator (must be unchanged).
12. **Tables across merge.** With `retain_iou=True`, `from_partials([..]).finalize_with_tables(per_pair=True)` matches `Evaluator(retain_iou=True).evaluate(union, tables=("per_pair",))` row-for-row. Pins the table parity story across the merge boundary.
13. **BackgroundEvaluator inheritance.** `BackgroundEvaluator.to_partial()` (drains worker, calls underlying streaming evaluator's method) plus `from_partials` reconstruct as expected. The background wrapper inherits the surface for free; the test pins that the inheritance doesn't quietly break.

Strict-mode tests for properties 1, 2, 3, 8 currently rely on the `(score, stream_position)` tiebreak that ADR-0013 reserved on `next_dt_id` but that the matching path does not yet consume. This ADR's strict-mode property tests are therefore conditionally skipped (`pytest.skip`) until the tiebreak ships, exactly the way `tests/python/parity/streaming/conftest.py::ROOT_STAYS` already documents. The skip message names the predecessor ADR.

### What this ADR explicitly does *not* decide

- **GPU all-reduce inside vernier.** The bytes interface intentionally does not assume the user has a GPU communicator. A user who wants all-reduce passes the partial bytes through their existing communicator. (`torch.distributed.all_gather_object` already does the right thing.)
- **Automatic torch / jax integration.** Vernier does not import torch. A user-side helper wrapping the gather call is two lines; vernier shipping the helper would make it the de-facto torch-version pin chase, which is exactly what ADR-0006 §"Considered options" rejected for its own reasons.
- **Dynamic rank addition mid-run.** All ranks must be present at `from_partials` time. A "ranks join over time" model is ill-posed for batch metrics: AP is computed over the union, and the union changes shape every time a rank joins.
- **Async / streaming merge.** `from_partials` is a single-shot call. Streaming merge (consume partials as they arrive over a socket) is a different shape; `BackgroundEvaluator`'s threading model is for ingest, not for cross-process orchestration.
- **Cross-version partial compatibility.** Refusing on `format_version` mismatch is a feature. A user who needs to merge a partial captured on vernier 0.6 against a 0.8 evaluator pins the version. Format evolution will use a migration-on-read path the same way ADR-0017's bench results do, *if and when* the streaming surface stabilizes; pre-1.0, no.
- **Distributed `error_decomposition` (TIDE).** TIDE per ADR-0021 has its own structural cross-class IoU work; a distributed TIDE merge is a different reduction (the cross-class IoU matrix is per-image, but the bin-attribution histograms aggregate differently). Tracked as a follow-up; this ADR's surface does not foreclose it.
- **Partial sharing across machines with different endianness.** rkyv's archived layout is little-endian; both x86_64 and aarch64 are LE in practice, and big-endian deployment is hypothetical. If a real user materializes, that's a separate ADR for an endian-agnostic re-archive path.
- **Compression of the partial format.** rkyv archived payloads are dense; for the cells store at COCO val2017 scale, a partial is ~600 KB / rank. `zstd` would shave it to ~250 KB but adds a top-level dep and a CPU cost the user's existing transport (`gather_object` pickles the bytes; pickle does compression already if the user opts in via `protocol=4`+) likely already covers. Punted.
- **Distributed semantic / panoptic merge.** `vernier.semantic` and `vernier.panoptic` have their own evaluator classes (per ADR-0028, ADR-0025) with their own state shapes. The wire-format magic + version + kernel-kind discriminator leaves room to extend, but the cell-store layout for confusion matrices (semantic) and PQ accumulators (panoptic) is different. Separate ADRs when the demand surfaces; this ADR is instance-detection only.

### Consequences

- **Positive.** The DDP / multi-node training story closes — every rank evaluates locally, gathers a small bytes payload (hundreds of KB per rank at COCO scale), and the head rank produces a Summary that is bit-equal to a hypothetical batch run over the union. ADR-0005's invariant survives a fifth orchestration layer being added (matching.rs and accumulate.rs untouched). `checkpoint()` and `restore()` ship in the same release: one wire format, one schema, one set of validation rules, one test surface. The strict-mode determinism story extends cleanly via `(rank_id, local_position)` — the future tiebreak ADR consumes a field this ADR already reserves. `BackgroundEvaluator` inherits the merge surface trivially because it already wraps a `StreamingEvaluator`. The MLOps persona's "deterministic CI gate even across distributed eval" claim becomes credible because every invariant is checked, named, and pinned by a property test.
- **Negative.** The wire format becomes a forward-compatibility commitment, even with the "private, no cross-version promises" caveat. Bumping `format_version` is operationally cheap but socially load-bearing: users with checkpoint files don't appreciate format churn. We mitigate by making the format the obvious answer to multiple problems (merge, checkpoint, future TIDE merge, future multi-host benchmarks) so the rev rate is low. The strict-mode bit-equality property tests are conditionally skipped until the `(score, stream_position)` tiebreak lands in the matching path — meaning this ADR ships a corrected-mode-bit-equal-on-union claim and a strict-mode-bit-equal-on-union *property* (asserted by the harness modulo skip). The skip is honest, but it's a UX wart for the strict-mode CI user who reads the determinism contract table and finds the strongest cell asterisked. Memory budget at merge time sums across partials; the merged evaluator's footprint is the sum, which can surprise a user who provisioned for one rank's footprint. The structured `OutOfBudgetError` breakdown helps but doesn't eliminate the surprise.
- **Neutral.** `rank_id` becomes an additional construction parameter on `StreamingEvaluator` for users who never plan to merge. Defaulting it to `None` keeps the single-rank ergonomics intact, but the parameter exists in the constructor signature and shows up in IDE auto-complete. The cells-store wire format is now part of vernier's externally-observable surface even though it's "private" — anyone willing to read the rkyv archived layout can build their own merger. We document this as a non-supported integration path and hope nobody does it; if someone does, the version byte gives us a cheap break-glass on cross-version compatibility. The validation order in `from_partials` is fixed and explicit; users debugging "why won't my partials merge" follow the order in the typed-error variant names. This is a feature, but it does mean we own the diagnostic UX as much as we own the format.

## Pros and cons of the options

### Axis A — API shape

**A3 (chosen) — `from_partials(&[bytes])`, N-way constructor**

- 👍 Validates the full partial set once. A 16-rank merge re-verifies dataset_hash zero times after the first.
- 👍 Allocates the final cells store at known capacity from header counts. No transient intermediate stores.
- 👍 Memory peak is `sum + max(any partial)` rather than `2 × sum`.
- 👍 Budget pre-check fits the structure naturally: see all partials before allocating any.
- 👎 Less idiomatic for users who think in `Iterator::reduce` shape. Mitigated by the python-side `from_partials(gt, list_of_bytes)` matching the gather pattern exactly.

**A1 — pairwise consuming `merge`**

- 👍 Familiar shape from `Iterator::reduce`. Composes with `.fold()` in user code.
- 👎 Re-validates header on every fold step; a 16-rank merge re-verifies the same dataset_hash 15 times.
- 👎 Allocates two cells stores during transient pairwise merge: the result of `a.merge(b)` lives until `c.merge(...)` consumes it.
- 👎 Encourages user-side associativity bugs (the result is well-defined but the transient memory footprint isn't).
- 👎 Makes the budget pre-check structurally awkward — you'd have to sum across the user's fold without seeing all partials.

**A2 — pairwise mutating `merge_in_place`**

- 👍 No transient allocation.
- 👎 Same per-step re-validation as A1.
- 👎 Mutates the receiver in a way that makes "fold from a fresh evaluator" the only sensible pattern, which is exactly what `from_partials` already does — A2 is a worse-API version of A3.

### Axis B — Wire format substrate

**B2 (chosen) — rkyv**

- 👍 Already in `vernier-core`'s deps for the RLE codec; no top-level dep change.
- 👍 ADR-0013 explicitly named it as the format for `cells` serialization in the deferred checkpoint plan — this ADR cashes the check.
- 👍 Zero-copy archive read on deserialize; we walk archived `cells` directly during the merge fold without materializing intermediate HashMaps.
- 👍 Structural validation via archived-type checks; field renames break parse loudly.
- 👎 Schema evolution is rkyv's responsibility, not ours. We inherit its semver discipline along with its archive layout.

**B1 — hand-rolled binary**

- 👍 Zero dependency cost.
- 👎 Every `PerImageEval` field becomes a hand-written serialize/deserialize pair, against the project's existing rkyv-via-mask-codec precedent. Maintenance burden.
- 👎 No archived-type validation; we'd be re-implementing rkyv's structural checks worse.

**B3 — Arrow IPC**

- 👍 Existing tables surface uses Arrow.
- 👎 The cells-store layout (`HashMap<(u32, u32, u32), PerImageEval>` with internal `dt_matched: Array2<bool>` arrays per cell) is not naturally columnar. Forcing it into Arrow's row/column model would either lose the sparse structure or add an indirection layer.
- 👎 Couples merge format evolution to tables format evolution. Two unrelated stories sharing a version byte is exactly the maintenance cost we want to avoid.

**B4 — JSON / CBOR**

- 👍 Human-debuggable.
- 👎 Parsing cost dominates merge wall-clock for large partials.
- 👎 No structural validation at parse time; field-mismatch errors surface as semantic bugs instead of typed format errors.

### Axis C — Strict-mode rank determinism

**C2 (chosen) — user-supplied `rank_id`**

- 👍 The only option that gives strict-mode users the bit-equals-batch guarantee they migrated to vernier for.
- 👍 Rank identity is explicit in user code; reviewing a training script tells you whether the merge will be deterministic.
- 👍 Extends cleanly to the future `(score, rank_id, local_position)` tiebreak.
- 👎 One more constructor parameter. Defaults to `None` so single-rank users don't see it.

**C1 — accept strict-mode ULP wobble**

- 👍 No new constructor parameter.
- 👎 Strict-mode users migrated from pycocotools precisely to get bit-equals-batch. Telling them "but only single-rank" undoes the migration story for multi-rank users.

**C3 — auto-derive `rank_id` from process metadata**

- 👍 No user-facing parameter.
- 👎 PIDs and hostnames are not stable across runs; results would be irreproducible across re-runs of the same training script. This is the worst possible failure mode for a parity-preserving evaluator.
- 👎 Containers and process-namespaces produce colliding PIDs across hosts.

### Axis D — Partial-overlap handling

**D1 (chosen) — disjoint partitions required, overlap is a typed error**

- 👍 Result is unambiguously `Evaluator.evaluate(union)`.
- 👍 Surfaces sampler bugs early.
- 👍 The error names both rank_ids and the colliding image_id; the user fixes their `DistributedSampler` and moves on.
- 👎 A user who wants to deduplicate before merging has to do it themselves. Fine; it's six lines of Python.

**D2 / D3 — first-wins / last-wins on partition overlap**

- 👍 Simpler for the user — they don't have to fix their sampler.
- 👎 Produces a number that doesn't match either rank's local view *or* a batch reference. The user doesn't know they have a sampler bug.
- 👎 The number depends on the order partials arrive in `from_partials`, which is non-deterministic for `dist.all_gather_object` under some backends.

**D4 — user-configurable overlap policy**

- 👍 Flexibility.
- 👎 Adds a parameter whose only sensible value is "error". The other values exist to let users silence a sampler bug, which is not something vernier should help with.

### Axis E — Validation strictness

**E2 (chosen) — strict typed errors**

- 👍 Metric correctness is the headline property; ambiguous merges undermine it.
- 👍 Validation cost is sub-millisecond per partial.
- 👍 Each error variant names the offending field; debugging is grep-friendly.
- 👎 Users with intentionally-different params across ranks (rare but possible — e.g., different `max_dets` per rank for some debugging reason) cannot merge.

**E1 — best-effort validation**

- 👍 Permissive.
- 👎 Same root failure as D2 / D3: produces numbers whose provenance is ambiguous.
- 👎 The cost of strict validation is sub-millisecond per partial; "best-effort" is solving a non-problem at the cost of correctness.

**E3 — configurable strict / lenient**

- 👍 Flexibility.
- 👎 Adds a parameter whose lenient setting exists to silence what should be loud errors.

## Links and references

- ADR-0001 — Record architecture decisions (§"Affect the public API", §"Cross the FFI boundary", §"Set a project-wide convention" all triggered).
- ADR-0002 — Three-tier parity model. Inherited unchanged through the merge boundary.
- ADR-0004 — Numerical layout policy. The 4-ULP aligned-tier envelope bounds the rank-order wobble in corrected mode.
- ADR-0005 — Lock the `Similarity` trait and matching-engine API. This ADR's invariant: no edits to `matching.rs` or `accumulate.rs`. Merge is orchestration above the spine.
- ADR-0006 — Threading model. Untouched: merge runs synchronously on the calling thread with the GIL released during the rkyv decode + cell-store fold.
- ADR-0013 — Streaming evaluator. The merge follow-up explicitly deferred there. This ADR closes that deferral and lands `checkpoint()` / `restore()` (currently `NotImplemented`) by composition.
- ADR-0014 — `BackgroundEvaluator`. Inherits `to_partial` / `from_partials` for free.
- ADR-0017 — Local bench harness. Future `--surface distributed` runner extends additively.
- ADR-0019 — Result tables. Tables across merge are folds over per-rank `meta_cells` + `retained_ious`; no surface change.
- ADR-0021 — TIDE error decomposition. Distributed TIDE is a separate ADR; this format does not foreclose it.
- ADR-0030 — Buffer-protocol ingest. Orthogonal: ranks ingest via array path or JSON; partials carry post-ingest cells regardless.
- Future ADR — `(score, stream_position)` strict tiebreak. Consumes `next_dt_id` from `stream.rs` and `rank_id` from this ADR. Lifts to `(score, rank_id, local_position)` lexicographic order.
