# ADR-0020: Parsed-once `Dataset` handle as the GT-side derivation cache

- **Status:** accepted
- **Date:** 2026-05-02
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

Today `Evaluator.evaluate(gt: bytes, dt: bytes)` accepts raw COCO JSON
bytes on every call. Each call re-parses the GT JSON and re-derives
every GT-side quantity the matching engine consumes — most expensively,
for boundary IoU, the per-annotation eroded-band segments (~36k erodes
on val2017, the hot path that ADRs/PRs #87–#89 spent five iterations
optimizing).

The training-loop persona that ADR-0014 builds for runs validation eval
many times against the *same* GT (every N steps for N→∞). The GT does
not change between epochs. Re-parsing JSON and re-deriving bands on
every validation pass is pure waste — and there is no way for the user
to express "this is the same GT" with the current API. The bytes are
the only surface; bytes do not carry identity.

The *internal* caches already exist: `vernier-core` ships
`BoundaryGtCache` and `SegmGtCache` (caller-owned, ratio-keyed,
threadsafe), with `evaluate_boundary_cached` / `evaluate_segm_cached`
public on the Rust side. They are not exposed at the FFI boundary
and not visible to Python users today. What's missing is the
user-facing primitive that owns those caches across `evaluate` calls.

This ADR introduces that primitive. It leaves the zero-overhead
default path untouched: callers who keep passing bytes keep paying
exactly what they pay today.

## Decision drivers

- **Eval is repeated; GT is not.** Training validation hits the same
  GT on every pass. The cache should be tied to the GT, not the
  evaluator (`Evaluator` is cheap; users may have several over one
  GT — different IoU kinds, parity modes).
- **Zero overhead for non-cache users.** ADR-0001 §"Affect the public
  API." Anyone passing bytes today must see no behavior change and no
  latency change.
- **Composition with ADR-0014.** `BackgroundEvaluator` shares state
  across worker threads via `Arc`. Whatever cache lives on GT must
  itself be `Send + Sync` and clone-as-`Arc` cheap.
- **Per-kernel, parameterized.** Boundary derivations key on
  `dilation_ratio`. Future kernels (segm GT counts, OKS visibility
  masks) want their own slots. The cache must be a typed map of slots,
  not one global blob.
- **Identity must be the user's variable**, not a hash or `id()`
  heuristic. Cache hit ⇔ "you reused the object." No fuzz.

## Considered options

1. **`GtCache` side-handle.** New public type that wraps the GT and
   holds the cache; user passes `cache` instead of `gt` to `evaluate`.
2. **Implicit on `Evaluator`.** Cache lives inside the (frozen)
   `Evaluator`; first call populates, subsequent structurally-equal GT
   reuses.
3. **`evaluator.with_gt(gt).evaluate(dt)`.** Method returning a bound
   evaluator; cache lives on the binding.
4. **Parsed-once `Dataset` handle.** Introduce a typed `Dataset` whose
   identity *is* "the loaded GT"; cache lives on it as interior-mutable
   per-kernel slots.

## Decision outcome

Chosen option: **(4) Parsed-once `Dataset` handle.** A/2/3 all bolt a
cache onto the side of an API designed around bytes; (4) names what
the user already has — "the loaded GT" — and lets caching fall out of
identity-by-construction. One new public type does both jobs.

The bytes-path stays exactly as today. `evaluate` becomes:

```python
def evaluate(self, gt: bytes | Dataset, dt: bytes) -> Summary: ...
```

Bytes branch: parse → derive → discard, identical to current behavior.
`Dataset` branch: parse-was-already-done, derivations consult the
cache and populate on miss.

### Consequences

- **Positive.** Training-loop validation across N epochs collapses
  GT-side cost from O(N) to O(1) for any kernel that registers a
  derivation slot. One handle, one `Arc` clone into
  `BackgroundEvaluator`. The cache is per-GT, so multiple `Evaluator`s
  (different IoU, parity modes) over one `Dataset` share derivations.
  No behavior change for current bytes callers.
- **Negative.** New public type on the surface. Long-lived training
  sessions accumulate cached derivations until the `Dataset` is
  dropped — fine in practice (one set per kernel × parameter, bounded
  by JSON GT size) but worth documenting.
- **Neutral.** Two evaluation entry points (bytes / `Dataset`) for
  one operation. Callers who don't reuse GT keep the bytes path; the
  pyi overloads make the choice explicit.

## Rust core surface

`vernier-core` already ships `BoundaryGtCache` and `SegmGtCache` —
caller-owned, mutex-guarded `HashMap<ann_id, Entry>` types passed as
`&cache` arguments to `evaluate_*_cached`. They are
`Send + Sync`, threadsafe, and ratio-keyed where parameterized
(`BoundaryGtCache::align_ratio`). Tests, an example
(`crates/vernier-core/examples/cache_speedup_val2017.rs`), and the
internal kernel wiring are all in place.

What this ADR adds is the *user-facing primitive that owns those
caches*: a `Dataset` type whose lifetime is the cache lifetime.
Bundling the existing caches behind one handle, not refactoring them.

The vernier-core surface stays as-is. The new wrapper lives at the
FFI boundary:

```rust
// crates/vernier-ffi/src/dataset.rs
pub struct PyDataset {
    gt: Arc<CocoDataset>,
    boundary_cache: Arc<BoundaryGtCache>,
    segm_cache: Arc<SegmGtCache>,
    // future: oks_cache, ...
}
```

Every cache is constructed eagerly (each is empty + cheap) and
populated lazily on its first cached `evaluate_*_cached` call. The
`Arc`s let `PyDataset::clone` produce a Python-side handle that
shares cache state — needed for `BackgroundEvaluator` (ADR-0014)
once it lands at the FFI surface.

This ADR ships the boundary + segm slots (both already exist in
core). OKS / future kernels add a slot each; no API change.

## Python surface

```python
class Dataset:
    @classmethod
    def from_json(cls, bytes: bytes) -> Dataset: ...
    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> Dataset: ...
```

`Dataset.from_path` is a thin convenience over `from_json` that reads
the file. We do not expose construction from typed Python objects;
JSON is the format users have. The wrapper holds an
`Arc<CocoDataset>`; cheap to clone, threadsafe.

`Evaluator.evaluate` accepts `bytes | Dataset` for the `gt`
parameter. The dispatch is a single `isinstance` check at the top of
the method — type-stable, zero-cost on the bytes branch.

We do *not* introduce a `Dataset.evaluate(...)` method. Evaluation is
the `Evaluator`'s job; the `Dataset` is data.

## What this ADR explicitly does not decide

- **Cache eviction.** None. The cache is bounded by the
  `Dataset`'s lifetime; users drop the handle when they are done.
  Adding eviction is a follow-up if a real workload shows the bound
  is wrong.
- **Disk persistence of derivations.** Cache lives in memory, dies
  with the `Dataset`. Persisting bands to disk for cross-process
  reuse is a separate ADR (the streaming-checkpoint surface from
  ADR-0013 is the natural place to fold it in).
- **Other kernels' slots.** Segm GT counts, OKS visibility, future
  Phase-3 derivations — all follow the same pattern, all land in
  follow-up PRs without changing the public API.
- **DT-side caching.** Detections change every call by definition.
  Out of scope.
- **Equality / hashing on `Dataset`.** Identity is object identity.
  Two `Dataset`s loaded from the same JSON are *not* equal and do
  *not* share cache; this is the principled answer (reload = fresh
  GT) and avoids hashing GT contents.

## Pros and cons of the options

### (1) `GtCache` side-handle

- 👍 Explicit, opt-in. Easy to add without touching `Dataset` (because
  there is no `Dataset` today).
- 👎 Two abstractions for one concept ("loaded GT" + "GtCache"). The
  user's mental model already has "the loaded GT"; the side-handle
  duplicates it.

### (2) Implicit on `Evaluator`

- 👍 Zero API change.
- 👎 Wrong axis: cache is per-evaluator, not per-GT. Multiple
  evaluators over one GT can't share. Cache identity becomes a hash
  problem (`id()` is fragile across reloads).

### (3) `evaluator.with_gt(gt).evaluate(dt)`

- 👍 Composable; fits the frozen-dataclass `Evaluator`.
- 👎 The intermediate type still needs a place to hold derivations;
  if it's per-call, no caching; if it's reused, it *is* (4) under a
  worse name.

### (4) Parsed-once `Dataset` handle (chosen)

- 👍 Names the missing primitive. Identity-by-construction —
  cache hit ⇔ object reuse, no hashing or `id()` heuristics. Per-GT
  axis is correct (multiple `Evaluator`s share). Composes with
  ADR-0014 via `Arc<CocoDataset>` clone. Forward-extends to other
  kernels' derivations via more slots, no API change.
- 👎 New public type. Two `evaluate` entry shapes (bytes / `Dataset`)
  to document.

## Links and references

- ADR-0001 — Record architecture decisions (§"Affect the public API",
  §"Cross the FFI boundary").
- ADR-0006 — Threading model. The `Arc<DerivationCache>` is the
  share-across-workers carrier ADR-0006 anticipated for a future
  parallelization story.
- ADR-0010 — Boundary IoU subsystem. The first cache slot lives here;
  the per-`(dilation_ratio, parity)` keying matches ADR-0010's quirk
  surface.
- ADR-0013 — Streaming evaluator. Detection identifiers and
  cross-call accounting; the GT cache here is the GT-side analogue
  of the streaming evaluator's running state.
- ADR-0014 — `BackgroundEvaluator`. The wrapper holds an
  `Arc<CocoDataset>`; this ADR ensures `clone()` shares the
  derivation cache for free.
