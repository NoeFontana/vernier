# ADR-0035: Consolidate the streaming/DDP/background public surface

- **Status:** proposed
- **Date:** 2026-05-09
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

## Context and problem statement

Vernier ships three near-overlapping public classes per paradigm:
`Evaluator` (frozen config; batch entry point), `StreamingEvaluator`
(in-loop submit/finalize, hosts the DDP wire-format methods), and
`BackgroundEvaluator` (a worker-thread wrapper over
`StreamingEvaluator`). The split made sense when ADR-0013 (streaming),
ADR-0014 (background), and ADR-0031/0032 (distributed eval) landed in
sequence — each ADR added the smallest surface it needed, on top of
the previous one. The accumulated result is that for a single-rank
user, `StreamingEvaluator` is "a more annoying `Evaluator`": same job,
different shape, different namespace.

Two specific overlaps make the surface noisier than necessary:

1. The DDP entry points (`to_partial`, `from_partials`) live on
   `StreamingEvaluator`, not on `Evaluator`. The training-loop
   tutorial and the distributed-eval how-to both push users through
   `StreamingEvaluator` even when they never need streaming submit.
2. `StreamingEvaluator.snapshot(running=True)` and
   `BackgroundEvaluator.snapshot(peek=True)` are biased fast paths
   that ADR-0013 §"Fast snapshot mode" itself flags as inappropriate
   for quality gates. They have shipped as `# TODO` stubs ever since
   (delegating to the parity-correct `snapshot()`); no real consumer
   has materialised. Same story for `checkpoint()` / `restore()`,
   which ship as `NotImplemented` thin wrappers around
   `from_partials([single_partial])`.

We are at 0.0.x. This is the moment to collapse the surface — before
1.0 freezes any of it — so each capability has one canonical home.

## Decision drivers

- Single source of truth for each capability (one canonical method
  per use case, not three classes that almost-but-not-quite agree).
- Pre-1.0 hard breaks are cheaper than post-1.0 deprecation cycles.
- Preserve the streaming substrate and the wire format (ADR-0031,
  ADR-0032). This is a public-surface refactor, not a kernel rewrite.
- Don't ship documented-as-broken features (the biased running/peek
  snapshots). If the use case revives, design it from scratch.

## Considered options

1. **Status quo.** Keep all three classes; add `to_partial` /
   `from_partials` on `Evaluator` *as well as* `StreamingEvaluator`.
2. **Merge `StreamingEvaluator` into `Evaluator` as a stateful
   handle.** Frozen-dataclass `Evaluator` becomes a mutable streaming
   evaluator; `evaluate(...)` mutates internal accumulator state.
3. **Remove the `StreamingEvaluator` pyclass from Python entirely;
   lift DDP onto `Evaluator` as new methods.** The streaming substrate
   stays in Rust (`vernier_core::stream`) where the FFI orchestrators
   and `BackgroundEvaluator`'s worker thread own it; the Python wheel
   never sees a streaming class. `BackgroundEvaluator` stays public.
   `snapshot(running=True)`, `BackgroundEvaluator.snapshot*`,
   `checkpoint`, `restore` go away.

## Decision outcome

Chosen option: **Option 3.** The frozen-dataclass shape of `Evaluator`
is preserved (ADR-0006 immutable-config invariant survives). DDP
methods sit naturally on the batch entry point: `evaluate_to_partial`
mirrors `evaluate`, and `from_partials` is a classmethod that
constructs and finalizes in one shot. The streaming class is removed
from the Python module entirely; six new PyO3 functions
(`evaluate_*_to_partial` / `merge_*_partials`, two per paradigm) wrap
the Rust streaming substrate (`StreamingEvaluator<K>`,
`StreamingPanopticEvaluator`, `StreamingSemanticEvaluator`) which
remains the implementation. `BackgroundEvaluator` keeps its public
shape and continues to wrap the same Rust substrate via its worker
thread.

`BackgroundEvaluator` keeps a small, sharp surface — `submit`,
`finalize`, `finalize_to_partial`, `finalize_with_tables`, context
manager — because the "evaluate off my training thread" use case
isn't covered by anything else. The `peek=True` / `snapshot(...)`
read-out paths go away alongside the streaming ones; users wanting an
unbiased mid-epoch readout call `Evaluator.evaluate(gt, accumulated_dt)`.

### Public surface after the cut

```python
# vernier.instance / vernier.panoptic / vernier.semantic each ship:
class Evaluator:                 # frozen config dataclass
    def evaluate(gt, dt) -> Summary: ...
    def evaluate_to_partial(gt, dt, *, rank_id) -> bytes: ...
    @classmethod
    def from_partials(gt, partials, /, **config) -> Summary: ...
    # plus paradigm-specific helpers (with_options on instance,
    # background factory on semantic/panoptic, ...)

class BackgroundEvaluator:       # worker-thread wrapper (FFI pyclass)
    def submit(...): ...
    def finalize() -> Summary: ...
    def finalize_to_partial() -> bytes: ...
    def finalize_with_tables(...) -> ...
    # plus context-manager + properties
```

`StreamingEvaluator`, `StreamingPanopticEvaluator`,
`StreamingSemanticEvaluator` are removed from the Python module
entirely (no `_impl` shim, no FFI re-export). The Rust substrate
remains; the new pyfunctions construct it internally. `Evaluator.stream(...)`
factories on semantic/panoptic are removed (they returned the now-removed
class).

### Removed surface

- `StreamingEvaluator`, `StreamingPanopticEvaluator`,
  `StreamingSemanticEvaluator` removed from `vernier._core` and from
  every paradigm's `__all__`. No private import path; the Rust struct
  is unreachable from Python.
- `StreamingEvaluator.snapshot(running=True)` and the underlying
  `vernier_core::stream::StreamingEvaluator::snapshot_running` method.
- `StreamingEvaluator.checkpoint` / `restore` (Rust core stubs).
- `BackgroundEvaluator.snapshot`, `snapshot(peek=True)`,
  `snapshot_with_tables`, `to_partial` (non-finalize) on all three
  paradigms.
- `from_partials` on `BackgroundPanopticEvaluator` /
  `BackgroundSemanticEvaluator` (the panoptic and semantic versions
  carried a return-type bug — they returned the streaming class — and
  no caller used them).
- `Evaluator.stream(...)` factory methods on semantic and panoptic.

The streaming substrate (`crates/vernier-core/src/stream.rs`,
`crates/vernier-panoptic/src/stream.rs`) keeps `snapshot()` (the
parity-correct path), `finalize`, `snapshot_to_partial`,
`finalize_to_partial`, and `from_partials`. The Python entry points
above delegate down to it.

### Consequences

- **Positive:** the public per-paradigm surface drops from three
  classes to two with a clear job split (batch+DDP / in-training).
  The DDP recipe in `docs/how-to/distributed-eval.md` no longer
  forces an import the user doesn't need. Three documented-as-biased
  features (running/peek/checkpoint) and one return-type bug exit
  the public API.
- **Negative:** existing users of `vernier.instance.StreamingEvaluator`
  must rewrite call sites against `Evaluator.evaluate_to_partial`
  (DDP submit) or `BackgroundEvaluator.submit` (training-loop
  submit). The streaming class has no Python-side replacement.
  `Evaluator.stream(...)` factories on semantic/panoptic are gone —
  use `BackgroundEvaluator(...)` directly or `Evaluator.evaluate_to_partial`
  for DDP. Panoptic `evaluate_to_partial` takes per-image tuples
  rather than the `Dataset/Predictions` pair `evaluate` consumes —
  `PanopticDataset` doesn't yet expose per-image accessors. A future
  ADR may close that asymmetry.
- **Neutral:** the streaming substrate, the `vernier-partial` wire
  format, `FORMAT_VERSION = 2`, paradigm enum, partition-disjointness,
  and the five paradigm-shared `Partial*` exception classes are
  unchanged. ADR-0031 / ADR-0032 decisions all carry over.

## Pros and cons of the options

### Option 1 (status quo + dual hosting)

- 👍 Pros: zero migration; existing scripts keep working unchanged.
- 👎 Cons: every capability now has *two* public homes; doubled
  surface for tests, types, docstrings; running/peek/checkpoint stay
  documented-as-broken. Worst possible long-term cost.

### Option 2 (merge into stateful Evaluator)

- 👍 Pros: maximally minimal — one class per paradigm.
- 👎 Cons: breaks the ADR-0006 immutable-config invariant the
  frozen dataclass enforces today; conflates "what the eval is
  configured to do" with "what state has been accumulated"; touches
  every existing call site of `Evaluator.evaluate`.

### Option 3 (chosen)

- 👍 Pros: minimal break, structured around the user's actual two
  use cases (batch+DDP / in-training). Frozen `Evaluator` stays
  immutable. The `_impl` substrate is unchanged below the FFI; this
  is a 200-line refactor of the public layer, not a kernel rewrite.
- 👎 Cons: the panoptic asymmetry (per-image tuples vs.
  `Dataset/Predictions`) is real until `PanopticDataset` grows
  per-image accessors. We accept that as a follow-up.

## Links and references

- ADR-0006 — immutable evaluator config.
- ADR-0013 — streaming evaluator (this ADR supersedes it; the Rust
  streaming substrate continues to exist as
  `vernier_core::stream::StreamingEvaluator<K>` and is reachable only
  via the new PyO3 functions and `BackgroundEvaluator`).
- ADR-0014 — background evaluator (this ADR amends the public
  surface: snapshot/peek paths removed; resource discipline unchanged).
- ADR-0029 — namespace restructure (per-paradigm submodules).
- ADR-0031 — distributed eval / partial wire format (this ADR amends
  the Python entry class for `to_partial` / `from_partials`; the wire
  format itself is unchanged).
- ADR-0032 — distributed eval across paradigms (same — wire format
  preserved, entry class moves).
