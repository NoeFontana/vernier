# ADR-0006: Threading model — GIL-drop at every PyO3 entry, single-threaded compute for Phase 1

- **Status:** proposed
- **Date:** 2026-04-26
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

Phase 1 ships a synchronous `Evaluator` Python class that wraps the
Rust core through PyO3. ADR-0005 already locked the Rust algorithmic
spine: `Similarity` impls are `Send + Sync`, and `match_image` /
`accumulate` take only owned data. The piece that ADR-0005
intentionally did not address is *what the FFI layer does with
threads* — specifically: does PyO3 hold the GIL across a Rust eval
pass, and where does inter-image parallelism live, if anywhere.

The question matters now because Phase 1 Week 5 is the first slice
that crosses the FFI boundary into Python. Shipping `Evaluator`
without a clear threading contract risks two failure modes:

1. **Holding the GIL across compute.** A naive PyO3 binding holds
   the GIL for the entire eval call. Multi-threaded Python callers
   (training-loop evaluation, robotics agents, `pytest-xdist`
   workers) end up *worse* than they would have been with pure-Python
   pycocotools, because pycocotools' inner loops at least call into
   numpy, which drops the GIL on most array ops. Vernier becoming a
   GIL-bound monolith is a regression on the migration path it claims
   to be.
2. **Drifting into rayon-by-default without an ADR.** Rayon is the
   obvious lever for inter-image parallelism, but it is a top-level
   dependency (per ADR-0001 §"Add or remove a top-level
   dependency") and a behavior change (the eval call now spawns
   threads the caller did not ask for). Adding it implicitly during
   Week 5 — even "just for the inner loop" — bypasses the discipline.

Phase 3 plans a `BackgroundEvaluator` that submits images to a
background worker and yields accumulated results asynchronously
(plan §"Phase 3 Week 3"). That work needs the Rust spine to be
thread-safe and the FFI to release the GIL — but it does not need to
land in Phase 1. The risk is that the Phase 1 binding shape silently
forecloses the Phase 3 design (e.g., by holding `&mut self` on the
evaluator across the call and forcing serialization at the Python
level).

## Decision drivers

- Don't make `vernier.Evaluator` worse than `pycocotools.COCOeval`
  for multi-threaded Python callers.
- Don't add a top-level dep (rayon, crossbeam-channel, tokio) until
  there is a feature that actually requires it. Each is its own
  ADR-0001 trigger when its time comes.
- Phase 3 `BackgroundEvaluator` must drop in without redesigning the
  FFI shape — i.e., the Rust core must already be `Send + Sync`,
  the GIL-drop pattern must already be in place, and the `Evaluator`
  Python class must not hold cross-call state that prevents a worker
  thread from owning a separate evaluator.
- ADR-0001 §"Change the threading model" requires an explicit ADR
  before code lands.
- The FFI layer is supposed to be data-conversion-only (per
  `crates/vernier-ffi/src/lib.rs` policy and the workspace's CLAUDE.md
  guidance). Threading decisions that leak business logic into
  `vernier-ffi` are the wrong shape.

## Considered options

1. **Hold the GIL for the entire call. No internal parallelism.**
   The simplest PyO3 binding shape. Wins on simplicity, loses on
   multi-threaded Python callers.
2. **Drop the GIL on every entry point. Single-threaded Rust
   compute.**
   PyO3's `Python::allow_threads` wraps each entry's compute body.
   The Rust core does no internal threading. Multi-threaded Python
   callers benefit immediately; Phase 3 `BackgroundEvaluator` lands
   without breaking changes.
3. **Drop the GIL + introduce rayon for per-image parallelism in
   Phase 1.** Best per-call latency. New top-level dep; needs its
   own ADR; introduces non-determinism in error reporting if the
   parallel iterators short-circuit.
4. **Async/await via `pyo3-asyncio` and tokio.** Heaviest. Mismatches
   pycocotools' synchronous API. Exposes runtime selection (tokio
   vs. async-std) to the user. Overkill for the migration story.

## Decision outcome

Chosen option: **Option 2 — drop the GIL on every PyO3 entry point;
keep Rust compute single-threaded for Phase 1.**

### Rule

Every `#[pyfunction]` and `#[pymethod]` that runs a non-trivial Rust
algorithm wraps its compute body in `py.allow_threads(...)`. "Non-
trivial" is defined as "anything that loops over images, GTs, or
DTs". Cheap accessors (e.g., reading a config field, returning a
`Vec<f64>` of stats already computed) are exempt — `allow_threads`
has a small but non-zero cost from re-acquiring the GIL on return,
and trivial ops do not benefit.

The wrapped closure must not touch Python objects. This matches our
FFI layering: data conversion happens at the boundary, before
`allow_threads`; the Rust core sees only owned Rust data; results are
converted back to Python on return, after the GIL is re-acquired.

### Cross-call state

`vernier.Evaluator` holds owned data only — the dataset, the params,
and the accumulated results so far. It is *not* a context manager
over a borrowed Rust resource. This makes it safe to construct and
destroy across threads, and lets Phase 3 wrap it in a worker without
a redesign.

The Python wrapper exposes no `&mut self`-style mutation that crosses
the FFI boundary; mutating params is done by reconstructing the
relevant builder, mirroring the public Rust API. Pycocotools' `Params`
mutability is faithfully reproduced inside the *drop-in* `COCOeval`
class only (per ADR-0007), and that class is single-threaded by
contract — users who want concurrency use `Evaluator`.

### Phase 3 forward compatibility

`BackgroundEvaluator` (plan §"Phase 3 Week 3") will be: a Rust
struct holding an `mpsc`-style queue and one or more worker threads
that own per-thread `Evaluator` state. The public Python wrapper has
`submit(image_id, gt, dt)` and `result()` methods. Two things must
already be true when that ships:

1. The Rust core's `Similarity` / `match_image` / `accumulate` are
   callable from any thread (`Send + Sync` per ADR-0005).
2. The PyO3 entry points already drop the GIL — otherwise the
   worker threads serialize on the Python side and the Background
   wrapper is theatre.

Both are true under this ADR. `BackgroundEvaluator` is not
implemented in Phase 1, but its existence is part of this decision —
it is the reason the FFI is shaped this way now rather than the
shape that would be slightly simpler if we knew there were no future
worker thread.

### Internal parallelism

Internal parallelism (rayon, scoped threads, pulp's parallel
abstractions) is **explicitly out of scope for Phase 1**. The 5–10×
performance claim against pycocotools is met by algorithmic and SIMD
wins (per ADR-0003 / ADR-0004), not by parallelism. When a use case
appears that justifies inter-image parallelism inside a single eval
call, it lands as its own ADR (top-level dep + threading-model
change).

### Consequences

- **Positive.** Multi-threaded Python callers benefit on day one
  with no code changes. Phase 3 `BackgroundEvaluator` lands as a
  pure addition. The FFI layer stays data-conversion-only —
  threading is a one-line wrapper, not a logic split.
- **Negative.** Single-threaded compute is slower per-call than a
  rayon-parallelized version would be. Single-image eval already
  fits in the SIMD/ algorithmic budget; multi-image batched eval is
  where parallelism would help, and we accept that gap until a
  follow-up ADR.
- **Neutral.** `py.allow_threads` requires the wrapped closure to be
  `Send`. The `Similarity` and matching APIs are already
  `Send + Sync`, so the constraint is satisfied for free.

## Pros and cons of the options

### Option 1 — Hold GIL, no parallelism

- 👍 Simplest binding code.
- 👎 Regression for multi-threaded Python callers vs. pycocotools.
- 👎 Forecloses Phase 3 BackgroundEvaluator without a redesign.

### Option 2 (chosen) — GIL-drop, single-threaded compute

- 👍 No regression vs. pycocotools for multi-threaded callers.
- 👍 Phase 3 BackgroundEvaluator lands as pure addition.
- 👍 No new top-level dep this week.
- 👎 Per-call latency is bounded by single-threaded compute until a
  later ADR introduces internal parallelism.

### Option 3 — GIL-drop + rayon now

- 👍 Best per-call latency for batched eval.
- 👎 New top-level dep without a use case yet — bypasses ADR-0001.
- 👎 Non-determinism in error reporting if parallel iterators short-
  circuit on different elements between runs.

### Option 4 — Async/await via pyo3-asyncio

- 👍 Most flexible threading model.
- 👎 Mismatches pycocotools synchronous API.
- 👎 Exposes runtime selection (tokio/async-std) to the user.
- 👎 Pulls in a heavy dep tree for capability we don't need.

## Links and references

- ADR-0001 — Record architecture decisions (§"Change the threading
  model", §"Add or remove a top-level dependency").
- ADR-0003 — `pulp` for stable-Rust SIMD with runtime CPU dispatch.
- ADR-0005 — Lock the `Similarity` trait and matching-engine API
  for Phases 1–3 (`Send + Sync` requirement).
- ADR-0007 — `patch_pycocotools` policy (the drop-in's relationship
  to threading).
- PyO3 docs — [`Python::allow_threads`](https://docs.rs/pyo3/latest/pyo3/marker/struct.Python.html#method.allow_threads).
