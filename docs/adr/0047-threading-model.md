# ADR-NNNN: Opt-in `num_threads` parallelism for `Evaluator` and `BackgroundEvaluator` — zero overhead on the default path

- **Status:** accepted
- **Date:** 2026-05-16
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

[ADR-0006](0006-threading-model.md) chose GIL-drop + single-threaded
Rust compute for Phase 1 and explicitly reserved a slot for this ADR:
"when a use case appears that justifies inter-image parallelism inside
a single eval call, it lands as its own ADR (top-level dep + threading-
model change)." [ADR-0014](0014-background-evaluator.md) reinforced
that stance for the in-training persona: "Bound the CPU footprint at
one core … single-worker, no internal parallelism inside the worker."
Both decisions were correct in their context and remain correct as
defaults — they encode the asymmetry that eval must not preempt the
trainer.

The use case ADR-0006 was waiting for has now landed in real harnesses.
Two personae share a hot path that single-threaded compute does not
serve well:

1. **End-of-epoch / CI / paper batch eval.** `Evaluator.evaluate(gt,
   dt)` on COCO val2017 holds at ~5 s on AMD EPYC-Milan
   ([`docs/benchmarks.md`](../benchmarks.md), bbox segm cells); LVIS
   sits in the tens of seconds; large custom benchmarks (50k+ images)
   are minutes. The user is sitting in front of a checkpoint waiting
   for the gate to clear. They have 16–128 idle cores in a box that's
   not currently training anything, and no way to spend them.
2. **Val-loader-driven streaming eval.** `BackgroundEvaluator` is
   increasingly used for the dedicated-validation-pass shape: the user
   spins up a `val_loader`, runs the model, `submit()`s every batch,
   `finalize()`s at the end. There is no trainer competing for cores
   in this loop — the model is doing inference (typically GPU-bound),
   not training. ADR-0014's one-core bound makes eval a serial
   bottleneck behind a parallel data path. The persona ADR-0014 was
   protecting against (training + eval contending for the same N
   cores) is not the persona who reaches for `BackgroundEvaluator` as
   a streaming validator.

The Phase 1–3 work that ADR-0006 was waiting to finish is finished.
The spine is `Send + Sync` (ADR-0005). The FFI drops the GIL via
`py.detach()` at every non-trivial entry point. The distributed-merge
machinery ([ADR-0031](0031-dist-eval.md),
[ADR-0032](0032-dist-eval-across-paradigms.md)) already proves
strict-mode bit-equality across partitions of per-image work — which
is exactly the property a parallel matching pass needs to preserve.
The streaming substrate (ADR-0013, surfaced via ADR-0035) is the
parallelism target on the `BackgroundEvaluator` side: each worker
update is one batched `update_parsed` call on the shared
`StreamingEvaluator<K>`.

What is not yet decided is the shape of the opt-in. This ADR is
constrained on four sides.

- **Zero observable cost on the default path.** Existing strict-mode
  CI gates, parity tests, and downstream callers must observe no
  measurable change in wall-time, allocations, or output bits when
  they do not pass `num_threads`. The pyo3 entry points cannot grow a
  branch that allocates a rayon pool, even an empty one. The release
  wheel cannot ship a runtime that pays a per-call check that
  short-circuits to "no". Today's benchmark numbers in
  [`docs/benchmarks.md`](../benchmarks.md) must reproduce byte-for-
  byte after this ADR lands.
- **Strict-mode parity preserved.** ADR-0002 strict mode is bit-
  identical to pycocotools. Adding parallelism cannot widen that
  contract by a single ULP. The parity harness must gain a thread-
  count axis and assert vernier-vs-vernier bit-equality across it,
  alongside the existing vernier-vs-pycocotools assertion.
- **Simple, single-source configuration.** One knob, one name, same
  semantics on the library API, on `BackgroundEvaluator`, on the CLI,
  and on the environment-variable fallback. No global mutable state
  (`vernier.set_num_threads(...)`), no implicit pickup of an unrelated
  library's pool (`RAYON_NUM_THREADS`), no Cargo feature flip that
  changes runtime behavior.
- **ADR-0014's bounded-resource discipline stays the default.** The
  trainer persona must observe exactly today's behavior. The val-
  loader persona must be able to opt out of the one-core bound with a
  single explicit kwarg.

The risk shape is mixed. This is a public-API ADR
([ADR-0001](0001-record-architecture-decisions.md) §"Affect the public
API"), an FFI ADR (§"Cross the FFI boundary"), a threading-model ADR
(§"Change the threading model"), a top-level-dep ADR (§"Add or remove
a top-level dependency" — rayon), and a parity-adjacent ADR (the
strict-mode contract has to survive the parallel reduction). One ADR
triggers five of ADR-0001's six gates.

## Decision drivers

- **Zero overhead on the default path — at runtime.** The
  `num_threads=None` callsite executes the exact instruction stream it
  does today: no rayon pool built, no `ThreadPoolBuilder` invocation,
  no atomic check on a "should I parallelize" flag inside the per-
  image loop, no detour through a parallel scheduler. The contract is
  about *runtime cost on the sequential path*, not about whether
  parallel code is linked into the binary — rayon is always present,
  always linked, and the parallel module is always compiled. One
  build, one published wheel, one strict-mode parity contract. The
  zero-overhead claim is a runtime property of the sequential path,
  not a compile-time feature of the build. This is the load-bearing
  driver and it constrains every other choice.
- **Strict-mode bit-equality across `num_threads`.** For every
  workload in the parity harness, `evaluate(..., num_threads=N)`
  produces stats bit-equal to `evaluate(..., num_threads=None)` for
  all `N` and every parity mode. This is a stronger property than
  vernier-vs-pycocotools — it's vernier-vs-itself — and it is the
  only check that catches a future refactor introducing a parallel
  reduction in the wrong place.
- **One knob, four touchpoints.** `num_threads: int | None = None` on
  the library `evaluate(...)`, on `Evaluator.background(...)`, on the
  `BackgroundEvaluator(...)` constructor; `--threads N` on the CLI;
  `VERNIER_NUM_THREADS=N` as a fallback consulted only when the
  library/CLI arg is unset. Same name. Same semantics. No
  `set_num_threads`, no per-evaluator config object whose threading
  field has to be looked up in three places.
- **Scoped thread pool, never global.** Rayon's default global pool is
  shared with every other crate that uses rayon in-process (polars,
  pyo3-polars, plenty of others). Two libraries fighting over one
  pool is a priority-inversion hazard under torch and OpenBLAS, and
  it makes per-call `num_threads` ineffective by construction. Pools
  in this ADR are scoped to the call that asked for them.
- **ADR-0014's resource discipline survives by being the default.**
  The `BackgroundEvaluator` default stays one worker, no internal
  parallelism. The trainer persona measures zero change. Opt-in only
  for the val-loader persona who has explicitly named a different
  resource policy.
- **Per ADR-0001:** §"Affect the public API" (new kwarg on every
  paradigm's `Evaluator.evaluate` / `background`, new CLI flag),
  §"Cross the FFI boundary" (PyO3 entry signatures), §"Change the
  threading model" (ADR-0006's reserved slot), §"Add or remove a top-
  level dependency" (rayon), §"Set a project-wide convention"
  (`num_threads` naming, env-var precedence).

## Considered options

This ADR has four orthogonal axes. Each is decided independently; the
chosen design is the combination of one option per axis.

### Axis A — Where parallelism lives in the call graph

1. **Inside `match_image`.** Parallelize the inner `O(T·D·G)` triple
   loop per (image, category) cell.
2. **Across (image, category) cells in `evaluate_with` and
   `StreamingEvaluator::update_parsed`.** Each call to `match_image`
   produces an owned `PerImageEval`; the cells store is write-once-
   per-cell; `accumulate()` consumes them in one serial pass
   afterwards.
3. **Inside `accumulate()`.** Parallel AP fold across `(K, A, M)`
   cells; or — more aggressively — parallel merged-stream sort.
4. **Multi-worker `BackgroundEvaluator`.** N workers each owning a
   `StreamingEvaluator<K>` with a unique `rank_id`; submissions hash
   by `image_id`; `finalize()` merges partials via the ADR-0031 wire
   format.

### Axis B — API knob shape

1. **One `num_threads: int | None = None` kwarg, threaded through to
   every entry point that does inter-image work; env-var fallback.**
   `None` (or `1`) is sequential and zero-overhead; `0` is auto via
   `std::thread::available_parallelism()`; `n ≥ 2` is fixed.
2. **Global setter (`vernier.set_num_threads(n)`).** Process-wide
   default; per-call kwarg overrides.
3. **Cargo feature `parallel` flipped on default-parallel.** Compile-
   time choice; users who want sequential build without `parallel`.
4. **Environment variable only (`VERNIER_NUM_THREADS` /
   `RAYON_NUM_THREADS`).** No API surface change; threading is a
   deployment knob.

### Axis C — Thread pool scope and lifetime

1. **Global rayon pool.** First call to any rayon API initializes the
   process-wide pool; `num_threads` is the size hint.
2. **Scoped per-call pool.** Each `evaluate(..., num_threads=n)` call
   builds a fresh `ThreadPool` via `ThreadPoolBuilder::new()`,
   `install`s the par_iter inside it, drops it on return.
3. **Pool owned by `Evaluator` / `BackgroundEvaluator`.** Pool is
   constructed at evaluator-construction time, reused across all
   `evaluate` calls on that evaluator, dropped with the evaluator.

### Axis D — `BackgroundEvaluator` parallelism shape

1. **Inner `num_threads` in the existing single worker.** The single
   worker thread is unchanged in count; it dispatches each batched
   `submit()`'s per-image matching through a scoped rayon pool of
   `num_threads`. Within the worker, accumulation stays serial.
2. **Outer `num_workers`.** N worker threads each owning a
   `StreamingEvaluator<K>` with a distinct `rank_id`; deterministic
   image-id-hash routing; `finalize()` merges partials.
3. **Both, layered.** `num_workers × num_threads_per_worker` total
   parallelism. The user composes them.

## Decision outcome

Chosen combination: **A2 + B1 + C2 + D1**, with **D2 explicitly
deferred to a follow-up ADR** rather than ruled out.

### The rule

Every public entry point that runs a non-trivial inter-image loop
accepts a `num_threads: Option<usize>` parameter (Python:
`num_threads: int | None = None`) with the following semantics:

| `num_threads` | Behavior |
|---|---|
| `None` (default) | Sequential. The runtime path is identical to today — no rayon code is entered, no pool is built, no per-iter dispatch overhead. The env-var fallback is consulted *here only*, before entering the compute path; if `VERNIER_NUM_THREADS` is set in the environment, its value replaces `None`. |
| `1` | Sequential. Same code path as `None`. The kwarg-vs-env distinction is documented but produces identical bits. |
| `n ≥ 2` | A scoped `rayon::ThreadPool` of exactly `n` threads is built around the inter-image loop, the work is dispatched via `par_iter` into a pre-sized output, and the pool is dropped on return. The pool is *not* the rayon global pool. |
| `0` | Auto. Resolved to `std::thread::available_parallelism().unwrap_or(NonZeroUsize::new(1))`, which is cgroup-aware on Linux. Subsequent behavior matches `n ≥ 2`. |

The same parameter, with the same name and the same semantics, is
exposed on every paradigm's `Evaluator.evaluate(...)`,
`Evaluator.evaluate_to_partial(...)`, `Evaluator.from_partials(...)`,
`Evaluator.background(...)`, and `BackgroundEvaluator(...)`
constructor. The CLI grows `--threads N` with default `1`. The env-var
fallback (`VERNIER_NUM_THREADS`) is consulted only when the API/CLI
arg is `None`, and the resolved value is logged once at the call site
with the line `vernier: num_threads=N (from VERNIER_NUM_THREADS)` so
the user can see the indirection in their job logs.

### Zero-overhead implementation contract

The sequential path is gated at the FFI boundary, not inside the Rust
core. The PyO3 signature is conceptually:

```rust
#[pyfunction]
fn evaluate(py: Python<'_>, gt: &Bound<'_, PyAny>, dt: &Bound<'_, PyAny>,
            num_threads: Option<usize>) -> PyResult<Summary> {
    let parsed = parse_inputs(py, gt, dt)?;          // GIL held
    py.detach(|| {
        match resolve_threads(num_threads) {
            ThreadPolicy::Sequential => evaluate_sequential(parsed),
            ThreadPolicy::Pool(n)    => evaluate_parallel(parsed, n),
        }
    })
}
```

`ThreadPolicy::Sequential` is the chosen variant for `None`, `Some(1)`,
and the env-var-resolved-to-1 case. `evaluate_sequential` is the exact
function called today, with the exact same signature it has today, in
the exact same `evaluate.rs` module. No rayon code is entered along
that path at runtime — the `match` falls through to the sequential
arm before the parallel arm's first instruction. `evaluate_parallel`
lives in `crates/vernier-core/src/evaluate_parallel.rs` and is
**always compiled in**. Rayon is an unconditional top-level
dependency, linked into every release build of the wheel. There is no
`parallel` Cargo feature, no compile-time toggle that changes whether
the parallel symbols exist, no build configuration in which
`num_threads > 1` is a typed error rather than a runtime dispatch.
The published wheel is one build, behaving one way per `num_threads`
value, with one strict-mode parity contract — and the zero-overhead
claim is verified by benchmark, not by the absence of code.

This is the principal-engineer move on this axis: Cargo features that
gate behavior fragment the parity contract ("strict-mode bit-equal
for this build of the wheel") and force downstream consumers to
choose between two slightly different vernier binaries. We pay the
~1.2 MB link-time cost once, unconditionally, and the runtime cost
exactly when and only when the user explicitly asks for parallelism.

Per-call pool construction cost is real (~50–200 µs on modern Linux);
it is amortized across the matching pass (seconds to tens of seconds
on the workloads that motivate this ADR) and is paid only by callers
who explicitly asked. There is no shared pool to leak across calls,
no pool to forget to drain on shutdown, no pool to coordinate with
torch / polars / pyo3-polars / any other rayon-using crate in the
same process.

### Parallel axis: per-image matching, serial accumulate

The chosen parallel axis (A2) is over (image, category) cells in
`evaluate_with` and per-image cells in
`StreamingEvaluator::update_parsed`. Each thread executes
`match_image` against owned inputs and writes its
`PerImageEval` into a pre-sized `Vec<Option<PerImageEval>>` indexed by
image-position, never into shared mutable state. After the par_iter
returns, the calling thread runs `accumulate()` exactly as it does
today — the same merged-stream sort, the same A1 stable-sort
tiebreak, the same C1 `partition_point` bucket lookup. Strict-mode
output is bit-identical to the sequential path *by construction*, not
by a tolerance argument: the cells store is order-equal across thread
counts, and the function consuming it is unchanged.

Three paradigm-specific notes:

- **Instance (bbox / segm / boundary / keypoints).** Cells store is
  indexed by `(K, A, image-position)`. Strict-mode bit-equality across
  `num_threads` is automatic. The matching-scaling doc
  (`docs/engineering/matching-scaling.md`) establishes per-image work
  is well-behaved O(T·D·G) — well-suited to par_iter dispatch at the
  image granularity. The inner `match_image` triple loop is *not*
  parallelized (option A1 rejected): per-image G·D is small on real
  workloads (val2017 median G·D = 1), and per-task overhead would
  dominate.
- **Semantic.** The confusion-matrix `u64` fold is associative and
  commutative; tree-reductions across threads are bit-equal to the
  serial fold trivially. Each thread maintains a thread-local
  `n_classes × n_classes` accumulator; the merge is a u64-additive
  reduction. No `retain_per_image_deltas` machinery needed; no
  constraints on routing.
- **Panoptic.** The per-image `PqStat` fold is `f64`-additive and
  *not* associative across orderings. Strict-mode bit-equality across
  `num_threads` is preserved via the same mechanism ADR-0031 §"Strict-
  mode merge" specifies: per-image deltas are retained, re-sorted by
  `image_id`, and re-summed in canonical order at the end. This is
  opt-in today (`retain_per_image_deltas=True`); for `num_threads > 1`
  on the panoptic surface, the flag is forced to `True` and a one-
  shot info-level log explains why. Without it, strict-mode users
  observe a corrected-tier 4-ULP envelope — louder than silent, but
  not bit-equal. The parity harness asserts the stronger property
  (bit-equality) under the forced flag.

`accumulate()` is *not* parallelized in this ADR (option A3 rejected).
The merged-stream score-descending sort is global and not naturally
parallel; parallelizing the `(K, A, M)` cell fan-out would buy a
small constant and risks introducing a parallel f64 reduction that
breaks strict-mode equality. The deferred "rayon AP fold" listed in
ADR-0033 §"Deferred (explicitly not in scope here)" is its own ADR
with its own parity story (likely corrected-tier only).

### `BackgroundEvaluator` shape

The `BackgroundEvaluator` constructor grows one parameter:
`num_threads: int | None = None`. Semantics are identical to the
batch-side knob. The single-worker thread is unchanged in count and
in lifecycle. When `num_threads > 1`, the worker's `update_parsed`
call dispatches per-image matching through a scoped rayon pool of
`num_threads` threads (pool is built once at construction and owned
by the worker, *not* per-`submit` — the worker is one long-lived
thread, the pool's lifetime is the worker's lifetime). Inside-worker
parallelism is bounded by the pool size; the worker's own thread
contributes one of those `num_threads` (rayon's main-thread-as-worker
convention).

ADR-0014's resource discipline survives by being the default:
`BackgroundEvaluator(gt)` is exactly today's behavior — one core, one
worker, sequential `update_parsed`, single-thread accumulate. The
training-loop persona never sees a change. The val-loader persona
writes `BackgroundEvaluator(gt, num_threads=8)` and gets a worker
that uses up to 8 cores during each `submit()`'s matching pass and
zero cores between submits. The CPU footprint is bounded *at the user's
explicit choice*, not at one core regardless.

`worker_nice` (ADR-0014 §"Worker scheduling") still applies to the
worker thread; the rayon-pool threads inherit the worker's nice value
by default on Linux (it's a process attribute through `setpriority`)
and are explicitly re-niced on platforms where it isn't.
`worker_affinity` is **not** propagated to pool threads — pinning all
N threads to one CPU defeats the purpose; pool threads run unpinned
unless the user explicitly composes affinity above the API.

### `num_workers` deferred — but the shape is documented

Multi-worker `BackgroundEvaluator` (Axis D2) is *deferred* to a
follow-up ADR rather than rejected. The shape is clear and the
substrate is in place:

- Each worker owns its own `StreamingEvaluator<K>` with a distinct
  `rank_id`.
- Submission routing hashes `image_id` to a fixed worker (deterministic
  routing — same image_id always lands on the same worker within one
  evaluator instance).
- `finalize()` collects partial wire-format blobs from every worker
  and merges via `Evaluator::from_partials(...)` (ADR-0031 /
  ADR-0032 / ADR-0035 surface).
- Strict-mode bit-equality is preserved by `rank_id` ordering for
  instance, by per-image-delta re-sort for panoptic, unconditionally
  for semantic — exactly the existing distributed-merge story, just
  in-process.

The reason for deferral is footgun risk: two-level parallelism
(`num_workers=4, num_threads=8` → 32 threads of potential
oversubscription) needs measurement on a real val-loader workload
before defaults can be picked. The single-worker + inner
`num_threads` shape lands first; the multi-worker extension follows
when there is a measured workload that the single-worker shape
underserves.

### Configuration discipline

A worked example of the four touchpoints and their precedence:

```python
# Library: one knob, threaded through every entry point
vernier.instance.Evaluator(iou=Bbox()).evaluate(gt, dt)                        # sequential, zero overhead
vernier.instance.Evaluator(iou=Bbox()).evaluate(gt, dt, num_threads=8)         # 8 threads, scoped pool
vernier.instance.Evaluator(iou=Bbox()).evaluate(gt, dt, num_threads=0)         # available_parallelism (cgroup-aware)
vernier.instance.Evaluator(iou=Bbox()).background(gt, num_threads=8)           # 1 worker, 8 inner threads

vernier.semantic.Evaluator(parity_mode="strict").evaluate(ds, preds, num_threads=8)
vernier.panoptic.Evaluator(parity_mode="strict").evaluate(ds, preds, num_threads=8)
#   ^ panoptic strict + num_threads > 1 forces retain_per_image_deltas=True; logs once.

# CLI: same name, same semantics
vernier eval --threads 8 ...                                                   # 8 threads
vernier eval --threads 0 ...                                                   # auto
vernier eval ...                                                               # default --threads 1, sequential

# Env var: fallback only, consulted when API/CLI arg is None
VERNIER_NUM_THREADS=8 python my_script.py
#   ^ takes effect only at evaluate(...) call sites that pass num_threads=None;
#     CLI runs override it because --threads defaults to 1, not None.
```

Precedence is documented as a single rule: **explicit beats implicit**.
Library kwarg `>` CLI flag `>` `VERNIER_NUM_THREADS` `>` default
(sequential). `RAYON_NUM_THREADS` is **never** consulted — vernier
does not use the global rayon pool, and inheriting an unrelated
library's deployment knob would silently change vernier's behavior.

### Composability

- **Re-entry detection.** If `rayon::current_thread_index().is_some()`
  when an entry point is called (the caller is itself running inside
  a rayon worker), vernier falls back to the sequential path and
  emits a one-time `UserWarning`. Oversubscription is the failure
  mode, not deadlock — but the failure mode is bad enough to refuse.
- **Cgroup awareness.** `num_threads=0` resolves through
  `std::thread::available_parallelism()`, which respects Linux cgroup
  v1/v2 CPU quotas since std 1.59. Container deployments get the
  right count without surprises.
- **torch / OpenBLAS / MKL.** Documented as an interaction, not
  enforced. The how-to guide recommends `num_threads = N // 2` when
  running alongside a torch model with default thread counts; the
  bench harness records both numbers for reproducibility.

### Determinism contract

This ADR extends the determinism table introduced by ADR-0013 §
"Determinism contract" and amended by ADR-0031 §"Determinism
contract" with one new column: `num_threads`-order invariant.

| Surface | Stream-order | Rank-order | `num_threads`-order | Bit-equals batch |
|---|:---:|:---:|:---:|:---:|
| `Evaluator.evaluate(..., num_threads=N)` (instance, strict) | n/a | n/a | **yes** | yes |
| `Evaluator.evaluate(..., num_threads=N)` (semantic, strict) | n/a | n/a | **yes** (u64 additive) | yes |
| `Evaluator.evaluate(..., num_threads=N)` (panoptic, strict, `retain_per_image_deltas=True`) | n/a | n/a | **yes** | yes |
| `Evaluator.evaluate(..., num_threads=N)` (any paradigm, corrected) | n/a | n/a | 4-ULP (ADR-0004) | yes-on-tolerance |
| `BackgroundEvaluator(num_threads=N).submit().finalize()` (any paradigm) | yes (single-writer worker) | n/a | matches `Evaluator.evaluate` above | yes |

The new column reads: "for fixed inputs, does varying `num_threads`
change the output bits?". For strict mode, the answer is **no** by
construction across all three paradigms, with the documented panoptic
caveat about `retain_per_image_deltas`.

### Test plan

The parity harness extends along the new axis additively. Each
existing fixture runs at `num_threads ∈ {None, 1, 2, 4, 8}` and
asserts:

1. **Strict-mode bit-equality across thread counts** (the new
   property). `evaluate(..., num_threads=N).stats` is bit-equal to
   `evaluate(..., num_threads=None).stats` for every N and every
   fixture. This is the load-bearing parity test for this ADR; it
   catches the failure mode of "future refactor accidentally
   introduces a parallel f64 reduction in the matching path".
2. **Strict-mode bit-equality vs pycocotools.** Unchanged; runs at
   `num_threads=None` as today and at every other thread count. The
   parallel path passes the same oracle check that the sequential
   path does.
3. **`num_threads=0` auto-resolution.** A fixture sets cgroup CPU
   quota to 2 cores and asserts the resolved thread count is 2, not
   the host's full core count.
4. **Re-entry warning.** A fixture calls `evaluate(num_threads=4)`
   from inside a rayon par_iter and asserts the `UserWarning` fires
   and the inner call took the sequential path.
5. **Panoptic strict-mode flag forcing.** A fixture constructs a
   panoptic evaluator with `retain_per_image_deltas=False,
   parity_mode="strict", num_threads=4` and asserts the flag is
   silently forced to `True`, the info-level log line fires once, and
   the result is bit-equal to a single-threaded run.
6. **`BackgroundEvaluator(num_threads=N)` matches batch.** A fixture
   runs `BackgroundEvaluator(num_threads=N).submit().finalize()` and
   asserts the result is bit-equal to `Evaluator.evaluate(..., num_threads=N)`
   on the same inputs — the substrate is shared, the property carries
   through.

The bench harness ([ADR-0017](0017-local-bench-harness.md),
[ADR-0033](0033-multi-paradigm-bench.md)) gains a `num_threads`
parameter on its workload definitions. The published scaling table
runs `{1, 2, 4, 8, 16}` on the existing val2017 / LVIS / panoptic-
val2017 / ADE20K cells. The headline number is the scaling curve, not
a single speedup ratio.

### Consequences

- **Positive.** The val-loader and end-of-epoch personae get the
  performance they're asking for, with no migration cost for existing
  callers. The single-thread path is observably identical to today —
  bit-equal output, bit-equal benchmark numbers, zero new symbols
  exercised at runtime. The `num_threads` knob composes uniformly
  across batch / streaming / background / CLI / env-var surfaces.
  Strict-mode parity is stronger than before: the harness now asserts
  vernier-vs-itself across thread counts in addition to vernier-vs-
  pycocotools. The deferred multi-worker
  `BackgroundEvaluator` follow-up lands as a pure composition over
  ADR-0031, ADR-0032, ADR-0035 — no spine rework needed.
- **Negative.** Rayon enters the top-level dependency set
  unconditionally — every release build of the wheel links it,
  whether the user ever passes `num_threads > 1` or not. The wheel
  grows by ~1.2 MB (rayon + crossbeam-deque + crossbeam-utils,
  measured on x86_64-unknown-linux-gnu, release profile), paid by
  every user. This is the deliberate trade for one wheel, one
  behavior, one strict-mode parity contract — and it is the right
  trade given the size budget is set in MB, not in tens of MB. The
  parity harness's runtime grows by ~4× on the strict-mode pass
  (five `num_threads` cells per fixture). A future `accumulate()`
  parallelization is foreclosed until its own ADR addresses the
  f64-reduction-order question. Panoptic users hit a strict-mode-
  vs-`retain_per_image_deltas` interaction that didn't exist before;
  mitigated by the forced-flag policy and the one-shot log, but it
  is genuinely one more thing to know.
- **Neutral.** The PyO3 entry signatures grow one keyword. The
  `vernier-ffi` policy of "data conversion only" is preserved — the
  match on `ThreadPolicy` is one branch at the boundary, not business
  logic. `vernier-core` grows a sibling module `evaluate_parallel.rs`,
  always compiled, always linked; the existing `evaluate.rs` is
  unchanged. No new Cargo feature is introduced.
  `RAYON_NUM_THREADS` is documented as explicitly ignored.

## Pros and cons of the options

### Axis A — Where parallelism lives

**A1 — inside `match_image`.**

- 👍 Highest possible per-image throughput.
- 👎 Per-image G·D is small on real workloads (val2017 median G·D = 1,
  99% of wall time in cells with G·D < 256 per
  `docs/engineering/benchmarking/2026-05-bbox-cdf.md`). Per-task
  rayon overhead would dominate; expected speedup is negative on
  median-shaped workloads.
- 👎 Edits `matching.rs`. ADR-0005's no-edits-to-spine invariant is
  load-bearing for parity confidence and would be violated.

**A2 — across (image, category) cells (chosen).**

- 👍 Embarrassingly parallel axis. Per-image work is independent.
  Cells store is write-once-per-cell at known indices, no shared
  mutable state, no synchronization in the hot path.
- 👍 `accumulate()` runs serially on the collected cells store
  afterwards — bit-equal to the sequential path by construction.
- 👍 Composes uniformly with `StreamingEvaluator::update_parsed` (the
  per-submit batch is the parallel work unit) and therefore with
  `BackgroundEvaluator`.
- 👎 Bounded by the slowest image in each batch (no work-stealing
  benefit on very unbalanced workloads — but rayon's work-stealing
  scheduler mitigates this without further effort).

**A3 — inside `accumulate()`.**

- 👍 The remaining single-threaded portion of the pipeline; cleaning
  it up would lift the asymptote on highly-multi-core boxes.
- 👎 The merged-stream score-descending sort is global. Parallelizing
  it is a hard problem with no off-the-shelf solution that preserves
  the A1 tiebreak.
- 👎 The `(K, A, M)` cell fan-out is f64-additive; a parallel
  reduction would break strict-mode bit-equality unless carefully
  ordered, and "carefully ordered" is the kind of constraint that
  silently regresses.
- 👎 Deferred to its own ADR (ADR-0033 §"Deferred").

**A4 — multi-worker `BackgroundEvaluator`.**

- 👍 The right shape for absorbing bursty submissions when matching
  is the bottleneck.
- 👎 Two-level parallelism (num_workers × inner num_threads) is a
  measurement-required regime. Defaults cannot be chosen without
  workload data. Deferred to a follow-up ADR for which the substrate
  this ADR ships is the prerequisite.

### Axis B — API knob shape

**B1 — `num_threads` kwarg + env-var fallback (chosen).**

- 👍 One knob, one name, one semantics across four surfaces. Explicit
  at every call site. Type-checkable.
- 👍 The kwarg's `None` default gives the zero-overhead path a clean
  type-level home.
- 👍 Env-var fallback is the deployment-knob ergonomic story without
  introducing process-wide mutable state.
- 👎 Every entry point's signature grows one parameter. Mitigated by
  the kwarg being the last positional / keyword-only — backward-
  compatible in Python, defaults-preserving in Rust.

**B2 — global `vernier.set_num_threads(n)`.**

- 👍 One call, all subsequent evals affected.
- 👎 Global mutable state. Tests interact via the global. Multi-
  tenant processes (a service serving multiple eval calls from
  multiple users) cannot disambiguate. Sets the same precedent rayon
  itself sets and that has caused real bugs in polars / pyo3-polars
  callsites that share rayon's pool with vernier — the precise
  precedent we want to *not* repeat.
- 👎 No good answer to "which call's setting wins?".

**B3 — Cargo feature `parallel` default-parallel.**

- 👍 Smallest API surface — no new kwarg.
- 👎 Compile-time choice; users who want sequential cannot opt out
  on the published wheel.
- 👎 Changes runtime behavior across two release builds of the same
  source. The bench harness now needs to declare which build it ran.
  Strict-mode parity becomes "for this Cargo build" rather than "for
  the published wheel". Bad shape.

**B4 — env var only.**

- 👍 Deployment-only knob; no library-API growth.
- 👎 Invisible from a script. The library has no way to assert "I
  want N threads on this call". CI tests cannot pin a thread count
  without process-wide environment manipulation. Loses the
  bit-equality-across-thread-counts harness's clean parametrization.

### Axis C — Pool scope

**C1 — global rayon pool.**

- 👍 Smallest implementation. One pool, initialized lazily.
- 👎 Shared with every other rayon-using crate in-process. polars,
  pyo3-polars, ndarray-rayon, plenty of others all draw from the
  same pool. Per-call `num_threads` cannot constrain a global pool
  without resizing it process-wide on every call — and resizing is
  not a supported rayon operation. The hazard is real and documented
  on the rayon issue tracker.
- 👎 `RAYON_NUM_THREADS` becomes a side channel that mutates
  vernier's behavior. We want the opposite.

**C2 — scoped per-call pool (chosen for batch; per-worker for background).**

- 👍 No shared state. `num_threads` constrains exactly the call /
  worker that asked for it. Other rayon users are unaffected.
- 👍 The `num_threads` kwarg is the pool's only configuration source.
  No env-var side channels.
- 👎 Pool construction cost (~50–200 µs) per call on the batch
  surface. Amortized across seconds-to-minutes of matching work; not
  measurable above the bench harness's IQR floor (5% relative).
- 👎 Idle pool threads after pool drop are returned to the OS via
  rayon's own shutdown path; no leakage.

**C3 — pool owned by `Evaluator`.**

- 👍 Pool reused across multiple `evaluate(...)` calls on the same
  evaluator.
- 👎 `Evaluator` is a frozen-dataclass-style config object
  (ADR-0006 §"Cross-call state", ADR-0035 invariant). It does not own
  resources. Giving it pool ownership requires either making it a
  context manager or accepting that pool lifetime is tied to GC
  finalization, which is exactly the resource-leak shape ADR-0014
  worked hard to avoid for the worker thread. C3 is the right shape
  on `BackgroundEvaluator` (where pool lifetime == worker lifetime
  is a natural fit, and that's what this ADR adopts there) and the
  wrong shape on the batch `Evaluator`.

### Axis D — `BackgroundEvaluator` parallelism

**D1 — inner `num_threads` in the single worker (chosen).**

- 👍 ADR-0014's bounded-resource discipline survives by being the
  default. Trainer persona observes zero change. Val-loader persona
  opts in with one kwarg.
- 👍 Pool lifetime == worker lifetime: pool built at construction,
  dropped at `finalize()` / context-manager exit. One pool, not one-
  per-submit.
- 👍 Strict-mode bit-equality is the same property the batch path
  guarantees; no new parity story.
- 👎 Per-submit batch size bounds the parallel speedup. Val loaders
  with batch size 1 see no benefit from this knob and need D2.

**D2 — outer `num_workers` (deferred follow-up).**

- 👍 Absorbs bursty submission when single-worker matching is the
  bottleneck.
- 👍 The strict-mode bit-equality story is already done (ADR-0031
  rank-id-ordered merge); this is pure composition.
- 👎 Two-level parallelism is a measurement-required regime. Default
  picking and the interaction-with-D1 documentation are too much for
  one ADR with five ADR-0001 triggers already.
- 👎 Deferred until a real val-loader workload exists where the D1
  shape is insufficient.

**D3 — both, layered.**

- 👍 Maximum throughput on big iron.
- 👎 Cannot land without D2 measurements; same deferral applies.

## Links and references

- [ADR-0001](0001-record-architecture-decisions.md) — Record
  architecture decisions. This ADR triggers §"Affect the public API",
  §"Cross the FFI boundary", §"Change the threading model", §"Add or
  remove a top-level dependency" (rayon), and §"Set a project-wide
  convention" (`num_threads` naming + env-var precedence).
- [ADR-0002](0002-three-tier-parity-model.md) — Three-tier parity
  model. The strict-mode contract is the load-bearing parity check
  this ADR must not widen.
- [ADR-0004](0004-numerical-layout-policy.md) — Numerical layout
  policy. The 4-ULP aligned-tier envelope bounds the corrected-mode
  wobble; this ADR introduces no new tolerance class.
- [ADR-0005](0005-similarity-trait-and-matching-engine-api.md) —
  Lock the `Similarity` trait and matching-engine API.
  `Send + Sync` is the precondition this ADR consumes. `matching.rs`
  is not edited (rejection of option A1).
- [ADR-0006](0006-threading-model.md) — Threading model. This ADR
  fills the slot ADR-0006 reserved ("when a use case appears that
  justifies inter-image parallelism inside a single eval call, it
  lands as its own ADR"). The GIL-drop discipline carries through
  unchanged; `py.detach()` wraps both the sequential and the
  parallel paths.
- [ADR-0013](0013-streaming-evaluator.md) — Streaming evaluator. The
  per-submit batch matching inside `update_parsed` is the parallel
  unit on the background surface.
- [ADR-0014](0014-background-evaluator.md) — `BackgroundEvaluator`.
  This ADR amends the resource-budget contract: ADR-0014's one-core
  bound becomes the default rather than the rule; the val-loader
  persona opts in to break it via `num_threads`.
- [ADR-0017](0017-local-bench-harness.md) — Local bench harness.
  Gains a `num_threads` axis on workload definitions; publishes the
  scaling table.
- [ADR-0020](0020-parsed-once-dataset-handle.md) — Parsed-once
  `Dataset` handle. Foresaw this ADR ("`Arc<DerivationCache>` is the
  share-across-workers carrier ADR-0006 anticipated for a future
  parallelization story"); the per-kernel GT-side derivation cache
  is shared across rayon worker threads via `Arc` clone, no
  duplication.
- [ADR-0031](0031-dist-eval.md) — Distributed eval / partial wire
  format. The substrate for the deferred multi-worker
  `BackgroundEvaluator` extension (Axis D2).
- [ADR-0032](0032-dist-eval-across-paradigms.md) — Distributed eval
  across paradigms. The paradigm-specific strict-mode merge stories
  this ADR inherits: instance preserves bit-equality unconditionally
  on the per-image axis; semantic preserves it via u64 additivity;
  panoptic preserves it via `retain_per_image_deltas=True`.
- [ADR-0033](0033-multi-paradigm-bench.md) — Multi-paradigm bench.
  §"Deferred (explicitly not in scope here)" lists "rayon AP fold"
  among Stage 2 optimization passes — that work remains deferred to
  its own ADR, distinct from this one.
- [ADR-0035](0035-api-surface-consolidation.md) — API surface
  consolidation. The `Evaluator.background(gt, num_threads=N)`
  factory and the `BackgroundEvaluator(num_threads=N)` constructor
  signatures fit cleanly into the consolidated surface; no new
  classes, no new entry points beyond the added kwarg.
- Rayon documentation — [`rayon::ThreadPoolBuilder`](https://docs.rs/rayon/latest/rayon/struct.ThreadPoolBuilder.html),
  [`ThreadPool::install`](https://docs.rs/rayon/latest/rayon/struct.ThreadPool.html#method.install),
  [`current_thread_index`](https://docs.rs/rayon/latest/rayon/fn.current_thread_index.html).
- PyO3 documentation — [`Python::detach`](https://docs.rs/pyo3/latest/pyo3/marker/struct.Python.html#method.detach)
  (the renamed `allow_threads` per PyO3 0.22+; ADR-0006's policy
  carries through under the new name).
- Rust stdlib — [`std::thread::available_parallelism`](https://doc.rust-lang.org/std/thread/fn.available_parallelism.html)
  (cgroup-aware on Linux since 1.59).

## Implementation notes

**Stage A — batch instance path.** Wires the `num_threads: int | None`
kwarg through every public surface listed in §"Configuration discipline"
*for the instance paradigm* (bbox, segm, boundary, keypoints; every
`evaluate_*_summary[*_with_dataset]` and `evaluate_*_grid[*_with_dataset]`
FFI entry; every `evaluate_*_partitioned` FFI entry; the
`vernier eval --threads N` CLI flag). The Rust core gains
`crates/vernier-core/src/evaluate_parallel.rs` with the parallel
sibling of `evaluate_with` and a public `evaluate_*_parallel` wrapper
per kernel. The parity harness gains the
`num_threads ∈ {None, 1, 2, 4, 8}` axis on every existing
bbox / segm / keypoints fixture; the `parity_threads` marker asserts
vernier-vs-vernier strict-mode bit-equality across thread counts. The
`VERNIER_NUM_THREADS` env-var fallback and the rayon re-entry
detection both land here. Rayon becomes an unconditional workspace
dependency.

**Stage B — streaming surfaces and remaining paradigms.** Adds:

- `StreamingEvaluator::update_parsed_parallel` (instance) and the
  matching `BackgroundConfig.num_threads` field; pool owned by the
  worker thread with lifetime == worker lifetime per
  §"`BackgroundEvaluator` shape". `worker_nice` re-applied on pool
  threads via `start_handler` for non-Linux nice inheritance;
  `worker_affinity` deliberately NOT propagated. `num_threads=None`
  is byte-identical to the pre-ADR worker (`pool = None`, no rayon
  symbol entered).
- Semantic-paradigm `StreamingSemanticEvaluator::update_parsed_parallel`
  + `evaluate_from_pngs_parallel` + `accumulate_confusion_parallel`.
  Per-thread confusion matrices; u64-additive reduce — trivially bit-
  equal across thread counts.
- Panoptic-paradigm `StreamingPanopticEvaluator::update_parsed_parallel`
  + `evaluate_per_image_parallel`. Per-image deltas re-sorted by
  `image_id` before the canonical f64 fold to preserve bit-equality
  regardless of par_iter completion order. The Python wrapper
  enforces the forced `retain_per_image_deltas = True` policy under
  `parity_mode="strict" && num_threads > 1` per §"Panoptic", with a
  one-shot info-level log.
- Bench harness `num_threads` axis on workload classes and CellSpec;
  `synthetic_threads_smoke` workload + `just bench-threads-smoke`
  recipe validate the plumbing end-to-end. The full scaling sweep
  across val2017 / LVIS / panoptic-val2017 / ADE20K is its own
  separate operation.

**Deferred — multi-worker `BackgroundEvaluator` (axis D2).** The
§"`num_workers` deferred but the shape is documented" extension
stays out of this rollout. Defaults cannot be picked without
measurement on a real val-loader workload; the single-worker shape
delivered above is the prerequisite for that measurement.
