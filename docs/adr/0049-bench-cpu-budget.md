# ADR-0049: Enforce a per-cell CPU budget and measure memory exactly

- **Status:** accepted
- **Date:** 2026-09-15
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** —

## Context and problem statement

The bench harness ([ADR-0017](0017-local-bench-harness.md),
[ADR-0033](0033-multi-paradigm-bench.md)) compares vernier against
third-party libraries, and its headline tables assumed every library ran
on a single thread. The `num_threads` axis ([ADR-0047](0047-threading-model.md))
was forwarded only to vernier. Other runners ignored it, which was
harmless while they had no thread pools.

That assumption no longer holds for the instance paradigm:

- **faster-coco-eval 1.8.0** (2026-08-24) parallelizes RLE IoU on a
  Python thread pool (`rle_iou_max_workers`, default 8), and parallelizes
  image evaluation and accumulation in C++ pools sized from
  `std::thread::hardware_concurrency()`. Those C++ pools have no user
  knob. Boundary preparation was already multi-threaded before 1.8
  (`boundary_cpu_count`, default `min(cpu_count, 4)`), so the published
  single-thread boundary headline was really vernier on 1 thread against
  faster-coco-eval on 4.
- **hotcoco** (Rust/PyO3, first benchmarked here at 1.0.1) evaluates and
  accumulates on rayon's global pool, which defaults to every visible
  CPU.

Left as is, the default cell would put vernier's sequential path against
libraries using all 8 vCPUs of the bench host. The comparison would say
more about core count than about the implementations.

## Decision drivers

- A headline ratio has to compare equal compute. Otherwise it measures
  the host, not the library.
- The budget has to be enforceable on libraries that expose no thread
  knob. A knob-only approach fails on faster-coco-eval's C++ pools.
- On SMT hosts, `nt=N` must not look worse than it is because it landed
  on hyperthread siblings of the same physical core.
- Every result should carry evidence that the budget held, so a reader
  does not have to trust the harness.
- Memory should be comparable too. `ru_maxrss` covers a whole process
  lifetime, so it folds each library's import cost into the number and
  cannot isolate what an evaluation actually needed.
- Blast radius: do not change measurement semantics for paradigms whose
  baselines did not change.

## Considered options

1. **Knobs only.** Forward `num_threads` to each library's own setting
   (`RAYON_NUM_THREADS`, `rle_iou_max_workers`, …).
2. **CPU affinity + knobs.** Pin each runner process to
   `cpu_budget(num_threads)` logical CPUs before `exec`, and also forward
   the budget to every knob that exists.
3. **Report library defaults.** Let each library use its out-of-the-box
   threading and document the core count.

## Decision outcome

Chosen option: **Option 2 (CPU affinity + knobs)**. Affinity is the only
constraint every implementation honours, including thread pools with no
knob. Forwarding the knobs as well keeps libraries from oversubscribing
the pinned set with threads that just wait.

- `cpu_budget(None) = 1`: the default (headline) cell is single-CPU for
  every impl. `nt=N` cells get `N` CPUs.
- CPUs are chosen one per physical core before any SMT sibling, starting
  from the highest-numbered core so the 1-CPU cell stays off CPU 0.
- Scope: every batch spawn path — instance, LVIS, panoptic and semantic.
  Streaming is unchanged (its cells measure a producer/consumer pipeline,
  not a single-shot evaluation). Semantic was pulled in once the CPU/wall
  column showed mmsegmentation running at ~4.0 CPU/wall (torch's intra-op
  pool) against vernier's 1.0; the budget is also forwarded to
  `torch.set_num_threads`.
- Every stage records process CPU time (`cpu_ns`), and the `total` stage
  records the affinity the runner observed. `docs/benchmarks.md` renders
  CPU/wall per impl.
- Every stage also records RSS at its start and the kernel's exact RSS
  high-water mark (`VmHWM`) while it ran, resetting `VmHWM` through
  `/proc/self/clear_refs` at each stage start. The reads sit outside the
  timed span. Two memory columns are published: absolute peak RSS over
  the timed stages, and that peak minus the pre-stage RSS ("eval Δ RSS"),
  which is the memory the evaluation itself needed.
- hotcoco joins the instance matrix (bbox / segm / keypoints) and the
  LVIS matrix (bbox). It is compared to vernier at the aligned tier, the
  same tier faster-coco-eval gets.

The out-of-the-box view from Option 3 is still published: on the 8-vCPU
bench host, the `nt=8` column of the thread-scaling tables is effectively
each library's default configuration.

### Consequences

- **Positive:** Headline ratios compare equal compute, and each result
  carries its CPU/wall ratio as evidence. Scaling tables cover every
  multi-threaded impl, not only vernier.
- **Negative:** Default cells for single-threaded baselines now run
  pinned to one CPU instead of floating. The median should not move
  materially, but numbers recorded before this ADR are not strictly the
  same measurement. faster-coco-eval's C++ pools still spawn
  `hardware_concurrency()` threads inside the pinned set. That is the
  library's own behaviour on a constrained host, and any cost from it is
  counted against it.
- **Negative (measurement side effect):** resetting `VmHWM` also resets
  the kernel accounting behind `getrusage(...).ru_maxrss`, so a runner's
  `ru_maxrss` no longer means "process lifetime peak" — it covers only
  the final stage onward. Everything that reports memory therefore reads
  the recorded stage peaks instead, and `ru_maxrss_bytes` survives on
  `RepResult` only to read results captured before this ADR.
- **Neutral:** Per-stage splits stay non-comparable across impls:
  vernier parses JSON inside `evaluate`, while pycocotools-shaped
  libraries parse in `load`. Only the total is rendered.

## Links and references

- Related ADRs: [ADR-0017](0017-local-bench-harness.md) (Axis G assumes
  Linux CPU pinning is available), [ADR-0033](0033-multi-paradigm-bench.md),
  [ADR-0047](0047-threading-model.md).
- faster-coco-eval 1.8.0 threading: `faster_coco_eval/core/faster_eval_api.py`
  (`rle_iou_max_workers`), `csrc/faster_eval_api/coco_eval/cocoeval.cpp`
  (`hardware_concurrency()` worker pools).
- hotcoco: <https://github.com/derekallman/hotcoco> (rayon global pool).
- `clear_refs` semantics: `Documentation/filesystems/proc.rst`, "Clearing
  the PG_Referenced and ACCESSED/YOUNG bits" — writing `5` resets the
  peak-RSS watermark.
