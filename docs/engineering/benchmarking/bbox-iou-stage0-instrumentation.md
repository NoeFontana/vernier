# bbox-IoU Stage 0 instrumentation

How to capture the per-call `(kernel, g, d, wall_ns)` distribution that
gates Stages 1b (SoA refactor) and 1c (explicit `pulp::Simd` lanes) of
the bbox-IoU optimization plan.

## Workflow

1. **Build vernier with the feature into the bench env's venv.**

   ```bash
   just bench-develop-histogram
   ```

   Compiles `--features bench-histogram` on `vernier-ffi` (which pulls
   `vernier-core`'s gated `histogram` module). Off by default; this
   recipe is the only way to flip it on without manual surgery.

2. **Run any workload that exercises the kernel.** The records
   accumulate in a process-global `Mutex<Vec<Record>>` over the
   lifetime of the runner subprocess. A real val2017 pass:

   ```bash
   VERNIER_COCO_GT_PATH=/path/to/instances_val2017.json \
     uv run --directory bench python -m bench run \
       --impl vernier \
       --workload coco_val2017_jittered_seed0 \
       --iou bbox \
       --mode profile
   ```

3. **Dump the buffer to CSV.** Histogram state lives in the *runner
   subprocess*, so the dump must happen before that subprocess exits.
   The bench harness does not currently call dump on shutdown — patch
   `bench/bench/runners/vernier_runner.py` to invoke
   `vernier._core.dump_bbox_iou_histogram(path)` between `summarize`
   and `write_outputs`, gating on a `VERNIER_BENCH_HISTOGRAM_PATH` env
   var so a non-instrumented build is unaffected. (Follow-up; not yet
   wired.)

   For a one-off single-process measurement (e.g., parity tests, a
   custom Python script), `vernier._core.dump_bbox_iou_histogram(path)`
   is callable directly:

   ```python
   import vernier._core as c
   # ... run evaluations ...
   n = c.dump_bbox_iou_histogram("/tmp/bbox-hist.csv")
   ```

4. **Reset to a stock build** when done so subsequent benches reflect
   the shipped wheel:

   ```bash
   just bench-sync
   ```

## CSV schema

```
kind,g,d,wall_ns
FullIou,4,30,512
OverlapMask,10,100,3201
...
```

* `kind` — `FullIou` (`BboxIou::compute`) or `OverlapMask`
  (`BboxIou::compute_overlap_mask`, the segm/boundary prefilter).
* `g`, `d` — GT and DT counts for the call (`u32`-saturated).
* `wall_ns` — `Instant::now()` delta around the kernel body. ~20–30 ns
  resolution on x86; for cells with `G·D < 4` this is comparable to the
  arithmetic itself. Use the time-weighted distribution, not raw counts.

## Decision criterion

From the optimization plan:

* If ≥ 80% of total wall time concentrates in cells with `G·D ≥ 256`,
  Stage 1c (explicit `pulp::Simd` lanes) is on the table.
* If ≥ 80% concentrates in cells with `G·D < 64`, drop 1c — the inner
  loop is too short to amortize explicit-lane overhead.

The intermediate band is judgment-call territory.

## Performance impact when feature is off

Zero. The `bench-histogram` feature gates the entire histogram module
out of the build, and the per-kernel `_guard` lines are erased by
`#[cfg(feature = "bench-histogram")]`. The shipped wheel never sees
this code.
