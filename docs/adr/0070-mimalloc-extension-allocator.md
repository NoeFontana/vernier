# ADR-0070: Ship mimalloc as the Python extension's global allocator

- **Status:** proposed
- **Date:** 2026-10-30
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

An evaluation allocates its working set once and touches all of it: the
parsed dataset and detections, per-cell scratch, then the precision and
scores tensors (7.8 MB each on COCO). Under glibc malloc, every block
above the mmap threshold is a fresh mapping, faulted in one 4 KiB page at
a time, and returned to the kernel when freed. On COCO val2017 jittered
bbox at one CPU, a fresh process takes ~14,000 minor page faults (55 MB)
between `evaluate` and `accumulate`, and `accumulate` runs twice as long
as it does in a warm process (23 vs 12 ms). Under threads the faults also
serialize on the process's memory map, so they cost more as threads are
added.

`vernier-ffi` has carried a `mimalloc-global` feature since the
bbox-scaling investigation (#256), off by default. Measured cold, it
removes almost all of those faults: mimalloc reserves its arena as one
mapping and advises Linux to back it with transparent huge pages
(`MADV_HUGEPAGE`), so the working set is faulted in 2 MiB at a time.
With that advice disabled (`MIMALLOC_ALLOW_THP=0`) mimalloc is as slow
as glibc, so this is where the gain comes from.

## Decision drivers

- One-shot evaluation (a script, a benchmark cell, the first `compute()`
  of a training run) is a fresh process; its fault cost is real time.
- No loss in a warm, long-running process (a training loop), in speed or
  in memory.
- Nothing users have to set: environment tunables reach only those who
  know to set them.
- No `unsafe` in vernier, no change to results.

## Considered options

1. **Keep the system allocator.**
2. **Document `GLIBC_TUNABLES=glibc.malloc.hugetlb=1`** for users who
   want the speedup.
3. **mimalloc v3** (the `mimalloc` crate's default) as the global
   allocator of the extension module.
4. **mimalloc v2** as the global allocator of the extension module.

## Decision outcome

Chosen option: **4**. The `mimalloc` feature of `vernier-ffi` is on by
default, so every build of `vernier._core` — the wheels, `uv run`'s
development build, the nextest binaries — runs the allocator it ships
with; `--no-default-features` gives the system allocator back for an A/B.
It uses mimalloc v2 with default options and `local_dynamic_tls`: the
extension is loaded with `dlopen`, where initial-exec TLS draws on a
fixed static-TLS reserve that other extensions share. The allocator
applies to Rust allocations only; Python's own objects stay on
`pymalloc`/libc. Nothing crosses allocators: numpy arrays and Arrow
buffers built from Rust memory are released through Rust's own drop, and
DLPack input is only borrowed.

Measured on Linux x86_64 (THP `madvise`, the Ubuntu default), fresh
process per run, CPUs pinned, median of five:

| cell (COCO val2017) | glibc | mimalloc v2 | peak RSS |
| --- | ---: | ---: | ---: |
| jittered bbox, 1 CPU | 167.6 ms | **140.5 ms** (−16 %) | 146 → 154 MiB |
| jittered keypoints, 1 CPU | 116.0 ms | **101.9 ms** (−12 %) | 103 → 111 MiB |
| jittered segm, 1 CPU | 406.7 ms | **364.6 ms** (−10 %) | 157 → 174 MiB |
| RF-DETR Nano bbox, 1 CPU | 498.6 ms | **441.1 ms** (−12 %) | 239 → 258 MiB |
| RF-DETR Nano bbox, 4 CPUs | 299.2 ms | **236.4 ms** (−21 %) | 245 → 305 MiB |
| RF-DETR Seg-Nano segm, 4 CPUs | 705.8 ms | **609.9 ms** (−14 %) | 488 → 529 MiB |

Each run reads the two JSON files, evaluates, accumulates and summarizes
through `vernier._core`, as the benchmark runner does. In a warm process
(the same evaluation repeated, GT parsed once) the times are within noise
of glibc, and after four bbox + segm rounds at four threads RSS settles
at 345–394 MiB against glibc's 421–428 MiB.

The one-CPU jittered bbox cell is the one ufcoco won at 0.6.0 (157 ms
there; 160 ms in this run).

### Consequences

- **Positive:** cold evaluation is 10–16 % faster at one CPU and 14–21 % at four,
  with no change to results, API or `unsafe` code. In a warm process the
  speed is unchanged and steady-state RSS is lower than glibc's (it
  fragments less across repeated evaluations).
- **Negative:** peak RSS of a one-shot run rises, from huge-page
  granularity and from freed memory mimalloc keeps for reuse. The wheels
  compile ~10k lines of C (MIT), so the build needs a C compiler on
  every target, which maturin's images already provide. vernier owns one
  more native dependency to keep current.
- **Neutral:** the gain is Linux THP. With THP set to `never` mimalloc is
  as fast as glibc; macOS and Windows have no THP advice, and their
  numbers are not measured here. The `vernier` CLI keeps the system
  allocator: ADR-0015 keeps its dependency tree pure Rust, and changing
  that is its own decision.

## Pros and cons of the options

### Option 1

- 👍 Nothing to own.
- 👎 Leaves 10–20 % of one-shot time in page faults.

### Option 2

- 👍 No dependency; also covers Python's own allocations (it speeds up
  reading the input files).
- 👎 Process-wide, set before Python starts, glibc only, and only for
  users who find it.

### Option 3

- 👍 The crate's default line.
- 👎 Same speed as v2 here, with 20–90 MB more peak RSS than glibc
  against v2's 8–60.

### Option 4 (chosen)

- 👍 The speedup with the smallest peak-RSS increase of the mimalloc
  lines.
- 👎 A C dependency in the wheel build.

## Links and references

- Related ADRs: [ADR-0006](0006-threading-model.md) (threading model), [ADR-0015](0015-vernier-cli.md) (pure-Rust CLI),
  [ADR-0047](0047-threading-model.md) (`num_threads`).
- mimalloc: <https://github.com/microsoft/mimalloc>; `local_dynamic_tls`
  in the `mimalloc` crate's features.
