# 2026-05 vernier-bench results

First release-mode capture from `vernier-bench` (ADR-0017). Every cell
captured this month — bbox/segm/boundary on val2017, a synthetic
stress smoke at n=500, and the harness's per-IoU fan-out smoke on the
parity fixture — consolidated here.

## Shared configuration

- **Harness:** release mode, N=10 measurement reps with 2 warmup
  discarded, randomised impl order per rep, IQR-relative-to-median
  gate at 5%. Every cell came in well under the threshold.
- **Parity tiers** (ADR-0002): strict (`np.array_equal` vs
  pycocotools), aligned (`np.allclose` at 4×ULP vs faster-coco-eval),
  strict vs the vendored `bowenc0221/boundary-iou-api` oracle for
  boundary. **Every cell parity OK.**
- **Pinned baselines:** pycocotools `2.0.11` and faster-coco-eval as
  resolved by `bench/envs/*/uv.lock`; bumping is an ADR-level
  decision.
- **Machine:** AMD EPYC-Milan / 4 cores / 32 GiB / kernel
  `6.8.0-107-generic`, fingerprint `82013f18a44d`, git_sha
  `58f09cb9149b`. cpufreq unexposed on this VM; the IQR gate carries
  the noise budget per the ADR-0017 "quiet on machines without
  cpufreq" clause.

## Headline — all cells

`× vernier` is total wall median for the baseline divided by
vernier's; greater than 1.0 means vernier is faster.

| workload                                  | iou      | vernier (median) | pycocotools     | faster-coco-eval | boundary-iou-api |
| ----------------------------------------- | -------- | ---------------: | --------------: | ---------------: | ---------------: |
| `coco_val2017_jittered_seed42`            | bbox     |       644.156 ms |    9.12× slower |     3.20× slower |             —    |
| `synthetic n=500,c=80,g=10,d=30,seed=0`   | bbox     |       116.763 ms |   12.57× slower |     2.35× slower |             —    |
| `coco_val2017_perfect_segm`               | segm     |         1.742 s  |    4.66× slower |     2.61× slower |             —    |
| `coco_val2017_perfect_segm`               | boundary |        75.861 s  |             —   |              —   | **0.84× — vernier 1.19× slower** |
| `smoke_perfect_match_segm`                | segm     |         0.280 ms |   23.26× slower |    19.81× slower |             —    |
| `smoke_perfect_match_segm`                | boundary |         0.355 ms |             —   |              —   |    20.71× slower |

Takeaways:

- **bbox is the load-bearing headline.** vernier is 9.1× faster than
  pycocotools and 3.2× faster than faster-coco-eval on val2017.
- **segm wins are real.** 4.66× / 2.61× on val2017 perfect-match.
- **Synthetic widens the gap.** 12.6× / 2.35× at n=500 — pycocotools'
  quadratic-ish hot loops scale less harshly when per-image work is
  small but uniform.
- **Boundary on val2017 is the one cell where vernier loses to a
  baseline.** 1.19× slower than `boundary-iou-api`. Cause is
  algorithmic and known; tracked as a follow-up, not a release
  blocker. See §Boundary regression below.
- **Smoke fixture sub-millisecond ratios** are dominated by fixed
  Python/PyO3 conversion overhead at that scale; treat them as parity
  smoke, not a perf claim.

---

## bbox — `coco_val2017_jittered_seed42`

`instances_val2017.json` (sha256-pinned `e8c7f7…`) with deterministic
Gaussian-noise jittered DT (seed 42), generated on the fly per
`bench/bench/workloads/jittered_predictions.py`.

| impl              | total median |    IQR (rel) | RAM (max RSS) | vs vernier |
| ----------------- | -----------: | -----------: | ------------: | ---------: |
| **vernier**       |   644.156 ms |  4.4 ms (0.7%) |     750.7 MiB |       1.0× |
| faster-coco-eval  |     2.058 s  | 64.3 ms (3.1%) |     644.7 MiB |  3.20× slower |
| pycocotools       |     5.871 s  | 82.7 ms (1.4%) |     558.3 MiB |  9.12× slower |

Per-stage medians:

| stage      | vernier  | faster-coco-eval | pycocotools |
| ---------- | -------: | ---------------: | ----------: |
| load       |  15.6 ms |         425.5 ms |    422.1 ms |
| evaluate   | 549.9 ms |        1632.6 ms |   4707.4 ms |
| accumulate |  78.1 ms |       *fused\**  |    747.6 ms |
| summarize  |  0.62 ms |          2.20 ms |     1.33 ms |

\* faster-coco-eval folds accumulate into evaluate; the timer reads zero.

`evaluate` dominates — pairwise IoU + per-image matching across ~5000
images and 80 categories.

---

## bbox — `synthetic:n=500`

500 images × 80 categories × 10 GT and 30 DT per image, seed 0.
Workload ID derives from the parameter tuple; cached at
`~/.cache/vernier-bench/synthetic/synthetic_n500_c80_g10_d30_s0_{gt,dt}.json`.

| impl              | total median |    IQR (rel) | RAM (max RSS) | vs vernier |
| ----------------- | -----------: | -----------: | ------------: | ---------: |
| **vernier**       |   116.763 ms | 2.95 ms (2.5%) |     147.7 MiB |       1.0× |
| faster-coco-eval  |   274.720 ms | 4.01 ms (1.5%) |     143.1 MiB |  2.35× slower |
| pycocotools       |     1.468 s  | 25.31 ms (1.7%) |     201.3 MiB | 12.57× slower |

The vernier-vs-pycocotools gap is **9.1× on val2017** but **12.6× on
synthetic n=500**: synthetic has uniformly small images and regular
per-image GT/DT counts; val2017 has wide image-size and per-image
distributions, which pycocotools handles less harshly than its
worst-case loops would suggest.

---

## segm — `coco_val2017_perfect_segm`

`instances_val2017.json` paired with `perfect_dt_segm.json` (the
parity-test "GT-as-DT" predictions generated by
`tools/make-perfect-dt.py`; 5000 images, ~36k annotations).

| impl              | total median |    IQR (rel) | RAM (max RSS) | vs vernier |
| ----------------- | -----------: | -----------: | ------------: | ---------: |
| **vernier**       |     1.742 s  |  55.1 ms (3.2%) |     765.3 MiB |       1.0× |
| faster-coco-eval  |     4.543 s  |  92.7 ms (2.0%) |     795.4 MiB |  2.61× slower |
| pycocotools       |     8.114 s  | 148.9 ms (1.8%) |     616.8 MiB |  4.66× slower |

Per-stage medians (ms):

| stage      | vernier  | faster-coco-eval | pycocotools |
| ---------- | -------: | ---------------: | ----------: |
| load       |   23.1   |          550.3   |      529.8  |
| evaluate   | 1658.9   |         3977.4   |     6854.4  |
| accumulate |   59.8   |       *fused\**  |      725.6  |
| summarize  |    0.62  |            2.30  |        1.34 |

**Caveat — perfect-match DT under-stresses matching.** Every DT lines
up with a GT exactly (score=1.0). Real detector output has unmatched
predictions which add false-positive accumulation work. These numbers
are realistic for the eval/accumulate hot path on real-mass mask
data; they understate the matching cost a Mask R-CNN inference would
exercise. Polygon-jitter DT generator is out of scope for the v1
harness (ADR-0017 §"Out of scope").

---

## boundary — `coco_val2017_perfect_segm` (regression)

| impl                | total median |     IQR (rel) | RAM (max RSS) | vs vernier |
| ------------------- | -----------: | ------------: | ------------: | ---------: |
| **vernier**         |    75.861 s  | 418.9 ms (0.6%) |     767.1 MiB |       1.0× |
| boundary-iou-api    |    63.642 s  | 928.3 ms (1.5%) |     705.6 MiB |  **0.84× — vernier 1.19× slower** |

Per-stage medians (ms):

| stage      | vernier   | boundary-iou-api |
| ---------- | --------: | ---------------: |
| load       |     23.3  |        55614.7   |
| evaluate   |  75778.4  |         7338.7   |
| accumulate |     59.1  |          709.7   |
| summarize  |      0.58 |            1.35  |

### Stage labels mislead — read the totals

`boundary-iou-api` precomputes a dilated boundary mask per annotation
inside its `cocoeval` constructor (i.e., during what our harness times
as "load"); the subsequent `evaluate()` then operates on cached
boundary masks. vernier does the boundary derivation lazily on each
IoU computation. So:

- "load" for boundary-iou-api is **mask augmentation**, not file I/O.
- "evaluate" for vernier includes **per-pair boundary derivation**,
  not just IoU.

The honest comparison is the **total** column. On val2017
perfect-match, boundary-iou-api's "augment once, evaluate cheaply"
trade beats vernier's "derive on demand" — vernier is **1.19× slower**
end-to-end here.

### Investigated causes (2026-05 perf push, single-threaded)

A single-threaded perf push attacked the three hypotheses from the
plan at `.claude/plans/let-s-optimize-the-heck-adaptive-ritchie.md`.
All three steps measured on this same workload (release, N=10 reps,
IQR gate ≤5%); the per-step deltas isolate each lever:

| step | what landed                                                             | total median | vs baseline | IQR (rel) |
| ---- | ----------------------------------------------------------------------- | -----------: | ----------: | --------: |
| 0    | baseline (HEAD `d1c5ad4`)                                               |    75.932 s  |          —  |     0.52% |
| 1    | `_into` scratch variants in `vernier-mask` (alloc reduction)            |    76.007 s  |     +0.10%  |     0.59% |
| 2    | bit-packed log-reduction `min_filter_binary`                            |    76.660 s  |     +0.96%  |     0.69% |
| 2′   | revert Step 2 → clean scalar van Herk                                   |        —     |          —  |        —  |
| 3    | shared `Mutex<ErodeScratch>` across cells via `BoundaryIouCached`       |    76.489 s  |     +0.73%  |     0.70% |

Findings:

1. **Allocation churn isn't the bottleneck.** The `_into` scratch
   variants amortize ~216 k mallocs in the hot path (~36 k anns × ~6
   `Vec<u8>` per `boundary_band`). Net delta: within IQR noise
   (+0.10% from baseline). Dataset-wide scratch reuse (Step 3,
   shared across cells) is also within noise (+0.73%). Conclusion:
   malloc cost on this workload is well below 100 ms — far from the
   ~12 s gap.
2. **Bit-packed binary erosion regressed.** Packing each row to u64
   bitstrings, sliding-AND of (2d+1) shifted copies, then unpacking
   was algorithmically tighter but the byte-by-byte pack/unpack
   loops dominated; net +0.96% on val2017. Reverted to scalar
   running-zero-count van Herk on `{0,1}` bytes.
3. **The actual gap is the inner sliding-min.** `boundary-iou-api`
   uses `cv2.erode` (hand-tuned C SIMD with structuring-element
   intrinsics); vernier's `min_filter_binary`
   (`crates/vernier-mask/src/ops/erode.rs`) is a scalar
   running-zero-count loop. ADR-0003 pins `pulp::Arch::dispatch` as
   the stable-Rust SIMD strategy but boundary IoU's tightest inner
   loop hasn't adopted it.

Hypotheses 1 and 2 from this section's earlier draft (boundary-mask
cache; per-IoU dilation as the dominant cost) are **not** the gap —
the cache benefit is below noise on val2017 because erosion itself
is the expensive step, not the malloc. Cache + scratch reuse remain
correct in principle (they're necessary infrastructure for any
future SIMD path that wants to amortize buffers across cells), but
they don't pay off on their own at this scale.

### Next push: byte-vectorized `pulp::Arch::dispatch` over `min_filter_binary`

The deferred lever — a `pulp` byte-vector path for the sliding-min
on `{0,1}` rows — is the structural fix expected to close the gap.
Tracked as a follow-up; not a release blocker for v0.0.x.

### Post-fix (2026-05-02 perf push closure)

The boundary regression is closed. Two follow-up PRs landed:

- **PR #86** — sparse-table AND-fold for `min_filter_binary`. Replaces
  the scalar running-zero-count van Herk loop with a level-`l` AND of
  two non-overlapping byte slices (`temp_l[i] = temp_{l-1}[i] &
  temp_{l-1}[i + 2^(l-1)]`); LLVM autovectorizes the contiguous-slice
  AND. Out-of-place ping-pong buffers avoid the scalar-fallback LLVM
  emits when the stride is below SIMD lane width.
- **This PR** — pre-decoded foreground-segment offsets +
  `intersect_area_offsets`. The per-pair RLE byte-stream state machine
  walks one run per loop iteration; replacing it with a two-pointer
  sweep over pre-decoded fg intervals (one decode per ann per call,
  amortised over the cell's pair count) skips the background runs that
  dominate `Rle::counts`. New `vernier_mask::SegmentTable` is the
  CSR-style flat-storage shared by segm + boundary kernels.

Final `coco_val2017_perfect_segm` cells (release, N=10 reps,
IQR ≤5%):

| iou      | impl                | total median | vs baseline | vs vernier |
| -------- | ------------------- | -----------: | ----------: | ---------: |
| segm     | **vernier**         |     1.759 s  | (was 1.742) |       1.0× |
| boundary | **vernier**         |    59.704 s  | (was 75.861) |       1.0× |
| boundary | boundary-iou-api    |    63.642 s  |          —  | **1.07× slower** |

Boundary `evaluate` stage alone: vernier **59.620 s**, IQR 0.85%.
Segm `evaluate` stage: 1.673 s, IQR 0.9%.

Boundary is now faster than `boundary-iou-api` on the headline cell;
the §"Headline — all cells" table above reflects the pre-fix numbers
and is the historical reference point.

### Post-fix #2 — band derivation single-pass decode

After PR #87 closed the regression, the dominant remaining boundary
cost was the per-annotation band raster → RLE → fg-offset
round-trip in `boundary_band_into` + `SegmentTable::push_from_rle`.
The next PR replaces it with a fused `boundary_band_segments_into`
that walks the XOR'd band raster once via the new
`SegmentTable::push_from_raster` primitive — emitting fg offsets and
counting band-area bytes in a single pass. Skips the intermediate
band-`Rle` allocation entirely.

`coco_val2017_perfect_segm` boundary, vernier (release, N=10, IQR ≤5%):

| stage    | post-#87 median | post-fix #2 median |    Δ |
| -------- | --------------: | -----------------: | ---: |
| total    |       59.704 s  |        51.444 s    | −13.8% |
| evaluate |       59.620 s  |        51.360 s    | −13.9% |

vs `boundary-iou-api` 63.642 s: vernier is now **1.24× faster**
(was 1.07× post-#87). Segm cell is unchanged within IQR
(dev-mode 1-rep control = 1.703 s vs 1.759 s release median),
matching the expectation that the fused path touches only the
boundary kernel.

### Post-fix #3 — fused-XOR scan + cache-friendly erode pad/strip

The follow-up PR after #88 bundles two adjacent levers on the
boundary band derivation hot path:

1. `SegmentTable::push_from_rasters_xor`: fuses the in-place
   `mask ^ eroded` step into the segment-emit scan. Instead of
   walking the band raster after a separate XOR pass, the new
   primitive reads the `(mask, eroded)` pair in 8-byte chunks,
   skipping windows where `mask == eroded` (covers both pure
   background `0=0` and pure interior `1=1` — the two regions that
   dominate a COCO-shaped band).
2. Per-column `copy_from_slice` in the erode pad/strip steps,
   replacing the previous nested per-byte loops over the
   non-contiguous column-major stride.

`coco_val2017_perfect_segm` boundary, vernier (release, N=10, IQR ≤5%):

| stage    | post-fix #2 median | post-fix #3 median |     Δ |
| -------- | -----------------: | -----------------: | ----: |
| total    |        51.444 s    |        40.510 s    | −21.3% |
| evaluate |        51.360 s    |        40.427 s    | −21.3% |

vs `boundary-iou-api` 63.642 s: vernier is now **1.57× faster**.
Segm release median is **1.759 s**, bit-identical to the post-#87
baseline — the fused path is boundary-only.

Cumulatively over the 2026-05 push, `coco_val2017_perfect_segm`
boundary went from 75.861 s (pre-fix) to 40.510 s (post-fix #3) —
**1.87× faster** end-to-end, sustained across the IQR gate.

---

## Smoke fan-out — `smoke_perfect_match_segm` (parity smoke, not a perf claim)

1-image / 1-annotation / 1-category parity case from
`tests/python/parity/fixtures/perfect_match_segm/`. Confirms the
harness fans out across IoU types and parity holds; sub-millisecond
ratios are dominated by fixed Python/PyO3 conversion overhead.

segm:

| impl              | total median |    IQR (rel) | RAM (max RSS) | vs vernier |
| ----------------- | -----------: | -----------: | ------------: | ---------: |
| **vernier**       |     0.280 ms |   10 μs (3.7%) |      73.5 MiB |       1.0× |
| faster-coco-eval  |     5.546 ms | 0.24 ms (4.3%) |      73.5 MiB | 19.81× slower |
| pycocotools       |     6.513 ms | 0.27 ms (4.2%) |      73.5 MiB | 23.26× slower |

boundary:

| impl              | total median |    IQR (rel) | RAM (max RSS) | vs vernier |
| ----------------- | -----------: | -----------: | ------------: | ---------: |
| **vernier**       |     0.355 ms |   13 μs (3.8%) |      73.4 MiB |       1.0× |
| boundary-iou-api  |     7.352 ms | 0.29 ms (4.0%) |      76.5 MiB | 20.71× slower |

---

## Cross-cutting caveats

- **Single machine.** Release-mode results aren't aggregated across
  hosts; this is one snapshot from the dev VM. The harness scopes
  every result by machine fingerprint per ADR-0017 §"Out of scope".
- **dev-VM thermals.** Absolute totals are upper-bound; the shape of
  each comparison (the speedup ratios) is the load-bearing claim.
  Bare-metal with `cpupower frequency-set -g performance` will likely
  come in cleaner.
- **bbox real-data uses jittered DT, not real detector output.** The
  matching distribution is bbox-realistic but synthetic. Real
  detector output (e.g., a Mask R-CNN dump) is out of scope for v1.

## Follow-ups

- **Byte-vectorized `pulp::Arch::dispatch` in `min_filter_binary`**
  — the actual structural fix for the val2017 boundary cell. The
  alloc / cache levers were tested in the 2026-05 push and are
  within IQR; the inner sliding-min on `{0,1}` bytes is what's left.
- **Polygon-jitter DT generator** (unlocks realistic segm + boundary
  workloads beyond perfect-match).
- **Synthetic ladder runs** at n=2000 and n=5000 — harness already
  accepts the parameters.
- **Keypoints workload** — no IoU type lists `keypoints` in
  `supported_iou_types` today; lands when the keypoints track from
  ADR-0012 ships.

## Reproduction

```bash
cd bench

VERNIER_COCO_GT_PATH=/path/to/instances_val2017.json \
  uv run python -m bench run --impl all --workload coco_val2017_jittered_seed42 --iou bbox --mode release

uv run python -m bench run --impl all \
  --workload "synthetic:n_images=500,n_categories=80,gt_per_image=10,dt_per_image=30,seed=0" \
  --iou bbox --mode release

VERNIER_COCO_GT_PATH=/path/to/instances_val2017.json \
VERNIER_COCO_DT_SEGM_PATH=/path/to/perfect_dt_segm.json \
  uv run python -m bench run --impl all --workload coco_val2017_perfect_segm --iou segm     --mode release

VERNIER_COCO_GT_PATH=/path/to/instances_val2017.json \
VERNIER_COCO_DT_SEGM_PATH=/path/to/perfect_dt_segm.json \
  uv run python -m bench run --impl all --workload coco_val2017_perfect_segm --iou boundary --mode release

uv run python -m bench run --impl all --workload smoke --iou segm     --mode release
uv run python -m bench run --impl all --workload smoke --iou boundary --mode release

uv run python -m bench report --since 1h
```

`tools/fetch-coco-val.sh` (and its `tools/make-perfect-dt.py`
companion) populates the val2017 GT and `perfect_dt_segm.json` in
`<repo>/.cache/coco-val2017/`; the harness fall-back picks them up
there if the env vars aren't set.
