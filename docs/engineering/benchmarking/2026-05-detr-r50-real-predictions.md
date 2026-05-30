# 2026-05 DETR-R50 real predictions — bbox cell

Engineer-facing snapshot of vernier vs the alternatives on a *real*
COCO val2017 prediction distribution — `facebook/detr-resnet-50` at hub
revision `1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b` (Apache-2.0,
CPU-runnable). Companion to the jittered-DT cells in
[2026-05-vernier-vs-alternatives.md](./2026-05-vernier-vs-alternatives.md).

Synthetic / GT-derived jitter doesn't exercise the long-tail score
distribution, low-confidence false-positive density, or class imbalance
that a SOTA detector produces in the wild. This snapshot closes that
gap for bbox / instance.

## Shared configuration

- **Harness mode**: release (N=10 + 2 warmup, randomised impl order,
  governor pre-flight, 5% relative-IQR gate per impl)
- **Git SHA**: `e1616fd77e7d`
- **Machine fingerprint**: `84edec51fd71` (AMD EPYC-Milan, x86_64).
  Different machine from the cross-paradigm snapshot
  (`37652a58e939`) — absolute numbers shouldn't be cross-compared,
  only ratios within this snapshot.
- **Build profile**: cargo release defaults (`opt-level=3`,
  `lto=thin`, `codegen-units=1`, no `target-cpu`). Same profile the
  PyPI wheel ships with — no benchmarking-only flags.
- **Parity**: strict-tier (vs pycocotools) and aligned-tier (vs
  faster-coco-eval) both pass on every reported cell.
- **Workload**: `coco_val2017_detr_r50_v1d5f47b` — bbox-only,
  150,680 detections across 4,977 of the 5,000 val2017 images
  (23 images produced no above-threshold output at the 0.05 score
  floor), 80 categories. Cache file 23 MB.

## Instance — `coco_val2017_detr_r50_v1d5f47b` (bbox)

| impl              |    median   |        IQR         |  RSS (max) | vs vernier |
| ----------------- | ----------: | -----------------: | ---------: | ---------: |
| **vernier**       |    552.8 ms |   12.1 ms (2.19%)  |   322 MiB  |  **1.00x** |
| faster-coco-eval  |    2.948 s  |   50.8 ms (1.72%)  |   830 MiB  |    5.33x   |
| pycocotools       |   11.104 s  |  112.5 ms (1.01%)  |   762 MiB  |   20.09x   |

### Per-stage breakdown (median ns)

The harness times four stages per rep: `load` (GT+DT JSON ingest),
`evaluate` (the per-(category, area-range, image) IoU sweep),
`accumulate` (PR-curve / scores tensor), `summarize` (12-stat output).

| impl              |      load   |    evaluate    |   accumulate   |
| ----------------- | ----------: | -------------: | -------------: |
| **vernier**       |    26.0 ms  |     392.4 ms   |     134.4 ms   |
| faster-coco-eval  |   742.3 ms  |   2,207.1 ms   |       fused\*  |
| pycocotools       |   729.2 ms  |   9,131.5 ms   |   1,251.0 ms   |

\* faster-coco-eval fuses accumulate into evaluate; the harness
records ~0 ns on the accumulate stage for fce and reports the wall
under evaluate.

### Raw measurements

For downstream consumers that need ratios without rounding loss.

| impl              |    median (ns) |       IQR (ns) |     RSS (B)   |
| ----------------- | -------------: | -------------: | ------------: |
| vernier           |    552,811,332 |     12,088,664 |   337,272,832 |
| faster-coco-eval  |  2,947,569,688 |     50,809,766 |   869,945,344 |
| pycocotools       | 11,103,797,514 |    112,479,404 |   798,515,200 |

### Read against the table

- **20.1x vs pycocotools / 5.3x vs faster-coco-eval** on real DETR-R50
  predictions. The pycocotools ratio is *wider* than on
  `coco_val2017_jittered_seed0` (15.5x) — the per-`(category, image)`
  framework overhead pycocotools pays scales with the populated
  `(K, I)` cells the COCOeval state machine walks, and DETR's bbox
  output covers more of that grid (80 categories × 5000 images, with
  most cells non-empty after the 0.05 floor) than the GT-jittered
  workload does.
- **Load stage is the most visible delta beyond the kernel.** vernier
  ingests GT+DT JSON via its binary FFI without materializing the
  Python dicts each oracle's loader keeps around — 26 ms vs 730+ ms
  on both alternatives. On a cell whose `evaluate` time is the
  dominant cost (DETR-R50 here), the load gap is ~3% of vernier's
  total but accounts for ~7% of pycocotools' and ~25% of fce's. The
  ratio compounds on shorter-running workloads.
- **RSS**: vernier holds 322 MiB peak vs 762–830 MiB on the
  alternatives — ~2.4-2.6× lower. Same trend as the jittered cell
  (261 MiB vs 576-661 MiB); the absolute numbers grow with the real
  detection density but the ratio holds.
- **IQR**: vernier's relative IQR widens to 2.19% on this cell vs
  1.10% on jittered seed0 — DETR's denser per-image walk lets
  per-rep noise (page cache, scheduler) accumulate more visibly, but
  the gate (5%) holds with margin. The two oracles tighten as their
  wall stretches: pycocotools at 1.01% (vs 3.39% on jittered).

## Parity

- **mAP**: `0.4168586010785383` — bit-identical across all three
  impls. Reproduces DETR-R50's published `box AP = 42.0` on
  COCO val2017 (the paper rounds to one decimal).
- **Strict-tier** (vs pycocotools) — exact equality on the 12-stat
  det summary, dense `precision` / `recall` / `counts` aggregates.
- **Aligned-tier** (vs pycocotools, 2 ULP relative on
  `eval_imgs.dtScores` + `scores` tensor) — absorbs the documented
  `serde_json` vs `strtod` rounding drift on near-tie JSON-encoded
  scores (e.g. `0.9992794394493103`). Tracked as a follow-up parity
  item; does not propagate past the score-threshold projection.
- **Aligned-tier** (vs faster-coco-eval) — passes; fce's reductions
  are bit-identical to pycocotools on this cell.

## Reproducing

The DETR-R50 prediction cache is generated by the SOTA harness
(`tests/python/integration/real_models/sota/`). First-run cost is
~12-15 h on an 8-core CPU (5000 images × ~9 s/image, DETR is
NMS-free so per-image cost is data-independent); cache hit is
seconds. The cache file is keyed on the full hub commit SHA so a
weights bump invalidates by construction.

```bash
# One-time: populate the prediction cache (~12-15h on 8-core CPU)
./tools/fetch-real-predictions.sh --detr

# Run the bench cell
VERNIER_COCO_CACHE=/path/to/coco-val2017 \
  just bench-run -- \
    --impl all \
    --workload coco_val2017_detr_r50_v1d5f47b \
    --paradigm instance \
    --iou bbox \
    --mode release
```

Results land under `bench/results/<git-sha>/<machine-fp>/instance/coco_val2017_detr_r50_v1d5f47b/bbox/`.
