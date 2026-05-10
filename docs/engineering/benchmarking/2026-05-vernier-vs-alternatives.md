# 2026-05 vernier vs alternatives — cross-paradigm scan

Engineer-facing snapshot of vernier vs the third-party libraries we
benchmark against, run across every IoU kind / paradigm we currently
support.

This doc is the **release-mode N=10 + IQR gate** rerun the previous
revision flagged for. Each cell discards 2 warmup reps and times 10
measurement reps with randomised impl order; the harness gates each
impl on relative IQR (Q3 − Q1) ≤ 5% of the median. Tables now carry an
IQR column; cells where the gate failed are flagged inline. Within-cell
variance is still dominated by the data (5000 val2017 images per cell),
not run-to-run jitter — the gate mostly fires on workloads that hit the
PNG decode / page-cache path, which is exactly where dev-mode N=1
wouldn't have surfaced the spread either.

## Shared configuration

- **Harness mode**: release (N=10 + 2 warmup, randomised impl order,
  governor pre-flight, 5% relative-IQR gate per impl)
- **Git SHA**: ``1fd5720bf56c``
- **Machine fingerprint**: ``1655eb18a194`` (AMD EPYC-Milan, x86_64).
  Distinct from the previous snapshot's ``5658de0e29a3`` — absolute
  numbers shouldn't be cross-compared, only ratios within this snapshot.
- **Build profile**: cargo release defaults (``opt-level=3``,
  ``lto=thin``, ``codegen-units=1``, no ``target-cpu``). Same profile
  the PyPI wheel ships with — no benchmarking-only flags.
- **Parity**: every reported cell passed strict-tier (vs pycocotools)
  and aligned-tier (vs faster-coco-eval) where applicable.
- **Workloads**: ``coco_val2017_jittered_seed0`` for instance/{bbox, segm,
  boundary}; the other paradigms documented below.

## Instance — `coco_val2017_jittered_seed0`

5000 val2017 images, deterministic Gaussian-jittered DT (seed 0).
Same workload across the three IoU kinds; the impl set varies by what
each library supports.

### bbox

| impl              |    median   |        IQR        |  RSS (max) | vs vernier |
| ----------------- | ----------: | ----------------: | ---------: | ---------: |
| **vernier**       |   360.0 ms  |   3.8 ms (1.07%)  |  236 MiB   |   **1.00x** |
| faster-coco-eval  |    2.127 s  |  21.8 ms (1.03%)  |  661 MiB   |    5.91x   |
| pycocotools       |    5.820 s  |  65.8 ms (1.13%)  |  576 MiB   |   16.17x   |

### segm

| impl              |    median   |        IQR        |  RSS (max) | vs vernier |
| ----------------- | ----------: | ----------------: | ---------: | ---------: |
| **vernier**       |   967.7 ms  |  13.4 ms (1.38%)  |  236 MiB   |   **1.00x** |
| faster-coco-eval  |    3.605 s  |  63.7 ms (1.77%)  |  721 MiB   |    3.73x   |
| pycocotools       |    6.853 s  |  71.8 ms (1.05%)  |  569 MiB   |    7.08x   |

### boundary

Boundary-IoU is a specialised metric. The reference implementation is
`boundary-iou-api`; faster-coco-eval ≥1.6 ships its own boundary
surface alongside the COCOeval drop-in. pycocotools doesn't expose
boundary natively.

| impl              |    median   |        IQR        |  RSS (max) | vs vernier |
| ----------------- | ----------: | ----------------: | ---------: | ---------: |
| **vernier**       |    3.130 s  |  21.7 ms (0.69%)  |  238 MiB   |   **1.00x** |
| faster-coco-eval  |   17.837 s  |  48.8 ms (0.27%)  |  794 MiB   |    5.70x   |
| boundary-iou-api  |   62.233 s  | 228.1 ms (0.37%)  |  666 MiB   |   19.88x   |

The faster-coco-eval boundary cell is timing-only — it isn't gated by
parity. The harness's boundary parity tier compares vernier vs
`boundary-iou-api` (the per-quirk strict reference for ADR-0010).
faster-coco-eval's boundary algorithm shares the 0.02 dilation-ratio
default but its band-derivation path differs in detail; pinning a
tolerance is a separate ADR-level decision, not a measurement
question.

### Raw measurements (instance)

For downstream consumers that need ratios without rounding loss.

| iou      | impl              |    median (ns) |       IQR (ns) |    RSS (B)    |
| -------- | ----------------- | -------------: | -------------: | ------------: |
| bbox     | vernier           |    360,025,645 |      3,849,963 |   247,070,720 |
| bbox     | faster-coco-eval  |  2,126,788,867 |     21,810,318 |   693,407,744 |
| bbox     | pycocotools       |  5,819,994,421 |     65,818,120 |   604,479,488 |
| segm     | vernier           |    967,653,356 |     13,373,942 |   247,648,256 |
| segm     | faster-coco-eval  |  3,604,756,101 |     63,713,149 |   755,798,016 |
| segm     | pycocotools       |  6,852,665,697 |     71,831,062 |   596,873,216 |
| boundary | vernier           |  3,130,462,065 |     21,686,559 |   249,831,424 |
| boundary | faster-coco-eval  | 17,837,169,630 |     48,809,327 |   832,741,376 |
| boundary | boundary-iou-api  | 62,233,375,428 |    228,058,216 |   698,830,848 |

### Read against the table

- **bbox** sits at 16.2x vs pycocotools / 5.9x vs fce. Bbox IoU compute
  is cheap, so the gap is mostly framework overhead — vernier's
  single-pass evaluator vs pycocotools' per-call cocoeval. Vernier's
  RSS is also a third of fce's; both are loading the same val2017
  GT/DT but vernier ingests via the binary FFI without materializing
  the JSON-shaped intermediate dicts each oracle keeps around. IQRs
  hold under 1.2% across all three impls — the cell is steady-state.
- **segm** is 7.1x vs pycocotools / 3.7x vs fce. The RLE
  intersection/union kernel still does most of the work for every
  impl, but the framework-overhead delta now exposes itself per-cell
  more visibly than the previous round. PR #183 fused the per-cell
  bbox+area+offsets walk; further compression here means going after
  the intersection sweep itself.
- **boundary** held the gap from the dev-mode N=1 snapshot: 19.9x vs
  `boundary-iou-api` and 5.7x vs faster-coco-eval. faster-coco-eval's
  boundary cell sits between the two — 5.7x slower than vernier, 3.5x
  faster than `boundary-iou-api` — which calibrates how much of
  vernier's boundary win is its specific erode pipeline (u64-packed
  row pass, bbox-cropped) vs general framework-overhead reduction.
  **Boundary was identified as the cell with the most absolute
  headroom per CPU cycle in the round-0 snapshot — that headroom is
  now realized and the release-mode rerun confirms the lead is real,
  not a single-rep artefact.**

## Instance — `coco_val2017_keypoints_jittered_seed0` (keypoints)

5000-image val2017 person subset, deterministic OKS-jittered DT.

| impl              |    median   |        IQR        |  RSS (max) | vs vernier |
| ----------------- | ----------: | ----------------: | ---------: | ---------: |
| **vernier**       |   135.7 ms  |   2.4 ms (1.76%)  |  102 MiB   |   **1.00x** |
| faster-coco-eval  |    1.700 s  |  20.1 ms (1.18%)  |  154 MiB   |   12.53x   |
| pycocotools       |    2.317 s  |  13.3 ms (0.57%)  |  163 MiB   |   17.07x   |

| iou       | impl              |     median (ns) |       IQR (ns) |    RSS (B)    |
| --------- | ----------------- | --------------: | -------------: | ------------: |
| keypoints | vernier           |     135,703,564 |      2,387,864 |   106,663,936 |
| keypoints | faster-coco-eval  |   1,700,207,231 |     20,136,823 |   161,927,168 |
| keypoints | pycocotools       |   2,316,928,190 |     13,286,803 |   171,167,744 |

The keypoints workload is small (val2017 keypoints subset, ~6k
annotations) so vernier's framework-overhead advantage dominates —
widest gap of any IoU type at 17.1x vs pycocotools. Memory now diverges
across impls (the previous dev-mode snapshot showed identical RSS
because the rusage high-water mark was dominated by the GT JSON load
in a single rep; with N=10 + warmup, vernier's tighter steady-state
working set shows up).

**Harness fix landed this phase**: keypoints jittered DT was
generating against `instances_val2017.json` (which lacks `keypoints`
fields, producing 0 detections). Patched to plumb
`person_keypoints_val2017.json` through `coco_val_cache`'s
`ensure_kp_gt()` + a new `kp_gt_path()` helper.

## Panoptic — `coco_panoptic_val2017_perfect`

5000 val2017 images, perfect-DT (GT-as-DT). Strict-tier parity
passes against panopticapi on every cell (PQ=1.0 by construction).

| impl                  |   median    |        IQR         | RSS (max)   | vs vernier_panoptic |
| --------------------- | ----------: | -----------------: | ----------: | ------------------: |
| **vernier_panoptic**  |   11.615 s  |  605.5 ms (5.21%)* |  117.7 MiB  |   **1.00x**         |
| panopticapi           |   35.327 s  |  344.5 ms (0.98%)  |  144.5 MiB  |    3.04x (slower)   |

\* Vernier's IQR is just over the 5% release-mode gate. The spread is
real, driven by PNG-decode I/O variance under randomised impl
ordering. The lead over panopticapi is 23.7 s of headroom, far larger
than the 0.6 s IQR band, so the comparison is still load-bearing; treat
the precise 3.04x ratio as "between roughly 2.9x and 3.2x" rather than
a fixed point.

| metric | impl             |    median (ns) |       IQR (ns) |    RSS (B)     |
| ------ | ---------------- | -------------: | -------------: | -------------: |
| pq     | vernier_panoptic | 11,615,017,169 |    605,449,804 |    123,432,960 |
| pq     | panopticapi      | 35,327,480,730 |    344,468,550 |    151,531,520 |

**Findings flag (resolved, then reopened by release-mode warmup).**
Three rounds of optimization closed the gap panopticapi held over
vernier on this cell, and the release-mode warmup surfaced an
additional cold-cache effect that dev-mode N=1 had been pricing in:

| round | wall | RSS | vs panopticapi |
| --- | ---: | ---: | ---: |
| 0 (eager decode) | 85.6 s | 21.17 GiB | 0.40x (slower) |
| 1 (streaming refactor, #187) | 51.0 s | 130 MiB | 0.68x (slower) |
| 2 (FxHash internal maps, #188, dev-mode N=1) | 32.3 s | 127 MiB | 1.11x (faster) |
| **3 (release-mode N=10 + warmup)** | **11.6 s** | **118 MiB** | **3.04x (faster)** |

Round 1 replaced the runner's eager-PNG-decode loop with
`vernier.panoptic.Evaluator.background()` — PNG decode runs on the
main thread while the Rust PQ kernel folds on a worker, so RSS is
bounded by `queue_capacity × image_size × 4 bytes` instead of
`n_images × …`. Round 2 swapped the per-image `HashMap<(u32, u32),
u32>` (intersection histogram) and `HashMap<u32, SegmentInfo>` (DT
validation) to `FxHashMap`: SipHash's DoS resistance was wasted work
on internal integer keys walked at ~1.5 B ops per cell. The `submit`
hot path dropped 24.9 s → 4.7 s (5.3x); the divan-level kernel arm
dropped 3.56 ms → 0.63 ms. Round 3 is not a code change — it's the
warmup reps populating the page cache so PNG reads hit warm files.
Dev-mode N=1 was paying the cold-cache penalty on every measurement;
release-mode hits steady state. The 5.21% IQR is the residual: even
with warm cache there's per-rep PNG-decode timing noise. Two
release-mode rounds at this SHA span 11.6–13.0 s on the same host
which is consistent with a thermal/I/O-jitter floor rather than a
methodology bug.

Strict-tier parity vs `pq_compute_single_core` still passes — FxHash
is deterministic and the histogram-iteration order is irrelevant per
quirk U9.

**Remaining headroom**: PNG decode is still the long pole and runs on
the main thread. The Rust PQ kernel is GIL-free on a worker; the cell
is decode-bound. Faster panoptic PNG decode (e.g., `image-rs` PNG over
`Pillow`) would shave wall-time further and would also shrink the IQR
(less I/O variance), but at that point we're optimizing PIL's
RGB→uint32 conversion, which panopticapi also pays.

**Harness fixes landed this phase**:
- ``GT_ZIP_SHA256`` placeholder pinned to the upstream `c05f76d2…`.
- The cache extractor was looking for val PNGs at
  `annotations/panoptic_val2017/` but they live in a NESTED zip
  (`annotations/panoptic_val2017.zip`); patched to extract the inner
  zip in a second pass.
- `categories.json` `isthing` normalized int 0/1 → bool to match
  vernier-panoptic's deserializer (the parity tests construct
  categories with Python booleans, so the COCO upstream int form
  hadn't been exercised end-to-end via the bench).
- CLI extended to dispatch `--paradigm panoptic` (was hard-gated to
  instance only); routes to the four-path GT/DT bundle.

## Semantic — `coco_val2017_semantic_perfect` and `synthetic_semantic:*`

**Cross-impl semantic now runs in the harness.** The vendored
`mmsegmentation` IoUMetric (ADR-0036, pinned at upstream commit
`c685fe6`) gets its own bench env and runner; the strict-tier parity
gate is bit-equality on the four per-class u64 marginals
(intersect / union / area_pred / area_label), which are
mmsegmentation's native output and the surface vernier projects its
NxN confusion matrix to. Both cells (synthetic and val2017
perfect-DT) pass strict-tier parity — equal marginals ⇒ equal
mIoU / FWIoU / pixel_accuracy / mean_accuracy under quirk AL2.

### `coco_val2017_semantic_perfect` (5000 images, 133 classes)

| impl              |    median   |        IQR         |  RSS (max) | vs vernier |
| ----------------- | ----------: | -----------------: | ---------: | ---------: |
| **vernier_semantic**  |   5.070 s  |  25.5 ms (0.50%)  |   92 MiB  |   **1.00x** |
| mmsegmentation         |  21.377 s  | 237.4 ms (1.11%)  |  648 MiB  |    4.22x   |

| metric | impl             |    median (ns) |       IQR (ns) |    RSS (B)     |
| ------ | ---------------- | -------------: | -------------: | -------------: |
| miou   | vernier_semantic |  5,069,581,870 |     25,529,121 |     97,005,568 |
| miou   | mmsegmentation   | 21,376,962,160 |    237,419,632 |    679,632,896 |

mIoU = 1.0 on both impls (perfect-DT). 4.2x faster wall-time and ~7x
lower peak RSS, single-threaded on each side. Vernier holds at 0.50%
IQR — `evaluate_from_pngs`'s fused libpng-decode + confusion-fold
(ADR-0037) reaches steady state once page cache warms.

### `synthetic_semantic:n_images=200,n_classes=19,seed=0`

| impl              |   median   |        IQR         | RSS (max) | mIoU    | vs vernier |
| ----------------- | ---------: | -----------------: | --------: | ------: | ---------: |
| **vernier_semantic**  |   63.1 ms  |  618.8 μs (0.98%)  |  88 MiB |  0.8180 |   **1.00x** |
| mmsegmentation        |  437.5 ms  |  46.5 ms (10.64%)* | 631 MiB |  0.8180 |    6.93x   |

\* mmsegmentation's IQR exceeds the 5% gate at this small workload —
200 images of jittered uint8 PNGs sits at the noise floor of
per-image torch-tensor + histc overhead. The val2017 cell at 5000
images is well inside the gate.

### Read against the table

- The cross-impl strict-tier parity that ADR-0028 promised is now a
  side effect of every bench run — the val2017 cell touches 1.5 B+
  pixels and bit-equates on every per-class u64 total. Headline
  speedup is 4.22x at val2017 scale and 6.93x at the synthetic
  workload; the gap widens at small workloads because vernier's
  framework overhead is closer to fixed-cost than mmseg's
  per-image torch.histc + numpy round-trip.
- mmsegmentation's RSS sits at ~650 MiB (val2017) — torch's
  per-process state dominates the working set, even with no
  models loaded. vernier holds at ~92 MiB by streaming PNGs
  through the libpng-decode + fold path under `py.detach`.
- The previous snapshot's PR-B7 numbers (oracle 26.6 s, vernier 5.2 s)
  were measured ad-hoc through the parity smoke at
  `tests/python/parity_semantic/`; this round measures the same
  oracle under the bench harness's release-mode rigor (10 reps + 2
  warmup, randomised impl order, IQR gate). mmseg lands ~4.5 s
  faster than the PR-B7 ad-hoc number because the bench runner
  feeds it native uint8 PNGs (mmseg's IoUMetric does its own
  `.float()` promotion inside `intersect_and_union`, so a pre-cast
  int64 only doubles the working set for no parity benefit).

**`ade20k_val_*` still raises** `NotImplementedError` from the
resolver — gated on the license-cleared ADE20K val cache, not on
the mmseg oracle (which is now wired).

## How to refresh

```bash
just bench-sync   # syncs every per-impl env

# Instance — bbox / segm / boundary at val2017 scale. Release mode
# does 2 warmup + 10 measurement reps with randomised impl ordering
# and gates each impl on relative IQR ≤ 5%; ~30 min wall for the
# boundary cell because boundary-iou-api is ~62 s/rep.
for iou in bbox segm boundary; do
  vernier-bench run --paradigm instance --impl all \
    --workload coco_val2017_jittered_seed0 --iou $iou --mode release
done

# Instance — keypoints (different workload; sub-second).
vernier-bench run --paradigm instance --impl all \
  --workload coco_val2017_keypoints_jittered_seed0 --iou keypoints --mode release

# Panoptic — needs the ~3 GB GT cache provisioned once.
python -m panoptic_val_cache
vernier-bench run --paradigm panoptic --impl all \
  --workload coco_panoptic_val2017_perfect --mode release

# Semantic — vernier_semantic vs the vendored mmsegmentation oracle
# (ADR-0036, pinned commit c685fe6). The first run materializes the
# synthetic cache under bench/.cache/synthetic_semantic/; the val2017
# cell reuses the panoptic GT cache.
vernier-bench run --paradigm semantic --impl all \
  --workload "synthetic_semantic:n_images=200,n_classes=19,seed=0" --mode release
vernier-bench run --paradigm semantic --impl all \
  --workload coco_val2017_semantic_perfect --mode release

# Re-pull the medians + IQRs + RSS from the result tree (release mode
# carries the aggregation block; the docs renderer reads from there):
for cell in instance/coco_val2017_jittered_seed0/{bbox,segm,boundary} \
            instance/coco_val2017_keypoints_jittered_seed0/keypoints \
            panoptic/coco_panoptic_val2017_perfect/pq \
            semantic/synthetic_semantic_n200_c19_s0/miou; do
  for f in bench/results/<sha>/<fp>/$cell/*.json; do
    [ -f "$f" ] && python -c "import json; d=json.load(open('$f')); a=d['aggregation']; t=a['stages']['total']; m=a['memory']; g=a['iqr_gate']; print(t['median_ns'], t['iqr_ns'], m['max_bytes'], g['relative'], g['passed'])"
  done
done

# Or regenerate docs/benchmarks.md from the freshly populated result tree:
python tools/render_benchmarks.py
```
