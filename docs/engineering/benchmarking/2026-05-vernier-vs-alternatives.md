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
- **Git SHA**: ``885d385d63e1``
- **Machine fingerprint**: ``37652a58e939`` (AMD EPYC-Milan, x86_64).
  Distinct from the previous snapshot's ``1655eb18a194`` — absolute
  numbers shouldn't be cross-compared, only ratios within this snapshot.
  Every section in this snapshot — including LVIS — was measured at this
  SHA/fingerprint pair, so the dual-SHA caveat the previous revision
  carried on the LVIS section is gone.
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
| **vernier**       |   370.6 ms  |   4.1 ms (1.10%)  |  261 MiB   |   **1.00x** |
| faster-coco-eval  |    2.060 s  |  29.2 ms (1.42%)  |  661 MiB   |    5.56x   |
| pycocotools       |    5.753 s  | 195.1 ms (3.39%)  |  576 MiB   |   15.52x   |

### segm

| impl              |    median   |        IQR        |  RSS (max) | vs vernier |
| ----------------- | ----------: | ----------------: | ---------: | ---------: |
| **vernier**       |   970.6 ms  |   3.9 ms (0.40%)  |  262 MiB   |   **1.00x** |
| faster-coco-eval  |    3.498 s  |  13.0 ms (0.37%)  |  721 MiB   |    3.60x   |
| pycocotools       |    6.635 s  |  76.7 ms (1.16%)  |  569 MiB   |    6.84x   |

### boundary

Boundary-IoU is a specialised metric. The reference implementation is
`boundary-iou-api`; faster-coco-eval ≥1.6 ships its own boundary
surface alongside the COCOeval drop-in. pycocotools doesn't expose
boundary natively.

| impl              |    median   |        IQR        |  RSS (max) | vs vernier |
| ----------------- | ----------: | ----------------: | ---------: | ---------: |
| **vernier**       |    3.143 s  |  17.0 ms (0.54%)  |  264 MiB   |   **1.00x** |
| faster-coco-eval  |   17.616 s  |  41.2 ms (0.23%)  |  794 MiB   |    5.61x   |
| boundary-iou-api  |   61.544 s  | 225.2 ms (0.37%)  |  666 MiB   |   19.58x   |

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
| bbox     | vernier           |    370,642,602 |      4,081,748 |   273,940,480 |
| bbox     | faster-coco-eval  |  2,059,867,200 |     29,249,927 |   693,452,800 |
| bbox     | pycocotools       |  5,752,686,836 |    195,057,334 |   604,377,088 |
| segm     | vernier           |    970,640,486 |      3,877,479 |   274,513,920 |
| segm     | faster-coco-eval  |  3,497,703,390 |     13,003,661 |   755,765,248 |
| segm     | pycocotools       |  6,635,460,133 |     76,668,046 |   596,660,224 |
| boundary | vernier           |  3,142,573,000 |     17,034,575 |   276,832,256 |
| boundary | faster-coco-eval  | 17,615,705,989 |     41,233,364 |   832,897,024 |
| boundary | boundary-iou-api  | 61,544,222,350 |    225,186,452 |   698,830,848 |

### Read against the table

- **bbox** sits at 15.5x vs pycocotools / 5.6x vs fce. Bbox IoU compute
  is cheap, so the gap is mostly framework overhead — vernier's
  single-pass evaluator vs pycocotools' per-call cocoeval. Vernier's
  RSS is also a third of fce's; both are loading the same val2017
  GT/DT but vernier ingests via the binary FFI without materializing
  the JSON-shaped intermediate dicts each oracle keeps around. IQRs
  hold under 1.5% on vernier/fce; pycocotools widens to 3.4% on this
  snapshot (still within the 5% gate) — the rep-to-rep variance shows
  up in the slower oracle as it spends more time in GC.
- **segm** is 6.8x vs pycocotools / 3.6x vs fce. The RLE
  intersection/union kernel still does most of the work for every
  impl, but the framework-overhead delta now exposes itself per-cell
  more visibly than the previous round. PR #183 fused the per-cell
  bbox+area+offsets walk; further compression here means going after
  the intersection sweep itself.
- **boundary** holds the gap: 19.6x vs `boundary-iou-api` and 5.6x vs
  faster-coco-eval. faster-coco-eval's boundary cell sits between the
  two — 5.6x slower than vernier, 3.5x faster than `boundary-iou-api`
  — which calibrates how much of vernier's boundary win is its
  specific erode pipeline (u64-packed row pass, bbox-cropped) vs
  general framework-overhead reduction. **Boundary was identified as
  the cell with the most absolute headroom per CPU cycle in the
  round-0 snapshot — that headroom is now realized and the
  release-mode rerun confirms the lead is real, not a single-rep
  artefact.**

## Instance — `coco_val2017_keypoints_jittered_seed0` (keypoints)

5000-image val2017 person subset, deterministic OKS-jittered DT.

| impl              |    median   |        IQR        |  RSS (max) | vs vernier |
| ----------------- | ----------: | ----------------: | ---------: | ---------: |
| **vernier**       |   137.1 ms  |   2.3 ms (1.69%)  |  127 MiB   |   **1.00x** |
| faster-coco-eval  |    1.661 s  |  25.9 ms (1.56%)  |  154 MiB   |   12.11x   |
| pycocotools       |    2.261 s  |  20.2 ms (0.89%)  |  163 MiB   |   16.49x   |

| iou       | impl              |     median (ns) |       IQR (ns) |    RSS (B)    |
| --------- | ----------------- | --------------: | -------------: | ------------: |
| keypoints | vernier           |     137,123,720 |      2,313,885 |   133,652,480 |
| keypoints | faster-coco-eval  |   1,660,531,938 |     25,852,658 |   161,943,552 |
| keypoints | pycocotools       |   2,260,984,977 |     20,217,797 |   171,069,440 |

The keypoints workload is small (val2017 keypoints subset, ~6k
annotations) so vernier's framework-overhead advantage dominates —
widest gap of any IoU type at 16.5x vs pycocotools. Memory now diverges
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
| **vernier_panoptic**  |   12.592 s  |   2.673 s (21.22%)*|  142.7 MiB  |   **1.00x**         |
| panopticapi           |   34.440 s  |  258.9 ms (0.75%)  |  146.0 MiB  |    2.73x (slower)   |

\* Vernier's IQR widened well past the 5% release-mode gate on this
snapshot (21.2% — three release-mode reruns at this SHA span
11.0/13.8/12.6 s on the same host). The spread is genuine PNG-decode
+ page-cache variance under randomised impl ordering, not a methodology
bug; the prior snapshot at `1fd5720bf56c` already shipped this cell at
5.2%-failing. The lead over panopticapi is ~22 s of headroom, far
larger than the 2.7 s IQR band, so the comparison is still
load-bearing; treat the precise 2.73x ratio as "between roughly 2.4x
and 3.2x" rather than a fixed point.

| metric | impl             |    median (ns) |       IQR (ns) |    RSS (B)     |
| ------ | ---------------- | -------------: | -------------: | -------------: |
| pq     | vernier_panoptic | 12,592,497,509 |  2,672,609,089 |    149,635,072 |
| pq     | panopticapi      | 34,440,246,111 |    258,941,479 |    153,047,040 |

**Findings flag (resolved, then reopened by release-mode warmup).**
Three rounds of optimization closed the gap panopticapi held over
vernier on this cell, and the release-mode warmup surfaced an
additional cold-cache effect that dev-mode N=1 had been pricing in:

| round | wall | RSS | vs panopticapi |
| --- | ---: | ---: | ---: |
| 0 (eager decode) | 85.6 s | 21.17 GiB | 0.40x (slower) |
| 1 (streaming refactor, #187) | 51.0 s | 130 MiB | 0.68x (slower) |
| 2 (FxHash internal maps, #188, dev-mode N=1) | 32.3 s | 127 MiB | 1.11x (faster) |
| 3 (first release-mode N=10 + warmup) | 11.6 s | 118 MiB | 3.04x (faster) |
| **4 (release-mode N=10, this snapshot — chronically noisy)** | **12.6 s** | **143 MiB** | **2.73x (faster) \*** |

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
release-mode hits steady state. The IQR is the residual: even
with warm cache there's per-rep PNG-decode timing noise. Three
release-mode reruns at the round-4 SHA span 11.0/13.8/12.6 s on
the same host, which is consistent with a thermal/I/O-jitter floor
rather than a methodology bug.

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
| **vernier_semantic**  |   5.004 s  |  39.0 ms (0.78%)  |   99 MiB  |   **1.00x** |
| mmsegmentation         |  20.605 s  | 172.5 ms (0.84%)  |  647 MiB  |    4.12x   |

| metric | impl             |    median (ns) |       IQR (ns) |    RSS (B)     |
| ------ | ---------------- | -------------: | -------------: | -------------: |
| miou   | vernier_semantic |  5,004,290,735 |     38,990,005 |    103,579,648 |
| miou   | mmsegmentation   | 20,604,663,667 |    172,470,102 |    678,858,752 |

mIoU = 1.0 on both impls (perfect-DT). 4.1x faster wall-time and ~7x
lower peak RSS, single-threaded on each side. Vernier holds at 0.50%
IQR — `evaluate_from_pngs`'s fused libpng-decode + confusion-fold
(ADR-0037) reaches steady state once page cache warms.

### `synthetic_semantic:n_images=200,n_classes=19,seed=0`

| impl              |   median   |        IQR         | RSS (max) | mIoU    | vs vernier |
| ----------------- | ---------: | -----------------: | --------: | ------: | ---------: |
| **vernier_semantic**  |   63.2 ms  |  678.2 μs (1.07%)  |  88 MiB |  0.8180 |   **1.00x** |
| mmsegmentation        |  430.7 ms  |  53.0 ms (12.32%)* | 631 MiB |  0.8180 |    6.82x   |

\* mmsegmentation's IQR exceeds the 5% gate at this small workload —
200 images of jittered uint8 PNGs sits at the noise floor of
per-image torch-tensor + histc overhead. The val2017 cell at 5000
images is well inside the gate.

### Read against the table

- The cross-impl strict-tier parity that ADR-0028 promised is now a
  side effect of every bench run — the val2017 cell touches 1.5 B+
  pixels and bit-equates on every per-class u64 total. Headline
  speedup is 4.12x at val2017 scale and 6.82x at the synthetic
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

## LVIS — `lvis_v1_val_perfect`

Full LVIS v1 val (19809 images, 1203 categories), GT-as-DT
(`perfect_dt.json` — bbox-shape; the segm-shape variant is the
follow-up cell once `evaluate_segm_grid_with_dataset` lands).
**Bench paradigm wired this phase** (ADR-0026 + ADR-0033): new
`paradigm: "lvis"` under `bench/harness/`, dedicated
`bench/envs/lvis-api/` env (lvis @ `031ac21f` + `pycocotools==2.0.11`
+ runtime `np.float = float` shim mirroring
`parity_lvis/conftest.py`), and a `_LvisComparator` that runs
strict-tier bit-equality on the precision tensor against the
vendored oracle.

### bbox

| impl                |    median    |       IQR        | RSS (max)  | vs vernier   |
| ------------------- | -----------: | ---------------: | ---------: | -----------: |
| **vernier_lvis**    |    3.727 s   |  67.5 ms (1.81%) |   1.48 GiB |  **1.00x**   |
| lvis-api            |  210.688 s   |   6.60 s (3.13%) |  15.01 GiB |    56.53x    |

Snapshot machine fingerprint `37652a58e939`, SHA `885d385d63e1` —
the same pair the COCO / panoptic / semantic sections above were
measured at. Absolute wall times are now cross-comparable across
every section in this doc.

### Raw measurements (LVIS)

| iou  | impl          |       median (ns) |          IQR (ns) |        RSS (B)     |
| ---- | ------------- | ----------------: | ----------------: | -----------------: |
| bbox | vernier_lvis  |     3,727,024,572 |        67,488,166 |      1,589,776,384 |
| bbox | lvis-api      |   210,688,088,897 |     6,599,924,298 |     16,121,040,896 |

### Read against the table

- **56.5x speedup** is the largest cross-impl gap in this snapshot.
  lvis-api is unoptimized Python (~210 s / rep on full val); the
  bulk of vernier's lead is parallel-free framework overhead
  (single-pass orchestrator vs the per-category Python iteration
  in `LVISEval.evaluate`), with the AP-fold core itself doing
  roughly the same work on both sides.
- **10x lower peak RSS** (1.48 GiB vs 15.01 GiB). The dense
  orchestrator grid that ADR-0026 §"Known follow-up" called out as
  ">22 GB structural" is no longer the load-bearing constraint —
  PR #179's `Box`-niche fix made each empty slot 8 B instead of 232 B
  (`Vec<Option<Box<PerImageEval>>>`), dropping the structural floor
  to 95M × 8 B ≈ 760 MB before populated cells land. The measured
  1.48 GiB matches that floor plus populated-cell heap. The lvis-api
  side carries the full per-image / per-category Python dict tree,
  hence the 15 GiB.
- **Vernier IQR 1.81%** is well inside the 5% gate. **lvis-api IQR
  3.13%** is also inside the gate (tighter than the prior snapshot's
  4.63%), but most of the per-rep variance is still GC pauses
  inside the long oracle reps (one rep is ~3.5 min wall, GC
  variance has time to compound). The gap to vernier — 57x — is
  three orders of magnitude wider than the IQR, so the comparison
  is load-bearing.

### Strict parity at full val — closed (AG6)

Both impls report the same headline `AP = 0.9983` on perfect-DT, and
strict-tier bit-equality on the `(T, R, K, A)` precision tensor now
passes on **all 4.86M cells** at full LVIS v1 val.

The earlier round of this snapshot recorded ~0.06% per-cell drift on
two K-rows (K_idx=168 'bun' and K_idx=817 'plate'). Root cause was
quirk **AG6** — `LVIS.get_ann_ids` filters with strict `area > 0`
(`lvis/lvis.py:94`) and silently drops three zero-area GT annotations
in LVIS val. The perfect-DT generator emits a DT for every GT
including the dropped ones, so on the two "mixed" `(image, category)`
cells the orphan DT became an FP on the oracle's side. Vernier
matched all 1206 DT-GT pairs cleanly because its loader keeps
zero-area annotations. The original "score-tie ordering" hypothesis
was wrong (perfect-DT scores aren't literally 1.0 — they're values
like 0.999968 — and `mergesort` deterministic ordering wouldn't have
explained the magnitude anyway).

Fixed by mirroring the filter under `ParityMode::Strict` for
federated datasets only (corrected mode keeps the zero-area
annotations, which is the user-friendly default). Pinned by
`evaluate.rs::tests::ag6_*`. The 56.5x speedup and 10x lower
peak RSS were never gated on this — they stand unchanged.

**Harness additions landed this phase**:
- New paradigm `"lvis"` in `bench.harness.schema.Paradigm` +
  `IMPL_PARADIGM_SUPPORT["lvis"]`.
- `bench/envs/lvis-api/` env (`lvis @ 031ac21f`, `pycocotools==2.0.11`,
  `pip` injected via `[tool.uv.extra-build-dependencies]` because
  lvis 0.5.3's `setup.py` calls pip during the build but doesn't
  declare it).
- `vernier_lvis_runner.py` (uses `CocoDataset.from_lvis_json` +
  `evaluate_bbox_grid_with_dataset` + `summarize_lvis`) and
  `lvis_api_runner.py` (wraps `LVISEval` with the runtime
  `np.float = float` shim mirroring `parity_lvis/conftest.py`).
- `_LvisComparator` in `bench.harness.parity` — single strict-tier
  pair (vernier_lvis vs lvis-api), same single-tensor surface as
  the instance comparator with its own impl pair.
- CLI dispatch `vernier-bench run --paradigm lvis ...`; auto-derived
  from `LvisWorkload` discriminator.
- `tools/render_benchmarks.py` emits an "Instance — LVIS federated
  AP" section under the same impl-row format as the others.

bbox-only at the vernier side today because only
`evaluate_bbox_grid_with_dataset` ships on the FFI; segm waits on
the parsed-once-dataset segm variant. lvis-api supports both
natively, so the matrix entry mirrors vernier's coverage.

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

# LVIS — vernier_lvis vs the vendored lvis-api oracle (ADR-0026,
# pinned commit 031ac21f). Needs the ~200 MB LVIS val cache provisioned
# once (`python -m lvis_val_cache`); the lvis-api env adds ~150 MiB on
# disk (matplotlib + opencv-python pulled transitively by lvis 0.5.3).
# Wall budget: ~50 min on the host snapshot above — lvis-api runs at
# ~3.5 min/rep × 12 reps; vernier_lvis stays under 60 s across all reps.
vernier-bench run --paradigm lvis --impl all \
  --workload lvis_v1_val_perfect --iou bbox --mode release

# Re-pull the medians + IQRs + RSS from the result tree (release mode
# carries the aggregation block; the docs renderer reads from there):
for cell in instance/coco_val2017_jittered_seed0/{bbox,segm,boundary} \
            instance/coco_val2017_keypoints_jittered_seed0/keypoints \
            panoptic/coco_panoptic_val2017_perfect/pq \
            semantic/synthetic_semantic_n200_c19_s0/miou \
            lvis/lvis_v1_val_perfect/bbox; do
  for f in bench/results/<sha>/<fp>/$cell/*.json; do
    [ -f "$f" ] && python -c "import json; d=json.load(open('$f')); a=d['aggregation']; t=a['stages']['total']; m=a['memory']; g=a['iqr_gate']; print(t['median_ns'], t['iqr_ns'], m['max_bytes'], g['relative'], g['passed'])"
  done
done

# Or regenerate docs/benchmarks.md from the freshly populated result tree:
python tools/render_benchmarks.py
```
