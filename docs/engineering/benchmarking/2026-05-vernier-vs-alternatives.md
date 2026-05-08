# 2026-05 vernier vs alternatives — cross-paradigm scan

Engineer-facing snapshot of vernier vs the third-party libraries we
benchmark against, run across every IoU kind / paradigm we currently
support. Companion to:

- [`v0.0.1-snapshot.md`](v0.0.1-snapshot.md) — release-mode baseline
  (placeholders today, fills in when a release-mode rerun lands)
- [`2026-05-scaling.md`](2026-05-scaling.md) — engineer-facing synthetic
  ladder (image-count scaling on bbox)

This doc is **dev-mode N=1**. Every cell is one measurement rep, no
warmup, no IQR gate. That's enough to read the gap between vernier and
each oracle on the same workload — within-cell variance is dominated
by the data (5000 val2017 images per cell), not run-to-run jitter.

## Shared configuration

- **Harness mode**: dev (N=1, no warmup, no governor pre-flight)
- **Machine fingerprint**: ``5658de0e29a3``
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

| impl              |    median   |  RSS (max) | vs vernier |
| ----------------- | ----------: | ---------: | ---------: |
| **vernier**       |   362.1 ms  |  235 MiB   |   **1.00x** |
| faster-coco-eval  |    2.106 s  |  661 MiB   |    5.81x   |
| pycocotools       |    5.779 s  |  576 MiB   |   15.96x   |

### segm

| impl              |    median   |  RSS (max) | vs vernier |
| ----------------- | ----------: | ---------: | ---------: |
| **vernier**       |   986.1 ms  |  236 MiB   |   **1.00x** |
| faster-coco-eval  |    3.517 s  |  721 MiB   |    3.57x   |
| pycocotools       |    6.819 s  |  568 MiB   |    6.92x   |

### boundary

Boundary-IoU is a specialised metric. The reference implementation is
`boundary-iou-api`; faster-coco-eval ≥1.6 ships its own boundary
surface alongside the COCOeval drop-in. pycocotools doesn't expose
boundary natively.

| impl              |    median   |  RSS (max) | vs vernier |
| ----------------- | ----------: | ---------: | ---------: |
| **vernier**       |    4.158 s  |  238 MiB   |   **1.00x** |
| faster-coco-eval  |   17.840 s  |  794 MiB   |    4.29x   |
| boundary-iou-api  |   62.075 s  |  663 MiB   |   14.93x   |

The faster-coco-eval boundary cell is timing-only — it isn't gated by
parity. The harness's boundary parity tier compares vernier vs
`boundary-iou-api` (the per-quirk strict reference for ADR-0010).
faster-coco-eval's boundary algorithm shares the 0.02 dilation-ratio
default but its band-derivation path differs in detail; pinning a
tolerance is a separate ADR-level decision, not a measurement
question.

### Raw measurements (instance)

For downstream consumers that need ratios without rounding loss.

| iou      | impl              |    median (ns) |    RSS (B)    |
| -------- | ----------------- | -------------: | ------------: |
| bbox     | vernier           |    362,102,286 |   246,702,080 |
| bbox     | faster-coco-eval  |  2,105,592,180 |   693,239,808 |
| bbox     | pycocotools       |  5,778,733,009 |   604,020,736 |
| segm     | vernier           |    986,142,427 |   246,964,224 |
| segm     | faster-coco-eval  |  3,517,070,528 |   755,789,824 |
| segm     | pycocotools       |  6,819,027,732 |   596,033,536 |
| boundary | vernier           |  4,158,086,331 |   249,311,232 |
| boundary | faster-coco-eval  | 17,840,309,949 |   832,577,536 |
| boundary | boundary-iou-api  | 62,074,594,905 |   695,209,984 |

### Read against the table

- **bbox** widens to 16x vs pycocotools / 5.8x vs fce. Bbox IoU compute
  is cheap, so the gap is mostly framework overhead — vernier's
  single-pass evaluator vs pycocotools' per-call cocoeval. Vernier's
  RSS is also a third of fce's; both are loading the same val2017
  GT/DT but vernier ingests via the binary FFI without materializing
  the JSON-shaped intermediate dicts each oracle keeps around.
- **segm** is 6.9x vs pycocotools / 3.6x vs fce. The RLE
  intersection/union kernel still does most of the work for every
  impl, but the framework-overhead delta now exposes itself per-cell
  more visibly than the previous round. PR #183 fused the per-cell
  bbox+area+offsets walk; further compression here means going after
  the intersection sweep itself.
- **boundary** moved the most this round. The previous snapshot was
  1.55x vs `boundary-iou-api` (the only oracle); after the bbox round
  + #181/#182/#184/#185 it's now 14.9x. faster-coco-eval's boundary
  cell sits between the two — 4.3x slower than vernier, 3.5x faster
  than `boundary-iou-api` — which calibrates how much of vernier's
  boundary win is its specific erode pipeline (u64-packed row pass,
  bbox-cropped) vs general framework-overhead reduction. **Boundary
  was identified as the cell with the most absolute headroom per CPU
  cycle in the previous snapshot — that headroom is now realized.**

## Instance — `coco_val2017_keypoints_jittered_seed0` (keypoints)

5000-image val2017 person subset, deterministic OKS-jittered DT.

| impl              |    median   |  RSS (max) | vs vernier |
| ----------------- | ----------: | ---------: | ---------: |
| **vernier**       |   146.7 ms  |  164 MiB   |   **1.00x** |
| faster-coco-eval  |    1.721 s  |  164 MiB   |   11.74x   |
| pycocotools       |    2.304 s  |  164 MiB   |   15.71x   |

| iou       | impl              |     median (ns) |    RSS (B)    |
| --------- | ----------------- | --------------: | ------------: |
| keypoints | vernier           |     146,668,518 |   171,753,472 |
| keypoints | faster-coco-eval  |   1,721,352,866 |   171,753,472 |
| keypoints | pycocotools       |   2,303,532,792 |   171,753,472 |

The keypoints workload is small (val2017 keypoints subset, ~6k
annotations) so vernier's framework-overhead advantage dominates —
widest gap of any IoU type. Identical RSS across impls reflects the
fixed cost of loading the GT/DT JSONs.

**Harness fix landed this phase**: keypoints jittered DT was
generating against `instances_val2017.json` (which lacks `keypoints`
fields, producing 0 detections). Patched to plumb
`person_keypoints_val2017.json` through `coco_val_cache`'s
`ensure_kp_gt()` + a new `kp_gt_path()` helper.

## Panoptic — `coco_panoptic_val2017_perfect`

5000 val2017 images, perfect-DT (GT-as-DT). Strict-tier parity
passes against panopticapi on every cell (PQ=1.0 by construction).

| impl                  |   median    | RSS (max)   | vs vernier_panoptic |
| --------------------- | ----------: | ----------: | ------------------: |
| **vernier_panoptic**  |   32.337 s  |  127.2 MiB  |   **1.00x**         |
| panopticapi           |   35.903 s  |  147.2 MiB  |    1.11x (slower)   |

| metric | impl             |    median (ns) |    RSS (B)     |
| ------ | ---------------- | -------------: | -------------: |
| pq     | vernier_panoptic | 32,337,000,000 |    133,373,952 |
| pq     | panopticapi      | 35,903,000,000 |    154,374,144 |

**Findings flag (resolved).** Two consecutive rounds closed the gap
panopticapi held over vernier on this cell:

| round | wall | RSS | vs panopticapi |
| --- | ---: | ---: | ---: |
| 0 (eager decode) | 85.6 s | 21.17 GiB | 0.40x (slower) |
| 1 (streaming refactor, #187) | 51.0 s | 130 MiB | 0.68x (slower) |
| **2 (FxHash internal maps, #188)** | **32.3 s** | **127 MiB** | **1.11x (faster)** |

Round 1 replaced the runner's eager-PNG-decode loop with
`vernier.panoptic.Evaluator.background()` — PNG decode runs on the
main thread while the Rust PQ kernel folds on a worker, so RSS is
bounded by `queue_capacity × image_size × 4 bytes` instead of
`n_images × …`. Round 2 swapped the per-image `HashMap<(u32, u32),
u32>` (intersection histogram) and `HashMap<u32, SegmentInfo>` (DT
validation) to `FxHashMap`: SipHash's DoS resistance was wasted work
on internal integer keys walked at ~1.5 B ops per cell. The `submit`
hot path dropped 24.9 s → 4.7 s (5.3x); the divan-level kernel arm
dropped 3.56 ms → 0.63 ms.

Strict-tier parity vs `pq_compute_single_core` still passes — FxHash
is deterministic and the histogram-iteration order is irrelevant per
quirk U9.

**Remaining headroom**: PNG decode (~33 s) is now the long pole and
runs on the main thread. The Rust PQ kernel is GIL-free on a worker;
the cell is decode-bound. Faster panoptic PNG decode (e.g.,
`image-rs` PNG over `Pillow`) would shave wall-time further, but at
that point we're optimizing PIL's RGB→uint32 conversion, which
panopticapi also pays.

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

## Semantic — `synthetic_semantic:*` and `ade20k_val_*`

**Vernier-only baseline runnable this phase.** The bench harness now
dispatches `--paradigm semantic`: `IMPL_PARADIGM_SUPPORT["semantic"] =
{"vernier_semantic": {"miou"}}`, the workload resolver takes
`synthetic_semantic:n_images=,n_classes=,seed=` (deterministic uint8
PNG label-map pairs cached under `bench/.cache/synthetic_semantic/`),
and `bench/bench/runners/vernier_semantic_runner.py` streams via
`vernier.semantic.Evaluator.stream()`. No oracle yet — the
`mmsegmentation` env (PR-B6/B7/B8) and ADE20K val cache (academic
license, no auto-download) are external blockers, so parity is
skipped while only one impl is registered.

A 200-image / 19-class / `jitter_rate=0.1` synthetic cell on this
host:

| impl              |   median   | RSS (max) | mIoU    |
| ----------------- | ---------: | --------: | ------: |
| vernier_semantic  |   200 ms   |  93.8 MiB |  0.8180 |

This is a measurement loop, not a comparison — the synthetic workload
is for iterating on the kernel under realistic-shaped data, not for
calibrating against a third party. ADE20K + mmseg parity (S3-B) is
where the comparable numbers will live.

**`ade20k_val_*` still raises** `NotImplementedError` from the
resolver, with an updated message pointing at the external blockers.

## How to refresh

```bash
just bench-sync   # syncs every per-impl env

# Instance — bbox / segm / boundary at val2017 scale.
for iou in bbox segm boundary; do
  vernier-bench run --paradigm instance --impl all \
    --workload coco_val2017_jittered_seed0 --iou $iou --mode dev
done

# Instance — keypoints (different workload; sub-second).
vernier-bench run --paradigm instance --impl all \
  --workload coco_val2017_keypoints_jittered_seed0 --iou keypoints --mode dev

# Panoptic — needs the ~3 GB GT cache provisioned once.
python -m panoptic_val_cache
vernier-bench run --paradigm panoptic --impl all \
  --workload coco_panoptic_val2017_perfect --mode dev

# Semantic — vernier-only baseline against a synthetic workload
# (no oracle until S3-B mmsegmentation env lands). The first run
# materializes the cache under bench/.cache/synthetic_semantic/.
vernier-bench run --paradigm semantic --impl all \
  --workload "synthetic_semantic:n_images=200,n_classes=19,seed=0" --mode dev

# Re-pull the medians + RSS from the result tree:
for cell in instance/coco_val2017_jittered_seed0/{bbox,segm,boundary} \
            instance/coco_val2017_keypoints_jittered_seed0/keypoints \
            panoptic/coco_panoptic_val2017_perfect/pq \
            semantic/synthetic_semantic_n200_c19_s0/miou; do
  for f in bench/results/<sha>/<fp>/$cell/*.json; do
    [ -f "$f" ] && python -c "import json; d=json.load(open('$f')); reps=[r for r in d['reps'] if not r['warmup']]; print(reps[0]['stages']['total']['wall_ns'], reps[0]['ru_maxrss_bytes'])"
  done
done
```
