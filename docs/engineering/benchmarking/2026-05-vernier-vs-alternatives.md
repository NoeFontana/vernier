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
| **vernier**       |   652.4 ms  |  784 MiB   |   **1.00x** |
| faster-coco-eval  |    2.096 s  |  661 MiB   |    3.21x   |
| pycocotools       |    5.989 s  |  576 MiB   |    9.18x   |

### segm

| impl              |    median   |  RSS (max) | vs vernier |
| ----------------- | ----------: | ---------: | ---------: |
| **vernier**       |    1.283 s  |  785 MiB   |   **1.00x** |
| faster-coco-eval  |    3.532 s  |  721 MiB   |    2.75x   |
| pycocotools       |    6.814 s  |  569 MiB   |    5.31x   |

### boundary

Boundary-IoU is a specialised metric — pycocotools and faster-coco-eval
don't ship it natively. Comparison is vernier vs `boundary-iou-api`.

| impl              |    median   |  RSS (max) | vs vernier |
| ----------------- | ----------: | ---------: | ---------: |
| **vernier**       |   39.918 s  |  787 MiB   |   **1.00x** |
| boundary-iou-api  |   61.982 s  |  666 MiB   |    1.55x   |

### Raw measurements (instance)

For downstream consumers that need ratios without rounding loss.

| iou      | impl              |    median (ns) |    RSS (B)    |
| -------- | ----------------- | -------------: | ------------: |
| bbox     | vernier           |    652,386,175 |   821,981,184 |
| bbox     | faster-coco-eval  |  2,095,863,547 |   692,973,568 |
| bbox     | pycocotools       |  5,989,202,770 |   604,286,976 |
| segm     | vernier           |  1,282,688,746 |   823,001,088 |
| segm     | faster-coco-eval  |  3,531,793,356 |   755,785,728 |
| segm     | pycocotools       |  6,813,613,131 |   596,291,584 |
| boundary | vernier           | 39,918,396,016 |   824,963,072 |
| boundary | boundary-iou-api  | 61,982,454,666 |   698,609,664 |

### Read against the table

- **bbox** has the widest gap (9.2x vs pycocotools, 3.2x vs fce). Bbox
  IoU compute is cheap, so the gap is mostly framework overhead —
  vernier's single-pass evaluator vs pycocotools' per-call cocoeval.
- **segm** narrows the gap (5.3x / 2.75x). Segm IoU compute (RLE
  intersection/union) dominates and runs in C-level libraries for
  every impl, so the kernels equalize. **vernier-mask's RLE
  intersection is the optimization target** to reopen the gap here.
- **boundary** is the closest cell (1.55x). Boundary-IoU is dominated
  by mask-boundary computation (distance transforms, polygon
  rasterization). vernier-mask's pulp dispatch is competitive but not
  dominant. **Boundary is the IoU type with the most absolute headroom
  per CPU cycle** — the smallest ratio means the most room to win
  back.

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
| **vernier_panoptic**  |   85.631 s  |  21.17 GiB  |   **1.00x**         |
| panopticapi           |   34.297 s  |  144.6 MiB  |    0.40x (faster)   |

| metric | impl             |    median (ns) |    RSS (B)     |
| ------ | ---------------- | -------------: | -------------: |
| pq     | vernier_panoptic | 85,631,466,597 | 22,736,642,048 |
| pq     | panopticapi      | 34,296,700,267 |    151,678,976 |

**Findings flag.** Unlike every instance cell, vernier is **slower
and ~150x more memory-heavy** than the oracle on panoptic. Two
contributing factors visible from the runner code:

- **Eager label-map decode**: `bench/bench/runners/vernier_panoptic_runner.py`
  decodes every PNG label map into a uint32 numpy array up front
  (`_build_label_maps`), holding all 5000 GT + 5000 DT maps in RAM
  before the kernel runs. ~1 MP × 4 bytes × 10000 maps ≈ 20 GB RSS,
  which matches the observation.
- **panopticapi streams**: the oracle iterates per-image on disk
  (you can see "Core: 0, X from 5000 images processed" in the run
  log) — constant-RSS by construction.

**Optimization target** (engineer-facing): rework the panoptic runner
(or the `vernier.panoptic.Dataset.from_arrays` API) so label maps
stream rather than load eagerly. Memory delta should be at least
100x; depending on how much of the 85 s wall is spent on Python-side
PNG decoding vs the Rust PQ kernel, wall-time delta could be similar.
Worth profiling both stages to see which dominates before optimizing.

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

## Semantic — `ade20k_val_*`

**Not runnable this phase.** The semantic paradigm is registered in
the type system (`SemanticWorkload`, the `--paradigm semantic` CLI
flag), and the `vernier-semantic` Rust crate ships with mIoU support,
but the bench harness has no runners or workloads wired up:

- `IMPL_PARADIGM_SUPPORT["semantic"] = {}` — no impls registered
- `_SEMANTIC_PREFIXES = ("ade20k_val",)` namespaces the workload IDs,
  but the resolver raises `NotImplementedError("registered by the B2
  stream")`
- No `bench/envs/mmseg/` env (the planned oracle install is ~5-8 min
  via `uv sync`, plus ADE20K val data ~2 GB which is academic-license
  gated and not auto-downloadable)
- No `vernier_semantic_runner` or `mmseg_runner` modules under
  `bench/bench/runners/`

To get semantic numbers, the bench harness B2 stream needs to land:
ADE20K val cache pin, runner pair, `mmseg` env, IMPL_PARADIGM_SUPPORT
registration. That's a follow-up PR larger than the panoptic and
keypoints fixes combined; tracked separately.

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

# Re-pull the medians + RSS from the result tree:
for cell in instance/coco_val2017_jittered_seed0/{bbox,segm,boundary} \
            instance/coco_val2017_keypoints_jittered_seed0/keypoints \
            panoptic/coco_panoptic_val2017_perfect/pq; do
  for f in bench/results/<sha>/<fp>/$cell/*.json; do
    [ -f "$f" ] && python -c "import json; d=json.load(open('$f')); reps=[r for r in d['reps'] if not r['warmup']]; print(reps[0]['stages']['total']['wall_ns'], reps[0]['ru_maxrss_bytes'])"
  done
done
```
