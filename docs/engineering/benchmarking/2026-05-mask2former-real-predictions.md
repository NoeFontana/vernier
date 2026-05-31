# 2026-05 Mask2Former real predictions — panoptic + semantic cells

Engineer-facing snapshot of vernier vs the alternatives on a *real*
SOTA prediction distribution from one architecture family. Two cells
in one doc because both share the Mask2Former Swin-T backbone and
the same SOTA-harness cache contract:

- **Panoptic** — `facebook/mask2former-swin-tiny-coco-panoptic` on
  COCO panoptic val2017 (Apache-2.0, CPU-runnable, 5000 images).
- **Semantic** — `facebook/mask2former-swin-tiny-ade-semantic` on
  ADE20K SceneParse150 val (Apache-2.0 model + research-use dataset,
  2000 images).

Synthetic perfect-DT (panoptic) and the COCO val2017 panoptic-derived
semantic GT-as-DT cell don't exercise the long-tail per-class
distribution a real SOTA model produces. These cells close that gap.

> **Status:** parity-validated on the live cache (2026-05-31). Both
> SOTA cells produced predictions on the full validation set
> (panoptic: 5000 / 5000 COCO val2017 images; semantic: 2000 / 2000
> ADE20K val images) and pass the parity gates documented below.
> Bench tables (median / IQR / RSS) are placeholders until the cells
> are timed via `just bench-run` on this snapshot's machine
> fingerprint; the parity claim above is the load-bearing
> correctness gate.

## Shared configuration

- **Harness mode**: release (N=10 + 2 warmup, randomised impl order,
  governor pre-flight, 5% relative-IQR gate per impl)
- **Git SHA**: `e146e16`
- **Machine fingerprint**: `84edec51fd71` (AMD EPYC-Milan, x86_64).
  Same machine as the DETR-R50 cell — absolute numbers can be
  cross-compared with that snapshot, not with the cross-paradigm
  snapshot at `37652a58e939`.
- **Build profile**: cargo release defaults (`opt-level=3`,
  `lto=thin`, `codegen-units=1`, no `target-cpu`). Same profile the
  PyPI wheel ships with.
- **Parity**: strict-tier (vs panopticapi for panoptic; vs
  mmsegmentation `IoUMetric` for semantic) passes on every reported
  cell. See "Parity" sections below.
- **Pinned model revisions** (resolved 2026-05-31, bumping is an
  ADR-level decision):
  - Panoptic: `df6b1142ff50c3276559d9d78f35f6a579c75a77`
  - Semantic: `c8cf1b5e823aee214d937d0d001c1850ba44ef6a`

## Panoptic — `coco_panoptic_val2017_mask2former_swin_t_v<sha>` (PQ)

| impl              |    median   |        IQR         |  RSS (max) | vs vernier |
| ----------------- | ----------: | -----------------: | ---------: | ---------: |
| **vernier**       |     `<TBD>` |          `<TBD>`   |    `<TBD>` |  **1.00x** |
| panopticapi       |     `<TBD>` |          `<TBD>`   |    `<TBD>` |    `<TBD>` |

### Per-stage breakdown (median ns)

| impl              |      load   |    evaluate    |   accumulate   |
| ----------------- | ----------: | -------------: | -------------: |
| **vernier**       |   `<TBD>`   |     `<TBD>`    |     `<TBD>`    |
| panopticapi       |   `<TBD>`   |     `<TBD>`    |     `<TBD>`    |

### Read against the table

(Populated post-inference; see the DETR-R50 cell at
[2026-05-detr-r50-real-predictions.md](./2026-05-detr-r50-real-predictions.md)
for the analytical shape — load-stage delta, RSS ratio, IQR
behaviour vs the synthetic counterpart.)

## Semantic — `ade20k_val_mask2former_swin_t_v<sha>` (mIoU)

| impl              |    median   |        IQR         |  RSS (max) | vs vernier |
| ----------------- | ----------: | -----------------: | ---------: | ---------: |
| **vernier**       |     `<TBD>` |          `<TBD>`   |    `<TBD>` |  **1.00x** |
| mmsegmentation    |     `<TBD>` |          `<TBD>`   |    `<TBD>` |    `<TBD>` |

### Per-stage breakdown (median ns)

| impl              |      load   |    evaluate    |   accumulate   |
| ----------------- | ----------: | -------------: | -------------: |
| **vernier**       |   `<TBD>`   |     `<TBD>`    |     `<TBD>`    |
| mmsegmentation    |   `<TBD>`   |     `<TBD>`    |     `<TBD>`    |

## Parity

### Panoptic (vs panopticapi)

- **Integer surface (strict-tier)**: bit-equality on the per-class
  `intersect_sum` / `union_sum` / `TP` / `FP` / `FN` counts that
  feed every reported metric. Drift here would mean a real divergence
  in the per-image PqStat fold; reaching the float averages confirms
  it doesn't.
- **Float averages (aligned-tier, 8 ULP relative)**: per-class PQ/SQ/RQ
  rows + the Things/Stuff bucket means. The `sum(iou)` over TPs per
  category and the `avg(metric)` over categories per bucket reduce
  in different orders between panopticapi (Python dict iteration over
  `pq_per_cat`) and vernier (its own per-category accumulator); the
  drift on the live cache is at most 2.5 ULP relative (per-class
  PQ for cat 3 = `5.55e-16` absolute, the worst entry in 5000 images
  × 133 classes). Bucket-level drift is ≤ 0.5 ULP.
- Reported on `tests/python/integration/real_models/sota/test_mask2former_panoptic_real_models.py`.
- **Headline numbers** (COCO panoptic val2017, 5000 images, 133
  categories):
  - Global PQ: `0.462607`, SQ: `0.815501`, RQ: `0.559097`
  - Things (80 classes): PQ `0.496540`, SQ `0.819126`, RQ `0.598265`
  - Stuff (53 classes): PQ `0.411386`, SQ `0.810030`, RQ `0.499976`
  - All 5000 images present in both GT and DT; all 133 classes
    accounted for in both buckets.

### Semantic (vs mmsegmentation IoUMetric)

- **Strict-tier**: bit-equality on the per-class u64 confusion-matrix
  totals (`intersect` / `union` / `pred` / `label`). Derived float
  scalars (mIoU, aAcc, per-class IoU/Acc) follow trivially from the
  same u64 inputs.
- Reported on `tests/python/integration/real_models/sota/test_mask2former_ade_real_models.py`.
- **Headline numbers** (ADE20K SceneParse150 val, 2000 images):
  - mIoU (mean over 150 classes): `0.462490`
  - aAcc (overall pixel accuracy): `0.819045`
  - Σ intersect: 367,015,797 pixels; Σ union: 529,188,289 pixels
  - All 150 classes present in both `label` and `pred`
  - vernier ↔ mmseg `IoUMetric`: bit-equal on `intersect`, `union`,
    `pred`, `label`

## Reproducing

The Mask2Former prediction caches are generated by the SOTA harness
(`tests/python/integration/real_models/sota/`). First-run costs are
~20-25h (panoptic, 5000 images x ~14-18s/image) and ~3-4h (semantic,
2000 images x ~5-7s/image) on an 8-core CPU. Cache hits are seconds.
The cache directory names embed the full hub commit SHA so a weights
bump invalidates by construction.

```bash
# One-time per host: pin MASK2FORMER_*_REVISION constants to current
# hub commit SHAs (the populator's preflight refuses to populate
# against the _UNPINNED_REVISION sentinel). Obtain the SHAs via:
python -c "from huggingface_hub import HfApi; print(HfApi().model_info('facebook/mask2former-swin-tiny-coco-panoptic').sha)"
python -c "from huggingface_hub import HfApi; print(HfApi().model_info('facebook/mask2former-swin-tiny-ade-semantic').sha)"
# Then edit MASK2FORMER_PANOPTIC_REVISION / MASK2FORMER_ADE_REVISION
# in tools/real_predictions_cache/real_predictions_cache/__init__.py.

# One-time: download the ADE20K val cache (~923MB challenge zip).
# First run prints the observed SHA; pin it in
# tools/ade20k_val_cache/ade20k_val_cache/__init__.py for strict
# subsequent verification.
python -m ade20k_val_cache

# Populate prediction caches (~20-25h panoptic, ~3-4h semantic).
./tools/fetch-real-predictions.sh --mask2former-panoptic
./tools/fetch-real-predictions.sh --mask2former-ade

# Run the bench cells.
VERNIER_COCO_CACHE=/path/to/coco-val2017 \
  just bench-run -- \
    --impl all \
    --workload coco_panoptic_val2017_mask2former_swin_t_v<short> \
    --paradigm panoptic \
    --mode release

just bench-run -- \
    --impl all \
    --workload ade20k_val_mask2former_swin_t_v<short> \
    --paradigm semantic \
    --mode release
```

Results land under `bench/results/<git-sha>/<machine-fp>/{panoptic,semantic}/...`.
