# Real-prediction parity gates (SOTA harness)

Sibling to [coco-val-parity.md](./coco-val-parity.md). The COCO val
parity smokes assert bit-equality on synthetic perfect-DT,
GT-jittered, or user-provided detector JSONs; this page is the
analogous gate on outputs from real Hugging Face SOTA models. Five
cells in one harness — one detector, two segmentation architectures,
one top-down keypoint estimator, and one LVIS-trained long-tail
detector — each pinned to a hub commit SHA so the cache key embeds
the model bytes by construction.

Synthetic / GT-derived fixtures don't exercise the long-tail score
distribution, low-confidence false-positive density, real class
imbalance, per-class mass concentration, per-joint heatmap confidence
surface, or 1203-category federated-evaluation semantics that a
trained SOTA model produces in the wild. These five cells close that
gap on the detection, panoptic, semantic, keypoints, and federated
LVIS paradigms.

> **What "parity" means on this page.** Every cell here gates
> `vernier ↔ <oracle>` agreement on the same `(GT JSON, DT JSON)`
> bytes: pycocotools, panopticapi, mmsegmentation `IoUMetric`,
> `lvis-api`, or vernier's own clean-room numpy oracle (calibration).
> The strict tier asserts bit-equality on the integer / boolean
> surfaces the orchestrator emits (`tp / fp / fn`, `dt_matches`,
> `dt_ignore`, `gt_ignore`, `counts`); the aligned tier asserts
> tolerated drift on the float surfaces (`precision`, summary
> `stats`, per-cell `dt_scores`) at the per-cell `rtol / atol` band
> documented under each section. **A cell that PASSES means: on the
> exact prediction bytes the populator wrote, the orchestrator and
> oracle agree to the documented tier**. A cell that PASSES does NOT
> mean: vernier's headline numbers reproduce the model card's
> published mAP. The "Headline snapshot" rows are cross-reference
> only; under-sampled prefixes (currently: LVIS at 1000/19,809) skew
> absolute mAP without affecting the parity claim, which holds at
> any N.

> **Status:** the three original SOTA parity tests (DETR,
> Mask2Former panoptic, Mask2Former ADE) pass on the live cache
> (machine `84edec51fd71`, 2026-05-31, harness SHA `b9dc053`).
> Bit-equality holds on the integer surfaces actually asserted by
> each test; aligned-tier float tolerances are documented per cell
> below and stay within their gates. Headline numerics in the
> sections below are SNAPSHOTS captured against the pinned model
> revisions listed under "Cells covered"; the tests gate
> vernier ↔ oracle parity, not absolute headline stability across
> dependency upgrades.

## Cells covered

| Paradigm    | Model                                           | Dataset                       | Oracle              | Test                                                                                                            |
| ----------- | ----------------------------------------------- | ----------------------------- | ------------------- | --------------------------------------------------------------------------------------------------------------- |
| Instance    | `facebook/detr-resnet-50`                       | COCO val2017 (bbox)           | `pycocotools 2.0.11`| `tests/python/integration/real_models/sota/test_detr_real_models.py::test_detr_r50_bbox_parity_vs_pycocotools` |
| Panoptic    | `facebook/mask2former-swin-tiny-coco-panoptic`  | COCO panoptic val2017         | `panopticapi`       | `tests/python/integration/real_models/sota/test_mask2former_panoptic_real_models.py::test_mask2former_panoptic_parity_vs_panopticapi` |
| Semantic    | `facebook/mask2former-swin-tiny-ade-semantic`   | ADE20K SceneParse150 val      | `mmsegmentation` `IoUMetric` | `tests/python/integration/real_models/sota/test_mask2former_ade_real_models.py::test_mask2former_ade_parity_vs_mmsegmentation` |
| Calibration | `facebook/detr-resnet-50` (cache reused)        | COCO val2017 (bbox)           | numpy reference (ADR-0018) | `tests/python/integration/real_models/sota/test_detr_calibration_real_models.py::test_detr_r50_calibration_parity_vs_numpy_oracle` |
| Keypoints   | `usyd-community/vitpose-base-simple`            | COCO val2017 (keypoints)      | `pycocotools 2.0.11`| `tests/python/integration/real_models/sota/test_vitpose_real_models.py::test_vitpose_keypoints_parity_vs_pycocotools` |
| Federated   | `facebook/deformable-detr-box-supervised`       | LVIS v1 val (bbox)            | vendored `lvis-api` | `tests/python/integration/real_models/sota/test_lvis_real_models.py::test_lvis_detector_parity_vs_lvis_api` |
| Boundary    | `rfdetr-segnano` (pip-version pinned)           | COCO val2017 (boundary IoU)   | vendored `boundary_iou_api` (bowenc0221) | `tests/python/integration/real_models/sota/test_rfdetr_boundary_real_models.py::test_rfdetr_segnano_boundary_parity_vs_bowenc0221` |

Five Hugging Face cells pinned to revisions captured in
`tools/real_predictions_cache/real_predictions_cache/__init__.py`:

- DETR-R50: `1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b`
- Mask2Former panoptic: `df6b1142ff50c3276559d9d78f35f6a579c75a77`
- Mask2Former ADE: `c8cf1b5e823aee214d937d0d001c1850ba44ef6a`
- ViTPose-base-simple: `a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45`
- Deformable-DETR LVIS: `d7710a91d4e58c2fccd29f53e0ca350093b934f3`

The boundary cell pins rfdetr by pip version
(`RFDETR_VERSION = "1.6.5.post0"`) rather than hub commit — same
ADR-level bump policy applies. The cache filename also embeds
`_RFDETR_CACHE_BLOB_VERSION = "v2"` so a harness-side change that
affects on-disk bytes (e.g. the `pin_inference_threads` landing in
the rfdetr predictor) forces a re-populate rather than silently
serving stale pre-pin bytes.

Bumping any of these is an ADR-level decision; the cache filename
embeds the full SHA (or pip-version + blob-version for the boundary
cell) so a pin bump invalidates by construction.

## Instance — DETR-R50 vs pycocotools (bbox)

- **Workload**: `coco_val2017_detr_r50_v1d5f47b` — 150,680 detections
  across 4,977 of 5,000 val2017 images (23 produce no above-threshold
  output at the 0.05 score floor), 80 categories.
- **Strict tier** — bit-equality on the 12-stat det summary, dense
  `precision` / `recall` / `counts` aggregates. These are the numbers
  a user reads from `Evaluator.summarize()`.
- **Aligned tier** — `eval_imgs.dtScores` and the COCOeval `scores`
  tensor (per-recall-grid score-threshold projection of dtScores) at
  `rtol = 2 * eps`. On the first DETR-R50 val2017 run, ~16% of
  dtScores diverge by exactly 1 ULP of float64 and the `scores` tensor
  inherits the same drift through the recall-grid projection. Root
  cause is documented in the test's module docstring: `serde_json` vs
  Python's `strtod` round near-tie JSON-encoded scores (e.g.
  `0.9992794394493103`) to different adjacent doubles. The divergence
  is parser-level and does **not** propagate past the score-threshold
  projection — precision / recall / mAP are bit-equal because AP
  depends only on detection order. Tolerance expressed as `rtol` (not
  `atol`) so the band tracks score magnitude across `[0.05, 1.0]`.
- **Aligned tier (faster-coco-eval)** — separately checked; fce's
  reductions are bit-identical to pycocotools on this cell, so the
  same strict-tier surface holds.
- **Headline numbers**
  - **mAP**: `0.4168586010785383` — bit-identical across vernier,
    pycocotools, and faster-coco-eval. Reproduces DETR-R50's published
    `box AP = 42.0` on COCO val2017 (paper rounds to one decimal).
  - All 12 stats (AP, AP50, AP75, APs/m/l, AR1/10/100, ARs/m/l)
    bit-equal.

Follow-up parity item: tighten `serde_json`'s f64 parser or normalise
scores at ingest to retire the aligned-tier band. Not a blocker for
shipping the real-prediction gate.

## Panoptic — Mask2Former Swin-T vs panopticapi (PQ)

- **Workload**: `coco_panoptic_val2017_mask2former_swin_t_v<sha>` —
  5,000 images, 133 categories (80 things + 53 stuff). Predictions
  land as rgb2id-encoded PNGs + per-image JSON sidecars + a single
  aggregated `panoptic_dt.json`; per-image writes are atomic so a
  SIGINT mid-run resumes cheaply.
- **Coverage gate** — the test asserts every GT image is present in
  the populator's aggregated DT. The aggregate JSON is only written
  at the end of a full run (the per-image sidecars cover resume), so
  its presence implies coverage; the assertion makes that contract
  explicit.
- **Strict tier** — per-category integer surface: `PQStatCat.tp / fp / fn`
  (panopticapi) vs `n_tp / n_fp / n_fn` (vernier `EvalResult.per_class`)
  are bit-equal across all 133 categories. Float reduction order
  cannot shift integers, so drift here is a real accumulator bug.
- **Aligned tier** — float reductions at `rtol = atol = 8 * eps`.
  Covers per-category `iou_sum` + per-class PQ/SQ/RQ rows + global /
  Things / Stuff bucket means. Both bounds are passed so a category
  whose oracle metric collapses to exact `0.0` while vernier yields a
  sub-ULP non-zero still passes (an `rtol`-only gate would fail at
  `rtol * 0 = 0`). Observed drift on the live cache:
  - Bucket level (Things SQ): max abs diff `1.11e-16` (= 0.5 ULP).
  - Per-class level (cat 3 PQ): max abs diff `5.55e-16` (= 2.5 ULP) —
    the worst entry across all 5000 images × 133 classes; 132 of 133
    rows are bit-equal.
  - 8 ULP keeps any genuine kernel divergence (e.g. a wrong IoU
    numerator) well above the gate.
- **Headline snapshot** (captured on the live cache at SHA `b9dc053`,
  machine `84edec51fd71`, 2026-05-31):
  - Global PQ: `0.462607`, SQ: `0.815501`, RQ: `0.559097`
  - Things (80 classes): PQ `0.496540`, SQ `0.819126`, RQ `0.598265`
  - Stuff (53 classes): PQ `0.411386`, SQ `0.810030`, RQ `0.499976`
  - Coverage: 5000 / 5000 val2017 images, 133 / 133 categories — the
    test asserts this explicitly. The metric values are recorded
    here for cross-reference only; the parity test gates
    vernier ↔ oracle equivalence, not absolute-metric stability under
    transformers / torch dependency upgrades.

## Semantic — Mask2Former Swin-T vs mmsegmentation `IoUMetric` (mIoU)

- **Workload**: `ade20k_val_mask2former_swin_t_v<sha>` — 2,000 images
  at ~512×512, single-channel label-map PNGs (train-id `0..149` +
  `255` ignore).
- **Strict tier** — bit-equality on the per-class u64 confusion-matrix
  totals: `intersect`, `union`, `pred`, `label`. Both
  mmsegmentation's `IoUMetric` and vernier-semantic produce identical
  totals from the same per-pixel (gt, dt) arrays. Derived float
  scalars (mIoU, aAcc, per-class IoU/Acc) follow trivially from the
  same u64 inputs — no aligned tier needed because there's no
  reduction-order ambiguity once the u64 surface matches.
- **Headline snapshot** (captured on the live cache at SHA `b9dc053`,
  machine `84edec51fd71`, 2026-05-31; the test asserts u64 array
  equality, not absolute headline values — those are recorded here
  for cross-reference, not pinned):
  - mIoU (mean over 150 classes): `0.462490`
  - aAcc (overall pixel accuracy): `0.819045`
  - Σ intersect: `367,015,797` pixels
  - Σ union: `529,188,289` pixels
  - All 150 classes present in both `label` and `pred`.

<<<<<<< HEAD
## Calibration — DETR-R50 vs numpy reference (ECE/MCE)

- **Workload**: same `coco_val2017_detr_r50_v1d5f47b` cache as the
  Instance cell — no new model, no new oracle, no new prediction
  cache. Cells lifted from `evaluate_bbox_grid(...).eval_imgs()` at
  the `a=0` (`all` area) bucket, the only bucket the ADR-0018
  calibration kernel consults. Both implementations receive the
  **identical** `dict[k -> list[PerImageCell]]` so the test isolates
  calibration-kernel parity from matching-kernel parity (the latter
  is the Instance cell's claim).
- **Oracle**: vernier's clean-room numpy implementation at
  `tests/python/parity_calibration/numpy_oracle.py`. Calibration
  metrics (ECE/MCE/Wilson) are vernier-specific per ADR-0018; no
  third-party calibration library is vendored as oracle.
- **Bin config**: ADR-0018 "DETR-aware defaults" — 15 quantile bins,
  `min_score=0.05`, Wilson 95% CIs. Same config the user-facing
  `result.calibration()` exposes.
- **Strict tier** — per-bin u64 `count` from the reliability table is
  bit-equal at every IoU threshold tested. Integer reductions cannot
  drift under reduction-order changes; a diff here would be an
  outright accumulator bug.
- **Aligned tier, 8 ULP rtol + atol** — `ece` / `mce` scalars and
  every float column of the reliability table (`mean_score`,
  `accuracy`, `gap`, `ci_lo`, `ci_hi`). Both bounds are passed so a
  bin whose oracle gap rounds to exact `0.0` while vernier yields a
  sub-ULP non-zero still passes (same symmetric-band rationale as the
  panoptic cell above). Synthetic fixtures pin at `4 * eps`; the
  real-prediction reduction over ~150k detections justifies widening
  to 8 ULP — same band the panoptic SOTA gate uses.
- **IoU coverage** — the test parametrises over `iou ∈ {0.5, 0.75,
  0.95}` against the canonical COCO 10-point ladder. ADR-0018's
  `calibrate(...)` surface picks one T-slot at a time, so the
  "0.5:0.95" mean-over-thresholds shape familiar from AP does not
  apply at the calibration kernel; the two endpoints + 0.75 give
  coverage of both the permissive and the strict-IoU regimes
  (`dt_matched` density differs by an order of magnitude across the
  span).
- **Headline numbers** (captured on the live cache at SHA
  `<populated by first real run>`):
  - **ECE @ iou=0.5**: `<populated by first real run>`
  - **MCE @ iou=0.5**: `<populated by first real run>`
  - **ECE @ iou=0.75**: `<populated by first real run>`
  - **MCE @ iou=0.75**: `<populated by first real run>`
  - **ECE @ iou=0.95**: `<populated by first real run>`
  - **MCE @ iou=0.95**: `<populated by first real run>`
  - n_detections / effective_n_bins per IoU:
    `<populated by first real run>`

This cell shares the inference cost with the Instance cell: both pull
the same `detr_predictions_path` session fixture, so adding the
calibration smoke is free on a populated cache.

## Keypoints — ViTPose-base vs pycocotools (OKS)

- **Workload**: `coco_val2017_vitpose_base_simple_va93ac0c` — one
  COCO keypoints result per GT person annotation in val2017
  (~11k crops across the 5,000-image val set), single category
  (`person`, COCO category_id=1), 17-joint COCO topology.
- **Inference shape** — ViTPose is *top-down*: one forward pass
  per person box. We feed GT person boxes from
  `person_keypoints_val2017.json` (not detector output) — this
  isolates the keypoint head's numerics from any detector quirks
  and mirrors the canonical mmpose / MMCV ViTPose eval
  configuration the published numbers use. The cache content is
  fully determined by the weights pin + the (SHA-pinned) GT JSON.
- **Strict tier** — bit-equality on the 10-stat OKS summary
  (re-indexed A-axis, no `_S` row per ADR-0012 quirk D5) +
  dense `precision` / `recall` / `counts` aggregates. These are
  the numbers a user reads from `Evaluator.summarize()`; they must
  match pycocotools' `iouType="keypoints"` exactly. The 10 stats
  are `AP / AP50 / AP75 / APm / APl / AR / AR50 / AR75 / ARm / ARl`.
- **Aligned tier** — `eval_imgs.dtScores` and the COCOeval `scores`
  tensor at `rtol = 2 * eps`. Same root cause as the DETR cell:
  `serde_json` vs Python's `strtod`-based parser round near-tie
  decimal scores to different adjacent doubles. The divergence is
  parser-level and does NOT propagate past the score-threshold
  projection — precision / recall / AP are bit-equal because
  ranking-based OKS depends only on detection order.
- **DT-side visibility** — the predictor projects per-joint heatmap
  scores to `v=2` above `0.3` and `v=1` below; pycocotools' OKS eval
  ignores DT-side `v` entirely (it uses only GT-side `v` for the
  labeled-joint mask), so the projection has no parity implication.
  Quirk F5 (per ADR-0012) lives on the GT side and is exercised by
  the synthetic keypoints parity fixtures, not by this real-prediction
  cell.
- **Headline snapshot** (captured on the live cache at SHA
  `0ba2b3c`, machine `84edec51fd71`, 2026-06-07):
  - 10-stat OKS summary:
    - **AP @ [.50:.95]**: `0.7626` — reproduces ViTPose-base's
      published `AP = 76.0` on COCO val2017 (top-down on GT person
      boxes, the canonical mmpose eval shape).
    - AP @ .50: `0.9314` &nbsp;&nbsp; AP @ .75: `0.8390`
    - AP @ medium: `0.7413` &nbsp;&nbsp; AP @ large: `0.7970`
    - AR @ [.50:.95]: `0.7995`
    - AR @ .50: `0.9449` &nbsp;&nbsp; AR @ .75: `0.8640`
    - AR @ medium: `0.7683` &nbsp;&nbsp; AR @ large: `0.8478`
  - Coverage: 10,777 DT records (one per GT person annotation with
    a usable box) across 2,693 images. The metric values are
    recorded here for cross-reference only; the parity test gates
    vernier ↔ pycocotools equivalence, not absolute-metric
    stability under transformers / torch upgrades.

## Federated — Deformable-DETR LVIS vs `lvis-api` (long-tail AP)

- **Workload**: `lvis_v1_val_deformable_detr_vd7710a9` — Deformable-
  DETR's 300-query head emits per-image detections over the LVIS v1
  category set (1,203 classes, 19,809 val images, reusing COCO 2017
  image bytes). The cache is sub-sampled at test time (default 1,000
  images, env-var override) so the test runs on a single 8–16 GB box
  in seconds while still exercising the federated-evaluation
  semantics on real-model outputs.
- **Strict tier** — per-category integer surface: `tp / fp / fn`
  bit-equal across all sampled categories vs the vendored
  `lvis-api` oracle. Federated cell-skip (AA4 — `eval_imgs[c, a, i]
  = None` when category `c` is not in image `i`'s federated set)
  and not-exhaustive `dt_ignore` propagation (AA3 — DT in
  `not_exhaustive` images flagged ignored not FP) are both
  exercised on real model outputs for the first time; the synthetic
  `federated_min` fixture covers only the topology.
- **Aligned tier** — `AP / AP50 / AP75 / APr / APc / APf`
  frequency-bucketed values at `rtol = atol = 8 * eps`. Bucket
  means over 1,203 categories have ~36× more reduction-order
  opportunities than the 133-class panoptic cell, so this band is
  expected to be wider than panoptic's observed 2.5 ULP; the
  rtol+atol gate is documented here and tightened post-first-run.
- **Coverage gate** — the test asserts every GT image is present
  in the populator's aggregated DT (asserted on pre-subsample
  bytes; the subsample step is deterministic on the env-var seed
  so the gate fires only on a genuinely incomplete populator run).
- **Headline snapshot** (captured on the live cache at SHA `4dfd374`
  rebased onto `9644f4e`, machine `84edec51fd71`, 2026-06-09,
  `VERNIER_LVIS_REAL_VAL_SAMPLE_IMAGES=1000` — the in-session
  default-prefix shape):
  - **AP @ [.50:.95]**: `0.3926` (full 19,809-image val publishes
    `box mAP = 31.7` per the model card; the 1000-image prefix is
    not a frequency-balanced sample of the full val, so this number
    is not directly comparable to the published headline — the
    parity gate is vernier ↔ `lvis-api`, not absolute mAP).
  - AP @ .50: `0.5212` &nbsp;&nbsp; AP @ .75: `0.4250`
  - AP_r (rare, ≤10 imgs): `0.2806`
    &nbsp;&nbsp; AP_c (common): `0.4000`
    &nbsp;&nbsp; AP_f (frequent): `0.3956`
  - AP_s / AP_m / AP_l: `0.2731` / `0.4822` / `0.5570`
  - AR @ 300: `0.4449`
    &nbsp;&nbsp; AR_s / AR_m / AR_l @ 300: `0.3061` / `0.5374` / `0.6372`
  - Coverage: 1000 / 19809 GT-image prefix (the populator was run
    with the same sub-sample knob; coverage-gate's partial-cache
    branch validated `dt ⊆ gt-prefix` with zero in-prefix skips).
    Full-corpus populate (~48–72 h on 8-core CPU) is the path to
    published-comparable numbers.

## Boundary — rfdetr-segnano vs `boundary_iou_api`

- **Workload**: `coco_val2017_rfdetr_segnano_v1.6.5.post0-v2` —
  reuses the rfdetr-segnano RLE-mask cache the TIDE harness already
  populates (no new inference is run here; boundary IoU is just a
  different metric over the same masks). 5,000 images, 80
  categories. rfdetr-segnano is a smoke-grade model — both vernier
  and bowenc0221 oracle agree bit-exact on the low headline numbers
  here; the parity claim is the deliverable, not the model quality.
- **Pip-version pinning analog**: rfdetr ships as a pip package
  rather than a Hugging Face hub model, so ``RFDETR_VERSION``
  (the pinned pip version, currently ``1.6.5.post0``) plays the
  same role as a hub commit SHA on the other cells. A pip-side
  bump invalidates the cache by construction. The cache filename
  also embeds ``_RFDETR_CACHE_BLOB_VERSION = "v2"`` so a harness-
  side change that affects on-disk bytes (currently:
  ``pin_inference_threads`` landing in the rfdetr predictor) forces
  a re-populate instead of silently serving stale pre-pin bytes.
- **Strict tier** — bit-equality on the dense ``precision`` (T × R × K
  × A × M = 10 × 101 × 80 × 4 × 3) + ``recall`` + ``counts`` aggregates
  + the 12-stat AP/AR boundary summary (`AP, AP50, AP75, APs/m/l,
  AR1/10/100, ARs/m/l`). Identical to the DETR-R50 (bbox) instance
  cell's contract — once a parity test holds the precision tensor
  bit-equal, the 12 derived stats follow by construction. Integer-
  reduction surfaces; reduction order cannot move them.
- **Aligned tier** — per-cell ``scores`` array at ``rtol = 2 * eps``,
  same documented ``serde_json`` vs Python ``strtod`` 1-ULP parser
  drift the DETR cell ships against. Boundary uses the same mask
  kernels so the same parser-drift band applies. Ranking-based AP
  is unaffected.
- **Headline snapshot** (captured on the live cache at SHA
  `e5abafe`, machine `84edec51fd71`, 2026-06-07):
  - boundary AP@[.5:.95]: `0.0001` &nbsp; AP@.50: `0.0002`
    &nbsp; AP@.75: `0.0001`
  - AP small / medium / large: `0.0005` / `0.0001` / `0.0001`
  - AR@1 / @10 / @100: `0.0040` / `0.0073` / `0.0087`
  - AR_s / AR_m / AR_l @100: `0.0201` / `0.0132` / `0.0082`
  - Coverage: 5000 / 5000 val2017 images, all 80 categories. The
    metric values are recorded here for cross-reference only — the
    parity test gates vernier ↔ bowenc0221 equivalence, not
    absolute-metric stability under transformers / torch upgrades.
    Numbers are low by design (rfdetr-segnano is the lightest
    rfdetr variant — minor model, real parity claim).

## Shared harness configuration

- **Streaming evaluation** — semantic streams (gt, dt) pairs through
  `parity_semantic.harness.run_streaming_pair`; peak RAM is one
  decoded label-map per side. Instance and panoptic load the full
  prediction JSON / PNG set (the COCO val2017 surface is bounded).
- **Inference determinism** — the SOTA conftest and
  `_harness_common` both set `OMP_NUM_THREADS=1` /
  `MKL_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` at import time
  (the env-var pin is the only path that reliably wins once torch's
  intra-op pool is live — a common scenario in a pytest process that
  also runs the rfdetr `tide/` cells); `pin_inference_threads`
  additionally calls `torch.set_num_threads(1)` as defence-in-depth.
  Every predictor uses `int64` `target_sizes` and loud-fails on
  unmapped class names. The cache filename embeds the pinned hub
  commit SHA so a weights bump invalidates by construction. See
  `tests/python/integration/real_models/sota/_harness_common.py` for
  the full discipline list.
- **Sibling-tree oracle resolution** — the SOTA conftest replicates
  two oracle installs that the `parity_semantic` / `parity_panoptic`
  conftests only fire when collecting under those trees:
  `mmsegmentation` is installed as a stub set via `_install_stubs()`
  (vendored at `parity_semantic/oracle/mmsegmentation/`);
  `panopticapi` is a real vendored package and only its parent
  directory is inserted onto `sys.path` (vendored at
  `parity_panoptic/oracle/panopticapi/`). Both installs are guarded
  so a vendored-oracle path rename fails a per-cell fixture skip
  instead of breaking SOTA collection wholesale.
- **Skip semantics** — each cell skips cleanly when:
  - the `real-models` extra is absent (`uv sync --extra real-models`),
  - the model revision is the `_UNPINNED_REVISION` sentinel,
  - the dataset cache (panoptic / ADE20K / COCO val2017 images +
    keypoints GT) is not provisioned.

## Reproducing

The parity tests pass on a populated cache in seconds; inference
itself is the cost driver (one-time per host per pinned SHA):

| Cell                  | First-run inference cost | Cache-hit cost |
| --------------------- | ------------------------ | -------------- |
| DETR-R50              | ~12-15 h (8-core CPU)    | seconds        |
| Mask2Former panoptic  | ~20-25 h (8-core CPU)    | seconds        |
| Mask2Former ADE       | ~3-4 h (8-core CPU)      | seconds        |
| ViTPose-base-simple   | ~2-3 h (8-core CPU)      | seconds        |

```bash
# One-time: populate prediction caches (see the per-cell bench docs
# for the full preflight, including SHA pinning and dataset caches).
./tools/fetch-real-predictions.sh --detr
./tools/fetch-real-predictions.sh --mask2former-panoptic
./tools/fetch-real-predictions.sh --mask2former-ade
./tools/fetch-real-predictions.sh --vitpose

# Run the SOTA parity tests.
VERNIER_COCO_CACHE=/path/to/coco-val2017 \
  uv run pytest tests/python/integration/real_models/sota/ -v
```

Per-cell benchmarking context (medians, IQR, RSS, per-stage
breakdowns) lives alongside the parity numbers in:

- [benchmarking/2026-05-detr-r50-real-predictions.md](./benchmarking/2026-05-detr-r50-real-predictions.md)
- [benchmarking/2026-05-mask2former-real-predictions.md](./benchmarking/2026-05-mask2former-real-predictions.md)

## What this gate doesn't catch

- **Other hub revisions of the same model.** The cache pins one SHA
  per cell; predictions from a different snapshot live behind their
  own cache key and would need their own pin bump (ADR-level).
- **Other datasets.** COCO val2017 and ADE20K val are the closed
  surface here; new datasets need their own cache module (cf.
  `panoptic_val_cache`, `ade20k_val_cache`).
- **`corrected` parity disposition divergences.** Vernier's
  `corrected` modes (ADR-0002) intentionally diverge from the
  oracles and are exercised by the per-quirk fixtures, not by these
  smokes. The Mask2Former-panoptic test uses `parity_mode="corrected"`
  for vernier's side because the oracles' bugs are documented under
  panoptic quirks; the strict-tier integer surface still matches by
  construction.
- **Other paradigms.** Boundary doesn't yet have a SOTA cell; the
  COCO val parity smoke is the headline gate for that paradigm. See
  [coco-val-parity.md](./coco-val-parity.md).
- **Oracle internal drift.** Each oracle is pinned to its current
  vendored snapshot (`pycocotools==2.0.11` in `pyproject.toml`;
  `panopticapi` / `mmsegmentation` checked out under the
  `parity_panoptic/oracle/` and `parity_semantic/oracle/` trees).
  Bumping follows the same ADR-level discipline as the pycocotools
  pin.
