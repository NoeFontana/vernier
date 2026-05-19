# How vernier compares

vernier sits in a small ecosystem of COCO-style evaluation libraries. This
page is a decision aid — when does vernier fit your case, when would you
still pick a specific alternative, and what does each alternative actually
provide. For mechanical "rewrite my imports" instructions, see the
[migration guides](migrate/README.md).

## At a glance

| Library | Paradigms | Parity contract | Performance vs vernier | When you'd still pick it |
|---|---|---|---|---|
| `pycocotools` | instance (bbox / segm / keypoints) | The reference | ~7–17× slower | You need the literal pycocotools printed table for an external system that scrapes it |
| `faster-coco-eval` | instance (bbox / segm / keypoints / boundary) | "Faster, mostly compatible" — quirks chosen silently | ~3.7–12× slower | You're already running it in production and don't need vernier's auditable parity surface |
| `panopticapi` | panoptic | The reference | ~3.3× slower on val2017 perfect-DT | You explicitly need the `pq_compute_*` script outputs unchanged |
| `lvis-api` | LVIS federated | The reference | ~57× slower on full v1 val · 10× lower peak RSS on vernier (1.5 GiB vs 15 GiB) | Your tooling depends on the `LVISEval` instance attributes |
| `boundary-iou-api` | boundary IoU only | The reference | ~19× slower on val2017 perfect-DT | You're running an external evaluation script that loads `boundary_iou.coco_instance_api.COCOeval` by name |
| `mmsegmentation` | semantic | One of three references vernier targets | ~7× slower on val2017 perfect-DT (`mmseg.IoUMetric` only — vernier-semantic vs vendored oracle, ADR-0036) | You need the full `mmseg.evaluation` registry surface (it's a training framework, not just an evaluator) |

Numbers above reference the [benchmarks page](benchmarks.md) and the
[engineering benchmarking notes](https://github.com/NoeFontana/vernier/tree/main/docs/engineering/benchmarking).

## `pycocotools`

[pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools)
is the reference COCO evaluation library — the source of truth that every
other library (including vernier) calibrates against. It ships `COCOeval`
for bbox / segm / keypoints AP, the `Mask` module for RLE codec and polygon
rasterization, and the `COCO` GT loader.

vernier reproduces `pycocotools==2.0.11` semantics bit-for-bit in strict
parity mode (the default). Every quirk — float casting in IoU computation,
the `setDetParams()` defaults, the `<` vs `<=` comparison in score-tied
matches — is filed in
[ADR-0002](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0002-three-tier-parity-model.md)
as either `strict` (bit-equal output) or `corrected` (opt-in opinionated
fix). The full table lives in
[`docs/engineering/pycocotools-quirks.md`](https://github.com/NoeFontana/vernier/blob/main/docs/engineering/pycocotools-quirks.md).

vernier is ~7–17× faster on val2017 across bbox / segm / keypoints (see the
[benchmarks](benchmarks.md)). The drop-in shim
([ADR-0007](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0007-patch-pycocotools-policy.md))
keeps `from pycocotools.cocoeval import COCOeval` working in existing
scripts via `vernier.patch_pycocotools()`.

**Pick `pycocotools` instead** when an external system parses
`COCOeval.summarize()`'s text output and you need that exact text. (Even
then, vernier's strict-mode `--emit text` matches it byte-for-byte —
[ADR-0015](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0015-vernier-cli.md)
pins the behavior — but if your dependency reaches into private attributes
of `COCOeval`, the shim is the safer path.)

## `faster-coco-eval`

[faster-coco-eval](https://github.com/MiXaiLL76/faster_coco_eval) is the
most prominent fast reimplementation of pycocotools' `COCOeval`. It exposes
a `COCOeval_faster` class with the same API and an `init_as_pycocotools()`
helper that monkey-patches `pycocotools.cocoeval.COCOeval`.

The contract is "faster, mostly compatible." Each pycocotools quirk gets
fixed or kept based on the maintainer's judgment, and the project doesn't
publish a quirks table. In practice this means a faster-coco-eval run will
sometimes diverge from the reference and you have to read source to know
why and where.

vernier targets the same drop-in pattern but with auditable parity. Every
quirk has a row in the disposition table; strict mode reproduces
pycocotools bit-for-bit; corrected fixes are listed and opt-in. The
performance gap is real on val2017 — vernier is ~3.7–12× faster on
bbox / segm / keypoints / boundary — but the headline benefit is "you can prove what
your numbers mean".

**Pick `faster-coco-eval` instead** when you have an existing CI pipeline
running it stably and the auditable-parity property doesn't justify a
migration. The migration cost is small (one-line shim) but real.

## `panopticapi`

[panopticapi](https://github.com/cocodataset/panopticapi) is the reference
panoptic-quality (PQ) evaluator. It ships `pq_compute_single_core` and
`pq_compute_multi_core` over the COCO panoptic GT format (RGB-encoded PNGs
with `id = R + G*256 + B*256²`).

[vernier-panoptic](https://github.com/NoeFontana/vernier/tree/main/crates/vernier-panoptic)
is a sibling crate to vernier-core; both depend on vernier-mask, neither
depends on the other. PQ and AP have different matching rules and different
data models — the architectural firewall keeps the two folds from drifting
toward each other (see
[ADR-0025](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0025-panoptic-api.md)).
Strict-mode parity is bit-equal at per-class TP/FP/FN counts.

vernier-panoptic is ~3.3× faster than panopticapi on val2017 perfect-DT
(post-#260 sparse-remap cache + SSSE3 RGB→u32 pack — see the
[panoptic cell of the benchmarks](benchmarks.md)). The gap is still
not the headline — panopticapi is already efficient and PNG decode
dominates both sides — the headline is the unified surface:
`vernier.panoptic.Evaluator` lives next to `vernier.instance.Evaluator`
and `vernier.semantic.Evaluator`, and the CLI covers all three.

**Pick `panopticapi` instead** when an external system parses the
`pq_compute_*` script's stdout output and you need that exact text. (As
above, vernier reproduces it in strict mode.)

## `lvis-api`

[lvis-api](https://github.com/lvis-dataset/lvis-api) extends pycocotools to
the LVIS dataset, with federated evaluation: the `not_exhaustive` and
`neg_category_ids` fields per image scope which categories actually
contribute to the AP fold. The library ships `LVISEval` and friends.

vernier's LVIS support
([ADR-0026](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0026-lvis-support.md))
implements federated evaluation in vernier-core's existing AP fold — no
fork of `matching.rs` or `accumulate.rs`, just a different category set per
image. The semantics match `lvis-api`'s outputs.

**Pick `lvis-api` instead** when your code depends on the `LVISEval`
instance's attribute layout. There's no equivalent bit-for-bit text
output to scrape (LVIS doesn't print a fixed-format summary), so the
choice is mostly about migration cost.

The original ADR-0026 §"Known follow-up" called out a >22 GB structural
peak from the dense `Vec<Option<PerImageEval>>` orchestrator grid (95M
slots × 232 B). PR #179 collapsed the slot type via the `Box`-niche
trick; the structural floor is now under 1 GB on full LVIS val. The
small per-cell precision-tensor drift that briefly tracked here as a
follow-up was root-caused to the oracle's `area > 0` GT filter (quirk
AG6) and is now mirrored in strict mode; the full-val bbox cell passes
bit-equal.

## `boundary-iou-api`

[boundary-iou-api](https://github.com/bowenc0221/boundary-iou-api) is the
reference for boundary IoU — the metric that ranks segmentation by how
well a prediction's *boundary* aligns with the ground truth, downweighting
the well-aligned interior. It ships a `COCOeval` subclass that swaps the
IoU kernel; everything else inherits from pycocotools.

vernier-core implements boundary IoU as an isolated subsystem
([ADR-0010](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0010-boundary-iou-isolated-subsystem.md))
with its own oracle and quirks file. The dilation ratio default (0.02)
matches `bowenc0221`'s reference value.

vernier is significantly faster on boundary IoU — the round 2026-05 perf
push (PRs #181/#182/#184/#185/#186) brought val2017 perfect-DT from
~21 s to ~3 s, post-bbox-cropped erode and bbox-cropped XOR scan. See the
[boundary cell of the benchmarks](benchmarks.md) for the current
margins.

**Pick `boundary-iou-api` instead** when an evaluation script imports
`boundary_iou.coco_instance_api.COCOeval` by name and you can't or
don't want to redirect the import.

## `mmsegmentation`

[mmsegmentation](https://github.com/open-mmlab/mmsegmentation) is a full
training framework for semantic segmentation, not just an evaluator. Its
`mmseg.evaluation.IoUMetric` is one of three references vernier-semantic
calibrates against (the other two are
[`mcordts/cityscapesScripts`](https://github.com/mcordts/cityscapesScripts)
and the Pascal VOC / ADE20K reference scripts).

[vernier-semantic](https://github.com/NoeFontana/vernier/tree/main/crates/vernier-semantic)
ships per-class IoU, mIoU, FWIoU, pAcc, and mAcc. The vendored oracle
harness lands under
[ADR-0036](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0036-vendor-mmsegmentation-ioumetric.md)
(still `proposed`); per-paradigm parity status lives in
[README §Status & validation](https://github.com/NoeFontana/vernier/#status--validation).

**Pick `mmsegmentation` instead** when you need the broader training
framework: model registry, training loop, dataset pipelines, config
system. vernier-semantic is the evaluation half only; the training half
is out of scope.

## Metrics beyond AP

The alternatives above all measure the same metric family — COCO-style
average precision (plus PQ / mIoU on the panoptic / semantic side).
vernier additionally ships two diagnostic surfaces that none of the
above expose, alongside the same matching pass:

- **LRP / oLRP error decomposition** (Oksuz et al., ECCV 2018 / TPAMI
  2021). `vernier.instance.optimal_lrp(...)` and
  `vernier eval --metric olrp` return `oLRP_Loc + oLRP_FP + oLRP_FN`
  plus a per-class operating-point `tau`. The first-party reference
  is [`kemaloksuz/LRP-Error`](https://github.com/kemaloksuz/LRP-Error)
  — vernier vendors it commit-pinned as a CI tripwire, not as a
  parity oracle ([ADR-0043](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0043-lrp-oracle-and-namespace.md)).
  The correctness contract is a clean-room NumPy oracle in
  `tests/python/oracle/lrp/`. Among the alternatives in the table,
  none expose an oLRP surface. See
  [LRP and its limits](explanation/lrp-and-its-limits.md).
- **Detection-family calibration** (ECE / MCE / reliability table).
  `Evaluator.evaluate(..., calibration=True)` plus the lazy
  `result.calibration(...)` fold over the retained per-image cell
  store ([ADR-0018](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0018-calibration.md)).
  The obvious comparison is *not* a COCO library but the general-
  purpose calibration toolkits — most prominently
  [`torchmetrics`](https://lightning.ai/docs/torchmetrics/) and
  [`netcal`](https://github.com/EFS-OpenSource/calibration-framework).
  Both compute ECE on a score / label pair, but neither knows what
  a COCO match is — the user has to thread the AP fold's per-pair
  TP/FP labels into the calibrator manually. vernier folds against
  the same ADR-0013 cell store the AP fold writes, so the matching
  pass is shared and the score / label pairs are bit-identical to
  the AP path. Calibration is detection-family today; panoptic and
  semantic are deferred (see
  [Calibration and its limits](explanation/calibration-and-its-limits.md)).

These surfaces are stacked on the matching engine, not bolted on:
the same `Evaluator.evaluate(...)` call returns AP-side metrics,
oLRP, and calibration over one pass. The alternative is N libraries
in N matching passes with N divergent definitions of "true positive."

## Scenario slicing and cross-run aggregation

The alternatives above evaluate one dataset against one detection set.
vernier additionally accepts a *partition manifest* — a tiny CSV/JSON
table mapping image ids to scenario axes (`weather`, `time_of_day`,
…) — and emits one row of headline metrics per `(axis, value)` cell
alongside the overall summary, in one matching pass
([ADR-0046](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0046-slice-and-aggregate.md)).
`vernier aggregate` then fans N corruption runs into a single
comparative table with mPC / rPC columns matching the Michaelis et
al. (NeurIPS-W 2019) corruption-benchmark convention. None of the
alternatives in the table above ship this lane; the typical pattern
elsewhere is N invocations + glue code per row. See
[Slice an evaluation by scenario](how-to/scenario-slicing.md).

## What vernier doesn't do (yet)

A short, honest list:

- **Visualization tooling.** vernier produces numbers and tables; it does
  not draw bounding boxes on images. Tools like
  [supervision](https://github.com/roboflow/supervision) and
  [fiftyone](https://github.com/voxel51/fiftyone) cover that ground.
- **Training-loop integration beyond two supported entry points.**
  `Evaluator.evaluate()` at end-of-epoch is the default; the
  `BackgroundEvaluator` surface
  ([ADR-0014](https://github.com/NoeFontana/vernier/blob/main/docs/adr/0014-background-evaluator.md))
  is the secondary one for in-loop submission without GIL stalls.
  Multi-rank rank-local + gather lives at
  [distributed-eval how-to](how-to/distributed-eval.md). Full
  callbacks-and-loggers integration is downstream-framework territory.
- **Pretty HTML reports.** The CLI emits text and JSON; HTML report
  generation is a follow-up tool that consumes the JSON output.
- **A model zoo / pretrained predictor.** vernier evaluates predictions
  you already have. Generating predictions is a different product
  (`mmsegmentation`, `mmdetection`, `detectron2`).
