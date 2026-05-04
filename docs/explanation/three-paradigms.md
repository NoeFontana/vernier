# Three paradigms: instance, panoptic, semantic

vernier ships three evaluation paradigms as **sibling classes**, not
as configurations of a single evaluator (per ADR-0011, ADR-0025,
ADR-0028, ADR-0029):

```python
import vernier

vernier.instance.Evaluator()    # bbox / segm / boundary / keypoints (AP fold)
vernier.panoptic.Evaluator()    # panoptic-quality (PQ)
vernier.semantic.Evaluator()    # semantic-segmentation (mIoU)
```

The submodules are mutually exclusive: a model emits *one* of these
output shapes, and the user evaluates against the matching paradigm.
This page explains why they are sibling submodules rather than knobs
on a single class, and helps you pick the right one for a model
output.

## What each paradigm consumes

| Paradigm | GT shape | DT shape | What's evaluated |
|---|---|---|---|
| **instance** | COCO JSON: `images` + `annotations` (bbox / RLE / polygons / keypoints) + `categories` | COCO JSON: list of detections with `image_id`, `category_id`, `score`, geometry | Per-detection matching at IoU thresholds, score-ranked, AP curves over T (10 IoU thresholds) × R (101 recalls) × A (4 area buckets) × M (3 max-dets) |
| **panoptic** | RGB-encoded segment-id PNG + `segments_info` JSON (per-segment `category_id`, `iscrowd`, `area`) | Same shape as GT | Per-segment one-to-one matching at `IoU > 0.5` (no score gradient); PQ = SQ × RQ over per-class buckets, with things/stuff splits |
| **semantic** | Single-channel `(H, W)` class-id label map per image (uint8 / uint16 / uint32) | Same shape as GT | Per-pixel class assignment; mIoU / FWIoU / pixel accuracy / mean accuracy from a `(n_classes, n_classes)` confusion matrix |

## What the kernels actually do

The three paradigms compute fundamentally different things:

- **Instance (AP fold)**: a *score-ranked greedy assignment* under
  per-class IoU thresholds. A high-confidence detection that matches
  a GT at IoU > 0.5 contributes a true positive at AP@0.5; the same
  detection contributes nothing at AP@0.95 unless it also crosses
  the higher threshold. The PR curve sweeps the score gradient.
- **Panoptic (PQ)**: a *one-to-one matching* at the single threshold
  `IoU > 0.5` (no score gradient). Every GT segment matches at most
  one DT segment and vice versa; SQ averages the matched IoU; RQ is
  `TP / (TP + 0.5·FP + 0.5·FN)`. PQ = SQ × RQ. Score is discarded.
- **Semantic (mIoU)**: a *per-pixel histogram*. There is no matching
  loop, no per-detection scoring, no PR curve. Each pixel
  contributes one cell increment to the `(gt_class, pred_class)`
  confusion matrix; per-class IoU = `TP_c / (TP_c + FP_c + FN_c)`
  derived at finalize time.

The data models, the matching rules, the parity oracles, and the
quirk surveys all differ. ADR-0005 locks the AP-fold matching engine
behind a structural firewall (the `vernier-core::matching.rs`
module); the panoptic and semantic crates **cannot** edit it.

## When to use which

> A model emits one of these output shapes. The shape determines the
> paradigm.

- **Bounding boxes / instance masks / keypoints with scores** →
  `vernier.instance`. Every COCO-style detection benchmark
  (`instances_val2017.json`, LVIS, etc.).
- **Panoptic PNGs (RGB segment ids) + segments_info JSON** →
  `vernier.panoptic`. COCO panoptic, Cityscapes panoptic, anything
  with the things-and-stuff partitioning.
- **Single-channel class-id label maps (or stacked per-class binary
  masks)** → `vernier.semantic`. Cityscapes 19-class semantic,
  ADE20K SceneParse150, Pascal VOC, free-space / drivable-surface /
  lane-segmentation heads in AV stacks, BEV semantics.

If your model emits *bounding boxes alongside a semantic mask*
(typical for AV stacks): evaluate the boxes via
`vernier.instance.Evaluator(iou=Bbox())` and the masks via
`vernier.semantic.Evaluator()` separately. They report different
numbers; mix them at your peril.

## Why mIoU diverges from PQ on stuff classes

A common question: "Panoptic *can* compute IoU on stuff-only data —
why have a separate semantic surface?"

ADR-0028 §"Context and problem statement" answers: SQ averages IoU
**only over matched segments** (panoptic survey U7: matching threshold
`IoU > 0.5`), whereas mIoU pools intersections and unions across all
images per class. A class your model gets at IoU=0.3 contributes zero
to SQ (no match) and the actual value 0.3 to mIoU (the union still
counts both pixels). The metric difference compounds with the data-
model mismatch (RGB-encoded segment-id PNGs vs. single-channel class-
id label maps) and the dataset conventions (panopticapi vs.
cityscapesScripts / mmsegmentation).

The two diverge most where you care most — on long-tail classes that
your model gets wrong. Use the paradigm that matches your downstream
question.

## Things vs. stuff (panoptic-only concept)

Panoptic distinguishes **things** (countable, instance-like:
person, car, bicycle) from **stuff** (uncountable, region-like: sky,
road, vegetation). The split is a panoptic-only concept — semantic
mIoU has no thing/stuff distinction (every class is a stuff-shaped
region in the per-pixel kernel) and instance evaluation only
considers things (stuff isn't instance-segmentable).

## See also

- [ADR-0011](../adr/0011-discriminated-kernel-config.md) — why the
  AP fold has a discriminated `IouKind` union (Bbox / Segm /
  Boundary / Keypoints) instead of separate evaluators.
- [ADR-0025](../adr/0025-panoptic-api.md) — why panoptic is a
  sibling crate, not an `IouKind::Panoptic` arm.
- [ADR-0028](../adr/0028-sem-seg.md) — why semantic is a sibling
  crate, not an `IouKind::Semantic` arm.
- [ADR-0029](../adr/0029-namespace.md) — per-paradigm submodule
  layout (`vernier.instance` / `vernier.panoptic` /
  `vernier.semantic`).
- [Migrating from `mmsegmentation`](../migrate/from-mmsegmentation.md)
  — semantic-side migration.
- [Migrating from `panopticapi`](panoptic-migration.md) —
  panoptic-side migration.
- [Migrating from `lvis-api`](../migrate/from-lvis-api.md) —
  long-tail-instance migration.
