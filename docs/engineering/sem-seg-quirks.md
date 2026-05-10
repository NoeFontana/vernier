# semantic segmentation quirks survey

A working note (not an ADR) cataloguing the numerical and structural
quirks of the three semantic-segmentation oracles vernier reckons
with: `mmsegmentation` (the de-facto research reference),
`cityscapesScripts` (the dataset-author reference for Cityscapes),
and the Pascal VOC / ADE20K reference scripts (treated as
documentation oracles, see disposition table below).

> **Status:** ratified by ADR-0028 on 2026-05-04. The
> `(quirk_id, oracle) → mode` disposition cells below are the
> contract `crates/vernier-semantic/src/parity.rs` and the
> `tests/python/parity_semantic/` harness implement against. ADR-0030
> will extend the survey with boundary-mIoU rows when that subsystem
> ships.

This survey is intentionally **independent of**
`docs/engineering/pycocotools-quirks.md`,
`docs/engineering/boundary-iou-quirks.md`,
`docs/engineering/panopticapi-quirks.md`, and
`docs/engineering/lvis-quirks.md`. The five documents share no
quirks, no fixtures, no parity harness, and no oracle. They share
only the `ParityMode` enum (`strict` / `corrected`, per ADR-0002
amended 2026-05-10) defined in `crates/vernier-core/src/parity.rs`.

## What's structurally new in this survey

Three oracles disagree on Cityscapes-19 in subtle, specific ways
(rows AB1, AC2, AC4, AD3 below have the worst splits). The
single-oracle disposition column from earlier surveys cannot honestly
represent that — a row that's `strict` against mmsegmentation might
be `aligned` against cityscapesScripts because the two upstreams
implement the same metric differently. ADR-0028 §"Parity strategy"
introduced **`(quirk, oracle) → mode`** keying for exactly this
case; this survey is its first realization.

Each row has up to three disposition cells, one per oracle:

- **MS** — mmsegmentation (default oracle; `parity_mode="strict"`
  on `vernier.semantic.Evaluator()` is keyed against this).
- **CS** — cityscapesScripts (Cityscapes-specific dataset-author
  oracle; the `SemanticDataset.cityscapes(...)` preset claims
  parity against this).
- **PA** — Pascal VOC / ADE20K reference scripts (documentation
  oracles; cross-oracle-tolerance parity claims, not bit-equal, per
  ADR-0028 §"Parity strategy").

A cell value is one of:

- **strict** — vernier reproduces this oracle's behavior bit-exactly.
- **aligned** — survey-row annotation for "vernier matches the
  *semantics* but may differ in incidental details" (cross-oracle
  tolerance, not the runtime `ParityMode`). Per-row `aligned`
  cell-values below are pending re-audit against the amended
  ADR-0002.
- **corrected** — vernier opts to fix this. Default behavior
  diverges from the oracle and the divergence is documented as an
  opinionated improvement.
- **n/a** — the oracle does not implement this code path
  (e.g., mmsegmentation does not implement Cityscapes-specific
  ID remapping in its `IoUMetric`; the row is `n/a` for MS).
- **informational** — pins a property of the upstream that vernier
  *relies on* but does not reproduce. No vernier-side behavior
  follows; the row exists as evidence for an ADR commitment.
- A trailing `(deferred)` qualifier marks a row whose disposition
  applies only when the relevant subsystem ships.

Where all three oracles agree, the row collapses to a single
disposition for brevity. Where they diverge, the row gets all
three cells.

The disposition columns are a **draft proposal** by the author of
this survey — ADR-0028 is the venue where each cell gets ratified
or revised.

## Source reference

Line numbers refer to the versions pinned in
`tests/python/parity_semantic/oracle/VENDORING.md`. As of writing:

- **mmsegmentation** at the version pinned in
  `ORACLE_MMSEG_VERSION` (commit SHA in `VENDORING.md`).
  `mmseg/evaluation/metrics/iou_metric.py` —
  `IoUMetric.compute_metrics`, `intersect_and_union`. The
  `IoUMetric` class is the de-facto research-reference
  implementation; many recent papers' eval scripts call this
  directly.
- **cityscapesScripts** at the version pinned in
  `ORACLE_CITYSCAPESSCRIPTS_VERSION`.
  `cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py`
  — `evaluatePair`, `printResults`, the `args` global. Custom
  19-class evaluation; the dataset-author reference.
- **Pascal VOC reference** at `VOCdevkit/VOCcode/VOCevalseg.m`
  (Matlab) and the de-facto Python reproductions in
  `voc12-segmentation` repos (line numbers approximate; cited as
  `voc:~`). Used as a documentation oracle only.
- **ADE20K reference** at the SceneParse150 development kit's
  `evaluationCode/utils_eval.py`. Used as a documentation
  oracle only.

Conventions used below:

- "ms" = mmsegmentation `iou_metric.py`,
  "cs" = cityscapesScripts `evalPixelLevelSemanticLabeling.py`,
  "voc" = Pascal VOC reference, "ade" = ADE20K reference.
- Line citations like `ms:120` mean `iou_metric.py:120`. A `~`
  prefix marks an approximate citation to a section.

The quirks survey starts at **AI** because pycocotools used A–L,
boundary M–Q, panoptic R–Z, and LVIS AA–AH. AI–AP keeps the
global namespace unambiguous when quirks are referenced in code
comments and tests across surveys.

---

## AI. Confusion-matrix kernel

These rows pin the central per-image kernel: turning a pair of
class-id tensors into a per-class confusion matrix.

| # | Quirk | Source | MS | CS | PA |
|---|---|---|---|---|---|
| AI1 | The per-image kernel is `np.bincount(gt * n_classes + pred, minlength=n_classes**2).reshape(n_classes, n_classes)` (or the equivalent histogram fold). One pass over the joint label map. | ms:~80 | **strict** | **aligned** (cs uses a Python loop with class enumeration; bit-equal output, different control flow) | **aligned** |
| AI2 | Pixels where `gt == ignore_label` are excluded from the bincount input (mask-out before histogram). The pred value at those pixels is irrelevant. | ms:~75 | **strict** | **strict** (cs's ignore_label is 255) | **strict** (Pascal VOC's is 255; ADE20K's is 0) |
| AI3 | Pixels where **`pred == ignore_label`** but `gt != ignore_label`: behavior diverges across oracles. mmsegmentation counts these as misses against the GT class (the pred value is recorded in the `pred=ignore_label` column, which lies *outside* the `n_classes × n_classes` confusion matrix and is silently dropped — so they contribute to FN but not to any FP cell). cityscapesScripts errors out if a prediction file contains the ignore label. | ms:~78 (silent drop), cs:~90 (error) | **strict** | **strict** (vernier's `parity_mode="strict"` against cs raises `PredictionContainsIgnore`) | **strict** (Pascal VOC follows mmsegmentation's silent-drop semantics) |
| AI4 | The bincount `minlength` is `n_classes**2`. If a pixel's `pred` is `>= n_classes`, the result has more entries than expected; mmsegmentation simply truncates by reshaping to `(n_classes, n_classes)` (the over-class entries are dropped silently). cityscapesScripts validates that prediction values are in the valid 19-class set up front. | ms:~80, cs:~85 | **strict** for legacy compatibility, **corrected** by default (vernier rejects `pred >= n_classes` at `SemanticPredictions::from_arrays` with `OutOfRangePrediction { image_id, class_id }`) | **strict** (cs's eager validation matches vernier's corrected default) | **strict** for legacy compat |
| AI5 | The confusion matrix is summed across images via element-wise integer addition. `u64` counts; safe to `2^64-1` per cell, well above the worst case (full Cityscapes train set: ~5000 images × 2048×1024 = 1.05e10 pixels, fits easily). | ms:~120 | **strict** (vernier uses `u64` per ADR-0028 §"Numerical layout") | **strict** | **strict** |
| AI6 | Per-image GT and pred shapes must match. mmsegmentation raises `ValueError` from NumPy on shape mismatch; cityscapesScripts asserts before the kernel; both error inputs at the kernel boundary. | ms:~70, cs:~80 | **corrected** (vernier surfaces `ShapeMismatch { image_id, gt_shape, pred_shape }` at `SemanticPredictions` construction, not at eval time) | **corrected** | **corrected** |
| AI7 | Per-image dtype: GT is typically `uint8` (Cityscapes) or `uint16` (ADE20K with 150 classes); predictions are typically `uint8`. The bincount fold promotes to `int64` internally; output cell counts are `int64` in mmsegmentation, accumulated to `int64` totals. | ms:~80 | **aligned** (vernier uses `u32` per pixel for the input, `u64` for the cell counts; bit-equal output for `n_classes ≤ 2^16`, which covers every realistic semseg dataset) | **aligned** | **aligned** |

## AJ. Ignore-label handling

These rows pin the per-dataset ignore-label conventions and how
they propagate through the kernel and the summary.

| # | Quirk | Source | MS | CS | PA |
|---|---|---|---|---|---|
| AJ1 | `ignore_label = 255` is the Cityscapes / Pascal VOC convention; `ignore_label = 0` is the ADE20K convention. mmsegmentation defaults to `ignore_label = 255` if not specified; cityscapesScripts hardcodes 255 for Cityscapes. | ms:~50 (config), cs:~30 (hardcoded), voc:~10, ade:~15 | **strict** | **strict** (vernier's `cityscapes()` preset sets it) | **strict** for VOC, **strict** for ADE20K (vernier's `ade20k()` preset sets it to 0) |
| AJ2 | Ignore-label pixels are excluded from BOTH numerator and denominator of per-class IoU. Equivalently, they contribute zero to TP, FP, and FN. This is the only sane semantics, but it must be stated — the alternative ("ignore label gets its own class row in the confusion matrix") would corrupt every metric. | ms:~75 | **strict** | **strict** | **strict** |
| AJ3 | An ignore-label class is **not** counted toward `n_classes` for the mean. mmsegmentation's `IoUMetric` reports mIoU over the explicit class list; the ignore label is filtered out before iteration. cityscapesScripts maintains a separate "trainId" vs "id" mapping that filters ignore labels at the remap step (AK1). | ms:~140, cs:~70 | **strict** | **strict** | **strict** |
| AJ4 | Multiple ignore labels per dataset: some configurations (e.g., mmsegmentation custom datasets) accept a *list* of ignore labels, masking pixels matching any of them. cityscapesScripts is single-ignore. | ms:~50 | **aligned** (vernier accepts `ignore_label: int | None` only — single ignore label per evaluator; rejects list inputs at construction with a typed error pointing to the multi-label workaround via `label_remap` AK2) | **strict** | **strict** |
| AJ5 | The `ignore_label` value can in principle equal a real class id (e.g., `ignore_label=0` on ADE20K, where class 0 is "other/background"). mmsegmentation's `IoUMetric` handles this correctly by treating the ignore label as a sentinel before the bincount; the histogram never sees the value. | ms:~75 | **strict**. The "ignore equals real class" case is metric-relevant on ADE20K. | **n/a** (Cityscapes ignore=255 doesn't collide) | **strict** for ADE20K |

## AK. ID remapping

These rows pin Cityscapes-specific ID remapping (training IDs vs.
evaluation IDs) and the per-dataset preset resolution.

| # | Quirk | Source | MS | CS | PA |
|---|---|---|---|---|---|
| AK1 | Cityscapes ships with 30+ class definitions ("labels" file); the standard 19-class evaluation maps these to a smaller set, with several IDs remapping to ignore (255). The mapping is in `cityscapesscripts/helpers/labels.py` as a list of `Label(name, id, trainId, ...)` tuples. cityscapesScripts uses `trainId` for evaluation; predictions are expected in `trainId` space. | cs:helpers/labels.py | **n/a** (mmsegmentation expects predictions already remapped to the 19-class space; user does the remap upstream) | **strict** (vernier's `cityscapes()` preset bakes the canonical 19-class remap table into the dataset constructor) | **n/a** |
| AK2 | The user-side path: `SemanticEvaluator(label_remap={...})` accepts a per-class remap dict. Predictions are remapped to evaluation classes before the bincount; pixels remapped to `ignore_label` are masked. The dict can be shipped per-dataset via constructors (`cityscapes()` populates the canonical Cityscapes remap; users with custom datasets pass their own dict). | (vernier-side) | **aligned** (mmsegmentation has a similar `reduce_zero_label` flag with narrower semantics; the corrected disposition is documented in the migration guide) | **strict** | **aligned** |
| AK3 | Pascal VOC has 21 classes (background + 20 objects); no remap needed. ADE20K has 150 classes; the SceneParse150 convention is direct (no remap). The `pascal_voc()` and `ade20k()` presets set `label_remap=None`. | (vernier-side) | **n/a** | **n/a** | **strict** |
| AK4 | `reduce_zero_label` (mmsegmentation): when `True`, predictions are decremented by 1 before evaluation, treating class 0 as "ignore" and shifting the remaining classes down. Used for ADE20K-style datasets where class 0 is background but the model emits 0-indexed class IDs. | ms:~55 | **strict** when `reduce_zero_label=True` is consumed via `label_remap={0: 255, 1: 0, 2: 1, ...}` (the equivalent dict). vernier doesn't ship a `reduce_zero_label` flag; the migration guide documents the equivalent dict. | **n/a** | **n/a** |
| AK5 | Remapping pixels to `ignore_label`: a `label_remap` entry pointing at the ignore label converts that class's pixels to ignore during the mask-out step. Useful for dropping rare classes from evaluation. | (vernier-side) | **aligned** | **strict** (cs's labels file uses this for "void" classes; remap to trainId=255) | **aligned** |

## AL. Per-class IoU formula and aggregation

These rows pin the load-bearing metric definitions: per-class IoU,
mIoU, FWIoU, pixel accuracy, mean accuracy.

| # | Quirk | Source | MS | CS | PA |
|---|---|---|---|---|---|
| AL1 | Per-class IoU: `IoU_c = TP_c / (TP_c + FP_c + FN_c)`, where `TP_c = confusion[c, c]`, `FP_c = sum(confusion[:, c]) - TP_c`, `FN_c = sum(confusion[c, :]) - TP_c`. Denominator is the union; class is from the GT row, prediction from the column. | ms:~140 | **strict** | **strict** | **strict** |
| AL2 | Per-class IoU when denominator is zero (class never seen and never predicted): mmsegmentation returns `nan`; cityscapesScripts returns `0.0`; ADE20K reference returns `nan`. The behavior diverges. | ms:~145, cs:~110, ade:~30 | **strict** (vernier's `parity_mode="strict"` against MS preserves NaN; `aligned` produces NaN; `corrected` returns 0.0 with a warning) | **aligned** (vernier's `cityscapes()` preset under MS-default reports NaN; the documented difference vs. cs's 0.0 is in the migration guide) | **strict** |
| AL3 | mIoU: unweighted mean of per-class IoU over classes where `TP_c + FP_c + FN_c > 0`. Categories with no support are excluded from the mean (mirrors panopticapi quirk **W2** and LVIS quirk **AB3**). NaN handling diverges per AL2. | ms:~155 | **strict** (mmsegmentation's `np.nanmean`) | **aligned** (cs averages over a fixed 19-class set; missing classes contribute 0.0 — see AL2) | **strict** |
| AL4 | FWIoU (frequency-weighted IoU): `FWIoU = sum_c (freq_c * IoU_c)` where `freq_c = (TP_c + FN_c) / total_evaluated_pixels`. Weighting reflects class prevalence in GT, not in predictions. | ms:~160 | **strict** | **n/a** (cs doesn't report FWIoU) | **strict** for VOC, **n/a** for ADE20K |
| AL5 | Pixel accuracy: `pAcc = sum_c TP_c / total_evaluated_pixels = trace(confusion) / sum(confusion)`. Single scalar. | ms:~165 | **strict** | **strict** | **strict** |
| AL6 | Mean accuracy (per-class recall, then mean): `mAcc = mean_c (TP_c / (TP_c + FN_c))` over classes with `TP_c + FN_c > 0`. Different from pixel accuracy: pAcc is global, mAcc averages per-class recall and is robust to class imbalance. | ms:~170 | **strict** | **n/a** (cs reports per-class IoU only) | **strict** |
| AL7 | The order of classes in the per-class output is the order of the categories list passed in. mmsegmentation preserves construction order; cityscapesScripts uses `trainId` order (which is the canonical 19-class Cityscapes order). | ms:~140, cs:~95 | **strict** | **strict** (cs's order is what users expect on Cityscapes; the `cityscapes()` preset preserves it) | **strict** |
| AL8 | The aggregated confusion matrix is itself a first-class output of mmsegmentation's `IoUMetric`. cityscapesScripts builds it implicitly but doesn't surface it. ADR-0028's `SemanticSummary.confusion_matrix` exposes it directly per F1; calibration / error-decomposition / model-diff tools consume it. | ms:~135 | **strict** | **aligned** (cs's per-class TP/FP/FN counts are equivalent; vernier surfaces a confusion matrix with the same data) | **aligned** |

## AM. Image-list and prediction-coverage protocols

These rows pin how the oracles handle missing predictions, missing
GT, and per-image iteration.

| # | Quirk | Source | MS | CS | PA |
|---|---|---|---|---|---|
| AM1 | Iteration is **GT-driven**: every GT image must have a prediction. mmsegmentation's `IoUMetric` is called per-image during a typical training/eval loop, so the prediction list is constructed alongside the GT list. cityscapesScripts requires prediction files matching the GT directory structure exactly. | ms:~95 (per-image API), cs:~50 | **strict** | **strict** (vernier surfaces `MissingPrediction { image_id }` at evaluator entry) | **strict** |
| AM2 | Predictions for images not in the GT set: mmsegmentation silently ignores; cityscapesScripts errors. | ms:~95, cs:~55 | **aligned** (vernier ignores extras silently; this is the COCO precedent and matches mmsegmentation) | **corrected** (vernier ignores extras even when the dataset is `cityscapes()`-loaded; documented divergence from cs's strict-error behavior in the migration guide) | **aligned** |
| AM3 | The `cityscapes()` preset auto-loads the 500-image val set from the Cityscapes directory layout. Image IDs derive from filenames (`<city>_<sequence>_<frame>_gtFine_labelTrainIds.png`). | cs:~40 (file globbing) | **n/a** (mmsegmentation accepts pre-loaded arrays; the loader path is upstream of `IoUMetric`) | **strict** | **n/a** |
| AM4 | The `ade20k()` preset auto-loads the SceneParse150 val set (~2000 images) with the convention `ADE_val_00000001.png` naming. ADE20K predictions are typically uint8 with class IDs in `[0, 150]`. | ade:~25 (file globbing) | **n/a** | **n/a** | **strict** |
| AM5 | Per-image evaluation order: mmsegmentation evaluates in the order images arrive (typically dataloader iteration order). cityscapesScripts sorts by filename. The order does not affect the final confusion matrix (additive aggregation per AI5), but a per-image table downstream sees a different order. | ms:~95, cs:~60 | **aligned** (vernier sorts by `image_id` for determinism; bit-equal aggregate output, ordered per-image rows) | **aligned** | **aligned** |

## AN. Multi-class predictions vs. binary masks

These rows pin the H2 case from ADR-0028 — the binary-mask-per-class
input shape used by AV / robotics free-space / drivable-surface
models.

| # | Quirk | Source | MS | CS | PA |
|---|---|---|---|---|---|
| AN1 | mmsegmentation, cityscapesScripts, and the ADE20K reference all expect a single class-id label map per image (multi-class `(H, W)` input). None of them accept per-class binary masks natively. | ms:~70, cs:~85 | **n/a** | **n/a** | **n/a** |
| AN2 | When predictions arrive as per-class binary masks (one mask per class, shape `(K, H, W)`), they need to be merged into a class-id label map before evaluation. The merge requires a precedence rule for overlapping pixels. mmsegmentation's reference loader uses `argmax` over the channel axis; for sigmoid-output multi-label models, the rule is "highest-score class wins." | ms:~65 (`reduce_pred`) | **strict** for argmax merge. **corrected** for sigmoid: vernier's `from_binary_masks` constructor accepts an explicit `merge: Literal["argmax", "first", "highest_class_id"]` selector, defaulting to `argmax`. | **n/a** | **n/a** |
| AN3 | When two binary masks both equal 1 at a pixel, the merge decision matters. `argmax` requires score channels (not just binary masks); `first` is order-dependent; `highest_class_id` is convention-dependent. ADR-0028 H2 ships `from_binary_masks` with explicit selector and documents the ambiguity. | (vernier-side) | **corrected** by construction. | **n/a** | **n/a** |
| AN4 | Pixels where all binary masks equal 0 (no class predicted): treated as `ignore_label` by default, since the absence of any class is structurally different from "background" (which is itself a class on Pascal VOC and ADE20K). User can override via the `unlabeled_class` parameter on `from_binary_masks`. | (vernier-side) | **corrected** | **n/a** | **n/a** |

## AO. Boundary-mIoU rows (ADR-0030)

These rows are deferred until ADR-0030 is implemented; the
boundary-mIoU evaluator is not part of the initial ADR-0028 ship.
Each row's full disposition is sketched here for the follow-up
ADR's quirks-survey extension.

| # | Quirk | Source | MS | CS | PA |
|---|---|---|---|---|---|
| AO1 | Boundary mIoU as IoU on per-class Chebyshev-eroded boundary bands. Implementation reuses `vernier_mask::ops::boundary_band` per ADR-0030 §"Per-class Chebyshev erosion". Default `dilation_ratio = 0.01` per ADR-0030 §D3 (Cheng et al. 2021 §6 published value for semantic). | (ADR-0030) | **strict** (deferred) | **strict** (deferred) | **strict** (deferred) |
| AO2 | Per-class boundary band construction inherits ADR-0010 quirks **M1–M5** (Chebyshev erosion algorithm) and **N1–N5** (padding and edge handling) verbatim. The single-pass separable erosion is bit-equal to the iterative reference. | boundary-iou-quirks **M1–M5, N1–N5** | **strict** (deferred) | **strict** (deferred) | **strict** (deferred) |
| AO3 | Per-pixel mismatch filter (the "obvious" implementation: a pixel is boundary iff any 3×3 neighbor has a different class) computes a *different metric* — the union of per-class eroded bands is not the same set as the per-pixel disagreement set. ADR-0030 §C2 documents this as a known wrong implementation. | (ADR-0030) | **corrected (deferred)** — vernier rejects this approach by construction; the C1 implementation is the only one shipped. | **corrected (deferred)** | **corrected (deferred)** |
| AO4 | The semantic-side `dilation_ratio` default (0.01) differs from the instance-side default (0.02 from ADR-0010). The per-paradigm defaults track published numbers. | (ADR-0030) | **strict (deferred)** | **strict (deferred)** | **strict (deferred)** |
| AO5 | BF-score (Csurka et al. 2013 contour F-measure) is a different metric from boundary mIoU. ADR-0030 §A2 defers BF-score to a future ADR; the initial boundary-mIoU ship does not implement it. | Csurka et al. 2013 | **corrected (deferred)** — vernier ships boundary mIoU, not BF-score. | **corrected (deferred)** | **corrected (deferred)** |

## AP. API and CLI surface

These rows pin the user-facing differences between the three
oracles and `vernier.semantic.Evaluator`.

| # | Quirk | Source | MS | CS | PA |
|---|---|---|---|---|---|
| AP1 | mmsegmentation's `IoUMetric` is constructed inside an evaluation loop, called per-batch with `process(data_batch, data_samples)`, and produces metrics via `compute_metrics(results)`. The lifecycle is "per-experiment instance"; reuse across experiments requires reset. | ms:~30 | **aligned** (vernier's `Evaluator` is immutable per ADR-0006; new evaluator per dataset/predictions pair) | **n/a** | **n/a** |
| AP2 | cityscapesScripts has a top-level `evaluatePair(predictionImg, groundTruthImg, args)` function plus a global `args` object holding configuration. The `args`-as-global pattern conflicts with vernier's frozen-evaluator (ADR-0006). | cs:~70 (function), cs:~30 (args) | **n/a** | **corrected** (vernier surfaces all configuration as constructor parameters; no globals) | **n/a** |
| AP3 | mmsegmentation prints metrics via `mmengine.runner.runner.Runner._log_metrics`, which formats a structured table. cityscapesScripts prints a plain-text table to stdout. ADE20K reference prints with `print()` calls. | ms:~180, cs:~115, ade:~35 | **corrected** (vernier returns structured `SemanticSummary`; CLI subcommand reproduces the print format under `--format text` per ADR-0015 verb extensibility) | **corrected** | **corrected** |
| AP4 | Errors raised mid-evaluation are bare `Exception` / `KeyError` / `ValueError` with f-string messages, parallel to pycocotools / panopticapi / lvis-api. | ms:~75, cs:~80, ade:~25 | **corrected** (vernier raises typed `SemanticError` variants matching each upstream raise site) | **corrected** | **corrected** |
| AP5 | mmsegmentation transitively pulls PyTorch (~3 GB on a clean install) via `mmengine`. ADR-0036 vendors `mmseg/evaluation/metrics/iou_metric.py` standalone at v1.2.2 (`c685fe6767c4cadf6b051983ca6208f1b9d1ccb8`) plus a hand-written `mmengine` / `mmseg.registry` / `prettytable` stub layer in `tests/python/parity_semantic/oracle/mmsegmentation/_mmengine_stub.py`. PyTorch remains a real test-only dep (`IoUMetric.intersect_and_union` calls `torch.histc`, no bit-exact numpy equivalent) but mmcv / mmengine / the mmsegmentation package itself never enter the test environment. | ms (package metadata) | **resolved (vendored)** | **n/a** (cs is small, no PyTorch dep) | **n/a** |
| AP6 | cityscapesScripts ships a `csCreateTrainIdLabelImgs.py` preprocessing tool to convert the dataset's full-class PNGs into trainId-space PNGs. Many users feed the tool's output directly to evaluation. | cs:tools | **n/a** | **strict** for the trainId convention; the `cityscapes()` preset accepts both raw and preprocessed PNGs and applies the canonical remap (AK1) at load if needed. | **n/a** |
| AP7 | mmsegmentation accepts a `format_only` flag that builds the metric structure without actually computing the metrics. Used during inference pipelines that emit predictions but defer evaluation. | ms:~60 | **n/a** for vernier (the `Evaluator.evaluate()` path always evaluates; no format-only mode) | **n/a** | **n/a** |

---

## Glossary cross-reference (for ADR-0028 authors and reviewers)

When ADR-0028 is reviewed, every row above must be cited by ID.
The disposition cells are the author's *proposal*; the ADR is the
venue where each cell is signed off. A short cheat-sheet:

- **Most rows: strict against MS (default oracle).** mmsegmentation
  is the most-cited semantic-segmentation reference. Reproducing
  it bit-for-bit is the headline parity claim.
- **Diverging rows where MS and CS disagree:** AL2 (NaN vs. 0.0
  for absent classes), AL3 (the mean handles them differently),
  AM2 (silent-ignore vs. error on extras). The migration guide
  surfaces each as a "if you're moving from cityscapesScripts,
  expect this difference" callout. Strict-against-MS is the
  default; users who need cs-equivalent behavior pass an
  explicit flag (documented but not auto-set per ADR-0028 F2's
  precedent against hidden coupling).
- **Aligned rows:** AI1 (different control flow, same output —
  bincount vs. Python loop), AI7 (different per-pixel dtype,
  same output for realistic class counts), AJ4
  (single-vs-multi-ignore via `label_remap` workaround), AK2
  (`reduce_zero_label` vs. equivalent dict), AL8 (confusion
  matrix surfaced directly vs. derived counts), AM5 (sorted
  vs. arrival-order iteration).
- **Corrected:** AI4 (out-of-range pred rejection), AI6 (typed
  shape mismatch), AM2 vs cs (silent-ignore divergence), AN2-AN4
  (binary-mask merge selector), AP1-AP4 (typed errors,
  structured return, no globals).
- **Resolved (vendored):** AP5 (mmsegmentation `IoUMetric` vendored at v1.2.2; ADR-0036 closed the PyTorch transitive concern).
- **Out-of-scope:** the boundary-mIoU rows AO1-AO5 are
  deferred to ADR-0030; the row dispositions are sketched but
  apply only when the boundary subsystem ships.

The ADR-0028 disposition table is the canonical source; this
survey is the per-row evidence base. The `(quirk, oracle) → mode`
keying is the genuinely new methodological contribution; ADR-0002
may be revisited after this survey lands to formalize the
multi-oracle case.

## Open questions

These are quirks where the reading is uncertain and we should
write a small reproducer before signing off.

1. **AL2 NaN-vs-zero divergence on real datasets.** Run
   mmsegmentation `IoUMetric` and cityscapesScripts on Cityscapes
   val. Identify classes with zero support (rare). Confirm the two
   produce different per-class IoU values for those classes (NaN
   vs. 0.0). Pin the divergence with a test fixture; document the
   expected behavior under each `parity_mode`. The migration
   guide's "moving from cityscapesScripts" section relies on this
   fixture being correct.

2. **AI3 silent-FN-on-pred-ignore.** Construct a fixture where
   prediction PNG contains the ignore label (255 on Cityscapes)
   for some pixels. Confirm:
   (a) mmsegmentation: silently drops those pixels, contributes
   to FN for the GT class but not FP for any pred class.
   (b) cityscapesScripts: errors out at load.
   (c) vernier `parity_mode="strict"` against MS: matches (a).
   (d) vernier `cityscapes()` preset: matches (b) by default; the
   dual-oracle disposition is captured at the row level (the survey
   below). The fixture pins both observable behaviors.

3. **AK4 `reduce_zero_label` equivalence with `label_remap`.**
   Run mmsegmentation `IoUMetric(reduce_zero_label=True)` on
   ADE20K val, and run vernier with the equivalent
   `label_remap={0: 255, 1: 0, 2: 1, ...}`. Assert byte-equal
   per-class IoU and mIoU. If they differ, the migration guide's
   ADE20K migration recipe is wrong; this is the fixture that
   catches it.

4. **AM3-AM4 preset auto-load behavior.** Construct a malformed
   Cityscapes directory (missing one image, extra one image,
   wrong filename pattern) and confirm vernier's `cityscapes()`
   constructor produces typed errors with image-id specificity,
   not bare KeyError. Same for ADE20K's directory layout.

5. **AN3 binary-mask merge ambiguity.** Construct a fixture with
   two binary masks that overlap on a 100-pixel region. Run with
   each of `merge="argmax"` (requires score channels — not
   applicable here, will error or fall back),
   `merge="first"` (first-class wins), and
   `merge="highest_class_id"` (highest ID wins). Confirm the
   per-class confusion matrix differs across the three; document
   the expected delta. The selector being explicit is what
   prevents this from being an invisible footgun.

6. **AP5 PyTorch dep avoidance.** *(Resolved 2026-05-09 — ADR-0036.)*
   `IoUMetric` cannot be exercised without `torch` —
   `intersect_and_union` calls `torch.histc(input.float(),
   bins=N, min=0, max=N-1)` for label binning, and `torch.histc`
   has no bit-exact numpy equivalent. **However**, mmcv +
   mmengine + the mmsegmentation package itself (which together
   account for the bulk of the ~3 GB transitive) are entirely
   stubbable. ADR-0036 vendors `iou_metric.py` standalone with
   hand-written `mmengine` / `mmseg.registry` / `prettytable`
   stubs in `tests/python/parity_semantic/oracle/mmsegmentation/`;
   `torch` remains a real test-only dep. Net result: from
   ~3 GB transitive to a ~600 MB CPU torch wheel, no rewrite of
   ADR-0028 §"Negative consequences" needed (the mitigation is
   the new ADR, not a change to the original cost analysis).

7. **MS and CS pycocotools transitive pin.** Cross-check whether
   mmsegmentation and cityscapesScripts pull `pycocotools` (they
   shouldn't — semantic eval doesn't need it) and confirm no
   version conflict with vernier's `==2.0.11` pin (ADR-0002).

Each open question is a ~30-minute fixture; a follow-up commit
should resolve them and update the disposition table.

## Links and references

- ADR-0001 — Record architecture decisions.
- ADR-0002 — Three-tier parity model. Vocabulary reused; the
  `(quirk, oracle) → mode` keying is a refinement that may
  warrant ADR-0002 revision after this survey lands.
- ADR-0005 — `Similarity` trait and matching-engine API lock.
  Validated by AE4 / AI5: the AP fold is untouched by semantic
  eval.
- ADR-0008 — Bbox IoU `f64` end-to-end. AL1-AL6 inherit it.
- ADR-0009 — `vernier-mask` as a pure-Rust leaf crate. The
  semantic kernel doesn't share `vernier-mask`; it adds the
  leaf-direction asymmetry called out in ADR-0028.
- ADR-0010 — Boundary IoU isolated subsystem. Quirks **M1-M5**
  and **N1-N5** are inherited verbatim by AO2 (deferred).
- ADR-0013 — Streaming evaluator. Confusion matrices add
  element-wise; streaming semantic eval falls out of AI5.
- ADR-0019 — Result tables. The follow-up extends per-class
  tables to semantic per ADR-0028 §"Result tables".
- ADR-0025 — Panoptic-quality evaluation. Sibling architecture
  precedent.
- ADR-0026 — LVIS federated evaluation. Precedent for a
  category-subset filter; semantic uses category masks
  differently (per-pixel ignore, not per-image-per-class
  skip).
- ADR-0027 — Documentation framework. The quirks browser
  surfaces this survey at `docs/reference/parity-quirks/semantic-segmentation.md`.
- ADR-0028 — Semantic evaluation as a sibling crate. This
  survey is the per-row evidence base for that ADR's parity
  strategy.
- ADR-0029 — Namespace restructure. `vernier.semantic.Evaluator`
  is where this survey's semantics surface.
- ADR-0030 — Boundary-mIoU. The AO section's deferred rows are
  resolved by ADR-0030.
- `docs/engineering/pycocotools-quirks.md` — pycocotools survey;
  no rows shared but the structural conventions are inherited.
- `docs/engineering/boundary-iou-quirks.md` — quirks **M1-M5,
  N1-N5** are inherited by AO2 when ADR-0030 ships.
- `docs/engineering/panopticapi-quirks.md` — quirk **W2** is
  the precedent for AL3's exclusion-of-zero-support classes from
  the mean.
- `docs/engineering/lvis-quirks.md` — quirk **AB3** is the
  same precedent as panopticapi **W2**, applied to LVIS.
- `tests/python/parity_semantic/oracle/VENDORING.md` — to be
  created alongside the ADR-0028 implementation; pins the three
  oracle versions.
- Long, Shelhamer, Darrell. *Fully Convolutional Networks for
  Semantic Segmentation.* CVPR 2015. arXiv:1411.4038.
- Cordts et al. *The Cityscapes Dataset for Semantic Urban
  Scene Understanding.* CVPR 2016.
- Zhou et al. *Scene Parsing through ADE20K Dataset.* CVPR 2017.
- `open-mmlab/mmsegmentation` — the strict-mode default oracle.
- `mcordts/cityscapesScripts` — the Cityscapes
  dataset-author oracle.
