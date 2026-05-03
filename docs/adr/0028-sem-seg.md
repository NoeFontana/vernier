# ADR-0028: Add semantic segmentation as a `vernier-semantic` sibling crate

- **Status:** proposed
- **Date:** 2026-05-03
- **Deciders:** @NoeFontana
- **Consulted:** —
- **Informed:** all contributors

## Context and problem statement

Semantic segmentation is the third leg of the "things and stuff"
evaluation surface, alongside instance segmentation (vernier-core)
and panoptic segmentation (vernier-panoptic, ADR-0025). For the
robotics / AV perception persona, semantic eval is not an
afterthought — it is the metric family for free-space estimation,
drivable-surface detection, lane segmentation, semantic occupancy
grids, and BEV semantics. A vernier release without it tells AV
users "this library covers the metrics paper-readers care about,
not the metrics deployment-shippers care about", which contradicts
the project positioning.

Through ADR-0025 we noted that panoptic *can* compute IoU on
stuff-only data, but it is the wrong tool: SQ averages IoU only
over **matched** segments (panoptic survey U7: `IoU > 0.5`),
whereas mIoU pools intersections and unions across all images per
class. A class your model gets at IoU=0.3 contributes zero to SQ
and the actual value 0.3 to mIoU; they diverge most where users
care most. The metric difference compounds with two structural
mismatches: panoptic input is RGB-encoded segment-id PNGs plus a
`segments_info` JSON, while semantic eval input is a single
`(H, W)` class-id tensor, and the standard semantic benchmarks
(Cityscapes, ADE20K, Pascal VOC) define `ignore_label`
conventions, ID remapping schemes, and image-list protocols that
COCO panoptic does not.

Three integration shapes were considered. **Module inside
vernier-core** drags `image-rs` decode and a new IoU paradigm
into the AP-fold crate, breaking ADR-0009's leaf discipline.
**Module inside vernier-panoptic** forces semantic users through
the panoptic data model (PNG + segments_info) and inherits the
panopticapi quirks they don't share. **New sibling crate**
mirrors ADR-0025's approach for the same reasons: different
algorithm, different data model, different benchmark
conventions, no kernel sharing with the AP fold or with PQ.

The decision in front of us is the integration shape (sibling
crate confirmed, but the boundaries of *what's in vs. out* are
the design content), the parity story under three competing
oracles with no single authoritative reference, the data-model
shape that supports Cityscapes / ADE20K / Pascal / custom datasets
without 1:1 dataset-to-loader code, and the public API shape
that fits cleanly alongside `Evaluator` (instance) and
`PanopticEvaluator` (panoptic).

This ADR triggers ADR-0001 §"Affect the public API" (new
`SemanticEvaluator` class, new `SemanticDataset` /
`SemanticPredictions` types, new `SemanticSummary` type),
§"Cross the FFI boundary" (new pyfunctions),
§"Add or remove a top-level dependency" (`png` crate via
`vernier-mask` if shared; `mmsegmentation`, `cityscapesscripts`,
optionally a Pascal/ADE20K reference, all test-only).

## Decision drivers

- **ADR-0005 invariant.** No edits to `matching.rs`,
  `accumulate.rs`, or the `Similarity` trait. Semantic eval has
  no per-detection matching, no score-ranked greedy assignment,
  no PR curve. The kernel is a per-image confusion matrix
  accumulator. ADR-0005 is preserved by construction — semantic
  eval doesn't touch the AP spine and doesn't touch the panoptic
  pipeline.
- **ADR-0009 leaf direction.** Pure-Rust kernels reusable outside
  evaluation belong in leaf crates. Semantic eval shares no
  kernel with `vernier-mask` (no RLE, no polygon rasterization,
  no boundary band). The shared leaves are at a higher level:
  `ParityMode` enum, the `Breakdown` axis (ADR-0016), `f64`
  numerical layout (ADR-0008). Those live in `vernier-core` and
  stay there.
- **ADR-0002 parity discipline, applied to *three* oracles.** No
  single pycocotools-equivalent reference exists for semantic
  segmentation. mmsegmentation is the de-facto research
  reference; `cityscapesScripts` is the dataset-author reference
  for Cityscapes; the ADE20K / Pascal communities each have
  their own scripts. The three-tier disposition model still
  applies, but the *table of dispositions* fragments: a
  Cityscapes user wants strict-mode equality vs. cityscapesScripts;
  a research user reproducing a paper wants strict-mode equality
  vs. mmsegmentation. Both must be available; vernier picks the
  default.
- **AV / robotics persona.** Semantic mIoU is a deployment-time
  KPI on free-space, drivable-surface, and lane masks. The
  user-facing API has to feel as natural as `Evaluator` does for
  detection — same construction shape, same parity-mode
  vocabulary, same streaming evaluator pattern.
- **Pre-1.0 freedom.** Project remains on 0.0.x patches.
  Semantic surface is provisional within the patch line; the
  parity contract (mmsegmentation default oracle, vendored at a
  pinned commit, strict-mode bit-equality on a canonical val
  workload) is durable from first ship.
- **Streaming naturally.** A confusion matrix is additively
  aggregable across images. Streaming semantic eval is one
  paragraph of orchestration code; no separate ADR like
  ADR-0013 needed because the spine is the same — the streaming
  evaluator from ADR-0013 takes per-image batches, and the
  semantic kernel takes per-image confusion matrices.

## Considered options

### Axis A — Workspace placement

1. Module inside `vernier-core`.
2. Module inside `vernier-panoptic`.
3. **New workspace crate `vernier-semantic`.**
4. Defer further; no semantic eval in the 0.0.x patch line.

### Axis B — Public-API integration

1. Extend `IouKind` with a `Semantic` arm.
2. **Sibling `SemanticEvaluator` class; `IouKind` unchanged.**
3. Top-level `vernier.evaluate_semantic(...)` function.
4. Restructure into a `vernier.semantic` / `vernier.instance` /
   `vernier.panoptic` namespace.

### Axis C — Input format

1. Pre-decoded numpy `uint8` / `uint16` arrays only; users decode
   their own PNGs / TIFFs.
2. **Pre-decoded arrays + native PNG decode (file paths) via
   `png`.**
3. Native decode for PNG, TIFF, NPY, paletted formats (`image`
   umbrella crate).

### Axis D — Default parity oracle

1. **mmsegmentation pinned + vendored, three-tier disposition.**
2. cityscapesScripts pinned; vendor mmsegmentation as second
   oracle.
3. Custom NumPy reference; both upstream oracles available as
   `corrected` cross-checks.
4. ADE20K / SceneParse150 reference scripts.

### Axis E — Per-dataset preset shape

1. **Constructors on the dataset type:
   `SemanticDataset.cityscapes(...)`,
   `.cityscapes_19class(...)`, `.ade20k(...)`,
   `.pascal_voc(...)`, `.from_arrays(...)`,
   `.from_files(...)`.**
2. Per-dataset `Cityscapes(...)` / `Ade20k(...)` discriminated
   union over `IouKind`.
3. No presets; users construct `SemanticDataset` from primitives
   only.

### Axis F — Metric set

1. **mIoU, FWIoU, pixel accuracy, mean accuracy, per-class IoU,
   per-class accuracy.**
2. Above plus Dice / F1 / class-balanced accuracy.
3. Above plus precision/recall per class and confusion-matrix
   pretty-printer.
4. mIoU only (the headline metric).

### Axis G — Boundary-aware semantic IoU

1. Ship boundary-mIoU together with mIoU.
2. **Defer; resolve in a follow-up ADR.**

### Axis H — Multi-class predictions vs. binary masks

1. Single class-id-tensor entry point (multi-class `(H, W)`,
   `uint8`/`uint16` per pixel).
2. **Class-id-tensor entry point, plus an opt-in
   `from_binary_masks` constructor for users with one
   binary-mask-per-class output (typical of free-space /
   drivable-surface model heads).**
3. Both entry points equally first-class, both consumed by the
   same kernel.

## Decision outcome

Chosen: **A3 + B2 + C2 + D1 + E1 + F1 + G2 + H2.**

### Workspace and dependency direction (A3)

A new workspace crate `crates/vernier-semantic/` is added. Its
dependency direction is:

- `vernier-semantic` → `vernier-core` (for `ParityMode`,
  `Breakdown`, the streaming-evaluator interface from ADR-0013,
  the result-tables interface from ADR-0019).
- `vernier-semantic` ⊥ `vernier-mask` (no RLE codec, no polygon
  rasterizer, no boundary band needed for semantic mIoU).
- `vernier-semantic` ⊥ `vernier-panoptic` (no shared kernels;
  the only thing they share is the `png` decode path, which is
  small enough to duplicate or factor through a helper module).
- `vernier-ffi` → `vernier-semantic` (Python bindings).

`vernier-core` does not depend on `vernier-semantic`. The
dependency edge `vernier-semantic` → `vernier-core` is the only
new asymmetry vs. ADR-0025 (where `vernier-panoptic` ⊥
`vernier-core`); the reason is concrete reuse — `Breakdown` and
the streaming-evaluator interface are useful enough for semantic
eval that duplicating them would be silly. Panoptic doesn't
reuse them because PQ doesn't have an A-axis or a streaming
formulation that benefits from `StreamingEvaluator`'s cell store.

The crate publishes to crates.io as `vernier-semantic`. SemVer
is independent of the wheel; the wheel pins a compatible range.
The reservation package, if any, follows
`docs/engineering/registry-reservations.md`.

PNG decode is the only nontrivial native dep. Two options were
considered: depend on `vernier-mask`'s `png` (if ADR-0025 lands
first and `vernier-mask` grows the dep), or depend on `png`
directly. `vernier-mask`'s charter is RLE / polygon /
mask ops — adding `png` decode there for a panoptic dep is a
stretch already; adding it for *two* dependents is worse.
`vernier-semantic` depends on `png` directly (~150 KB compiled,
same crate `vernier-panoptic` would use under ADR-0025 §E2). If
both crates end up with the same dep, factoring through a tiny
`vernier-image` leaf is a follow-up cleanup, not blocking.

### Public Python surface (B2 + E1 + H2)

```python
@dataclass(frozen=True, slots=True)
class SemanticEvaluator:
    """Semantic-segmentation evaluator (ADR-0028).

    Sibling to :class:`Evaluator` (instance) and
    :class:`PanopticEvaluator` (panoptic). Computes mIoU, FWIoU,
    pixel accuracy, mean accuracy, per-class IoU, and per-class
    accuracy from per-image confusion matrices accumulated into
    a global confusion matrix.
    """

    parity_mode: ParityMode = "corrected"
    # Pixels where GT == ignore_label are excluded from BOTH
    # numerator and denominator. None means "no ignore". Default
    # is None to keep the evaluator semantics dataset-agnostic;
    # the per-dataset constructors (E1) set it to the canonical
    # value (e.g., 255 for Cityscapes).
    ignore_label: int | None = None
    # Optional: when present, predictions of these class ids
    # are remapped to ignore_label before accumulation. Used by
    # Cityscapes 19-class evaluation where 30+ training-set ids
    # collapse to 19 eval ids (and the rest are ignored).
    label_remap: Mapping[int, int] | None = None

    def evaluate(
        self,
        gt: SemanticDataset,
        dt: SemanticPredictions,
    ) -> SemanticSummary: ...

    def stream(
        self,
        n_classes: int,
        ignore_label: int | None = None,
    ) -> StreamingSemanticEvaluator: ...


@dataclass(frozen=True, slots=True)
class SemanticDataset:
    """Semantic-segmentation ground truth.

    Constructors (E1):

    - :meth:`from_arrays` — pre-decoded ``uint8`` / ``uint16``
      class-id tensors (one ``ndarray`` per image), with a
      ``categories`` list giving the eval class layout.
    - :meth:`from_files` — PNG paths; Rust decodes via the
      ``png`` crate.
    - :meth:`from_binary_masks` — H2 entry point. One binary
      mask per (image, class), stacked. Used by free-space /
      drivable-surface models with one head per class.
    - :meth:`cityscapes` — Cityscapes preset (19-class eval
      layout, ignore_label=255, the official train-id-to-eval-id
      remap built in).
    - :meth:`cityscapes_19class` — alias for the 19-class default;
      retained for explicit migration from cityscapesScripts.
    - :meth:`ade20k` — ADE20K preset (150 classes, ignore_label=0
      per the ADE20K convention where label 0 is "other/unlabeled").
    - :meth:`pascal_voc` — Pascal VOC preset (21 classes including
      background, ignore_label=255 for `void` boundary).

    @classmethod
    def from_arrays(
        cls,
        label_maps: Mapping[ImageId, np.ndarray],
        categories: Sequence[CategoryMeta],
        ignore_label: int | None = None,
    ) -> Self: ...

    @classmethod
    def from_files(
        cls,
        png_paths: Mapping[ImageId, str | Path],
        categories: Sequence[CategoryMeta],
        ignore_label: int | None = None,
    ) -> Self: ...

    @classmethod
    def from_binary_masks(
        cls,
        masks: Mapping[ImageId, np.ndarray],  # shape (n_classes, H, W) uint8
        categories: Sequence[CategoryMeta],
    ) -> Self: ...

    @classmethod
    def cityscapes(cls, png_paths: Mapping[ImageId, str | Path]) -> Self: ...
    @classmethod
    def ade20k(cls, png_paths: Mapping[ImageId, str | Path]) -> Self: ...
    @classmethod
    def pascal_voc(cls, png_paths: Mapping[ImageId, str | Path]) -> Self: ...


@dataclass(frozen=True, slots=True)
class SemanticPredictions:
    """Semantic-segmentation predictions; same constructors as
    :class:`SemanticDataset`."""


@dataclass(frozen=True, slots=True)
class SemanticSummary:
    """Per-:meth:`SemanticEvaluator.evaluate` result.

    Sibling to :class:`Summary` and :class:`PanopticSummary`. The
    six headline scalars on top, plus the per-class breakdown.
    """
    miou: float            # F1: mean intersection-over-union, ignore-aware.
    fwiou: float           # F1: frequency-weighted IoU.
    pixel_accuracy: float  # F1: total_correct / total_evaluated_pixels.
    mean_accuracy: float   # F1: mean per-class recall.
    # Per-category rows.
    per_class: Mapping[int, ClassSemanticStats]
    # The accumulated confusion matrix is a first-class output —
    # downstream tools (calibration, error decomposition, model-diff)
    # consume it directly.
    confusion_matrix: ConfusionMatrix


@dataclass(frozen=True, slots=True)
class ClassSemanticStats:
    iou: float
    accuracy: float       # Per-class recall: TP / (TP + FN).
    precision: float      # Per-class precision: TP / (TP + FP).
    n_gt_pixels: int
    n_dt_pixels: int


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """N×N integer matrix, rows = GT class, cols = prediction class.

    Cells are accumulated counts. The ``ignore_label`` axis is
    *not* present — ignore-label pixels are excluded from
    accumulation entirely (per F1 ignore-aware semantics).
    """
    counts: np.ndarray  # shape (N, N), dtype uint64
    categories: Sequence[CategoryMeta]
```

`SemanticEvaluator` is **not** added to `IouKind` (B1 rejected).
`IouKind` is the AP-fold kernel discriminator (per ADR-0011);
semantic eval is not an AP variant. Adding a `Semantic` arm
would silently overload the union — every `match self.iou:` site
in `Evaluator` would have no meaningful arm for it. The
sibling-class shape mirrors ADR-0025 panoptic.

`SemanticSummary` is a sibling type to `Summary` and
`PanopticSummary`, not a subtype. The dimensions differ enough
(no T-axis, no R-axis, no M-axis, no IoU-threshold ladder) that
a unified type would be mostly `None` for two of the three
pipelines. Same precedent as ADR-0019's `EvalResult` / `Summary`
relationship.

### FFI surface (B2 + C2)

Three pyfunctions in `vernier-ffi`:

```rust
fn evaluate_semantic_from_arrays<'py>(
    py: Python<'py>,
    gt_label_maps: Bound<'py, PyDict>,    // ImageId -> uintN ndarray
    dt_label_maps: Bound<'py, PyDict>,
    n_classes: usize,
    ignore_label: Option<u32>,
    label_remap: Option<Bound<'py, PyDict>>,
    parity_mode: &str,
) -> PyResult<SemanticSummaryFfi> { ... }

fn evaluate_semantic_from_files(
    gt_png_paths: BTreeMap<ImageId, PathBuf>,
    dt_png_paths: BTreeMap<ImageId, PathBuf>,
    n_classes: usize,
    ignore_label: Option<u32>,
    label_remap: Option<BTreeMap<u32, u32>>,
    parity_mode: &str,
) -> PyResult<SemanticSummaryFfi> { ... }

fn evaluate_semantic_from_binary_masks<'py>(
    py: Python<'py>,
    gt_masks: Bound<'py, PyDict>,         // ImageId -> uint8 (n_classes, H, W)
    dt_masks: Bound<'py, PyDict>,
    n_classes: usize,
    parity_mode: &str,
) -> PyResult<SemanticSummaryFfi> { ... }
```

All three drop the GIL via `py.detach` per ADR-0006 and run
single-threaded compute through Phase 5. The `from_arrays` and
`from_binary_masks` paths are zero-copy on the ndarrays via
`numpy::PyReadonlyArrayDyn`. The `from_files` path streams
decode-then-accumulate per image so peak memory is bounded by
two label maps (GT + DT) at a time, not by the full dataset.

### Parity strategy (D1)

The parity machinery follows ADR-0010 / ADR-0025 shape, with
**three** vendored oracles instead of one. Default oracle is
mmsegmentation; the others are corrected-mode cross-checks for
their respective dataset presets.

- **Vendored, pinned default oracle.** mmsegmentation
  `IoUMetric` at a frozen commit SHA, vendored at
  `tests/python/parity_semantic/oracle/mmsegmentation/` with
  `VENDORING.md` recording the SHA, the PyTorch / mmcv
  versions the oracle was tested against, and a fork plan.
  mmsegmentation is the de-facto research reference; the strict
  `Evaluator(parity_mode="strict")` claim is against this
  oracle. License: Apache-2.0 (compatible with vernier's
  MIT/Apache-2.0 dual license).
- **cityscapesScripts dataset-author oracle.** Vendored at
  `tests/python/parity_semantic/oracle/cityscapesscripts/` for
  Cityscapes-specific dispositions. Used to validate the
  `SemanticDataset.cityscapes(...)` preset against the
  dataset-author reference. License: MIT.
- **Pascal VOC / ADE20K reference scripts.** Vendored
  similarly. Less rigid — the official scripts are simpler
  (one-page Python) and mmsegmentation already replicates
  their behavior; we use them as documentation oracles, not
  bit-equality oracles. The corresponding presets get
  *aligned*-mode parity claims, not strict.
- **Quirks survey.** New file
  `docs/engineering/semantic-segmentation-quirks.md`
  enumerating the choice points across the three oracles:
  ignore-label semantics, ID remapping, NaN handling for
  classes absent from GT, the
  per-image-vs-pooled-confusion-matrix question, the
  255-as-ignore-vs-255-as-class question (Cityscapes uses
  255 as ignore; SceneParse150 uses 0). Initial pass:
  ~25 rows. Each row gets a strict / aligned / corrected
  disposition before this ADR merges. The disposition
  *table* is keyed on `(quirk_id, oracle)` since the same
  vernier behavior may be `strict` against mmsegmentation
  and `aligned` against cityscapesScripts.
- **Three-tier dispositions, scoped to semantic.** The global
  `ParityMode` enum is reused; the disposition tables it
  consults are semantic-specific. Strict-mode parity claim:
  bit-equal mIoU / FWIoU / pAcc / mAcc / per-class IoU on
  Cityscapes val (default `cityscapes()` preset) against the
  pinned mmsegmentation SHA, and `aligned`-mode parity on
  the same workload against `cityscapesScripts`.
- **Constants module.** `crates/vernier-semantic/src/parity.rs`
  pins `MMSEG_VERSION`, `CITYSCAPESSCRIPTS_VERSION`,
  `SEMANTIC_PARITY_EPS`, the ignore-label conventions
  (`CITYSCAPES_IGNORE_LABEL = 255`, `ADE20K_IGNORE_LABEL = 0`,
  `PASCAL_VOC_IGNORE_LABEL = 255`), and the canonical class
  ID remap tables for each preset.

### Numerical layout

Confusion matrix counts are `u64` integers — pixel counts at
Cityscapes scale are below `u32::MAX` per class per image but
well above it when accumulated across a full val set
(`2048 × 1024 × 500 ≈ 10⁹` pixels per class, up to ~10¹¹ for the
road class). `u64` is the only safe choice. Per-class IoU,
mIoU, and FWIoU are computed in `f64` end-to-end at summarize
time; per ADR-0008's precedent. No SIMD dispatch (`pulp`) on
the v1 semantic path — the kernel is integer-bound (one
addition per pixel) and dominated by memory bandwidth, not
arithmetic throughput; if profiling shows otherwise, SIMD lands
in a follow-up under ADR-0003.

### Algorithm

The kernel is the simplest in vernier:

```rust
fn accumulate_confusion(
    gt: &[u32],     // flattened (H, W), values in 0..n_classes or ignore_label
    dt: &[u32],     // same shape
    n_classes: usize,
    ignore_label: Option<u32>,
    confusion: &mut Array2<u64>,  // (n_classes, n_classes)
) {
    debug_assert_eq!(gt.len(), dt.len());
    for (&g, &d) in gt.iter().zip(dt.iter()) {
        if Some(g) == ignore_label { continue; }
        // DT classes that fall outside [0, n_classes) are clamped
        // to a "spurious-class" bucket — semantic-segmentation
        // quirk SS3 in the survey, dispositioned per oracle.
        let g = g as usize;
        let d = if (d as usize) < n_classes { d as usize } else { /* see SS3 */ };
        confusion[[g, d]] += 1;
    }
}
```

That's the whole load-bearing kernel. The summary metrics
derive from the confusion matrix at finalize:

```
TP_c        = confusion[c, c]
FP_c        = sum(confusion[:, c]) - TP_c
FN_c        = sum(confusion[c, :]) - TP_c
IoU_c       = TP_c / (TP_c + FP_c + FN_c)            # NaN when denom == 0
Acc_c       = TP_c / (TP_c + FN_c)                   # per-class recall
mIoU        = mean(IoU_c) over classes with TP+FP+FN > 0
FWIoU       = sum(freq_c * IoU_c) where freq_c = (TP_c + FN_c) / total_pixels
pAcc        = sum(diag(confusion)) / sum(confusion)
mAcc        = mean(Acc_c) over classes with TP+FN > 0
```

The "exclude classes with zero support from the mean" rule
mirrors panopticapi quirk **W2** and LVIS quirk **AB3** —
a class never seen has undefined IoU, and including a
NaN-or-zero in the mean would corrupt the global. Different
oracles have subtly different rules here (mmsegmentation drops
NaN; cityscapesScripts errors out; some forks use 0); the
quirks survey resolves each, and `parity_mode` selects between
them.

### Streaming

Streaming semantic eval is a three-line follow-on. The
streaming evaluator from ADR-0013 takes per-batch updates and
folds them additively. Confusion matrices add element-wise.
`StreamingSemanticEvaluator.update(image_id, gt, dt)`
accumulates into a per-instance confusion matrix; `snapshot()`
and `finalize()` derive the summary metrics on demand. No
"running mode" / "fast mode" distinction (per ADR-0013
§"Fast snapshot mode") because the full snapshot is already
constant-time relative to image count — it's a fold over a
fixed-size N×N matrix, not over the per-image cell store.

### Result tables (ADR-0019 extension)

`SemanticSummary` carries the confusion matrix as a
first-class output. The follow-up extension to ADR-0019 adds
`per_class` (mirroring the existing detection per_class table,
adapted for the semantic metric set) and `confusion` (a
PyCapsule-backed `RecordBatch` over the confusion matrix in
long-form). Out of scope for this ADR; covered in ADR-0019's
"What this ADR explicitly does *not* decide" follow-up bucket.

## What this ADR explicitly does *not* decide

- **Boundary-aware semantic IoU (boundary-mIoU).** Bowenc0221's
  boundary-iou-api covers boundary-aware Cityscapes panoptic
  but not boundary-aware Cityscapes semantic. There is a
  separate body of work (Csurka et al. 2013 trimap-based
  contour evaluation, the "BF-score") that some semantic-
  segmentation papers report. Resolving the right oracle and
  the right composition rule is its own ADR; until then,
  `SemanticEvaluator` does not accept a boundary flag.
- **3D / BEV semantic segmentation.** Same shape (per-pixel
  class assignment, per-class IoU) but the data model is
  voxel- or BEV-grid-shaped. A BEV-semantic evaluator could
  reuse the confusion-matrix kernel from this ADR, but the
  data ingestion is a different design. Tracked under
  `Possible_Extensions` 3D/BEV bullet; separate package
  decision when demand materializes.
- **CLI subcommand.** `vernier semantic` follows the
  `vernier panoptic` deferral from ADR-0025: in-Python first,
  CLI follow-up. JSON schema bump (per
  `docs/reference/cli-output-schema.md`) at that time.
- **Patching mmsegmentation / cityscapesScripts in-process.**
  No `patch_mmsegmentation()` shim. mmsegmentation's
  `IoUMetric` is a class users instantiate inside an
  evaluation pipeline, not a top-level function — the
  `patch_pycocotools` (ADR-0007) shape doesn't translate.
  Migration guide does the work.
- **Multi-resolution aggregation (sliding-window inference).**
  Predictions are accepted as final per-image label maps;
  vernier does not perform tiling, multi-scale aggregation, or
  TTA on the prediction side. That is an inference-pipeline
  responsibility upstream of evaluation.
- **Per-image mIoU.** The "per-image mIoU" metric is not
  well-defined (a class absent from one image is undefined IoU
  for that image) and biases comparisons toward easy images.
  Same rationale as ADR-0019's per-image-AP omission. The
  per-image confusion matrix *is* exposed via the result-tables
  follow-up; users can derive whatever per-image metric they
  want, but vernier does not pick one.
- **Class-incremental / open-set evaluation.** Domain shift
  detection, novel-class recognition, and OOD pixel
  identification are research problems with no standard
  metric set. Out of scope; revisit when the field
  consolidates.

## Consequences

- **Positive.** The robotics / AV persona gets first-class
  support across all three segmentation paradigms (instance,
  panoptic, semantic) at the same release. ADR-0005's
  invariant survives a third evaluation paradigm being added
  to vernier without modification — confusion matrices flow
  through their own kernel; the AP fold is untouched. The
  `Breakdown` axis (ADR-0016) and the streaming-evaluator
  interface (ADR-0013) are validated as cross-paradigm
  abstractions: both reused by `vernier-semantic` without
  modification. The three-oracle parity story extends the
  three-tier disposition model to "(quirk, oracle) → mode"
  cleanly. Per-dataset presets (E1) lower the per-user
  setup cost to ~3 lines for Cityscapes / ADE20K / Pascal.
- **Negative.** Adds a new workspace crate. Adds a new
  top-level Rust dep (`png`, ~150 KB compiled — same as the
  panoptic crate would; if both ship, the duplication is a
  follow-up cleanup). Adds three new test-only Python deps
  (`mmsegmentation`, `cityscapesscripts`, plus their
  transitives — `mmsegmentation` notably pulls `mmcv` and
  PyTorch, ~3 GB on a clean install). Adds a new parity
  surface with three oracles instead of one. Doubles the
  documentation surface for "how to evaluate" again
  (instance vs panoptic vs semantic users follow different
  tutorials, different migration guides, different reference
  pages). The `SemanticEvaluator` / `PanopticEvaluator` /
  `Evaluator` triple is a real ergonomic cost; the namespace
  question (B4 — `vernier.semantic.Evaluator` /
  `vernier.instance.Evaluator` / `vernier.panoptic.Evaluator`)
  is deferred to a follow-up but worth flagging. The
  `vernier-semantic` → `vernier-core` dep edge is the first
  asymmetry vs. ADR-0025 — `vernier-panoptic` ⊥
  `vernier-core`, but `vernier-semantic` reuses
  `Breakdown` and the streaming interface; honest reuse is
  better than duplication, but the architecture diagram
  becomes asymmetric.
- **Neutral.** `IouKind` discriminated union remains
  AP-shaped. `Summary`, `PanopticSummary`, and
  `SemanticSummary` are siblings, not variants.
  `patch_pycocotools` does not grow; no `patch_mmsegmentation`
  is added. The `f64` end-to-end choice is the same one
  ADR-0008 set; no new precedent. The three-tier parity
  model gains a `(quirk, oracle)` keying refinement, but the
  three modes themselves are unchanged.

## Pros and cons of the options

### A. Workspace placement

- **A1 module in vernier-core.** 👍 single crate. 👎 forces
  every `vernier-core` consumer to pull in the `png` crate
  and a fundamentally different IoU paradigm; muddles the AP
  identity.
- **A2 module in vernier-panoptic.** 👍 reuses `png` decode
  if vernier-panoptic has it. 👎 forces semantic users
  through panoptic data model; inherits panopticapi quirks
  they don't share; couples two evaluation paradigms with
  different oracles.
- **A3 new crate (chosen).** 👍 honors leaf-direction
  discipline; clean dep graph; semantic users never touch
  panoptic code and vice versa. 👎 yet another crate.
- **A4 defer.** 👍 zero new code. 👎 the AV/robotics
  persona case is decided; deferring further is an adoption
  cost.

### B. Public-API integration

- **B1 IouKind arm.** 👍 single evaluator class. 👎 every
  AP-shaped `match self.iou:` site has to grow a Semantic
  arm with no meaningful behavior; ADR-0011's
  exhaustiveness becomes a footgun.
- **B2 sibling class (chosen).** 👍 ADR-0011 unchanged;
  types stay honest. 👎 three evaluator classes.
- **B3 top-level function.** 👍 minimal surface. 👎 no place
  to hang `parity_mode`, `ignore_label`, `label_remap`,
  future options.
- **B4 namespace restructure.** 👍 cleanest long-term shape.
  👎 breaks every existing import; pre-1.0 freedom is real
  but this is a meaningful enough churn to defer until the
  three-evaluator surface has settled.

### C. Input format

- **C1 arrays only.** 👍 keeps `png` dep out. 👎 forces
  every user to install Pillow / OpenCV / imageio for
  decode; the CLI subcommand has nowhere to land; per-dataset
  presets become harder.
- **C2 arrays + native PNG (chosen).** 👍 covers both
  personas without compromise; the `png` crate is tight;
  zero-copy on the array path. 👎 two FFI entry points,
  two test surfaces.
- **C3 image umbrella.** 👍 covers PNG, TIFF, NPY,
  paletted formats. 👎 ~5 MB compiled; we pay for formats
  semantic eval doesn't need (TIFF is the only credible
  case, used by some satellite-imagery datasets; revisit
  if a real user appears).

### D. Default parity oracle

- **D1 mmsegmentation default (chosen).** 👍 the de-facto
  research reference; every recent paper uses it. 👎 large
  install footprint; pulls PyTorch transitively.
- **D2 cityscapesScripts default.** 👍 dataset-author
  authority. 👎 not the reference for non-Cityscapes
  workloads; users reproducing recent papers want
  mmsegmentation parity.
- **D3 custom NumPy reference.** 👍 no upstream coupling.
  👎 we become the reference; every PR re-litigates "is our
  NumPy reproduction correct?"
- **D4 ADE20K reference.** 👍 simple. 👎 same problem as D2,
  one dataset community.

### E. Per-dataset preset shape

- **E1 constructors on dataset (chosen).** 👍 single
  evaluator class; presets are data, not configuration;
  `cityscapes()` is one line. 👎 dataset-class API surface
  grows.
- **E2 discriminated union over kernel.** 👍 mirrors
  ADR-0011 shape. 👎 wrong axis — the kernel is identical
  across datasets; what differs is ignore_label,
  label_remap, and class layout, which are dataset
  metadata, not kernel parameters.
- **E3 no presets.** 👍 minimal surface. 👎 every Cityscapes
  user repeats the same 30-line setup (ID remap table,
  ignore_label=255, 19-class layout); we'd be shipping a
  semantic eval library that doesn't know about the most
  common semantic eval workload.

### F. Metric set

- **F1 mIoU / FWIoU / pAcc / mAcc / per-class (chosen).**
  👍 the mmsegmentation `IoUMetric` set + per-class
  precision; 5 scalars + per-class breakdown is the
  canonical research report. 👎 some users want Dice / F1.
- **F2 plus Dice / F1.** 👍 covers medical-imaging users.
  👎 medical imaging has its own conventions (3D volumes,
  surface-distance metrics) we'd need to address fully or
  not at all; a half-supported medical mode misleads.
- **F3 plus precision/recall + confusion-matrix
  pretty-printer.** 👍 maximally informative. 👎 the
  pretty-printer is presentation, not metric — belongs in
  a result-tables follow-up, not the core summary.
- **F4 mIoU only.** 👍 minimal. 👎 mmsegmentation's
  `IoUMetric` reports the F1 set; matching the reference
  output is metric-relevant.

### G. Boundary-aware semantic IoU

- **G1 ship together.** 👍 closes the boundary surface in
  one go. 👎 doubles initial scope; bowenc0221 doesn't
  cover semantic boundary; resolving the oracle and
  composition rule is its own pass.
- **G2 follow-up ADR (chosen).** 👍 cleaner initial scope;
  matches ADR-0025 §F2 pattern. 👎 boundary-mIoU users
  wait.

### H. Multi-class predictions vs. binary masks

- **H1 single class-id-tensor entry point.** 👍 one path.
  👎 forces free-space / drivable-surface users (one
  binary-mask-per-class outputs) to pre-merge into a
  class-id tensor with arbitrary precedence rules — the
  precedence rule is a model-specific decision they
  shouldn't have to encode in evaluation code.
- **H2 plus binary-mask constructor (chosen).** 👍 covers
  the AV / robotics use case where a head emits multiple
  binary masks (drivable, sidewalk, lane); one extra
  constructor on the dataset, same kernel underneath. 👎
  extra constructor.
- **H3 both equally first-class.** 👍 no preferred path.
  👎 documentation has to teach both as if equivalent;
  they aren't (class-id is more compact and avoids
  precedence ambiguity).

## Links and references

- ADR-0001 — Record architecture decisions.
- ADR-0002 — Three-tier parity model. The vocabulary is
  reused; the disposition table extends to
  `(quirk, oracle) → mode` keying.
- ADR-0005 — `Similarity` trait and matching-engine API
  lock. Validated again: `vernier-semantic` requires no
  edits to `matching.rs` or `accumulate.rs`.
- ADR-0006 — Threading model. Honored: `py.detach` at FFI
  entry, single-threaded compute through Phase 5.
- ADR-0008 — Bbox IoU `f64` end-to-end. Precedent for the
  `f64` choice on the per-class metric path.
- ADR-0009 — `vernier-mask` as a pure-Rust leaf crate.
  Shaped this ADR's leaf-direction decision; no shared
  kernel with `vernier-mask` here.
- ADR-0010 — Boundary IoU as an isolated subsystem.
  Structural precedent for the parity strategy.
- ADR-0011 — Discriminated kernel config. Unchanged by
  this ADR: `IouKind` remains AP-shaped.
- ADR-0013 — Streaming evaluator. Reused without
  modification: confusion matrices add element-wise; the
  per-image cell store is the substrate.
- ADR-0015 — `vernier-cli`. The `vernier semantic`
  subcommand is a follow-up.
- ADR-0016 — Generalized `Breakdown` axis. Reused for
  area-bucketed mIoU when the user opts into it
  (`SemanticSummary` carries an `mIoU_per_breakdown`
  field when the dataset and evaluator are configured
  with breakdowns).
- ADR-0019 — Result tables. Per-class and confusion-matrix
  table surfaces are a follow-up extension.
- ADR-0025 — Panoptic-quality evaluation as a sibling
  crate. Architectural precedent; this ADR follows the
  same pattern with one asymmetry (`vernier-semantic`
  depends on `vernier-core`; `vernier-panoptic` does not).
- `docs/engineering/semantic-segmentation-quirks.md` — the
  quirks survey ratified by this ADR.
- `tests/python/parity_semantic/oracle/VENDORING.md` — to
  be created alongside the implementation; pins
  mmsegmentation, cityscapesScripts, and Pascal/ADE20K
  reference scripts.
- Long, Shelhamer, Darrell. *Fully Convolutional Networks
  for Semantic Segmentation.* CVPR 2015.
  arXiv:1411.4038. (The mIoU formulation this ADR
  reproduces.)
- Cordts et al. *The Cityscapes Dataset for Semantic Urban
  Scene Understanding.* CVPR 2016. (`cityscapesScripts`
  and the 19-class eval convention.)
- Zhou et al. *Scene Parsing through ADE20K Dataset.*
  CVPR 2017. (ADE20K / SceneParse150.)
- `open-mmlab/mmsegmentation` — the strict-mode default
  oracle (commit pinned in `VENDORING.md`).
- `mcordts/cityscapesScripts` — the Cityscapes
  dataset-author oracle.
